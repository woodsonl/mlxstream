"""basalt's expert ops (monolith/ops/basalt.py) in compiled programs on the GPU: one static program per window width,
T = 1 .. 8 (basalt captures one graph per width), each run bitwise equal to the merged single-kernel harness
(tests/kernels/basalt_gpu.py) on the same bytes, under shader validation.

- The routed experts (``basalt_experts``) on the IEEE q8_1 of the FFN input, t * 10 entries (cap_groups = cap_entries
  = t * 10, basalt's call), with the plan basalt's ``resident_plan`` makes (not ported; transcribed here): blobs split
  over two host buffers the session allocates, so that grp_ptr crosses buffers as the harness builds it
  (``basalt_gpu.py:90-93``), at 2560 x 640 (the two-row gate/up and 16-lane down basalt runs) and 512 x 256 (the
  one-row kernels); and blobs in a synthetic .basalt file's ``experts`` section, through its window and as an exact
  copy. The IEEE rows, the gate, up and q8_1 scratch and the output against ``basalt_gpu.q8_1`` and ``experts``
  (``:65-126``); an empty plan (basalt's for a non-resident id) writes nothing.
- The drafter's experts (``basalt_s2_plan``, ``basalt_q8_0_scaled``, ``basalt_s2_experts``) with the device plan over
  a synthetic ``mtp/experts`` section, through its window and as an exact copy: the plan's tables, the q8_0 rows and
  scales, the scratch and the output against ``basalt_gpu.q8_0_scaled`` and ``s2_experts(resident=...)``
  (``:129-212``).
- ``Engine`` refuses a program whose ``indirect_reads`` names a buffer it lacks, and one whose session cache is not
  passed or is short.
- The device plan's tables, valid and equal to a host transcription of ``group_resident_kernel``, for every routing
  pattern the merged drafter tests use (tests/kernels/test_basalt_s2_experts.py) and the edges of its range; the
  tables' padding keeps its bytes.

The program's outputs and workspaces start as 0xff bytes and the harness's as other patterns, so a value neither
writes cannot match."""

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler import emit_program
from monolith.compiler.emit import basalt_windows
from monolith.core import Graph
from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.contract.test_basalt_expert_ops import (BLOB, CACHE, I32, K, PLAN, PROFILE, drafter, expert_file, plan,
                                                   routed, section)
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_experts import random_blob
from tests.kernels.test_basalt_s2_experts import fill_blob, fill_x, groups_by_expert, signed

F32 = np.float32
H, FF = kernels.S2_H, kernels.S2_FF
WIDTHS = range(1, 9)
E = 16                                          # experts in each section of the file and in the session's cache
PAD = 512                                       # bytes between the cache's blobs, past each one's end
PIPELINES = {}                                  # one pipeline cache for every engine of the module


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The synthetic .basalt file: E routed blobs (Q2_0 rows at 2560 x 640) in ``experts``, E drafter blobs (planar
    Q2_0, ``fill_blob``) in ``mtp/experts``."""
    rng = np.random.default_rng(2026)
    rblobs = np.stack([random_blob(rng, H, FF) for _ in range(E)])
    mblobs = np.stack([fill_blob(rng) for _ in range(E)])
    path = expert_file(tmp_path_factory.mktemp("experts") / "experts.basalt", rblobs.tobytes(), mblobs.tobytes())
    return path, rblobs, mblobs


def emit(g, src, t):
    return emit_program(g, pack=src, profile=PROFILE, t=t, tail=None)


def bound(prog, function, slot):
    """The buffer the first dispatch of ``function`` binds at ``slot``."""
    o = next(o for o in prog.ops if prog.kernels[o.kernel].function == function)
    return next(b for s, b, _ in o.bindings if s == slot)


def run(eng, inputs, fill=()):
    """One step of ``eng`` with ``inputs`` ({buffer: bytes}) written and the buffers ``fill`` set to 0xff bytes."""
    for name, data in inputs.items():
        eng.buffers[name].write(data, 0)
    for name in fill:
        eng.buffers[name].fill(0xFF)
    assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1


def routing(rng, t, n_expert=E, k=K):
    """The router's ids for t tokens: k distinct experts a token (its top-k), shared across tokens."""
    return np.concatenate([rng.permutation(n_expert)[:k] for _ in range(t)]).astype(np.int32)


def resident_plan(ids, k=K):
    """basalt's ``resident_plan`` (``verify_kernels.cu:964-1013``), transcribed: a group per distinct expert in
    first-appearance order, its entries in routing order, ``ent_dst`` the routing index and ``ent_tok`` that index /
    k. Returns (experts, grp_start, ent_dst, ent_tok)."""
    experts, start, order = groups_by_expert([int(i) for i in ids])
    order = np.asarray(order, np.int32)
    return experts, start, order, order // k


def plan_bytes(name, cap, ptr, start, dst, tok, n_groups):
    """The plan's tables as the program's inputs ``name.*`` hold them: grp_ptr padded with null addresses, grp_start
    with -1 (the kernels read neither past the plan)."""
    p = np.zeros(cap, np.uint64)
    p[: len(ptr)] = ptr
    s = np.full(cap + 1, -1, np.int32)
    s[: len(start)] = start
    data = [p, s, np.int32([n_groups]), np.asarray(dst, np.int32), np.asarray(tok, np.int32)]
    return {f"{name}.{t}": d.tobytes() for t, d in zip(PLAN, data)}


def check_routed(eng, prog, t, x, want):
    """The program's IEEE rows, gate, up and q8_1 scratch and output bit for bit the harness's."""
    assert eng.read("moe.xq") == gpu.q8_1(x.ravel(), "ieee").tobytes()
    for name, w in ((bound(prog, "basalt_expert_gu", 6), want["gate"]), (bound(prog, "basalt_expert_gu", 7),
                                                                         want["up"]),
                    (bound(prog, "basalt_expert_swiglu_q8_1", 4), want["hq"]), ("moe.out", want["out"])):
        assert eng.read(name) == np.ascontiguousarray(w).tobytes(), (name, t)


# ---- the routed experts ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_embd, n_ff", [(H, FF), (512, 256)], ids=["2560x640", "512x256"])
def test_routed_over_the_session_cache(files, n_embd, n_ff):
    """The session's cache: E blobs, the first half in cache.0 and the rest in cache.1, each PAD bytes past the previous
    one's end, so that a call's addresses cross buffers. At each width, the plan of t tokens' routing, then an empty
    plan, which must leave the output as it was."""
    rng = np.random.default_rng(31 + n_embd)
    lay = kernels.basalt_expert_layout(n_embd, n_ff)
    arenas = [np.zeros(PAD + E // 2 * (lay["bytes"] + PAD), np.uint8) for _ in CACHE]
    where = []                                                       # expert -> (cache buffer, byte offset)
    for e in range(E):
        i, at = e // (E // 2), PAD + e % (E // 2) * (lay["bytes"] + PAD)
        arenas[i][at: at + lay["bytes"]] = random_blob(rng, n_embd, n_ff)
        where.append((i, at))
    dev = gpu.device()
    cache = {name: gpu.nt.Buffer(dev, a.tobytes()) for name, a in zip(CACHE, arenas)}
    src = BasaltSource(files[0])
    for t in WIDTHS:
        cap = t * K
        g = Graph("step")
        routed(g, t, n_embd=n_embd, n_ff=n_ff)
        prog = emit(g, src, t)
        eng = Engine(prog, dev, buffers=cache, pipeline_cache=PIPELINES)
        experts, start, dst, tok = resident_plan(routing(rng, t))
        ptr = [cache[CACHE[where[e][0]]].gpu_address + where[e][1] for e in experts]
        x = rng.standard_normal((t, n_embd)).astype(F32)
        scratch = [bound(prog, "basalt_expert_gu", 6), bound(prog, "basalt_expert_gu", 7),
                   bound(prog, "basalt_expert_swiglu_q8_1", 4)]
        run(eng, {"moe.x": x.tobytes(), **plan_bytes("moe", cap, ptr, start, dst, tok, len(experts))},
            ["moe.out", *scratch])
        want = gpu.experts(arenas, [where[e][1] for e in experts], start, tok, dst, gpu.q8_1(x.ravel(), "ieee"),
                           n_embd, n_ff, cap, cap_groups=cap, cap_entries=cap, grp_buf=[where[e][0] for e in experts],
                           out_fill=0xFE, scratch_fill=0x7F)
        assert sorted({where[e][0] for e in experts}) == [0, 1]      # the call reads both buffers
        check_routed(eng, prog, t, x, want)
        run(eng, plan_bytes("moe", cap, ptr, start, dst, tok, 0), ["moe.out"])
        assert set(eng.read("moe.out")) == {0xFF}, t                 # an empty plan writes no row


@pytest.mark.parametrize("copy", [False, True], ids=["window", "exact-copy"])
def test_routed_over_a_file_section(files, copy):
    """Blobs read where the file holds them: its ``experts`` section, mapped as a window (or copied into a buffer of
    exactly its bytes), named in the gate/up and down dispatches' indirect reads."""
    path, rblobs, _ = files
    src, rng, dev = BasaltSource(path, copy=copy), np.random.default_rng(41), gpu.device()
    for t in WIDTHS:
        cap = t * K
        g = Graph("step")
        section(g, "experts", E)
        routed(g, t, blobs=["experts"])
        prog = emit(g, src, t)
        win, off = basalt_windows(src, g.values.values(), t)[1]["experts"]
        assert [o.meta.get("indirect_reads") for o in prog.ops[1:]] == [[win], None, [win]]
        assert (prog.buffers[win].file is None) == copy
        eng = Engine(prog, dev, pipeline_cache=PIPELINES)
        experts, start, dst, tok = resident_plan(routing(rng, t))
        ptr = [eng.buffers[win].gpu_address + off + e * BLOB for e in experts]
        x = rng.standard_normal((t, H)).astype(F32)
        scratch = [bound(prog, "basalt_expert_gu", 6), bound(prog, "basalt_expert_gu", 7),
                   bound(prog, "basalt_expert_swiglu_q8_1", 4)]
        run(eng, {"moe.x": x.tobytes(), **plan_bytes("moe", cap, ptr, start, dst, tok, len(experts))},
            ["moe.out", *scratch])
        want = gpu.experts(rblobs.reshape(-1), [e * BLOB for e in experts], start, tok, dst,
                           gpu.q8_1(x.ravel(), "ieee"), H, FF, cap, cap_groups=cap, cap_entries=cap, out_fill=0xFE,
                           scratch_fill=0x7F)
        check_routed(eng, prog, t, x, want)


def test_engine_refuses_missing_blob_buffers(files):
    """The session's cache is the program's host buffers, so ``Engine`` requires both, each holding at least a blob
    (engine.py's host-buffer check); a program whose ``indirect_reads`` names a buffer it does not declare is refused
    (engine.py's indirect-read check)."""
    g = Graph("step")
    routed(g, 2)
    prog = emit(g, BasaltSource(files[0]), 2)
    dev = gpu.device()
    cache = {name: gpu.nt.Buffer(dev, BLOB) for name in CACHE}
    with pytest.raises(ValueError, match=r"host buffer cache.1: the session allocates it"):
        Engine(prog, dev, buffers={"cache.0": cache["cache.0"]}, pipeline_cache=PIPELINES)
    with pytest.raises(ValueError, match=rf"host buffer cache.1: {BLOB - 2} bytes, program needs {BLOB}"):
        Engine(prog, dev, buffers={**cache, "cache.1": gpu.nt.Buffer(dev, BLOB - 2)}, pipeline_cache=PIPELINES)
    Engine(prog, dev, buffers=cache, pipeline_cache=PIPELINES)
    del prog.buffers["cache.1"]
    with pytest.raises(ValueError, match=r"reaches 'cache.1' through a GPU address, which is not a program buffer"):
        Engine(prog, dev, buffers=cache, pipeline_cache=PIPELINES)


# ---- the drafter's experts -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("copy", [False, True], ids=["window", "exact-copy"])
def test_drafter_with_the_device_plan(files, copy):
    """The drafter's MoE at each width: the device plan over t tokens' routing of the file's ``mtp/experts``, the
    quantizer on N(0, 1) activations, the experts; against the harness's plan, quantizer and experts over the same
    blobs in a buffer of its own (grp_ptr compared as offsets from each one's experts)."""
    path, _, mblobs = files
    src, rng, dev = BasaltSource(path, copy=copy), np.random.default_rng(51), gpu.device()
    resident = gpu.nt.Buffer(dev, mblobs.tobytes())
    for t in WIDTHS:
        n = t * K
        g = Graph("step")
        drafter(g, t, n_expert=E)
        prog = emit(g, src, t)
        win, off = basalt_windows(src, g.values.values(), t)[1]["mtp/experts"]
        eng = Engine(prog, dev, pipeline_cache=PIPELINES)
        ids, x = routing(rng, t), rng.standard_normal((t, H)).astype(F32)
        scratch = bound(prog, "basalt_s2_gu", 6)
        run(eng, {"mtp.ids": ids.tobytes(), "mtp.x": x.tobytes()}, ["mtp.out", scratch])
        xq, xs = gpu.q8_0_scaled(x.ravel())
        want = gpu.s2_experts([resident], None, None, 0, None, None, xq, xs, n, n, n, resident=(ids, K, 0, 0))
        assert eng.read("mtp.xq") == xq.tobytes() and eng.read("mtp.xs") == xs.tobytes()
        ng = int(want["n_groups"][0])
        got = {s: np.frombuffer(eng.read(f"mtp.{s}"), np.uint64 if s == "grp_ptr" else np.int32) for s in PLAN}
        assert got["n_groups"][:2].tolist() == want["n_groups"][:2].tolist() == [ng, n]
        assert np.array_equal(got["grp_ptr"][:ng] - np.uint64(eng.buffers[win].gpu_address + off),
                              want["grp_ptr"][:ng] - np.uint64(resident.gpu_address))
        for s in ("grp_start", "ent_dst", "ent_tok"):
            m = ng + 1 if s == "grp_start" else n
            assert np.array_equal(got[s][:m], want[s][:m]), (s, t)
        assert eng.read("mtp.out") == want["out"].tobytes(), t
        sc, s = kernels.basalt_s2_scratch(n), np.frombuffer(eng.read(scratch), np.uint8)
        for a, b in ((0, 4 * 2 * n * FF), (sc["hq"], sc["hq"] + n * FF // 32 * 34), (sc["hs"], sc["hs"] + n * FF // 8)):
            assert np.array_equal(s[a:b], want["scratch"][a:b]), (a, t)     # gate/up, the intermediate and its scales


def merged_routings():
    """(name, ids, k_per_tok): the routings of tests/kernels/test_basalt_s2_experts.py as a router's ids in routing
    order, each drawn as that test draws it, and the edges of the plan's range. The host-planned tests' entries become
    ids by their routing index (ent_dst); the device plan's token map is index / k_per_tok, which those tests' token
    rows need not follow."""
    pats = []
    rng = np.random.default_rng(5)                                   # test_group_resident_drafter_shape
    pats.append(("drafter shape: 5 tokens x 10 of 12", np.concatenate([rng.permutation(12)[:K] for _ in range(5)]),
                 K))
    order, ids, at = np.random.default_rng(4).permutation(8 * K), np.empty(8 * K, np.int32), 0   # test_grouped_8_tokens
    for gi, size in enumerate([8, 1, 5, 2, 7, 3, 1, 6, 4, 8, 1, 2, 3, 4, 5, 6, 7, 1, 1, 5]):
        ids[order[at: at + size]] = 11 if gi == 3 else gi % 11
        at += size
    pats.append(("8 tokens x 10 in groups of 1 to 8", ids, K))
    rng = np.random.default_rng(3)                                   # test_verify_window_rows: its 29 hits
    rng.permutation(4 * K)
    pats.append(("29 hits of 4 tokens x 10", rng.integers(0, 12, 4 * K)[:29], K))
    pats.append(("one token, 7 of 10 counted", np.array([(i * 7 + 1) % 12 for i in range(7)]), K))   # test_device_count
    pats.append(("3 tokens on one expert", np.zeros(3), 1))          # test_whole_expert_against_float64
    pats.append(("2 tokens on one expert", np.zeros(2), 1))          # test_nan_bit_pattern
    rng = np.random.default_rng(14)                                  # test_planted_tiny_activations
    fill_x(rng, 4)
    for lo, hi in ((-149, -100), (-78, -72), (-140, -60), (-126, -110)):
        signed(rng, H // 32, lo, hi)
    ids = np.empty(10, np.int32)
    ids[rng.permutation(10)] = [2] * 4 + [11] * 3 + [5] * 3
    pats.append(("planted: 10 entries of 3 experts", ids, K))
    rng = np.random.default_rng(61)
    pats += [("one id", np.array([7]), 1), ("128 ids: 8 tokens x 16 of 16", routing(rng, 8, E, 16), 16),
             ("128 ids of one expert", np.full(128, E - 1), 16), ("16 distinct ids", rng.permutation(E), 16)]
    return [(name, np.asarray(ids, np.int32), k) for name, ids, k in pats]


ROUTINGS = merged_routings()


@pytest.mark.parametrize("name, ids, k", ROUTINGS, ids=[r[0] for r in ROUTINGS])
def test_plan_tables_are_valid(files, name, ids, k):
    """The device plan in a static program, over tables of 130 entries that start as 0xff bytes: the group and entry
    counts; grp_start from 0, increasing, to n; every routing index an entry once, its token the index / k; each
    group's entries in routing order on one expert, distinct across groups, in first-appearance order, the group's
    address that expert's blob (2-byte aligned) in the window; each group of a router's routing (distinct ids within
    a token) at most a token's worth of entries (the drafter's experts take 8); all of it the host transcription's;
    every table's padding left as it was."""
    src, dev, n, cap = BasaltSource(files[0]), gpu.device(), len(ids), 130
    g = Graph("step")
    plan(g, g.input("p.ids", (n,), I32), section(g, "mtp/experts", E), name="p", rows=(cap, cap + 1, 4, cap, cap),
         k_per_tok=k, n_expert=E)
    prog = emit(g, src, 1)
    win, off = basalt_windows(src, g.values.values(), 1)[1]["mtp/experts"]
    eng = Engine(prog, dev, pipeline_cache=PIPELINES)
    run(eng, {"p.ids": ids.tobytes()}, [f"p.{s}" for s in PLAN])
    base = eng.buffers[win].gpu_address + off
    ptr = np.frombuffer(eng.read("p.grp_ptr"), np.uint64).astype(object)
    start, counts, dst, tok = (np.frombuffer(eng.read(f"p.{s}"), np.int32) for s in PLAN[1:])
    ng = int(counts[0])
    assert counts[:2].tolist() == [len(set(ids.tolist())), n]
    assert start[0] == 0 and start[ng] == n and np.all(np.diff(start[: ng + 1]) > 0)
    assert sorted(dst[:n].tolist()) == list(range(n)) and np.array_equal(tok[:n], dst[:n] // k)
    first = []
    for grp in range(ng):
        ent = dst[start[grp]: start[grp + 1]]
        assert len(set(ids[ent].tolist())) == 1 and np.all(np.diff(ent) > 0), grp
        assert ptr[grp] == base + int(ids[ent[0]]) * BLOB and ptr[grp] % 2 == 0
        first.append(int(ent[0]))
    assert first == sorted(first) and len({int(ids[f]) for f in first}) == ng
    if all(len(set(ids[i: i + k].tolist())) == len(ids[i: i + k]) for i in range(0, n, k)):
        assert np.diff(start[: ng + 1]).max() <= -(-n // k)
    experts, want_start, order = groups_by_expert(ids.tolist())
    assert start[: ng + 1].tolist() == want_start and dst[:n].tolist() == order and len(experts) == ng
    padding = (ptr[ng:], start[ng + 1:], counts[2:], dst[n:], tok[n:])
    assert all(np.all(p == (2 ** 64 - 1 if i == 0 else -1)) for i, p in enumerate(padding))
