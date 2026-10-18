# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/q5k_head_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): the rows' rescaling and the planted row (centered here).
"""basalt's Q5_K head ops (monolith/ops/basalt.py) in compiled static programs on the GPU, as basalt captures them: the
verify head (``q5k_head_mmvq`` over ``output.weight``, T columns, no map) in one program per width T = 1 .. 8, and a
drafter's head, one program per draft-head tier and width (``round_exec_[tier][T]``, mtp.cpp:1516; ``mtp.cpp:1006-1023``
picks the launches): at T = 1 the proxy top-1 over the 2-bit copy of the first n_tier1 rows (tier 0) or the top-1 with
the map over every gathered row (tier 1), above T = 1 the head over the tier's rows and the mask on each column; and the
top-1 over ``output.weight`` without a map (a drafter with no draft vocabulary). The file is bound through its windows
and through exact per-tensor copies; the gathered rows, their map and the 2-bit copy are the session's host buffers, one
allocation every draft program binds.

Synthetic Q5_K rows at n_in 2560 (q5k_head_parity's rows, rescaled so that the logits spread 3, as its proxy test does)
and q8_1 columns, each a noise vector plus one planted row, centered and lifted so that its logit stands above every
other row's (by 11 or more in these columns; uncentered, the rows' common mean lifts every logit with it): the row of a
token the draft vocabulary lacks until a flush brings it in.

- Each output bitwise equal to the merged single-kernel harness (tests/kernels/basalt_gpu.py) on the same bytes: the
  logits, the id, the probability and the top-1's partials; the proxy top-1's id and probability with the score's
  candidates and tile partials; the masked head's columns against the harness's head and mask on each column. The
  program's outputs and scratch start as 0xff bytes, the harness's as 0xee.
- The session's flush (``ctx_flush``, mtp.cpp:1439-1452): the planted tokens' rows gathered into the next context
  rows, the 2-bit copy packed over them by the harness's direct pack dispatch (row0 > 0) on top of the load's pack,
  their ids written into the map. Every draft program scored again equals the harness on the new bytes, its pick
  moves to the planted token, and the copy equals a pack of the new rows at load.
- Each program run twice gives identical bits."""

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler import emit_program
from monolith.core import Graph
from monolith.core.shapes import T
from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.contract.test_basalt_ops import PROFILE, weight, write_file
from tests.contract.test_basalt_q5k_ops import QCOL, ROW, U8, draft_inputs, head, proxy_top1, top1
from tests.kernels import basalt_gpu as gpu
from tests.kernels import test_basalt_q5k_head as q5k

F32 = np.float32
N_IN, N_VOCAB = 2560, 6000
N_BASE, N_XCAP, N_ADD = 3000, 64, 300                   # the draft vocabulary: base ids, context rows, added ids
N_CAP, N_DVOCAB = N_BASE + N_XCAP, N_BASE + N_XCAP + N_ADD  # tier 0's rows (n_tier1_), tier 1's (n_dvocab_)
WIDTHS = range(1, 9)
PIPELINES = {}                                           # one pipeline cache for every engine of the module


@pytest.fixture(scope="module")
def data():
    """The head's rows, the draft vocabulary's map (base ids, the context rows' -1, added ids: mtp.cpp:647-652), the
    flush's tokens (outside the vocabulary) and 8 q8_1 columns, column j planted with the row of token j % 4: its
    values less their mean at 60 / |row - mean|^2 (q5k_head_parity.cpp:263-293 plants 20 / |row|^2)."""
    rng = np.random.default_rng(2560)
    w = q5k.weights(N_VOCAB, rng)
    l0 = gpu.mmvq(13, w, gpu.q8_1(rng.standard_normal(N_IN).astype(F32), "native"), N_IN, N_VOCAB, 1)[0]
    f = F32(3.0 / np.std(l0.astype(np.float64)))          # q5k_head_parity.cpp:215-235: the logits spread 3
    blk = w.reshape(-1, 176)
    blk[:, :4] = (blk[:, :4].copy().view(np.float16).astype(F32) * f).astype(np.float16).view(np.uint8).reshape(-1, 4)
    ids = rng.permutation(N_VOCAB)
    dvocab = np.concatenate([np.sort(ids[:N_BASE]), np.full(N_XCAP, -1), np.sort(ids[N_BASE: N_BASE + N_ADD])])
    tokens = ids[N_BASE + N_ADD: N_BASE + N_ADD + 4]
    cols = []
    for j in range(8):
        dq = q5k.dequant(w[tokens[j % 4]]).astype(np.float64)
        dq -= dq.mean()
        noise = F32(0.8) * rng.standard_normal(N_IN).astype(F32)
        cols.append((noise + (60.0 / np.sum(dq * dq) * dq).astype(F32)).astype(F32))
    return {"w": w, "dvocab": dvocab.astype(np.int32), "tokens": tokens,
            "xq": gpu.q8_1(np.concatenate(cols), "native")}


def run(eng, inputs, read, fill):
    """One step of ``eng`` on ``inputs``, after every buffer of ``read`` ({name: bytes}) is filled with ``fill``."""
    for name, b in inputs.items():
        eng.buffers[name].write(b, 0)
    for name in read:
        eng.buffers[name].fill(fill)
    assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
    return {name: eng.read(name, nb) for name, nb in read.items()}


def runs_twice(eng, inputs, read):
    """Two runs, outputs and scratch starting as 0xff and then 0x5a: (the first's bytes, the second's)."""
    return run(eng, inputs, read, 0xFF), run(eng, inputs, read, 0x5A)


def workspace(prog, function, slot):
    """The buffer a dispatch of ``function`` binds at ``slot`` (the partials or the proxy's scratch) and its size."""
    o = next(o for o in prog.ops if prog.kernels[o.kernel].function == function)
    return o.bindings[[b[0] for b in o.bindings].index(slot)][1]


# ---- the verify head and the plain top-1, over the file -------------------------------------------------------------

@pytest.fixture(scope="module", params=[False, True], ids=["windows", "exact-copies"])
def file_runs(request, data, tmp_path_factory):
    """``output.weight`` in a .basalt file; the verify head at each width and the top-1 without a map, each run twice."""
    w = data["w"]
    path = write_file(tmp_path_factory.mktemp("q5k") / "head.basalt", {"output.weight": (13, N_VOCAB, N_IN, w.tobytes())},
                      {"pad": (1, 64, bytes(128))})
    src = BasaltSource(path, copy=request.param)
    progs = {}
    for t in WIDTHS:
        g = Graph("step")
        head(g, "y", g.input("xq", (T, QCOL), U8), weight(g, "gguf/output.weight", src))
        progs[t] = emit_program(g, pack=src, profile=PROFILE, t=t, tail=None)
    g = Graph("step")
    top1(g, "t", g.input("x1", (1, QCOL), U8), weight(g, "gguf/output.weight", src))
    progs["top1"] = emit_program(g, pack=src, profile=PROFILE, t=1, tail=None)
    dev = gpu.device()
    shared, engines, out = {}, {}, {}
    for key, prog in progs.items():
        engines[key] = Engine(prog, dev, buffers=shared, pipeline_cache=PIPELINES)
        shared.update({n: b for n, b in engines[key].buffers.items() if prog.buffers[n].role == "weights"})
    xq = data["xq"]
    for t in WIDTHS:
        out[t] = runs_twice(engines[t], {"xq": xq[: t * QCOL].tobytes()}, {"y": 4 * t * N_VOCAB})
    grid = kernels.basalt_q5k_head_grid(N_VOCAB)
    part = workspace(progs["top1"], "basalt_q5k_head_top1", 4)
    out["top1"] = runs_twice(engines["top1"], {"x1": xq[:QCOL].tobytes()},
                             {"t.y": 4 * N_VOCAB, "t.id": 4, "t.p": 4, part: 12 * grid})
    return {"src": src, "progs": progs, "engines": engines, "out": out, "part": part}


def test_verify_head_is_the_harness(data, file_runs):
    """basalt's H3 (verify.cpp:1936): each width's logits bit for bit the harness's launch for T columns."""
    w, xq = data["w"], data["xq"]
    for t in WIDTHS:
        y = np.frombuffer(file_runs["out"][t][0]["y"], F32).reshape(t, N_VOCAB)
        assert y.tobytes() == gpu.q5k_head(w, xq[: t * QCOL], N_IN, N_VOCAB, t).tobytes(), t
        assert np.all(np.isfinite(y)) and np.argmax(y[0]) == data["tokens"][0], t


def test_top1_without_a_map_is_the_harness(data, file_runs):
    """The top-1 over output.weight without a map: logits, id, probability and the partials, bit for bit."""
    got = file_runs["out"]["top1"][0]
    y, tid, p, part = gpu.q5k_top1(data["w"], data["xq"][:QCOL], N_VOCAB)
    assert (got["t.y"], got["t.p"], got[file_runs["part"]]) == (y.tobytes(), p.tobytes(), part.tobytes())
    assert np.frombuffer(got["t.id"], np.int32)[0] == tid == data["tokens"][0]


def test_file_programs_run_twice_alike(file_runs):
    for key, (first, second) in file_runs["out"].items():
        assert first == second, key


# ---- the draft head: one program per tier and width over the session's host buffers ----------------------------------

def draft_program(tier, t):
    """Tier 0 reads the first N_CAP gathered rows (the proxy top-1 at T = 1), tier 1 all of them (the top-1 with the
    map); above T = 1 both run the head over their rows and mask each column (mtp.cpp:1006-1023)."""
    g = Graph("step")
    rows, mp, proxy = draft_inputs(g, rows=N_DVOCAB, cap=N_CAP)
    if t == 1:
        x1 = g.input("x1", (1, QCOL), U8)
        if tier == 0:
            proxy_top1(g, "p", x1, proxy, rows, mp, n_rows=N_CAP, n_cap=N_CAP)
        else:
            top1(g, "t", x1, rows, mp)
    else:
        head(g, "y", g.input("xq", (T, QCOL), U8), rows, mp, width=(N_CAP, N_DVOCAB)[tier])
    return emit_program(g, pack=[], profile=PROFILE, t=t, tail=None)


def draft_reads(prog, tier, t):
    """{buffer: bytes} a draft program's run reads: its outputs and the scratch its second dispatch reads."""
    if t > 1:
        return {"y": 4 * t * (N_CAP, N_DVOCAB)[tier]}
    if tier == 0:
        return {"p.id": 4, "p.p": 4,
                workspace(prog, "basalt_q5k_proxy_score", 3): kernels.basalt_q5k_proxy_scratch_bytes(N_CAP)}
    grid = kernels.basalt_q5k_head_grid(N_DVOCAB)
    return {"t.y": 4 * N_DVOCAB, "t.id": 4, "t.p": 4,
            workspace(prog, "basalt_q5k_head_top1", 4): kernels.basalt_q5k_top1_part_bytes(grid)}


@pytest.fixture(scope="module")
def draft_runs(data):
    """The session: its host buffers after the load (the gather with the context rows reading the first base id's row,
    the copy packed over its N_CAP rows, mtp.cpp:657-680), every draft program run twice; then its flush (the four
    tokens' rows into context rows N_BASE .., the pack over them at row0 = N_BASE, their ids into the map) and every
    program run twice again. Returns the programs, engines, buffers and each state's bytes and outputs."""
    w, dvocab = data["w"], data["dvocab"].copy()
    dhead = w[np.where(dvocab < 0, dvocab[0], dvocab)]
    proxy = gpu.q5k_proxy_pack(dhead, N_CAP, [(0, N_CAP)])
    dev = gpu.device()
    session = {"dhead": gpu.nt.Buffer(dev, dhead.tobytes()), "dvocab": gpu.nt.Buffer(dev, dvocab.tobytes()),
               "proxy": gpu.nt.Buffer(dev, proxy.tobytes())}
    progs = {(tier, t): draft_program(tier, t) for tier in (0, 1) for t in WIDTHS}
    engines = {k: Engine(p, dev, buffers=session, pipeline_cache=PIPELINES) for k, p in progs.items()}
    xq = data["xq"]

    def score():
        out = {}
        for (tier, t), eng in engines.items():
            inputs = {"x1" if t == 1 else "xq": xq[: t * QCOL].tobytes()}
            out[tier, t] = runs_twice(eng, inputs, draft_reads(progs[tier, t], tier, t))
        return out

    states = {"load": {"dhead": dhead.copy(), "dvocab": dvocab.copy(), "proxy": proxy, "out": score()}}
    at, k = N_BASE, len(data["tokens"])                   # xbase_ + n_xused_: the first free context row
    dhead[at: at + k] = w[data["tokens"]]                 # gather_rows (the session's)
    session["dhead"].write(dhead[at: at + k].tobytes(), at * ROW)
    proxy = gpu.q5k_proxy_pack(dhead, N_CAP, [(at, k)], proxy=proxy)     # q5k_proxy_pack(dhead_, at, n, ...)
    session["proxy"].write(proxy.tobytes(), 0)
    dvocab[at: at + k] = data["tokens"]                   # the id map's copy
    session["dvocab"].write(dvocab[at: at + k].tobytes(), 4 * at)
    states["flush"] = {"dhead": dhead, "dvocab": dvocab, "proxy": proxy, "out": score()}
    return {"progs": progs, "engines": engines, "session": session, "states": states}


@pytest.mark.parametrize("state", ["load", "flush"])
def test_proxy_top1_is_the_harness(data, draft_runs, state):
    """Tier 0 at T = 1: the id, the probability, the score's candidates and tile partials bit for bit the harness's
    q5k_proxy_top1 on the state's copy, rows and map. The load's pick is a base row; the flush's is the token."""
    s = draft_runs["states"][state]
    got = s["out"][0, 1][0]
    tid, p, cand, part = gpu.q5k_proxy_top1(s["proxy"], s["dhead"], data["xq"][:QCOL], N_CAP, N_CAP, s["dvocab"])
    scratch = got[workspace(draft_runs["progs"][0, 1], "basalt_q5k_proxy_score", 3)]
    t = kernels.basalt_q5k_proxy_tiles(N_CAP)            # n_rows = n_cap: the candidates, then the tiles' partials
    assert (scratch[: 32 * t], scratch[32 * t: 40 * t]) == (cand.tobytes(), part.tobytes())
    assert np.frombuffer(got["p.id"], np.int32)[0] == tid and got["p.p"] == p.tobytes()
    assert (tid == data["tokens"][0]) == (state == "flush") and tid in s["dvocab"][:N_CAP]


@pytest.mark.parametrize("state", ["load", "flush"])
def test_top1_with_the_map_is_the_harness(data, draft_runs, state):
    """Tier 1 at T = 1: the logits (masked rows -inf), the id through the map, the probability and the partials bit for
    bit the harness's q5k_head_top1 on the state's rows and map. The flush's pick is the token."""
    s = draft_runs["states"][state]
    got = s["out"][1, 1][0]
    y, tid, p, part = gpu.q5k_top1(s["dhead"], data["xq"][:QCOL], N_DVOCAB, s["dvocab"])
    part_name = workspace(draft_runs["progs"][1, 1], "basalt_q5k_head_top1", 4)
    assert (got["t.y"], got["t.p"], got[part_name]) == (y.tobytes(), p.tobytes(), part.tobytes())
    assert np.frombuffer(got["t.id"], np.int32)[0] == tid
    assert np.all(np.isneginf(y[s["dvocab"] < 0])) and np.all(np.isfinite(y[s["dvocab"] >= 0]))
    assert (tid == data["tokens"][0]) == (state == "flush")


@pytest.mark.parametrize("state", ["load", "flush"])
@pytest.mark.parametrize("tier", [0, 1])
def test_masked_head_is_the_harness(data, draft_runs, tier, state):
    """Each tier above T = 1: column j of the T-wide masked head bit for bit the harness's head and mask on column j
    over the tier's rows (basalt masks each column in turn, mtp.cpp:1022-1023); after the flush column j's best row is
    the context row of the token planted in it."""
    s, nv = draft_runs["states"][state], (N_CAP, N_DVOCAB)[tier]
    want = [gpu.q5k_head_masked(s["dhead"], data["xq"][j * QCOL: (j + 1) * QCOL], N_IN, nv, s["dvocab"])
            for j in range(8)]
    for t in WIDTHS[1:]:
        y = np.frombuffer(s["out"][tier, t][0]["y"], F32).reshape(t, nv)
        for j in range(t):
            assert y[j].tobytes() == want[j].tobytes(), (t, j)
            if state == "flush":
                assert np.argmax(y[j]) == N_BASE + j % 4, (t, j)


def test_the_flush_packs_what_a_load_would(draft_runs):
    """The copy after the load's pack and the flush's pack over the appended rows equals one pack of the new rows (the
    session's two direct dispatches give a load's bytes), and the session's buffers hold what the harness wrote."""
    s, session = draft_runs["states"]["flush"], draft_runs["session"]
    assert np.array_equal(s["proxy"], gpu.q5k_proxy_pack(s["dhead"], N_CAP, [(0, N_CAP)]))
    for name in ("dhead", "dvocab", "proxy"):
        assert session[name].read(0, session[name].nbytes) == np.ascontiguousarray(s[name]).tobytes(), name


def test_draft_programs_share_the_sessions_buffers(draft_runs):
    """Every tier's program at every width binds the session's one allocation of each host buffer, and a program is
    refused without them (the session allocates them: Engine)."""
    for key, eng in draft_runs["engines"].items():
        assert all(eng.buffers[n] is b for n, b in draft_runs["session"].items()), key
    with pytest.raises(ValueError, match="host buffer dhead: the session allocates it"):
        Engine(draft_runs["progs"][0, 1], gpu.device(), pipeline_cache=PIPELINES)


def test_draft_programs_run_twice_alike(draft_runs):
    for state, s in draft_runs["states"].items():
        for key, (first, second) in s["out"].items():
            assert first == second, (state, key)
