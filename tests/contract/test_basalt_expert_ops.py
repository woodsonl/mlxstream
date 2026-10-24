"""basalt's expert ops (monolith/ops/basalt.py: ``basalt_experts``, ``basalt_s2_plan``, ``basalt_q8_0_scaled``,
``basalt_s2_experts``) as compiler/emit_basalt.py emits them over a synthetic .basalt file and the session's buffers:
every dispatch (function, macros, grid, threadgroup, bindings, params, scratch sizes and written slots) against
monolith/kernels.py's builders, ``indirect_reads`` on exactly the dispatches that read blobs through grp_ptr, the
session's blob buffers declared host, the follow-up dispatches behind their barriers in the finished program, and each
refusal, a valid input with exactly one defect refused by that defect's own check (its message). The GPU side is
tests/kernels/test_basalt_expert_ops_program.py."""

import math

import pytest

from monolith import kernels
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import basalt_windows
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.ops import OPS
from monolith.packs.basalt import BasaltSource
from monolith.runtime.program import BufferSpec
from tests.contract.test_basalt_ops import PROFILE, emit, op, pack, q8_1, spliced
from tests.contract.test_basalt_reader import write_basalt

F32, I32, I64, U8 = DType.F32, DType.I32, DType.I64, DType.U8
KINDS = ("basalt_experts", "basalt_s2_plan", "basalt_q8_0_scaled", "basalt_s2_experts")
H, FF, K = kernels.S2_H, kernels.S2_FF, 10          # basalt's experts are 2560 x 640; a token routes to 10
BLOB = kernels.S2_BLOB_BYTES                         # a blob, routed (Q2_0 GGUF rows) or drafter (planar): 1,382,400
N_EXPERT = 4                                         # blobs in each section of the synthetic file
CACHE = ["cache.0", "cache.1"]                       # the session's expert cache: two buffers
PLAN = ("grp_ptr", "grp_start", "n_groups", "ent_dst", "ent_tok")
TABLES = (I64, I32, I32, I32, I32)
READERS = ("basalt_expert_gu", "basalt_expert_down", "basalt_s2_gu", "basalt_s2_down")   # read blobs through grp_ptr
SOURCES = {"quantize_q8_1": kernels.basalt_q8_1_source, "basalt_expert_gu": kernels.basalt_experts_source,
           "basalt_expert_swiglu_q8_1": kernels.basalt_experts_source, "basalt_expert_down": kernels.basalt_experts_source,
           **{f: kernels.basalt_s2_experts_source for f in ("basalt_s2_group_resident", "basalt_quantize_q8_0_scaled",
                                                           "basalt_s2_gu", "basalt_s2_swiglu", "basalt_s2_down")}}


def expert_file(path, experts=None, mtp=None):
    """A .basalt file whose routed ``experts`` and drafter ``mtp/experts`` sections hold N_EXPERT blobs each (zeros
    unless given), then ``meta``, the last section as in basalt's pack, so that no expert window reaches the file's last
    partial page."""
    path.write_bytes(write_basalt({"experts": bytes(N_EXPERT * BLOB) if experts is None else experts,
                                   "mtp/experts": bytes(N_EXPERT * BLOB) if mtp is None else mtp, "meta": b"{}"}))
    return path


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """The sections as zeros (the emitter never reads them)."""
    return BasaltSource(expert_file(tmp_path_factory.mktemp("experts") / "experts.basalt"))


# ---- graph builders (the GPU test uses them too) -------------------------------------------------------------------

def section(g, name, n_expert=N_EXPERT):
    """The value that declares a whole section of the file (its ``n_expert`` blobs), once a graph."""
    return g.values.get(name) or g.const(name, (n_expert * BLOB,), U8)


def dims(r):
    """A table's shape: its rows, or a whole shape."""
    return r if isinstance(r, tuple) else (r,)


def plan_inputs(g, name, cap_groups, cap_entries, rows=None, dtypes=TABLES):
    """A plan the session writes: the five tables as inputs (basalt's ``resident_plan`` is not ported yet)."""
    rows = rows or (cap_groups, cap_groups + 1, 1, cap_entries, cap_entries)
    return [g.input(f"{name}.{s}", dims(r), dt) for s, r, dt in zip(PLAN, rows, dtypes)]


def routed(g, t, *, name="moe", n_embd=H, n_ff=FF, tables=None, rows=None, dtypes=TABLES, xq=None,
           quantizer="ieee", out_rows=None, out_shape=None, out_dtype=F32, **attrs):
    """basalt's decode call of the routed experts at t tokens (``verify.cpp:1731``, ``:1744``, ``:1764``): the FFN
    input's IEEE q8_1 rows, a plan of the call's t * K entries, and native_expert_grouped with k_per_tok K (so
    cap_groups = cap_entries = t * K) into a row per entry, its blobs in the session's cache. ``attrs`` override the
    op's attrs (None drops one)."""
    cap = t * K
    if xq is None:
        xq = q8_1(g, f"{name}.xq", [g.input(f"{name}.x", (t, n_embd), F32)], quantizer, rows=t)
    tables = tables or plan_inputs(g, name, cap, cap, rows, dtypes)
    out = g.value(f"{name}.out", out_shape or (cap if out_rows is None else out_rows, n_embd), out_dtype)
    a = {"n_ff": n_ff, "k_per_tok": K, "blobs": CACHE, **attrs}
    op(g, "basalt_experts", [*tables, xq], [out], **{k: v for k, v in a.items() if v is not None})
    return out


def plan(g, ids, base, *, name="mtp", rows=None, dtypes=TABLES, **attrs):
    """``basalt_s2_plan`` of the routed ``ids`` over ``base``, into tables of the ids' n entries (``rows`` otherwise)."""
    n = ids.shape[0]
    rows = rows or (n, n + 1, 2, n, n)
    tables = [g.value(f"{name}.{s}", dims(r), dt) for s, r, dt in zip(PLAN, rows, dtypes)]
    a = {"k_per_tok": K, "n_expert": N_EXPERT, **attrs}
    op(g, "basalt_s2_plan", [ids, base], tables, **{k: v for k, v in a.items() if v is not None})
    return tables


def q8_0(g, x, name="mtp", t=None, widths=None):
    """``basalt_q8_0_scaled`` of ``x`` into ``name.xq`` and ``name.xs`` (t rows; ``widths`` overrides theirs)."""
    t = x.shape[0] if t is None else t
    wq, ws = widths or (x.shape[1] // 32 * kernels.Q8_0_BLOCK_BYTES, x.shape[1] // 32)
    xq, xs = g.value(f"{name}.xq", (t, wq), U8), g.value(f"{name}.xs", (t, ws), F32)
    op(g, "basalt_q8_0_scaled", [x], [xq, xs])
    return xq, xs


def drafter(g, t, *, name="mtp", n=None, n_expert=N_EXPERT, tables=None, xq=None, xs=None, out_shape=None,
            out_dtype=F32, **attrs):
    """basalt's drafter MoE at t tokens (``mtp.cpp:931-935``): moe_group_resident over its t * K routed ids and the
    resident experts (``mtp/experts``, ``n_expert`` blobs), quantize_q8_0_scaled of the FFN input, and moe_grouped_s2
    with cap_groups = cap_entries = t * K into a row per routed id. Given ``xq``, the op reads it and ``xs`` (None: no
    scales) instead of the quantizer's rows."""
    n = t * K if n is None else n
    if tables is None:
        tables = plan(g, g.input(f"{name}.ids", (n,), I32), section(g, "mtp/experts", n_expert), name=name,
                      n_expert=n_expert)
    if xq is None:
        xq, xs = q8_0(g, g.input(f"{name}.x", (t, H), F32), name)
    out = g.value(f"{name}.out", out_shape or (n, H), out_dtype)
    a = {"cap_groups": n, "cap_entries": n, **attrs}
    op(g, "basalt_s2_experts", [*tables, xq] + ([] if xs is None else [xs]), [out],
       **{k: v for k, v in a.items() if v is not None})
    return out


def all_ops(g, t):
    """Every kind as basalt runs it at t tokens: the routed experts on one IEEE q8_1 of the FFN input, their blobs in
    the session's cache (two host buffers) and in the file's ``experts`` section (its window, as tests bind them); the
    drafter's plan, quantizer and experts over ``mtp/experts``."""
    xq = q8_1(g, "moe.xq", [g.input("moe.x", (t, H), F32)], "ieee", rows=t)
    routed(g, t, xq=xq)
    section(g, "experts")
    routed(g, t, name="moe_w", xq=xq, blobs=["experts"])
    drafter(g, t)


# ---- registration, each dispatch against the builders, indirect reads, host buffers, barriers ----------------------

def test_registered_and_covered(src):
    for kind in KINDS:
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
    g = Graph("step")
    all_ops(g, 2)
    assert {o.kind for o in g.ops} == set(KINDS) | {"basalt_q8_1"}
    check_coverage(g, PROFILE)


def described(prog):
    """Each dispatch as (function, macros, grid, threadgroup, bindings, written slots, indirect reads), a params record
    or workspace named by its first appearance (``params<i>``, ``ws<i>``), and what those hold: a record's bytes, a
    workspace's size. Every kernel is its builder's source, with no threadgroup memory declared."""
    tags = {}

    def bind(slot, name, off):
        spec = prog.buffers[name]
        if spec.role == "params" or name.startswith("ws."):
            name = tags.setdefault(name, f"{'params' if spec.role == 'params' else 'ws'}{len(tags)}")
        return slot, name, off

    ops = []
    for o in prog.ops:
        k = prog.kernels[o.kernel]
        assert k.source == spliced(SOURCES[k.function]()) and o.threadgroup_memory == [], o.name
        ops.append((k.function, k.macros, o.grid, o.threadgroup, [bind(*b) for b in o.bindings], o.meta["writes"],
                    o.meta.get("indirect_reads")))
    held = {tag: prog.buffers[n].init if prog.buffers[n].role == "params" else prog.buffers[n].nbytes
            for n, tag in tags.items()}
    return ops, held


def routed_dispatches(name, t, ws, reads):
    """The routed op's three dispatches at t tokens, its workspaces and params named ``ws`` (gate, up, params, hq)."""
    cap, (gate, up, prm, hq) = t * K, ws
    macros, th = kernels.basalt_expert_macros(H, FF), (kernels.EXPERT_THREADS, 1, 1)
    ggu, gsw, gdn = kernels.basalt_expert_grids(macros, H, FF, cap, cap)
    ptr, start, count = (1, f"{name}.grp_ptr", 0), (2, f"{name}.grp_start", 0), (3, f"{name}.n_groups", 0)
    return [("basalt_expert_gu", macros, (*ggu, 1), th, [ptr, start, count, (4, f"{name}.ent_tok", 0),
                                                         (5, "moe.xq", 0), (6, gate, 0), (7, up, 0), (8, prm, 0)],
             [6, 7], reads),
            ("basalt_expert_swiglu_q8_1", macros, (gsw, 1, 1), th, [(0, gate, 0), (1, up, 0), start, count,
                                                                    (4, hq, 0), (5, prm, 0)], [4], None),
            ("basalt_expert_down", macros, (*gdn, 1), th, [ptr, start, count, (4, f"{name}.ent_dst", 0), (5, hq, 0),
                                                          (6, f"{name}.out", 0), (7, prm, 0)], [6], reads)]


@pytest.mark.parametrize("t", range(1, 9))
def test_dispatches_are_the_builders(src, t):
    """``all_ops`` at T = t: basalt's launches in its order, each with the builders' macros, grid, threadgroup, params
    and scratch sizes (cap_groups = cap_entries = t * 10, basalt's): the routed experts' gate/up, SwiGLU and down over
    one params record, their gate, up and hq workspaces; the plan on one threadgroup of 128, its base the window of
    ``mtp/experts``; the quantizer; the drafter's gate/up, SwiGLU, quantizer and down over one scratch at basalt's
    offsets. The session's tables and activations bind at 0 of their own buffers; no StepState."""
    g = Graph("step")
    all_ops(g, t)
    ops, held = described(emit(g, src, t))
    w = basalt_windows(src, g.values.values(), t)[1]
    cap, th = t * K, (kernels.S2_THREADS, 1, 1)
    grids, sc = kernels.basalt_s2_grids(cap, cap), kernels.basalt_s2_scratch(cap)
    mtp = {s: (i, f"mtp.{s}", 0) for i, s in enumerate(PLAN)}
    assert ops == [
        ("quantize_q8_1", {}, (-(-t * H // 256), 1, 1), (256, 1, 1), [(0, "moe.x", 0), (1, "moe.xq", 0),
                                                                      (2, "params0", 0)], [1], None),
        *routed_dispatches("moe", t, ("ws1", "ws2", "params3", "ws4"), CACHE),
        *routed_dispatches("moe_w", t, ("ws5", "ws6", "params7", "ws8"), [w["experts"][0]]),
        ("basalt_s2_group_resident", {}, (1, 1, 1), (kernels.S2_RESIDENT_THREADS, 1, 1),
         [(0, "mtp.ids", 0), (1, *w["mtp/experts"])] + [(i + 2, f"mtp.{s}", 0) for i, s in enumerate(PLAN)]
         + [(7, "params9", 0)], [2, 3, 4, 5, 6], None),
        ("basalt_quantize_q8_0_scaled", {}, (kernels.basalt_q8_0_scaled_grid(t * H), 1, 1), th,
         [(0, "mtp.x", 0), (1, "mtp.xq", 0), (2, "mtp.xs", 0), (3, "params10", 0)], [1, 2], None),
        ("basalt_s2_gu", {}, (*grids["gu"], 1), th, [mtp["grp_ptr"], mtp["grp_start"], mtp["n_groups"],
                                                     (3, "mtp.ent_tok", 0), (4, "mtp.xq", 0), (5, "mtp.xs", 0),
                                                     (6, "ws11", 0), (7, "params12", 0)], [6], [w["mtp/experts"][0]]),
        ("basalt_s2_swiglu", {}, (grids["swiglu"], 1, 1), th, [(0, "ws11", 0), (1, "params12", 0)], [0], None),
        ("basalt_quantize_q8_0_scaled", {}, (grids["quant"], 1, 1), th,
         [(0, "ws11", 0), (1, "ws11", sc["hq"]), (2, "ws11", sc["hs"]), (3, "params13", 0)], [1, 2], None),
        ("basalt_s2_down", {}, (*grids["down"], 1), th, [mtp["grp_ptr"], mtp["grp_start"], mtp["n_groups"],
                                                         (3, "mtp.ent_dst", 0), (4, "ws11", sc["hq"]),
                                                         (5, "ws11", sc["hs"]), (6, "mtp.out", 0)], [6],
         [w["mtp/experts"][0]])]
    gate, up, hq = kernels.basalt_expert_scratch(cap, FF)
    assert (gate, up, hq) == (4 * cap * FF, 4 * cap * FF, cap * FF // 32 * 36)       # the harness's sizes
    expert = kernels.basalt_expert_params(H, FF)
    assert held == {"params0": kernels.basalt_q8_1_params(t * H), "ws1": gate, "ws2": up, "params3": expert,
                    "ws4": hq, "ws5": gate, "ws6": up, "params7": expert, "ws8": hq,
                    "params9": kernels.basalt_s2_resident_params(cap, K),
                    "params10": kernels.basalt_q8_0_scaled_params(t * H), "ws11": sc["bytes"],
                    "params12": kernels.basalt_s2_params(cap), "params13": kernels.basalt_q8_0_scaled_params(cap * FF)}


def test_indirect_reads_name_the_blob_buffers(src):
    """Exactly the dispatches that read blobs through grp_ptr (both ops' gate/up and down) carry ``indirect_reads``,
    naming the buffers the addresses point into: the session's cache, the window of the file's ``experts`` section, the
    window the drafter's plan binds as its base. The plan binds that window; the SwiGLUs and quantizers read none."""
    g = Graph("step")
    all_ops(g, 3)
    prog = emit(g, src, 3)
    w = basalt_windows(src, g.values.values(), 3)[1]
    reads = [(prog.kernels[o.kernel].function, o.name.split(":")[1], o.meta.get("indirect_reads")) for o in prog.ops]
    assert [r for r in reads if r[2] is not None] == [
        ("basalt_expert_gu", "moe.out", CACHE), ("basalt_expert_down", "moe.out", CACHE),
        ("basalt_expert_gu", "moe_w.out", [w["experts"][0]]), ("basalt_expert_down", "moe_w.out", [w["experts"][0]]),
        ("basalt_s2_gu", "mtp.out", [w["mtp/experts"][0]]), ("basalt_s2_down", "mtp.out", [w["mtp/experts"][0]])]
    assert {f for f, _, r in reads if r is not None} == set(READERS)
    (resident,) = [o for o in prog.ops if prog.kernels[o.kernel].function == "basalt_s2_group_resident"]
    assert resident.bindings[1] == (1, *w["mtp/experts"])
    for o in prog.ops:                                         # the reads OpSpec.access reports (the barrier pass's)
        assert set(o.meta.get("indirect_reads", ())) <= o.access()[0]


def test_session_buffers_are_host(src):
    """A blob buffer that is no graph value is the session's: the program declares it with role host, holding at least
    one blob of each op that names it (the largest), so that Engine requires the session to pass it; one declaration
    for every op that names it. The windows stay the planner's, mapped from the file."""
    g = Graph("step")
    xq = q8_1(g, "small.xq", [g.input("small.x", (2, 512), F32)], "ieee", rows=2)
    routed(g, 2, name="small", n_embd=512, n_ff=256, xq=xq, blobs=["cache.0"])
    all_ops(g, 2)
    prog = emit(g, src, 2)
    small = kernels.basalt_expert_layout(512, 256)["bytes"]
    assert small < BLOB == kernels.basalt_expert_layout(H, FF)["bytes"]
    assert {n: b for n, b in prog.buffers.items() if b.role == "host"} == {n: BufferSpec(BLOB, None, "host")
                                                                          for n in CACHE}
    w = basalt_windows(src, g.values.values(), 2)[0]
    assert all(prog.buffers[n] == spec and spec.file == src.path for n, spec in w.items())


def test_blobs_in_a_graph_value_keep_its_buffer(src):
    """Blobs in a graph value the session writes (an input): no dispatch binds its buffer, but the gate/up and down
    reach it through grp_ptr, so the program keeps it (the emitter drops only arena buffers no dispatch uses) and
    declares nothing host."""
    g = Graph("step")
    g.input("moe.cache", (BLOB,), U8)
    routed(g, 2, blobs=["moe.cache"])
    prog = emit(g, src, 2)
    assert prog.buffers["moe.cache"] == BufferSpec(BLOB, None, "arena")
    assert [o.meta.get("indirect_reads") for o in prog.ops[1:]] == [["moe.cache"], None, ["moe.cache"]]
    assert not any(b.role == "host" for b in prog.buffers.values())


@pytest.mark.parametrize("t", (1, 3, 8))
def test_follow_ups_wait_for_their_first_dispatch(src, t):
    """In the finished program (place_barriers, then the backend's finalize) under ``barriers="minimal"``, each SwiGLU
    and down, and the drafter's intermediate quantizer, carry ``barrier_before``: each reads what the dispatch before it
    wrote. The first routed gate/up waits for the IEEE rows; the second runs beside the first's down; the plan and the
    drafter's standalone quantizer run beside the dispatches before them (they share nothing), and the drafter's gate/up
    waits for both."""
    g = Graph("step")
    all_ops(g, t)
    prog = emit(g, src, t, barriers="minimal")
    flags = [(prog.kernels[o.kernel].function, o.barrier_before) for o in prog.ops]
    assert flags == [("quantize_q8_1", True),
                     ("basalt_expert_gu", True), ("basalt_expert_swiglu_q8_1", True), ("basalt_expert_down", True),
                     ("basalt_expert_gu", False), ("basalt_expert_swiglu_q8_1", True), ("basalt_expert_down", True),
                     ("basalt_s2_group_resident", False), ("basalt_quantize_q8_0_scaled", False),
                     ("basalt_s2_gu", True), ("basalt_s2_swiglu", True), ("basalt_quantize_q8_0_scaled", True),
                     ("basalt_s2_down", True)]


def test_dynamic_t_refused(src):
    """A dynamic-T program is refused for every kind. The routed and drafter experts' own refusals are unreachable:
    their IEEE rows' quantizer and their plan are emitted before them and refuse first."""
    for build in (lambda g: all_ops(g, 2), lambda g: routed(g, 2), lambda g: drafter(g, 2)):
        g = Graph("step")
        build(g)
        with pytest.raises(ValueError, match="a dynamic-T program cannot run them"):
            emit(g, src, 2, dynamic_t=True)


# ---- refusals: each case's valid input emits, and the same input with its one defect is refused by its own check ---

def case(fn, good, bad, emit_kw=None):
    """The graph ``fn(g, **good)``, or ``fn(g, **bad)``; ``emit_kw``: emit_program's arguments with the defect."""
    def build(g, src, is_bad, tmp):
        fn(g, **(bad if is_bad else good))
        return dict(emit_kw or {}) if is_bad else {}
    return build


def r2(g, **kw):
    return routed(g, 2, **kw)


def d2(g, t=2, **kw):
    return drafter(g, t, **kw)


def plan_only(g, n=20, ids_dtype=I32, ids_shape=None, ids=None, base=None, **kw):
    """The drafter's plan alone over n routed ids (or ``ids``) and ``mtp/experts`` (or ``base``)."""
    return plan(g, ids if ids is not None else g.input("mtp.ids", ids_shape or (n,), ids_dtype),
                base if base is not None else section(g, "mtp/experts"), **kw)


def q8_0_only(g, rows=2, n_in=H, **kw):
    """The drafter's quantizer alone on ``rows`` rows of ``n_in`` values."""
    return q8_0(g, g.input("mtp.x", (rows, n_in), F32), **kw)


def _routed_xq(width):
    """The routed experts at 512 x 256 on IEEE q8_1 rows of ``width`` values."""
    def build(g, src, bad, tmp):
        xq = q8_1(g, "w.xq", [g.input("w.x", (2, width if bad else 512), F32)], "ieee", rows=2)
        routed(g, 2, n_embd=512, n_ff=256, xq=xq)
        return {}
    return build


def _routed_session_rows(g, src, bad, tmp):
    """The routed experts on q8_1 rows the session writes, or the IEEE quantizer's."""
    xq = g.input("w.xq", (2, 2880), U8) if bad else None
    routed(g, 2, xq=xq)
    return {}


def _routed_other_kind(g, src, bad, tmp):
    """The routed experts at 4352 x 256 on 4896-byte rows, as wide as the IEEE q8_1 of 4352 values: the IEEE
    quantizer's, or basalt_q8_0_scaled's of 4608 values, its op carrying the attr quantizer "ieee"."""
    if bad:
        xq = g.value("w.xq", (2, 4896), U8)
        op(g, "basalt_q8_0_scaled", [g.input("w.x", (2, 4608), F32)], [xq, g.value("w.xs", (2, 144), F32)],
           quantizer="ieee")
    else:
        xq = q8_1(g, "w.xq", [g.input("w.x", (2, 4352), F32)], "ieee", rows=2)
    routed(g, 2, n_embd=4352, n_ff=256, xq=xq)
    return {}


def _routed_table_at(i, at, good=None):
    """The routed experts with plan table i a pack constant at byte ``at`` (valid: ``good``, its alignment)."""
    def build(g, src, bad, tmp):
        rows = (20, 21, 1, 20, 20)
        tables = [g.const(f"p.{s}", (r,), dt) if j == i else g.input(f"moe.{s}", (r,), dt)
                  for j, (s, r, dt) in enumerate(zip(PLAN, rows, TABLES))]
        routed(g, 2, tables=tables)
        return {"packs": [pack(tmp, {f"p.{PLAN[i]}": (at if bad else good, rows[i] * TABLES[i].itemsize)})]}
    return build


def _plan_at(what, at, good, dtype=U8, shape=(N_EXPERT * BLOB,)):
    """The plan with its ids (i32 [20]) or its base (``dtype`` ``shape``) a pack constant at byte ``at``; valid: a base
    of u8 [n_expert * blob], and both at ``good``."""
    def build(g, src, bad, tmp):
        dt, sh = (I32, (20,)) if what == "ids" else (dtype, shape) if bad else (U8, (N_EXPERT * BLOB,))
        plan_only(g, **{what: g.const(f"p.{what}", sh, dt)})
        return {"packs": [pack(tmp, {f"p.{what}": (at if bad else good, math.prod(sh) * dt.itemsize)})]}
    return build


def _q8_0_at(g, src, bad, tmp):
    """The quantizer on a pack constant at byte 2 (4: valid)."""
    q8_0(g, g.const("p.x", (2, H), F32))
    return {"packs": [pack(tmp, {"p.x": (2 if bad else 4, 2 * H * 4)})]}


def _s2_rows_at(which):
    """The drafter's experts on q8_0 rows and scales the session's pack holds, ``which`` at byte 2 (4: valid)."""
    def build(g, src, bad, tmp):
        xq, xs = g.const("p.xq", (2, 2720), U8), g.const("p.xs", (2, 80), F32)
        drafter(g, 2, xq=xq, xs=xs)
        at = {"xq": 2 if bad and which == "xq" else 4, "xs": 2 if bad and which == "xs" else 4}
        return {"packs": [pack(tmp, {"p.xq": (at["xq"], 2 * 2720), "p.xs": (8192 + at["xs"], 2 * 80 * 4)})]}
    return build


def _s2_session_rows(xq_w, xs_w):
    """The drafter's experts on q8_0 rows (``xq_w`` bytes a token) and scales (``xs_w``) the session writes."""
    def build(g, src, bad, tmp):
        xq = g.input("mtp.xq", (2, xq_w if bad else 2720), U8)
        xs = g.input("mtp.xs", (2, xs_w if bad else 80), F32)
        drafter(g, 2, xq=xq, xs=xs)
        return {}
    return build


def _s2_without_scales(g, src, bad, tmp):
    xq, xs = q8_0(g, g.input("mtp.x", (2, H), F32))
    drafter(g, 2, xq=xq, xs=None if bad else xs)
    return {}


def _s2_session_tables(g, src, bad, tmp):
    """The drafter's experts on the session's tables, or on the plan's."""
    drafter(g, 2, tables=plan_inputs(g, "mtp", 20, 20, rows=(20, 21, 2, 20, 20)) if bad else None)
    return {}


def _s2_tables_out_of_order(g, src, bad, tmp):
    """The plan's tables with ent_dst and ent_tok swapped."""
    tables = plan(g, g.input("mtp.ids", (20,), I32), section(g, "mtp/experts"))
    if bad:
        tables[3], tables[4] = tables[4], tables[3]
    drafter(g, 2, tables=tables)
    return {}


ROUTED_TABLE = [(f"routed {s} {d}", case(r2, {}, {"rows": rows, "dtypes": dts}), why) for s, d, rows, dts, why in (
    ("grp_ptr", "i32", None, (I32,) + TABLES[1:], r"'moe.grp_ptr' is i32 \[20\], not i64 \[>= 20\] \(an address"),
    ("grp_ptr", "a group short", (19, 21, 1, 20, 20), TABLES, r"'moe.grp_ptr' is i64 \[19\], not i64 \[>= 20\]"),
    ("grp_start", "without the end", (20, 20, 1, 20, 20), TABLES, r"'moe.grp_start' is i32 \[20\], not i32 \[>= 21\]"),
    ("n_groups", "f32", None, TABLES[:2] + (F32,) + TABLES[3:], r"'moe.n_groups' is f32 \[1\], not i32 \[>= 1\]"),
    ("ent_dst", "an entry short", (20, 21, 1, 19, 20), TABLES, r"'moe.ent_dst' is i32 \[19\], not i32 \[>= 20\]"),
    ("ent_dst", "of two dims", (20, 21, 1, (20, 1), 20), TABLES, r"'moe.ent_dst' is i32 \[20, 1\], not i32 \[>= 20\]"),
    ("ent_tok", "i64", None, TABLES[:4] + (I64,), r"'moe.ent_tok' is i64 \[20\], not i32 \[>= 20\] \(a token"))]
PLAN_TABLE = [(f"plan {s} {d}", case(plan_only, {}, {"rows": rows, "dtypes": dts}), why) for s, d, rows, dts, why in (
    ("grp_ptr", "i32", None, (I32,) + TABLES[1:], r"'mtp.grp_ptr' is i32 \[20\], not i64 \[>= 20\]"),
    ("grp_ptr", "a group short", (19, 21, 2, 20, 20), TABLES, r"'mtp.grp_ptr' is i64 \[19\], not i64 \[>= 20\]"),
    ("grp_start", "without the end", (20, 20, 2, 20, 20), TABLES, r"'mtp.grp_start' is i32 \[20\], not i32 \[>= 21\]"),
    ("grp_start", "of two dims", (20, (21, 1), 2, 20, 20), TABLES,
     r"'mtp.grp_start' is i32 \[21, 1\], not i32 \[>= 21\]"),
    ("counts", "one count", (20, 21, 1, 20, 20), TABLES, r"'mtp.n_groups' is i32 \[1\], not i32 \[>= 2\] \(the group"),
    ("ent_dst", "an entry short", (20, 21, 2, 19, 20), TABLES, r"'mtp.ent_dst' is i32 \[19\], not i32 \[>= 20\]"),
    ("ent_tok", "i64", None, TABLES[:4] + (I64,), r"'mtp.ent_tok' is i64 \[20\], not i32 \[>= 20\]"))]
ALIGN = r"binds at byte {} of [\w.]+; slot {} needs {}-byte alignment"

REFUSALS = [
    # the routed experts
    ("routed n_ff missing", case(r2, {}, {"n_ff": None}), "attrs n_ff, k_per_tok must be positive ints"),
    ("routed k_per_tok 0", case(r2, {}, {"k_per_tok": 0}), r"must be positive ints, got \[640, 0\]"),
    ("routed blobs empty", case(r2, {}, {"blobs": []}), "attr blobs must list the buffers its blobs live in"),
    ("routed blobs a string", case(r2, {}, {"blobs": "cache.0"}), "attr blobs must list the buffers"),
    ("routed blobs not names", case(r2, {}, {"blobs": ["cache.0", 1]}), "attr blobs must list the buffers"),
    ("routed out a row short", case(r2, {}, {"out_rows": 19}),
     r"'moe.out' is f32 \[19, 2560\]; .* so out is f32 \[>= T x k_per_tok \(20\), n_embd\]"),
    ("routed out i32", case(r2, {}, {"out_dtype": I32}), r"'moe.out' is i32 \[20, 2560\]; an entry writes the row"),
    ("routed out of three dims", case(r2, {}, {"out_shape": (20, H, 1)}),
     r"'moe.out' is f32 \[20, 2560, 1\]; an entry writes the row"),
    ("routed tables of 2 tokens, xq of 3", case(routed, {"t": 2, "rows": (20, 21, 1, 20, 20), "out_rows": 30},
                                                 {"t": 3, "rows": (20, 21, 1, 20, 20), "out_rows": 30}),
     r"'moe.grp_ptr' is i64 \[20\], not i64 \[>= 30\] \(an address per group\)"),
    ("routed n_ff not whole 64-value blocks", case(r2, {}, {"n_ff": 96}),
     "n_embd must be a multiple of 256 and n_ff of 64"),
    ("routed q8_1 rows of another width", _routed_xq(2560),
     r"'w.xq' is u8 \[2, 2880\], not u8 \[2, 576\] \(q8_1 rows of the output's width\)"),
    ("routed on native q8_1 rows", case(r2, {}, {"quantizer": "native"}), "is not a basalt_q8_1 op's ieee rows"),
    ("routed on rows the session writes", _routed_session_rows, r"'w.xq' is not a basalt_q8_1 op's ieee rows"),
    ("routed on another kind's rows", _routed_other_kind, r"'w.xq' is not a basalt_q8_1 op's ieee rows"),
    *ROUTED_TABLE,
    ("routed grp_ptr off 8 bytes", _routed_table_at(0, 4, 8), ALIGN.format(4, 1, 8)),
    ("routed grp_start off 4 bytes", _routed_table_at(1, 2, 4), ALIGN.format(2, 2, 4)),
    ("routed n_groups off 4 bytes", _routed_table_at(2, 2, 4), ALIGN.format(2, 3, 4)),
    ("routed ent_dst off 4 bytes", _routed_table_at(3, 2, 4), "'p.ent_dst' " + ALIGN.format(2, 4, 4)),
    ("routed ent_tok off 4 bytes", _routed_table_at(4, 2, 4), "'p.ent_tok' " + ALIGN.format(2, 4, 4)),
    # the drafter's plan
    ("plan dynamic-T", case(plan_only, {}, {}, {"dynamic_t": True}), "a dynamic-T program cannot run them"),
    ("plan k_per_tok 0", case(plan_only, {}, {"k_per_tok": 0}), r"attrs k_per_tok, n_expert must be positive ints"),
    ("plan n_expert missing", case(plan_only, {}, {"n_expert": None}), r"got \[10, None\]"),
    ("plan ids i64", case(plan_only, {}, {"ids_dtype": I64}), r"the ids 'mtp.ids' are i64 \[20\], not i32 \[n\]"),
    ("plan ids of two rows", case(plan_only, {"rows": (20, 21, 2, 20, 20)},
                                  {"ids_shape": (2, 10), "rows": (20, 21, 2, 20, 20)}),
     r"the ids 'mtp.ids' are i32 \[2, 10\], not i32 \[n\]"),
    ("plan of 129 ids", case(plan_only, {"n": 128}, {"n": 129}), "moe_group_resident: n must be 1..128"),
    *PLAN_TABLE,
    ("plan base of fewer blobs than n_expert", case(plan_only, {}, {"n_expert": N_EXPERT + 1}),
     rf"the router's ids reach n_expert \(5\) blobs of {BLOB} bytes from it, u8 \[>= {5 * BLOB}\]"),
    ("plan base i32", _plan_at("base", 2, 2, I32), r"base 'p.base' is i32 \[5529600\]; the router's ids reach"),
    ("plan base of two dims", _plan_at("base", 2, 2, shape=(N_EXPERT * BLOB, 1)),
     r"base 'p.base' is u8 \[5529600, 1\]; the router's ids reach"),
    ("plan base at an odd byte", _plan_at("base", 1, 2), "'p.base' " + ALIGN.format(1, 1, 2)),
    ("plan ids off 4 bytes", _plan_at("ids", 2, 4), "'p.ids' " + ALIGN.format(2, 0, 4)),
    # the quantizer
    ("q8_0 dynamic-T", case(q8_0_only, {}, {}, {"dynamic_t": True}), "a dynamic-T program cannot run them"),
    ("q8_0 rows not whole blocks", case(q8_0_only, {"n_in": 64}, {"n_in": 48}),
     "a row of 48 values is not whole 32-value blocks"),
    ("q8_0 input of other rows", case(q8_0_only, {"t": 2}, {"rows": 3, "t": 2}),
     r"'mtp.x' is f32 \[3, 2560\], not f32 \[2, 2560\] \(the quantized rows\)"),
    ("q8_0 blocks of another width", case(q8_0_only, {}, {"widths": (2754, 80)}),
     r"'mtp.xq' is u8 \[2, 2754\], not u8 \[2, 2720\] \(block_q8_0 rows of the input's rows\)"),
    ("q8_0 scales of another width", case(q8_0_only, {}, {"widths": (2720, 81)}),
     r"'mtp.xs' is f32 \[2, 81\], not f32 \[2, 80\] \(an fp32 scale per block\)"),
    ("q8_0 input off 4 bytes", _q8_0_at, "'p.x' " + ALIGN.format(2, 0, 4)),
    # the drafter's experts
    ("s2 without the scales", _s2_without_scales,
     r"6 inputs; it takes \(grp_ptr, grp_start, counts, ent_dst, ent_tok, xq, xs\)"),
    ("s2 cap_groups missing", case(d2, {}, {"cap_groups": None}), "attrs cap_groups, cap_entries must be positive ints"),
    ("s2 on the session's tables", _s2_session_tables, "its tables are not one basalt_s2_plan's outputs"),
    ("s2 on the plan's tables out of order", _s2_tables_out_of_order, "its tables are not one basalt_s2_plan's outputs"),
    ("s2 cap_groups below the plan's ids", case(d2, {}, {"cap_groups": 19}), "cap_groups 19 < the plan's 20 ids"),
    ("s2 cap_entries below the plan's ids", case(d2, {}, {"cap_entries": 19}), "cap_entries 19 < the plan's 20 entries"),
    ("s2 of 9 tokens", case(d2, {"t": 8}, {"t": 9}), "9 tokens; the kernels take at most 8 entries a group"),
    ("s2 ids of more tokens than xq holds", case(d2, {"t": 3, "n": 30}, {"t": 2, "n": 30}),
     "the plan's 30 ids are 3 tokens of 10; xq holds 2"),
    ("s2 q8_0 rows of another width", _s2_session_rows(2754, 80),
     r"'mtp.xq' is u8 \[2, 2754\], not u8 \[2, 2720\] \(block_q8_0 rows of 2560 values\)"),
    ("s2 scales of another width", _s2_session_rows(2720, 81),
     r"'mtp.xs' is f32 \[2, 81\], not f32 \[2, 80\] \(their fp32 scales\)"),
    ("s2 out a row short", case(d2, {}, {"out_shape": (19, H)}), r"'mtp.out' is f32 \[19, 2560\]; .* f32 \[>= 20, 2560\]"),
    ("s2 out of another width", case(d2, {}, {"out_shape": (20, 2048)}), r"'mtp.out' is f32 \[20, 2048\]; an entry"),
    ("s2 out i32", case(d2, {}, {"out_dtype": I32}), r"'mtp.out' is i32 \[20, 2560\]; an entry writes the row"),
    ("s2 out of three dims", case(d2, {}, {"out_shape": (20, H, 1)}), r"'mtp.out' is f32 \[20, 2560, 1\]; an entry"),
    ("s2 cap_groups past CUDA's grid", case(d2, {"cap_groups": 65535}, {"cap_groups": 65536}),
     r"moe_grouped_s2: cap_groups must be 1..65535, got 65536"),
    ("s2 cap_entries past CUDA's grid", case(d2, {"cap_entries": 858_993_458}, {"cap_entries": 858_993_459}),
     r"quantize_q8_0_scaled: 549755813760 values need 2147483648 threadgroups"),
    ("s2 q8_0 rows off 4 bytes", _s2_rows_at("xq"), "'p.xq' " + ALIGN.format(2, 4, 4)),
    ("s2 scales off 4 bytes", _s2_rows_at("xs"), "'p.xs' " + ALIGN.format(8194, 5, 4)),
]


@pytest.mark.parametrize("build, why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(src, tmp_path, build, why):
    for bad in (False, True):
        g = Graph("step")
        kw = build(g, src, bad, tmp_path / f"pack{int(bad)}")
        packs = kw.pop("packs", [])
        if not bad:
            emit(g, src, 2, packs, **kw)
            continue
        with pytest.raises(ValueError, match=why):
            emit(g, src, 2, packs, **kw)
