"""basalt's Q5_K head ops (monolith/ops/basalt.py: ``basalt_q5k_head``, ``basalt_q5k_top1``, ``basalt_q5k_proxy_top1``)
as compiler/emit_basalt.py emits them over a synthetic .basalt file and the session's host inputs: every dispatch
(function, macros, grid, threadgroup, bindings, params and written slots) against monolith/kernels.py's builders, the
follow-up dispatches behind their barriers in the finished program, the merge's partial count and the scratch sizes,
and each refusal, a valid input with exactly one defect refused by that defect's own check (its message). The GPU side
is tests/kernels/test_basalt_q5k_ops_program.py."""

import struct

import pytest

from monolith import kernels
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import basalt_windows
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.core.shapes import T
from monolith.formats.gguf import ggml_row_bytes
from monolith.ops import OPS
from monolith.packs.basalt import BasaltSource
from tests.contract.test_basalt_ops import PROFILE, emit, op, pack, spliced, weight, write_file

F32, I32, U8 = DType.F32, DType.I32, DType.U8
KINDS = ("basalt_q5k_head", "basalt_q5k_top1", "basalt_q5k_proxy_top1")
QCOL, ROW = 2880, 1760                      # bytes of a 2560-value q8_1 column and of a 2560-value Q5_K row
N_VOCAB, N_DVOCAB, N_CAP = 4000, 3600, 3064  # the head's rows; the gathered rows (tier 1); the copy's (tier 0)
GGUF = {"head": (13, N_VOCAB, 2560), "head2048": (13, 64, 2048), "q4k": (12, 64, 2560)}
FOLLOW_UPS = ("basalt_q5k_head_mask_unmapped", "basalt_q5k_head_top1_merge", "basalt_q5k_proxy_final")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """The tensors as zeros (the emitter never reads them), and a BF16 row the file's dense index needs."""
    gguf = {n: (t, o, i, bytes(o * ggml_row_bytes(t, i))) for n, (t, o, i) in GGUF.items()}
    return BasaltSource(write_file(tmp_path_factory.mktemp("q5k") / "q5k.basalt", gguf, {"pad": (1, 64, bytes(128))}))


# ---- graph builders (the GPU test uses them too) -------------------------------------------------------------------

def draft_inputs(g, rows=N_DVOCAB, cap=N_CAP):
    """The session's host inputs: the gathered head rows, their map and the 2-bit copy of the first ``cap`` rows."""
    return (g.input("dhead", (rows, ROW), U8, host=True), g.input("dvocab", (rows,), I32, host=True),
            g.input("proxy", (kernels.basalt_q5k_proxy_bytes(cap),), U8, host=True))


def head(g, name, xq, rows, *mp, width=None, dtype=F32):
    """``basalt_q5k_head`` of the q8_1 columns ``xq`` by the first ``width`` (default: all) of ``rows``."""
    y = g.value(name, (xq.shape[0], rows.shape[0] if width is None else width), dtype)
    op(g, "basalt_q5k_head", [xq, rows, *mp], [y])
    return y


def top1(g, name, x1, rows, *mp, width=None, dtypes=(F32, I32, F32)):
    """``basalt_q5k_top1`` of the q8_1 column ``x1``: ``name.y`` (the logits of one column), ``name.id``, ``name.p``."""
    shapes = ((1, rows.shape[0] if width is None else width), (1,), (1,))
    outs = [g.value(f"{name}.{s}", sh, dt) for s, sh, dt in zip(("y", "id", "p"), shapes, dtypes)]
    op(g, "basalt_q5k_top1", [x1, rows, *mp], outs)
    return outs


def proxy_top1(g, name, x1, proxy, rows, *mp, n_rows=N_CAP, n_cap=N_CAP, dtypes=(I32, F32)):
    """``basalt_q5k_proxy_top1`` over the first ``n_rows`` rows of the copy ``proxy`` (``n_cap`` rows)."""
    outs = [g.value(f"{name}.{s}", (1,), dt) for s, dt in zip(("id", "p"), dtypes)]
    attrs = {k: v for k, v in (("n_rows", n_rows), ("n_cap", n_cap)) if v is not None}
    op(g, "basalt_q5k_proxy_top1", [x1, proxy, rows, *mp], outs, **attrs)
    return outs


def all_ops(g, src):
    """Every kind as basalt calls it: the verify head over the file's output rows (T columns); the draft head over the
    first N_CAP gathered rows with the map (tier 0's width, masked); the top-1 over the file's rows without a map and
    over the gathered rows with it; the proxy top-1 over the copy of the first N_CAP rows (tier 0)."""
    xq, x1 = g.input("xq", (T, QCOL), U8), g.input("x1", (1, QCOL), U8)
    w = weight(g, "gguf/head", src)
    rows, mp, proxy = draft_inputs(g)
    head(g, "y.head", xq, w)
    head(g, "y.masked", xq, rows, mp, width=N_CAP)
    top1(g, "top1", x1, w)
    top1(g, "top1m", x1, rows, mp)
    proxy_top1(g, "proxy", x1, proxy, rows, mp)


# ---- registration, each dispatch against the builders, barriers, partial counts, scratch -----------------------------

def test_registered_and_covered(src):
    for kind, klass in zip(KINDS, (OpClass.MAP, OpClass.REDUCE, OpClass.REDUCE)):
        od = OPS.get(kind)
        assert od.klass == klass and od.binding_for(PROFILE.key).kernel == kind
    g = Graph("step")
    all_ops(g, src)
    assert {o.kind for o in g.ops} == set(KINDS)
    check_coverage(g, PROFILE)


def described(prog):
    """Each dispatch as (function, macros, grid, threadgroup, bindings, written slots), a params record or workspace
    named by its first appearance (``params<i>``, ``ws<i>``), and what those hold: a record's bytes, a workspace's
    size. Every kernel is the Q5_K head's source."""
    tags = {}

    def bind(slot, name, off):
        spec = prog.buffers[name]
        if spec.role == "params" or name.startswith("ws."):
            name = tags.setdefault(name, f"{'params' if spec.role == 'params' else 'ws'}{len(tags)}")
        return slot, name, off

    ops = []
    for o in prog.ops:
        k = prog.kernels[o.kernel]
        assert k.source == spliced(kernels.basalt_q5k_head_source()) and o.threadgroup_memory == [], o.name
        ops.append((k.function, k.macros, o.grid, o.threadgroup, [bind(*b) for b in o.bindings], o.meta["writes"]))
    held = {tag: prog.buffers[n].init if prog.buffers[n].role == "params" else prog.buffers[n].nbytes
            for n, tag in tags.items()}
    return ops, held


@pytest.mark.parametrize("t", range(1, 9))
def test_dispatches_are_the_builders(src, t):
    """``all_ops`` at T = t: basalt's launches in its order, each with the builders' macros, grid, threadgroup and
    params; the file's rows bound at the window and offset the planner gives them, the session's at 0 of their own
    buffers; the masked head's mask once per column, at that column's logits; one params record per op, shared by its
    dispatches, and one partials workspace between the top-1 and its merge, which a dummy buffer joins in the map's slot
    when there is no map; no StepState and no threadgroup memory."""
    g = Graph("step")
    all_ops(g, src)
    ops, held = described(emit(g, src, t))
    w = basalt_windows(src, g.values.values(), t)[1]["gguf/head"]
    nc, n1, tg = {"NC": str(t)}, {"NC": "1"}, (kernels.Q5K_HEAD_THREADS, 1, 1)
    gv, gc, gd = (kernels.basalt_q5k_head_grid(n) for n in (N_VOCAB, N_CAP, N_DVOCAB))
    mg, mt = kernels.basalt_q5k_mask_grid(N_CAP)
    assert ops == [
        ("basalt_q5k_head", nc, (gv, 1, 1), tg, [(0, *w), (1, "xq", 0), (2, "y.head", 0), (3, "params0", 0)], [2]),
        ("basalt_q5k_head", nc, (gc, 1, 1), tg, [(0, "dhead", 0), (1, "xq", 0), (2, "y.masked", 0),
                                                 (3, "params1", 0)], [2]),
        *[("basalt_q5k_head_mask_unmapped", nc, (mg, 1, 1), (mt, 1, 1),
           [(0, "y.masked", 4 * N_CAP * j), (1, "dvocab", 0), (2, "params1", 0)], [0]) for j in range(t)],
        ("basalt_q5k_head_top1", n1, (gv, 1, 1), tg, [(0, *w), (1, "x1", 0), (2, "top1.y", 0), (3, "ws2", 0),
                                                       (4, "ws3", 0), (5, "params4", 0)], [2, 4]),
        ("basalt_q5k_head_top1_merge", n1, (1, 1, 1), tg, [(0, "ws3", 0), (1, "ws2", 0), (2, "top1.id", 0),
                                                            (3, "top1.p", 0), (4, "params4", 0)], [2, 3]),
        ("basalt_q5k_head_top1", n1, (gd, 1, 1), tg, [(0, "dhead", 0), (1, "x1", 0), (2, "top1m.y", 0),
                                                       (3, "dvocab", 0), (4, "ws5", 0), (5, "params6", 0)], [2, 4]),
        ("basalt_q5k_head_top1_merge", n1, (1, 1, 1), tg, [(0, "ws5", 0), (1, "dvocab", 0), (2, "top1m.id", 0),
                                                            (3, "top1m.p", 0), (4, "params6", 0)], [2, 3]),
        ("basalt_q5k_proxy_score", {}, (kernels.basalt_q5k_proxy_tiles(N_CAP), 1, 1), (kernels.PX_THREADS_SCORE, 1, 1),
         [(0, "proxy", 0), (1, "x1", 0), (2, "dvocab", 0), (3, "ws7", 0), (4, "params8", 0)], [3]),
        ("basalt_q5k_proxy_final", {}, (1, 1, 1), (kernels.PX_THREADS_FINAL, 1, 1),
         [(0, "dhead", 0), (1, "x1", 0), (2, "dvocab", 0), (3, "ws7", 0), (4, "proxy.id", 0), (5, "proxy.p", 0),
          (6, "params8", 0)], [4, 5])]
    assert held == {"params0": kernels.basalt_q5k_head_params(2560, N_VOCAB),
                    "params1": kernels.basalt_q5k_head_params(2560, N_CAP),
                    "ws2": 16,                                                 # the dummy map: 4 bytes, the floor 16
                    "ws3": kernels.basalt_q5k_top1_part_bytes(gv),
                    "params4": kernels.basalt_q5k_head_params(2560, N_VOCAB, False, gv),
                    "ws5": kernels.basalt_q5k_top1_part_bytes(gd),
                    "params6": kernels.basalt_q5k_head_params(2560, N_DVOCAB, True, gd),
                    "ws7": kernels.basalt_q5k_proxy_scratch_bytes(N_CAP),
                    "params8": kernels.basalt_q5k_proxy_params(N_CAP, n_rows=N_CAP)}


def test_host_inputs_are_the_sessions(src):
    """The gathered rows, their map and the 2-bit copy are host buffers (the session allocates them and passes the same
    ones to every program); an input not declared host stays the program's own."""
    g = Graph("step")
    all_ops(g, src)
    prog = emit(g, src, 1)
    assert {n: prog.buffers[n].role for n in ("dhead", "dvocab", "proxy", "xq", "x1")} == {
        "dhead": "host", "dvocab": "host", "proxy": "host", "xq": "arena", "x1": "arena"}
    assert prog.buffers["proxy"].nbytes == kernels.basalt_q5k_proxy_bytes(N_CAP)


def test_host_inputs_are_sized_exactly(src):
    """A host buffer is its value's bytes, a one-id map 4 (basalt allocates ``map.size() * 4``, mtp.cpp:656), so Engine
    takes the session's exact allocation; the program's own buffers keep their 16-byte floor."""
    g = Graph("step")
    x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g, rows=1, cap=1)
    proxy_top1(g, "p", x1, proxy, rows, mp, n_rows=1, n_cap=1)
    prog = emit(g, src, 1)
    assert {n: prog.buffers[n].nbytes for n in ("dhead", "dvocab", "proxy", "p.id")} == {
        "dhead": ROW, "dvocab": 4, "proxy": kernels.basalt_q5k_proxy_bytes(1), "p.id": 16}


@pytest.mark.parametrize("t", (1, 3, 8))
def test_follow_ups_wait_for_their_first_dispatch(src, t):
    """In the finished program (place_barriers, then the backend's finalize) under ``barriers="minimal"``, each mask,
    merge and final carries ``barrier_before``: it reads what the dispatch before it wrote. The first dispatches here
    read only host buffers, inputs and windows, so the pass lets the second and later ones run beside the first."""
    g = Graph("step")
    all_ops(g, src)
    prog = emit(g, src, t, barriers="minimal")
    fn = [prog.kernels[o.kernel].function for o in prog.ops]
    assert fn.count("basalt_q5k_head_mask_unmapped") == t
    assert [o.barrier_before for o, f in zip(prog.ops, fn) if f in FOLLOW_UPS] == [True] * (t + 3)
    assert [o.barrier_before for o, f in zip(prog.ops, fn) if f not in FOLLOW_UPS] == [True, False, False, False,
                                                                                         False]


@pytest.mark.parametrize("width", [N_VOCAB, 2720, 1000, 8])
def test_merge_counts_the_first_grid(src, width):
    """The merge's params give n_blocks = the first dispatch's grid, which is basalt's rule at its 170 multiprocessors:
    the cap of 340 threadgroups from 2,720 rows, fewer below (125 at 1,000, one at 8); with and without the map."""
    g = Graph("step")
    x1, w = g.input("x1", (1, QCOL), U8), weight(g, "gguf/head", src)
    rows, mp, _ = draft_inputs(g, rows=N_VOCAB)
    top1(g, "a", x1, w, width=width)
    top1(g, "b", x1, rows, mp, width=width)
    prog = emit(g, src, 1)
    firsts = [o for o in prog.ops if prog.kernels[o.kernel].function == "basalt_q5k_head_top1"]
    merges = [o for o in prog.ops if prog.kernels[o.kernel].function == "basalt_q5k_head_top1_merge"]
    assert len(firsts) == len(merges) == 2
    want = kernels.basalt_q5k_head_grid(width)
    assert want == {N_VOCAB: 340, 2720: 340, 1000: 125, 8: 1}[width]
    for first, merge in zip(firsts, merges):
        n_in, n_out, has_map, n_blocks = struct.unpack("<IIII", prog.buffers[merge.bindings[-1][1]].init)
        assert (n_in, n_out, n_blocks) == (2560, width, first.grid[0]) and first.grid[0] == want
        assert has_map == (first is firsts[1])


@pytest.mark.parametrize("width,n_rows,n_cap", [(N_VOCAB, N_CAP, N_CAP), (2720, 1000, N_CAP), (1000, 32, 33)])
def test_scratch_sizes_are_the_builders(src, width, n_rows, n_cap):
    """The top-1's partials are kernels.basalt_q5k_top1_part_bytes(grid), shared with its merge; the proxy's scratch
    is kernels.basalt_q5k_proxy_scratch_bytes(n_cap), sized by the copy's capacity as basalt's (mtp.cpp:676-677), while
    the score's grid covers the n_rows rows it scores."""
    g = Graph("step")
    x1 = g.input("x1", (1, QCOL), U8)
    rows, mp, proxy = draft_inputs(g, cap=n_cap)
    top1(g, "a", x1, weight(g, "gguf/head", src), width=width)
    proxy_top1(g, "p", x1, proxy, rows, mp, n_rows=n_rows, n_cap=n_cap)
    prog = emit(g, src, 1)
    first, merge, score, final = prog.ops
    assert first.bindings[4][1] == merge.bindings[0][1] and score.bindings[3][1] == final.bindings[3][1]
    assert prog.buffers[first.bindings[4][1]].nbytes == kernels.basalt_q5k_top1_part_bytes(first.grid[0]) > 16
    assert prog.buffers[score.bindings[3][1]].nbytes == kernels.basalt_q5k_proxy_scratch_bytes(n_cap) > 16
    assert score.grid == (kernels.basalt_q5k_proxy_tiles(n_rows), 1, 1)


# ---- refusals: each case's valid input emits, and the same input with its one defect is refused by its own check ---

def _dynamic(kind):
    def build(g, src, bad, tmp):
        x1, w = g.input("x1", (1, QCOL), U8), weight(g, "gguf/head", src)
        rows, mp, proxy = draft_inputs(g)
        {"basalt_q5k_head": lambda: head(g, "y", x1, w), "basalt_q5k_top1": lambda: top1(g, "y", x1, w),
         "basalt_q5k_proxy_top1": lambda: proxy_top1(g, "y", x1, proxy, rows, mp)}[kind]()
        return {"dynamic_t": bad}
    return build


def _arity(kind):
    """An extra input: a second map."""
    def build(g, src, bad, tmp):
        x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g)
        extra = [g.input("map2", (N_DVOCAB,), I32, host=True)] if bad else []
        {"basalt_q5k_head": lambda: head(g, "y", x1, rows, mp, *extra),
         "basalt_q5k_top1": lambda: top1(g, "y", x1, rows, mp, *extra),
         "basalt_q5k_proxy_top1": lambda: proxy_top1(g, "y", x1, proxy, rows, mp, *extra)}[kind]()
        return {}
    return build


def _rows_type(g, src, bad, tmp):
    head(g, "y", g.input("x1", (1, QCOL), U8), weight(g, "gguf/q4k" if bad else "gguf/head", src), width=64)
    return {}


def _rows_not_host(g, src, bad, tmp):
    head(g, "y", g.input("x1", (1, QCOL), U8), g.input("dhead", (64, ROW), U8, host=not bad))
    return {}


def _host_rows(name="dhead", shape=(64, ROW), dtype=U8):
    """The head over host rows ``dhead`` u8 [64, ROW], or (bad) the given name, shape or dtype."""
    def build(g, src, bad, tmp):
        rows = g.input(name if bad else "dhead", shape if bad else (64, ROW), dtype if bad else U8, host=True)
        head(g, "y", g.input("x1", (1, QCOL), U8), rows)
        return {}
    return build


def _rows_held(kind):
    """The op reads one row more than the rows hold (the head and the top-1: their logits' width; the proxy: n_rows,
    within a copy that has room for it)."""
    def build(g, src, bad, tmp):
        x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g, cap=N_DVOCAB + 1)
        n = N_DVOCAB + 1 if bad else N_DVOCAB
        mp = g.input("map_long", (N_DVOCAB + 1,), I32, host=True)
        {"basalt_q5k_head": lambda: head(g, "y", x1, rows, mp, width=n),
         "basalt_q5k_top1": lambda: top1(g, "y", x1, rows, mp, width=n),
         "basalt_q5k_proxy_top1": lambda: proxy_top1(g, "y", x1, proxy, rows, mp, n_rows=n,
                                                      n_cap=N_DVOCAB + 1)}[kind]()
        return {}
    return build


def _head_shape(good, bad):
    """The head of ``good`` = (columns, blocks a row) over host rows of that width, or of ``bad``: q5k_head_supported's
    rule (1 to 64 blocks, 1 to 8 columns, their staging within 96 KiB). Static counts, as the default t_max (8) allows
    no T past 8."""
    def build(g, src, is_bad, tmp):
        cols, blocks = bad if is_bad else good
        head(g, "y", g.input("xq", (cols, blocks * 8 * kernels.Q8_1_BLOCK_BYTES), U8),
             g.input("dhead", (64, blocks * 176), U8, host=True))
        return {}
    return build


def _head_xq_width(g, src, bad, tmp):
    head(g, "y", g.input("xq", (T, 2304 if bad else 2304 + 576), U8), weight(g, "gguf/head", src))
    return {}


def _head_logits(g, src, bad, tmp):
    head(g, "y", g.input("xq", (T, QCOL), U8), weight(g, "gguf/head", src), dtype=DType.BF16 if bad else F32)
    return {}


def _map_short(kind):
    def build(g, src, bad, tmp):
        x1, rows = g.input("x1", (1, QCOL), U8), g.input("dhead", (N_CAP, ROW), U8, host=True)
        proxy = g.input("proxy", (kernels.basalt_q5k_proxy_bytes(N_CAP),), U8, host=True)
        mp = g.input("dvocab", (N_CAP - 1 if bad else N_CAP,), I32, host=True)
        {"basalt_q5k_head": lambda: head(g, "y", x1, rows, mp),
         "basalt_q5k_top1": lambda: top1(g, "y", x1, rows, mp),
         "basalt_q5k_proxy_top1": lambda: proxy_top1(g, "y", x1, proxy, rows, mp)}[kind]()
        return {}
    return build


def _head_map(shape=(N_DVOCAB,), dtype=I32):
    """The head with the map i32 [N_DVOCAB], or (bad) of the given shape or dtype."""
    def build(g, src, bad, tmp):
        mp = g.input("dvocab", shape if bad else (N_DVOCAB,), dtype if bad else I32, host=True)
        head(g, "y", g.input("x1", (1, QCOL), U8), g.input("dhead", (N_DVOCAB, ROW), U8, host=True), mp)
        return {}
    return build


def _top1_n_in(kind):
    """Rows of 2,048 values (a valid head shape) for the top-1s, which take 2,560 only."""
    def build(g, src, bad, tmp):
        x1 = g.input("x1", (1, 2048 // 32 * 36 if bad else QCOL), U8)
        if kind == "basalt_q5k_top1":
            top1(g, "y", x1, weight(g, "gguf/head2048" if bad else "gguf/head", src), width=64)
        else:
            _, mp, proxy = draft_inputs(g, cap=64)
            rows = g.input("rows", (64, 1408 if bad else ROW), U8, host=True)
            proxy_top1(g, "y", x1, proxy, rows, mp, n_rows=64, n_cap=64)
        return {}
    return build


def _one_column(kind):
    def build(g, src, bad, tmp):
        x = g.input("x", (2 if bad else 1, QCOL), U8)
        rows, mp, proxy = draft_inputs(g)
        if kind == "basalt_q5k_top1":
            top1(g, "y", x, weight(g, "gguf/head", src))
        else:
            proxy_top1(g, "y", x, proxy, rows, mp)
        return {}
    return build


def _out_dtype(kind, which):
    """Output ``which`` (0 the logits, 1 the id, 2 the probability; the proxy's: 1 and 2) of another dtype."""
    def build(g, src, bad, tmp):
        x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g)
        dts = [F32, I32, F32]
        if bad:
            dts[which] = {0: DType.BF16, 1: DType.U32, 2: DType.BF16}[which]
        if kind == "basalt_q5k_top1":
            top1(g, "y", x1, rows, mp, dtypes=dts)
        else:
            proxy_top1(g, "y", x1, proxy, rows, mp, dtypes=dts[1:])
        return {}
    return build


def _proxy_map(g, src, bad, tmp):
    x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g)
    proxy_top1(g, "y", x1, proxy, rows, *([] if bad else [mp]))
    return {}


def _proxy_attrs(attr):
    def build(g, src, bad, tmp):
        x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g)
        proxy_top1(g, "y", x1, proxy, rows, mp, **{attr: None if bad else N_CAP})
        return {}
    return build


def _proxy_rows_past_cap(g, src, bad, tmp):
    """n_rows one past the copy's n_cap (the copy itself sized for n_cap: its size check holds)."""
    x1, (rows, mp, proxy) = g.input("x1", (1, QCOL), U8), draft_inputs(g, cap=N_CAP - 1)
    proxy_top1(g, "y", x1, proxy, rows, mp, n_rows=N_CAP if bad else N_CAP - 1, n_cap=N_CAP - 1)
    return {}


def _proxy_not_host(g, src, bad, tmp):
    x1, (rows, mp, _) = g.input("x1", (1, QCOL), U8), draft_inputs(g)
    proxy = g.input("proxy2", (kernels.basalt_q5k_proxy_bytes(N_CAP),), U8, host=not bad)
    proxy_top1(g, "y", x1, proxy, rows, mp)
    return {}


def _proxy_size(g, src, bad, tmp):
    """A copy of n_cap + 32 rows' bytes: one tile more, which would move the scales the kernels read."""
    x1, (rows, mp, _) = g.input("x1", (1, QCOL), U8), draft_inputs(g)
    proxy = g.input("proxy2", (kernels.basalt_q5k_proxy_bytes(N_CAP + 32 if bad else N_CAP),), U8, host=True)
    proxy_top1(g, "y", x1, proxy, rows, mp)
    return {}


def _from_pack(kind, value, offset=16, extra=0):
    """``kind`` over 64 rows (the proxy: n_rows = n_cap = 64) with the map, its ``value`` (``rows``: a q5_k weight;
    ``xq``: the q8_1 column; ``map``) a pack entry at byte 16 holding exactly its bytes (valid), or at byte ``offset``
    holding ``extra`` bytes more; the other values the program's and the session's inputs."""
    def build(g, src, bad, tmp):
        v, nbytes = {"rows": lambda: (g.weight("p", (64, 2560), "q5_k"), 64 * ROW),
                     "xq": lambda: (g.const("p", (1, QCOL), U8), QCOL),
                     "map": lambda: (g.const("p", (64,), I32), 4 * 64)}[value]()
        x1 = v if value == "xq" else g.input("x1", (1, QCOL), U8)
        rows = v if value == "rows" else g.input("dhead", (64, ROW), U8, host=True)
        mp = v if value == "map" else g.input("dvocab", (64,), I32, host=True)
        if kind == "basalt_q5k_proxy_top1":
            proxy = g.input("proxy", (kernels.basalt_q5k_proxy_bytes(64),), U8, host=True)
            proxy_top1(g, "y", x1, proxy, rows, mp, n_rows=64, n_cap=64)
        else:
            (head if kind == "basalt_q5k_head" else top1)(g, "y", x1, rows, mp)
        return {"packs": [pack(tmp, {"p": (offset, nbytes + extra) if bad else (16, nbytes)})]}
    return build


REFUSALS = [
    *[(f"dynamic-T {k}", _dynamic(k), "a dynamic-T program cannot run them") for k in KINDS],
    *[(f"{k} with a second map", _arity(k), r"4 inputs; it takes \(xq, rows\[, map\]\)") for k in KINDS[:2]],
    ("proxy top-1 with a second map", _arity("basalt_q5k_proxy_top1"),
     r"5 inputs; it takes \(xq, proxy, rows, map\)"),
    ("head on Q4_K rows", _rows_type, r"of ggml type \['q5_k'\]"),
    ("head rows an input the program allocates", _rows_not_host, "is an input each program allocates"),
    ("head host rows not whole Q5_K blocks", _host_rows(shape=(64, ROW - 60)),
     r"u8 \[64, 1700\], not u8 \[rows, whole 176-byte"),
    ("head host rows not u8", _host_rows(dtype=F32), r"'dhead' are f32 \[64, 1760\], not u8 \[rows, whole 176-byte"),
    ("head host rows of rank 3", _host_rows(shape=(64, ROW, 2)),
     r"'dhead' are u8 \[64, 1760, 2\], not u8 \[rows, whole 176-byte"),
    ("head host rows of no blocks", _host_rows(shape=(64, 0)), r"unsupported shape \(n_in 0, ncols 1\)"),
    *[(f"head host rows named after StepState's {f}", _host_rows(name=n),
       f"the host input '{n}' is named after StepState's '{f}'")
      for n, f in (("draft_tokens", "draft_tokens"), ("tokens", "pending_tokens"))],
    *[(f"{k} reads a row past its rows", _rows_held(k), f"'dhead' holds {N_DVOCAB} rows; the op reads {N_DVOCAB + 1}")
      for k in KINDS],
    ("head of 16 columns", _head_shape((8, 10), (16, 10)), r"unsupported shape \(n_in 2560, ncols 16\)"),
    ("head of no columns", _head_shape((1, 10), (0, 10)), r"unsupported shape \(n_in 2560, ncols 0\)"),
    ("head rows of 65 blocks", _head_shape((1, 64), (1, 65)), r"unsupported shape \(n_in 16640, ncols 1\)"),
    ("head of 3 columns of 64 blocks", _head_shape((2, 64), (3, 64)), r"unsupported shape \(n_in 16384, ncols 3\)"),
    ("head q8_1 rows of another width", _head_xq_width, r"not u8 \[2, 2880\] \(q8_1 rows of the rows' width\)"),
    ("head logits not F32", _head_logits, r"not f32 \[2, 4000\] \(contiguous columns of the logits\)"),
    *[(f"{k} map a row short", _map_short(k), rf"\[{N_CAP - 1}\]; it maps the {N_CAP} rows read") for k in KINDS],
    ("head map not I32", _head_map(dtype=DType.U32), r"the map 'dvocab' is u32"),
    ("head map of rank 2", _head_map(shape=(N_DVOCAB, 2)), rf"the map 'dvocab' is i32 \[{N_DVOCAB}, 2\]"),
    ("top-1 on rows of 2048", _top1_n_in("basalt_q5k_top1"), "rows of 2048 values; q5k_head_top1 takes 2560"),
    ("proxy top-1 on rows of 2048", _top1_n_in("basalt_q5k_proxy_top1"),
     "rows of 2048 values; the 2-bit copy holds rows of 2560"),
    *[(f"{k} of two columns", _one_column(k), r"not u8 \[1, 2880\] \(one q8_1 column") for k in KINDS[1:]],
    ("top-1 logits not F32", _out_dtype("basalt_q5k_top1", 0), r"\(the logits of one column\)"),
    *[(f"{k} id not I32", _out_dtype(k, 1), r"\(the best row's id\)") for k in KINDS[1:]],
    *[(f"{k} probability not F32", _out_dtype(k, 2), r"\(its probability\)") for k in KINDS[1:]],
    ("proxy top-1 without the map", _proxy_map, r"3 inputs; .* basalt refuses it without the map"),
    ("proxy top-1 without n_rows", _proxy_attrs("n_rows"), f"must be ints, got None and {N_CAP}"),
    ("proxy top-1 without n_cap", _proxy_attrs("n_cap"), f"must be ints, got {N_CAP} and None"),
    ("proxy top-1 n_rows past n_cap", _proxy_rows_past_cap,
     rf"rows out of range \(row0 0, n 0, n_rows {N_CAP}, n_cap {N_CAP - 1}\)"),
    ("proxy top-1 copy not a host input", _proxy_not_host, "the 2-bit copy 'proxy2' is not a host input"),
    ("proxy top-1 copy of another size", _proxy_size, r"\(the 2-bit copy of n_cap rows\)"),
    *[(f"{k} rows off 2 bytes", _from_pack(k, "rows", 1), "'p' binds at byte 1 of .*; slot 0 needs 2-byte alignment")
      for k in KINDS],
    *[(f"{k} q8_1 column off 4 bytes", _from_pack(k, "xq", 2), "'p' binds at byte 2 of .*; slot 1 needs 4-byte")
      for k in KINDS],
    *[(f"{k} map off 4 bytes", _from_pack(k, "map", 2), f"'p' binds at byte 2 of .*; slot {s} needs 4-byte")
      for k, s in zip(KINDS, (1, 3, 2))],            # the mask's slot, the top-1's, the score's
    *[(f"{k} rows entry a row short", _from_pack(k, "rows", extra=-ROW),
       r"'p' is 112640 bytes of q5_k \[64, 2560\], but its pack entry holds 110880") for k in KINDS],
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
