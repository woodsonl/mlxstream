"""basalt's prompt-path op kinds (monolith/ops/basalt_prompt.py) as compiler/emit_basalt_prompt.py emits them over the
synthetic .basalt file of tests/contract/test_basalt_ops.py: both kinds registered and covered; every dispatch equal to
monolith/kernels.py's builders (function, macros, source, grid, threadgroup, params, written slot), with StepState bound
at slot 15 and the row macros of the program's form (prompt_rows); no per-step value in a params record (invariant 4);
the GEMM's static slices tiling every product once; and each refusal, a valid input with exactly one defect refused by
that defect's own check (its message). The GPU side is tests/kernels/test_basalt_prompt_program.py."""

import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import HANDLERS, _Ctx, basalt_windows
from monolith.compiler.emit_basalt_prompt import BASALT_PROMPT_HANDLERS, prompt_rows
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.core.shapes import N_INJ, T
from monolith.core.step_state import StepStateLayout
from monolith.ops import OPS
from monolith.runtime.program import Program
from tests.contract.test_basalt_ops import emit, op, pack, src, weight  # noqa: F401  (src: the .basalt file fixture)
from tests.contract.test_basalt_windows import PROFILE

F32, F16, BF16 = DType.F32, DType.F16, DType.BF16
KINDS = ("basalt_prompt_gemm", "basalt_prompt_convert")
LAYOUT = StepStateLayout(t_max=128)
TILE_K2560 = 64 * 64 * 2560          # one 64 x 64 tile's multiply-adds at K 2560: a budget that slices rows and columns

# The products basalt_prompt_gemm runs on this pack, (N, K) (spec-kernels-gemm.md section 3; prefill.cpp lines):
# :1569 and :1571 the hc down and up (Gemm::bf16 on the copies' BF16 images, hc8's entry), :1572 the inject, :1585 and
# :1535 qkv and the PLE key, :1586 the GDN gate, :1587-1588 alpha and beta, :1592 and :1688 ssm_out and attn_output,
# :1605-1606, :1609 and :1698 at 512 x 2560, :1607 attn_q, :1608 the indexer's k, :1702-1703 the shared expert's gate and
# up, :1705 its down, :1707 its scalar gate, :1538 the PLE value; and N 4 and 72 at K 2560 (column tiles cut short).
PRODUCTS = ((320, 10240), (10240, 320), (4, 10240), (10240, 2560), (6144, 2560), (48, 2560), (2560, 6144), (512, 2560),
            (12288, 2560), (128, 2560), (640, 2560), (2560, 640), (1, 2560), (2560, 2560), (4, 2560), (72, 2560))


def prompt_graph(g, src, rows=T):
    """Both kinds at ``rows`` rows: to_f16 of x [rows, 2560] F32, and Gemm::bf16 of xb [rows, 2560] BF16 by the file's
    two BF16 matrices (128 outputs: two column tiles; 1 output: one partial tile)."""
    x = g.input("x", (rows, 2560), F32)
    op(g, "basalt_prompt_convert", [x], [g.value("x16", (rows, 2560), F16)])
    xb = g.input("xb", (rows, 2560), BF16)
    for name in ("mmvf.q", "mmvf.gate"):
        w = weight(g, f"dense/{name}", src)
        op(g, "basalt_prompt_gemm", [xb, w], [g.value(f"y.{name}", (rows, w.shape[0]), F32)], entry="bf16")


def row_macros(rows, static):
    """The macros prompt_rows(st) reads: T_SRC 0 in a dynamic-T program, 2 with the count for a static one."""
    m = {"STEP_STATE": "1", "T_SRC": "2" if static else "0", "ROWS_MAX": f"{rows}u"}
    return {**m, "T_STATIC_ROWS": f"{rows}u"} if static else m


def record(prog, o):
    """The params record a dispatch binds (its one buffer of role params)."""
    (prm,) = [prog.buffers[b] for _, b, _ in o.bindings if prog.buffers[b].role == "params"]
    return prm.init


# ---- registration, and each dispatch against the builders ----------------------------------------------------------

def test_registered_and_covered(src):
    for kind in KINDS:
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
    g = Graph("step")
    prompt_graph(g, src)
    assert {o.kind for o in g.ops} == set(KINDS)
    check_coverage(g, PROFILE)


# (dynamic-T program, the values' rows, the program's t): T rows in a dynamic program read t_this_step; a static count
# in a dynamic program and every row count of a static one (T included: rows_of reports it as source 0 there too, and a
# static program's advance resets t_this_step to 1) compile T_SRC 2.
FORMS = [(True, T, 128), (True, 17, 128), (False, T, 17), (False, 5, 17)]


@pytest.mark.parametrize("dynamic, rows, t", FORMS, ids=["dynamic", "dynamic-static-rows", "static", "static-rows"])
def test_dispatches_bind_step_state(src, monkeypatch, dynamic, rows, t):
    """Every dispatch of both kinds is the builders' and binds (15, step_state, 0) once, with STEP_STATE 1, ROWS_MAX
    the compiled rows B and T_SRC 0 (dynamic-T, T rows) or T_SRC 2 with T_STATIC_ROWS B. Under a budget of one tile
    the products split into static slices, one dispatch each in basalt_prompt_gemm_slices' order, each with its own
    record (row0, col0) and the grid of its block; the conversion is one dispatch over the bound's elements."""
    monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", TILE_K2560)
    g = Graph("step")
    prompt_graph(g, src, rows)
    prog = emit(g, src, t, dynamic_t=dynamic, layout=LAYOUT)
    _, bound = basalt_windows(src, g.values.values(), t)
    b = t if rows is T else rows
    macros = row_macros(b, not dynamic or rows is not T)
    source = kernels.basalt_prompt_gemm_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
    # per dispatch: (function, its macro, the value slots, params, grid, label, written slots)
    want = [("basalt_prompt_convert", {"CV_BF16": "0"}, [(0, "x", 0), (1, "x16", 0)],
             kernels.basalt_prompt_convert_params(2560), (-(-b * 2560 // 256), 1, 1), "basalt_prompt_convert:to_f16:x", [1])]
    for name in ("mmvf.q", "mmvf.gate"):
        n = src.entries[f"dense/{name}"].shape[0]
        values = [(0, "xb", 0), (1, *bound[f"dense/{name}"]), (2, f"y.{name}", 0)]
        want += [("basalt_prompt_gemm", {"GEMM_BF16": "1"}, values, kernels.basalt_prompt_gemm_params(n, 2560, row0, col0),
                  (-(-ncols // 64), nrows // 64, 1), f"basalt_prompt_gemm:bf16:dense/{name}", [2])
                 for row0, nrows, col0, ncols in kernels.basalt_prompt_gemm_slices(b, n, 2560)]
    assert [w[5] for w in want].count("basalt_prompt_gemm:bf16:dense/mmvf.q") == (4 if b > 64 else 2)   # sliced
    assert len(prog.ops) == len(want)
    for o, (fn, macro, values, params, grid, label, writes) in zip(prog.ops, want):
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, {**macros, **macro}, source), label
        assert (o.name, o.grid, o.threadgroup, o.threadgroup_memory, o.meta["writes"]) == (
            label, grid, (256, 1, 1), [], writes)
        slot, prm, off = o.bindings[len(values)]
        assert o.bindings == values + [(slot, prm, off), (15, prog.step_state, 0)], label
        assert (slot, off, prog.buffers[prm].role, prog.buffers[prm].init) == (len(values), 0, "params", params), label


def test_rows_from_reads_its_value(src):
    """prompt_rows' ``rows_from``: an op whose only output is a placeholder takes its rows from the value it names (the
    same rule as an output's), and without it from that output, a static count."""
    g = Graph("step")
    k = g.input("k", (T, 2560), F32)
    placeholder = g.value("ph", (1,), DType.U32)
    o = op(g, "basalt_prompt_convert", [k], [placeholder])
    for dynamic, want in ((True, row_macros(64, False)), (False, row_macros(64, True))):
        ctx = _Ctx(Program(kernels={}, buffers={}, ops=[], layout=LAYOUT), [src], LAYOUT, 64, 1, 256, dynamic_t=dynamic)
        assert prompt_rows(ctx, o, rows_from=k) == (64, want)
        assert prompt_rows(ctx, o) == (1, row_macros(1, True))


def test_per_step_values_stay_out_of_params(src, monkeypatch):
    """Invariant 4: the program compiled at bound B serves every chunk, so its records hold geometry only. Two emissions
    give byte-identical records and macros; every product record is exactly (n, k, row0, col0), its slice's 64-row and
    64-column origin, every conversion record exactly (n,): no row count, position or pointer."""
    monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", TILE_K2560)
    progs = []
    for _ in range(2):
        g = Graph("step")
        prompt_graph(g, src)
        progs.append(emit(g, src, 128, dynamic_t=True, layout=LAYOUT))
    a, b = ([(record(p, o), p.kernels[o.kernel].macros) for o in p.ops] for p in progs)
    assert a == b
    origins = {(n, row0, col0) for n in (128, 1) for row0, _, col0, _ in kernels.basalt_prompt_gemm_slices(128, n, 2560)}
    seen = set()
    for o in progs[0].ops:
        rec = record(progs[0], o)
        if o.name.startswith("basalt_prompt_convert:"):
            assert struct.unpack("<I", rec) == (2560,)
            continue
        n, k, row0, col0 = struct.unpack("<4I", rec)
        assert k == 2560 and row0 % 64 == 0 and col0 % 64 == 0 and (n, row0, col0) in origins, o.name
        seen.add((n, row0, col0))
    assert seen == origins and len(origins) == 6


@pytest.mark.parametrize("budget", ["tile-", "c0", "c64", "cN", "constant"])
@pytest.mark.parametrize("rows", [1, 17, 256, 512])
def test_slices_cover_the_product(monkeypatch, rows, budget):
    """basalt_prompt_gemm_slices tiles [0, 64 * ceil(B / 64)) x [0, N) exactly once for every product of the pack, under
    a budget below one tile, budgets that make the column slice 0 (rows sliced instead), 64 and N columns wide, and the
    module's own constant: every row slice a multiple of 64 tall, every column slice but the one ending at N a multiple
    of 64 wide (the property the kernel's t < R, n < N stores rely on), each within the budget or a single tile."""
    bp = 64 * -(-rows // 64)
    for n, k in PRODUCTS:
        macs = {"tile-": 64 * 64 * k - 1, "c0": bp * 64 * k - 1, "c64": bp * 64 * k, "cN": bp * 64 * -(-n // 64) * k,
                "constant": kernels.PROMPT_GEMM_MACS}[budget]
        monkeypatch.setattr(kernels, "PROMPT_GEMM_MACS", macs)
        slices = kernels.basalt_prompt_gemm_slices(rows, n, k)
        cover = np.zeros((bp, n), np.uint8)
        for row0, nrows, col0, ncols in slices:
            assert row0 % 64 == 0 and nrows % 64 == 0 and nrows > 0 and row0 + nrows <= bp, (n, k, row0, nrows)
            assert col0 % 64 == 0 and ncols > 0 and col0 + ncols <= n, (n, k, col0, ncols)
            assert ncols % 64 == 0 or col0 + ncols == n, (n, k, col0, ncols)
            assert nrows * 64 * -(-ncols // 64) * k <= macs or (nrows == 64 and ncols <= 64), (n, k, row0, col0)
            cover[row0:row0 + nrows, col0:col0 + ncols] += 1
        assert (cover == 1).all(), (n, k)
        widths = {ncols for _, _, _, ncols in slices}
        if budget == "tile-":                   # single tiles, each over the budget
            assert len(slices) == bp // 64 * -(-n // 64) and widths <= set(range(1, 65)), (n, k)
        if budget == "c0" and bp > 64:          # rows sliced: Bp - 64 rows, then 64
            assert sorted({nrows for _, nrows, _, _ in slices}) == [64, bp - 64], (n, k)
        if budget == "cN":
            assert widths == {n} and len(slices) == 1, (n, k)


# ---- refusals: each case's valid input emits, and the same input with its one defect is refused by its own check ---

def gemm(g, src, entry="bf16", w="dense/mmvf.q", x_dtype=BF16, y_dtype=F32, y_extra=0, rows=T):
    """``basalt_prompt_gemm`` of x [rows, K] by ``w`` into y [rows, N + y_extra]."""
    wv = weight(g, w, src)
    x = g.input("x", (rows, wv.shape[1]), x_dtype)
    op(g, "basalt_prompt_gemm", [x, wv], [g.value("y", (rows, wv.shape[0] + y_extra), y_dtype)], entry=entry)


def convert(g, x_dtype=F32, y_dtype=F16, y_extra=0):
    """``basalt_prompt_convert`` of x [T, 64] into y [T, 64 + y_extra]."""
    x = g.input("x", (T, 64), x_dtype)
    op(g, "basalt_prompt_convert", [x], [g.value("y", (T, 64 + y_extra), y_dtype)])


def _gemm(**defect):
    def build(g, src, bad, tmp):
        gemm(g, src, **(defect if bad else {}))
        return {}
    return build


def _convert(**defect):
    def build(g, src, bad, tmp):
        convert(g, **(defect if bad else {}))
        return {}
    return build


def _k16(g, src, bad, tmp):
    """A BF16 matrix of 4 rows over K 40 (valid: 48) from a pack entry at byte 16."""
    k = 40 if bad else 48
    w = g.const("p", (4, k), BF16)
    op(g, "basalt_prompt_gemm", [g.input("x", (T, k), BF16), w], [g.value("y", (T, 4), F32)], entry="bf16")
    return {"packs": [pack(tmp, {"p": (16, 4 * k * 2)})]}


REFUSALS = [
    ("entry outside bf16 and native", _gemm(entry="f16"), r"attr entry must be one of \('bf16', 'native'\), got 'f16'"),
    ("bf16 entry on GGUF blocks", _gemm(w="gguf/mmvq.12"), r"is not a weight or constant .* of ggml type \['bf16'\]"),
    # the valid native op (F16 rows, GGUF blocks) until dequant's kernels merge into this branch
    ("native before dequant", _gemm(entry="native", w="gguf/mmvq.12", x_dtype=F16),
     "dequantizes through dequant's basalt_dequant_g32, which this branch does not have yet"),
    ("bf16 entry on F16 rows", _gemm(x_dtype=F16), r"'x' is f16 \[2, 2560\], not bf16 \[2, 2560\]"),
    ("product output not F32", _gemm(y_dtype=F16), r"'y' is f16 \[2, 128\], not f32 \[2, 128\]"),
    ("product output wider than the weight's rows", _gemm(y_extra=1), r"'y' is f32 \[2, 129\], not f32 \[2, 128\]"),
    ("K not a multiple of 16", _k16, "k must be a positive multiple of 16, got 40"),
    ("rows from n_inject", _gemm(rows=N_INJ), r"'y' has the rows n_inject \(T_SRC 1\)"),
    ("convert to F32", _convert(y_dtype=F32), r"'y' is f32; the output is F16 \(to_f16\) or BF16 \(to_bf16\)"),
    # the valid to_bf16 op until dequant's bf16_rne_nonan merges into this branch
    ("convert to BF16 before dequant", _convert(y_dtype=BF16), "bf16_rne_nonan, which this branch does not have yet"),
    ("convert of F16 rows", _convert(x_dtype=F16), r"'x' is f16 \[2, 64\], not f32 \[2, 64\]"),
    ("convert output of another width", _convert(y_extra=1), r"'y' is f16 \[2, 65\], not f16 \[2, 64\]"),
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


# ---- the hyper-connection kinds (HC-6; kernels/common/basalt_prompt_hc.metal) -------------------------------------

HC_KINDS = {"basalt_prompt_hc_broadcast": "prompt_gr_broadcast", "basalt_prompt_hc_norm": "prompt_gr_norm_rs",
            "basalt_prompt_hc_silu": "prompt_gr_silu", "basalt_prompt_hc_mix": "prompt_gr_mix_r",
            "basalt_prompt_hc_write": "prompt_gr_write", "basalt_prompt_hc_write_norm": "prompt_gr_write_norm_rs"}
HC_N, HC_LR = 2560, 320                         # basalt's N and LR (src/prefill/kernels.cu:18-19)
HC_EPS = "as_type<float>(0x358637BDu)"          # basalt's EPS 1e-6 (prefill.cpp:57) as float32 bits
HC_EPS0 = "as_type<float>(0x00000000u)"         # the kinds without eps compile 0


def hc_graph(g, rows=T, n=HC_N, lr=HC_LR):
    """One half of a layer's hyper-connection at ``rows`` rows, every kind once and the mix in both forms: broadcast
    of e into r; norm of r by w ``[4n]`` into rs, xn; silu of lo; mix of r, rs, w, gated into mixed, mixed_bf, mixed_h
    and, without the BF16 image (the drafter's KV pass), into mixed2, mixed_h2; write_norm of r, bo, inj by w_next
    ``[1, 4n]`` into r2, rs2, xn2; write of r2, bo, inj into r3."""
    def inp(name, cols):
        return g.input(name, (rows, cols), F32)

    def out(name, cols, dtype=F32):
        return g.value(name, (rows, cols), dtype)

    e, lo, gated, bo, inj = inp("e", n), inp("lo", lr), inp("gated", 4 * n), inp("bo", n), inp("inj", 4)
    w, w_next = g.input("w", (4 * n,), F32), g.input("w_next", (1, 4 * n), F32)
    r, rs, r2 = out("r", 4 * n), out("rs", 4), out("r2", 4 * n)
    op(g, "basalt_prompt_hc_broadcast", [e], [r])
    op(g, "basalt_prompt_hc_norm", [r, w], [rs, out("xn", 4 * n, BF16)], eps=1e-6)
    op(g, "basalt_prompt_hc_silu", [lo], [out("lo16", lr, BF16)])
    mix = [r, rs, w, gated]
    op(g, "basalt_prompt_hc_mix", mix, [out("mixed", n), out("mixed_bf", n, BF16), out("mixed_h", n, F16)])
    op(g, "basalt_prompt_hc_mix", mix, [out("mixed2", n), out("mixed_h2", n, F16)])
    op(g, "basalt_prompt_hc_write_norm", [r, bo, inj, w_next], [r2, out("rs2", 4), out("xn2", 4 * n, BF16)], eps=1e-6)
    op(g, "basalt_prompt_hc_write", [r2, bo, inj], [out("r3", 4 * n)])


def test_prompt_hc_registered():
    """HC-6: the six kinds registered (MAP over rows, each bound to its own name on every profile), each with its
    handler in BASALT_PROMPT_HANDLERS, the one emit.HANDLERS calls."""
    for kind in HC_KINDS:
        od = OPS.get(kind)
        assert (od.klass, od.domain_kind, od.binding_for(PROFILE.key).kernel) == (OpClass.MAP, "rows", kind), kind
        assert HANDLERS[kind] is BASALT_PROMPT_HANDLERS[kind], kind


@pytest.mark.parametrize("dynamic, t", [(True, 128), (False, 17)], ids=["dynamic", "static"])
def test_prompt_hc_covered_and_emits(src, dynamic, t):
    """HC-6: a graph using all six kinds passes the coverage guard and emits, one dispatch per op (one slice: B = t
    is below PROMPT_HC_ROWS), each the builders': its entry of basalt_prompt_hc_source with the rows' macros and HC_N,
    HC_LR, MIX_BF, HC_EPS (silu at HC_N 256, its file's least; HC_LR 0, MIX_BF 0 and eps 0 where the kernel reads
    none), the values at the kernel's slots, the slice's record (row0 0) at slot 7 and StepState at 15, basalt's launch
    shape (T * HC threadgroups of 256 for the two norms, blocks_for(T * width) for the others, kernels.cu:867-900), the
    outputs' slots as its writes and the label of its kind and weight (else its first input)."""
    g = Graph("step")
    hc_graph(g)
    assert {o.kind for o in g.ops} == set(HC_KINDS)
    check_coverage(g, PROFILE)
    prog = emit(g, src, t, dynamic_t=dynamic, layout=LAYOUT)
    source = kernels.basalt_prompt_hc_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
    n, lr = HC_N, HC_LR

    def blocks(width):
        return -(-t * width // 256)

    # per dispatch: (kind, HC_N, HC_LR, MIX_BF, HC_EPS, the values bound from slot 0 (None: no binding), grid, writes,
    # the label's value)
    want = [("broadcast", n, 0, 0, HC_EPS0, ["e", "r"], blocks(4 * n), [1], "e"),
            ("norm", n, 0, 0, HC_EPS, ["r", "w", "rs", "xn"], 4 * t, [2, 3], "w"),
            ("silu", 256, lr, 0, HC_EPS0, ["lo", "lo16"], blocks(lr), [1], "lo"),
            ("mix", n, 0, 1, HC_EPS0, ["r", "rs", "w", "gated", "mixed", "mixed_bf", "mixed_h"], blocks(n), [4, 5, 6],
             "w"),
            ("mix", n, 0, 0, HC_EPS0, ["r", "rs", "w", "gated", "mixed2", None, "mixed_h2"], blocks(n), [4, 6], "w"),
            ("write_norm", n, 0, 0, HC_EPS, ["r", "bo", "inj", "w_next", "r2", "rs2", "xn2"], 4 * t, [4, 5, 6],
             "w_next"),
            ("write", n, 0, 0, HC_EPS0, ["r2", "bo", "inj", "r3"], blocks(4 * n), [3], "r2")]
    assert len(prog.ops) == len(want)
    for o, (kind, hc_n, hc_lr, mix_bf, eps, values, grid, writes, label) in zip(prog.ops, want):
        kind = f"basalt_prompt_hc_{kind}"
        k = prog.kernels[o.kernel]
        macros = {**row_macros(t, not dynamic), "HC_N": str(hc_n), "HC_LR": str(hc_lr), "MIX_BF": str(mix_bf),
                  "HC_EPS": eps}
        assert (k.function, k.macros, k.source) == (HC_KINDS[kind], macros, source), kind
        assert (o.name, o.grid, o.threadgroup, o.threadgroup_memory, o.meta["writes"]) == (
            f"{kind}:{label}", (grid, 1, 1), (256, 1, 1), [], writes), kind
        prm = o.bindings[-2][1]
        assert o.bindings == [(s, v, 0) for s, v in enumerate(values) if v] + [(7, prm, 0), (15, prog.step_state, 0)]
        assert (prog.buffers[prm].role, prog.buffers[prm].init) == ("params", kernels.basalt_prompt_hc_params(0)), kind


def hc_op(g, kind, rows=T, n=HC_N, lr=HC_LR, cols=None, dtype=(), outs=3, eps=1e-6, w=None):
    """One op of ``basalt_prompt_hc_<kind>`` over fresh inputs at ``rows`` rows: hc_graph's shapes and dtypes at the
    defaults, or one defect: streams of ``n`` values (silu: rows of ``lr``), ``r`` of ``cols`` columns, ``dtype`` =
    ``(output, dtype)`` for one output, a mix of ``outs`` outputs (1 to 4), the attr ``eps`` (None: absent), or the
    norm weights ``w`` = ``(shape, dtype)``."""
    bad = dict([dtype]) if dtype else {}

    def inp(name, c):
        return g.input(name, (rows, c), F32)

    def out(name, c, dt=F32):
        return g.value(name, (rows, c), bad.get(name, dt))

    def weights(name, shape):
        return g.input(name, *(w or (shape, F32)))

    r = cols or 4 * n
    if kind == "broadcast":
        ins, os = [inp("e", n)], [out("r", r)]
    elif kind == "norm":
        ins, os = [inp("r", r), weights("w", (4 * n,))], [out("rs", 4), out("xn", 4 * n, BF16)]
    elif kind == "silu":
        ins, os = [inp("lo", lr)], [out("lo16", lr, BF16)]
    elif kind == "mix":
        ins = [inp("r", r), inp("rs", 4), weights("w", (4 * n,)), inp("gated", 4 * n)]
        names = {1: ["mixed"], 2: ["mixed", "mixed_h"], 3: ["mixed", "mixed_bf", "mixed_h"],
                 4: ["mixed", "mixed_bf", "mixed_h", "mixed_x"]}[outs]
        os = [out(m, n, {"mixed_bf": BF16, "mixed_h": F16}.get(m, F32)) for m in names]
    elif kind == "write":
        ins, os = [inp("r", r), inp("bo", n), inp("inj", 4)], [out("r2", 4 * n)]
    else:
        ins = [inp("r", r), inp("bo", n), inp("inj", 4), weights("w_next", (1, 4 * n))]
        os = [out("r2", 4 * n), out("rs", 4), out("xn", 4 * n, BF16)]
    attrs = {"eps": eps} if kind in ("norm", "write_norm") and eps is not None else {}
    op(g, f"basalt_prompt_hc_{kind}", ins, os, **attrs)


STREAMS = ("broadcast", "norm", "mix", "write", "write_norm")      # the kinds with streams (silu has none)
OTHER = {F32: F16, BF16: F16, F16: BF16}                          # a wrong dtype for each output dtype
OUT_DTYPES = [("broadcast", "r", F32), ("norm", "rs", F32), ("norm", "xn", BF16), ("silu", "lo16", BF16),
              ("mix", "mixed", F32), ("mix", "mixed_bf", BF16), ("mix", "mixed_h", F16), ("write", "r2", F32),
              ("write_norm", "r2", F32), ("write_norm", "rs", F32), ("write_norm", "xn", BF16)]
EPS_BAD = [("nan", float("nan")), ("0", 0.0), ("1e-45", 1e-45), ("2^-103", 2.0 ** -103), ("absent", None)]

# (id, kind, the valid op's arguments, the defective op's, the refusal's message)
HC_REFUSALS = [
    # every stream kind: the file compiles only at HC_N >= 256 (gr_write_norm_rs's v[HC_N / 256]), so 256 is the least
    *[(f"{k} streams below 256", k, {"n": 256}, {"n": 252}, "streams of 252 values; the hyper-connection kernels take "
       "at least 256") for k in STREAMS],
    *[(f"{k} streams not whole sweeps", k, {}, {"n": 2564}, "streams of 2564 values are not whole sweeps of 256 "
       "threads") for k in ("norm", "write_norm")],
    *[(f"{k} streams not a multiple of 4", k, {}, {"n": 2562}, "streams of 2562 values, not a multiple of 4")
      for k in ("mix", "write", "broadcast")],
    ("silu rows not a multiple of 4", "silu", {}, {"lr": 322}, "rows of 322 values, not a positive multiple of 4"),
    ("silu rows of no values", "silu", {"lr": 4}, {"lr": 0}, "rows of 0 values, not a positive multiple of 4"),
    # broadcast's n is the embedding's, so five streams of it; the others take n from r: columns not a multiple of 4
    *[(f"{k} r not four streams", k, {}, {"cols": 5 * HC_N if k == "broadcast" else 4 * HC_N + 2},
       r"'r' has \d+ columns, not four streams of 2560 \(basalt's HC = 4\)") for k in STREAMS],
    ("norm w as BF16", "norm", {}, {"w": ((4 * HC_N,), BF16)}, r"'w' is bf16 \[10240\], not 10240 f32 values"),
    *[(f"{k} {v} as {OTHER[dt]}", k, {}, {"dtype": (v, OTHER[dt])}, rf"'{v}' is {OTHER[dt]} \[2, \d+\], not {dt} ")
      for k, v, dt in OUT_DTYPES],
    *[(f"mix of {c} outputs", "mix", {}, {"outs": c}, f"4 inputs and {c} outputs; it takes 4 inputs and 2 or 3 outputs")
      for c in (1, 4)],
    # the least eps accepted is 2^-102 (the row norm's plain / and + give IEEE's bits from there up)
    *[(f"{k} eps {name}", k, {"eps": 2.0 ** -102}, {"eps": e}, r"attr eps must be a finite float32 of at least 2\^-102")
      for k in ("norm", "write_norm") for name, e in EPS_BAD],
    *[(f"{k} rows from n_inject", k, {}, {"rows": N_INJ}, r"has the rows n_inject \(T_SRC 1\)")
      for k in ("broadcast", "norm", "silu", "mix", "write", "write_norm")],
    # the kernels index in 32 bits: B * 4n <= 2^32 - 256 (419430 * 10240 = 2^32 - 4096; 419431 rows overflow)
    ("norm past 32-bit indices", "norm", {"rows": 419430}, {"rows": 419431},
     "419431 rows of 10240 values overflow the kernels' 32-bit indices"),
]


@pytest.mark.parametrize("kind, good, bad, why", [r[1:] for r in HC_REFUSALS], ids=[r[0] for r in HC_REFUSALS])
def test_prompt_hc_refusals(src, kind, good, bad, why):
    """HC-6's refusals, one case each: the op of ``kind`` emits with ``good``, and the same op with ``bad`` instead,
    its one defect, is refused by that defect's own check (its message)."""
    for args in (good, bad):
        g = Graph("step")
        hc_op(g, kind, **args)
        if args is good:
            emit(g, src, 2)
            continue
        with pytest.raises(ValueError, match=why):
            emit(g, src, 2)


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic", "static"])
def test_prompt_hc_per_step_values_stay_out_of_params(src, monkeypatch, dynamic):
    """HC-6, invariant 4: the program compiled at bound B serves every chunk, so a hyper-connection dispatch's macros
    are exactly the rows' (STEP_STATE, T_SRC, ROWS_MAX, and T_STATIC_ROWS in a static program) and the op's shapes'
    (HC_N, HC_LR, HC_EPS, MIX_BF), and its one params record, at slot 7, holds only its slice's first row, a
    compile-time value. With PROMPT_HC_ROWS at 64, B = 256 gives each op four dispatches, of rows 0, 64, 128 and 192,
    each with a 64-row grid and StepState at slot 15, which differ only in that record; two emits are byte-identical."""
    monkeypatch.setattr(kernels, "PROMPT_HC_ROWS", 64)
    layout = StepStateLayout(t_max=256)
    progs = []
    for _ in range(2):
        g = Graph("step")
        hc_graph(g)
        progs.append(emit(g, src, 256, dynamic_t=dynamic, layout=layout))
    assert progs[0].to_json() == progs[1].to_json()
    prog = progs[0]
    keys = {"STEP_STATE", "T_SRC", "ROWS_MAX", "HC_N", "HC_LR", "HC_EPS", "MIX_BF"} | (set() if dynamic else
                                                                                     {"T_STATIC_ROWS"})
    kinds = ("broadcast", "norm", "silu", "mix", "mix", "write_norm", "write")             # hc_graph's ops, in order
    # a 64-row slice: 4 * 64 threadgroups for the two row kernels, 64 * width / 256 for the element kernels
    grids = {"broadcast": 64 * 4 * HC_N // 256, "norm": 4 * 64, "silu": 64 * HC_LR // 256, "mix": 64 * HC_N // 256,
             "write_norm": 4 * 64, "write": 64 * 4 * HC_N // 256}
    assert len(prog.ops) == 4 * len(kinds)

    def but_record(o):
        return o.kernel, [b for b in o.bindings if b[0] != 7], o.threadgroup, o.meta["writes"]

    for i, o in enumerate(prog.ops):
        kind, row0 = kinds[i // 4], 64 * (i % 4)
        assert o.name.startswith(f"basalt_prompt_hc_{kind}:") and o.grid == (grids[kind], 1, 1), o.name
        assert set(prog.kernels[o.kernel].macros) == keys, o.name
        assert [(s, off) for s, b, off in o.bindings if prog.buffers[b].role == "params"] == [(7, 0)], o.name
        assert record(prog, o) == struct.pack("<I", row0), o.name
        assert [b for b in o.bindings if b[0] == 15] == [(15, prog.step_state, 0)], o.name
        assert but_record(o) == but_record(prog.ops[i - i % 4]), o.name
