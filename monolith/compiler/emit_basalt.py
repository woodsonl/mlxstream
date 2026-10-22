"""The emitter's handlers for basalt's ops (``monolith/ops/basalt.py``, ``monolith/ops/basalt_gdn.py``), merged into
``emit.HANDLERS``. Each handler emits one dispatch with the macros, grid, threadgroup and params of
``monolith/kernels.py``'s builders, after the checks basalt's launchers make (shapes, column counts, pointer alignment)
and the ones its call sites satisfy (the IEEE q8_1 rows feed only the routed experts; outputs are contiguous columns); a
static program only (one per window width; the GDN commit's, one per session, run after every window).
"""

# The launcher checks are adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:1465-1475, :1490-1494,
# :1549-1552 and :1563-1568, src/kernels/cuda/dense_tc.cu:1077-1082 and :1107-1113,
# src/kernels/cuda/native_bf16.cu:139-171 and src/kernels/cuda/iq_kernels.cu:2450-2455 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import Op, Value
from ..formats.gguf import GGML_TYPES, ggml_row_bytes

if TYPE_CHECKING:
    from .emit import _Ctx

GGML_IDS = {name: type_id for type_id, (name, _, _) in GGML_TYPES.items()}
Q8_1_KERNELS = {"native": "native_quantize_q8_1", "swiglu": "native_swiglu_quantize_q8_1", "ieee": "quantize_q8_1"}
IEEE_READERS = ("basalt_experts",)      # the routed experts (OI-013 PR 4): basalt's only reader of the IEEE rows (F4)


def _columns(ctx: _Ctx, op: Op) -> int:
    """T, the window's columns: the first output's rows. basalt bakes T into each width's graph; a dynamic-T program
    would bind StepState, which none of its kernels reads, and run every narrower window at the compiled width."""
    if ctx.dynamic_t:
        raise ValueError(f"{op.kind}: basalt's kernels take the window's width as a compiled count, one static program "
                         "per width; a dynamic-T program cannot run them")
    return ctx.rows_of(op)[0]


def _expect(ctx: _Ctx, op: Op, v: Value, shape: Tuple[int, int], dtype: DType, why: str) -> None:
    if (ctx.shape(v), v.dtype) != (shape, dtype):
        raise ValueError(f"{op.kind}: {v.name!r} is {v.dtype} {list(ctx.shape(v))}, not {dtype} {list(shape)} ({why})")


def _weight(ctx: _Ctx, op: Op, w: Value, types: Sequence[int]) -> Tuple[int, int, int]:
    """``(ggml type id, n_out, n_in)`` of a weight as shipped: a weight whose format, or a constant whose dtype, names a
    ggml type in ``types``, ``[n_out, n_in]`` values, and not a pack's block-lane-major slab (a ``bf16`` slab's rows
    are re-laid-out; the kernels read rows as stored). A pack's aux entry holds exactly those rows, as a ``.basalt``
    entry does (``BasaltSource.bind``): the kernels read every row, and a shorter entry would send them past it."""
    name = w.format if w.is_weight else w.dtype.short if w.is_const else None
    shape = ctx.shape(w)
    if GGML_IDS.get(name) not in types or len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"{op.kind}: the weight {w.name!r} ({name} {list(shape)}) is not a weight or constant "
                         f"[n_out, n_in] of ggml type {sorted(GGML_TYPES[t][0] for t in types)}")
    if any(w.name in pk.slabs for pk in ctx.packs):
        raise ValueError(f"{op.kind}: the weight {w.name!r} is a block-lane-major pack slab, not rows as stored")
    tid, (n_out, n_in) = GGML_IDS[name], shape
    aux = next((pk.aux[w.name] for pk in ctx.packs if w.name in pk.aux), None)
    if aux is not None and int(aux["nbytes"]) != (need := n_out * ggml_row_bytes(tid, n_in)):
        raise ValueError(f"{op.kind}: the weight {w.name!r} is {need} bytes of {name} {list(shape)}, but its pack "
                         f"entry holds {aux['nbytes']}")
    return tid, n_out, n_in


def _bind(ctx: _Ctx, op: Op, slot: int, v: Value, align: int) -> Tuple[int, str, int]:
    """``(slot, buffer, offset)`` of ``v``, refused unless the offset is a multiple of ``align`` bytes (basalt's
    launchers check most of these pointers, ``ops/basalt.py`` says which; a window's buffer starts on a page)."""
    buf, off = ctx.buf(v)
    if off % align:
        raise ValueError(f"{op.kind}: {v.name!r} binds at byte {off} of {buf}; slot {slot} needs {align}-byte "
                         "alignment")
    return slot, buf, off


def _q8_1(ctx: _Ctx, op: Op) -> None:
    kind = op.attrs.get("quantizer")
    if kind not in Q8_1_KERNELS:
        raise ValueError(f"basalt_q8_1: attr quantizer must be one of {sorted(Q8_1_KERNELS)}, got {kind!r}")
    t = _columns(ctx, op)
    xs, (y,) = list(op.inputs), op.outputs
    if len(xs) != (2 if kind == "swiglu" else 1):
        raise ValueError(f"basalt_q8_1 {kind}: {len(xs)} inputs")
    n_in = ctx.shape(xs[0])[-1]
    if n_in < 1 or n_in % 32:
        raise ValueError(f"basalt_q8_1: a column of {n_in} values is not whole 32-value blocks")
    if kind != "ieee" and t > 8:     # validate_shape (native_mmvq.cu:1465-1470); quantize_q8_1_rows has none
        raise ValueError(f"basalt_q8_1 {kind}: {t} columns; basalt quantizes 1 to 8")
    for x in xs:
        _expect(ctx, op, x, (t, n_in), DType.F32, "the quantized columns")
    _expect(ctx, op, y, (t, n_in // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the input's columns")
    if kind == "ieee":
        other = sorted({c.kind for c in y.consumers} - set(IEEE_READERS))
        if other:
            raise ValueError(f"basalt_q8_1 ieee: {y.name!r} is read by {other}; basalt's IEEE rows feed only the "
                             f"routed experts ({', '.join(IEEE_READERS)})")
        shared = [y.view_of[0]] if y.view_of else [v.name for v in ctx.values.values() if v.view_of and
                                                   v.view_of[0] == y.name]
        if shared:                   # a view's readers are not the base's consumers, nor the base's the view's
            raise ValueError(f"basalt_q8_1 ieee: {y.name!r} shares its buffer with {shared}; basalt's IEEE rows are a "
                             "buffer of their own, read only by the routed experts")
    n = t * n_in
    k = ctx.kernel("basalt_q8_1", kernels.basalt_q8_1_source(), Q8_1_KERNELS[kind], {})
    prm = ctx.params("basalt_q8_1", kernels.basalt_q8_1_params(n))
    grid, threads = kernels.basalt_q8_1_grid(n)
    bindings = [_bind(ctx, op, i, v, 4) for i, v in enumerate(xs + [y])]
    ctx.add(k, bindings + [(len(xs) + 1, prm, 0)], (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{kind}", writes=[len(xs)])


def _gemv_values(ctx: _Ctx, op: Op, t: int, xq: Value, n_in: int, ys: Sequence[Value], n_outs: Sequence[int]) -> None:
    _expect(ctx, op, xq, (t, n_in // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the weight's columns")
    for y, n_out in zip(ys, n_outs):
        _expect(ctx, op, y, (t, n_out), DType.F32, "contiguous columns of the weight's rows")


def _mmvq(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (xq, w), (y,) = op.inputs, op.outputs
    tid, n_out, n_in = _weight(ctx, op, w, kernels.MMVQ_TRAITS)
    macros = kernels.basalt_mmvq_macros(tid, n_in, t)
    _gemv_values(ctx, op, t, xq, n_in, [y], [n_out])
    grid, threads = kernels.basalt_mmvq_grid(macros, n_out)
    k = ctx.kernel("basalt_mmvq", kernels.basalt_mmvq_source(), "basalt_mmvq", macros)
    prm = ctx.params("basalt_mmvq", kernels.basalt_mmvq_params(n_in, n_out))
    ctx.add(k, [_bind(ctx, op, 0, w, 4), _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)],
            (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w.name}", writes=[2])


def _dense_tc(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (xq, w), (y,) = op.inputs, op.outputs
    tid, n_out, n_in = _weight(ctx, op, w, kernels.DENSE_TC_SB)
    macros = kernels.basalt_dense_tc_macros(tid, n_in, n_out)       # basalt's order: 170 multiprocessors, aligned
    _gemv_values(ctx, op, t, xq, n_in, [y], [n_out])
    grid, threads = kernels.basalt_dense_tc_grid(macros, n_out)
    k = ctx.kernel("basalt_dense_tc", kernels.basalt_dense_tc_source(), "basalt_dense_tc", macros)
    prm = ctx.params("basalt_dense_tc", kernels.basalt_dense_tc_params(n_in, n_out, t))
    ctx.add(k, [_bind(ctx, op, 0, w, 4), _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)],
            (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w.name}", writes=[2])


def _dense_tc_pair(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (xq, w0, w1), (y0, y1) = op.inputs, op.outputs
    t0, n_out0, n_in = _weight(ctx, op, w0, kernels.DENSE_TC_SB)
    t1, n_out1, n_in1 = _weight(ctx, op, w1, kernels.DENSE_TC_SB)
    if n_in1 != n_in:
        raise ValueError(f"basalt_dense_tc_pair: {w0.name!r} has {n_in} columns, {w1.name!r} {n_in1}; the pair reads "
                         "one input")
    macros = kernels.basalt_dense_tc_pair_macros(t0, t1, n_in)
    _gemv_values(ctx, op, t, xq, n_in, [y0, y1], [n_out0, n_out1])
    grid, threads = kernels.basalt_dense_tc_pair_grid(n_out0, n_out1)
    k = ctx.kernel("basalt_dense_tc", kernels.basalt_dense_tc_source(), "basalt_dense_tc_pair", macros)
    prm = ctx.params("basalt_dense_tc_pair", kernels.basalt_dense_tc_params(n_in, n_out0, t, n_out1))
    bindings = [_bind(ctx, op, 0, w0, 16), _bind(ctx, op, 1, w1, 16), _bind(ctx, op, 2, xq, 4),
                _bind(ctx, op, 3, y0, 4), _bind(ctx, op, 4, y1, 4), (5, prm, 0)]
    ctx.add(k, bindings, (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w0.name}+{w1.name}", writes=[3, 4])


def _mmvf(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (x, w), (y,) = op.inputs, op.outputs
    _, n_out, n_in = _weight(ctx, op, w, (GGML_IDS["bf16"],))
    ldy = ctx.shape(y)[-1]
    if ldy < n_out:                  # basalt never checks it: every call passes ldy == n_out
        raise ValueError(f"basalt_mmvf: {y.name!r} holds {ldy} values a row and the weight has {n_out} rows; output "
                         "row k starts at k * ldy, so ldy >= n_out")
    macros = kernels.basalt_mmvf_macros(n_in, n_out, t)
    _expect(ctx, op, x, (t, n_in), DType.F32, "contiguous columns of the weight's width")
    _expect(ctx, op, y, (t, n_out), DType.F32, "contiguous columns of the weight's rows")
    grid, threads = kernels.basalt_mmvf_grid(macros, n_out)
    k = ctx.kernel("basalt_mmvf", kernels.basalt_mmvf_source(), "basalt_mmvf", macros)
    prm = ctx.params("basalt_mmvf", kernels.basalt_mmvf_params(n_in, n_out, t))
    ctx.add(k, [_bind(ctx, op, 0, x, 8), _bind(ctx, op, 1, w, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)],
            (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w.name}", writes=[2])


def _vector(ctx: _Ctx, op: Op, v: Value, n: int) -> None:
    """A vector as shipped, ``_weight``'s rules for ``(n,)`` F32 values: a weight whose format, or a constant whose
    dtype, is the ggml type, not a pack's block-lane-major slab, and a pack's aux entry of exactly those values (the
    kernels read every one)."""
    name = v.format if v.is_weight else v.dtype.short if v.is_const else None
    if name != "f32" or ctx.shape(v) != (n,):
        raise ValueError(f"{op.kind}: the vector {v.name!r} ({name} {list(ctx.shape(v))}) is not a weight or constant "
                         f"[{n}] of ggml type f32")
    if any(v.name in pk.slabs for pk in ctx.packs):
        raise ValueError(f"{op.kind}: the vector {v.name!r} is a block-lane-major pack slab, not values as stored")
    aux = next((pk.aux[v.name] for pk in ctx.packs if v.name in pk.aux), None)
    if aux is not None and int(aux["nbytes"]) != 4 * n:
        raise ValueError(f"{op.kind}: the vector {v.name!r} is {4 * n} bytes of f32 [{n}], but its pack entry holds "
                         f"{aux['nbytes']}")


# The GDN window's checks and bindings are adapted from basalt (jesdga95/basalt) src/kernels/cuda/dec_gdn_rec.cu:30-33
# and :355-367 and include/basalt/kernels/dec_gdn.hpp:16-39 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
# third_party/NOTICE).
def _gdn_window(ctx: _Ctx, op: Op) -> None:
    """``dec_gdn_rec`` (ops/basalt_gdn.py): HV from ``dt``, C from ``conv_w``, h_k = (C - 128 HV) / 256 and n_embd from
    ``w_alpha``, refused where host ``dec_gdn_rec`` exits (``kernels.basalt_gdn_params``); input i at buffer i, yq at
    14, the record at 16, and 15 (StepState's) unbound: T is the program's width. One 512-thread threadgroup per value
    head."""
    if (len(op.inputs), len(op.outputs)) != (14, 1):
        raise ValueError(f"{op.kind}: {len(op.inputs)} inputs and {len(op.outputs)} outputs; the window reads 14 (the "
                         "state, qkv, conv_w, x, the gate rows, dt, ssm_a, z, gamma, its four stores) and writes yq")
    state, qkv, conv_w, x, wa, wb, dt, ssm_a, z, gamma, *stores = op.inputs
    (yq,) = op.outputs
    t = _columns(ctx, op)
    hv = ctx.shape(dt)[0]
    _vector(ctx, op, dt, hv)
    _, c, taps = _weight(ctx, op, conv_w, (GGML_IDS["f32"],))
    if taps != 4:
        raise ValueError(f"{op.kind}: the conv weight {conv_w.name!r} holds {taps} taps a channel; basalt's conv has 4")
    n = ctx.shape(wa)[-1]
    for w in (wa, wb):
        if _weight(ctx, op, w, (GGML_IDS["bf16"],))[1:] != (hv, n):
            raise ValueError(f"{op.kind}: the gate rows {w.name!r} are {list(ctx.shape(w))}, not [{hv}, {n}]: one row "
                             "of n_embd per value head")
    _vector(ctx, op, ssm_a, hv)
    _vector(ctx, op, gamma, kernels.GDN_S)
    zv = kernels.GDN_S * hv
    _expect(ctx, op, qkv, (t, c), DType.F32, "the window's q, k and v projections")
    _expect(ctx, op, x, (t, n), DType.F32, "the gates' input rows")
    _expect(ctx, op, z, (t, zv), DType.F32, "the output gate's rows")
    _expect(ctx, op, yq, (t, zv // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the window's output")
    shapes = [(kernels.GDN_S * zv + 3 * c,), (kernels.GDN_MT, c), (kernels.GDN_MT, c), (kernels.GDN_MT, hv),
              (kernels.GDN_MT, hv)]
    for s, shape in zip([state, *stores], shapes):
        if not s.is_state:
            raise ValueError(f"{op.kind}: {s.name!r} is not a graph state; the state and the window stores are a "
                             "session's buffers, shared by name across its programs")
        _expect(ctx, op, s, shape, DType.F32, "the recurrent state, then the conv history" if s is state else
                f"the window's {kernels.GDN_MT} rows, a commit's replay")
    ups, names = set(op.attrs.get("updates", ())), [s.name for s in stores]
    if state.name in ups:
        raise ValueError(f"{op.kind}: updates names {state.name!r}; the window reads the state and never writes it "
                         "(invariant 3: the commit does)")
    if ups != set(names):
        raise ValueError(f"{op.kind}: updates names {sorted(ups)}; the window writes exactly its stores {names}")
    eps = op.attrs.get("eps")
    if eps is None:
        raise ValueError(f"{op.kind}: needs attr eps, the norms' epsilon")
    prm = ctx.params("basalt_gdn_window", kernels.basalt_gdn_params(t, c, n, (c - zv) // 256, hv, eps))
    k = ctx.kernel("basalt_gdn", kernels.basalt_gdn_source(), "dec_gdn_rec", {})
    bindings = [_bind(ctx, op, i, v, 16 if i in (3, 4, 5) else 4) for i, v in enumerate(op.inputs)]
    ctx.add(k, bindings + [_bind(ctx, op, 14, yq, 4), (16, prm, 0)], (hv, 1, 1), (kernels.GDN_THREADS, 1, 1),
            f"{op.kind}:{state.name}", writes=[10, 11, 12, 13, 14])


def _gdn_commit(ctx: _Ctx, op: Op) -> None:
    """``gdn_commit_layers`` for one layer (ops/basalt_gdn.py): MT and C from ``win_h``, HV from ``win_gate``, refused
    where ``kernels.basalt_gdn_commit_params`` refuses; input i at buffer i, ``used`` at 6, the record at 7, and no
    StepState: the host-written record ``commit`` carries the count. One 128-thread threadgroup per item."""
    if ctx.dynamic_t:
        raise ValueError(f"{op.kind}: the commit's count is word 0 of its host-written record, not StepState's; a "
                         "dynamic-T program cannot run it")
    if (len(op.inputs), len(op.outputs)) != (6, 1):
        raise ValueError(f"{op.kind}: {len(op.inputs)} inputs and {len(op.outputs)} outputs; the commit reads 6 (the "
                         "state, win_h, win_qkv, win_gate, win_beta, the record) and writes used")
    state, win_h, win_qkv, win_gate, win_beta, commit = op.inputs
    (used,) = op.outputs
    for s in (state, win_h, win_qkv, win_gate, win_beta):
        if not s.is_state:
            raise ValueError(f"{op.kind}: {s.name!r} is not a graph state; the state and the window stores are a "
                             "session's buffers, shared by name with its window programs")
    if not commit.is_input:
        raise ValueError(f"{op.kind}: {commit.name!r} is not a graph input; the record is the host's, written once it "
                         "has chosen n_keep")
    ups = set(op.attrs.get("updates", ()))
    if ups != {state.name}:
        raise ValueError(f"{op.kind}: updates names {sorted(ups)}; the commit writes exactly the state {state.name!r}")
    mt, c, hv = ctx.shape(win_h)[0], ctx.shape(win_h)[-1], ctx.shape(win_gate)[-1]
    _expect(ctx, op, win_h, (mt, c), DType.F32, "the window's rows after the conv, the SiLU and the q/k L2 norm")
    _expect(ctx, op, win_qkv, (mt, c), DType.F32, f"the window's qkv rows; MT and C from {win_h.name!r}")
    _expect(ctx, op, win_gate, (mt, hv), DType.F32, f"each value head's decay gate per row; MT from {win_h.name!r}")
    _expect(ctx, op, win_beta, (mt, hv), DType.F32,
            f"each value head's beta per row; MT from {win_h.name!r}, HV from {win_gate.name!r}")
    _expect(ctx, op, state, (kernels.GDN_S * kernels.GDN_S * hv + 3 * c,), DType.F32,
            f"the recurrent state, then the conv history; HV from {win_gate.name!r}, C from {win_h.name!r}")
    _expect(ctx, op, commit, (2 + mt,), DType.I32, "the record: n_keep, n_keep - 1, then a position per window row")
    _expect(ctx, op, used, (1,), DType.U32, "the n the commit used")
    prm = ctx.params("basalt_gdn_commit", kernels.basalt_gdn_commit_params(c, hv, mt))
    k = ctx.kernel("basalt_gdn_commit", kernels.basalt_gdn_commit_source(), "gdn_commit_layers", {})
    bindings = [_bind(ctx, op, i, v, 4) for i, v in enumerate([*op.inputs, used])]
    if bindings[5][1] == ctx.program.step_state:
        raise ValueError(f"{op.kind}: {commit.name!r} names a StepState field; the commit record is a host-written "
                         "input buffer of its own, never StepState")
    ctx.add(k, bindings + [(7, prm, 0)], (kernels.basalt_gdn_commit_grid(c, hv), 1, 1), (128, 1, 1), op.kind,
            writes=[0, 6])


BASALT_HANDLERS: Dict[str, object] = {"basalt_q8_1": _q8_1, "basalt_mmvq": _mmvq, "basalt_dense_tc": _dense_tc,
                                      "basalt_dense_tc_pair": _dense_tc_pair, "basalt_mmvf": _mmvf,
                                      "basalt_gdn_window": _gdn_window, "basalt_gdn_commit": _gdn_commit}
