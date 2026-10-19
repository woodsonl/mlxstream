"""The emitter's handlers for basalt's ops (``monolith/ops/basalt.py``), merged into ``emit.HANDLERS``. Each handler
emits basalt's launches in its order (one dispatch for a GEMV or a quantizer, three or four for the experts) with the
macros, grids, threadgroups, params and scratch sizes of ``monolith/kernels.py``'s builders, after the checks basalt's
launchers make (shapes, column counts, pointer alignment) and the ones its call sites satisfy (the IEEE q8_1 rows feed
only the routed experts; outputs are contiguous columns); a static program only (one per window width). A follow-up
dispatch reads what the one before it wrote, which ``writes`` declares, so the barrier pass orders it.
"""

# The launcher checks are adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:1465-1475, :1490-1494,
# :1549-1552 and :1563-1568, src/kernels/cuda/dense_tc.cu:1077-1082 and :1107-1113,
# src/kernels/cuda/native_bf16.cu:139-171 and src/kernels/cuda/iq_kernels.cu:2450-2455 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import Op, Value
from ..formats.gguf import GGML_TYPES, ggml_row_bytes
from ..runtime.program import BufferSpec

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


BASALT_HANDLERS: Dict[str, object] = {"basalt_q8_1": _q8_1, "basalt_mmvq": _mmvq, "basalt_dense_tc": _dense_tc,
                                      "basalt_dense_tc_pair": _dense_tc_pair, "basalt_mmvf": _mmvf}


# ---- the routed and drafter experts (kernels/common/basalt_experts.metal, basalt_s2_experts.metal) ------------------
# The launchers' checks and their call sites are adapted from basalt (jesdga95/basalt)
# src/kernels/cuda/iq_kernels.cu:2553-2603, src/kernels/cuda/s2_expert_grouped.cu:504-508, :745 and :1076-1122,
# src/kernels/cuda/quantize_act.cu:219-238, src/core/verify.cpp:1731 and :1744-1794 and src/core/mtp.cpp:931-935 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).

def _counts(op: Op, *names: str) -> List[int]:
    """The launcher's counts the op takes as attrs ``names``: positive ints."""
    vals = [op.attrs.get(n) for n in names]
    if not all(type(v) is int and v > 0 for v in vals):
        raise ValueError(f"{op.kind}: attrs {', '.join(names)} must be positive ints, got {vals}")
    return vals


def _table(ctx: _Ctx, op: Op, v: Value, dtype: DType, rows: int, what: str) -> None:
    """A plan table: 1-D ``dtype`` of at least ``rows`` entries."""
    shape = ctx.shape(v)
    if v.dtype != dtype or len(shape) != 1 or shape[0] < rows:
        raise ValueError(f"{op.kind}: {v.name!r} is {v.dtype} {list(shape)}, not {dtype} [>= {rows}] ({what})")


def _blob_buffers(ctx: _Ctx, names: Sequence[str], blob: int) -> List[str]:
    """The buffers the plan's addresses point into: a graph value's buffer (a ``.basalt`` section's window), else a
    buffer the session allocates (the expert cache), which the program declares with role "host", holding at least a
    blob, so that Engine requires the session to pass it."""
    bufs = []
    for name in names:
        buf = ctx.buf(ctx.values[name])[0] if name in ctx.values else name
        spec = ctx.program.buffers.setdefault(buf, BufferSpec(blob, None, "host"))
        if spec.role == "host":
            spec.nbytes = max(spec.nbytes, blob)
        bufs.append(buf)
    return list(dict.fromkeys(bufs))


def _experts(ctx: _Ctx, op: Op) -> None:
    _columns(ctx, op)                                       # a static program; T is xq's rows
    grp_ptr, grp_start, n_groups, ent_dst, ent_tok, xq = op.inputs
    (out,) = op.outputs
    n_ff, k_per_tok = _counts(op, "n_ff", "k_per_tok")
    blobs = op.attrs.get("blobs")
    if isinstance(blobs, str) or not blobs or not all(isinstance(b, str) for b in blobs):
        raise ValueError(f"basalt_experts: attr blobs must list the buffers its blobs live in, got {blobs!r}")
    t = ctx.shape(xq)[0]
    cap = t * k_per_tok                 # cap_groups = cap_entries: T tokens' routed entries (verify.cpp:1744, :1764)
    shape = ctx.shape(out)
    if out.dtype != DType.F32 or len(shape) != 2 or shape[0] < cap:
        raise ValueError(f"basalt_experts: {out.name!r} is {out.dtype} {list(shape)}; an entry writes the row of its "
                         f"routing index, so out is f32 [>= T x k_per_tok ({cap}), n_embd]")
    n_embd = shape[1]
    lay = kernels.basalt_expert_layout(n_embd, n_ff)        # native_expert_supported (iq_kernels.cu:2511-2515)
    macros = kernels.basalt_expert_macros(n_embd, n_ff)     # basalt's kernels for the shape
    _expect(ctx, op, xq, (t, n_embd // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the output's width")
    q = xq.producer
    if q is None or q.kind != "basalt_q8_1" or q.attrs.get("quantizer") != "ieee":
        raise ValueError(f"basalt_experts: {xq.name!r} is not a basalt_q8_1 op's ieee rows; basalt's routed experts "
                         "read the IEEE q8_1 of the FFN input (quantize_q8_1_rows, verify.cpp:1731)")
    for v, dt, rows, what in ((grp_ptr, DType.I64, cap, "an address per group"),
                              (grp_start, DType.I32, cap + 1, "a start per group and the end"),
                              (n_groups, DType.I32, 1, "the group count"),
                              (ent_dst, DType.I32, cap, "a row per entry"),
                              (ent_tok, DType.I32, cap, "a token per entry")):
        _table(ctx, op, v, dt, rows, what)
    ptr, start, count = _bind(ctx, op, 1, grp_ptr, 8), _bind(ctx, op, 2, grp_start, 4), _bind(ctx, op, 3, n_groups, 4)
    dst, tok = _bind(ctx, op, 4, ent_dst, 4), _bind(ctx, op, 4, ent_tok, 4)
    xb, ob = _bind(ctx, op, 5, xq, 4), _bind(ctx, op, 6, out, 4)
    reads = _blob_buffers(ctx, blobs, lay["bytes"])
    gate, up, hq = (ctx.scratch(f"basalt_experts.{s}", nb)
                    for s, nb in zip(("gate", "up", "hq"), kernels.basalt_expert_scratch(cap, n_ff)))
    prm = ctx.params("basalt_experts", kernels.basalt_expert_params(n_embd, n_ff))
    ggu, gsw, gdn = kernels.basalt_expert_grids(macros, n_embd, n_ff, cap, cap)     # gy: min(cap, 24)
    src, th = kernels.basalt_experts_source(), (kernels.EXPERT_THREADS, 1, 1)
    ctx.add(ctx.kernel("basalt_experts", src, "basalt_expert_gu", macros),
            [ptr, start, count, tok, xb, (6, gate, 0), (7, up, 0), (8, prm, 0)], (*ggu, 1), th, f"{op.kind}:{out.name}",
            writes=[6, 7], indirect_reads=reads)
    ctx.add(ctx.kernel("basalt_experts", src, "basalt_expert_swiglu_q8_1", macros),
            [(0, gate, 0), (1, up, 0), start, count, (4, hq, 0), (5, prm, 0)], (gsw, 1, 1), th,
            f"{op.kind}.swiglu:{out.name}", writes=[4])
    ctx.add(ctx.kernel("basalt_experts", src, "basalt_expert_down", macros),
            [ptr, start, count, dst, (5, hq, 0), ob, (7, prm, 0)], (*gdn, 1), th, f"{op.kind}.down:{out.name}",
            writes=[6], indirect_reads=reads)


def _s2_plan(ctx: _Ctx, op: Op) -> None:
    _columns(ctx, op)
    ids, base = op.inputs
    grp_ptr, grp_start, counts, ent_dst, ent_tok = op.outputs
    k_per_tok, n_expert = _counts(op, "k_per_tok", "n_expert")
    shape = ctx.shape(ids)
    if ids.dtype != DType.I32 or len(shape) != 1:
        raise ValueError(f"basalt_s2_plan: the ids {ids.name!r} are {ids.dtype} {list(shape)}, not i32 [n]")
    n = shape[0]
    params = kernels.basalt_s2_resident_params(n, k_per_tok)        # 1 <= n <= 128 (s2_expert_grouped.cu:1079)
    for v, dt, rows, what in ((grp_ptr, DType.I64, n, "an address per group"),
                              (grp_start, DType.I32, n + 1, "a start per group and the end"),
                              (counts, DType.I32, 2, "the group and entry counts"),
                              (ent_dst, DType.I32, n, "a row per entry"),
                              (ent_tok, DType.I32, n, "a token per entry")):
        _table(ctx, op, v, dt, rows, what)
    need, bshape = n_expert * kernels.S2_BLOB_BYTES, ctx.shape(base)
    if base.dtype != DType.U8 or len(bshape) != 1 or bshape[0] < need:
        raise ValueError(f"basalt_s2_plan: base {base.name!r} is {base.dtype} {list(bshape)}; the router's ids reach "
                         f"n_expert ({n_expert}) blobs of {kernels.S2_BLOB_BYTES} bytes from it, u8 [>= {need}]")
    bindings = [_bind(ctx, op, 0, ids, 4), _bind(ctx, op, 1, base, 2), _bind(ctx, op, 2, grp_ptr, 8),
                _bind(ctx, op, 3, grp_start, 4), _bind(ctx, op, 4, counts, 4), _bind(ctx, op, 5, ent_dst, 4),
                _bind(ctx, op, 6, ent_tok, 4)]
    prm = ctx.params("basalt_s2_plan", params)
    ctx.add(ctx.kernel("basalt_s2_experts", kernels.basalt_s2_experts_source(), "basalt_s2_group_resident", {}),
            bindings + [(7, prm, 0)], (1, 1, 1), (kernels.S2_RESIDENT_THREADS, 1, 1), f"{op.kind}:{ids.name}",
            writes=[2, 3, 4, 5, 6])


def _q8_0_scaled(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (x,), (xq, xs) = op.inputs, op.outputs
    n_in = ctx.shape(x)[-1]
    if n_in < 1 or n_in % 32:                               # quantize_act.cu:221-224, a row at a time
        raise ValueError(f"basalt_q8_0_scaled: a row of {n_in} values is not whole 32-value blocks")
    _expect(ctx, op, x, (t, n_in), DType.F32, "the quantized rows")
    _expect(ctx, op, xq, (t, n_in // 32 * kernels.Q8_0_BLOCK_BYTES), DType.U8, "block_q8_0 rows of the input's rows")
    _expect(ctx, op, xs, (t, n_in // 32), DType.F32, "an fp32 scale per block")
    prm = ctx.params("basalt_q8_0_scaled", kernels.basalt_q8_0_scaled_params(t * n_in))
    ctx.add(ctx.kernel("basalt_s2_experts", kernels.basalt_s2_experts_source(), "basalt_quantize_q8_0_scaled", {}),
            [_bind(ctx, op, 0, x, 4), _bind(ctx, op, 1, xq, 1), _bind(ctx, op, 2, xs, 4), (3, prm, 0)],
            (kernels.basalt_q8_0_scaled_grid(t * n_in), 1, 1), (kernels.S2_THREADS, 1, 1), f"{op.kind}:{x.name}",
            writes=[1, 2])


def _s2_experts(ctx: _Ctx, op: Op) -> None:
    _columns(ctx, op)                                       # a static program; T is xq's rows
    if len(op.inputs) != 7:
        raise ValueError(f"basalt_s2_experts: {len(op.inputs)} inputs; it takes (grp_ptr, grp_start, counts, ent_dst, "
                         "ent_tok, xq, xs), the fp32 scales included, as basalt's drafter passes them")
    tables, (xq, xs), (out,) = list(op.inputs[:5]), op.inputs[5:], op.outputs
    cap_groups, cap_entries = _counts(op, "cap_groups", "cap_entries")
    plan = tables[0].producer
    if plan is None or plan.kind != "basalt_s2_plan" or list(plan.outputs) != tables:
        raise ValueError("basalt_s2_experts: its tables are not one basalt_s2_plan's outputs (grp_ptr, grp_start, "
                         "counts, ent_dst, ent_tok) in order; basalt's drafter plans on the device, and the plan's "
                         "base is the buffer its blobs live in")
    ids, base = plan.inputs
    n, k_per_tok = ctx.shape(ids)[0], plan.attrs.get("k_per_tok")
    if cap_groups < n:
        raise ValueError(f"basalt_s2_experts: cap_groups {cap_groups} < the plan's {n} ids; the plan makes up to a "
                         "group an id, and each group takes a threadgroup row")
    if cap_entries < n:
        raise ValueError(f"basalt_s2_experts: cap_entries {cap_entries} < the plan's {n} entries, which the scratch "
                         "holds")
    t = ctx.shape(xq)[0]
    if t > kernels.S2_GMAX:                     # static_assert(GMAX >= kVerifyMaxT), s2_expert_grouped.cu:745
        raise ValueError(f"basalt_s2_experts: {t} tokens; the kernels take at most {kernels.S2_GMAX} entries a group, "
                         "one a token routed to its expert")
    if -(-n // k_per_tok) > t:
        raise ValueError(f"basalt_s2_experts: the plan's {n} ids are {-(-n // k_per_tok)} tokens of {k_per_tok}; xq "
                         f"holds {t}")
    _expect(ctx, op, xq, (t, kernels.S2_H // 32 * kernels.Q8_0_BLOCK_BYTES), DType.U8, "block_q8_0 rows of 2560 values")
    _expect(ctx, op, xs, (t, kernels.S2_H // 32), DType.F32, "their fp32 scales")
    shape = ctx.shape(out)
    if out.dtype != DType.F32 or len(shape) != 2 or shape[1] != kernels.S2_H or shape[0] < n:
        raise ValueError(f"basalt_s2_experts: {out.name!r} is {out.dtype} {list(shape)}; an entry writes the row of "
                         f"its routing index, so out is f32 [>= {n}, {kernels.S2_H}]")
    grids, sc = kernels.basalt_s2_grids(cap_groups, cap_entries), kernels.basalt_s2_scratch(cap_entries)
    ptr, start, count = (_bind(ctx, op, 0, tables[0], 8), _bind(ctx, op, 1, tables[1], 4),
                         _bind(ctx, op, 2, tables[2], 4))
    dst, tok = _bind(ctx, op, 3, tables[3], 4), _bind(ctx, op, 3, tables[4], 4)
    xb, sb, ob = _bind(ctx, op, 4, xq, 4), _bind(ctx, op, 5, xs, 4), _bind(ctx, op, 6, out, 4)
    reads = [ctx.buf(base)[0]]
    scratch = ctx.scratch("basalt_s2_experts.scratch", sc["bytes"])
    prm = ctx.params("basalt_s2_experts", kernels.basalt_s2_params(cap_entries))
    qprm = ctx.params("basalt_s2_experts.quant", kernels.basalt_q8_0_scaled_params(cap_entries * kernels.S2_FF))
    src, th = kernels.basalt_s2_experts_source(), (kernels.S2_THREADS, 1, 1)
    ctx.add(ctx.kernel("basalt_s2_experts", src, "basalt_s2_gu", {}),
            [ptr, start, count, tok, xb, sb, (6, scratch, 0), (7, prm, 0)], (*grids["gu"], 1), th,
            f"{op.kind}:{out.name}", writes=[6], indirect_reads=reads)
    ctx.add(ctx.kernel("basalt_s2_experts", src, "basalt_s2_swiglu", {}), [(0, scratch, 0), (1, prm, 0)],
            (grids["swiglu"], 1, 1), th, f"{op.kind}.swiglu:{out.name}", writes=[0])
    ctx.add(ctx.kernel("basalt_s2_experts", src, "basalt_quantize_q8_0_scaled", {}),
            [(0, scratch, 0), (1, scratch, sc["hq"]), (2, scratch, sc["hs"]), (3, qprm, 0)], (grids["quant"], 1, 1),
            th, f"{op.kind}.quant:{out.name}", writes=[1, 2])
    ctx.add(ctx.kernel("basalt_s2_experts", src, "basalt_s2_down", {}),
            [ptr, start, count, dst, (4, scratch, sc["hq"]), (5, scratch, sc["hs"]), ob], (*grids["down"], 1), th,
            f"{op.kind}.down:{out.name}", writes=[6], indirect_reads=reads)


BASALT_HANDLERS.update({"basalt_experts": _experts, "basalt_s2_plan": _s2_plan, "basalt_q8_0_scaled": _q8_0_scaled,
                        "basalt_s2_experts": _s2_experts})
