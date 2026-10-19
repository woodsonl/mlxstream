"""The emitter's handlers for basalt's ops (``monolith/ops/basalt.py``), merged into ``emit.HANDLERS``. Each handler
emits basalt's launches in its order (one dispatch for a GEMV or a quantizer; the Q5_K head's second dispatches after
it) with the macros, grid, threadgroup, params and scratch sizes of ``monolith/kernels.py``'s builders, after the
checks basalt's launchers make (shapes, column counts, pointer alignment) and the ones its call sites satisfy (the IEEE
q8_1 rows feed only the routed experts; outputs are contiguous columns); a static program only (one per window width).
A follow-up dispatch reads what the one before it wrote, which ``writes`` declares, so the barrier pass orders it.
"""

# The launcher checks are adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:1465-1475, :1490-1494,
# :1549-1552 and :1563-1568, src/kernels/cuda/dense_tc.cu:1077-1082 and :1107-1113,
# src/kernels/cuda/native_bf16.cu:139-171, src/kernels/cuda/iq_kernels.cu:2450-2455 and
# src/kernels/cuda/q5k_head.cu:372-374, :406-408, :411-412 and :516-518, and the Q5_K head's per-column mask from
# src/core/mtp.cpp:1024-1025 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).

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


# ---- the Q5_K head (kernels/common/basalt_q5k_head.metal) -----------------------------------------------------------

Q5_K = GGML_IDS["q5_k"]
Q5K_BLOCK = GGML_TYPES[Q5_K][2]         # bytes of a 256-value block
TOP1_N_IN = 2560                        # the top-1s' rows: q5k_head_top1 refuses others, the 2-bit copy holds 40 words


def _q5k_rows(ctx: _Ctx, op: Op, w: Value, n: int) -> int:
    """n_in of the Q5_K rows ``w`` an op reads the first ``n`` of: a ``q5_k`` weight ``[rows, n_in]`` (a ``.basalt``
    tensor, the verify head's ``output.weight``) or a host input ``U8 [rows, n_in / 256 * 176]``, the rows as stored
    (the drafter's gathered head rows, which the session writes at load and at every flush). basalt passes such a
    pointer with a row count: its tier 0 reads the leading n_tier1_ of the gathered rows (mtp.cpp:1006, :1016-1021)."""
    if w.is_input:
        shape = ctx.shape(w)
        if not w.is_host:
            raise ValueError(f"{op.kind}: {w.name!r} is an input each program allocates; Q5_K rows that are no file "
                             "tensor are the session's gathered head rows, a host input")
        if w.dtype != DType.U8 or len(shape) != 2 or shape[1] % Q5K_BLOCK:   # no blocks: each op refuses n_in 0
            raise ValueError(f"{op.kind}: the host rows {w.name!r} are {w.dtype} {list(shape)}, not u8 [rows, whole "
                             f"{Q5K_BLOCK}-byte Q5_K blocks] (the rows as stored)")
        rows, n_in = shape[0], shape[1] // Q5K_BLOCK * 256
    else:
        _, rows, n_in = _weight(ctx, op, w, (Q5_K,))
    if rows < n:
        raise ValueError(f"{op.kind}: {w.name!r} holds {rows} rows; the op reads {n}")
    return n_in


def _map(ctx: _Ctx, op: Op, m: Value, n: int) -> None:
    """The map: an id per row read, negative for a row that holds no token (basalt's ``dvocab_``)."""
    shape = ctx.shape(m)
    if m.dtype != DType.I32 or len(shape) != 1 or shape[0] < n:
        raise ValueError(f"{op.kind}: the map {m.name!r} is {m.dtype} {list(shape)}; it maps the {n} rows read, "
                         f"i32 [>= {n}]")


def _q5k_head(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    xq, w, *mp = op.inputs
    (y,) = op.outputs
    if len(mp) > 1:
        raise ValueError(f"basalt_q5k_head: {len(op.inputs)} inputs; it takes (xq, rows[, map])")
    n_out = ctx.shape(y)[-1]
    n_in = _q5k_rows(ctx, op, w, n_out)
    if mp:
        _map(ctx, op, mp[0], n_out)
    macros = kernels.basalt_q5k_head_macros(n_in, t)        # q5k_head_supported (q5k_head.cu:412)
    _expect(ctx, op, xq, (t, n_in // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the rows' width")
    _expect(ctx, op, y, (t, n_out), DType.F32, "contiguous columns of the logits")
    src = kernels.basalt_q5k_head_source()
    prm = ctx.params("basalt_q5k_head", kernels.basalt_q5k_head_params(n_in, n_out))
    head = [_bind(ctx, op, 0, w, 2), _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)]
    ctx.add(ctx.kernel("basalt_q5k_head", src, "basalt_q5k_head", macros), head,
            (kernels.basalt_q5k_head_grid(n_out), 1, 1), (kernels.Q5K_HEAD_THREADS, 1, 1), f"{op.kind}:{w.name}",
            writes=[2])
    if mp:                           # basalt masks each column's logits (mtp.cpp:1024-1025)
        km = ctx.kernel("basalt_q5k_head", src, "basalt_q5k_head_mask_unmapped", macros)
        grid, threads = kernels.basalt_q5k_mask_grid(n_out)
        _, yb, yo = head[2]
        for j in range(t):
            ctx.add(km, [(0, yb, yo + 4 * n_out * j), _bind(ctx, op, 1, mp[0], 4), (2, prm, 0)], (grid, 1, 1),
                    (threads, 1, 1), f"{op.kind}.mask:{w.name}", writes=[0])


def _q5k_top1(ctx: _Ctx, op: Op) -> None:
    _columns(ctx, op)
    xq, w, *mp = op.inputs
    y, tok, prob = op.outputs
    if len(mp) > 1:
        raise ValueError(f"basalt_q5k_top1: {len(op.inputs)} inputs; it takes (xq, rows[, map])")
    n_out = ctx.shape(y)[-1]
    n_in = _q5k_rows(ctx, op, w, n_out)
    if n_in != TOP1_N_IN:            # q5k_head.cu:374
        raise ValueError(f"basalt_q5k_top1: rows of {n_in} values; q5k_head_top1 takes {TOP1_N_IN}")
    if mp:
        _map(ctx, op, mp[0], n_out)
    _expect(ctx, op, xq, (1, TOP1_N_IN // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8,
            "one q8_1 column: basalt runs the top-1 at T = 1, mtp.cpp:1016")
    _expect(ctx, op, y, (1, n_out), DType.F32, "the logits of one column")
    _expect(ctx, op, tok, (1,), DType.I32, "the best row's id")
    _expect(ctx, op, prob, (1,), DType.F32, "its probability")
    src, macros = kernels.basalt_q5k_head_source(), kernels.basalt_q5k_head_macros(TOP1_N_IN, 1)
    grid = kernels.basalt_q5k_head_grid(n_out)        # once: the merge combines this many partials in basalt's order
    prm = ctx.params("basalt_q5k_top1", kernels.basalt_q5k_head_params(TOP1_N_IN, n_out, bool(mp), grid))
    part = ctx.scratch("basalt_q5k_top1.part", kernels.basalt_q5k_top1_part_bytes(grid))
    _, mb, mo = _bind(ctx, op, 3, mp[0], 4) if mp else (3, ctx.scratch("basalt_q5k_top1.no_map", 4), 0)
    ctx.add(ctx.kernel("basalt_q5k_head", src, "basalt_q5k_head_top1", macros),
            [_bind(ctx, op, 0, w, 2), _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, y, 4), (3, mb, mo), (4, part, 0),
             (5, prm, 0)], (grid, 1, 1), (kernels.Q5K_HEAD_THREADS, 1, 1), f"{op.kind}:{w.name}", writes=[2, 4])
    ctx.add(ctx.kernel("basalt_q5k_head", src, "basalt_q5k_head_top1_merge", macros),
            [(0, part, 0), (1, mb, mo), _bind(ctx, op, 2, tok, 4), _bind(ctx, op, 3, prob, 4), (4, prm, 0)],
            (1, 1, 1), (kernels.Q5K_HEAD_THREADS, 1, 1), f"{op.kind}.merge:{w.name}", writes=[2, 3])


def _q5k_proxy_top1(ctx: _Ctx, op: Op) -> None:
    _columns(ctx, op)
    xq, proxy, rows, *mp = op.inputs
    tok, prob = op.outputs
    if len(mp) != 1:                 # q5k_head.cu:518 refuses a call without the map
        raise ValueError(f"basalt_q5k_proxy_top1: {len(op.inputs)} inputs; it takes (xq, proxy, rows, map), and basalt "
                         "refuses it without the map")
    n_rows, n_cap = op.attrs.get("n_rows"), op.attrs.get("n_cap")
    if not (isinstance(n_rows, int) and isinstance(n_cap, int)):
        raise ValueError(f"basalt_q5k_proxy_top1: attrs n_rows and n_cap must be ints, got {n_rows!r} and {n_cap!r}")
    params = kernels.basalt_q5k_proxy_params(n_cap, n_rows=n_rows)          # 1 <= n_rows <= n_cap (q5k_head.cu:518)
    if not proxy.is_host:
        raise ValueError(f"basalt_q5k_proxy_top1: the 2-bit copy {proxy.name!r} is not a host input; the session packs "
                         "it at load and at every flush")
    _expect(ctx, op, proxy, (kernels.basalt_q5k_proxy_bytes(n_cap),), DType.U8, "the 2-bit copy of n_cap rows")
    n_in = _q5k_rows(ctx, op, rows, n_rows)
    if n_in != TOP1_N_IN:
        raise ValueError(f"basalt_q5k_proxy_top1: rows of {n_in} values; the 2-bit copy holds rows of {TOP1_N_IN}")
    _map(ctx, op, mp[0], n_rows)
    _expect(ctx, op, xq, (1, TOP1_N_IN // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8,
            "one q8_1 column: basalt runs the top-1 at T = 1, mtp.cpp:1009")
    _expect(ctx, op, tok, (1,), DType.I32, "the best row's id")
    _expect(ctx, op, prob, (1,), DType.F32, "its probability")
    src = kernels.basalt_q5k_head_source()
    prm = ctx.params("basalt_q5k_proxy_top1", params)
    scratch = ctx.scratch("basalt_q5k_proxy.scratch", kernels.basalt_q5k_proxy_scratch_bytes(n_cap))
    xb, (_, mb, mo) = _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, mp[0], 4)
    ctx.add(ctx.kernel("basalt_q5k_head", src, "basalt_q5k_proxy_score", {}),
            [_bind(ctx, op, 0, proxy, 16), xb, (2, mb, mo), (3, scratch, 0), (4, prm, 0)],
            (kernels.basalt_q5k_proxy_tiles(n_rows), 1, 1), (kernels.PX_THREADS_SCORE, 1, 1),
            f"{op.kind}:{proxy.name}", writes=[3])
    ctx.add(ctx.kernel("basalt_q5k_head", src, "basalt_q5k_proxy_final", {}),
            [_bind(ctx, op, 0, rows, 2), xb, (2, mb, mo), (3, scratch, 0), _bind(ctx, op, 4, tok, 4),
             _bind(ctx, op, 5, prob, 4), (6, prm, 0)], (1, 1, 1), (kernels.PX_THREADS_FINAL, 1, 1),
            f"{op.kind}.final:{rows.name}", writes=[4, 5])


BASALT_HANDLERS: Dict[str, object] = {"basalt_q8_1": _q8_1, "basalt_mmvq": _mmvq, "basalt_dense_tc": _dense_tc,
                                      "basalt_dense_tc_pair": _dense_tc_pair, "basalt_mmvf": _mmvf,
                                      "basalt_q5k_head": _q5k_head, "basalt_q5k_top1": _q5k_top1,
                                      "basalt_q5k_proxy_top1": _q5k_proxy_top1}
