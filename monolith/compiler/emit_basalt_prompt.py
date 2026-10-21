# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/prefill/prefill.cpp:1188-1199 and src/prefill/kernels.cu:63, :867-887 and
# :893-900 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""The emitter's handlers for basalt's prompt path (``monolith/ops/basalt_prompt.py``), merged into ``emit.HANDLERS``.
Every dispatch binds StepState at slot 15 and reads the chunk's rows only through ``prompt_rows(st)``
(``kernels.PROMPT_ROWS``; ``Gemm::native``'s dequantization reads none), in the dynamic-T prompt program and in a
static one; its macros, grid, threadgroup and params come from ``monolith/kernels.py``'s builders only, and a params
record holds geometry, never a row count, a position or a pointer (invariant 4). The weight checks are
``native_proj``'s and ``bf16_proj``'s (``prefill.cpp:1188-1199``): the entry must agree with the weight."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import Op, Value
from ..core.shapes import T
from .emit_basalt import GGML_IDS, _bind, _expect, _weight

if TYPE_CHECKING:
    from .emit import _Ctx

GEMM_ENTRIES = ("bf16", "native")      # hc8 adds "bf16_w"


def prompt_rows(ctx: _Ctx, op: Op, rows_from: Optional[Value] = None) -> Tuple[int, Dict[str, str]]:
    """``(B, macros)`` of a prompt dispatch: B, the compiled rows of ``op``'s first output (of ``rows_from`` instead, for
    an op whose only output is a placeholder), by ``ctx.rows_of``'s rule, and the macros ``prompt_rows(st)`` reads:
    STEP_STATE 1, ROWS_MAX B, and T_SRC 0 for T rows in a dynamic-T program, T_SRC 2 with T_STATIC_ROWS B for a static
    row count and in a static program (``rows_of`` reports T rows as source 0 there too, but its advance resets
    t_this_step to 1). Rows from any other StepState field are refused. Every prompt dispatch lists
    ``(15, ctx.program.step_state, 0)`` itself: ``ctx.add`` appends it in a dynamic-T program only."""
    from .emit import ROW_SOURCE, STATIC_ROWS              # emit imports this module: resolved at call time
    v = op.outputs[0] if rows_from is None else rows_from
    rows, src = ctx.rows_of(replace(op, outputs=[v]))
    if src not in (ROW_SOURCE[T], STATIC_ROWS):
        raise ValueError(f"{op.kind}: {v.name!r} has the rows {v.shape[0]} (T_SRC {src}); a prompt kernel's rows are "
                         "the chunk's (T, t_this_step) or a static count")
    src = src if ctx.dynamic_t else STATIC_ROWS
    macros = {"STEP_STATE": "1", "T_SRC": str(src), "ROWS_MAX": f"{rows}u"}
    if src == STATIC_ROWS:
        macros["T_STATIC_ROWS"] = f"{rows}u"
    return rows, macros


def _gemm(ctx: _Ctx, op: Op) -> None:
    """``Gemm::bf16`` (``gemm.cu:122-132``): the product on the BF16 tensor in place. ``Gemm::native`` (``:169-185``):
    ``dequant_f16`` of the whole matrix (dequant's ``basalt_dequant_g32`` at ``DQ_OUT 1``) into the program's one FP16
    scratch, grown to its largest image (basalt slices only a matrix past its 32M-value scratch, none on this pack;
    each output reads only its own image row, so the bits are the same), then ``Gemm::f16``'s product on that image.
    The dequantization lists the scratch in ``writes``: the barrier pass orders it after the products that read the
    previous op's image and before its own, and ``reuse_arenas`` may pack the scratch."""
    entry = op.attrs.get("entry")
    if entry not in GEMM_ENTRIES:
        raise ValueError(f"basalt_prompt_gemm: attr entry must be one of {GEMM_ENTRIES}, got {entry!r}")
    (x, w), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op)
    native = entry == "native"
    tid, n, k = _weight(ctx, op, w, kernels.DEQUANT16_TYPES if native else (GGML_IDS["bf16"],))
    _expect(ctx, op, x, (rows, k), DType.F16 if native else DType.BF16,
            f"Gemm::{entry}'s {'FP16' if native else 'BF16'} rows of the weight's width")
    _expect(ctx, op, y, (rows, n), DType.F32, "contiguous columns of the weight's rows")
    if not native:
        return _product(ctx, op, _bind(ctx, op, 1, w, 2), True, rows, n, k, macros)
    img = (1, ctx.scratch("prompt_gemm_f16", 2 * n * k, shared=True), 0)    # the dequantization's out, the products' w
    # ponytail: one dequantization dispatch per matrix. On this Mac (M3 Max) a Q6_K matrix of 10240 or 12288 x 2560
    # takes 1.42-1.70 ms, over a dispatch's 1 ms, and would need two row0 halves; no Q6_K tensor of this pack has that
    # shape (its longest, 2560 x 6144, 0.85 ms; every other type at most 0.84 ms at its shapes here).
    kern = ctx.kernel("basalt_dequant16", kernels.basalt_dequant16_source(), "basalt_dequant_g32",
                      kernels.basalt_dequant16_macros(tid, "f16"))
    prm = ctx.params("basalt_dequant16", kernels.basalt_dequant16_params(tid, 0, n, k))
    grid, threads = kernels.basalt_dequant16_grid(n, k)
    ctx.add(kern, [_bind(ctx, op, 0, w, 1), img, (2, prm, 0), (15, ctx.program.step_state, 0)], (grid, 1, 1),
            (threads, 1, 1), f"{op.kind}:dequant_f16:{w.name}", writes=[1])
    _product(ctx, op, img, False, rows, n, k, macros)


def _product(ctx: _Ctx, op: Op, w: Tuple[int, str, int], bf16: bool, rows: int, n: int, k: int,
             macros: Dict[str, str]) -> None:
    """The product (``Gemm::bf16``'s, ``gemm.cu:122-132``, or ``Gemm::f16``'s, ``:158-167``) of x, ``op``'s first
    input, by the 16-bit rows ``w`` binds at slot 1 (a BF16 tensor in place, or a scratch's image) into y, its first
    output: one dispatch per static slice of ``kernels.basalt_prompt_gemm_slices(B, N, K)``, each with its own params
    record, every slice writing y."""
    slices = kernels.basalt_prompt_gemm_slices(rows, n, k)            # refuses k % 16
    kern = ctx.kernel("basalt_prompt_gemm", kernels.basalt_prompt_gemm_source(), "basalt_prompt_gemm",
                      {**macros, "GEMM_BF16": str(int(bf16))})
    bindings = [_bind(ctx, op, 0, op.inputs[0], 2), w, _bind(ctx, op, 2, op.outputs[0], 4)]
    for row0, nrows, col0, ncols in slices:
        prm = ctx.params("basalt_prompt_gemm", kernels.basalt_prompt_gemm_params(n, k, row0, col0))
        ctx.add(kern, bindings + [(3, prm, 0), (15, ctx.program.step_state, 0)], (-(-ncols // 64), -(-nrows // 64), 1),
                (256, 1, 1), f"{op.kind}:{op.attrs['entry']}:{op.inputs[1].name}", writes=[2])


def _convert(ctx: _Ctx, op: Op) -> None:
    """``to_f16`` (``kernels.cu:818-821``) or ``to_bf16`` without ``ylo`` (``:826-833``), as the output's dtype
    says: one thread per element of the bound's rows."""
    (x,), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op)
    n = ctx.shape(x)[-1]
    _expect(ctx, op, x, (rows, n), DType.F32, "the converted rows")
    if y.dtype not in (DType.F16, DType.BF16):
        raise ValueError(f"basalt_prompt_convert: {y.name!r} is {y.dtype}; the output is F16 (to_f16) or BF16 (to_bf16)")
    bf16 = y.dtype == DType.BF16
    _expect(ctx, op, y, (rows, n), y.dtype, "the input's rows, converted")
    kern = ctx.kernel("basalt_prompt_gemm", kernels.basalt_prompt_gemm_source(), "basalt_prompt_convert",
                      {**macros, "CV_BF16": str(int(bf16))})
    prm = ctx.params("basalt_prompt_convert", kernels.basalt_prompt_convert_params(n))
    ctx.add(kern, [_bind(ctx, op, 0, x, 4), _bind(ctx, op, 1, y, 2), (2, prm, 0), (15, ctx.program.step_state, 0)],
            (-(-rows * n // 256), 1, 1), (256, 1, 1), f"{op.kind}:{'to_bf16' if bf16 else 'to_f16'}:{x.name}",
            writes=[1])


BASALT_PROMPT_HANDLERS: Dict[str, object] = {"basalt_prompt_gemm": _gemm, "basalt_prompt_convert": _convert}


# ---- the hyper-connection kinds (kernels/common/basalt_prompt_hc.metal; src/prefill/kernels.cu:85-204) ----
# basalt's launchers (kernels.cu:867-887, :893-900) run T * HC threadgroups of 256 for gr_norm_rs and gr_write_norm_rs,
# one per (token, stream) row, and blocks_for(T * width) (:63) for the others; here each static row slice (row0, S) of
# kernels.basalt_prompt_hc_slices(B) is one dispatch of that shape at S rows, with its own params record.

HC_ROW_ENTRIES = ("prompt_gr_norm_rs", "prompt_gr_write_norm_rs")   # a threadgroup per row (t, c); both take eps
HC_SILU_N = 256   # silu's HC_N: no stream; the file compiles only at HC_N >= 256 (write_norm's v[HC_N / 256])


def _hc_io(op: Op, n_in: int, n_out: Tuple[int, ...]) -> Tuple[List[Value], List[Value]]:
    """``op``'s inputs and outputs, refused unless there are ``n_in`` and one of the counts ``n_out``."""
    if len(op.inputs) != n_in or len(op.outputs) not in n_out:
        raise ValueError(f"{op.kind}: {len(op.inputs)} inputs and {len(op.outputs)} outputs; it takes {n_in} inputs "
                         f"and {' or '.join(map(str, n_out))} outputs")
    return list(op.inputs), list(op.outputs)


def _hc_n(ctx: _Ctx, op: Op, r: Value, n: Optional[int] = None) -> int:
    """n, a stream's width: ``r``'s columns over basalt's four streams (or the given ``n``, the embedding's), refused
    unless ``r`` holds exactly four streams of n (HC = 4, kernels.cu:18-19)."""
    cols = ctx.shape(r)[-1]
    n = cols // 4 if n is None else n
    if cols != 4 * n:
        raise ValueError(f"{op.kind}: {r.name!r} has {cols} columns, not four streams of {n} (basalt's HC = 4)")
    return n


def _hc_weight(ctx: _Ctx, op: Op, w: Value, n: int) -> None:
    """``w``: the four streams' norm weights, 4n F32 values in any shape (``[4n]``, ``[1, 4n]``)."""
    if w.dtype != DType.F32 or int(np.prod(ctx.shape(w))) != 4 * n:
        raise ValueError(f"{op.kind}: {w.name!r} is {w.dtype} {list(ctx.shape(w))}, not {4 * n} f32 values (the four "
                         "streams' norm weights)")


def _hc_kernel(ctx: _Ctx, op: Op, entry: str, n: int, lr: int = 0, mix_bf: int = 0) -> Tuple[int, str]:
    """``(B, kernel)`` of a hyper-connection dispatch: ``prompt_rows``' macros and HC_N ``n``, HC_LR ``lr``, MIX_BF
    ``mix_bf`` and HC_EPS, the float32 bits of the attr ``eps`` for the two row kernels and 0 for the others, which do
    not read it. Every pipeline compiles all six entries, so each kind passes all four macros and n is at least 256 for
    every kind (gr_write_norm_rs keeps n / 256 values per thread, :128). An eps below 2^-102 is refused: the row norm's
    plain quotient and sum give basalt's bits from 2^-102 up, which also keeps fast::rsqrt's argument normal."""
    B, macros = prompt_rows(ctx, op)
    row = entry in HC_ROW_ENTRIES
    if n < 256:
        raise ValueError(f"{op.kind}: streams of {n} values; the hyper-connection kernels take at least 256 "
                         "(gr_write_norm_rs keeps n / 256 per thread, and every hc pipeline compiles it)")
    if row and n % 256:
        raise ValueError(f"{op.kind}: streams of {n} values are not whole sweeps of 256 threads (d = k + 256 j)")
    if not row and n % 4:
        raise ValueError(f"{op.kind}: streams of {n} values, not a multiple of 4")
    if entry == "prompt_gr_silu" and (lr < 4 or lr % 4):
        raise ValueError(f"{op.kind}: rows of {lr} values, not a positive multiple of 4")
    eps = np.float32(0.0)
    if row:
        eps = np.float32(op.attrs.get("eps", np.nan))
        # 2^-102, the least eps whose half ulp absorbs any subnormal S / n: IEEE's fl(S / n + eps) by plain / and +
        if not (np.isfinite(eps) and eps >= np.float32(2.0 ** -102)):
            raise ValueError(f"{op.kind}: attr eps must be a finite float32 of at least 2^-102, got "
                             f"{op.attrs.get('eps')!r}")
    if B * max(4 * n, lr) > (1 << 32) - 256:
        raise ValueError(f"{op.kind}: {B} rows of {max(4 * n, lr)} values overflow the kernels' 32-bit indices")
    macros.update(HC_N=str(n), HC_LR=str(lr), MIX_BF=str(mix_bf),
                  HC_EPS=f"as_type<float>(0x{int(eps.view(np.uint32)):08X}u)")
    return B, ctx.kernel("basalt_prompt_hc", kernels.basalt_prompt_hc_source(), entry, macros)


def _hc_add(ctx: _Ctx, op: Op, B: int, kern: str, ins: List[Tuple[int, Value]], outs: List[Tuple[int, Value]],
            width: Optional[int], label: Value) -> None:
    """One dispatch per static row slice ``(row0, S)`` of ``kernels.basalt_prompt_hc_slices(B)``, binding the ``(slot,
    value)`` pairs ``ins`` and ``outs`` (F32 values at 4-byte alignment, 16-bit ones at 2), the slice's params record at
    7 and StepState at 15, and writing ``outs``' slots: 4 S threadgroups of 256 for a row kernel (``width`` None),
    ceil(S * width / 256) for an element kernel over rows of ``width`` values."""
    bindings = [_bind(ctx, op, slot, v, 4 if v.dtype == DType.F32 else 2) for slot, v in ins + outs]
    writes = [slot for slot, _ in outs]
    for row0, rows in kernels.basalt_prompt_hc_slices(B):
        prm = ctx.params("basalt_prompt_hc", kernels.basalt_prompt_hc_params(row0))
        grid = (4 * rows, 1, 1) if width is None else (-(-rows * width // 256), 1, 1)
        ctx.add(kern, bindings + [(7, prm, 0), (15, ctx.program.step_state, 0)], grid, (256, 1, 1),
                f"{op.kind}:{label.name}", writes=writes)


def _hc_broadcast(ctx: _Ctx, op: Op) -> None:
    """``gr_broadcast`` (kernels.cu:199-204): each of a token's four streams a 32-bit copy of its embedding row."""
    (e,), (r,) = _hc_io(op, 1, (1,))
    n = _hc_n(ctx, op, r, ctx.shape(e)[-1])
    B, k = _hc_kernel(ctx, op, "prompt_gr_broadcast", n)
    _expect(ctx, op, e, (B, n), DType.F32, "the embedding rows")
    _expect(ctx, op, r, (B, 4 * n), DType.F32, "four streams of the embedding")
    _hc_add(ctx, op, B, k, [(0, e)], [(1, r)], 4 * n, e)


def _hc_norm(ctx: _Ctx, op: Op) -> None:
    """``gr_norm_rs`` (kernels.cu:85-102): each (token, stream) row's rsqrt(mean(r^2) + eps) and its normed BF16
    image."""
    (r, w), (rs, xn) = _hc_io(op, 2, (2,))
    n = _hc_n(ctx, op, r)
    B, k = _hc_kernel(ctx, op, "prompt_gr_norm_rs", n)
    _expect(ctx, op, r, (B, 4 * n), DType.F32, "a token's four streams")
    _hc_weight(ctx, op, w, n)
    _expect(ctx, op, rs, (B, 4), DType.F32, "the streams' row norms")
    _expect(ctx, op, xn, (B, 4 * n), DType.BF16, "the normed streams' BF16 image")
    _hc_add(ctx, op, B, k, [(0, r), (1, w)], [(2, rs), (3, xn)], None, w)


def _hc_silu(ctx: _Ctx, op: Op) -> None:
    """``gr_silu`` (kernels.cu:161-170): silu(lo / 4) as BF16."""
    (lo,), (lo16,) = _hc_io(op, 1, (1,))
    m = ctx.shape(lo)[-1]
    B, k = _hc_kernel(ctx, op, "prompt_gr_silu", HC_SILU_N, lr=m)
    _expect(ctx, op, lo, (B, m), DType.F32, "the low-rank rows")
    _expect(ctx, op, lo16, (B, m), DType.BF16, "their SiLU's BF16 image")
    _hc_add(ctx, op, B, k, [(0, lo)], [(1, lo16)], m, lo)


def _hc_mix(ctx: _Ctx, op: Op) -> None:
    """``gr_mix_r`` (kernels.cu:103-124): the mean over the four streams of the normed stream times sigm(gated), as
    F32, BF16 and FP16 (three outputs, MIX_BF 1), or without the BF16 image (two outputs, MIX_BF 0: the drafter's KV
    pass, prefill.cpp:1019)."""
    (r, rs, w, gated), outs = _hc_io(op, 4, (2, 3))
    mixed, *bf, mixed_h = outs
    n = _hc_n(ctx, op, r)
    B, k = _hc_kernel(ctx, op, "prompt_gr_mix_r", n, mix_bf=len(bf))
    _expect(ctx, op, r, (B, 4 * n), DType.F32, "a token's four streams")
    _expect(ctx, op, rs, (B, 4), DType.F32, "the streams' row norms")
    _hc_weight(ctx, op, w, n)
    _expect(ctx, op, gated, (B, 4 * n), DType.F32, "the four streams' gates")
    _expect(ctx, op, mixed, (B, n), DType.F32, "the mix of the four streams")
    for v in bf:
        _expect(ctx, op, v, (B, n), DType.BF16, "the mix's BF16 image")
    _expect(ctx, op, mixed_h, (B, n), DType.F16, "the mix's FP16 image")
    written = [(4, mixed), *[(5, v) for v in bf], (6, mixed_h)]          # MIX_BF 0 binds no slot 5
    _hc_add(ctx, op, B, k, [(0, r), (1, rs), (2, w), (3, gated)], written, n, w)


def _hc_write(ctx: _Ctx, op: Op) -> None:
    """``gr_write`` (kernels.cu:192-198): r' = r + bo * 2 sigm(inj / 4), stream by stream."""
    (r, bo, inj), (r2,) = _hc_io(op, 3, (1,))
    n = _hc_n(ctx, op, r)
    B, k = _hc_kernel(ctx, op, "prompt_gr_write", n)
    _expect(ctx, op, r, (B, 4 * n), DType.F32, "a token's four streams")
    _expect(ctx, op, bo, (B, n), DType.F32, "the block's output")
    _expect(ctx, op, inj, (B, 4), DType.F32, "the four streams' injections")
    _expect(ctx, op, r2, (B, 4 * n), DType.F32, "the written streams")
    _hc_add(ctx, op, B, k, [(0, r), (1, bo), (2, inj)], [(3, r2)], 4 * n, r)


def _hc_write_norm(ctx: _Ctx, op: Op) -> None:
    """``gr_write_norm_rs`` (kernels.cu:125-160): ``gr_write``, then ``gr_norm_rs`` on the written streams with
    w_next."""
    (r, bo, inj, w), (r2, rs, xn) = _hc_io(op, 4, (3,))
    n = _hc_n(ctx, op, r)
    B, k = _hc_kernel(ctx, op, "prompt_gr_write_norm_rs", n)
    _expect(ctx, op, r, (B, 4 * n), DType.F32, "a token's four streams")
    _expect(ctx, op, bo, (B, n), DType.F32, "the block's output")
    _expect(ctx, op, inj, (B, 4), DType.F32, "the four streams' injections")
    _hc_weight(ctx, op, w, n)
    _expect(ctx, op, r2, (B, 4 * n), DType.F32, "the written streams")
    _expect(ctx, op, rs, (B, 4), DType.F32, "the written streams' row norms")
    _expect(ctx, op, xn, (B, 4 * n), DType.BF16, "the normed streams' BF16 image")
    _hc_add(ctx, op, B, k, [(0, r), (1, bo), (2, inj), (3, w)], [(4, r2), (5, rs), (6, xn)], None, w)


BASALT_PROMPT_HANDLERS.update({"basalt_prompt_hc_broadcast": _hc_broadcast, "basalt_prompt_hc_norm": _hc_norm,
                               "basalt_prompt_hc_silu": _hc_silu, "basalt_prompt_hc_mix": _hc_mix,
                               "basalt_prompt_hc_write": _hc_write, "basalt_prompt_hc_write_norm": _hc_write_norm})
