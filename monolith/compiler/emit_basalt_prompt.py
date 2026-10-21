# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/prefill/prefill.cpp:1188-1199 and src/prefill/gemm.cu:134-156 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""The emitter's handlers for basalt's prompt path (``monolith/ops/basalt_prompt.py``), merged into ``emit.HANDLERS``.
Every dispatch binds StepState at slot 15 and reads the chunk's rows only through ``prompt_rows(st)``
(``kernels.PROMPT_ROWS``; ``Gemm::native``'s dequantization and ``Gemm::bf16_w``'s copy image read none), in the
dynamic-T prompt program and in a static one; its macros, grid, threadgroup and params come from
``monolith/kernels.py``'s builders only, and a params record holds geometry, never a row count, a position or a
pointer (invariant 4). The weight checks are ``native_proj``'s and ``bf16_proj``'s (``prefill.cpp:1188-1199``): the
entry must agree with the weight; and ``Gemm::bf16_w``'s (``src/prefill/gemm.cu:137-141``): a copy must be the
product's N x K."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import Op, Value
from ..core.shapes import T
from .emit_basalt import GGML_IDS, _bind, _expect, _weight

if TYPE_CHECKING:
    from .emit import _Ctx

GEMM_ENTRIES = ("bf16", "native", "bf16_w")


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
    previous op's image and before its own, and ``reuse_arenas`` may pack the scratch. ``Gemm::bf16_w`` on an int8
    copy: ``_gemm_copy``. Each entry's input count is checked before its inputs are unpacked."""
    entry = op.attrs.get("entry")
    if entry not in GEMM_ENTRIES:
        raise ValueError(f"basalt_prompt_gemm: attr entry must be one of {GEMM_ENTRIES}, got {entry!r}")
    n_in = 3 if entry == "bf16_w" else 2
    if len(op.inputs) != n_in:
        raise ValueError(f"basalt_prompt_gemm: entry {entry!r} takes {n_in} inputs, "
                         f"{'(x, wq, ws)' if n_in == 3 else '(x, w)'}, got {len(op.inputs)}")
    if entry == "bf16_w":
        return _gemm_copy(ctx, op)
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


def _gemm_copy(ctx: _Ctx, op: Op) -> None:
    """``Gemm::bf16_w`` on an int8 hyper-connection copy (``gemm.cu:134-154``): ``hc8_dequant_bf16``
    (``hc_fused.cu:1328-1333``; ``basalt_hc8_dequant_bf16``) of the whole copy, codes ``wq`` and scales ``ws``, into the
    program's shared GEMM scratch, then ``Gemm::bf16``'s product on that BF16 image, every slice reading the one image.
    basalt's row-slice loop (``:143``, ``:148-152``) runs once on this pack (every copy fits its scratch) and is not
    ported. The image is rebuilt on every call, as basalt's is, and lists the scratch in ``writes``, so the barrier
    pass orders it behind the previous op's products and ahead of its own. N and K are the product's (y's columns,
    x's), as ``bf16_w`` takes them; a copy of another shape is refused with basalt's message (``:137-141``, where it
    exits). The image reads every byte of the codes and the scales, so each must be a constant held as rows (not a
    weight, not a pack slab) whose pack row, if a pack holds it, is exactly its bytes, as ``_weight`` requires of a
    weight's rows; basalt's reads stay inside the copy it allocated at its size (``hc_fused.cu:1212-1217``)."""
    (x, wq, ws), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op)
    n, k = ctx.shape(y)[-1], ctx.shape(x)[-1]
    _expect(ctx, op, x, (rows, k), DType.BF16, "Gemm::bf16_w's BF16 rows")
    _expect(ctx, op, y, (rows, n), DType.F32, "the product's rows")
    if ctx.shape(wq) != (n, k):
        raise ValueError(f"{op.kind}: {wq.name!r}: prefill gemm: an 8-bit copy is "
                         f"{' x '.join(map(str, ctx.shape(wq)))}, the product wants {n} x {k}")
    _expect(ctx, op, wq, (n, k), DType.U8, "the copy's int8 codes")
    _expect(ctx, op, ws, (n, (k // 32 + 7) & ~7), DType.F16, "the copy's fp16 scales, (K / 32 + 7) & ~7 a row")
    for v in (wq, ws):
        if not v.is_const or any(v.name in pk.slabs for pk in ctx.packs):
            raise ValueError(f"{op.kind}: {v.name!r} is {'a pack slab' if v.is_const else 'not a constant'}; a copy's "
                             "codes and scales are the constants hc8/<name>/q and /s, rows as stored")
        aux = next((pk.aux[v.name] for pk in ctx.packs if v.name in pk.aux), None)
        if aux is not None and int(aux["nbytes"]) != (need := math.prod(ctx.shape(v)) * v.dtype.itemsize):
            raise ValueError(f"{op.kind}: the copy's {v.name!r} is {need} bytes of {v.dtype.short} "
                             f"{list(ctx.shape(v))}, but its pack entry holds {aux['nbytes']}")
    img = ctx.scratch("prompt_gemm_f16", 2 * n * k, shared=True)
    kern = ctx.kernel("basalt_hc8_dequant_bf16", kernels.basalt_prompt_hc8_source(), "basalt_hc8_dequant_bf16", {})
    prm = ctx.params("hc8_image", kernels.basalt_hc8_image_params(n, k))
    grid, threads = kernels.basalt_hc8_image_grid(n, k)
    ctx.add(kern, [_bind(ctx, op, 0, wq, 1), _bind(ctx, op, 1, ws, 2), (2, img, 0), (3, prm, 0),
                   (15, ctx.program.step_state, 0)], (grid, 1, 1), (threads, 1, 1),
            f"{op.kind}:hc8_dequant_bf16:{wq.name}", writes=[2])
    _product(ctx, op, (1, img, 0), True, rows, n, k, macros)


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
    """``to_f16`` (``kernels.cu:818-821``): one thread per element of the bound's rows."""
    (x,), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op)
    n = ctx.shape(x)[-1]
    _expect(ctx, op, x, (rows, n), DType.F32, "the converted rows")
    if y.dtype == DType.BF16:
        raise ValueError(f"basalt_prompt_convert: {y.name!r} is BF16 (to_bf16), which rounds with dequant's "
                         "bf16_rne_nonan, which this branch does not have yet")
    if y.dtype != DType.F16:
        raise ValueError(f"basalt_prompt_convert: {y.name!r} is {y.dtype}; the output is F16 (to_f16) or BF16 (to_bf16)")
    _expect(ctx, op, y, (rows, n), DType.F16, "the input's rows, converted")
    kern = ctx.kernel("basalt_prompt_gemm", kernels.basalt_prompt_gemm_source(), "basalt_prompt_convert",
                      {**macros, "CV_BF16": "0"})
    prm = ctx.params("basalt_prompt_convert", kernels.basalt_prompt_convert_params(n))
    ctx.add(kern, [_bind(ctx, op, 0, x, 4), _bind(ctx, op, 1, y, 2), (2, prm, 0), (15, ctx.program.step_state, 0)],
            (-(-rows * n // 256), 1, 1), (256, 1, 1), f"{op.kind}:to_f16:{x.name}", writes=[1])


BASALT_PROMPT_HANDLERS: Dict[str, object] = {"basalt_prompt_gemm": _gemm, "basalt_prompt_convert": _convert}
