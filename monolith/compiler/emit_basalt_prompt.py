# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/prefill/prefill.cpp:1188-1199 @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE).
"""The emitter's handlers for basalt's prompt path (``monolith/ops/basalt_prompt.py``), merged into ``emit.HANDLERS``.
Every dispatch binds StepState at slot 15 and reads the chunk's rows through ``prompt_rows(st)``
(``kernels.PROMPT_ROWS``), in the dynamic-T prompt program and in a static one; its macros, grid, threadgroup and
params come from ``monolith/kernels.py``'s builders only, and a params record holds geometry, never a row count, a
position or a pointer (invariant 4). The weight checks are ``native_proj``'s and ``bf16_proj``'s
(``prefill.cpp:1188-1199``): the entry must agree with the weight."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Optional, Tuple

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
    """``Gemm::bf16`` (``gemm.cu:122-132``): one product dispatch per static slice of
    ``kernels.basalt_prompt_gemm_slices(B, N, K)``, each with its own params record, every slice writing y."""
    entry = op.attrs.get("entry")
    if entry not in GEMM_ENTRIES:
        raise ValueError(f"basalt_prompt_gemm: attr entry must be one of {GEMM_ENTRIES}, got {entry!r}")
    if entry == "native":
        raise ValueError("basalt_prompt_gemm: entry 'native' dequantizes through dequant's basalt_dequant_g32, which "
                         "this branch does not have yet")
    (x, w), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op)
    _, n, k = _weight(ctx, op, w, (GGML_IDS["bf16"],))
    _expect(ctx, op, x, (rows, k), DType.BF16, "Gemm::bf16's BF16 rows of the weight's width")
    _expect(ctx, op, y, (rows, n), DType.F32, "contiguous columns of the weight's rows")
    slices = kernels.basalt_prompt_gemm_slices(rows, n, k)            # refuses k % 16
    kern = ctx.kernel("basalt_prompt_gemm", kernels.basalt_prompt_gemm_source(), "basalt_prompt_gemm",
                      {**macros, "GEMM_BF16": "1"})
    bindings = [_bind(ctx, op, 0, x, 2), _bind(ctx, op, 1, w, 2), _bind(ctx, op, 2, y, 4)]
    for row0, nrows, col0, ncols in slices:
        prm = ctx.params("basalt_prompt_gemm", kernels.basalt_prompt_gemm_params(n, k, row0, col0))
        ctx.add(kern, bindings + [(3, prm, 0), (15, ctx.program.step_state, 0)], (-(-ncols // 64), -(-nrows // 64), 1),
                (256, 1, 1), f"{op.kind}:{entry}:{w.name}", writes=[2])


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
