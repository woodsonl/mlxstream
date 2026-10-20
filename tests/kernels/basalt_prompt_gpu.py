"""Single-kernel runs of basalt's prompt path (kernels/common/basalt_prompt_*.metal) for the GPU tests. Every prompt
kernel reads its rows from StepState at slot 15 (monolith/kernels.py's PROMPT_ROWS), so ``run_rows`` is
basalt_gpu.run with a StepState record bound there. A source compiles as the emitter registers it (the StepState
struct spliced in after the prelude) with the dynamic-T program's row macros: STEP_STATE 1, T_SRC 0 and ROWS_MAX the
compiled bound B; the static form (T_SRC 2) is the contract and program tiers' concern."""

import numpy as np

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from monolith.runtime import _native as nt
from tests.contract.test_basalt_ops import spliced
from tests.kernels.basalt_gpu import device, pipeline


def prompt_pipeline(source, name, B, macros=()):
    """Kernel ``name`` of ``source`` as a dynamic-T program at bound ``B`` compiles it, with ``macros`` beside."""
    m = {"STEP_STATE": "1", "T_SRC": "0", "ROWS_MAX": f"{B}u", **dict(macros)}
    return pipeline(spliced(source), name, tuple(sorted(m.items())))


def run_rows(pso, buffers, params, slot, grid, threads, rows, position, fill=None, done=0):
    """basalt_gpu.run with a StepState record (the layout ``spliced`` gives the source; ``t_this_step`` = ``rows``,
    ``position``, ``done``) bound at slot 15 and returned last. ``grid`` (threadgroups) and ``threads`` are 3-tuples."""
    dev = device()
    state = StepStateLayout().pack({"t_this_step": rows, "position": position, "done": done})
    srcs = [*buffers, state]
    bufs = [nt.Buffer(dev, b) for b in srcs]
    for buf, src in zip(bufs, buffers):
        if fill is not None and not isinstance(src, (bytes, bytearray)):
            buf.fill(fill)
    d = nt.Dispatch().pipeline(pso)
    for k, b in enumerate(bufs[:-1]):
        d = d.buffer(k, b)
    d = d.buffer(15, bufs[-1])
    if params is not None:
        d = d.bytes(slot, params)
    r = nt.Queue(dev).run([d.grid(*grid).threadgroup(*threads)])
    assert not r.error, r.error
    return [b.read(0, len(src) if isinstance(src, (bytes, bytearray)) else src) for b, src in zip(bufs, srcs)]


def prompt_gemm(xb, wb, *, bf16, B, rows=None, slices=None, fill=0xFF, done=0):
    """basalt_prompt_gemm: ``Gemm::bf16``'s product (``bf16``) or ``Gemm::native``'s FP16 product of the 16-bit bits
    ``xb`` [B, K] and ``wb`` [N, K] (gemm.cu:122-132, :158-185), compiled at bound ``B`` and run with ``rows`` (default
    B) and ``done`` in StepState: one run_rows per slice ``(row0, nrows, col0, ncols)`` into one y of exactly B * N
    floats whose bytes start as ``fill``. ``slices`` default to the handler's rule, kernels.basalt_prompt_gemm_slices(B,
    N, K). Returns y's bits, uint32 [B, N]."""
    xb, wb = np.ascontiguousarray(xb, np.uint16), np.ascontiguousarray(wb, np.uint16)
    (nx, k), (n, kw) = xb.shape, wb.shape
    assert nx == B and kw == k, f"x {xb.shape} and w {wb.shape} at bound {B}"
    pso = prompt_pipeline(kernels.basalt_prompt_gemm_source(), "basalt_prompt_gemm", B, {"GEMM_BF16": str(int(bf16))})
    y = bytes([fill]) * (4 * B * n)
    for row0, nrows, col0, ncols in kernels.basalt_prompt_gemm_slices(B, n, k) if slices is None else slices:
        y = run_rows(pso, [xb.tobytes(), wb.tobytes(), y], kernels.basalt_prompt_gemm_params(n, k, row0, col0), 3,
                     (-(-ncols // 64), -(-nrows // 64), 1), (256, 1, 1), B if rows is None else rows, 0, done=done)[2]
    return np.frombuffer(y, np.uint32).reshape(B, n)


def convert(x, *, bf16, B, rows, fill=0xFF):
    """basalt_prompt_convert: ``to_bf16`` (``bf16``) or ``to_f16`` of the floats ``x`` [B, n] (kernels.cu:818-833),
    compiled at bound ``B`` and run with ``rows`` in StepState, one thread per element, into one y of exactly B * n
    halves whose bytes start as ``fill``. ``x``'s bits reach the kernel as they are (NaN payloads included). Returns
    y's bits, uint16 [B, n]."""
    x = np.ascontiguousarray(x)
    assert x.dtype == np.float32 and x.ndim == 2 and len(x) == B, f"x {x.dtype} {x.shape} at bound {B}"
    n = x.shape[1]
    pso = prompt_pipeline(kernels.basalt_prompt_gemm_source(), "basalt_prompt_convert", B, {"CV_BF16": str(int(bf16))})
    y = run_rows(pso, [x.tobytes(), 2 * B * n], kernels.basalt_prompt_convert_params(n), 2, (-(-B * n // 256), 1, 1),
                 (256, 1, 1), rows, 0, fill=fill)[1]
    return np.frombuffer(y, np.uint16).reshape(B, n)
