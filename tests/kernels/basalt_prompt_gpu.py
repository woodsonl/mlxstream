"""Single-kernel runs of basalt's prompt path (kernels/common/basalt_prompt_*.metal) for the GPU tests. Every prompt
kernel reads its rows from StepState at slot 15 (monolith/kernels.py's PROMPT_ROWS), so ``run_rows`` is
basalt_gpu.run with a StepState record bound there. A source compiles as the emitter registers it (the StepState
struct spliced in after the prelude) with the dynamic-T program's row macros: STEP_STATE 1, T_SRC 0 and ROWS_MAX the
compiled bound B; the static form (T_SRC 2) is the contract and program tiers' concern. The copy image
(basalt_prompt_hc8.metal) reads no rows, so ``hc8_image`` is basalt_gpu.run alone."""

import numpy as np

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from monolith.formats.gguf import ggml_row_bytes
from monolith.runtime import _native as nt
from tests.contract.test_basalt_ops import spliced
from tests.kernels.basalt_gpu import device, pipeline, run


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


def native(type_id, blocks, xb, *, B, rows=None, fill=0xFF):
    """``Gemm::native`` (gemm.cu:169-185) of the fp16 bits ``xb`` [B, K] by the GGUF rows ``blocks`` [N, row bytes] of
    type ``type_id``: basalt_dequant_g32 at DQ_OUT 1 (dequant_f16, through the dequantizer's builders) over all N rows
    into an fp16 image of exactly N * K halves, then prompt_gemm's FP16 product on that image (rows and y as there).
    The image's and y's bytes start as ``fill``. Two runs here; the program's two dispatches in one command buffer are
    the program tier's. Returns (y's bits uint32 [B, N], the image's bits uint16 [N, K])."""
    blocks = np.ascontiguousarray(blocks, np.uint8)
    n, k = len(blocks), np.shape(xb)[1]
    assert blocks.shape[1] == ggml_row_bytes(type_id, k), f"blocks {blocks.shape} of type {type_id} at K {k}"
    pso = pipeline(kernels.basalt_dequant16_source(), "basalt_dequant_g32",
                   tuple(kernels.basalt_dequant16_macros(type_id, "f16").items()))
    image = run(pso, [blocks.tobytes(), 2 * n * k], kernels.basalt_dequant16_params(type_id, 0, n, k), 2,
                *kernels.basalt_dequant16_grid(n, k), fill)[1]
    image = np.frombuffer(image, np.uint16).reshape(n, k)
    return prompt_gemm(xb, image, bf16=False, B=B, rows=rows, fill=fill), image


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


def hc8_image(q, s, rows, cols, row0=0, n=None, guard=0):
    """basalt_hc8_dequant_bf16: hc8_dequant_bf16's BF16 image (hc_fused.cu:1229-1244, launched as :1328-1333) of rows
    [row0, row0 + n) (n None: every row from row0) of the ``rows`` x ``cols`` copy with codes ``q`` (int8 bit
    patterns, [rows, cols]) and scales ``s`` (fp16 bits, uint16 [rows, srow]), each bound as exactly its bytes. One
    dispatch, compiled as the emitter registers the source, into a buffer of n * cols values and ``guard`` more, every
    byte 0xFF: with ``guard`` 0 the binding is exactly the image, so shader validation reports any store past it.
    Returns the buffer's bits, uint16 [n * cols + guard]. The image reads no StepState."""
    n = rows - row0 if n is None else n
    q, s = np.ascontiguousarray(q), np.ascontiguousarray(s)
    assert (q.dtype.itemsize == 1 and q.shape == (rows, cols) and s.dtype == np.uint16
            and s.shape == (rows, (cols // 32 + 7) & ~7)), \
        f"q {q.dtype} {q.shape} and s {s.dtype} {s.shape}: not a {rows} x {cols} copy"
    pso = pipeline(spliced(kernels.basalt_prompt_hc8_source()), "basalt_hc8_dequant_bf16")
    grid, threads = kernels.basalt_hc8_image_grid(n, cols)
    out = run(pso, [q.tobytes(), s.tobytes(), 2 * (n * cols + guard)],
              kernels.basalt_hc8_image_params(rows, cols, row0, n), 3, grid, threads, fill=0xFF)[2]
    return np.frombuffer(out, np.uint16)
