"""Single-kernel runs of basalt's prompt path (kernels/common/basalt_prompt_*.metal) for the GPU tests. Every prompt
kernel reads its rows from StepState at slot 15 (monolith/kernels.py's PROMPT_ROWS), so ``run_rows`` is
basalt_gpu.run with a StepState record bound there. A source compiles as the emitter registers it (the StepState
struct spliced in after the prelude) with the dynamic-T program's row macros: STEP_STATE 1, T_SRC 0 and ROWS_MAX the
compiled bound B; the static form (T_SRC 2) is the contract and program tiers' concern."""

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



def prompt_hc(entry, ins, out_bytes, *, B, rows, n=2560, lr=320, mix_bf=1, static=False, slices=None, fill=0xFF,
              done=0):
    """basalt_prompt_hc's kernel ``entry`` compiled at bound ``B`` with the handler's macros (HC_N ``n``, HC_LR
    ``lr``, MIX_BF ``mix_bf``, HC_EPS 1e-6f), dynamic (``rows`` and ``done`` in StepState) or, with ``static``, a
    static program's (T_STATIC_ROWS ``rows``; t_this_step 1, as its advance leaves it). ``ins`` (bytes) bind at slots
    0, 1, ... and the outputs (``out_bytes`` byte counts, exactly B rows each, starting as ``fill`` bytes) after them,
    the kernel's slots; one run_rows per slice ``(row0, S)`` (default the handler's, kernels.basalt_prompt_hc_slices(B))
    on the same outputs, 4 S threadgroups of 256 for a row kernel, ceil(S * W / 256) for an element kernel. Returns
    every input's and output's bytes as the last run left them."""
    eps = int(np.float32(1e-6).view(np.uint32))
    macros = {"HC_N": str(n), "HC_LR": str(lr), "MIX_BF": str(mix_bf), "HC_EPS": f"as_type<float>(0x{eps:08X}u)"}
    if static:
        macros.update(T_SRC="2", T_STATIC_ROWS=f"{rows}u")
    pso = prompt_pipeline(kernels.basalt_prompt_hc_source(), entry, B, macros)
    width = {"prompt_gr_mix_r": n, "prompt_gr_silu": lr, "prompt_gr_write": 4 * n,   # an element kernel's W
             "prompt_gr_broadcast": 4 * n}.get(entry)                                   # (row kernels: None)
    bufs = [*ins, *(bytes([fill]) * k for k in out_bytes)]
    for row0, s in kernels.basalt_prompt_hc_slices(B) if slices is None else slices:
        grid = (4 * s, 1, 1) if width is None else (-(-s * width // 256), 1, 1)
        bufs = run_rows(pso, bufs, kernels.basalt_prompt_hc_params(row0), 7, grid, (256, 1, 1), 1 if static else rows,
                        0, done=done)[:-1]
    return bufs


def hc_probe(fn, x):
    """``fast::rsqrt`` ("rsqrt"), ``expf_ieee(sign_flip(x))`` ("exp_neg") or ``fast::exp2`` ("exp2") of each float of
    ``x`` on this GPU, one thread per value (a test-only kernel, no op kind): the approximate operations' values the
    host twins take. Returns float32 of ``x``'s shape."""
    probe = {"rsqrt": "fast::rsqrt(v)", "exp_neg": "expf_ieee(sign_flip(v))", "exp2": "fast::exp2(v)"}[fn]
    src = (kernels.PRELUDE + kernels.template("basalt_fp.metal") + kernels.template("basalt_ieee.metal")
           + f"#define PROBE(v) {probe}\n"
           + "kernel void hc_probe(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],\n"
           + "                     uint i [[thread_position_in_grid]]) { y[i] = PROBE(x[i]); }\n")
    x = np.ascontiguousarray(x, np.float32)
    m = -(-max(x.size, 1) // 256) * 256                      # whole threadgroups of 256: x padded with zeros
    xp = np.zeros(m, np.float32)
    xp.view(np.uint32)[: x.size] = x.view(np.uint32).ravel()   # bits, NaN payloads included
    y = run(pipeline(src, "hc_probe"), [xp.tobytes(), 4 * m], None, 0, m // 256, 256)[1]
    return np.frombuffer(y, np.float32)[: x.size].reshape(x.shape)
