"""Single-kernel runs of the n-gram block's Metal port (kernels/common/basalt_gr_norm.metal) for the GPU tests, on
tests/kernels/basalt_gpu.py's device, pipelines and runs: one wrapper per entry, every output starting as 0xFF bytes,
and GPU probes of the basalt_fp.metal helpers whose values the numpy emulation (tests/kernels/basalt_emu.py) takes
from the GPU."""

import numpy as np

from monolith import kernels
from tests.kernels import basalt_gpu as gpu


def gr_norm(x, gamma, n_rows, n_cols, gamma_rows, eps):
    """``basalt_gr_norm`` in basalt's launch (``basalt_gr_norm_macros``): ``n_rows`` rows of ``n_cols`` float32 values
    ``x``, row r with row r % ``gamma_rows`` of ``gamma``, into an output that starts as 0xFF bytes; (n_rows, n_cols)
    float32. The block size fixes the reduction order, so the pipeline must take a threadgroup of that size."""
    macros = kernels.basalt_gr_norm_macros(n_cols)
    pso = gpu.pipeline(kernels.basalt_gr_norm_source(), "basalt_gr_norm", tuple(sorted(macros.items())))
    grid, threads = kernels.basalt_gr_norm_grid(n_rows, macros)
    assert pso.max_threads_per_threadgroup >= threads, (pso.max_threads_per_threadgroup, threads)
    ins = [np.ascontiguousarray(a, np.float32).tobytes() for a in (x, gamma)]
    out = gpu.run(pso, ins + [4 * n_rows * n_cols], kernels.basalt_gr_norm_params(n_cols, gamma_rows, eps), 3, grid,
                  threads, 0xFF)
    return np.frombuffer(out[2], np.float32).reshape(n_rows, n_cols)


def _helper(name, x):
    """basalt_fp.metal's float helper ``name`` on the GPU, elementwise (basalt_gpu.expf_ftz's pattern)."""
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + f"""
kernel void probe(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {{
    if (i < n) y[i] = {name}(x[i]);
}}
"""
    x = np.ascontiguousarray(x, np.float32)
    flat = x.reshape(-1)
    out = gpu.run(gpu.pipeline(src, "probe"), [flat.tobytes(), 4 * flat.size], np.uint32(flat.size).tobytes(), 2,
                  -(-flat.size // 256), 256, 0xFF)
    return np.frombuffer(out[1], np.float32).reshape(x.shape)


def rsqrtf_ftz(x):
    """basalt_fp.metal's rsqrtf_ftz (``rsqrtf`` in a fast-math file) on the GPU, elementwise: the values the norm
    uses."""
    return _helper("rsqrtf_ftz", x)
