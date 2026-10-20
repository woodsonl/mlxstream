"""Single-kernel runs of the n-gram block's Metal port (kernels/common/basalt_gr_norm.metal and basalt_ple.metal) for
the GPU tests, on tests/kernels/basalt_gpu.py's device, pipelines and runs: one wrapper per entry, every output
starting as 0xFF bytes, and GPU probes of the basalt_fp.metal helpers whose values the numpy emulation
(tests/kernels/basalt_emu.py) takes from the GPU."""

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


def gate(key, query):
    """``basalt_ple_gate`` in basalt's launch (``PLE_GRIDS["gate"]``: one threadgroup of 512 threads per stream) on the
    PLE_H rows of PLE_N normalized key and query values, into gates that start as 0xFF bytes; (PLE_H,) float32. The
    block size fixes the reduction order, so the pipeline must take a threadgroup of that size."""
    pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_gate")
    grid, threads = kernels.PLE_GRIDS["gate"]
    assert pso.max_threads_per_threadgroup >= threads, (pso.max_threads_per_threadgroup, threads)
    ins = [np.ascontiguousarray(a, np.float32).reshape(grid * kernels.PLE_N).tobytes() for a in (key, query)]
    out = gpu.run(pso, ins + [4 * grid], kernels.basalt_ple_gate_params(), 3, grid, threads, 0xFF)
    return np.frombuffer(out[2], np.float32)


def broadcast(value, gate):
    """``basalt_ple_broadcast`` in basalt's launch: the PLE_N values times each of the PLE_H gates, into a PLE_D output
    that starts as 0xFF bytes; float32."""
    pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_broadcast")
    ins = [np.ascontiguousarray(value, np.float32).reshape(kernels.PLE_N).tobytes(),
           np.ascontiguousarray(gate, np.float32).reshape(kernels.PLE_H).tobytes()]
    out = gpu.run(pso, ins + [4 * kernels.PLE_D], None, None, *kernels.PLE_GRIDS["broadcast"], 0xFF)
    return np.frombuffer(out[2], np.float32)


def conv(hist, normalized, taps, hidden, gated):
    """``basalt_ple_conv`` in basalt's launch: ``hist`` the history in basalt's layout (row r of channel c at
    c * 9 + r), ``taps`` the fp16 kernel (float16 or its bits) at [k + 4 c], ``normalized``, ``hidden`` and ``gated``
    PLE_D values each; conv and result start as 0xFF bytes. Returns (conv, result), PLE_D float32 each."""
    pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_conv")
    d = kernels.PLE_D

    def f32(a, n):
        return np.ascontiguousarray(a, np.float32).reshape(n).tobytes()

    ins = [f32(hist, kernels.PLE_HIST * d), f32(normalized, d),
           np.ascontiguousarray(taps).view(np.uint16).reshape(4 * d).tobytes(), f32(hidden, d), f32(gated, d)]
    out = gpu.run(pso, ins + [4 * d, 4 * d], None, None, *kernels.PLE_GRIDS["conv"], 0xFF)
    return np.frombuffer(out[5], np.float32), np.frombuffer(out[6], np.float32)


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


def sqrtf_ftz(x):
    """basalt_fp.metal's sqrtf_ftz (``sqrtf`` in a fast-math file) on the GPU, elementwise: the values the gate uses."""
    return _helper("sqrtf_ftz", x)
