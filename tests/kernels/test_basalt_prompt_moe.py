# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) tests/cuda/prefill_route_count_test.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): its ids, widths, accumulation and pass
# criterion; with the rules of src/kernels/router_top10_parity.cpp:33-60, :86-90, :119-146 and
# src/kernels/shared_expert_parity.cpp:54-62, :250-285, :445-462 as the nearest bounds.
"""basalt's prompt-path MoE element-wise kernels (src/prefill/kernels.cu:626-723: route_kernel<16>, route_count,
swiglu_pair and moe_combine) on Metal (kernels/common/basalt_prompt_moe.metal), and the library expf the router calls
(kernels/common/basalt_libdevice.metal's nv_expf), against host references (tests/kernels/basalt_emu.py).

- ``test_nv_expf_bits``: nv_expf, libdevice's __nv_expf as nvcc 13.0 inlines it at src/prefill/kernels.cu:642,
  through a probe kernel, bitwise equal to its host twin ``basalt_emu.nv_expf_emu`` (prefill_kernels.md section 3's
  PTX in IEEE float32, its ex2.approx.ftz the GPU's fast::exp2 from ``hc_probe("exp2", .)`` at the twin's
  arguments), a NaN result by NaN-ness only (the Metal multiply's NaN bits are not CUDA's 0x7fffffff; route_kernel
  turns a NaN probability into -FLT_MAX, :645, so no output carries them); and on [-103.97, -87.34], where expf is
  subnormal, against float64 exp: nonzero wherever exp(x) >= 2^-149 and within one subnormal step (2^-149, one ulp)
  of exp(x) correctly rounded to float32.
  Arguments (``nv_expf_args``): every float in [-104, -87.3], 513 floats around each argument where t * 252 crosses an
  integer, 2^16 uniform draws over [-104, 89], 513 floats around ln(FLT_MAX) and the specials (+-0, subnormals,
  +-2^-126, +-inf, NaNs). The test also asserts that t saturates at 0 and at 1, that nv_expf's correction of
  floor(fl(t * 252)) runs and gives fma.rm's floor of the exact t * 252 on every argument, and that the results include
  subnormals, and +0 and +inf from finite arguments."""

import numpy as np

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu
from tests.kernels.basalt_prompt_gpu import hc_probe

F32, U32 = np.float32, np.uint32

# ---- nv_expf (src/prefill/kernels.cu:642, the library expf) -------------------------------------------------------

C1 = U32(0x3BBB989D).view(F32)                  # log2(e) / 252, __nv_expf's first fma: t = sat(x C1 + 1/2)

NV_EXPF = """
kernel void probe_nv_expf(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = as_type<uint>(nv_expf(x[i]));
}
"""


def nv_expf_args():
    """test_nv_expf_bits's arguments, float32 (the module docstring's list)."""
    rng = np.random.default_rng(642)

    def around(v):                              # the 513 floats centred on each float32 of v (none near 0)
        b = np.asarray(v, F32).view(U32).astype(np.int64)
        return (b[..., None] + np.arange(-256, 257)).astype(U32).view(F32).ravel()

    m = np.delete(np.arange(253), 126)          # t * 252 = m at x = (m / 252 - 1/2) / C1 (m = 126: x = 0)
    lo, hi = np.array([-87.3, -104.0], F32).view(U32)                   # a negative float's bits grow with |x|
    sub = rng.integers(1, 1 << 23, 64, dtype=U32)
    specials = np.r_[U32([0, 1, 2, 0x007FFFFF, 0x00800000, 0x7F800000, 0x7FC00000, 0x7F800001, 0x7FFFFFFF]), sub]
    return np.concatenate([np.arange(lo, hi + 1, dtype=U32).view(F32), around((m / 252 - 0.5) / np.float64(C1)),
                           rng.uniform(-104.0, 89.0, 1 << 16).astype(F32), around(np.log(np.finfo(F32).max)),
                           np.r_[specials, specials | U32(0x80000000)].view(F32)])


def test_nv_expf_bits():
    """The module docstring's test_nv_expf_bits."""
    x = nv_expf_args()
    with np.errstate(invalid="ignore"):                                 # the NaN arguments
        t = emu.fma32(x, C1, F32(0.5))
    t = np.where(t > 0, np.minimum(t, F32(1)), F32(0))                  # cvt.sat, a NaN to +0
    k = np.floor(t * F32(252))                                          # nv_expf's k before its correction
    fix = emu.fma32(t, F32(252), -k) < 0
    assert np.array_equal(k - fix, np.floor(t.astype(np.float64) * 252))
    assert fix.any() and (t == 0).any() and (t == 1).any()
    want = emu.nv_expf_emu(x, lambda f: hc_probe("exp2", f))
    w = want.view(U32) & U32(0x7FFFFFFF)
    finite = np.isfinite(x)
    assert ((w > 0) & (w < 0x00800000)).any() and (finite & (w == 0)).any() and (finite & (w == 0x7F800000)).any()

    pso = basalt_gpu.pipeline(kernels.basalt_sampler_source() + NV_EXPF, "probe_nv_expf")
    out = basalt_gpu.run(pso, [x.tobytes(), 4 * x.size], U32(x.size).tobytes(), 2, -(-x.size // 256), 256, fill=0xFF)
    got = np.frombuffer(out[1], U32)
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got.view(F32)), nan)
    bad = np.flatnonzero((got != want.view(U32)) & ~nan)
    assert bad.size == 0, [f"{x[i]!r}: {got[i]:#010x} != {want.view(U32)[i]:#010x}" for i in bad[:8]]

    sub = (x >= F32(-103.97)) & (x <= F32(-87.34))
    ref, g = np.exp(x[sub].astype(np.float64)), got[sub]
    assert (g.view(F32)[ref >= 2.0 ** -149] > 0).all()
    # The bit distance to exp(x) correctly rounded to float32 (numpy's cast) counts 2^-149 steps here: at most one, so
    # within 1.5 steps of exp(x) itself. One step from the unrounded exp(x) needs an exp2 within about half an ulp of
    # 2^f near f = 0: fast::exp2 (invariant 5's ex2.approx.ftz) is within one ulp of the correctly rounded 2^f, and
    # with it nv_expf lands up to 1.22 steps off on [2^-127, 2^-126); the PTX ISA bounds ex2.approx.f32 by 2 ulp only.
    steps = np.abs(g.astype(np.int64) - ref.astype(F32).view(U32).astype(np.int64))
    assert steps.max() <= 1, (steps.max(), x[sub][steps.argmax()])
