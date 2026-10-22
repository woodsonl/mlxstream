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
  subnormals, and +0 and +inf from finite arguments.

swiglu_pair (kernels.cu:698-704, the shared expert's SwiGLU into saturated FP16; no basalt test launches it) runs at
B = 256 rows of 640 through basalt_prompt_gpu.run_rows (the rows from StepState), its halves starting as 0xA5A5, on
g, u = default_rng(22).normal(0, 3) (the distribution and seed of elementwise_parity.cpp:161-162's silu draws):
- ``test_swiglu_pair_emulation_bit_for_bit`` (SP-1): with row 0 opening with ``SP_EDGES``' 13 (g, u) pairs, every
  half equals ``basalt_emu.swiglu_pair_emu``'s, whose __expf(-g) is ``hc_probe("exp_neg", g)``; the twin stores the
  edge pairs' halves sm_120 does, and with every operation flushed it differs on the first four (a subnormal q or u,
  which a flushing port loses). The fixture holds FP16 subnormals, the saturated 65504, -0 and NaN halves.
- ``test_swiglu_pair_float64_reference`` (SP-2): on the draws without the edge pairs, against c = g / (1 + exp(-g)) u
  in float64, clamped to +-65504 as hf_sat clamps, with shared_expert_parity.cpp's SwiGLU rules as the bound (a
  no-reference choice): first its trap, the rival reading silu(u) g at least rel_l1 0.5 from c (``if (!(rel_wrong >
  0.5)) ++bad;``, :267); no non-finite output (:275, :282-285); every half c rounded down or up to fp16; and
  rel_l1(c, got) <= 1e-3 (rel_l1 :54-62, the fp16 intermediate's tolerance :3-5).
- ``test_swiglu_pair_rows`` (SP-3): a 1-row run's row 0 and a 3-row run's rows (row 0's data at index 2, StepState's
  position 2) equal their rows of the 256-row run bit for bit; rows at or past the count, and every row of a run with
  done = 1, keep 0xA5A5."""

from functools import partial

import numpy as np

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu
from tests.kernels.basalt_prompt_gpu import hc_probe, prompt_pipeline, run_rows

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


# ---- swiglu_pair (src/prefill/kernels.cu:698-704, the shared expert's SwiGLU) ---------------------------------------

B, F = 256, 640                                 # the bound (the perplexity gate's chunk) and the shared expert's width
EXP_NEG = partial(hc_probe, "exp_neg")          # this GPU's expf_ieee(sign_flip(x)): __expf(-x) for the twins
# (g, u, the half sm_120 stores) in row 0's first columns. A port that flushes a subnormal q or u stores 0x0000,
# 0x0000, 0x7fff and 0x0000 for the first four and the same halves for the rest.
SP_EDGES = ((1e-39, 1e33, 0x0008), (1e33, 1e-39, 0x0011), (1e-39, np.inf, 0x7BFF), (2.0 ** -126, 2.0 ** 120, 0x2000),
            (0.0, np.inf, 0x7FFF), (-1e-30, 1e-10, 0x8000), (300.0, 300.0, 0x7BFF), (-300.0, 300.0, 0x8000),
            (-90.0, 1.0, 0x8000), (90.0, 1.0, 0x55A0), (np.inf, 2.0, 0x7BFF), (-np.inf, 2.0, 0x7FFF),
            (np.nan, 1.0, 0x7FFF))
SP_BOUND = ("a no-reference choice (no basalt test launches swiglu_pair_kernel): shared_expert_parity.cpp's SwiGLU "
            "rules")


def sp_fixture(edges=True):
    """g, u float32 [B, F] = default_rng(22).normal(0, 3), with SP_EDGES in row 0's first columns unless not
    ``edges``."""
    g, u = np.random.default_rng(22).normal(0.0, 3.0, (2, B, F)).astype(F32)
    if edges:
        g[0, : len(SP_EDGES)], u[0, : len(SP_EDGES)] = np.array([e[:2] for e in SP_EDGES], F32).T
    return g, u


def swiglu_pair(g, u, rows, position=0, done=0):
    """basalt_prompt_swiglu_pair compiled at bound B (prompt_pipeline's row macros, basalt_prompt_moe_launch's widths,
    grid and threadgroup) on g and u [B, F], with ``rows``, ``position`` and ``done`` in StepState, into h16 [B, F]
    whose halves start as 0xA5A5. Returns h16's bits, uint16 [B, F]."""
    macros, grid, threads = kernels.basalt_prompt_moe_launch("basalt_prompt_swiglu_pair", B)
    pso = prompt_pipeline(kernels.basalt_prompt_moe_source(), "basalt_prompt_swiglu_pair", B, macros)
    out = run_rows(pso, [g.tobytes(), u.tobytes(), 2 * B * F], None, 0, grid, threads, rows, position, fill=0xA5,
                   done=done)
    return np.frombuffer(out[2], np.uint16).reshape(B, F)


def rel_l1(a, b):
    """shared_expert_parity.cpp:54-62: sum |a - b| / sum |a| in float64, the denominator floored at 1e-30."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.abs(a).sum()
    return np.abs(a - b).sum() / (m if m > 1e-30 else 1e-30)


def test_swiglu_pair_emulation_bit_for_bit():
    """SP-1 (the module docstring's)."""
    g, u = sp_fixture()
    want = emu.swiglu_pair_emu(g, u, EXP_NEG)
    n = len(SP_EDGES)
    assert want[0, :n].tolist() == [e[2] for e in SP_EDGES], [f"{h:#06x}" for h in want[0, :n]]
    a = want & 0x7FFF                                                   # the fixture's classes
    assert ((a > 0) & (a < 0x0400)).any() and (a == 0x7BFF).any() and (want == 0x8000).any() and (want == 0x7FFF).any()
    got = swiglu_pair(g, u, B)
    bad = np.argwhere(got != want)
    if len(bad):
        t, c = bad[0]
        raise AssertionError(f"{len(bad)} halves differ from swiglu_pair_emu; the first at ({t}, {c}): g {g[t, c]!r}, "
                             f"u {u[t, c]!r}, got {got[t, c]:#06x}, want {want[t, c]:#06x}")
    flushed = emu.swiglu_pair_emu(g, u, EXP_NEG, flush=True)
    assert (flushed[0, :4] != want[0, :4]).all(), [f"{h:#06x}" for h in flushed[0, :4]]


def test_swiglu_pair_float64_reference():
    """SP-2 (the module docstring's)."""
    g, u = sp_fixture(edges=False)
    g64, u64 = g.astype(np.float64), u.astype(np.float64)
    c = np.clip(g64 / (1.0 + np.exp(-g64)) * u64, -65504.0, 65504.0)        # the reference saturates as hf_sat does
    wrong = np.clip(u64 / (1.0 + np.exp(-u64)) * g64, -65504.0, 65504.0)    # the rival reading, silu(u) * g
    rel_wrong = rel_l1(c, wrong)
    print(f"SP-2: the rival reading is rel_l1 {rel_wrong:.3f} apart")
    assert rel_wrong > 0.5, f"{SP_BOUND}:267 if (!(rel_wrong > 0.5)) ++bad;: {rel_wrong}"
    h = swiglu_pair(g, u, B).view(np.float16)
    got = h.astype(np.float64)
    assert np.count_nonzero(~np.isfinite(got)) == 0, f"{SP_BOUND}:275, :282-285: a non-finite output"
    up, down = (np.nextafter(h, np.float16(s)).astype(np.float64) for s in (np.inf, -np.inf))
    faithful = (got == c) | ((got < c) & (up > c)) | ((got > c) & (down < c))       # c rounded down or up to fp16
    assert faithful.all(), f"{np.count_nonzero(~faithful)} halves are neither fp16 neighbour of the float64 value"
    rel = rel_l1(c, got)
    print(f"SP-2: rel_l1 {rel:.3e}")
    assert rel <= 1e-3, f"{SP_BOUND}, the fp16 intermediate's ~1e-3 (:3-5) as rel_l1 (:54-62): {rel:.3e}"


def test_swiglu_pair_rows():
    """SP-3 (the module docstring's)."""
    g, u = sp_fixture()
    full = swiglu_pair(g, u, B)
    one = swiglu_pair(g, u, 1)
    perm = [1, 2, 0]                                                    # row 0's data at index 2 of 3 rows
    g3, u3 = g.copy(), u.copy()
    g3[:3], u3[:3] = g[perm], u[perm]
    three = swiglu_pair(g3, u3, 3, position=2)
    assert np.array_equal(one[0], full[0]), "row 0 alone differs from row 0 of 256"
    assert np.array_equal(three[:3], full[perm]), "a row of the 3-row run differs from its row in 256"
    assert (one[1:] == 0xA5A5).all() and (three[3:] == 0xA5A5).all(), "a row at or past the count was written"
    assert (swiglu_pair(g, u, B, done=1) == 0xA5A5).all(), "a run with done = 1 stored"
