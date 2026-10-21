// IEEE binary32 operations built on the software binary64, written for this port (the repository's own code; no
// upstream copy).
//
// CUDA's binary32 arithmetic in an IEEE-built kernel keeps subnormal operands and results; this GPU flushes them
// (tests/kernels/test_basalt_platform.py). Where basalt_ieee.metal's integer rebuilds do not reach, the exact result
// comes from basalt_f64.metal's binary64 rounded to odd, then to binary32. Needs basalt_ieee.metal and basalt_f64.metal
// ahead of it (monolith/kernels.py's basalt_gdn_source()).
#pragma STDC FP_CONTRACT OFF

// fma(a, b, c) with IEEE subnormals: CUDA's fma.rn.f32 in an IEEE-built kernel. The hardware's fused fma rounds the
// exact a * b + c once, a product below the normal range included (basalt_sampler.metal's ieee_fma_count), but reads a
// subnormal operand as zero and flushes an exact result below 2^-126, even one IEEE rounds up to 2^-126. So r is
// IEEE's when each operand is zero or at least 2^-100 by its bits (IEEE_SMALL; inf and NaN pass) and r is at least
// 2^-100, infinite or NaN (a product below the normal range cannot move such an r: half an ulp of any |c| >= 2^-100
// is at least 2^-124). A zero r passes only when a or b is zero by its bits: the exact result is then c or a signed
// zero, which the hardware gives (tests/kernels/test_basalt_gdn_helpers.py, H.2); from two nonzero factors a zero r
// may be a flushed subnormal. Every other case is the exact a * b + c rounded to odd in binary64 and then to nearest
// even in binary32, the correctly rounded fma (53 >= 24 + 2: Boldo and Melquiond); a NaN from it is 0x7fffffff.
static inline float fma_ieee(float a, float b, float c) {
    const float r = fma(a, b, c);
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b), cb = as_type<uint>(c);
    const uint ma = ab & 0x7fffffffu, mb = bb & 0x7fffffffu, mc = cb & 0x7fffffffu;
    if ((ma == 0u || ma >= IEEE_SMALL) && (mb == 0u || mb >= IEEE_SMALL) && (mc == 0u || mc >= IEEE_SMALL)
        && (!(fabs(r) < 0x1p-100f) || ma == 0u || mb == 0u))
        return r;
    return as_type<float>(f32_from_f64(f64_fma_odd(f64_from_f32(ab), f64_from_f32(bb), f64_from_f32(cb))));
}
