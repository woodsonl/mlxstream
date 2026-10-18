// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/sampler.cu, include/basalt/kernels/sampler.hpp and
// include/basalt/core/coupled_draft.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// written in the rounding sequence nvcc compiles for them (docs/research/basalt-nvcc/sampler.md).
//
// basalt's sampler family on Metal. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal and
// basalt_libdevice.metal ahead of it (monolith/kernels.py's basalt_sampler_source()).
#pragma STDC FP_CONTRACT OFF

// ---- Keys and compares on the bits ----------------------------------------------------------------------------------
//
// CUDA's setp and max.f32 without .ftz order subnormals exactly and compare -0 equal to +0; this GPU's <, > and ==
// read a subnormal as zero (tests/kernels/test_basalt_platform.py), so every compare that orders or selects a float
// here runs on its bits (tests/kernels/test_basalt_sampler_helpers.py, H.6).

// The order-preserving key of a non-NaN float's bits: a < b exactly when okey(a) < okey(b), and a == b exactly when
// the keys are equal (-0 and +0 share one key).
static inline uint okey(uint b) {
    b = (b == 0x80000000u) ? 0u : b;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static inline bool fnan(uint b) { return (b & 0x7fffffffu) > 0x7f800000u; }
// CUDA's ordered > and < (false when either operand is NaN)
static inline bool f_gt(uint a, uint b) { return !fnan(a) && !fnan(b) && okey(a) > okey(b); }
static inline bool f_lt(uint a, uint b) { return !fnan(a) && !fnan(b) && okey(a) < okey(b); }

// CUDA's fmaxf (max.f32 without .ftz): the other operand when one is NaN, 0x7fffffff (NVIDIA's canonical NaN) when
// both are, else the larger by okey; on -0 against +0 the first operand (no output depends on that zero's sign).
static inline uint fmaxf_bits(uint a, uint b) {
    if (fnan(a)) return fnan(b) ? 0x7fffffffu : b;
    if (fnan(b)) return a;
    return okey(b) > okey(a) ? b : a;
}

// ---- IEEE division and fma where a subnormal can occur -------------------------------------------------------------
//
// CUDA's div.rn.f32 and fma.rn.f32 in an IEEE-built kernel keep subnormal operands and results; this GPU flushes them,
// an fma on its exact value, before rounding (tests/kernels/test_basalt_platform.py). As ieee_mul does,
// each helper takes the hardware's correctly rounded result when every operand has a nonzero exponent field (normal,
// infinite or NaN) and the result is at least 2^-125, infinite or NaN: the hardware rounds such an exact value once,
// a product below the normal range or beyond FLT_MAX in the fma included (tests/kernels/test_basalt_sampler_helpers.py,
// H.3). Every other case (a zero or subnormal operand, a result the hardware may have flushed) is rounded from the
// software binary64. A NaN from the software is 0x7fffffff, NVIDIA's canonical NaN; the hardware's NaN keeps its own
// bits.

// a / b, and ieee_div(1, x) for rcp.rn.f32: the binary64 quotient, rounded to nearest even and then to binary32. That
// double rounding is the correctly rounded quotient: on the normal grid since 53 >= 2 * 24 + 2 (Figueroa), and on the
// subnormal grid because a quotient of floats off a midpoint lies more than 2^-48 of itself from it, beyond binary64's
// rounding, while one on a midpoint stays on it (H.3).
static inline float ieee_div(float a, float b) {
    const float q = a / b;
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b);
    if ((ab & 0x7f800000u) != 0u && (bb & 0x7f800000u) != 0u && !(fabs(q) < 0x1p-125f)) return q;
    return as_type<float>(f32_from_f64(f64_div(f64_from_f32(ab), f64_from_f32(bb))));
}

// fma(a, b, c), the penalty's fma(float(count), freq, present) (sampler.cu:75): the exact a * b + c rounded to odd in
// binary64 and then to nearest even in binary32, which is the correctly rounded fma (53 >= 24 + 2: Boldo and
// Melquiond). f32_from_f64(f64_fma(...)) would round twice and can land on the wrong neighbour of a binary32 midpoint.
static inline float ieee_fma_count(float a, float b, float c) {
    const float r = fma(a, b, c);
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b), cb = as_type<uint>(c);
    if ((ab & 0x7f800000u) != 0u && (bb & 0x7f800000u) != 0u && (cb & 0x7f800000u) != 0u && !(fabs(r) < 0x1p-125f))
        return r;
    return as_type<float>(f32_from_f64(f64_fma_odd(f64_from_f32(ab), f64_from_f32(bb), f64_from_f32(cb))));
}

// ---- The approximate exponential ------------------------------------------------------------------------------------

// CUDA's __expf(x) in an IEEE-built kernel (sampler.cu:311): ex2.approx.f32 of the IEEE product x * log2(e)
// (mul.f32 by 0x3fb8aa3b). Invariant 5 maps it to fast::exp2 of that product, without ex2.approx's branch for
// subnormal results; the product is ieee_mul's, since x = l[i] - m can be subnormal.
static inline float expf_approx(float x) { return fast::exp2(ieee_mul(x, as_type<float>(0x3fb8aa3bu))); }

// ---- Philox (sampler.cu:37-59) --------------------------------------------------------------------------------------
//
// basalt's own variant, not Random123's Philox4x32-10: the multipliers are the Weyl constants, the round key is
// (i, 0), and the seed fills counter words c2 and c3. Integers only, so it ports bit for bit. The draw is
// (c0 >> 8) * 2^-24, exact: a multiple of 2^-24 in [0, 1 - 2^-24].
static inline float philox_uniform(ulong seed, ulong counter) {
    uint c0 = uint(counter), c1 = uint(counter >> 32), c2 = uint(seed), c3 = uint(seed >> 32);
    for (uint i = 0u; i < 10u; ++i) {                                   // bounded: ten rounds
        const uint hi0 = mulhi(0x9E3779B9u, c0), hi1 = mulhi(0xBB67AE85u, c2);
        const uint lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return float(c0 >> 8) * 0x1p-24f;
}
