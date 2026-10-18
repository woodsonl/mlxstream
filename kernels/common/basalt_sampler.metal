// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/sampler.cu, include/basalt/kernels/sampler.hpp and
// include/basalt/core/coupled_draft.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// written in the rounding sequence nvcc compiles for them (docs/research/basalt-nvcc/sampler.md).
//
// basalt's sampler family on Metal. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal and
// basalt_libdevice.metal ahead of it (monolith/kernels.py's basalt_sampler_source()).
#pragma STDC FP_CONTRACT OFF

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
