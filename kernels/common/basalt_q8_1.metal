// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:146-177, src/kernels/cuda/iq_kernels.cu:1880-1900
// and include/basalt/kernels/q8_1_finite.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// which adapt ggml-org/llama.cpp ggml/src/ggml-cuda/quantize.cu @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT;
// third_party/NOTICE). block_sum is adapted from basalt src/prefill/kernels.cu:45-58 (basalt's own code)
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
//
// basalt's q8_1 activation quantizers. A q8_1 block is 36 bytes: the fp16 scale d (bytes 0-1), the fp16 sum s of the
// block's 32 raw values (bytes 2-3), 32 int8 codes (bytes 4-35); column j's blocks follow column j - 1's, so a vector
// of ncols * n_in values (n_in % 32 == 0) is ncols columns. One thread per value, 256 per threadgroup, SIMD-group =
// block (whole SIMD-groups past the end return). Needs basalt_fp.metal and basalt_ieee.metal ahead of it. A NaN in
// d or s is stored as sm_120's fp16 0x7fff (cuda_nan_f16; docs/research/basalt-nvcc/nan.md); a NaN quotient's code
// is 0, as on sm_120.
//
// - native_quantize_q8_1: the sequence nvcc compiles for native_mmvq.cu (--use_fast_math; matvec.md section 1):
//   flush-to-zero abs, max and add; d = fl(amax * fl(1/127)) (ptxas replaces the approximate division by 127 with
//   this multiply); the codes' xi / d as MUFU.RCP then FMUL.FTZ, here the correctly rounded reciprocal and the same
//   multiply; roundf; the +-127 clamp with NaN to 0.
// - native_swiglu_quantize_q8_1: the shared expert's SwiGLU value (native_mmvq.cu:170; matvec.md section 2), then the
//   native sequence. __expf is MUFU.EX2 of fl(g * -log2 e), here fast::exp2 of the same product.
// - quantize_q8_1: quantize_q8_1_kernel (iq_kernels.cu, nvcc's IEEE defaults; matvec.md section 3): correctly rounded
//   divisions and no flushing. This GPU flushes subnormal inputs and results of add and division, and its fmax
//   compares a subnormal as zero, returning the second operand of a tie rather than the larger
//   (tests/kernels/test_basalt_platform.py), so wherever a subnormal can occur the IEEE result is rebuilt from integer
//   operations: amax as the largest bit pattern, the block sum and the codes on operands scaled by 2^64, d = amax / 127
//   on the subnormal grid by integer division. Above 2^-100 every one of these is the plain operation. The IEEE
//   helpers (basalt_ieee.metal's ieee_add and ieee_mul; ieee_div127, ieee_rcp, ieee_fma, ieee_div, warp_sum_ieee
//   and block_sum here) serve every IEEE-built port kernel.
#pragma STDC FP_CONTRACT OFF

struct Q81Params { uint n; uint pad0; uint pad1; uint pad2; };   // n: values (ncols * n_in), a multiple of 32

constant constexpr float Q8_1_HALF_MAX = 65504.0f;

static inline float q8_1_finite(float v) { return fabs(v) > Q8_1_HALF_MAX ? copysign(Q8_1_HALF_MAX, v) : v; }

// (int8_t) of the clamped roundf: q > 127 -> 127, q < -127 -> -127, NaN -> 0 (F2I.TRUNC.NTZ / cvt.rzi of NaN).
static inline char q8_1_code(float q) {
    return isnan(q) ? char(0) : char(int(q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q));
}

// q8_1_ds (q8_1_finite.hpp:29-31): make_half2(__float2half(...)), F2FP.F16.F32.PACK_AB on sm_120, whose NaN is 0x7fff
static inline void q8_1_put(device uchar* y, uint i, char code, float d, float sum) {
    y[36u * (i / 32u) + 4u + i % 32u] = as_type<uchar>(code);
    if (i % 32u == 0u) {
        *(device uint*) (y + 36u * (i / 32u)) = uint(cuda_nan_f16(f16_bits(q8_1_finite(d))))
                                                | (uint(cuda_nan_f16(f16_bits(q8_1_finite(sum)))) << 16);
    }
}

// The native sequence for value xi of block i / 32 (lane i % 32): the device function every native call site
// shares (native_quantize_q8_1, the SwiGLU quantizer, and the fused epilogues that reproduce it).
static inline void q8_1_store_native(float xi, device uchar* y, uint i) {
    const float amax = warp_max_ftz(abs_ftz(xi));
    const float sum = warp_sum_ftz(xi);
    const float d = q8_1_finite(mul_ftz(amax, as_type<float>(0x3C010204u)));   // fl(1/127)
    const char code = amax == 0.0f ? char(0) : q8_1_code(round(mul_ftz(xi, rcp_ftz(d))));
    q8_1_put(y, i, code, d, sum);
}

kernel void native_quantize_q8_1(device const float* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                                 constant Q81Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;                    // n is a multiple of 32: only whole SIMD-groups return
    q8_1_store_native(x[i], y, i);
}

kernel void native_swiglu_quantize_q8_1(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                                        device uchar* y [[buffer(2)]], constant Q81Params& p [[buffer(3)]],
                                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float g = gate[i];
    const float e = ftz(fast::exp2(mul_ftz(g, as_type<float>(0xBFB8AA3Bu))));   // __expf(-g): ex2.approx(g * -log2 e)
    const float s = mul_ftz(g, rcp_ftz(add_ftz(1.0f, e)));                      // __fdividef(g, __fadd_rn(1, e))
    q8_1_store_native(mul_ftz(s, up[i]), y, i);                                 // __fmul_rn(s, up)
}

// ---- IEEE operations where a subnormal can occur (with basalt_ieee.metal's) ----------------------------------

// amax / 127 rounded as IEEE does, for amax >= 0 or NaN, decided on its bits (a compare may flush): the plain division
// from 2^-100 (and for 0 and NaN); from 2^-119, where the quotient is normal, on amax scaled by 2^64; below, on the
// subnormal grid by integer division (127 is odd: no ties).
static inline float ieee_div127(float amax) {
    const uint b = as_type<uint>(amax);
    if (b >= IEEE_SMALL || b == 0u) return amax / 127.0f;
    if (b >= (8u << 23)) return (ieee_up64(amax) / 127.0f) * 0x1p-64f;
    const uint m = (b & 0x7f800000u) == 0u ? b : ((b & 0x007fffffu) | 0x00800000u) << ((b >> 23) - 1u);
    return as_type<float>(m / 127u + (m % 127u >= 64u ? 1u : 0u));
}

// 1 / s rounded as IEEE does (rcp.rn.f32 without .ftz) for |s| <= 2^126 (the quantizer's s is at most FLT_MAX / 127),
// an infinity or NaN: below 2^-100 (zero included) on s scaled by 2^64, rcp.rn.f32's own slow path: the correctly
// rounded reciprocal of a normal value, scaled back exactly or overflowing where 1 / s does.
static inline float ieee_rcp(float s) {
    if ((as_type<uint>(s) & 0x7fffffffu) >= IEEE_SMALL) return 1.0f / s;
    return (1.0f / ieee_up64(s)) * 0x1p64f;
}

// fma(a, n, c) rounded as IEEE does (fma.rn.f32 without .ftz) for an integer n, |n| < 2^24, decided on bits:
// - a and c each zero or at least 2^-100: a * n + c is a multiple of 2^-123, zero or normal: the hardware's fma.
// - a nonzero below 2^-100: |a * n| < 2^-76. With |c| < 2^62, on the operands scaled by 2^64 (exactly): a * n + c is
//   a multiple of 2^-149, so a subnormal result is exact in the scaled form and scales back exactly (ieee_down64), and
//   a normal one rounds as its scaled form does. With a larger c (or an infinite or NaN one) the sum rounds to c
//   whether or not the hardware flushes a: its fma.
// - c nonzero below 2^-100, a zero or at least 2^-100: a finite a * n of zero leaves c. Otherwise |a * n| >= 2^-100
//   (or a is infinite or NaN): the sum is at least 2^-124, and a subnormal c changes its rounding only by breaking a
//   tie, as any value of its sign below 2^-123 does: the hardware's fma with c at +-2^-126.
static inline float ieee_fma(float a, float n, float c) {
    const uint ua = as_type<uint>(a) & 0x7fffffffu, uc = as_type<uint>(c) & 0x7fffffffu;
    const bool a_ok = ua == 0u || ua >= IEEE_SMALL, c_ok = uc == 0u || uc >= IEEE_SMALL;
    if (a_ok && c_ok) return fma(a, n, c);
    if (!a_ok) return uc < (189u << 23) ? ieee_down64(fma(ieee_up64(a), n, ieee_up64(c))) : fma(a, n, c);
    if ((ua == 0u || n == 0.0f) && ua < 0x7f800000u) return c;
    return fma(a, n, uc < 0x00800000u ? copysign(0x1p-126f, c) : c);
}

// a / b rounded as IEEE does (div.rn.f32 without .ftz) for b >= 1, infinite or NaN (the SwiGLU's 1 + e): the
// hardware's quotient when a or b is infinite or NaN, or when |a| >= 2^-100 and the quotient is at least 2^-125
// (normal operands and result: Metal's division is IEEE's, tests/kernels/test_basalt_platform.py); a zero a gives its
// signed zero. Otherwise the quotient is below 2^-100 and is rebuilt from the integer quotient of the significands,
// rounded to nearest even on the grid of its binade (k = 0 the subnormal grid), as ieee_mul does.
static inline float ieee_div(float a, float b) {
    const float q = a / b;
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b), ua = ab & 0x7fffffffu;
    if (ua >= 0x7f800000u || (bb & 0x7f800000u) == 0x7f800000u) return q;
    if (ua >= IEEE_SMALL && !(fabs(q) < 0x1p-125f)) return q;
    const uint sign = (ab ^ bb) & 0x80000000u;
    if (ua == 0u) return as_type<float>(sign);
    uint ea = ua >> 23, ma = ua & 0x7fffffu;
    if (ea == 0u) ea = 1u; else ma |= 0x800000u;
    const int z = int(clz(ma)) - 8;                                    // normalize ma to [2^23, 2^24)
    ma <<= uint(z);
    const uint mb = (bb & 0x7fffffu) | 0x800000u;
    const int s = int(ea) - z - int((bb >> 23) & 0xffu) + 149;        // |a / b| = ma / mb * 2^s units of 2^-149
    const int k = max(0, s - (ma < mb ? 1 : 0) - 23);                 // the grid: 24 significant bits, or 2^-149
    ulong num = ulong(ma), den = ulong(mb);
    if (s - k >= 0) num <<= uint(s - k);
    else if (k - s < 40) den <<= uint(k - s);
    else num = 0ul;
    ulong r = num / den;
    const ulong rem = num - r * den;
    if (2ul * rem > den || (2ul * rem == den && (r & 1ul) != 0ul)) r += 1ul;
    return as_type<float>(sign | (uint(k) << 23) + uint(r));            // r == 2^24 carries into the exponent
}

// basalt's warp_sum with IEEE adds (the XOR butterfly the port shares; lane 0 is a shift-down tree's lane 0 too)
static inline float warp_sum_ieee(float x) {
    for (ushort o = 16; o > 0; o >>= 1) x = ieee_add(x, shfl_xor(x, o));
    return x;
}

// basalt's block_sum (src/prefill/kernels.cu:45-58): the sum of v over a threadgroup of ntg <= 1024 threads (1-D),
// returned to every thread; warp_sum_ieee for warp_sum and a threadgroup barrier for each __syncthreads. Each
// SIMD-group's lane 0 stores its sum in sh[w]; SIMD-group 0 sums sh[0 .. nw) and zeros, nw = (ntg + 31) >> 5 (:52).
// At ntg = 256 the eight partials fold as ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)). Every thread of the
// threadgroup calls it; callers pass their dispatch width and declare threadgroup float sh[32].
static inline float block_sum(float v, threadgroup float* sh, uint tid, uint ntg) {
    const uint lane = tid & 31u, w = tid >> 5;
    v = warp_sum_ieee(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sh[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nw = (ntg + 31u) >> 5;
    float t = tid < nw ? sh[tid] : 0.0f;
    if (w == 0u) t = warp_sum_ieee(t);
    if (tid == 0u) sh[0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}

static inline void q8_1_store_ieee(float xi, device uchar* y, uint i) {
    int key = isnan(xi) ? -1 : int(as_type<uint>(fabs(xi)));        // fmaxf drops a NaN operand
    float sum = xi;
    for (ushort o = 16; o > 0; o >>= 1) {
        key = max(key, simd_shuffle_xor(key, o));
        sum = ieee_add(sum, shfl_xor(sum, o));
    }
    const float amax = key < 0 ? as_type<float>(0x7fffffffu) : as_type<float>(uint(key));
    const bool small = uint(key) < IEEE_SMALL;                        // NaN (key -1) is not small
    const float d = q8_1_finite(ieee_div127(amax));
    char code = char(0);
    if (key != 0) code = q8_1_code(round(small ? ieee_up64(xi) / ieee_up64(d) : xi / d));
    q8_1_put(y, i, code, d, sum);
}

kernel void quantize_q8_1(device const float* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                          constant Q81Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    q8_1_store_ieee(x[i], y, i);
}
