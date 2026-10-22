// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:146-177, src/kernels/cuda/iq_kernels.cu:1880-1900
// and include/basalt/kernels/q8_1_finite.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// which adapt ggml-org/llama.cpp ggml/src/ggml-cuda/quantize.cu @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT;
// third_party/NOTICE).
// Moved from kernels/common/basalt_q8_1.metal.
//
// IEEE binary32 add and multiply where a subnormal can occur: this GPU flushes subnormal operands and results
// (tests/kernels/test_basalt_platform.py), so ieee_add and ieee_mul rebuild IEEE's result. Needs only the prelude.
#pragma STDC FP_CONTRACT OFF

// ---- IEEE operations where a subnormal can occur ---------------------------------------------------------------

constant constexpr uint IEEE_SMALL = 27u << 23;   // 2^-100: below it an operand or result may be subnormal

// x * 2^64, exactly, for |x| < 2^-100 (a subnormal's significand rebuilt as an integer: the hardware would flush it)
static inline float ieee_up64(float x) {
    const uint b = as_type<uint>(x);
    if ((b & 0x7f800000u) != 0u) return x * 0x1p64f;
    return copysign(float(b & 0x007fffffu) * 0x1p-85f, x);
}

// s * 2^-64 rounded as IEEE would: exact when the result is normal; a subnormal result is exact too (see ieee_add)
static inline float ieee_down64(float s) {
    if (fabs(s) >= 0x1p-62f || s == 0.0f) return s * 0x1p-64f;
    return as_type<float>((as_type<uint>(s) & 0x80000000u) | uint(fabs(s) * 0x1p85f));
}

// a + b with IEEE subnormals: below 2^-100 both operands are scaled by 2^64 (exactly), added, and scaled back; a sum
// whose exact value is subnormal is representable (Hauser), so the scaled sum is exact and the integer rebuild too.
static inline float ieee_add(float a, float b) {
    if ((as_type<uint>(a) & 0x7fffffffu) >= IEEE_SMALL || (as_type<uint>(b) & 0x7fffffffu) >= IEEE_SMALL) return a + b;
    return ieee_down64(ieee_up64(a) + ieee_up64(b));
}

// a * b with IEEE subnormals (CUDA's mul.rn.f32 in an IEEE-built kernel). Two operands with nonzero exponent fields
// (normal, infinite or NaN) whose product is at least 2^-125, infinite or NaN take the hardware's correctly rounded
// product. An infinity or NaN against a zero or a subnormal (which the hardware flushes, making a subnormal times inf
// 0 * inf) gives IEEE's result from the bits: NaN (CUDA's canonical 0x7fffffff) for a NaN or a zero operand, else the
// signed infinity. Otherwise (a subnormal operand, or a product the hardware may flush) the product is rebuilt from
// the exact integer product m of the significands: |a * b| = m * 2^e units of 2^-149, rounded to nearest even on the
// grid of its binade (k = 0 is the subnormal grid and binade -126), so that the float's bits are (k << 23) + r for r
// units of 2^(k - 149).
static inline float ieee_mul(float a, float b) {
    const float p = a * b;
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b);
    uint ea = (ab >> 23) & 0xffu, eb = (bb >> 23) & 0xffu, ma = ab & 0x7fffffu, mb = bb & 0x7fffffu;
    if (ea != 0u && eb != 0u && !(fabs(p) < 0x1p-125f)) return p;      // p infinite or NaN included
    const uint sign = (ab ^ bb) & 0x80000000u;
    const bool za = (ea | ma) == 0u, zb = (eb | mb) == 0u;             // zeros by their bits (a compare may flush)
    if (ea == 0xffu || eb == 0xffu)                                     // the other operand is a zero or a subnormal
        return as_type<float>((ea == 0xffu && ma != 0u) || (eb == 0xffu && mb != 0u) || za || zb
                              ? 0x7fffffffu : sign | 0x7f800000u);
    if (za || zb) return as_type<float>(sign);
    if (ea == 0u) ea = 1u; else ma |= 0x800000u;
    if (eb == 0u) eb = 1u; else mb |= 0x800000u;
    const ulong m = ulong(ma) * ulong(mb);
    const int e = int(ea + eb) - 151;                                   // |a * b| = m * 2^e units of 2^-149
    const int lead = 63 - int(clz(m));
    const int k = max(0, lead + e - 23);                                // the grid: 24 significant bits, or 2^-149
    const int s = k - e;                                                // m * 2^e / 2^k = m >> s
    ulong r;
    if (s <= 0) {
        r = m << uint(-s);
    } else if (s > 63) {
        r = 0;
    } else {
        r = m >> uint(s);
        const ulong rem = m & ((1ul << uint(s)) - 1ul), tie = 1ul << uint(s - 1);
        if (rem > tie || (rem == tie && (r & 1ul) != 0ul)) r += 1ul;
    }
    return as_type<float>(sign | (uint(k) << 23) + uint(r));            // r == 2^24 carries into the exponent
}

// ---- __expf and rsqrtf as an IEEE-built kernel computes them --------------------------------------------------------
//
// nvcc compiles both without .ftz (docs/research/basalt-nvcc/dec_gdn.md, Notation: ex2n, rsqn). __expf(x) is an IEEE
// multiply by log2(e) (0x3FB8AA3B), then MUFU.EX2 with an argument below -126 halved and the result squared by an IEEE
// multiply, so that subnormal results survive; rsqrtf(x) is MUFU.RSQ with a zero or subnormal argument scaled by 2^24
// and the result by 2^12. fast::exp2 and fast::rsqrt stand in for MUFU.EX2 and MUFU.RSQ; the product, the square and
// the 2^24 and 2^12 scalings go through ieee_mul (the halving is exact).

// -x by its bits: the sign flipped, every other bit kept (unary minus on a subnormal is not measured on this GPU)
static inline float sign_flip(float x) { return as_type<float>(as_type<uint>(x) ^ 0x80000000u); }

// __expf(x). The halving test is FSETP.GEU's complement, e < -126 ordered: a subnormal e reads as zero here and is not
// below -126, nor is NaN. __expf(-x) is expf_ieee(sign_flip(x)): nvcc folds the negation into the constant
// (0xBFB8AA3B), and a correctly rounded product, ieee_mul's included, changes only its sign with an operand's.
static inline float expf_ieee(float x) {
    float e = ieee_mul(x, as_type<float>(0x3FB8AA3Bu));
    const bool s = e < -126.0f;
    if (s) e = e * 0.5f;
    float r = fast::exp2(e);
    if (s) r = ieee_mul(r, r);
    return r;
}

// rsqrtf(x). small is FSETP.GEU's complement, |x| < 2^-126 ordered: a zero or a subnormal, read from the bits. A
// scaled result is at least 2^51 in magnitude, infinite or NaN, so the postscale's ieee_mul is the hardware product.
static inline float rsqrt_ieee(float x) {
    const bool small = (as_type<uint>(x) & 0x7fffffffu) < 0x00800000u;
    const float r = fast::rsqrt(small ? ieee_mul(x, 0x1p24f) : x);
    return small ? ieee_mul(r, 4096.0f) : r;
}
