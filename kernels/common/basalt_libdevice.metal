// Transcribed from NVIDIA libdevice's __nv_exp and __nv_logf as nvcc 13.0 (V13.0.48, -arch=sm_120 -lineinfo -O3
// -DNDEBUG) inlined them into basalt's sampler.cu (jesdga95/basalt src/kernels/cuda/sampler.cu
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36; third_party/NOTICE); PTX listing and transcription:
// docs/research/basalt-nvcc/sampler.md sections 2.5-2.6. __nv_log1pf as the same nvcc inlined it into
// src/kernels/cuda/dec_gdn_rec.cu and src/kernels/cuda/verify_kernels.cu @ the same commit; SASS listing and
// transcription: docs/research/basalt-nvcc/dec_gdn.md section 6a.
//
// CUDA's binary64 exp and binary32 logf and log1pf, which are not correctly rounded, so no Metal function gives their
// bits: the sampled tail's exp((double) x - (double) max) and logf(min_p), and the GDN gate's softplus, operation for
// operation. Needs basalt_ieee.metal and basalt_f64.metal ahead of it.
//
// - nv_exp (binary64 bits in and out): each fma.rn.f64, add.rn.f64 and mul.f64 of the PTX is the software binary64's
//   operation, each integer step the same operation on 32-bit words. A NaN argument gives F64_NAN.
// - nv_logf: the PTX's binary32 operations, fma() at its twelve fma.rn.f32 and a statement for each other one.
//   A subnormal argument's 2^23 pre-scale goes through ieee_mul (the hardware would flush it) and every compare is on
//   bits, so no subnormal reaches the hardware's arithmetic or compares. A NaN result (a NaN or negative argument)
//   has the hardware fma's NaN bits [H: CUDA's FFMA gives 0x7fffffff]; the tail passes only min_p > 0.
// - log1pf_nv: the SASS's binary32 operations, fma() at its twelve FFMAs and a statement for each other one, with
//   three changes of form that change no bit (at log1pf_nv below). It calls only basalt_ieee.metal's ieee_mul.
#pragma STDC FP_CONTRACT OFF

// exp(double): t = x / ln 2 + 1.5 * 2^52 holds i = round(x / ln 2) in its low word, j is i as a double, r = x - j ln 2
// in two fmas (ln 2 split hi and lo), p = e^r in eleven Horner fmas, and e^x = p * 2^i through the exponent field. For
// |x| >= 708.4 the result is 0 (x < 0, ordered) or x + inf (+inf, NaN), except that below 745 the scale is split as
// 2^(i - h) * 2^h, h = i / 2, so one mul.f64 rounds a subnormal or overflowing result once. The range tests read the
// high word of x as a float (abs.f32, setp.lt.f32 and setp.geu.f32): on clear-sign words that is the unsigned order,
// NaN and infinite words failing < both times, so they compare the word as a uint.
static inline ulong nv_exp(ulong x) {
    const ulong t = f64_fma(x, 0x3FF71547652B82FEul, 0x4338000000000000ul);
    const int i = as_type<int>(lo(t));
    const ulong j = f64_add(t, 0xC338000000000000ul);
    ulong r = f64_fma(j, 0xBFE62E42FEFA39EFul, x);
    r = f64_fma(j, 0xBC7ABC9E3B39803Ful, r);
    ulong p = f64_fma(r, 0x3E5ADE1569CE2BDFul, 0x3E928AF3FCA213EAul);
    p = f64_fma(p, r, 0x3EC71DEE62401315ul);
    p = f64_fma(p, r, 0x3EFA01997C89EB71ul);
    p = f64_fma(p, r, 0x3F2A01A014761F65ul);
    p = f64_fma(p, r, 0x3F56C16C1852B7AFul);
    p = f64_fma(p, r, 0x3F81111111122322ul);
    p = f64_fma(p, r, 0x3FA55555555502A1ul);
    p = f64_fma(p, r, 0x3FC5555555555511ul);
    p = f64_fma(p, r, 0x3FE000000000000Bul);
    p = f64_fma(p, r, 0x3FF0000000000000ul);
    p = f64_fma(p, r, 0x3FF0000000000000ul);
    ulong res = bits(lo(p), hi(p) + (uint(i) << 20));                       // shl.b32, add.s32
    const uint hx = hi(x) & 0x7fffffffu;
    if (!(hx < 0x4086232Bu)) {                                              // |x| >= 708.396...
        res = f64_lt(x, 0ul) ? 0ul : f64_add(x, F64_INF);                   // setp.lt.f64, add.f64 x + inf, selp
        if (hx < 0x40874800u) {                                             // |x| < 745
            const int h = i / 2;                                            // shr.u32 31, add.s32, shr.s32 1
            res = f64_mul(bits(0u, (uint(i - h) << 20) + 0x3FF00000u), bits(lo(p), hi(p) + (uint(h) << 20)));
        }
    }
    return res;
}

// logf(float): an x below 2^-126 (setp.lt.f32: subnormals, zeros and negatives) is pre-scaled by 2^23 and fe starts
// at -23; subtracting the exponent field e of xs - 0x3F2AAAAB (integers) moves xs to m in [2/3, 4/3), and fe gains
// e * 2^-23; f = m - 1 is exact, and log(x) = fe * ln 2 + f + f^2 * q(f), q in eight Horner fmas. Bits above
// 0x7f7fffff (+inf, NaN, negatives) give fma(xs, inf, inf), +inf or NaN, and a zero xs gives -inf.
static inline float nv_logf(float x) {
    const uint b = as_type<uint>(x);
    const bool small = (b & 0x7fffffffu) <= 0x7f800000u && as_type<int>(b) < 0x00800000;   // x < 2^-126, ordered
    const float xs = small ? ieee_mul(x, as_type<float>(0x4B000000u)) : x;                 // mul.f32 by 2^23, selp
    float fe = small ? as_type<float>(0xC1B80000u) : 0.0f;
    const uint ib = as_type<uint>(xs);
    const uint e = (ib + 0xC0D55555u) & 0xFF800000u;                        // add.s32 -1059760811, and.b32 -8388608
    const float m = as_type<float>(ib - e);
    fe = fma(float(as_type<int>(e)), as_type<float>(0x34000000u), fe);      // cvt.rn.f32.s32 (exact), times 2^-23
    const float f = m + as_type<float>(0xBF800000u);                        // add.f32 m, -1
    float q = fma(f, as_type<float>(0xBE055027u), as_type<float>(0x3E1039F6u));
    q = fma(q, f, as_type<float>(0xBDF8CDCCu));
    q = fma(q, f, as_type<float>(0x3E0F2955u));
    q = fma(q, f, as_type<float>(0xBE2AD8B9u));
    q = fma(q, f, as_type<float>(0x3E4CED0Bu));
    q = fma(q, f, as_type<float>(0xBE7FFF22u));
    q = fma(q, f, as_type<float>(0x3EAAAA78u));
    q = fma(q, f, as_type<float>(0xBF000000u));
    float s = f * q;                                                        // mul.f32
    s = fma(s, f, f);
    float r = fma(fe, as_type<float>(0x3F317218u), s);                      // + fe * ln 2
    if (ib > 0x7F7FFFFFu) r = fma(xs, as_type<float>(0x7F800000u), as_type<float>(0x7F800000u));   // setp.gt.u32
    if ((ib & 0x7fffffffu) == 0u) r = as_type<float>(0xFF800000u);          // setp.eq.f32 xs, 0: xs is never subnormal
    return r;
}

// log1pf(float), as the GDN gate's softplus calls it (dec_gdn_rec.cu:111, verify_kernels.cu:154 and :209): t = x + 1
// rounded toward zero (FADD.RZ); k = floor(log2(t / 0.75)) << 23, from the bits of t less those of 0.75, scales x to
// m = x 2^-k + (2^-k - 1) (2^-k for 2^-(k >> 23)), in [-1/4, 1/2) for x > -1; log1p(x) = m + m (m q(m)) + k ln 2, q in
// eight Horner fmas. Bits at or above 0x7f800000 (+inf, NaN, the sign set) enter the special block: +inf, NaN and
// x < -1 give fma(x, inf, inf) (+inf, or NaN with this GPU's bits: a store a NaN can reach goes through cuda_nan), and
// -0 gives -0. Three changes of form, none of a bit (dec_gdn.md section 6a):
// (a) a zero or subnormal x returns x: the sequence returns it (t is 1 or its predecessor, k = 0, m = x, and every
//     later step keeps x), where this GPU's m + 0 would flush it;
// (b) FADD.RZ is the round-to-nearest sum stepped one ulp toward zero (bits - 1) when the sum's exact error, Fast2Sum's
//     of the operand larger in magnitude and the other, has the opposite sign; for a normal x every value of the
//     Fast2Sum is zero or normal (the error is x itself or a multiple of 2^-48), so no flush touches it;
// (c) FMUL p, m, p is ieee_mul: it is subnormal for |m| < 2^-125, where the next fma returns m either way.
// For a normal x no other step meets a subnormal: m is zero or at least 2^-126 in magnitude, and so is r.
static inline float log1pf_nv(float x) {
    const uint xb = as_type<uint>(x);
    if ((xb & 0x7f800000u) == 0u) return x;                                 // (a)
    const bool big = fabs(x) > 1.0f;                                         // (b) FADD.RZ t, x, 1
    const float hi = big ? x : 1.0f, lo = big ? 1.0f : x;
    const float rn = x + 1.0f;
    const float z = rn - hi;
    const float err = lo - z;
    const bool down = (err < 0.0f && rn > 0.0f) || (err > 0.0f && rn < 0.0f);
    const float t = down ? as_type<float>(as_type<uint>(rn) - 1u) : rn;
    const uint k = (as_type<uint>(t) - 0x3F400000u) & 0xFF800000u;          // IADD3, LOP3.LUT
    float m = as_type<float>(xb - k);                                        // IADD3 m, x, -k (integer)
    float s = as_type<float>(0x40800000u - k);                               // IADD3 s, -k, 0x40800000 (integer)
    s = fma(s, 0.25f, -1.0f);
    m = m + s;
    float kf = float(as_type<int>(k));                                       // I2FP.F32.S32: exact
    kf = kf * as_type<float>(0x34000000u);                                   // * 2^-23
    float p = fma(m, as_type<float>(0xBD39BF78u), as_type<float>(0x3DD80012u));
    p = fma(p, m, as_type<float>(0xBE0778E0u));
    p = fma(p, m, as_type<float>(0x3E146475u));
    p = fma(p, m, as_type<float>(0xBE2A68DDu));
    p = fma(p, m, as_type<float>(0x3E4CAF9Eu));
    p = fma(p, m, as_type<float>(0xBE800042u));
    p = fma(p, m, as_type<float>(0x3EAAAAE6u));
    p = fma(p, m, as_type<float>(0xBF000000u));
    p = ieee_mul(m, p);                                                      // (c) FMUL p, m, p
    float r = fma(m, p, m);
    r = fma(kf, as_type<float>(0x3F317218u), r);                             // + k ln 2
    if (xb >= 0x7F800000u) {                                                 // ISETP.GE.U32 P0
        if (as_type<int>(xb) > as_type<int>(0xBF800000u)) r = fma(x, INFINITY, INFINITY);   // ISETP.GT P2
        if ((xb & 0x7fffffffu) == 0u) r = as_type<float>(0x80000000u);      // FSEL on x == 0: -0, returned by (a)
    }
    return r;
}
