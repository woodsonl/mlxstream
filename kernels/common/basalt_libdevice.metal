// Transcribed from NVIDIA libdevice's __nv_exp, __nv_logf and __nv_powf as nvcc 13.0 (V13.0.48, -arch=sm_120
// -lineinfo -O3 -DNDEBUG) inlined them into basalt's sampler.cu (__nv_exp, __nv_logf) and prefill/kernels.cu
// (__nv_powf) (jesdga95/basalt src/kernels/cuda/sampler.cu and src/prefill/kernels.cu
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36; third_party/NOTICE); PTX listings and transcriptions:
// docs/research/basalt-nvcc/sampler.md sections 2.5-2.6 and docs/research/basalt-nvcc/prefill_kernels.md section 4.
//
// CUDA's binary64 exp and binary32 logf, which are not correctly rounded, so no Metal function gives their bits: the
// sampled tail's exp((double) x - (double) max) and logf(min_p), operation for operation. Needs basalt_ieee.metal and
// basalt_f64.metal ahead of it.
//
// - nv_exp (binary64 bits in and out): each fma.rn.f64, add.rn.f64 and mul.f64 of the PTX is the software binary64's
//   operation, each integer step the same operation on 32-bit words. A NaN argument gives F64_NAN.
// - nv_logf: the PTX's binary32 operations, fma() at its twelve fma.rn.f32 and a statement for each other one.
//   A subnormal argument's 2^23 pre-scale goes through ieee_mul (the hardware would flush it) and every compare is on
//   bits, so no subnormal reaches the hardware's arithmetic or compares. A NaN result (a NaN or negative argument)
//   has the hardware fma's NaN bits [H: CUDA's FFMA gives 0x7fffffff]; the tail passes only min_p > 0.
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

// ---- the prompt rotation's angle (rope_kernel<false>, src/prefill/kernels.cu:751), as nvcc inlines libdevice into
// that IEEE-built kernel; PTX and transcription: docs/research/basalt-nvcc/prefill_kernels.md section 4. Each
// fma.rn.f32 is an fma(), each other operation a statement of its own (ptxas contracts nothing further). On the
// rotation's arguments every intermediate is normal or zero, so the hardware's flush never acts and plain Metal
// arithmetic rounds as CUDA's does.

// powf(x, y) for x = theta_scale in [2^(-125/31), 1] (the rope handler refuses any other: x and every power x^0 ..
// x^31 are normal) and y = 0, 1, ..., 31, so libdevice's other branches (a subnormal, zero, negative, infinite or NaN
// x; |y log2 x| > 152) cannot run and are left out. log2(x) = lh + ll from m = x / 2^e in [2^-1/2, 2^1/2) and
// u = 2 (m - 1) / (m + 1); y log2(x) = yh + yl; 2^f (f = yh - n + yl, n = rint(yh)) in six Horner fmas; then 2^n
// through two exponent-field multiplies. The one approximate instruction, rcp.approx.ftz.f32 of m + 1, is the
// correctly rounded 1.0f / (m + 1) (invariant 5): CUDA's bits wherever no result depends on its last bits, which
// holds for every pair at this pack's 0x3F1AB32B [M] (prefill_kernels.md section 4); elsewhere invariant 5's mapping.
static inline float nv_powf(float x, float y) {
    if (y == 0.0f || x == 1.0f) return 1.0f;
    const int ex = (as_type<int>(x) - 0x3F3504F3) & int(0xFF800000u);
    const float m = as_type<float>(as_type<int>(x) - ex);            // in [2^-1/2, 2^1/2)
    const float e = fma(float(ex), 0x1p-23f, 0.0f);                    // the exponent, exact
    const float m1 = m + -1.0f, rc = 1.0f / (m + 1.0f);               // rcp.approx.ftz.f32 (invariant 5)
    const float u = (m1 + m1) * rc, uu = u * u;
    float d = m1 - u;
    d = d + d;
    const float ul = rc * fma(-u, m1, d);
    float p = fma(uu, as_type<float>(0x3A2C32E4u), as_type<float>(0x3B52E7DBu));
    p = fma(p, uu, as_type<float>(0x3C93BB73u));
    p = fma(p, uu, as_type<float>(0x3DF6384Fu));
    p = p * uu;
    const float l2e = as_type<float>(0x3FB8AA3Bu), h = fma(u, l2e, e);
    float t = e - h;
    t = fma(u, l2e, t);
    t = fma(ul, l2e, t);
    t = fma(u, as_type<float>(0x32A55E34u), t);
    t = fma(p * 3.0f, ul, t);
    t = fma(p, u, t);
    const float lh = h + t, ll = t - (lh - h);                         // log2(x) = lh + ll
    const float yh = lh * y, yl = fma(ll, y, fma(lh, y, -yh));
    const float n = rint(yh), f = (yh - n) + yl;
    float q = fma(f, as_type<float>(0x391FCB8Eu), as_type<float>(0x3AAF85EDu));
    q = fma(q, f, as_type<float>(0x3C1D9856u));
    q = fma(q, f, as_type<float>(0x3D6357BBu));
    q = fma(q, f, as_type<float>(0x3E75FDECu));
    q = fma(q, f, as_type<float>(0x3F317218u));
    q = fma(q, f, 1.0f);                                               // 2^f
    const uint b = n > 0.0f ? 0u : 0x83000000u;                        // 2^127 * 2^(n - 127), or 2^-123 * 2^(n + 123)
    return (q * as_type<float>(b + 0x7F000000u)) * as_type<float>((uint(int(n)) << 23) - b);
}
