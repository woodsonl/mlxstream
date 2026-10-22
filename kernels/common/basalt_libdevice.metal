// Transcribed from NVIDIA libdevice's __nv_exp, __nv_logf and __nv_expf as nvcc 13.0 (V13.0.48, -arch=sm_120
// -lineinfo -O3 -DNDEBUG) inlined them into basalt's sampler.cu (__nv_exp, __nv_logf) and prefill/kernels.cu
// (__nv_expf) (jesdga95/basalt src/kernels/cuda/sampler.cu and src/prefill/kernels.cu
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36; third_party/NOTICE); PTX listings and transcriptions:
// docs/research/basalt-nvcc/sampler.md sections 2.5-2.6 and docs/research/basalt-nvcc/prefill_kernels.md section 3.
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

// libdevice __nv_expf as nvcc 13.0 inlines it into an IEEE-built kernel (kernels.cu:642, :770).
// expf(float) for src/prefill/kernels.cu:642, the prompt router's library expf; PTX listing and transcription:
// docs/research/basalt-nvcc/prefill_kernels.md section 3. Every step but ex2.approx.ftz (fast::exp2, invariant 5) is
// exact on this GPU: fl(t * 252) rounds up at most to the next integer, which the sign of the fma corrects to
// fma.rm's floor; t and f are never subnormal for a normal x, and a subnormal x gives 1.0 on both machines; only the
// scale multiply can write a subnormal (x in about (-104, -87.3]), so it is ieee_mul. expf(-inf) = +0. A NaN x gives
// a NaN with the hardware multiply's bits [H: CUDA's mul.f32 gives 0x7fffffff].
static inline float nv_expf(float x) {
    const float t = saturate(fma(x, as_type<float>(0x3BBB989Du), 0.5f));   // fma.rn, cvt.sat: t in [0, 1]
    float k = floor(t * 252.0f);                                              // fma.rm(t, 252, 12582913) - 12582913
    if (fma(t, 252.0f, -k) < 0.0f) k -= 1.0f;                                 //   = floor(exact t * 252)
    const float j = 12582913.0f + k;                                          // exact: integers below 2^24
    const float n = j + as_type<float>(0xCB40007Fu);                          // add.f32 j - 12583039 = k - 126
    float f = fma(x, as_type<float>(0x3FB8AA3Bu), -n);                        // x * log2(e) - n
    f = fma(x, as_type<float>(0x32A57060u), f);                               // + x * (log2(e) - 0x3FB8AA3B)
    const float s = as_type<float>(as_type<uint>(j) << 23);                   // 2^n, n in [-126, 126]
    return ieee_mul(s, fast::exp2(f));                                        // ex2.approx.ftz -> fast::exp2
}
