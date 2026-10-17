// IEEE 754 binary64 in software, written for this port (the repository's own code; no upstream copy).
//
// The sampler's double-precision tail (docs/research/basalt-nvcc/sampler.md section 2.4) on a GPU without double: a
// binary64 is its bit pattern in a ulong. NVIDIA's fp64 rounds to nearest even and never flushes, so every operation
// here is correctly rounded with gradual underflow, propagates NaN and rounds overflow to infinity. Any correct
// implementation gives NVIDIA's bits; tests/kernels/test_basalt_sampler_helpers.py checks this one against the host's
// binary64 and binary32 bit for bit. Needs only the prelude. Every loop is bounded (normalization uses clz).
//
// - hi, lo, bits(l, h): the high and low 32-bit words of a binary64, and the binary64 of two words.
// - f64_from_f32 (float bits in): exact; a float subnormal becomes a normal double. f32_from_f64 (float bits out):
//   rounded to nearest even, float subnormals produced, never flushed; the caller writes as_type<float>(bits).
// - f64_add, f64_sub: the exact sum, rounded once; x + (-x) = +0, -0 + -0 = -0, inf - inf = NaN.
// - f64_lt, f64_le, f64_ge: ordered compares on order-preserving keys: false when either operand is NaN, -0 == +0.
// - f64_fma: the exact a * b + c rounded once. f64_mul(a, b) is f64_fma(a, b, -0). f64_div: restoring division in
//   exactly 55 steps, rounded once.
// - f64_fma_odd: the exact a * b + c rounded to odd (toward zero, the inexact flag ORed into the last bit; a nonzero
//   value never gives 0, overflow gives +-DBL_MAX, specials as f64_fma). For binary32 operands, f32_from_f64 of it is
//   their fma correctly rounded (53 >= 24 + 2: Boldo and Melquiond's rounding to odd), where
//   f32_from_f64(f64_fma(...)) can round twice to the wrong neighbour.
// - A NaN result is F64_NAN, and f32_from_f64 of a NaN is 0x7fffffff, NVIDIA's canonical single-precision NaN [H: the
//   bits NVIDIA's fp64 and its conversion give for a NaN were not checked; no basalt check depends on them].
#pragma STDC FP_CONTRACT OFF

constant constexpr ulong F64_SIGN = 0x8000000000000000ul;
constant constexpr ulong F64_INF = 0x7ff0000000000000ul;
constant constexpr ulong F64_NAN = 0x7ffffffffffffffful;
constant constexpr ulong F64_FRAC = 0x000ffffffffffffful;

static inline uint hi(ulong x) { return uint(x >> 32); }
static inline uint lo(ulong x) { return uint(x); }
static inline ulong bits(uint l, uint h) { return (ulong(h) << 32) | ulong(l); }

static inline bool f64_isnan(ulong x) { return (x & ~F64_SIGN) > F64_INF; }

// m >> n, every bit shifted out ORed into bit 0 (the sticky bit), for any n (a shift by 64 or more is not left to MSL)
static inline ulong f64_shr_jam(ulong m, uint n) {
    if (n == 0u) return m;
    if (n >= 64u) return ulong(m != 0ul);
    return (m >> n) | ulong((m << (64u - n)) != 0ul);
}

// The binary64 nearest (ties to even) to (-1)^s * m * 2^(e - 1085): bit 63 of s is the sign; m < 2^63 has its
// leading bit at 62, or e <= 1. A finite double's biased exponent E and 53-bit significand M give e = E (1 for a
// subnormal) and m = M << 10: the ten low bits lie below the result's last place, and bit 0 also holds the sticky bit
// of anything below them. A value under the normal range (e < 1) is denormalized first, its shifted-out bits jammed,
// so it rounds once, on the subnormal grid. The significand's carry adds into the exponent field: a rounded-up
// subnormal becomes 2^-1022, and the top binade rounds up to infinity, as does any e above it.
static inline ulong f64_round(ulong s, int e, ulong m) {
    if (e > 2046) return s | F64_INF;
    if (e < 1) {
        m = f64_shr_jam(m, uint(1 - e));
        e = 1;
    }
    const ulong q = m >> 10, rest = m & 0x3fful;
    const ulong up = ulong(rest > 0x200ul || (rest == 0x200ul && (q & 1ul) != 0ul));
    return s | ((ulong(e - 1) << 52) + q + up);
}

static inline ulong f64_add(ulong a, ulong b) {
    ulong x = a & ~F64_SIGN, y = b & ~F64_SIGN;                            // the magnitudes
    if (x > F64_INF || y > F64_INF) return F64_NAN;
    if (x < y) {                                                            // a is the larger in magnitude
        const ulong t = a, u = x;
        a = b; b = t; x = y; y = u;
    }
    const bool opp = ((a ^ b) & F64_SIGN) != 0ul;
    if (x == F64_INF) return (opp && y == F64_INF) ? F64_NAN : a;
    if (y == 0ul) return x == 0ul ? (a & b) : a;                            // -0 only from -0 + -0
    int ea = int(x >> 52), eb = int(y >> 52);
    ulong ma = (x & F64_FRAC) << 10, mb = (y & F64_FRAC) << 10;
    if (ea != 0) ma |= 1ul << 62; else ea = 1;                              // a subnormal: binade 1, no leading 1
    if (eb != 0) mb |= 1ul << 62; else eb = 1;
    mb = f64_shr_jam(mb, uint(ea - eb));                                    // b on a's grid, the rest jammed
    ulong m;
    if (!opp) {
        m = ma + mb;                                                        // below 2^64
        if ((m >> 63) != 0ul) {
            m = f64_shr_jam(m, 1u);
            ea += 1;
        }
    } else {
        m = ma - mb;                                                        // |a| >= |b|: no borrow out
        if (m == 0ul) return 0ul;                                           // x + (-x) = +0
        const int n = min(int(clz(m)) - 1, ea - 1);                         // the leading bit back to 62, or binade 1
        m <<= uint(n);
        ea -= n;
    }
    return f64_round(a & F64_SIGN, ea, m);
}

static inline ulong f64_sub(ulong a, ulong b) { return f64_add(a, b ^ F64_SIGN); }

// order-preserving key of a non-NaN binary64, -0 and +0 equal
static inline ulong f64_key(ulong x) {
    x = x == F64_SIGN ? 0ul : x;
    return (x & F64_SIGN) != 0ul ? ~x : (x | F64_SIGN);
}
static inline bool f64_lt(ulong a, ulong b) { return !f64_isnan(a) && !f64_isnan(b) && f64_key(a) < f64_key(b); }
static inline bool f64_le(ulong a, ulong b) { return !f64_isnan(a) && !f64_isnan(b) && f64_key(a) <= f64_key(b); }
static inline bool f64_ge(ulong a, ulong b) { return !f64_isnan(a) && !f64_isnan(b) && f64_key(a) >= f64_key(b); }

static inline ulong f64_from_f32(uint f) {
    const ulong s = ulong(f & 0x80000000u) << 32;
    const uint e = (f >> 23) & 0xffu, m = f & 0x7fffffu;
    if (e == 0xffu) return m != 0u ? F64_NAN : (s | F64_INF);
    if (e != 0u) return s | (ulong(e + 896u) << 52) | (ulong(m) << 29);      // 896 = 1023 - 127
    if (m == 0u) return s;
    const uint lead = 31u - clz(m);                                         // m * 2^-149 = 2^(lead - 149) * 1.f
    return s | (ulong(lead + 874u) << 52) | ((ulong(m) << (52u - lead)) & F64_FRAC);
}

static inline uint f32_from_f64(ulong d) {
    const uint s = hi(d) & 0x80000000u;
    const int e = int(hi(d) >> 20) & 0x7ff;
    if (e == 0x7ff) return (d & F64_FRAC) != 0ul ? 0x7fffffffu : (s | 0x7f800000u);
    const int ef = e - 896;                                                 // the float's biased exponent
    if (ef > 254) return s | 0x7f800000u;                                   // |d| >= 2^128
    if (ef < -23) return s;                                                 // |d| < 2^-150: half the least subnormal
    const ulong m = (d & F64_FRAC) | (1ul << 52);                           // d is normal here
    const uint k = uint(29 + max(0, 1 - ef));                               // m's bits below the float's last place
    const ulong q = m >> k, rest = m & ((1ul << k) - 1ul), mid = 1ul << (k - 1u);
    const uint up = uint(rest > mid || (rest == mid && (q & 1ul) != 0ul));
    return s | ((uint(max(ef - 1, 0)) << 23) + uint(q) + up);              // a carry moves the exponent up
}

// ---- multiply, fma, divide -----------------------------------------------------------------------------------------

constant constexpr ulong F64_MAX = 0x7feffffffffffffful;

// f64_round's input rounded to odd: denormalized with the jam as there, truncated toward zero, and the inexact flag
// ORed into the last bit. A nonzero value never becomes 0 (a tiny one gives the least subnormal), and a value past
// the largest finite gives it, +-DBL_MAX (odd), where f64_round gives infinity.
static inline ulong f64_round_odd(ulong s, int e, ulong m) {
    if (e > 2046) return s | F64_MAX;
    if (e < 1) {
        m = f64_shr_jam(m, uint(1 - e));
        e = 1;
    }
    return s | ((ulong(e - 1) << 52) + ((m >> 10) | ulong((m & 0x3fful) != 0ul)));
}

// a finite nonzero magnitude x as m * 2^(e - 1075), m in [2^52, 2^53): a subnormal is normalized with clz
static inline ulong f64_sig(ulong x, thread int& e) {
    e = int(x >> 52);
    if (e != 0) return (x & F64_FRAC) | (1ul << 52);
    const uint n = uint(clz(x)) - 11u;                                      // the leading bit up to 52
    e = 1 - int(n);
    return x << n;
}

// the 128-bit (h, l) >> n, every bit shifted out ORed into bit 0, for any n (no shift by 64 or more is left to MSL)
static inline void u128_shr_jam(thread ulong& h, thread ulong& l, uint n) {
    if (n == 0u) return;
    if (n < 64u) {
        l = (l >> n) | (h << (64u - n)) | ulong((l << (64u - n)) != 0ul);
        h >>= n;
    } else {
        l = f64_shr_jam(h, n - 64u) | ulong(l != 0ul);                      // n >= 128 leaves only the sticky bit
        h = 0ul;
    }
}

// the 128-bit (h, l) << n, n < 128, when no set bit leaves
static inline void u128_shl(thread ulong& h, thread ulong& l, uint n) {
    if (n == 0u) return;
    if (n < 64u) {
        h = (h << n) | (l >> (64u - n));
        l <<= n;
    } else {
        h = l << (n - 64u);
        l = 0ul;
    }
}

// The exact a * b + c, rounded once: to nearest even, or to odd (f64_round_odd). The product of the two 53-bit
// significands is exact in 128 bits, its leading bit at 126: (h, l) * 2^(e - 1149), so the high word is f64_round's
// m. c joins it on that grid, the smaller operand shifted with its lost bits jammed (a cancellation of more than one
// bit needs exponents at most 1 apart, where the shift loses nothing, so it is exact), the sum is normalized with
// clz, and the low word is jammed into bit 0, ten bits below the last place. Exact cancellation gives +0, as does
// +0 + -0; -0 + -0 gives -0.
static inline ulong f64_fma_rounded(ulong a, ulong b, ulong c, bool odd) {
    const ulong x = a & ~F64_SIGN, y = b & ~F64_SIGN, z = c & ~F64_SIGN, ps = (a ^ b) & F64_SIGN;
    if (x > F64_INF || y > F64_INF || z > F64_INF) return F64_NAN;
    if (x == F64_INF || y == F64_INF) {                                     // inf * 0, or an infinite product
        if (x == 0ul || y == 0ul || (z == F64_INF && (c & F64_SIGN) != ps)) return F64_NAN;
        return ps | F64_INF;
    }
    if (z == F64_INF) return c;
    if (x == 0ul || y == 0ul) return z == 0ul ? (ps & c) : c;               // an exact zero product
    int ea, eb;
    const ulong ma = f64_sig(x, ea) << 10, mb = f64_sig(y, eb) << 11;
    ulong h = mulhi(ma, mb), l = ma * mb, s = ps;                           // in [2^125, 2^127)
    int e = ea + eb - 1022;
    if ((h >> 62) == 0ul) {
        u128_shl(h, l, 1u);
        e -= 1;
    }
    if (z != 0ul) {
        int ec;
        ulong ch = f64_sig(z, ec) << 10, cl = 0ul, cs = c & F64_SIGN;
        if (ec > e || (ec == e && ch > h)) {                                // (h, l) is the larger magnitude
            const ulong th = h, tl = l, ts = s;
            const int te = e;
            h = ch; l = cl; s = cs; e = ec;
            ch = th; cl = tl; cs = ts; ec = te;
        }
        u128_shr_jam(ch, cl, uint(e - ec));
        if (s == cs) {
            const ulong t = l + cl;
            h += ch + ulong(t < l);                                         // below 2^64: both below 2^63
            l = t;
            if ((h >> 63) != 0ul) {
                u128_shr_jam(h, l, 1u);
                e += 1;
            }
        } else {
            const ulong t = l - cl;
            h -= ch + ulong(l < cl);                                        // |(h, l)| >= |(ch, cl)|: no borrow out
            l = t;
            if ((h | l) == 0ul) return 0ul;
            const uint n = (h != 0ul ? uint(clz(h)) : 64u + uint(clz(l))) - 1u;
            u128_shl(h, l, n);
            e -= int(n);
        }
    }
    const ulong m = h | ulong(l != 0ul);
    return odd ? f64_round_odd(s, e, m) : f64_round(s, e, m);
}

static inline ulong f64_fma(ulong a, ulong b, ulong c) { return f64_fma_rounded(a, b, c, false); }
static inline ulong f64_fma_odd(ulong a, ulong b, ulong c) { return f64_fma_rounded(a, b, c, true); }

// -0 is the sum's identity: +0 + -0 = +0, -0 + -0 = -0, and a product that rounds to 0 keeps its sign
static inline ulong f64_mul(ulong a, ulong b) { return f64_fma(a, b, F64_SIGN); }

// Restoring division of the normalized significands: 55 quotient bits (the 53 kept, the round bit and one below it),
// one per step in exactly 55 steps, then the remainder's sticky bit, rounded once by f64_round.
static inline ulong f64_div(ulong a, ulong b) {
    const ulong x = a & ~F64_SIGN, y = b & ~F64_SIGN, s = (a ^ b) & F64_SIGN;
    if (x > F64_INF || y > F64_INF) return F64_NAN;
    if (x == F64_INF) return y == F64_INF ? F64_NAN : (s | F64_INF);
    if (y == F64_INF) return s;
    if (y == 0ul) return x == 0ul ? F64_NAN : (s | F64_INF);
    if (x == 0ul) return s;
    int ea, eb;
    ulong r = f64_sig(x, ea);
    const ulong d = f64_sig(y, eb);
    int e = ea - eb + 1023;
    if (r < d) {                                                            // the quotient r / d in [1, 2)
        r <<= 1;
        e -= 1;
    }
    ulong q = 0ul;
    for (uint i = 0u; i < 55u; ++i) {                                       // r < 2d throughout
        const bool ge = r >= d;
        q = (q << 1) | ulong(ge);
        r = (ge ? r - d : r) << 1;
    }
    return f64_round(s, e, (q << 8) | ulong(r != 0ul));                     // q's leading bit at 54, moved to 62
}
