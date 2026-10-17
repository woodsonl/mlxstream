# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): the host arithmetic its references compute in (binary64 sums and the float casts),
# checked here bit for bit on the port's helpers.
"""The sampler family's shared helpers on Metal (kernels/common/basalt_f64.metal, basalt_libdevice.metal and
basalt_sampler.metal), each through an inline kernel appended to ``basalt_sampler_source()`` (test_basalt_platform.py's
pattern), against host references. Every comparison is bitwise.

- H.1: the software binary64's conversions (``f64_from_f32``, ``f32_from_f64``), ``f64_add``, ``f64_sub``, the
  ordered compares and the word helpers (``hi``, ``lo``, ``bits``) against the host's binary64 (numpy's float64, the
  arithmetic of Python's ``float``) and numpy's float32 conversion, on 2^16 uniform bit patterns and planted classes:
  exponent gaps over every branch of the alignment shift, near and exact cancellation, subnormals, ties, carries past
  a tie by the sticky bit alone, overflow, every pair of specials, and float-range doubles, float midpoints and edges
  for the conversion. A NaN result must be ``F64_NAN`` (0x7fffffffffffffff), and ``f32_from_f64`` of a NaN
  0x7fffffff.
- H.2: ``f64_mul``, ``f64_fma``, ``f64_fma_odd`` and ``f64_div`` against the host's binary64 (numpy's ``*`` and
  ``/``, ``math.fma``) and an exact integer reference that rounds once, to nearest even or to odd (``round64``, its
  nearest-even results checked against the host's on every row), on uniform patterns and planted classes: every
  exponent gap between the product and ``c`` over [-140, 140] and beyond, cancellation (shallow, deep into the low
  word, and exact, which gives +0 in both roundings), exact and near ties of products, sums and quotients (exact ties
  on the subnormal grid at every denormalization shift), subnormal results, overflow (to inf, and to DBL_MAX when
  rounded to odd) and every triple of specials.
"""

import math
import struct

import numpy as np

from monolith import kernels
from tests.kernels import basalt_gpu as gpu

U32, U64, I64, F32, F64 = np.uint32, np.uint64, np.int64, np.float32, np.float64
SIGN, FRAC, MAG = U64(1 << 63), U64((1 << 52) - 1), U64((1 << 63) - 1)
EXP = U64(0x7FF << 52)
F64_NAN, F32_NAN = U64(0x7FFFFFFFFFFFFFFF), U64(0x7FFFFFFF)


def run_kernel(src, name, inputs, cols, dtype=U64):
    """The inline kernel ``name`` (``src``, appended to basalt_sampler_source()), one thread per row of ``inputs``
    (equal-length arrays at buffers 0, 1, ...); then the output, ``cols`` values of ``dtype`` per row, starting as 0xFF
    bytes, and the row count at the next slot. Returns the output as (rows, cols)."""
    n = len(inputs[0])
    pso = gpu.pipeline(kernels.basalt_sampler_source() + src, name)
    bufs = [np.ascontiguousarray(x).tobytes() for x in inputs] + [cols * n * np.dtype(dtype).itemsize]
    out = gpu.run(pso, bufs, np.uint32(n).tobytes(), len(inputs) + 1, -(-n // 256), 256, fill=0xFF)
    return np.frombuffer(out[-1], dtype).reshape(n, cols)


# ---- H.1: binary64 conversions, add, sub, compares, words ----------------------------------------------------------

H1 = """
kernel void h1_f64(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                   device const uint* f [[buffer(2)]], device ulong* out [[buffer(3)]],
                   constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const ulong x = a[i], y = b[i];
  device ulong* o = out + 6 * i;
  o[0] = f64_add(x, y);
  o[1] = f64_sub(x, y);
  o[2] = f64_from_f32(f[i]);
  o[3] = ulong(f32_from_f64(x));
  o[4] = ulong(f64_lt(x, y)) | (ulong(f64_le(x, y)) << 1) | (ulong(f64_ge(x, y)) << 2);
  o[5] = bits(lo(y), hi(x));
}
"""


def words(rng, n, dtype=U64):
    return np.frombuffer(rng.bytes(np.dtype(dtype).itemsize * n), dtype).copy()


def doubles(rng, exp):
    """Random signs and fractions on the biased exponents ``exp``."""
    return (words(rng, len(exp)) & (SIGN | FRAC)) | (np.asarray(exp).astype(U64) << U64(52))


def offset(x, k):
    """x with its magnitude moved by k (int64) units in the last place, clamped to [0, DBL_MAX]; x's sign."""
    mag = np.clip((x & MAG).view(I64) + np.asarray(k, I64), 0, 0x7FEFFFFFFFFFFFFF)
    return mag.view(U64) | (x & SIGN)


def biased(x):
    return ((x & EXP) >> U64(52)).astype(I64)


def subnormal(x):
    return ((x & EXP) == 0) & ((x & FRAC) != 0)


SPECIALS = np.array([0, 1, 2, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x0010000000000001,      # +0, subnormals, DBL_MIN
                     0x3CA0000000000000, 0x3FEFFFFFFFFFFFFF, 0x3FF0000000000000, 0x3FF0000000000001,  # 2^-53, ~1
                     0x7C8FFFFFFFFFFFFF, 0x7C90000000000000, 0x7FEFFFFFFFFFFFFF,     # 2^970 (DBL_MAX's half ulp)
                     0x7FF0000000000000, 0x7FF0000000000001, 0x7FF8000000000000, 0x7FFFFFFFFFFFFFFF], U64)
# f32_from_f64's edges and the float bits they must give (both signs are run)
F32_EDGES = [(0x3690000000000000, 0), (0x3690000000000001, 1), (0x368FFFFFFFFFFFFF, 0),   # 2^-150: the tie to 0
             (0x380FFFFFE0000000, 0x00800000),                    # 2^-126 - 2^-150: the subnormal tie, up to 2^-126
             (0x47EFFFFFF0000000, 0x7F800000), (0x47EFFFFFEFFFFFFF, 0x7F7FFFFF),   # FLT_MAX + half an ulp, and below
             (0x47F0000000000000, 0x7F800000), (0x0000000000000001, 0), (0x000FFFFFFFFFFFFF, 0)]   # 2^128, subnormals


def h1_pairs(rng):
    """The H.1 operand pairs (uint64 bit patterns) by class."""
    n = 1 << 14
    k = np.arange(n)
    pairs = {"uniform": (words(rng, 1 << 16), words(rng, 1 << 16))}
    # exponent gaps over every branch of the alignment shift (0, 1, 2, 52-56, 62-65, past 64), either operand larger
    gap = rng.choice(np.r_[0:70, 0:3, 52:57, 62:66, 70, 100, 1100], n)
    ea = rng.integers(1, 2047, n)
    x, y = doubles(rng, ea), doubles(rng, np.clip(ea - gap, 0, 2046))
    swap = rng.integers(0, 2, n).astype(bool)
    pairs["gaps"] = (np.where(swap, y, x), np.where(swap, x, y))
    # near cancellation: b = -(a moved by up to 1000 ulps), exactly -a for every 8th; half at biased exponents 1-60
    # (subnormal results: the normalization stops at binade 1), a quarter with tiny fractions (a binade crossed)
    x = doubles(rng, np.where(k % 2 == 0, rng.integers(1, 61, n), rng.integers(1, 2047, n)))
    x = np.where(k % 4 == 1, (x & ~FRAC) | rng.integers(0, 8, n).astype(U64), x)
    pairs["cancel"] = (x, offset(x, np.where(k % 8 == 0, 0, rng.integers(-1000, 1001, n))) ^ SIGN)
    # subnormals and the two lowest normal binades, every sign: sums crossing 2^-1022, differences below it
    pairs["subnormal"] = (doubles(rng, rng.integers(0, 3, n)), doubles(rng, rng.integers(0, 3, n)))
    # ties: b = +-half an ulp of a (exactly, or one ulp of b off it), either sign
    ea = rng.integers(54, 2047, n)
    x = doubles(rng, ea)
    tie = (words(rng, n) & SIGN) | ((ea - 53).astype(U64) << U64(52))
    pairs["ties"] = (x, offset(tie, rng.integers(-1, 2, n)))
    # carries past a tie only by the sticky bit: a = 2^(E+1) - u (u: a's ulp), b = 2^(E-d) * (1 + f * 2^-52) with
    # f = K * 2^(d+1) + t, 0 < t < 2^(d-9), so a + b = 2^(E+1) + (2j + 1) u + t * 2^-d * u: above the next binade's
    # tie by less than u / 512, which the carry's one-bit shift must keep (a same sign, rounded up)
    d = rng.integers(11, 52, n)
    ea = rng.integers(60, 2046, n)
    f = (rng.integers(0, 2 ** (51 - d)) << (d + 1)) + rng.integers(1, 2 ** (d - 9))
    x = (words(rng, n) & SIGN) | (ea.astype(U64) << U64(52)) | FRAC
    pairs["carry"] = (x, (x & SIGN) | ((ea - d).astype(U64) << U64(52)) | f.astype(U64))
    # overflow: the top three binades, mostly of one sign
    ea = rng.integers(2044, 2047, n)
    x, y = doubles(rng, ea), doubles(rng, ea - rng.integers(0, 4, n))
    pairs["overflow"] = (x, np.where(k % 4 == 0, y, (y & MAG) | (x & SIGN)))
    s = np.r_[SPECIALS, SPECIALS | SIGN]
    pa, pb = np.meshgrid(s, s)
    pairs["specials"] = (pa.ravel(), pb.ravel())
    # f32_from_f64: float-range doubles (float normals, subnormals, underflow to 0, overflow)
    pairs["to_f32"] = (doubles(rng, rng.integers(1023 - 152, 1023 + 130, n)), words(rng, n))
    # midpoints of adjacent floats (a quarter subnormal), exactly or one double ulp off
    fx = np.where(k % 4 == 0, rng.integers(0, 0x007FFFFF, n), rng.integers(0, 0x7F7FFFFF, n)).astype(U32)
    mid = (fx.view(F32).astype(F64) + (fx + U32(1)).view(F32).astype(F64)) * 0.5
    pairs["midpoints"] = (offset(mid.view(U64) | (words(rng, n) & SIGN), rng.integers(-1, 2, n)), words(rng, n))
    e = np.array([v for v, _ in F32_EDGES], U64)
    pairs["edges"] = (np.r_[e, e | SIGN], words(rng, 2 * len(e)))
    return pairs


def h1_floats(rng, n):
    """f64_from_f32's inputs: uniform float bits, a quarter subnormal (or zero), and planted specials."""
    f = words(rng, n, U32)
    f[: n // 4] &= U32(0x807FFFFF)
    planted = [0, 1, 2, 0x7FFFFF, 0x800000, 0x800001, 0x3F800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000,
               0x7FFFFFFF] + [1 << j for j in range(23)] + [(2 << j) - 1 for j in range(23)]
    p = np.array(planted, U32)
    f[n // 4: n // 4 + 2 * len(p)] = np.r_[p, p | U32(0x80000000)]
    return f


def canon(x):
    """float64 values as bits, every NaN as F64_NAN."""
    return np.where(np.isnan(x), F64_NAN, x.view(U64))


def test_basalt_f64_basics_vs_host():
    rng = np.random.default_rng(21)
    pairs = h1_pairs(rng)
    a = np.concatenate([v[0] for v in pairs.values()])
    b = np.concatenate([v[1] for v in pairs.values()])
    f = h1_floats(rng, len(a))
    assert a.dtype == b.dtype == U64 and f.dtype == U32
    got = run_kernel(H1, "h1_f64", [a, b, f], 6)
    A, B = a.view(F64), b.view(F64)
    with np.errstate(all="ignore"):
        add, sub, wide = canon(A + B), canon(A - B), canon(f.view(F32).astype(F64))
        narrow = np.where(np.isnan(A), F32_NAN, A.astype(F32).view(U32).astype(U64))
    cmp = (A < B).astype(U64) | ((A <= B).astype(U64) << U64(1)) | ((A >= B).astype(U64) << U64(2))
    words_ = (a & U64(0xFFFFFFFF00000000)) | (b & U64(0xFFFFFFFF))
    for col, name, want in [(0, "f64_add", add), (1, "f64_sub", sub), (2, "f64_from_f32", wide),
                            (3, "f32_from_f64", narrow), (4, "f64_lt/le/ge", cmp), (5, "bits(lo, hi)", words_)]:
        bad = np.flatnonzero(got[:, col] != want)
        assert bad.size == 0, (f"{name}: {bad.size} of {len(a)} differ; first a={a[bad[0]]:#018x} b={b[bad[0]]:#018x} "
                               f"f={f[bad[0]]:#010x}: got {got[bad[0], col]:#x}, want {want[bad[0]]:#x}")
    # the planted classes are what they claim
    at, n0 = {}, 0
    for name, (x, _) in pairs.items():
        at[name], n0 = slice(n0, n0 + len(x)), n0 + len(x)
    ga, gb = a[at["gaps"]], b[at["gaps"]]
    shift = np.abs(np.maximum(biased(ga), 1) - np.maximum(biased(gb), 1))
    assert {0, 1, 2, 52, 53, 54, 55, 56, 62, 63, 64, 65} <= set(shift.tolist()) and shift.max() > 64
    for name in ("cancel", "subnormal"):
        assert np.count_nonzero(subnormal(add[at[name]])) > 100, name                   # subnormal sums
    c = at["cancel"]
    assert np.count_nonzero(add[c] == 0) > 100                                           # x + (-x) = +0
    assert np.count_nonzero((add[c] != 0) & (biased(add[c]) < biased(a[c]) - 20)) > 100  # deep normalization
    t, ta, tb = at["ties"], a[at["ties"]], b[at["ties"]]
    exact = ((tb & FRAC) == 0) & (biased(tb) == biased(ta) - 53) & (((ta ^ tb) & SIGN) == 0)
    up = (add[t] & MAG) == (ta & MAG) + U64(1)
    assert np.count_nonzero(exact & up) > 100 and np.count_nonzero(exact & ((add[t] & MAG) == (ta & MAG))) > 100
    assert np.all(biased(add[at["carry"]]) == biased(a[at["carry"]]) + 1)                # every carry class sum carries
    fin = np.isfinite(A) & np.isfinite(B)
    assert np.count_nonzero(fin & ((add & MAG) == U64(0x7FF0000000000000))) > 100       # finite sums overflowing
    sub32 = ((narrow & U64(0x7F800000)) == 0) & ((narrow & U64(0x7FFFFF)) != 0)
    assert np.count_nonzero(sub32) > 1000                                                # float subnormals produced
    e = at["edges"]
    want_edges = [w for _, w in F32_EDGES]
    assert narrow[e].tolist() == want_edges + [w | 0x80000000 for w in want_edges]


# ---- H.2: binary64 multiply, fma, fma rounded to odd, divide -------------------------------------------------------

H2 = """
kernel void h2_f64(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                   device const ulong* c [[buffer(2)]], device ulong* out [[buffer(3)]],
                   constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const ulong x = a[i], y = b[i], z = c[i];
  device ulong* o = out + 4 * i;
  o[0] = f64_mul(x, y);
  o[1] = f64_fma(x, y, z);
  o[2] = f64_fma_odd(x, y, z);
  o[3] = f64_div(x, y);
}
"""

SPECIAL, EXACT, INEXACT, NEAR, DOWN, UP = range(-1, 5)   # round64's cases; NEAR: within 2^-40 of a tie, off it
INF_BITS, MAX_BITS, NAN_BITS = 0x7FF0000000000000, 0x7FEFFFFFFFFFFFFF, int(F64_NAN)


def f64_bits(x):
    return NAN_BITS if math.isnan(x) else struct.unpack("<Q", struct.pack("<d", x))[0]


def round64(n, e, odd=False):
    """n * 2**e (an integer n != 0) rounded once to binary64, with gradual underflow: to nearest even (overflow to
    inf), or to odd (toward zero, the inexact flag ORed into the last bit; overflow to DBL_MAX). Returns the bits and
    the case: EXACT, INEXACT, NEAR, or a tie, DOWN or UP by its rounding to even."""
    s, n = (1 << 63) if n < 0 else 0, abs(n)
    k = max(n.bit_length() + e - 53, -1074) - e                         # the bits below the result's last place
    case = EXACT
    if k > 0:
        q, rest, half = n >> k, n & ((1 << k) - 1), 1 << (k - 1)
        case = (EXACT if rest == 0 else (UP if q & 1 else DOWN) if rest == half
                else NEAR if abs(rest - half) << 40 < half else INEXACT)
        n, e = (q | (rest != 0)) if odd else q + (rest > half or case == UP), e + k
    if n.bit_length() + e > 1024:
        return s | (MAX_BITS if odd else INF_BITS), case
    return s | f64_bits(math.ldexp(n, e)), case


def ref_fma(x, y, z, odd=False):
    """x * y + z (floats) rounded once, as IEEE's fusedMultiplyAdd or (``odd``) to odd, and round64's case. The
    specials follow from the operand classes: a NaN, inf * 0 and inf - inf give NaN; a finite product and an
    infinite z give z; an exact zero is +0, or -0 from -0 + -0."""
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        return NAN_BITS, SPECIAL
    if math.isinf(x) or math.isinf(y):
        return f64_bits(x * y + z), SPECIAL
    if math.isinf(z):
        return f64_bits(z), SPECIAL
    (nx, dx), (ny, dy), (nz, dz) = x.as_integer_ratio(), y.as_integer_ratio(), z.as_integer_ratio()
    n = nx * ny * dz + nz * dx * dy
    if n == 0:
        zero_neg = x * y == 0 and math.copysign(1.0, x * y) < 0 and math.copysign(1.0, z) < 0
        return (1 << 63) * zero_neg, SPECIAL
    return round64(n, 1 - (dx * dy * dz).bit_length(), odd)


def ref_div(x, y):
    """x / y (finite, nonzero floats) through round64: the quotient to 120 bits and the remainder's sticky bit."""
    (nx, dx), (ny, dy) = x.as_integer_ratio(), y.as_integer_ratio()
    num, den = abs(nx) * dy, dx * abs(ny)
    k = max(0, 120 - num.bit_length() + den.bit_length())
    q, r = divmod(num << k, den)
    return round64((-1 if (x < 0) != (y < 0) else 1) * (2 * q + (r != 0)), -k - 1)


def lg(x, y=1.0):
    """floor(log2 |x * y|), exactly, for finite nonzero floats."""
    (nx, dx), (ny, dy) = x.as_integer_ratio(), y.as_integer_ratio()
    return abs(nx * ny).bit_length() - (dx * dy).bit_length()


def dbl(m, e, neg=False):
    """The bits of +-m * 2^(e - 1075), m a 53-bit significand, e the biased exponent (1 to 2046)."""
    return (int(neg) << 63) | (e << 52) | (m - (1 << 52))


def odd53(rng):
    return int(rng.integers(1 << 52, 1 << 53)) | 1


def h2_rows(rng):
    """The H.2 operand triples (uint64 bit patterns) by class."""
    n = 1 << 12
    k = np.arange(2 * n)
    sign = lambda m: words(rng, m) & SIGN
    rows = {"uniform": tuple(words(rng, 1 << 15) for _ in range(3))}
    # every exponent gap between the product and c over [-140, 140] and +-300, +-2100 (clamped at the range's ends):
    # each branch of the 128-bit alignment shift, either operand the larger, either sign
    ea, eb = rng.integers(623, 1424, 4 * n), rng.integers(623, 1424, 4 * n)
    gap = rng.choice(np.r_[-140:141, -300, 300, -2100, 2100], 4 * n)
    rows["gaps"] = (doubles(rng, ea), doubles(rng, eb), doubles(rng, np.clip(ea + eb - 1023 + gap, 0, 2046)))
    # cancellation: c = -RN(a * b) moved by up to 4 ulps (exactly -RN(a * b), the product's rounding error, for every
    # 8th), and exact zeros: c = -a * b on 26-bit significands
    a, b = doubles(rng, rng.integers(723, 1324, 2 * n)), doubles(rng, rng.integers(723, 1324, 2 * n))
    p = (a.view(F64) * b.view(F64)).view(U64)
    rows["cancel"] = (a, b, offset(p, np.where(k % 8 == 0, 0, rng.integers(-4, 5, 2 * n))) ^ SIGN)
    sig26 = lambda m: ((rng.integers(1 << 25, 1 << 26, m) | 1) << 27).astype(U64) & FRAC
    a = sign(n) | sig26(n) | (rng.integers(723, 1324, n).astype(U64) << U64(52))
    b = sign(n) | sig26(n) | (rng.integers(723, 1324, n).astype(U64) << U64(52))
    rows["zeros"] = (a, b, (a.view(F64) * b.view(F64)).view(U64) ^ SIGN)
    # deep cancellation into the low word: Mb = R / Ma mod 2^53 (Ma odd) and c = -(Ma * Mb without its low 53 bits),
    # so the exact result is R at the product's last place, R of every bit length 1 to 53
    deep = []
    for i in range(2 * n):
        r = int(rng.integers(1 << (i % 53), 2 << (i % 53)))
        while True:
            ma = odd53(rng)
            mb = r * pow(ma, -1, 1 << 53) % (1 << 53)
            if mb >= 1 << 52:
                break
        ea, eb, neg = int(rng.integers(823, 1224)), int(rng.integers(823, 1224)), i % 2 == 1
        top = math.ldexp((ma * mb) >> 53, ea + eb - 2097)
        deep.append((dbl(ma, ea, neg), dbl(mb, eb), f64_bits(top if neg else -top)))
    rows["deep"] = tuple(np.array(v, U64) for v in zip(*deep))
    # products on a tie: u * v odd of 54 bits (exact ties, either way to even), or near one, Ma * Mb = 2^52 + d
    # mod 2^53 (Ma odd, 0 < |d| <= 3); c = +-0 (the product alone) or +-2^-150 of it (a sticky bit past a tie)
    ties = []
    for i in range(2 * n):
        if i % 2 == 0:
            ma = int(rng.integers(1 << 26, 1 << 27)) | 1
            mb = int(rng.integers(-(-(1 << 53) // ma), (1 << 54) // ma)) | 1
        else:
            d = int(rng.choice([-3, -2, -1, 1, 2, 3]))
            while True:
                ma = odd53(rng)
                mb = (2 ** 52 + d) * pow(ma, -1, 1 << 53) % (1 << 53)
                if mb >= 1 << 52:
                    break
        x = math.copysign(math.ldexp(ma, int(rng.integers(-500, 450))), float(rng.choice([-1.0, 1.0])))
        y = math.copysign(math.ldexp(mb, int(rng.integers(-500, 450))), float(rng.choice([-1.0, 1.0])))
        z = math.copysign(0.0 if i % 4 < 2 else math.ldexp(1.0, lg(x, y) - 150), float(rng.choice([-1.0, 1.0])))
        ties.append((f64_bits(x), f64_bits(y), f64_bits(z)))
    rows["mul_ties"] = tuple(np.array(v, U64) for v in zip(*ties))
    # sums on a tie: a * b = +-half an ulp of c (a a power of 2) times 1 - 2^-53, 1 or 1 + 2^-52
    ec = rng.integers(200, 1901, n)
    one = np.array([0x3FEFFFFFFFFFFFFF, 0x3FF0000000000000, 0x3FF0000000000001], U64)
    rows["sum_ties"] = (sign(n) | ((ec - 53).astype(U64) << U64(52)), rng.choice(one, n), doubles(rng, ec))
    # subnormal results: products at biased exponents -60 to 2, and subnormal a times 1 to 2^60; c +-0, subnormal,
    # in the two lowest binades, or near the product
    t, ea = rng.integers(-60, 3, 2 * n), rng.integers(1, 961, 2 * n)
    a = np.where(k % 4 == 3, doubles(rng, np.zeros(2 * n, int)), doubles(rng, ea))
    b = np.where(k % 4 == 3, doubles(rng, rng.integers(1023, 1084, 2 * n)), doubles(rng, t + 1023 - ea))
    c = np.select([k % 4 == 0, k % 4 == 1, k % 4 == 2],
                  [sign(2 * n), doubles(rng, np.zeros(2 * n, int)), doubles(rng, rng.integers(1, 3, 2 * n))],
                  doubles(rng, np.clip(t + rng.integers(-2, 3, 2 * n), 0, 2046)))
    rows["subnormal"] = (a, b, c)
    # ties on the subnormal grid: x = +-(2q + 1) * 2^(s - 1075), q of every bit length 0 to 52 (the denormalization
    # shifts 53 down to 1), exactly or one ulp of x off; times 2^-s and over 2^s, c = +-0
    s = rng.integers(1, 1001, 2 * n)
    q = [0 if j == 0 else int(rng.integers(1 << (j - 1), 1 << j)) for j in (k % 53).tolist()]
    x = np.array([f64_bits(math.ldexp(2 * qi + 1, si - 1075)) for qi, si in zip(q, s.tolist())], U64)
    x = offset(x, rng.choice([-1, 0, 0, 1], 2 * n)) | sign(2 * n)
    rows["sub_ties_mul"] = (x, ((1023 - s).astype(U64) << U64(52)), sign(2 * n))
    rows["sub_ties_div"] = (x, ((1023 + s).astype(U64) << U64(52)), sign(2 * n))
    # overflow: products and quotients at biased exponents 2044 to 2048, c near the top, of either sign
    t = rng.integers(2044, 2049, n)
    ea = rng.integers(1100, 2047, n)
    rows["overflow"] = (doubles(rng, ea), doubles(rng, t + 1023 - ea), doubles(rng, rng.integers(2040, 2047, n)))
    ea = rng.integers(1026, 2047, n)
    rows["overflow_div"] = (doubles(rng, ea), doubles(rng, ea + 1023 - t), doubles(rng, rng.integers(2040, 2047, n)))
    # quotients near a tie: Ma * 2^54 = K * Mb + d (Mb odd, 0 < |d| <= 3) puts Ma / Mb within 3 / Mb of K * 2^-54, a
    # midpoint for K odd (the quotient below 1; above it, 2^53 and Ma + Mb); a quarter of them subnormal
    near = []
    for i in range(2 * n):
        d = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        while True:
            mb = odd53(rng)
            ma = d * pow(2, -54 if i % 2 else -53, mb) % mb + (0 if i % 2 else mb)
            if (1 << 52) <= ma < (1 << 53):
                break
        ea = int(rng.integers(623, 1424)) if i % 4 != 3 else int(rng.integers(1, 901))
        eb = int(rng.integers(623, 1424)) if i % 4 != 3 else ea + 1023 - int(rng.integers(-60, 1))
        near.append((dbl(ma, ea, i % 8 < 4), dbl(mb, eb, i % 3 == 0), 0))
    rows["div_ties"] = tuple(np.array(v, U64) for v in zip(*near))
    sp = np.r_[SPECIALS, SPECIALS | SIGN]
    rows["specials"] = tuple(v.ravel() for v in np.meshgrid(sp, sp, sp, indexing="ij"))
    return rows


def test_basalt_f64_mul_fma_div_vs_host():
    rng = np.random.default_rng(22)
    rows = h2_rows(rng)
    a, b, c = (np.concatenate([v[j] for v in rows.values()]) for j in range(3))
    got = run_kernel(H2, "h2_f64", [a, b, c], 4)
    A, B, C = a.view(F64), b.view(F64), c.view(F64)
    with np.errstate(all="ignore"):
        mul, div = canon(A * B), canon(A / B)
    ref = []
    for x, y, z in zip(A.tolist(), B.tolist(), C.tolist()):
        fin = math.isfinite(x) and math.isfinite(y) and x != 0 and y != 0
        try:
            host = f64_bits(math.fma(x, y, z))
        except OverflowError:
            host = INF_BITS                                             # its sign is not the host's to say
        except ValueError:
            host = NAN_BITS
        ref.append((*ref_fma(x, y, z), *ref_fma(x, y, z, True), *ref_fma(x, y, -0.0),
                    *(ref_div(x, y) if fin else (0, SPECIAL)), host))
    fma, fcase, odd, _, mref, mcase, dref, dcase, host = (np.array(v, object) for v in zip(*ref))
    fma, odd, mref, dref, host = (v.astype(U64) for v in (fma, odd, mref, dref, host))
    fcase, mcase, dcase = (v.astype(int) for v in (fcase, mcase, dcase))
    # the reference rounds to nearest even as the host does
    assert np.array_equal(mref, mul), "round64's products differ from the host's"
    fin = dcase != SPECIAL
    assert np.array_equal(dref[fin], div[fin]), "round64's quotients differ from the host's"
    over = host == INF_BITS
    assert np.array_equal(fma[~over], host[~over]) and np.all((fma[over] & MAG) == U64(INF_BITS)), \
        "the fma reference differs from math.fma"
    for col, name, want in [(0, "f64_mul", mul), (1, "f64_fma", fma), (2, "f64_fma_odd", odd), (3, "f64_div", div)]:
        bad = np.flatnonzero(got[:, col] != want)
        assert bad.size == 0, (f"{name}: {bad.size} of {len(a)} differ; first a={a[bad[0]]:#018x} b={b[bad[0]]:#018x} "
                               f"c={c[bad[0]]:#018x}: got {got[bad[0], col]:#x}, want {want[bad[0]]:#x}")
    # the planted classes are what they claim
    at, n0 = {}, 0
    for name, (x, _, _) in rows.items():
        at[name], n0 = slice(n0, n0 + len(x)), n0 + len(x)
    ok = lambda v: math.isfinite(v) and v != 0
    gaps = {lg(x, y) - lg(z) for x, y, z in zip(A[at["gaps"]].tolist(), B[at["gaps"]].tolist(),
                                                C[at["gaps"]].tolist()) if ok(x) and ok(y) and ok(z)}
    assert {0, 1, 2, 63, 64, 65, 127, 128, 129} <= {abs(g) for g in gaps if g in gaps and -g in gaps}
    assert max(abs(g) for g in gaps) > 129
    d = at["deep"]
    drop = [lg(x, y) - lg(r) for x, y, r in zip(A[d].tolist(), B[d].tolist(), fma[d].view(F64).tolist())]
    assert sum(v > 64 for v in drop) > 1000 and max(drop) > 100                       # the low word's normalization
    z = at["zeros"]
    assert np.all(fma[z] == 0) and np.all(odd[z] == 0) and np.all(got[z, 1:3] == 0)  # exact cancellation: +0
    for case, name in [(mcase, "mul_ties"), (mcase, "sub_ties_mul"), (dcase, "sub_ties_div"), (fcase, "sum_ties")]:
        v = case[at[name]]
        assert np.count_nonzero(v == DOWN) > 100 and np.count_nonzero(v == UP) > 100, name   # ties, either way
    for case, name in [(mcase, "mul_ties"), (fcase, "mul_ties"), (fcase, "sum_ties"), (dcase, "div_ties"),
                       (mcase, "sub_ties_mul"), (dcase, "sub_ties_div")]:
        assert np.count_nonzero(case[at[name]] == NEAR) > 500, name                  # near ties, off them
    for col in range(4):
        assert np.count_nonzero(subnormal(got[:, col])) > 1000, col                  # subnormal results
    exact_nonzero = (fcase != SPECIAL)
    assert np.count_nonzero(exact_nonzero & ((fma & MAG) == 0) & ((odd & MAG) == 1)) > 100   # tiny: 0 or the least
    assert np.count_nonzero(fcase[:] == INEXACT) > 10000 and np.count_nonzero(odd != fma) > 10000
    fin3 = np.isfinite(A) & np.isfinite(B) & np.isfinite(C)
    for v in (mul, fma, div):
        assert np.count_nonzero(fin3 & ((v & MAG) == U64(INF_BITS))) > 100                # overflow to inf
    assert np.count_nonzero(fin3 & ((odd & MAG) == U64(MAX_BITS)) & ((fma & MAG) == U64(INF_BITS))) > 100
