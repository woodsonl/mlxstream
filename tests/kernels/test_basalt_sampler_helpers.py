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
  on the subnormal grid at every denormalization shift), results decided by a sticky bit alone (a carry into the next
  binade, and products whose lowest bits only the alignment's sticky bit sees), subnormal results, overflow (to inf,
  and to DBL_MAX when rounded to odd) and every triple of specials.
- H.3: the binary32 operations with IEEE subnormals: ``ieee_add`` and ``ieee_mul`` (basalt_ieee.metal), ``ieee_div``
  and ``ieee_fma_count`` (basalt_sampler.metal), against each exact value as a ``Fraction`` rounded once to nearest
  even (``round32``, checked against the host's float32 add, multiply and divide and basalt_emu's fma), on uniform
  patterns and planted classes, more than half of the rows with a subnormal operand or result: every exponent field,
  subnormals and the lowest binades, products and quotients at the bottom of the range, a subnormal c that moves a
  normal fma, exact results just below 2^-126 that round up to it (which the GPU flushes) and the exact pairs at
  2^-126 and above FLT_MAX, ties and near ties (the subnormal grid at every shift, quotients on its ties by divisors
  that are not powers of two, quotients and fmas near a midpoint
  just above the hardware path's 2^-125 bound, with divisors of 2^126 and products below 2^-142), fma gaps,
  cancellation and exact zeros, products beyond FLT_MAX beside a large or an infinite c, fmas that rounding to
  nearest twice (binary64, then binary32) gets wrong, on both paths, overflow and every triple of specials. A NaN
  result must be NaN, and 0x7fffffff where an operand is zero or subnormal (the software route).
- H.4: ``nv_exp`` and ``nv_logf`` (CUDA's binary64 exp and logf) against the reference module's transcriptions
  (tests/basalt_sampler_ref.py), on uniform bit patterns and planted classes: the tail's arguments, both range
  thresholds of exp and the split scale's subnormal and overflowing results, tiny arguments, arguments near a midpoint
  of the reduction's rounding, logf's subnormal, negative, above-1 and near-1 arguments, the edges of its binade split
  and the specials. A NaN result must be NaN: ``F64_NAN`` for ``nv_exp``; ``nv_logf``'s carries the hardware fma's
  payload. The references must also differ from ``math.exp`` and float32 ``math.log`` somewhere, by at most an ulp.
- H.5: ``philox_uniform`` against the reference module's transcription of basalt's host Philox
  (sampler_parity.cpp:134-158), on uniform (seed, counter) words, seeds above and below 2^32, counters near 2^64
  (where a window's counter + t wraps) and about 2^32, position-like counters and every pair of edge words. Each row
  also runs with the seed's high word changed and with the counter's, which must change the draw: c3 and c1 are used.
- H.6: the keys and compares on bits, ``okey``, ``fnan``, ``f_gt``, ``f_lt`` and ``fmaxf_bits``, against numpy's
  float32 (IEEE order, subnormals exact, NaN unordered, -0 equal to +0): the keys order every pair, and sort every
  value, as numpy does; ``fmaxf_bits`` against ``std::fmax`` as the host references use it (sampler_parity.cpp:56: the
  other operand when one is NaN, else the larger, the first on a tie), with 0x7fffffff for two NaNs. On uniform
  patterns, subnormals and the lowest binades, adjacent, opposite and equal values and every pair of specials, many of
  them ordered differently by the GPU's own compares (subnormals read as zero). Also ``expf_approx(-g)`` against
  basalt_fp.metal's ``expf_neg(g)``, bit for bit (NaN by NaN-ness): one operation, equal even where ``ieee_mul`` keeps
  a subnormal product that the plain multiply flushes (``fast::exp2`` gives 1.0 for both).
"""

import math
import struct
from fractions import Fraction

import numpy as np

from monolith import kernels
from tests.basalt_sampler_ref import nv_exp, nv_logf, philox_uniform, u2d
from tests.kernels import basalt_emu as emu
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


# 53-bit significand pairs whose product is H * 2^76 + L (RUN) or H * 2^76 + 2^75 + L (TIE), 1 <= L <= 3: a run of
# 74 zero bits (73 and a one for TIE) above the two lowest. Found as Mb = L / Ma mod 2^76 (Ma odd) below 2^53;
# test_basalt_f64_mul_fma_div_vs_host checks the claim.
RUN = [(0x1B5D11E09CEFBB, 0x101FC2C5367D73), (0x110B418C767F71, 0x1320303030C322), (0x19C09973F61FAD, 0x118A3A1D41384A),
       (0x13E35659E33BDF, 0x1E455985F4C05D), (0x19C09973F61FAD, 0x1A4F572BE1D46F), (0x11186E6E05622B, 0x136835134E4C83),
       (0x1B36C87C818151, 0x14F1035A1CCF62), (0x13527DA0291695, 0x177A95C820097A)]
TIE = [(0x1A6409955206D9, 0x1B1592CC6032D2), (0x1CD41FF62F6D49, 0x1F00B42B5A29F2), (0x16B2537DF2BCD9, 0x12F129B99406D2),
       (0x110B418C767F71, 0x1CB048484924B3), (0x1DB47C8A1C3901, 0x1E428D9194C701), (0x1DA4C9E7C0BB91, 0x1B948F829D6571),
       (0x1B36C87C818151, 0x1F6985072B3713), (0x1E6188A3E3CDB5, 0x11CB0AFC26789D)]


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
    # a carry into the next binade with only a sticky bit below: Ma = 2^52 + u and Mb = 2^53 - 2u give a * b = 2^105 -
    # 2u^2 units, and c is 2u^2 units plus g of its ulps, which lie below the 128-bit grid's bit 1: the exact sum is
    # 2^105 units plus a part that only the jam of the carry's one-bit shift keeps (rounded to odd: 2^105 units + ulp)
    carry = []
    for i in range(n):
        u = int(rng.integers(1, 1 << 12))
        ea, eb, sa, sb = int(rng.integers(823, 1224)), int(rng.integers(823, 1224)), i % 2 == 1, i % 4 > 1
        c0 = f64_bits(math.ldexp(2 * u * u, ea + eb - 2150)) | (int(sa != sb) << 63)
        g = int(rng.integers(1, 1 << (32 - (2 * u * u).bit_length())))
        carry.append((dbl((1 << 52) + u, ea, sa), dbl((1 << 53) - 2 * u, eb, sb), c0 + g))
    rows["carry"] = tuple(np.array(v, U64) for v in zip(*carry))
    # RUN and TIE products, c larger with its ulp at 2^76 product units, added or subtracted: the two lowest bits lie
    # past the alignment shift, and its sticky bit alone decides f64_fma_odd (RUN) or f64_fma's tie (TIE); c's
    # significand Mc makes Mc +- H even, so the result is not the one the sticky bit's loss would give
    for name, pairs in (("run", RUN), ("tie", TIE)):
        trip = []
        for i in range(64 * len(pairs)):
            ma, mb = pairs[i // 2 % len(pairs)]
            ea, eb, sa, sb = int(rng.integers(823, 1224)), int(rng.integers(823, 1224)), i % 3 == 1, i % 5 > 2
            mc = int(rng.integers((1 << 52) + (1 << 31), (1 << 53) - (1 << 31)))
            mc += (mc + ((ma * mb) >> 76)) & 1
            trip.append((dbl(ma, ea, sa), dbl(mb, eb, sb), dbl(mc, ea + eb - 999, (sa != sb) != (i % 2 == 1))))
        rows[name] = tuple(np.array(v, U64) for v in zip(*trip))
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
    cy = at["carry"]
    assert np.all((fma[cy] & FRAC) == 0) and np.array_equal(odd[cy], fma[cy] + U64(1))   # 2^k, and 2^k plus an ulp
    assert {(x * y) % (1 << 76) for x, y in RUN} <= {1, 2, 3}
    assert {(x * y) % (1 << 76) - (1 << 75) for x, y in TIE} <= {1, 2, 3}
    r, t = at["run"], at["tie"]
    step = np.abs((odd[r] & MAG).view(I64) - (fma[r] & MAG).view(I64))
    assert np.all((fma[r] & U64(1)) == 0) and np.all(step == 1)   # inexact only by the sticky bit: odd is fma +- 1
    assert np.all((fma[t] & U64(1)) == 1) and np.array_equal(fma[t], odd[t])   # a tie only the sticky bit breaks
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


# ---- H.3: binary32 add, multiply, divide and fma with IEEE subnormals ----------------------------------------------

H3 = """
kernel void h3_ieee(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                    device const uint* c [[buffer(2)]], device uint* out [[buffer(3)]],
                    constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float x = as_type<float>(a[i]), y = as_type<float>(b[i]), z = as_type<float>(c[i]);
  device uint* o = out + 4 * i;
  o[0] = as_type<uint>(ieee_add(x, y));
  o[1] = as_type<uint>(ieee_mul(x, y));
  o[2] = as_type<uint>(ieee_div(x, y));
  o[3] = as_type<uint>(ieee_fma_count(x, y, z));
}
"""

INF32, SIGN32 = 0x7F800000, 1 << 31


def round32(q):
    """The nonzero Fraction q rounded once to binary32: to nearest even on the grid of its binade (2^-149 below 2^-126,
    gradual underflow), infinity from 2^128 - 2^103 up. Returns the bits and round64's case (NEAR: within 2^-21 of a
    half unit, off it)."""
    n, d = abs(q.numerator), q.denominator
    e = n.bit_length() - d.bit_length()
    if (n << max(0, -e)) < (d << max(0, e)):
        e -= 1                                                          # 2^e <= |q| < 2^(e + 1)
    k = max(e, -126) - 23                                               # the grid 2^k
    den = d << max(0, k)
    m, r = divmod(n << max(0, -k), den)
    case = (EXACT if r == 0 else (UP if m & 1 else DOWN) if 2 * r == den
            else NEAR if abs(2 * r - den) << 20 < den else INEXACT)
    m += 2 * r > den or case == UP
    return (int(q < 0) << 31) | min(((k + 149) << 23) + m, INF32), case  # a carry moves the exponent up


def neg(v):
    return math.copysign(1.0, v) < 0


def ref_ieee(x, y, z):
    """IEEE binary32 x + y, x * y, x / y and fma(x, y, z) on float32 values (Python floats): each exact value a
    Fraction, rounded once (round32); the specials and the signs of an exact zero by IEEE 754's rules. Returns four
    (bits, case) pairs; a NaN's bits are None."""
    F, nan, inf = Fraction, math.isnan, math.isinf
    sign, nan_ = (neg(x) != neg(y)) << 31, (None, SPECIAL)
    if nan(x) or nan(y) or (inf(x) and inf(y) and x != y):
        add = nan_
    elif inf(x) or inf(y):
        add = (f32_bits(x if inf(x) else y), SPECIAL)
    else:
        s = F(x) + F(y)
        add = round32(s) if s else ((neg(x) and neg(y)) << 31, SPECIAL)   # -0 only from -0 + -0
    if nan(x) or nan(y) or (inf(x) and y == 0) or (inf(y) and x == 0):
        mul = nan_
    elif inf(x) or inf(y) or x == 0 or y == 0:
        mul = (sign | (INF32 if inf(x) or inf(y) else 0), SPECIAL)
    else:
        mul = round32(F(x) * F(y))
    if nan(x) or nan(y) or (x == 0 and y == 0) or (inf(x) and inf(y)):
        div = nan_
    elif inf(x) or y == 0 or inf(y) or x == 0:
        div = (sign | (INF32 if inf(x) or y == 0 else 0), SPECIAL)
    else:
        div = round32(F(x) / F(y))
    if nan(x) or nan(y) or nan(z) or ((inf(x) or inf(y)) and (x == 0 or y == 0 or (inf(z) and neg(z) != bool(sign)))):
        fma = nan_
    elif inf(x) or inf(y):
        fma = (sign | INF32, SPECIAL)
    elif inf(z):
        fma = (f32_bits(z), SPECIAL)
    else:
        v = F(x) * F(y) + F(z)
        fma = round32(v) if v else ((x * y == 0 and bool(sign) and neg(z)) << 31, SPECIAL)   # -0: -0 + -0 only
    return add, mul, div, fma


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def floats(rng, exp):
    """Random signs and fractions on the biased exponents ``exp`` (float32 bits)."""
    return (words(rng, len(exp), U32) & U32(0x807FFFFF)) | (np.asarray(exp).astype(U32) << U32(23))


def f32(v):
    """float64 values as float32 bits, rounded to nearest even."""
    return np.asarray(v, F64).astype(F32).view(U32)


def inv32(a):
    """a^-1 mod 2^32 for odd uint64 a < 2^32 (Newton: each step doubles the correct low bits, 3 to 48)."""
    m, x = U64(0xFFFFFFFF), a.copy()
    for _ in range(4):
        x = (x * ((U64(2) - (a * x & m)) & m)) & m
    return x


# 2^25 - 1 = 31 * 601 * 1801 and 2^24 - 1 = 4095 * 4097: products at 2^-126 - 2^-151 (rounds up to 2^-126), at the
# tie 2^-126 - 2^-150 (to even: 2^-126) and at 2^128 - 2^103 (the tie above FLT_MAX: to even, infinity)
EDGE_PAIRS = [(31 * 2.0 ** -70, 1082401 * 2.0 ** -81), (601 * 2.0 ** -75, 55831 * 2.0 ** -76),
              (1801 * 2.0 ** -75, 18631 * 2.0 ** -76), (4095 * 2.0 ** -75, 4097 * 2.0 ** -75),
              (18631 * 2.0 ** 50, 1801 * 2.0 ** 53), (601 * 2.0 ** 52, 55831 * 2.0 ** 51)]
SPECIALS32 = np.array([0, 1, 0x400000, 0x7FFFFF, 0x800000, 0x1000000, 0x3F800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001,
                       0x7FC00000], U32)   # 0, subnormals, 2^-126, 2^-125, 1, FLT_MAX, inf, NaNs


def offset32(x, k):
    """float32 bits x with the magnitude moved by k units in the last place, clamped to [0, FLT_MAX]; x's sign."""
    mag = np.clip((x & U32(0x7FFFFFFF)).astype(I64) + np.asarray(k, I64), 0, 0x7F7FFFFF)
    return mag.astype(U32) | (x & U32(SIGN32))


def pow2(e):
    """2^e (integer e within the normal range) as float32 bits."""
    return (np.asarray(e) + 127).astype(U32) << U32(23)


def h3_rows(rng):
    """The H.3 operand triples (float32 bit patterns) by class."""
    n = 1 << 12
    k = np.arange(2 * n)
    sign = lambda m: words(rng, m, U32) & U32(SIGN32)
    rows = {"uniform": tuple(words(rng, n, U32) for _ in range(3))}
    # exponent fields 0 to 255 for a and b (subnormal, normal, inf, NaN), c within 30 binades of the product
    ea, eb = rng.integers(0, 256, 2 * n), rng.integers(0, 256, 2 * n)
    rows["exponents"] = (floats(rng, ea), floats(rng, eb),
                         floats(rng, np.clip(ea + eb - 127 + rng.integers(-30, 31, 2 * n), 0, 255)))
    # subnormals (half of the operands) and the two lowest binades, every sign: sums across 2^-126, differences below
    # it, quotients of them
    rows["subnormal"] = tuple(floats(rng, rng.choice([0, 0, 1, 2], 2 * n)) for _ in range(3))
    # products (even rows) and quotients (odd rows) at biased exponents -25 to 3: subnormal, or about 2^-126; c +-0,
    # subnormal, or near the product
    k4 = np.arange(4 * n)
    t, ea = rng.integers(-25, 4, 4 * n), rng.integers(30, 101, 4 * n)
    c = np.select([k4 % 3 == 0, k4 % 3 == 1], [sign(4 * n), floats(rng, np.zeros(4 * n, int))],
                  floats(rng, np.clip(t + rng.integers(-2, 3, 4 * n), 0, 254)))
    rows["sub_results"] = (floats(rng, ea), floats(rng, np.where(k4 % 2 == 0, t + 127 - ea, ea + 127 - t)), c)
    # a subnormal c beside a product of 2^-125 to 2^-106, where it changes a normal fma; a subnormal a (or b) times
    # 2^-60 to 2^60 and divided by it (normal products and quotients of a subnormal), c in the low binades
    t, ea = rng.integers(2, 22, 2 * n), rng.integers(30, 98, 2 * n)
    rows["sub_c"] = (floats(rng, ea), floats(rng, t + 127 - ea), floats(rng, np.zeros(2 * n, int)))
    s, x = floats(rng, np.zeros(2 * n, int)), floats(rng, rng.integers(67, 188, 2 * n))
    rows["sub_ab"] = (np.where(k % 2 == 0, s, x), np.where(k % 2 == 0, x, s), floats(rng, rng.integers(1, 64, 2 * n)))
    # results within about 2^-150 of +-2^-126 or +-2^-125: b = RN(target / a) (c = 0, or b = RN((target - c) / a) for
    # c = +-j 2^-126) and a = RN(target * b) in every 4th row; the exact pairs EDGE_PAIRS (2^-126 and FLT_MAX's edges)
    # and the tie 2^-126 - 2^-150 as +-(2^24 - 1) 2^E / 2^(E + 150), every E: no other quotient of floats lies in
    # [2^-126 - 2^-150, 2^-126), 2^-150 being 2^-24 of it
    target = np.where(k % 3 == 0, 2.0 ** -125, 2.0 ** -126) * (1 + rng.integers(-40, 41, 2 * n) * 2.0 ** -26)
    target *= np.where(k % 2 == 0, 1.0, -1.0)
    quot = k % 4 == 3
    x = floats(rng, np.where(quot, rng.integers(128, 178, 2 * n), rng.integers(77, 127, 2 * n))).view(F32).astype(F64)
    cz = np.where(k % 4 < 2, 0.0, rng.integers(-3, 4, 2 * n) * 2.0 ** -126)
    p = np.array(EDGE_PAIRS)
    sg = np.repeat([1.0, -1.0], len(p))
    e = np.arange(-149, -22)
    rows["edges"] = (np.r_[np.where(quot, f32(target * x), f32(x)), f32(np.r_[p[:, 0], p[:, 0]] * sg),
                           f32(np.ldexp(2.0 ** 24 - 1, np.r_[e, e])) ^ np.repeat([U32(0), U32(SIGN32)], len(e))],
                     np.r_[np.where(quot, f32(x), f32((target - cz) / x)), f32(np.r_[p[:, 1], p[:, 1]]),
                           pow2(np.r_[e, e] + 150)],
                     np.r_[np.where(quot, 0, f32(cz)), np.zeros(2 * len(p) + 2 * len(e), U32)])
    # ties, exactly or (some) one ulp off: products of 13-bit significands (25 or 26 bits); on the subnormal grid,
    # 2^p * (2j + 1) 2^(-150 - p) and (2j + 1) 2^E / 2^(E + 150), j of every bit length 0 to 22 (every denormalization
    # shift); sums with b = +-half an ulp of a; fmas with a * b = +-half an ulp of c (2^-150 for a subnormal c), or
    # 1 + 2^-23 times it
    j = np.array([0 if v == 0 else int(rng.integers(1 << (v - 1), 1 << v)) for v in (k[:n] % 23).tolist()])
    odd, pw, e = (2 * j + 1).astype(F64), rng.integers(-60, -23, n), rng.integers(-126, -23, n)
    sig13 = lambda m: (((rng.integers(1 << 12, 1 << 13, m) - (1 << 12)) << 11).astype(U32)
                       | (rng.integers(64, 190, m).astype(U32) << U32(23)))
    ea = rng.integers(64, 190, n)
    a = floats(rng, ea)
    ec = rng.integers(0, 254, n)
    h = np.maximum(ec, 1) - 151                                          # log2 of half an ulp of c
    pa = h // 2 + rng.integers(-10, 11, n)
    rows["ties"] = (
        np.r_[sign(n) | sig13(n), pow2(pw), offset32(f32(odd * np.ldexp(1.0, e)), rng.choice([-1, 0, 0, 1], n)), a,
              pow2(pa)],
        np.r_[sig13(n), f32(odd * np.ldexp(1.0, -150 - pw)), pow2(e + 150),
              offset32(pow2(ea - 151) | sign(n), rng.choice([-1, 0, 0, 1], n)),
              f32(np.ldexp(np.where(k[:n] % 2 == 0, 1.0, 1 + 2.0 ** -23), h - pa)) | sign(n)],
        np.r_[words(rng, 4 * n, U32), floats(rng, ec)])
    # quotients on the subnormal grid's ties by divisors that are not powers of two: +-(2j + 1) M 2^E / (M 2^(E + 150))
    # = (2j + 1) 2^-150, M odd from 3 to 4095, (2j + 1) M below 2^24 (1 / b is inexact in binary64)
    mo = rng.integers(1, 1 << 11, n) * 2 + 1
    odd = 2 * np.floor(rng.random(n) * ((2 ** 24 // mo - 1) // 2)) + 1
    e = rng.integers(-149, -35, n)
    rows["div_ties"] = (f32(odd * mo * np.ldexp(1.0, e)) | sign(n), f32(mo * np.ldexp(1.0, e + 150)) | sign(n),
                        np.zeros(n, U32))
    # quotients near a midpoint: Ma * 2^24 (2^25 for Ma < Mb) = K * Mb + d, 0 < |d| <= 3, Mb odd: at divisors of
    # 2^126 and above with quotients of 2^-125 to 2^-119 (the guard's edge), and at random exponents
    near = []
    for i in range(n):
        d = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        while True:
            mb = int(rng.integers(1 << 23, 1 << 24)) | 1
            ma = d * pow(2, -25 if i % 2 else -24, mb) % mb + (0 if i % 2 else mb)
            if (1 << 23) <= ma < (1 << 24):
                break
        if i % 4 < 2:
            eb = int(rng.integers(253, 255))
            ea = int(rng.integers(2, 8)) + eb - 127 + i % 2                 # the quotient's biased exponent: 2 to 7
        else:
            ea, eb = int(rng.integers(64, 191)), int(rng.integers(64, 191))
        near.append((int(i % 8 < 4) << 31 | ea << 23 | (ma - (1 << 23)),
                     int(i % 3 == 0) << 31 | eb << 23 | (mb - (1 << 23))))
    na, nb = (np.array(v, U32) for v in zip(*near))
    # fmas near a midpoint at c of 2^-125 to 2^-118, beside a product below 2^-142: a power of two a times b, a * b half
    # an ulp of c times 1 - j 2^-24, 1 or 1 + j 2^-23 (|j| <= 4)
    ec, pa = rng.integers(2, 9, n), rng.integers(-60, -30, n)
    mm = 1 + rng.choice([-4, -3, -2, -1, 0, 0, 0, 1, 2, 3, 4], n) * np.where(k[:n] % 2 == 0, 2.0 ** -24, 2.0 ** -23)
    rows["near"] = (np.r_[na, pow2(pa) | sign(n)], np.r_[nb, f32(np.ldexp(mm, ec - 151 - pa))],
                    np.r_[np.zeros(n, U32), floats(rng, ec)])
    # the fma: gaps of -50 to 50 binades between the product and c; c = -RN(a * b) moved by up to 4 ulps (the
    # product's rounding error for every 8th: subnormal for products below about 2^-102); exact zeros (12-bit
    # significands, c = -a * b); products of 2^128 to 2^130 beside a c of the other sign in the top two binades, or
    # an infinite c (every 4th)
    ea, eb = rng.integers(1, 255, n), rng.integers(1, 255, n)
    rows["fma_gaps"] = (floats(rng, ea), floats(rng, eb),
                        floats(rng, np.clip(ea + eb - 127 + rng.integers(-50, 51, n), 0, 254)))
    ea = rng.integers(1, 128, n)
    a = floats(rng, ea)
    b = floats(rng, np.clip(np.where(k[:n] % 2 == 0, rng.integers(128, 160, n), rng.integers(2, 60, n)) + 127 - ea,
                            1, 254))
    pr = f32(a.view(F32).astype(F64) * b.view(F32).astype(F64))
    rows["cancel"] = (a, b, offset32(pr, np.where(k[:n] % 8 == 0, 0, rng.integers(-4, 5, n))) ^ U32(SIGN32))
    sig12 = lambda m: sign(m) | (((rng.integers(1 << 11, 1 << 12, m) - (1 << 11)) << 12).astype(U32)) \
        | (rng.integers(64, 191, m).astype(U32) << U32(23))
    a, b = sig12(n), sig12(n)
    rows["zeros"] = (a, b, f32(a.view(F32).astype(F64) * b.view(F32).astype(F64)) ^ U32(SIGN32))
    ea = rng.integers(128, 255, n)
    a, b = floats(rng, ea), floats(rng, np.clip(rng.integers(255, 257, n) + 127 - ea, 1, 254))
    big = (~(a ^ b) & U32(SIGN32)) | (rng.integers(253, 255, n).astype(U32) << U32(23)) | (words(rng, n, U32)
                                                                                        & U32(0x7FFFFF))
    rows["fma_over"] = (a, b, np.where(k[:n] % 4 == 0, sign(n) | U32(INF32), big))
    # double rounding: Ma * Mb = 2^31 + D mod 2^32 (D = +-1 to 3), c's ulp 2^32 units of the product, so a * b + c is a
    # binary32 midpoint plus D / 8 of a binary64 ulp: rounding to nearest twice gives the even neighbour, the wrong one
    # for about half of the rows. c's exponent field 0 (c above 2^-127, D = +-1) and 1 take the software route
    m = rng.integers(1 << 22, 1 << 23, 1 << 21).astype(U64) * U64(2) + U64(1)
    dd = rng.choice(np.array([-3, -2, -1, 1, 2, 3]), len(m))
    ec = np.select([np.arange(len(m)) % 3 == 0, np.arange(len(m)) % 3 == 1], [0, 1], rng.integers(2, 254, len(m)))
    dd = np.where(ec == 0, np.sign(dd), dd)
    mb = (((U64(1 << 31) + dd.astype(I64).view(U64)) & U64(0xFFFFFFFF)) * inv32(m)) & U64(0xFFFFFFFF)
    ok = (mb >= U64(1 << 23)) & (mb < U64(1 << 24))
    nt_ = n // 2
    ma, mb, ec = m[ok][:nt_], mb[ok][:nt_], ec[ok][:nt_]
    assert len(ma) == nt_
    s = np.maximum(ec, 1) - 182                                          # the product's last place: 2^s
    sa = s // 2
    cf = np.where(ec == 0, rng.integers(1 << 22, (1 << 23) - (1 << 16), nt_), rng.integers(0, 1 << 23, nt_))
    rows["traps"] = (sign(nt_) | ((sa + 150).astype(U32) << U32(23)) | (ma.astype(U32) - U32(1 << 23)),
                     sign(nt_) | ((s - sa + 150).astype(U32) << U32(23)) | (mb.astype(U32) - U32(1 << 23)),
                     sign(nt_) | (ec.astype(U32) << U32(23)) | cf.astype(U32))
    # results about FLT_MAX: products (even rows) and quotients (odd rows) at biased exponents 252 to 256, sums with a
    # b in the top binades (every 3rd), and FLT_MAX plus 2^103 (half its ulp: the tie, to infinity), plus the float
    # below 2^103, and plus FLT_MAX
    ea, t = rng.integers(128, 255, n), rng.integers(252, 257, n)
    b = np.where(k[:n] % 3 == 0, floats(rng, rng.integers(252, 255, n)),
                 floats(rng, np.clip(np.where(k[:n] % 2 == 0, t + 127 - ea, ea + 127 - t), 1, 254)))
    rows["overflow"] = (np.r_[floats(rng, ea), np.full(3, 0x7F7FFFFF, U32)],
                        np.r_[b, np.array([0x73000000, 0x72FFFFFF, 0x7F7FFFFF], U32)],
                        np.r_[floats(rng, rng.integers(250, 255, n)), np.zeros(3, U32)])
    sp = np.r_[SPECIALS32, SPECIALS32 | U32(SIGN32)]
    rows["specials"] = tuple(v.ravel() for v in np.meshgrid(sp, sp, sp, indexing="ij"))
    return rows


def test_basalt_ieee_add_mul_div_fma_vs_fraction():
    rng = np.random.default_rng(23)
    rows = h3_rows(rng)
    a, b, c = (np.concatenate([v[j] for v in rows.values()]) for j in range(3))
    assert a.dtype == b.dtype == c.dtype == U32
    got = run_kernel(H3, "h3_ieee", [a, b, c], 4, U32)
    A, B, C = (v.view(F32) for v in (a, b, c))
    ref = [ref_ieee(x, y, z) for x, y, z in zip(A.tolist(), B.tolist(), C.tolist())]
    want = np.array([[-1 if r[0] is None else r[0] for r in row] for row in ref], I64)       # -1: NaN
    case = np.array([[r[1] for r in row] for row in ref], int)
    nan, mag = want < 0, want & 0x7FFFFFFF
    # the reference rounds as the host's IEEE float32 add, multiply and divide do, and as basalt_emu's fma
    with np.errstate(all="ignore"):
        host = [A + B, A * B, A / B, emu.fma32(A, B, C)]
    for col, h in enumerate(host):
        assert np.array_equal(np.isnan(h), nan[:, col]), col
        assert np.array_equal(h.view(U32)[~nan[:, col]], want[~nan[:, col], col]), col
    for col, name in enumerate(["ieee_add", "ieee_mul", "ieee_div", "ieee_fma_count"]):
        g = got[:, col]
        bad = np.flatnonzero(np.where(nan[:, col], (g & U32(0x7FFFFFFF)) <= U32(INF32), g != want[:, col]))
        assert bad.size == 0, (f"{name}: {bad.size} of {len(a)} differ; first a={a[bad[0]]:#010x} b={b[bad[0]]:#010x} "
                               f"c={c[bad[0]]:#010x}: got {g[bad[0]]:#x}, want {want[bad[0], col]:#x}")
    # a NaN from the software route (a zero or subnormal operand) is 0x7fffffff
    low = lambda x: (x & U32(INF32)) == 0
    for col, sw in ((1, low(a) | low(b)), (2, low(a) | low(b)), (3, low(a) | low(b) | low(c))):
        assert np.count_nonzero(sw & nan[:, col]) > 10 and np.all(got[sw & nan[:, col], col] == U32(0x7FFFFFFF)), col
    # the planted classes are what they claim
    at = slices({name: v[0] for name, v in rows.items()})
    sub = lambda x: low(x) & ((x & U32(0x7FFFFF)) != 0)
    wsub = (mag > 0) & (mag < 0x800000) & ~nan                                         # subnormal results
    assert 2 * np.count_nonzero(sub(a) | sub(b) | sub(c) | wsub.any(axis=1)) >= len(a)   # half subnormal-heavy
    normal = ~nan & (mag >= 0x800000) & (mag < INF32)
    big = nan | (mag >= 0x1000000)                                                     # 2^-125 and above
    hw = {2: ~low(a) & ~low(b) & big[:, 2], 3: ~low(a) & ~low(b) & ~low(c) & big[:, 3]}   # the hardware's results
    for col, fast in hw.items():
        assert np.count_nonzero(fast) > 20000 and np.count_nonzero(~fast & normal[:, col]) > 2000, col
        assert np.count_nonzero(~fast & wsub[:, col]) > 5000, col
    with np.errstate(all="ignore"):
        exact = [None, A.astype(F64) * B, A.astype(F64) / B, A.astype(F64) * B + C]    # float64: classes only
    e = at["edges"]
    for col in (1, 2, 3):                                     # below 2^-126, rounded up to it (the GPU flushes these)
        assert np.count_nonzero((np.abs(exact[col][e]) < 2.0 ** -126) & (mag[e, col] == 0x800000)) > 50, col
    for name, col in [("ties", 0), ("ties", 1), ("ties", 2), ("ties", 3), ("div_ties", 2)]:
        v = case[at[name], col]
        assert np.count_nonzero(v == DOWN) > 100 and np.count_nonzero(v == UP) > 100, (name, col)  # either way
    assert np.count_nonzero(case[at["near"], 2] == NEAR) > 500 and np.count_nonzero(case[at["near"], 3] == NEAR) > 500
    s = at["sub_c"]
    assert np.count_nonzero(normal[s, 3] & (want[s, 3] != want[s, 1])) > 1000         # a subnormal c moves the fma
    s = at["sub_ab"]
    assert np.count_nonzero(normal[s, 1]) > 500 and np.count_nonzero(normal[s, 2]) > 500   # of a subnormal operand
    s = at["fma_over"]
    beyond = np.abs(exact[1][s]) >= 2.0 ** 128
    assert np.count_nonzero(beyond & normal[s, 3]) > 100 and np.all(beyond)
    assert np.count_nonzero(beyond & ((c[s] & U32(0x7FFFFFFF)) == U32(INF32)) & (want[s, 3] == c[s])) > 500
    assert np.count_nonzero(wsub[at["cancel"], 3]) > 100 and np.all(want[at["zeros"], 3] == 0)   # +0 when exact
    # the double-rounding traps: rounding to nearest in binary64 first, then to binary32, differs from the fma
    t = at["traps"]
    twice = []
    for x, y, z in zip(A[t].tolist(), B[t].tolist(), C[t].tolist()):
        v = Fraction(x) * Fraction(y) + Fraction(z)
        bits, _ = round64(v.numerator, 1 - v.denominator.bit_length())
        twice.append(bits)
    twice = np.array(twice, U64).view(F64).astype(F32).view(U32).astype(I64)
    differ = twice != want[t, 3]
    assert np.count_nonzero(differ & ~hw[3][t]) > 300 and np.count_nonzero(differ & hw[3][t]) > 100


# ---- H.4: libdevice's exp and logf ---------------------------------------------------------------------------------

H4 = """
kernel void h4_libdevice(device const ulong* x [[buffer(0)]], device const uint* f [[buffer(1)]],
                         device ulong* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[2 * i] = nv_exp(x[i]);
  out[2 * i + 1] = ulong(as_type<uint>(nv_logf(as_type<float>(f[i]))));
}
"""

# exp's edges, run at both signs: 0, inf, 1, ln 2, the least subnormal, DBL_MIN, DBL_MAX, the range thresholds
# 708.396... (1022 ln 2) and 745, and the |x| whose results are 2^-+1023, the least subnormal and half of it (either
# side), and DBL_MAX (either side)
EXP_EDGES = [0.0, math.inf, 1.0, math.log(2.0), 5e-324, 2.0 ** -1022, 1.7976931348623157e308, 708.3964185322641,
             745.0, 746.0, 709.0895657128241, 744.4400719213812, 745.1332191019411, 745.1332191019412,
             709.782712893384, 709.7827128933841]
EXP_NANS = [0x7FF0000000000001, 0x7FF8000000000000, 0xFFF8000000000000, 0x7FFFFFFFFFFFFFFF, 0xFFF0000000000001]
# logf's edges, run at both signs: 0, subnormals and the normal range's ends, 1 and its neighbours, the binade split
# 0x3F2AAAAB, inf, NaNs, every power of two and all-ones significand among the subnormals, and the fixtures' min_p
LOGF_EDGES = ([0, 1, 2, 0x7FFFFF, 0x800000, 0x800001, 0x3F800000, 0x3F7FFFFF, 0x3F800001, 0x3F2AAAAA, 0x3F2AAAAB,
               0x3F2AAAAC, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]
              + [1 << j for j in range(23)] + [(2 << j) - 1 for j in range(23)]
              + [0x3D4CCCCD, 0x3DCCCCCD, 0x3E4CCCCD, 0x3E99999A, 0x3ECCCCCD, 0x3F000000, 0x3F666666])
ONE = U64(0x3FF0000000000000)


def h4_exp_args(rng):
    """nv_exp's arguments (binary64 bit patterns) by class."""
    n = 1 << 12
    args = {"uniform": words(rng, 1 << 15),                                     # every class of bit pattern
            "tail": rng.uniform(-50.0, 0.0, 2 * n).view(U64),                   # the tail's x - max <= 0
            "underflow": rng.uniform(-746.0, -708.0, n).view(U64),              # the split scale: subnormal results
            "overflow": rng.uniform(708.0, 746.0, 2 * n).view(U64)}             # and overflowing ones
    # high words at each range threshold and either side of it, random low words, either sign
    hw = rng.choice(np.array([0x4086232A, 0x4086232B, 0x4086232C, 0x408747FF, 0x40874800, 0x40874801], U64), n)
    args["thresholds"] = (hw << U64(32)) | (words(rng, n) & U64(0xFFFFFFFF)) | (words(rng, n) & SIGN)
    # every binade below 2^-20, the subnormals included: i is 0 and r is x
    args["tiny"] = doubles(rng, rng.integers(0, 1023 - 20, n))
    # x / ln 2 within 10^-12 of k + 1/2, a midpoint of t's rounding to an integer, for every k of the range
    mid = ((np.arange(-1075, 1024) + 0.5) * math.log(2.0)).view(U64)
    args["midpoints"] = np.concatenate([offset(mid, d) for d in (-2, -1, 0, 1, 2)])
    e = np.array(EXP_EDGES, F64).view(U64)
    args["edges"] = np.r_[e, e | SIGN, np.array(EXP_NANS, U64)]
    return args


def h4_logf_args(rng):
    """nv_logf's arguments (binary32 bit patterns) by class."""
    n = 1 << 12
    k = np.arange(n)
    args = {"uniform": words(rng, 1 << 13, U32),                                # every class of bit pattern
            "tail": rng.uniform(1e-6, 1.0, n).astype(F32).view(U32),           # min_p's range
            "decades": (10.0 ** rng.uniform(-46.0, 38.5, n)).astype(F32).view(U32),   # 0, subnormals, ... 3e38
            # subnormals, a quarter negative
            "subnormal": rng.integers(1, 1 << 23, n).astype(U32) | np.where(k % 4 == 0, U32(1 << 31), U32(0)),
            # 1 + j ulps for |j| <= 2048: f = m - 1 down to 2^-24
            "near_one": (np.int64(0x3F800000) + rng.integers(-2048, 2049, n)).astype(U32)}
    # the significands 0x2AAAAB - 3 to + 3 about the split of [2/3, 4/3) in every binade
    split = (np.arange(1, 255, dtype=U32)[:, None] << U32(23)) | (U32(0x2AAAAB) + np.arange(-3, 4).astype(U32))
    args["split"] = split.ravel()
    e = np.array(LOGF_EDGES, U32)
    args["edges"] = np.r_[e, e | U32(1 << 31)]
    return args


def slices(args):
    """Each class's slice of the concatenated arguments."""
    at, n0 = {}, 0
    for name, v in args.items():
        at[name], n0 = slice(n0, n0 + len(v)), n0 + len(v)
    return at


def test_basalt_nv_exp_nv_logf_vs_ref():
    rng = np.random.default_rng(24)
    xa, fa = h4_exp_args(rng), h4_logf_args(rng)
    x, f = np.concatenate(list(xa.values())), np.concatenate(list(fa.values()))
    assert x.dtype == U64 and f.dtype == U32
    n = max(len(x), len(f))
    got = run_kernel(H4, "h4_libdevice", [np.r_[x, np.zeros(n - len(x), U64)], np.r_[f, np.zeros(n - len(f), U32)]], 2)
    gx, gf = got[: len(x), 0], got[: len(f), 1].astype(U32)
    X, Fl = x.view(F64), f.view(F32)
    wx = canon(np.array([nv_exp(v) for v in X.tolist()], F64))
    wfl = np.array([nv_logf(v) for v in Fl], F32)
    wf = wfl.view(U32)
    bad = np.flatnonzero(gx != wx)
    assert bad.size == 0, (f"nv_exp: {bad.size} of {len(x)} differ; first x={x[bad[0]]:#018x}: got {gx[bad[0]]:#x}, "
                           f"want {wx[bad[0]]:#x}")
    nan_want, nan_got = np.isnan(wfl), (gf & U32(0x7FFFFFFF)) > U32(0x7F800000)
    bad = np.flatnonzero(np.where(nan_want, ~nan_got, gf != wf))
    assert bad.size == 0, (f"nv_logf: {bad.size} of {len(f)} differ; first x={f[bad[0]]:#010x}: got {gf[bad[0]]:#x}, "
                           f"want {wf[bad[0]]:#x}")
    # the planted classes are what they claim
    ax, af = slices(xa), slices(fa)
    hx = (x >> U64(32)) & U64(0x7FFFFFFF)
    assert np.count_nonzero((hx >= 0x4086232B) & (hx < 0x40874800)) > 4000            # the split scale
    assert np.count_nonzero(hx >= 0x40874800) > 1000                                  # 0, inf or NaN outright
    assert np.count_nonzero(subnormal(wx[ax["underflow"]])) > 1000                    # subnormal results
    assert np.count_nonzero(wx[ax["underflow"]] == 0) > 10                            # and underflows to 0
    over = wx[ax["overflow"]]
    assert np.count_nonzero(over == U64(INF_BITS)) > 1000 and np.count_nonzero(over < U64(INF_BITS)) > 100
    assert np.count_nonzero(wx[ax["tiny"]] == ONE) > 1000                             # exp of a tiny x is 1
    assert np.count_nonzero(wx[ax["tiny"]] != ONE) > 50
    c = Fraction(u2d(0x3FF71547652B82FE))                                             # nv_exp's 1 / ln 2
    dev = [float(Fraction(v) * c % 1 - Fraction(1, 2)) for v in X[ax["midpoints"]].tolist()]
    assert max(map(abs, dev)) < 1e-12 and min(dev) < 0 < max(dev)                     # either side of k + 1/2
    e, m = wx[ax["edges"]], len(EXP_EDGES)
    assert [e[0], e[1], e[m], e[m + 1]] == [ONE, INF_BITS, ONE, 0]                    # exp(+-0) = 1, exp(+-inf)
    assert np.all(e[2 * m:] == F64_NAN)
    pos = (f & U32(1 << 31)) == 0
    sub = ((f & U32(0x7F800000)) == 0) & ((f & U32(0x7FFFFF)) != 0)
    assert np.all(np.isfinite(wfl[sub & pos])) and np.count_nonzero(sub & pos) > 3000   # no subnormal is flushed
    neg = ~pos & ((f & U32(0x7FFFFFFF)) != 0)
    assert np.all(np.isnan(wfl[neg])) and np.count_nonzero(neg & sub) > 1000           # log of a negative is NaN
    assert np.all(wfl[(f & U32(0x7FFFFFFF)) == 0] == -np.inf) and np.all(wfl[f == U32(0x7F800000)] == np.inf)
    assert np.count_nonzero(pos & (f > U32(0x3F800000)) & (f < U32(0x7F800000))) > 3000    # above 1
    # CUDA's functions, not the platform's: within an ulp of math.exp and float32 math.log, and not equal to them
    t = ax["tail"]
    lib = np.array([math.exp(v) for v in X[t].tolist()], F64).view(U64)
    d = np.abs(wx[t].view(I64) - lib.view(I64))
    assert d.max() <= 1 and np.count_nonzero(d) > 100, ("exp", d.max(), np.count_nonzero(d))
    t = af["tail"]
    lib = np.array([math.log(v) for v in Fl[t].tolist()], F64).astype(F32).view(U32)
    d = np.abs(wf[t].astype(I64) - lib.astype(I64))
    assert d.max() <= 1 and np.count_nonzero(d) > 100, ("logf", d.max(), np.count_nonzero(d))


# ---- H.5: Philox ---------------------------------------------------------------------------------------------------

H5 = """
kernel void h5_philox(device const ulong* s [[buffer(0)]], device const ulong* c [[buffer(1)]],
                      device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = as_type<uint>(philox_uniform(s[i], c[i]));
}
"""

EDGE_WORDS = np.array([0, 1, (1 << 32) - 1, 1 << 32, (1 << 40) + 123, 1 << 63, (1 << 64) - 2, (1 << 64) - 1], U64)


def h5_rows(rng):
    """The H.5 (seed, counter) pairs (uint64) by class."""
    n = 1 << 12
    hi = lambda m: words(rng, m) | U64(1 << 32)                                  # seeds above 2^32
    rows = {"uniform": (words(rng, 1 << 13), words(rng, 1 << 13)),
            "top": (hi(n), U64((1 << 64) - 1) - rng.integers(0, 64, n).astype(U64)),   # the high word 0xffffffff
            "carry": (hi(n), (np.int64(1 << 32) + rng.integers(-64, 64, n)).astype(U64)),   # about 2^32
            "small": (words(rng, n) & U64(0xFFFFFFFF), np.arange(n, dtype=U64))}       # seeds below 2^32, positions
    s, c = np.meshgrid(EDGE_WORDS, EDGE_WORDS)
    rows["edges"] = (s.ravel(), c.ravel())
    return rows


def test_basalt_philox_uniform_vs_host():
    rng = np.random.default_rng(25)
    rows = h5_rows(rng)
    s0, c0 = (np.concatenate([v[j] for v in rows.values()]) for j in range(2))
    n = len(s0)
    flip = lambda: (rng.integers(1, 1 << 32, n).astype(U64) << U64(32))          # a nonzero change of the high word
    s, c = np.r_[s0, s0 ^ flip(), s0], np.r_[c0, c0, c0 ^ flip()]
    assert s.dtype == c.dtype == U64
    got = run_kernel(H5, "h5_philox", [s, c], 1, U32)[:, 0]
    want = np.array([philox_uniform(x, y) for x, y in zip(s.tolist(), c.tolist())], F32).view(U32)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (f"philox_uniform: {bad.size} of {len(s)} differ; first seed={s[bad[0]]:#018x} "
                           f"counter={c[bad[0]]:#018x}: got {got[bad[0]]:#010x}, want {want[bad[0]]:#010x}")
    u = got.view(F32).astype(F64)
    assert np.all(u * 2.0 ** 24 == np.floor(u * 2.0 ** 24)) and u.min() >= 0 and u.max() <= 1 - 2.0 ** -24
    # the seed's high word (c3) and the counter's (c1) each change the draw (two equal draws: 2^-24 per row)
    base, seed_hi, ctr_hi = got[:n], got[n: 2 * n], got[2 * n:]
    assert np.count_nonzero(seed_hi == base) <= 2 and np.count_nonzero(ctr_hi == base) <= 2


# ---- H.6: keys, compares, fmaxf, the approximate exponential ------------------------------------------------------

H6 = """
kernel void h6_keys(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                    device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const uint x = a[i], y = b[i];
  device uint* o = out + 6 * i;
  o[0] = okey(x);
  o[1] = okey(y);
  o[2] = uint(fnan(x)) | (uint(f_gt(x, y)) << 1) | (uint(f_lt(x, y)) << 2);
  o[3] = fmaxf_bits(x, y);
  o[4] = as_type<uint>(expf_approx(as_type<float>(x ^ 0x80000000u)));
  o[5] = as_type<uint>(expf_neg(as_type<float>(x)));
}
"""


def nan32(x):
    return (x & U32(0x7FFFFFFF)) > U32(INF32)


def h6_pairs(rng):
    """The H.6 operand pairs (float32 bit patterns) by class."""
    n = 1 << 13
    k = np.arange(n)
    low = lambda m: floats(rng, rng.choice([0, 0, 1, 2], m))                    # subnormals, the two lowest binades
    pairs = {"uniform": (words(rng, 1 << 16, U32), words(rng, 1 << 16, U32)),
             "subnormal": (low(2 * n), low(2 * n))}
    # b a unit in the last place either side of a, or a itself; a's sign flipped; a quarter of them across 2^-126,
    # an eighth from a signed zero
    x = np.where(k % 2 == 0, low(n), floats(rng, rng.integers(0, 256, n)))
    x = np.where(k % 4 == 1, (x & U32(SIGN32)) | U32(0x800000), x)
    x = np.where(k % 8 == 3, x & U32(SIGN32), x)
    pairs["adjacent"] = (np.r_[x, x], np.r_[offset32(x, rng.choice([-1, 0, 1], n)), x ^ U32(SIGN32)])
    sp = np.r_[SPECIALS32, np.array([2, 0x800001, 0x7FFFFFFF], U32)]
    sp = np.r_[sp, sp | U32(SIGN32)]
    pa, pb = np.meshgrid(sp, sp)
    pairs["specials"] = (pa.ravel(), pb.ravel())
    return pairs


def test_basalt_keys_compares_fmaxf_vs_numpy():
    rng = np.random.default_rng(26)
    pairs = h6_pairs(rng)
    a, b = (np.concatenate([v[j] for v in pairs.values()]) for j in range(2))
    assert a.dtype == b.dtype == U32
    got = run_kernel(H6, "h6_keys", [a, b], 6, U32)
    A, B = a.view(F32), b.view(F32)
    na, nb = np.isnan(A), np.isnan(B)
    flags = na.astype(U32) | ((A > B).astype(U32) << U32(1)) | ((A < B).astype(U32) << U32(2))
    fmax = np.where(na & nb, U32(0x7FFFFFFF), np.where(na, b, np.where(nb, a, np.where(B > A, b, a))))
    for col, name, want in [(2, "fnan | f_gt << 1 | f_lt << 2", flags), (3, "fmaxf_bits", fmax)]:
        bad = np.flatnonzero(got[:, col] != want)
        assert bad.size == 0, (f"{name}: {bad.size} of {len(a)} differ; first a={a[bad[0]]:#010x} b={b[bad[0]]:#010x}: "
                               f"got {got[bad[0], col]:#x}, want {want[bad[0]]:#x}")
    # okey orders every non-NaN pair as numpy does, and sorts every non-NaN value as numpy does
    ka, kb, ok = got[:, 0], got[:, 1], ~na & ~nb
    assert np.array_equal((ka < kb)[ok], (A < B)[ok]) and np.array_equal((ka == kb)[ok], (A == B)[ok])
    v, key = np.r_[A[~na], B[~nb]], np.r_[ka[~na], kb[~nb]]
    order = np.argsort(v, kind="stable")
    v, key = v[order], key[order]
    assert np.all(key[1:] >= key[:-1]) and np.array_equal(key[1:] == key[:-1], v[1:] == v[:-1])
    # the rows are what they claim: the GPU's own compares (a subnormal read as zero) order many of them differently,
    # and -0 meets +0
    fa, fb = emu.ftz(A), emu.ftz(B)
    assert np.count_nonzero(ok & (((fa > fb) != (A > B)) | ((fa < fb) != (A < B)))) > 5000
    assert np.count_nonzero(ok & (A == B) & (a != b)) > 500
    # expf_approx(-g) is expf_neg(g) bit for bit, also where ieee_mul keeps a subnormal product the plain one flushes
    e1, e2 = got[:, 4], got[:, 5]
    bad = np.flatnonzero((nan32(e1) != nan32(e2)) | (~nan32(e1) & (e1 != e2)))
    assert bad.size == 0, (f"expf_approx(-g) != expf_neg(g): {bad.size} of {len(a)}; first g={a[bad[0]]:#010x}: "
                           f"{e1[bad[0]]:#010x} against {e2[bad[0]]:#010x}")
    p = np.abs(A[~na].astype(F64)) * float(np.uint32(0x3FB8AA3B).view(F32))
    assert np.count_nonzero((p > 0) & (p < 2.0 ** -126)) > 5000
