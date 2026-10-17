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
"""

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
