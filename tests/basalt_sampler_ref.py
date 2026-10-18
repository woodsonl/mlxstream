# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its host references, fixtures and pass criteria.
# nv_exp and nv_logf: transcribed from NVIDIA libdevice's __nv_exp and __nv_logf as nvcc 13.0 (V13.0.48,
# -arch=sm_120 -lineinfo -O3 -DNDEBUG) inlined them into basalt's sampler.cu; PTX listing and transcription:
# docs/research/basalt-nvcc/sampler.md sections 2.5-2.6.
"""The sampler family's host references, numpy only (no Metal import), imported as ``from tests.basalt_sampler_ref
import ...``.

- ``nv_exp`` (binary64, on a Python ``float``) and ``nv_logf`` (binary32, on an ``np.float32``): CUDA's library
  ``exp(double)`` and ``logf`` as nvcc inlined them into the sampled tail, operation for operation (``fma64`` for
  ``fma.rn.f64``, ``fma32`` for ``fma.rn.f32``). Neither is correctly rounded, so these give CUDA's bits where
  ``math.exp`` and ``math.log`` miss them; kernels/common/basalt_libdevice.metal is the kernels' transcription.
- ``f32_rn``: a ``Fraction`` rounded to the nearest float32, ties to even (normal range); ``fma32``: the float32
  ``a * b + c`` rounded once through it.
- ``philox_uniform``: basalt's host Philox (``host_philox_uniform``, sampler_parity.cpp:134-158), the kernel's draw
  (sampler.cu:37-59) transcribed; basalt's own variant, not a library generator. The draw for row t is
  ``philox_uniform(seed, (counter + t) mod 2^64)``.
"""

import math
import struct
from fractions import Fraction

import numpy as np

f32 = np.float32
M = 0xFFFFFFFF


def d2u(x): return struct.unpack("<Q", struct.pack("<d", x))[0]
def u2d(u): return struct.unpack("<d", struct.pack("<Q", u & 0xFFFFFFFFFFFFFFFF))[0]
def f2u(x): return struct.unpack("<I", struct.pack("<f", x))[0]
def u2f(u): return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
def hexd(c): return u2d(c)
def hexf(c): return np.float32(u2f(c))


def fma64(a, b, c):
    """fma.rn.f64: ``math.fma``, which raises where IEEE returns NaN (inf * 0, inf - inf) or a signed infinity (an
    overflow)."""
    try:
        return math.fma(a, b, c)
    except ValueError:
        return math.nan
    except OverflowError:
        return math.copysign(math.inf, a * b + c)


def nv_exp(x):
    """CUDA's exp(double) (libdevice ``__nv_exp``) on a Python float: sampler.md section 2.6, operation for
    operation."""
    t = fma64(x, hexd(0x3FF71547652B82FE), hexd(0x4338000000000000))
    i = d2u(t) & 0xFFFFFFFF                                   # lo32(t): round(x / ln2) as int32 bits
    i = i - (1 << 32) if i & 0x80000000 else i
    j = t + hexd(0xC338000000000000)                          # add.rn.f64
    r = fma64(j, hexd(0xBFE62E42FEFA39EF), x)
    r = fma64(j, hexd(0xBC7ABC9E3B39803F), r)
    p = fma64(r, hexd(0x3E5ADE1569CE2BDF), hexd(0x3E928AF3FCA213EA))
    for c in (0x3EC71DEE62401315, 0x3EFA01997C89EB71, 0x3F2A01A014761F65, 0x3F56C16C1852B7AF, 0x3F81111111122322,
              0x3FA55555555502A1, 0x3FC5555555555511, 0x3FE000000000000B, 0x3FF0000000000000, 0x3FF0000000000000):
        p = fma64(p, r, hexd(c))
    pu = d2u(p)
    lo, hi = pu & 0xFFFFFFFF, pu >> 32
    res = u2d(lo | (((hi + (i << 20)) & 0xFFFFFFFF) << 32))
    hx = (d2u(x) >> 32) & 0x7FFFFFFF                          # abs.f32 of hi32(x) seen as a float: an integer compare
    if not hx < 0x4086232B:
        res = 0.0 if x < 0.0 else x + math.inf
        if hx < 0x40874800:
            h = (i + ((i >> 31) & 1)) >> 1                    # trunc(i / 2)
            a = u2d(lo | (((hi + (h << 20)) & 0xFFFFFFFF) << 32))
            b = u2d((((i - h) << 20) + 0x3FF00000 & 0xFFFFFFFF) << 32)
            res = b * a
    return res


def f32_rn(q):
    """Fraction -> nearest-even float32 (normal range only; enough for logf's intermediates)."""
    if q == 0: return np.float32(0.0)
    s, q = (-1, -q) if q < 0 else (1, q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q: e -= 1
    m = q / Fraction(2) ** (e - 23)                           # in [2^23, 2^24)
    n, rem = divmod(m.numerator, m.denominator)
    if 2 * rem > m.denominator or (2 * rem == m.denominator and n & 1): n += 1
    return np.float32(s * float(Fraction(n) * Fraction(2) ** (e - 23)))


def fma32(a, b, c): return f32_rn(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def nv_logf(x):
    """CUDA's logf (libdevice ``__nv_logf``, the IEEE build) on an np.float32: sampler.md section 2.5, operation for
    operation."""
    x = np.float32(x)
    small = x < np.float32(u2f(0x00800000))
    with np.errstate(over="ignore"):
        xs = np.float32(x * np.float32(8388608.0)) if small else x  # exact: a subnormal times 2^23
    fe = hexf(0xC1B80000) if small else np.float32(0.0)
    ib = f2u(float(xs))
    e = ((ib - 0x3F2AAAAB) & 0xFF800000) & 0xFFFFFFFF          # add 0xC0D55555 (-1059760811), and 0xFF800000
    m = np.float32(u2f((ib - e) & 0xFFFFFFFF))
    es = e - (1 << 32) if e & 0x80000000 else e
    fe = fma32(np.float32(es), hexf(0x34000000), fe)          # cvt.rn.f32.s32 exact here (multiple of 2^23)
    f = np.float32(m + hexf(0xBF800000))
    pp = fma32(f, hexf(0xBE055027), hexf(0x3E1039F6))
    for c in (0xBDF8CDCC, 0x3E0F2955, 0xBE2AD8B9, 0x3E4CED0B, 0xBE7FFF22, 0x3EAAAA78, 0xBF000000):
        pp = fma32(pp, f, hexf(c))
    q = np.float32(f * pp)
    q = fma32(q, f, f)
    r = fma32(fe, hexf(0x3F317218), q)
    if f2u(float(xs)) > 0x7F7FFFFF:                           # +inf, NaN, negative: fma(xs, inf, inf)
        r = np.float32(float(xs) * math.inf + math.inf)
    if xs == 0: r = np.float32(-math.inf)
    return r


def philox_uniform(seed, counter):
    c0, c1, c2, c3 = counter & M, (counter >> 32) & M, seed & M, (seed >> 32) & M
    for i in range(10):
        p0, p1 = 0x9E3779B9 * c0, 0xBB67AE85 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ i) & M, p1 & M, ((p0 >> 32) ^ c3) & M, p0 & M
    return f32(c0 >> 8) * f32(1.0 / 16777216.0)
