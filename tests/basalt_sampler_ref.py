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
- sampler_parity's host references, on float32 scalars one operation at a time (``SamplerParams``, ``fmax``):
  ``reference_pick`` / ``reference_cut``; the kernels' own list and tail, ``mirror_select`` and ``mirror_pick``, with
  ``sampled_k`` and ``window_of``; ``greedy_pen`` for the greedy references local to its main. ``penal`` is their
  penalty, rounded as the device rounds it. ``mirror_pick`` calls ``nv_exp`` and ``nv_logf`` where basalt's host
  calls std::exp(double) and logf; ``reference_pick`` and ``reference_cut`` keep ``math.exp``: no kernel computes
  their sums.
- The vectorized forms the large fixtures need: ``penal_vec``, ``select_vec`` and ``mirror_select`` itself, equal
  bit for bit to basalt's loop, ``mirror_select_scalar``.
"""

import functools
import math
import struct
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple

import numpy as np

f32 = np.float32
M = 0xFFFFFFFF
NINF = f32(-np.inf)


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


@dataclass
class SamplerParams:
    """basalt's SamplerParams (sampler.hpp:15-29): its fields and defaults, without ``logit_bias`` (the references
    take rows the caller biased). The float fields are rounded to float32 where they are read, as the C++ fields hold
    them. A fixture's variant is ``dataclasses.replace(p, ...)``."""
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    temperature: float = 1.0
    min_keep: int = 1
    penalty_last_n: int = 0
    penalty_repeat: float = 1.0
    penalty_freq: float = 0.0
    penalty_present: float = 0.0
    seed: int = 0
    counter: int = 0
    greedy: bool = False


def fmax(a, b):
    """std::fmax on float32 scalars: the other operand when one is NaN, else the larger, the first on a tie."""
    return b if a != a else (a if b != b else max(a, b))


@functools.lru_cache(maxsize=1 << 18)
def _nv_exp_bits(u): return nv_exp(u2d(u))


def nv_exp_memo(x):
    """``nv_exp`` memoized on the argument's bits: the sampled tail's std::exp(double)."""
    return _nv_exp_bits(d2u(x))


# ---- reference_pick, reference_cut (sampler_parity.cpp:34-102) -----------------------------------------------------

def _head(l, p, temp_first):
    """Their top-k (:37-53, :73-86): k = top_k, or the whole row when top_k <= 0 (not capped at 64); each round the
    first untaken id wins outright, then strict > (the lowest id on a tie). ``temp_first`` ranks l * (1 / T)."""
    nv = len(l)
    inv_t = f32(1) / f32(p.temperature) if f32(p.temperature) > 0 else f32(0)
    val = (lambda v: f32(l[v]) * inv_t) if temp_first else (lambda v: f32(l[v]))
    k = p.top_k if p.top_k > 0 else nv
    taken, ids = [False] * nv, []
    for _ in range(k):
        best, bv = -1, f32(0)
        for v in range(nv):
            if taken[v]: continue
            if best < 0 or val(v) > bv: best, bv = v, val(v)
        taken[best] = True; ids.append(best)
    return ids, val


def _top_p_cut(ids, val, p):
    """top_p over the whole list (:54-66, :87-99): base fmax of the list, binary64 sums in list order from 0, each
    quotient recomputed; at least min_keep (at most the list). std::exp is math.exp: no gate reads its last bit."""
    mx = val(ids[0])
    for v in ids: mx = fmax(mx, val(v))
    s = 0.0
    for v in ids: s += math.exp(float(val(v)) - float(mx))
    cum, cut = 0.0, len(ids)
    for i, v in enumerate(ids):
        cum += math.exp(float(val(v)) - float(mx)) / s
        if cum >= float(f32(p.top_p)): cut = i + 1; break
    if cut < p.min_keep: cut = p.min_keep if p.min_keep < len(ids) else len(ids)
    return cut


def reference_pick(l, p, temp_first):
    """The greedy pick in top_k -> top_p -> temperature -> argmax order (``temp_first``: temperature first); no
    penalties, min_p, bias or seed. The cut keeps ids[0], so it is the lowest-index argmax of the ranked values."""
    ids, val = _head(l, p, temp_first)
    if f32(p.top_p) < 1: ids = ids[:_top_p_cut(ids, val, p)]
    return ids[0]


def reference_cut(l, p, temp_first):
    """The number of top_p survivors in that order: the quantity the order changes."""
    ids, val = _head(l, p, temp_first)
    return _top_p_cut(ids, val, p) if f32(p.top_p) < 1 else len(ids)


# ---- the penalty, and the kernels' own semantics (sampler_parity.cpp:267-345) ---------------------------------------

def penal(logit, count, p):
    """apply_penalties (sampler.cu:71-77; the host's :165-170) for a logit seen ``count`` times in the window: a
    non-positive logit is multiplied by penalty_repeat and any other (NaN included) divided, then count * freq +
    present is subtracted, that sum rounded once (``fma32``) as the device's fma rounds it. basalt's host rounds the
    product first, the same value wherever the product is exact (every fixture that compares with a host
    reference). ``count <= 0`` leaves the logit."""
    x = f32(logit)
    if count <= 0: return x
    x = x * f32(p.penalty_repeat) if x <= 0 else x / f32(p.penalty_repeat)
    return x - fma32(f32(count), f32(p.penalty_freq), f32(p.penalty_present))


class SelList(NamedTuple):
    """A top-k list (:273-276): ids and their penalised float32 logits, in list order."""
    ids: list
    logit: list


def sampled_k(top_k, nv):
    """The sampled list's width (:345, sampler.cu:339-342): top_k for 1..63, 64 for anything else, at most nv."""
    return min(nv, top_k if 0 < top_k < 64 else 64)


def penal_vec(l, window, p):
    """``penal`` over a row: the window counted with bincount (-1 and ids outside [0, nv) dropped, as the kernels
    drop them), the penalised entries' repeat step elementwise (NaN and positive values divide, like setp.gtu), and
    one ``fma32`` per distinct count. numpy's float32 operations round once each and never contract."""
    x = np.array(l, np.float32)
    w = np.asarray(window, np.int64)
    cnt = np.bincount(w[(w >= 0) & (w < len(x))], minlength=len(x))
    m = cnt > 0
    rep = f32(p.penalty_repeat)
    xm = x[m]
    xm = np.where(~(xm <= 0), xm / rep, xm * rep)
    t = {c: fma32(f32(c), f32(p.penalty_freq), f32(p.penalty_present)) for c in np.unique(cnt[m])}
    x[m] = xm - np.array([t[c] for c in cnt[m]], np.float32)
    return x


def select_vec(s, k):
    """The kernels' top-k over penalised logits ``s``: NaN and -inf never listed, value descending with ties (-0 and
    +0 alike) to the lowest id by a stable sort (+inf first), padded to k with the empty round's (id 0, -inf)."""
    s = np.asarray(s, np.float32)
    idx = np.nonzero(~np.isnan(s) & (s != NINF))[0]
    order = idx[np.argsort(-s[idx], kind="stable")][:k]
    pad = k - len(order)
    return SelList([int(i) for i in order] + [0] * pad, [s[i] for i in order] + [NINF] * pad)


def mirror_select(l, nv, window, p, k):
    """The top-k list the kernels build for one row (:279-305), vectorized: ``window`` is the counted history
    (``window_of``). Equal bit for bit to ``mirror_select_scalar`` (tests/contract/test_basalt_sampler_refs.py)."""
    return select_vec(penal_vec(l[:nv], window, p), k)


def mirror_select_scalar(l, nv, window, p, k):
    """mirror_select as basalt wrote it, O(k * nv): each round the largest untaken value by strict > from -inf (NaN
    and -inf never picked, the lowest id on a tie); a round that finds nothing stores id 0 with -inf and takes id 0."""
    s = [penal(l[v], sum(1 for h in window if h == v), p) for v in range(nv)]
    ids, lg, taken = [], [], [False] * nv
    for _ in range(k):
        best, bv = nv, NINF
        for v in range(nv):
            if not taken[v] and s[v] > bv: bv, best = s[v], v
        i = best if best < nv else 0
        taken[i] = True; ids.append(i); lg.append(bv)
    return SelList(ids, lg)


def mirror_pick(sel, k, p, row):
    """The tail over sel[0:k] (:308-343): top_p with base fmax of the list and its min_keep clamp, then the min_p
    prefix cut (strict <), the temperature (a float32 product), and the draw philox_uniform(seed, counter + row):
    the first kept entry whose cumulative probability exceeds it, else the last kept one. Binary64 sums in list order
    from 0, with CUDA's exp and logf (``nv_exp_memo``, ``nv_logf``) where basalt's host calls std::exp and logf: the
    port's fixtures are its own, so a draw may land where only the last bit decides."""
    ids, lg = sel
    n_keep, mx = k, lg[0]
    for i in range(1, k): mx = fmax(mx, lg[i])
    if f32(p.top_p) < 1:
        s = 0.0
        for i in range(k): s += nv_exp_memo(float(lg[i]) - float(mx))
        cum, cut = 0.0, k
        for i in range(k):
            cum += nv_exp_memo(float(lg[i]) - float(mx)) / s
            if cum >= float(f32(p.top_p)): cut = i + 1; break
        if cut < p.min_keep: cut = p.min_keep if p.min_keep < k else k
        n_keep = cut
    if f32(p.min_p) > 0:
        thresh = lg[0] + nv_logf(f32(p.min_p))
        for i in range(n_keep):
            if lg[i] < thresh: n_keep = i; break
    inv_t = f32(1) / f32(p.temperature) if f32(p.temperature) > 0 else f32(0)
    smx = lg[0] * inv_t
    for i in range(1, n_keep): smx = fmax(smx, lg[i] * inv_t)
    s = 0.0
    for i in range(n_keep): s += nv_exp_memo(float(lg[i] * inv_t) - float(smx))
    u = philox_uniform(p.seed, (p.counter + row) & (2**64 - 1))
    cum, pick = 0.0, ids[n_keep - 1 if n_keep > 0 else 0]
    for i in range(n_keep):
        cum += nv_exp_memo(float(lg[i] * inv_t) - float(smx)) / s
        if float(u) < cum: pick = ids[i]; break
    return pick


def window_of(hist, hist_len, t, last_n):
    """Row t's counted window (:504-510): the last min(last_n, hist_len) of its hist_len slots, as the kernels count
    them (sampler.cu:97-103); empty without a history or without penalties."""
    if hist_len <= 0 or last_n <= 0: return []
    h = min(last_n, hist_len)
    return hist[t * hist_len + hist_len - h:(t + 1) * hist_len]


# ---- the greedy references local to main ---------------------------------------------------------------------------

def greedy_pen(l, hist, p, divide_unconditionally=False):
    """One scan for every greedy reference in sampler_parity's main: each id in order, the first wins outright, then
    strict > (the lowest id on a tie), each logit through ``penal`` with its count over all of ``hist`` (pass the
    window). Fixture 4's ``pick`` (:890-905; ``divide_unconditionally`` is its wrong rule for a counted id,
    l / repeat - c * freq - present, two subtractions), fixture 8's ``pick_clamped`` (:1075-1086: ``hist`` the last 4
    or all 8 slots), fixture 12's ``greedy_pen`` (:1177-1190), fixture 13's inline reference (:1262-1271) and the
    logit_bias host argmax (:758-761: no ``hist``). Their host roundings are ``penal``'s wherever they are used: every
    count * freq they see is exact."""
    cnt = Counter(int(h) for h in hist)
    best, bv = 0, f32(0)
    for v in range(len(l)):
        c = cnt[v]
        if divide_unconditionally and c > 0:
            s = f32(l[v]) / f32(p.penalty_repeat) - f32(c) * f32(p.penalty_freq) - f32(1) * f32(p.penalty_present)
        else:
            s = penal(l[v], c, p)
        if v == 0 or s > bv: best, bv = v, s
    return best
