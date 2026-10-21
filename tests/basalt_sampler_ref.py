# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp, include/basalt/kernels/sampler.hpp
# (penalty_rows) and include/basalt/core/coupled_draft.hpp (the coupled_* helpers)
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): its host references, fixtures and pass
# criteria; its penalty reference (penal) is basalt's transcription of ggml-org/llama.cpp src/llama-sampler.cpp
# (llama_sampler_penalties_apply) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
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
  ``reference_pick`` / ``reference_cut``; the sampled chain in llama.cpp's order, ``sampled_reference``, and its
  survivor count ``sampled_cut``; the kernels' own list and tail, ``mirror_select`` and ``mirror_pick``, with
  ``sampled_k`` and ``window_of``; the probe's probability, ``mirror_prob``; ``greedy_pen`` for the greedy references
  local to its main; ``penalty_rows``, basalt's host builder of a verify window's history rows (sampler.hpp:86-97);
  the coupled draft's ``coupled_draft_cell``, ``coupled_draft_counter``, ``coupled_hist_len``,
  ``coupled_hist_start`` and ``coupled_hist_base`` (coupled_draft.hpp:65-83).
  ``penal`` is their penalty, rounded as the device rounds it. ``sampled_reference``,
  ``sampled_cut`` and ``mirror_pick`` call ``nv_exp`` and ``nv_logf`` where basalt's host calls std::exp(double) and
  logf; ``reference_pick``, ``reference_cut`` and ``mirror_prob`` keep ``math.exp``: no kernel computes their sums,
  and ``mirror_prob`` is the independent check of the kernel's probability.
- The vectorized forms the large fixtures need: ``penal_vec``, ``select_vec`` and ``mirror_select`` itself, equal
  bit for bit to basalt's loop, ``mirror_select_scalar``.
- The exact fixtures of its main, which draw nothing (spec T§8): ``fixture2_data``, ``fixture4_data``,
  ``fixture6_data`` to ``fixture10_data``, each basalt's rows, history and base parameters.
- The fixture generators, one per random block of sampler_parity (spec T§4): ``normal_rows`` for the plain fills,
  ``fixture12_data``, ``fixture13_data``, ``fixture16_data``, ``fixture17_data``, ``fixture18_data``, ``probe_rows``,
  ``bench_rows`` and ``coupled_cases``; ``raw32`` gives basalt's ``rng()`` draws and ``roundf`` C's rounding.
"""

import functools
import math
import struct
from collections import Counter
from dataclasses import dataclass, replace
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


def fma_exact(a, b, c):
    """IEEE's fma on Python floats, exactly, for Pythons without ``math.fma`` (before 3.13): a * b + c rounded once to
    nearest even (an int quotient is correctly rounded, subnormals included), NaN for a NaN operand, inf * 0 or
    inf - inf, the signed infinity on overflow; an exact zero is +0 unless the product is a zero and both it and c
    are negative zeros."""
    inf_p = math.isinf(a) or math.isinf(b)
    if math.isnan(a) or math.isnan(b) or math.isnan(c) or (inf_p and (a == 0 or b == 0)):
        return math.nan
    ps = math.copysign(1.0, a) * math.copysign(1.0, b)
    if inf_p:
        return math.nan if math.isinf(c) and math.copysign(1.0, c) != ps else math.copysign(math.inf, ps)
    if math.isinf(c):
        return c
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        return -0.0 if (a == 0 or b == 0) and ps < 0 and math.copysign(1.0, c) < 0 else 0.0
    try:
        return exact.numerator / exact.denominator
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def fma64(a, b, c):
    """fma.rn.f64: ``math.fma`` (Python 3.13 and later), which raises where IEEE returns NaN (inf * 0, inf - inf) or a
    signed infinity (an overflow); ``fma_exact`` before 3.13."""
    if not hasattr(math, "fma"):
        return fma_exact(a, b, c)
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


# ---- sampled_reference, sampled_cut (sampler_parity.cpp:160-265) and mirror_prob (:347-380) -------------------------
# Transcribed apart from mirror_pick, though they share its tail's steps: test_basalt_sampler_refs compares mirror_pick
# over the vectorized list with sampled_reference, which a shared helper would compare with itself.

def _select_first(l, hist, p):
    """The top-k of sampled_reference and sampled_cut (:165-189): k = sampled_k(top_k, nv); each round the first untaken
    id wins outright, even at -inf or NaN, then strict > (the lowest id on a tie), every logit through ``penal`` with
    its count over all of ``hist`` (the callers pass the window). Each id's count and penalty are computed once rather
    than once per round, the same values."""
    nv, cnt = len(l), Counter(int(h) for h in hist)
    s = [penal(l[v], cnt[v], p) for v in range(nv)]
    taken, ids, lg = [False] * nv, [], []
    for _ in range(sampled_k(p.top_k, nv)):
        best, bv = -1, f32(0)
        for v in range(nv):
            if taken[v]: continue
            if best < 0 or s[v] > bv: best, bv = v, s[v]
        taken[best] = True; ids.append(best); lg.append(bv)
    return ids, lg


def sampled_reference(l, hist, p, row):
    """The sampled chain in llama.cpp's order (:160-222) on one row ``l`` and its counted window ``hist``:
    ``_select_first``'s list; top_p over the whole list with base lg[0] (no max taken) and the min_keep clamp; the
    min_p prefix cut (strict <) on top_p's survivors; the temperature (float32 products with 1 / T, 0 when T <= 0),
    base the fmax of the kept products; the draw philox_uniform(seed, counter + row): the first kept entry whose
    cumulative probability exceeds it, else the last kept one. Binary64 sums in list order from 0, with ``nv_exp_memo``
    and ``nv_logf`` where the host calls std::exp(double) and logf. It equals ``mirror_pick`` over ``mirror_select``'s
    list on rows without NaN that hold at least k values above -inf (spec T§3.10); the mirror is the reference
    elsewhere."""
    ids, lg = _select_first(l, hist, p)
    n_sel = n_keep = len(ids)
    if f32(p.top_p) < 1:
        s = 0.0
        for x in lg: s += nv_exp_memo(float(x) - float(lg[0]))
        cum, cut = 0.0, n_sel
        for i, x in enumerate(lg):
            cum += nv_exp_memo(float(x) - float(lg[0])) / s
            if cum >= float(f32(p.top_p)): cut = i + 1; break
        if cut < p.min_keep: cut = p.min_keep if p.min_keep < n_sel else n_sel
        n_keep = cut
    if f32(p.min_p) > 0:
        thresh = lg[0] + nv_logf(f32(p.min_p))
        for i in range(n_keep):
            if lg[i] < thresh: n_keep = i; break
    assert n_keep > 0, "min_p > 1 kept nothing: basalt's host then reads sel_ids[-1]; use the mirror"
    inv_t = f32(1) / f32(p.temperature) if f32(p.temperature) > 0 else f32(0)
    sc = [x * inv_t for x in lg]
    smx = sc[0]
    for i in range(1, n_keep): smx = fmax(smx, sc[i])
    s = 0.0
    for i in range(n_keep): s += nv_exp_memo(float(sc[i]) - float(smx))
    u = philox_uniform(p.seed, (p.counter + row) & (2**64 - 1))
    cum, pick = 0.0, ids[n_keep - 1]
    for i in range(n_keep):
        cum += nv_exp_memo(float(sc[i]) - float(smx)) / s
        if float(u) < cum: pick = ids[i]; break
    return pick


def sampled_cut(l, hist, p):
    """The survivors of the min_p and top_p cuts (:224-265), in the REVERSE of the chain's order: min_p over the whole
    ``_select_first`` list (threshold sel[0] + logf(min_p), strict <), then top_p over its survivors with base sel[0]
    and the min_keep clamp. Its one caller, fixture 7, has top_p = 1, where the order cannot matter; it is not the
    chain where top_p < 1 and min_p > 0 together. ``nv_exp_memo`` and ``nv_logf``, as in sampled_reference."""
    _, sel = _select_first(l, hist, p)
    n_minp = len(sel)
    if f32(p.min_p) > 0:
        thresh = sel[0] + nv_logf(f32(p.min_p))
        for i in range(len(sel)):
            if sel[i] < thresh: n_minp = i; break
    if f32(p.top_p) >= 1: return n_minp
    s = 0.0
    for i in range(n_minp): s += nv_exp_memo(float(sel[i]) - float(sel[0]))
    cum, cut = 0.0, n_minp
    for i in range(n_minp):
        cum += nv_exp_memo(float(sel[i]) - float(sel[0])) / s
        if cum >= float(f32(p.top_p)): cut = i + 1; break
    if cut < p.min_keep: cut = p.min_keep if p.min_keep < n_minp else n_minp
    return cut


def mirror_prob(sel, k, p, q):
    """The probe's double reference (:347-380): the final probability of id ``q`` under the tail over sel[0:k], 0
    outside the survivors. Its survivors are mirror_pick's, computed again here: top_p with base the fmax of the list
    and the min_keep clamp, then min_p. Then the kept logits times 1 / T in float32 (unguarded: its callers have T >
    0), base the fmax of those products as doubles from -inf, and the probability of the LAST kept entry whose id is
    ``q`` (the kernel's qprob takes the first, sampler.cu:433-434; only id 0 can repeat, from empty rounds). Every
    exponential is ``math.exp``, as basalt's host std::exp: it is the independent double check behind P.3's 1e-6, not
    the kernels' exp; the min_p threshold uses ``nv_logf``, the one logf helper."""
    ids, lg = sel
    n_keep, mx = k, lg[0]
    for i in range(1, k): mx = fmax(mx, lg[i])
    if f32(p.top_p) < 1:
        s = 0.0
        for i in range(k): s += math.exp(float(lg[i]) - float(mx))
        cum, cut = 0.0, k
        for i in range(k):
            cum += math.exp(float(lg[i]) - float(mx)) / s
            if cum >= float(f32(p.top_p)): cut = i + 1; break
        if cut < p.min_keep: cut = p.min_keep if p.min_keep < k else k
        n_keep = cut
    if f32(p.min_p) > 0:
        thresh = lg[0] + nv_logf(f32(p.min_p))
        for i in range(n_keep):
            if lg[i] < thresh: n_keep = i; break
    inv_t = f32(1) / f32(p.temperature)
    smx = -math.inf
    for i in range(n_keep): smx = fmax(smx, float(lg[i] * inv_t))
    s = pq = 0.0
    for i in range(n_keep):
        e = math.exp(float(lg[i] * inv_t) - smx)
        s += e
        if ids[i] == q: pq = e
    return pq / s if n_keep > 0 else 0.0


def window_of(hist, hist_len, t, last_n):
    """Row t's counted window (:504-510): the last min(last_n, hist_len) of its hist_len slots, as the kernels count
    them (sampler.cu:97-103); empty without a history or without penalties."""
    if hist_len <= 0 or last_n <= 0: return []
    h = min(last_n, hist_len)
    return hist[t * hist_len + hist_len - h:(t + 1) * hist_len]


def penalty_rows(tail, window, T, h, out=None):
    """basalt's penalty_rows (sampler.hpp:86-97): the penalty-history rows of a verify window, T rows of h slots, flat
    int32, written into ``out`` (every slot) or a new array. Row t is the last h tokens of ``tail`` followed by
    window[0..t], most recent last, -1 in the unused front slots: ``tail`` is what the state consumed before the
    window, window[0] the fed-back token and window[1..] the drafts, so row t is the history plain decode counts when
    it picks the token after window[t]."""
    tail = np.asarray(tail, np.int32)
    seq = np.concatenate([tail, np.asarray(window, np.int32)[:T]])  # (tail, window)
    out = np.empty(T * h, np.int32) if out is None else out
    for t in range(T):
        avail = len(tail) + t + 1                                     # tail + window[0..t]
        take = min(h, avail)
        out[t * h:t * h + h - take] = -1
        out[t * h + h - take:(t + 1) * h] = seq[avail - take:avail]
    return out


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


# ---- the exact fixtures of main (spec T§8) -------------------------------------------------------------------------
# They draw nothing, so spec T§8 pins their host results. Each returns basalt's rows ((rows, nv) float32), its history
# (flat int32, as window_of reads it) where it has one, and its base SamplerParams; a check's variants are
# dataclasses.replace of them.

def fixture2_data():
    """Fixture 2, the order fixture (:810-846): 4 rows of 512 at -1000 but 3.0 at id 0 and 2.6f - 0.05f * v at ids
    1-7 (the float32 product rounded, then the difference); greedy, top_k 0, top_p 0.5, temperature 0.5."""
    l = np.full((4, 512), -1000, np.float32)
    l[:, 0] = 3.0
    l[:, 1:8] = f32(2.6) - f32(0.05) * np.arange(1, 8, dtype=np.float32)
    return l, SamplerParams(top_k=0, top_p=0.5, temperature=0.5, min_keep=1, greedy=True)


def fixture4_data():
    """Fixture 4, greedy penalties (:881-955): two cases of 2 rows of 8 at -8, with 4-slot histories. A: -1.0 at id 0
    and -1.2 at id 1, history [0, -1, -1, -1]. B: 5.0 at id 0 and 3.4 at id 1, history [0, 0, -1, -1], run with
    ``replace(p, penalty_present=1.5, penalty_freq=0.0)``. Greedy, top_k 0, top_p 1, penalty_last_n 4, repeat 2.
    Returns (la, hist_a, lb, hist_b, p)."""
    la, lb = np.full((2, 8), -8, np.float32), np.full((2, 8), -8, np.float32)
    la[:, :2] = -1.0, -1.2
    lb[:, :2] = 5.0, 3.4
    hist_a, hist_b = np.tile(np.int32([0, -1, -1, -1]), 2), np.tile(np.int32([0, 0, -1, -1]), 2)
    p = SamplerParams(top_k=0, top_p=1.0, temperature=1.0, greedy=True, penalty_last_n=4, penalty_repeat=2.0)
    return la, hist_a, lb, hist_b, p


def fixture6_data():
    """Fixture 6, penalties in the sampled chain (:986-1019): 2 rows of 8 at -8 but 9.0, 4.5, 4.4, 4.3 at ids 0-3;
    4-slot histories [0, -1, -1, -1] and [0, 0, -1, -1] (token 0 once in row 0, twice in row 1); top_k 5, top_p 0.9,
    temperature 0.8, seed 9, counter 0, penalty_last_n 4, repeat 3, freq 0.2, present 0.6."""
    l = np.full((2, 8), -8, np.float32)
    l[:, :4] = 9.0, 4.5, 4.4, 4.3
    return l, np.int32([0, -1, -1, -1, 0, 0, -1, -1]), SamplerParams(
        top_k=5, top_p=0.9, temperature=0.8, seed=9, counter=0, penalty_last_n=4, penalty_repeat=3.0,
        penalty_freq=0.2, penalty_present=0.6)


def fixture7_data():
    """Fixture 7, min_p (:1021-1059): 2 rows of 8 at -8 but 4.0, 3.5, 3.2, 3.1, 2.0 at ids 0-4; top_k 6, top_p 1,
    temperature 0.9, seed 77 (its checks set min_p 0, 0.5 and 0.9)."""
    l = np.full((2, 8), -8, np.float32)
    l[:, :5] = 4.0, 3.5, 3.2, 3.1, 2.0
    return l, SamplerParams(top_k=6, top_p=1.0, temperature=0.9, seed=77)


def fixture8_data():
    """Fixture 8, the penalty window is the tail (:1061-1093): one row of 8 at -8 but 6.0 at id 0 and 6.5 at id 3; the
    8-slot history [0, 0, 0, 0, 3, 3, 3, 3]; greedy, top_k 0, top_p 1, penalty_last_n 4 (the last four slots count),
    repeat 3, freq 0.3, present 0.5."""
    l = np.full((1, 8), -8, np.float32)
    l[0, 0], l[0, 3] = 6.0, 6.5
    return l, np.int32([0, 0, 0, 0, 3, 3, 3, 3]), SamplerParams(
        top_k=0, top_p=1.0, temperature=1.0, greedy=True, penalty_last_n=4, penalty_repeat=3.0, penalty_freq=0.3,
        penalty_present=0.5)


def fixture9_data():
    """Fixture 9, one penalties stage (:1095-1119): 8,000 rows of two logits at 0.0, each with the 1-slot history [0];
    top_k 2, top_p 1, min_p 0, temperature 0.7, seed 53, counter 0, penalty_last_n 1, repeat 1, freq 0, present
    1.5."""
    return np.zeros((8000, 2), np.float32), np.zeros(8000, np.int32), SamplerParams(
        top_k=2, top_p=1.0, min_p=0.0, temperature=0.7, seed=53, counter=0, penalty_last_n=1, penalty_repeat=1.0,
        penalty_freq=0.0, penalty_present=1.5)


def fixture10_data():
    """Fixture 10, top_p before min_p (:1121-1143): 256 rows of 8, logf(0.4f), logf(0.3f), logf(0.2f), logf(0.1f)
    (``nv_logf``), then -30 four times; top_k 4, top_p 0.75, min_p 0.3, temperature 1, seed 10, counter 0."""
    row = np.array([nv_logf(f32(x)) for x in (0.4, 0.3, 0.2, 0.1)] + [-30.0] * 4, np.float32)
    return np.tile(row, (256, 1)), SamplerParams(top_k=4, top_p=0.75, min_p=0.3, temperature=1.0, seed=10, counter=0)


# ---- fixture generators (spec T§4) ---------------------------------------------------------------------------------
# The port draws its own rows, not basalt's (libstdc++'s normal_distribution is not numpy's): one RandomState per block,
# seeded with basalt's seed unless a gate whose truth rests on the data needs another (M12.O, M17.2: change the seed,
# never the criterion), with basalt's distributions, transforms and constructed overwrites in float32, and raw draws
# where basalt has ``rng() % n``. Logits are (rows, nv) float32; histories flat int32, row t at [t * H, (t + 1) * H),
# as ``window_of`` reads them.

def raw32(rs, n):
    """n raw 32-bit draws of RandomState ``rs``, as int64: std::mt19937's rng() for the same seed (numpy's legacy
    seeding is init_genrand), so basalt's ``rng() % m`` is ``raw32(rs, n) % m``."""
    return rs.randint(0, 2**32, size=n, dtype=np.uint64).astype(np.int64)


def _normal(rs, shape, scale):
    """normal_distribution<float>(0, scale) values: float32 standard normals times f32(scale), the product rounded in
    float32 as libstdc++'s ``ret * stddev`` is."""
    return rs.standard_normal(shape).astype(np.float32) * np.float32(scale)


def roundf(x):
    """C's roundf on float32 values: halves away from zero, where np.round rounds them to even; in float64, where the
    + 0.5 is exact for a float32."""
    x = np.asarray(x, np.float32).astype(np.float64)
    return np.copysign(np.floor(np.abs(x) + 0.5), x).astype(np.float32)


def normal_rows(seed, rows=4, nv=512, scale=1.0):
    """A (rows, nv) fill of normal(0, 1) times ``scale`` from RandomState(seed), for the blocks that are one plain fill:
    logit_bias fixture B's ``zeros`` (seed 5, one row of 248,320; :781-782), fixtures 1, 3 and 11 (seeds 3, 5 and 11,
    4 x 512) and fixture 14 (seed 14, scale 0.3: ``g(rng) * 0.3f``, :1282-1284)."""
    return _normal(np.random.RandomState(seed), (rows, nv), scale)


def fixture12_data(seed=12):
    """Fixture 12 (:1191-1204) from RandomState(seed): the 5,000-token tail (raw draws mod 512), then, for T in (1, 2,
    4, 8) and H in (1, 64, 1024, 4096) within each T, a (T, 512) normal(0, 1) fill whose row t holds 5.0 at its draft
    100 + 37 t. Returns the tail and the 16 cases (T, H, window, logits) in that order; the case's histories are
    ``penalty_rows(tail, window, T, H)``."""
    rs = np.random.RandomState(seed)
    tail, cases = (raw32(rs, 5000) % 512).astype(np.int32), []
    for T in (1, 2, 4, 8):
        for H in (1, 64, 1024, 4096):
            window = (100 + 37 * np.arange(T)).astype(np.int32)
            l = _normal(rs, (T, 512), 1.0)
            l[np.arange(T), window] = 5.0
            cases.append((T, H, window, l))
    return tail, cases


def fixture13_data(seed=13):
    """Fixture 13 (:1238-1250) from RandomState(seed): a (4, 512) normal(0, 1) fill, then each row's 16 history slots
    in order, a raw draw mod 512 in each slot j % 3 != 0 (drawn only there) and junk in the others (512, 1512,
    0x7fffffff, -5, 1 << 20, 512); then the id in row t's slot 1 gets the logit 4.0. Returns the logits, the history
    and ``valid_only`` (the drawn ids, -1 in the junk slots), flat int32 of 4 x 16."""
    rs = np.random.RandomState(seed)
    junk = (512, 1512, 0x7FFFFFFF, -5, 1 << 20)
    l = _normal(rs, (4, 512), 1.0)
    hist, valid = np.full(64, -1, np.int32), np.full(64, -1, np.int32)
    for t in range(4):
        for j in range(16):
            hist[16 * t + j] = junk[j // 3 % 5] if j % 3 == 0 else raw32(rs, 1)[0] % 512
            if j % 3: valid[16 * t + j] = hist[16 * t + j]
        l[t, hist[16 * t + 1]] = 4.0
    return l, hist, valid


def fixture16_data(nv, seed=None):
    """Fixture 16 (:1380-1422) for vocabulary nv, from RandomState(1600 + nv) or ``seed``: a (4, nv) fill of
    floor(normal(0, 1) * 4) / 4 (quarter steps), then basalt's constructed rows in its order, spread(j) being
    (7919 j + 13) % nv:
    - row 0: 6.25 at spread(j) for j < 20 and 6.0 for 20 <= j < 320, then +inf at nv / 2 and nv - 1;
    - row 1: -|v|, each zero signed by its id's parity (-0 at even ids), then +inf at 3;
    - row 2: -inf, but 1.0 at spread(400 .. 404) and 0.5 at spread(405 .. 409), then NaN at spread(500) and
      spread(501) and +inf at spread(502): 11 candidates;
    - row 3: 9.0 at each window edge below nv (0, 1023, 1024, 4095, 4096, 8191, 8192, 12345, nv / 2 + 1, nv - 2),
      3.75 at spread(600 .. 614) and 3.5 at spread(615 .. 629), then +inf at nv / 2, outside the window.
    Returns the logits and the history, flat int32 of 4 x 64: -1 except row 3's first slots, which hold the edges in
    order (an edge stored into a slot that is the third, sixth, ... is stored once more: counted twice), then nv,
    nv + 77 and -5, outside the vocabulary."""
    rs = np.random.RandomState(1600 + nv if seed is None else seed)
    inf, sp = np.float32(np.inf), lambda j: (7919 * j + 13) % nv
    l = np.floor(_normal(rs, (4, nv), 1.0) * np.float32(4)) / np.float32(4)
    for j in range(320): l[0, sp(j)] = 6.25 if j < 20 else 6.0
    l[0, nv // 2] = l[0, nv - 1] = inf
    r1 = -np.abs(l[1])
    l[1] = np.where(r1 == 0, np.where(np.arange(nv) & 1, np.float32(0.0), np.float32(-0.0)), r1)
    l[1, 3] = inf
    l[2] = -inf
    for j in range(10): l[2, sp(j + 400)] = 1.0 if j < 5 else 0.5
    l[2, sp(500)] = l[2, sp(501)] = np.nan
    l[2, sp(502)] = inf
    h3 = []
    for e in (0, 1023, 1024, 4095, 4096, 8191, 8192, 12345, nv // 2 + 1, nv - 2):
        if e < nv and e != nv // 2:
            h3.append(e)
            if len(h3) % 3 == 0: h3.append(e)
            l[3, e] = 9.0
    h3 += [nv, nv + 77, -5]
    for j in range(30): l[3, sp(j + 600)] = 3.75 if j < 15 else 3.5
    l[3, nv // 2] = inf
    hist = np.full(4 * 64, -1, np.int32)
    hist[192:192 + len(h3)] = h3
    return l, hist


def fixture17_data(nv, seed=None):
    """Fixture 17 (:1471-1487) for vocabulary nv, from RandomState(1700 + nv) or ``seed``: a (17, nv) fill of
    floor(normal(0, 1.5) * 2) / 2 (half steps; at nv = 512 then floor(v / 2), whole steps), then each row's 64 history
    slots, one raw draw each: the draw mod nv in odd slots, head[draw mod len(head)] in even ones, head being the ids
    (ascending) whose logit is at least the row's maximum minus 1. Returns the logits and the history, flat int32 of
    17 x 64."""
    rs = np.random.RandomState(1700 + nv if seed is None else seed)
    l = np.floor(_normal(rs, (17, nv), 1.5) * np.float32(2)) / np.float32(2)
    if nv == 512: l = np.floor(l / np.float32(2))
    hist = np.empty(17 * 64, np.int32)
    for t in range(17):
        head, r = np.nonzero(l[t] >= l[t].max() - np.float32(1))[0], raw32(rs, 64)
        hist[64 * t:64 * (t + 1)] = np.where(np.arange(64) & 1, r % nv, head[r % len(head)])
    return l, hist


def fixture18_data(nv, T, seed):
    """Fixture 18's ``make`` (:1556-1563) from RandomState(seed): a (T, nv) fill of floor(normal(0, 1.5) * 2) / 2, then
    T x 64 history slots of raw draws mod nv, flat int32. basalt calls it as (248320, 3, 1800) and as (512, T, 1810 + T)
    for T in (64, 70)."""
    rs = np.random.RandomState(seed)
    l = np.floor(_normal(rs, (T, nv), 1.5) * np.float32(2)) / np.float32(2)
    return l, (raw32(rs, 64 * T) % nv).astype(np.int32)


def probe_rows(T, seed=4242):
    """probe_parity's rows (:387-396) for T in (7, 70): one RandomState(seed) for both windows, normal(0, 3), the 7-row
    fill first and the 70-row fill after it; then every even row t gets 14 + t added at (977 t) % 248,320."""
    rs = np.random.RandomState(seed)
    for n in (7, 70)[:(7, 70).index(T) + 1]:
        l = _normal(rs, (n, 248320), 3.0)
    t = np.arange(0, T, 2)
    l[t, t * 977 % 248320] += np.float32(14) + t.astype(np.float32)
    return l


def bench_rows(T, seed=20):
    """bench_sampled's rows (:517-521) for T in (1, 4, 8): one RandomState(seed), normal(0, 3), a (T, 248,320) fill per
    window, in that order."""
    rs = np.random.RandomState(seed)
    for n in (1, 4, 8)[:(1, 4, 8).index(T) + 1]:
        l = _normal(rs, (n, 248320), 3.0)
    return l


# ---- coupled draft sampling's host helpers (include/basalt/core/coupled_draft.hpp:65-83) ----------------------------

COUPLED_HIST_CAP = 4096   # kCoupledHistCap: the longest penalty window the coupled drafter mirrors (serve's cap)


def coupled_draft_cell(p, a, j):
    """The MTP cell of chain step j after a window at position p with a drafts accepted."""
    return p + a + j


def coupled_draft_counter(cell):
    """The draw counter of the verify row that checks the draft made at MTP cell ``cell``: cell + 1, as uint64."""
    return (cell + 1) & (2**64 - 1)


def coupled_hist_len(penalty_last_n, cap):
    """The penalty window the draft layer and the target use: min(penalty_last_n, cap), 0 when penalties are off."""
    return 0 if penalty_last_n <= 0 else min(penalty_last_n, cap)


def coupled_hist_start(cap, j, h):
    """Where draft j's penalty window starts in the ring (the base in [cap - h, cap), draft i at cap + i)."""
    return cap + j - h


def coupled_hist_base(tail, nxt, h):
    """The ring's base on the host: the last h tokens of (tail, nxt), -1 padded in front, the h slots the host stages
    in [cap - h, cap) (penalty_rows' row 0 for the one-token window [nxt]); empty when h <= 0."""
    return penalty_rows(tail, [nxt], 1, h) if h > 0 else np.empty(0, np.int32)


COUPLED_VOCABS = (106299, 40525, 4096, 4097, 300, 20, 7)


class CoupledCase(NamedTuple):
    """One coupled_parity case (:603-621): the draft vocabulary nv (subset index i is token id i + 25, of nv + 50 ids),
    the case number c, its logits (nv,), params, draft index j, the 264-slot ring and the step record's cell."""
    nv: int
    c: int
    logits: np.ndarray
    p: SamplerParams
    j: int
    ring: np.ndarray
    cell: int


def coupled_cases(seed=77):
    """coupled_parity's 420 cases (:579-621), in its order, from one RandomState(seed) over ``COUPLED_VOCABS``, 60 cases
    each. Case c: nv values of normal(0, 1 if c % 3 == 0 else 4), then roundf(v * 4) / 4 if c % 4 == 1 (exact ties),
    then l[0] at every third id if c % 7 == 2 (a wide tie); top_k (20, 0, 1, 64, 5, 20, 100)[c % 7], top_p 1 if
    c % 5 == 0 else 0.95, min_p 0.05 if c % 6 == 3 else 0, temperature 0.6, seed 1000 + c, and if c % 3 == 1 the
    penalties last_n 64, repeat 1.1, freq 0.05, present 0.1; j = c % 8; the ring's 264 slots -1 except 176 .. 255 + j,
    which hold subset tokens (a raw draw mod min(nv, 40), plus 25); then the cell, a raw draw mod 100,000. After the
    first vocabulary's cases the stream gives the coupled bench's normal(0, 3) row (:659-660): drawn and dropped, as
    basalt draws it before the next vocabulary."""
    rs = np.random.RandomState(seed)
    for nv in COUPLED_VOCABS:
        for c in range(60):
            l = _normal(rs, nv, 1.0 if c % 3 == 0 else 4.0)
            if c % 4 == 1: l = roundf(l * np.float32(4)) / np.float32(4)
            if c % 7 == 2: l[::3] = l[0]
            p = SamplerParams(top_k=(20, 0, 1, 64, 5, 20, 100)[c % 7], top_p=1.0 if c % 5 == 0 else 0.95,
                              min_p=0.05 if c % 6 == 3 else 0.0, temperature=0.6, seed=1000 + c)
            if c % 3 == 1: p = replace(p, penalty_last_n=64, penalty_repeat=1.1, penalty_freq=0.05, penalty_present=0.1)
            ring = np.full(264, -1, np.int32)
            ring[176:256 + c % 8] = raw32(rs, 80 + c % 8) % min(nv, 40) + 25
            yield CoupledCase(nv, c, l, p, c % 8, ring, int(raw32(rs, 1)[0] % 100000))
        if nv == 106299: _normal(rs, nv, 3.0)
