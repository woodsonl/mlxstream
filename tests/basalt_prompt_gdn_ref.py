# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/prefill/gdn_rec_parity.cu, src/kernels/dec_gdn_parity.cpp and
# src/kernels/elementwise_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE):
# gdn_rec_parity's fixture (mix, uni, gauss, fill_gauss, l2_rows, fill_gates :372-403, Inputs :484-505, the widths and
# seeds :1038-1041), its word count (count_diff :404-409, diff_words :475-482), close_to (:556-573) and the fp32
# arithmetic of its copies of gdn_rec_wl_kernel and gdn_out_norm_kernel (:236-352, with warp_sum, sigm and hf :44-50);
# cpu_seg's recurrence and gated norm (dec_gdn_parity.cpp:339-366), make_layer's and parity_seg's conv distributions
# (:130, :434, :437); elementwise_parity's gate fixture (:59-99) and double softplus gate (:78-80).
"""The prompt GDN kernels' host references and fixtures, numpy only (no Metal import), imported as ``from
tests.basalt_prompt_gdn_ref import ...``. The geometry, ``randn``, ``cpu_seg``, ``max_err``, ``n_bad`` and ``conv_tail``
are the gdn lane's (tests/basalt_gdn_ref.py) and are used from there, never copied: one float64 module of the GDN
arithmetic, each lane its own driver; so are the fp32 steps the emulation composes (tests/kernels/basalt_emu.py).

- ``gdn_prompt_ref``: the float64 recurrence and gated norm of dec_gdn_parity's ``cpu_seg`` (:339-366) on the prompt
  layout (spec T§4.3), from an h whose q and k heads are already L2-normed. Per token and value head hd, of q/k head
  ``QH[hd]`` = hd % 16: kv = S^T k on the old state, delta = (v - g kv) b with g = exp(gate), S = g S + k delta, oc =
  S^T q on the new state over sqrt(128); y = oc * scale * gamma * sigmoid(z), scale = 1 / sqrt(mean(oc^2) + EPS), left
  to right, rounded to float32 once. The un-normed oc (gdn_rec_wl_kernel's y) and the final state stay float64,
  unrounded, for W's ``close_to`` (gdn_rec_parity.cu:585-586); y is D-y16's reference (``max_err``).
- ``gates64``: gdn_gates_kernel's gate and beta (kernels.cu:207-216) in float64, elementwise_parity's reference.
- ``conv_l2_64``: D-conv's float64 h, ``cpu_seg``'s conv, SiLU and q/k L2 (dec_gdn_parity.cpp:320-338).
- ``close_to``: W's metric (gdn_rec_parity.cu:556-573); ``diff_words``: B-rec's and B-norm's count of differing raw
  32-bit words (:404-409, :475-482). Spec T§4.1-4.2.
- The fixtures. ``inputs(T, seed)``: gdn_rec_parity's ``Inputs`` (:484-505) in numpy (spec T§5.2), the hash generator
  ``mix``, ``uni``, ``gauss`` (:373-381) keyed by ``keys``, ``fill_gauss`` (:382-385), the L2 of ``l2_rows`` in float64,
  ``fill_gates`` (:397-403); basalt's widths ``TS`` with seeds ``SEED_B + i`` (B-rec, B-norm) and ``SEED_W + i`` (W).
  ``gates_inputs(T, seed)``: D-gates' (elementwise_parity.cpp:59-99 on T rows) and ``conv_inputs(T, seed)``: D-conv's
  (dec_gdn_parity's distributions), each on ``default_rng(seed)`` (spec K§9.2). P-edge's: ``edge_rec()`` (spec T§5.3,
  E1-E6), ``edge_gates()`` and ``edge_conv()`` (spec K§9.2's gate, beta, conv, L2 and history classes).
- The fp32 emulation (spec T§4.4), the bit-for-bit tests' reference, on the gdn lane's ``basalt_emu`` steps with
  ``exp2`` and ``rsqrt`` injected (the GPU's fast::exp2 and fast::rsqrt in the kernel tests): ``rec_emu``,
  gdn_rec_wl_kernel's oc and state (kernels.cu:486-585), each column's row sums as wl's 8 lanes of 16-row chains
  (``LANE_ROWS``) and their ``octet`` butterfly; ``out_norm_emu``, gdn_out_norm_kernel's y16 bits (:609-623). NaNs as
  sm_120 stores them (``cuda_nan``; 0x7fff).
"""

import math

import numpy as np

from tests.basalt_gdn_ref import C, EPS, F32, F64, HK, HV, REC, S, SF, ZV, cpu_seg, randn
from tests.kernels.basalt_emu import (RSQRT128, cuda_half_bits, expf_ieee, gdn_delta, gdn_kv, gdn_out, gdn_sq32,
                                      gdn_sum4, gdn_update)

QH = np.arange(HV) % HK                  # value head hd reads q/k head hd % HK (`const int qh = hd % HK;`)
U64, U32 = np.uint64, np.uint32
M64 = (1 << 64) - 1
TS = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 257, 1000, 4099)   # gdn_rec_parity.cu:1038
SEED_B, SEED_W = 11, 27                  # :1039-1041: check_bits(Ts[i], 11 + i), then check_warp(Ts[i], 27 + i)


def gdn_prompt_ref(h, gate, beta, z, gamma, state0):
    """float64 transcription of dec_gdn_parity.cpp:339-366 on the prompt layout, vectorized over the 48 value heads (the
    per-head sums in another order: in double a value moves by about 128 * 2^-53 relative). h (T, C) float32 with q and
    k L2-normed; gate, beta (T, HV) float32; z (T, HV, S) or (T, HV * S) float32; gamma (S,) float32; state0 (S, HV, S)
    float32, state[row][head][col]. Returns oc (T, HV, S) float64, y (T, HV, S) float32 and the final state (S, HV, S)
    float64."""
    h = np.asarray(h, F32).reshape(-1, C)
    T = h.shape[0]
    gate, beta = (np.asarray(a, F32).reshape(T, HV) for a in (gate, beta))
    z = np.asarray(z, F32).reshape(T, HV, S)
    sd = np.asarray(state0, F32).reshape(S, HV, S).astype(F64)
    gam, eps = np.asarray(gamma, F32).astype(F64), F64(EPS)          # `ss / S + EPS`: the float 1e-6f, promoted
    oc = np.empty((T, HV, S))
    y = np.empty((T, HV, S), F32)
    for t in range(T):
        ht = h[t].astype(F64)
        q = ht[:HK * S].reshape(HK, S)[QH]                  # q[hd, i] = ht[qh * S + i]
        k = ht[HK * S:2 * HK * S].reshape(HK, S)[QH]        # k[hd, i] = ht[qk + qh * S + i]
        v = ht[2 * HK * S:].reshape(HV, S)                  # v[hd, j] = ht[2 * qk + hd * S + j]
        g = np.exp(gate[t].astype(F64))[:, None]            # (HV, 1)
        b = beta[t].astype(F64)[:, None]
        kv = np.einsum("hi,ihj->hj", k, sd)                 # kv[hd, j] = sum_i sd[i, hd, j] * k[hd, i]
        delta = (v - g * kv) * b
        sd = g[None] * sd + k.T[:, :, None] * delta[None]   # v = g * v + k[i] * delta[j]
        oc[t] = np.einsum("hi,ihj->hj", q, sd) / math.sqrt(S)   # o on the updated state
        ss = np.einsum("hj,hj->h", oc[t], oc[t])[:, None]
        scale = 1.0 / np.sqrt(ss / S + eps)
        sig = 1.0 / (1.0 + np.exp(-z[t].astype(F64)))
        y[t] = (oc[t] * scale * gam * sig).astype(F32)
    return oc, y, sd


def gates64(alpha, beta_logit, dt, ssm_a):
    """gdn_gates_kernel's gate and beta (kernels.cu:207-216) in float64, as elementwise_parity.cpp:78-80 computes its
    want: u = alpha + dt summed in double, gate = (u > 20 ? u : log1p(exp(u))) * ssm_a; and beta = 1 / (1 + exp(-x)).
    alpha, beta_logit (T, HV) float32 (basalt's ab[t * 96 + h] and ab[t * 96 + 48 + h]); dt, ssm_a (HV,) float32.
    Returns (gate, beta), (T, HV) float64 each, unrounded: elementwise_parity casts its want to float once (:80), which
    the comparison does; rounded here, the branch would not show below exp's double overflow (log1p(exp(u)) is u to
    within e^-u)."""
    u = np.asarray(alpha, F32).astype(F64) + np.asarray(dt, F32).astype(F64)
    sp = np.where(u > 20.0, u, np.log1p(np.exp(np.minimum(u, 20.0))))
    return sp * np.asarray(ssm_a, F32).astype(F64), 1.0 / (1.0 + np.exp(-np.asarray(beta_logit, F32).astype(F64)))


def conv_l2_64(state, qkv, conv_w):
    """D-conv's float64 h (spec K§9.2): ``cpu_seg``'s conv and SiLU, rounded to float32, then the q/k L2 from those
    floats, rounded again (dec_gdn_parity.cpp:320-338), over the rows of qkv (T, C) from the conv history in ``state``
    (SF floats, the history [C][3] at REC, oldest first) with conv_w (C, 4). Taken as the gdn lane's tests take it
    (test_basalt_dec_gdn.py:218-219): its recurrence fed a zero gate and beta, whose y nothing reads."""
    n = len(qkv)
    nil = np.zeros((n, HV), F32)
    return cpu_seg(state, qkv, np.zeros((n, ZV), F32), conv_w, np.ones(S, F32), nil, nil, n).h


def close_to(x, ref):
    """close_to (gdn_rec_parity.cu:556-573), W's metric: (the relative RMS of x - ref, max |x - ref| / max |ref|) in
    float64; where ref is all zero, sqrt(sum (x - ref)^2) and max |x - ref| instead (``r2 > 0 ? std::sqrt(d2 / r2) :
    std::sqrt(d2)``); (nan, nan) if the candidate x has a value that is not finite. The reference is not tested: a NaN
    or infinite one makes the RMS NaN, which W's ``std::isfinite(co.rms)`` fails. The maxima fold as ``std::max(dm,
    std::fabs(d))`` does, which drops a NaN (np.fmax; spec T§4.1's ``.max`` would return it, against its own text);
    the sums run in BLAS order, which moves the RMS by about n 2^-53 relative (spec T§4.1)."""
    x = np.asarray(x, F32).ravel()
    if not np.isfinite(x).all():
        return math.nan, math.nan
    y = np.asarray(ref).astype(F64).ravel()
    d = x.astype(F64) - y
    d2, r2 = float(d @ d), float(y @ y)
    dm, rm = (float(np.fmax.reduce(np.abs(v), initial=0.0)) for v in (d, y))
    return (math.sqrt(d2 / r2) if r2 > 0 else math.sqrt(d2)), (dm / rm if rm > 0 else dm)


def diff_words(a, b):
    """diff_words (gdn_rec_parity.cu:475-482, count_diff :404-409): how many raw 32-bit words of two arrays of one size
    differ (-0 and +0 differ, two NaNs of one pattern agree); FP16 arrays compare as pairs of halves, B-norm's unit."""
    a, b = (np.ascontiguousarray(v).reshape(-1).view(U32) for v in (a, b))
    assert a.shape == b.shape, f"{a.size} words against {b.size}"
    return int(np.count_nonzero(a != b))


def mix(x):                                      # gdn_rec_parity.cu:373-376, x a uint64 array (wraps mod 2^64 as C)
    x = x ^ (x >> U64(33)); x = x * U64(0xFF51AFD7ED558CCD); x = x ^ (x >> U64(33))
    x = x * U64(0xC4CEB9FE1A85EC53); x = x ^ (x >> U64(33))
    return (x & U64(0xFFFFFFFF)).astype(U32)


def uni(key):                                    # :377: exact int -> float, a float32 add (ties to even), exact scale
    return ((mix(key) >> U32(8)).astype(F32) + F32(0.5)) * F32(2.0 ** -24)


def gauss(key):                                  # :378-381, numpy's float32 log and cos for CUDA's logf, cosf
    u1, u2 = uni(key * U64(2) + U64(1)), uni(key * U64(2) + U64(2))
    return np.sqrt(F32(-2.0) * np.log(u1)) * np.cos(F32(6.2831853) * u2)


def keys(seed, mult, n, stride=1):               # seed * mult + stride * i, mod 2^64
    return U64((seed * mult) & M64) + U64(stride) * np.arange(n, dtype=U64)


def fill_gauss(n, scale, seed, offset=0.0):      # :382-385
    return (F32(offset) + F32(scale) * gauss(keys(seed, 0x9E3779B97F4A7C15, n))).astype(F32)


def inputs(T, seed):
    """gdn_rec_parity's Inputs(T, seed) (:484-505) in numpy (spec T§5.2): h (T, C) N(0, 1) with each q and k head of a
    token L2-normed (l2_rows, :386-395, in float64 and rounded once), gate (T, HV) = -0.02 - 0.8 u^3 and beta (T, HV) =
    sigmoid(N(0, 1)) (fill_gates, :396-403), z (T, HV, S) N(0, 1), gamma (S,) 1 + 0.1 N(0, 1) and state0 (S, HV, S)
    0.05 N(0, 1), state[row][head][col]; float32, keyed by seed .. seed + 4 as basalt keys them. The same generator,
    so each test maps to basalt's (T, seed), but not basalt's bits (CUDA's logf, cosf, expf and rsqrtf; spec T§5.1):
    the GPU tests and the references read these arrays."""
    h = fill_gauss(T * C, 1.0, seed).reshape(T, C)
    qk = h[:, :2 * HK * S].reshape(T, 2 * HK, S).astype(np.float64)                  # l2_rows (:386-395), in float64
    h[:, :2 * HK * S] = (qk / np.sqrt((qk * qk).sum(-1, keepdims=True) + 1e-6)).reshape(T, -1)
    u = uni(keys(seed + 1, 0x9E3779B97F4A7C15, T * HV, stride=7))                     # fill_gates (:396-403)
    gate = (F32(-0.02) - F32(0.8) * u * u * u).reshape(T, HV)
    beta = (F32(1) / (F32(1) + np.exp(-gauss(keys(seed + 1, 0xD1B54A32D192ED03, T * HV))))).reshape(T, HV)
    z = fill_gauss(T * HV * S, 1.0, seed + 2).reshape(T, HV, S)
    gamma = fill_gauss(S, 0.1, seed + 3, 1.0)
    state0 = fill_gauss(S * HV * S, 0.05, seed + 4).reshape(S, HV, S)
    return h, gate, beta, z, gamma, state0


def gates_inputs(T, seed):
    """D-gates' fixture (spec K§9.2): elementwise_parity's (:59-99) on T rows, from default_rng(seed). Over the flat
    index i = t * HV + h, the large heads i % 3 == 0 (every third head of every row) take alpha = 22 + 6 (i % 17), so u
    = alpha + dt crosses 20 and reaches about 120, past float's exp overflow at 88 (:65-74); the others 3 N(0, 1); per
    head dt = 0.5 N(0, 1) and ssm_a = -(|N(0, 1)| + 0.1), negative (:75-77); beta logits 3 N(0, 1). Returns alpha,
    beta_logit (T, HV) (basalt's ab[t * 96 + h] and ab[t * 96 + 48 + h], two buffers here) and dt, ssm_a (HV,),
    float32. Both properties are asserted as basalt asserts them (:82-99), in the contract test."""
    rng = np.random.default_rng(seed)
    dt = randn(rng, HV, 0.5)
    ssm_a = -(np.abs(randn(rng, HV, 1.0)) + F32(0.1))
    i = np.arange(T * HV).reshape(T, HV)
    alpha = np.where(i % 3 == 0, F32(22) + F32(6) * (i % 17).astype(F32), randn(rng, (T, HV), 3.0))
    return alpha, randn(rng, (T, HV), 3.0), dt, ssm_a


def conv_inputs(T, seed):
    """D-conv's fixture (spec K§9.2) in dec_gdn_parity's distributions (make_layer :130, parity_seg :434, :437), from
    default_rng(seed): conv_w (C, 4) N(0, 0.5), a state of SF floats N(0, 0.1), whose conv history (the 3 C floats at
    REC, [C][3], oldest first) is non-zero, then qkv (T, C) N(0, 1); conv_w and the state do not depend on T."""
    rng = np.random.default_rng(seed)
    conv_w, state = randn(rng, (C, 4), 0.5), randn(rng, SF, 0.1)
    return randn(rng, (T, C), 1.0), conv_w, state


def edge_rec():
    """P-edge's recurrence and norm fixture (spec T§5.3): inputs(3, 43), the seed after basalt's last, with
    - E1, a subnormal decay: gate -95 at token 0 on value heads 0, 16 and 32 (__expf's t = -137.06 < -126: halved,
      MUFU.EX2, squared to g of about 2^-137) and their key head 0's k = 0 at token 0, so p = k delta = 0, the stored
      state is g s (about 2^-141) and oc a sum of subnormal products; E5 is the norm on those rows (ss underflows to 0);
    - E2, the subnormal state carried into the next token: v = 0 at token 1 on value head 0, columns 0..63;
    - E3, a subnormal initial state: value head 1's state0[:, 1, :] = 1e-39 N(0, 1) (default_rng(43), rounded once),
      and v = 0 on value head 1 at token 0;
    - E4, the norm's extremes: z = -200, -88, 88, 200, -inf, +inf on token 2, value head 3, columns 0..5;
    - E6, a NaN: v = NaN at token 2, value head 2, column 7.
    Returns inputs()'s six arrays."""
    h, gate, beta, z, gamma, state0 = inputs(3, 43)
    V = 2 * HK * S                                                   # v of value head hd: h[t, V + hd * S + j]
    gate[0, [0, HK, 2 * HK]] = -95.0                                 # E1
    h[0, HK * S:HK * S + S] = 0.0                                    # E1: k of key head 0 at token 0
    h[1, V:V + 64] = 0.0                                             # E2
    state0[:, 1, :] = 1e-39 * np.random.default_rng(43).standard_normal((S, S))   # E3
    h[0, V + S:V + 2 * S] = 0.0                                      # E3
    z[2, 3, :6] = [-200.0, -88.0, 88.0, 200.0, -np.inf, np.inf]      # E4
    h[2, V + 2 * S + 7] = np.nan                                     # E6
    return h, gate, beta, z, gamma, state0


def edge_gates():
    """P-edge's gates fixture (spec K§2, §9.2): four rows, dt = 0 on every head but the last (the subnormal 2^-140,
    so u = alpha + dt is alpha elsewhere, and +0 for either zero alpha), ssm_a = -0.3, -0.75, -1.5, -3 by head (both
    sides of 1), the other values 3 N(0, 1) from default_rng(43):
    - row 0: u through [-104, -87.3], __expf's halved branch (t = u log2(e) < -126 below -87.34): a subnormal e, which
      log1pf passes on, times ssm_a; beta logits through [-88.7, -87.35]: 1 + e above 2^126, a subnormal beta;
    - row 1: u on both sides of the -126 test and of u > 20 (20 itself takes log1pf), past 88 (where log1pf(exp(u))
      overflows), +-inf, NaN, alpha +-0; beta logits on both sides of -87.34 and -88.72, far below, +-inf, NaN, +-0;
    - row 3: a subnormal alpha on the last head (a subnormal sum u) and subnormal beta logits.
    Returns alpha, beta_logit (4, HV), dt, ssm_a (HV,), float32."""
    rng = np.random.default_rng(43)
    alpha, logit = randn(rng, (4, HV), 3.0), randn(rng, (4, HV), 3.0)
    alpha[0], logit[0] = np.linspace(-104.0, -87.3, HV), np.linspace(-88.7, -87.35, HV)
    u1 = [-87.4, -87.35, -87.34, -87.336, -87.33, -87.3, -86.5, -1.0, 0.0, -0.0, 1.0, 19.5, 20.0,
          np.nextafter(F32(20), F32(np.inf)), 20.25, 22.0, 50.0, 87.9, 88.0, 88.8, 100.0, 120.0, 1e4, np.inf, -np.inf,
          np.nan]
    b1 = [-87.3, -87.33, -87.34, -87.35, -88.7, -88.72, -88.73, -88.8, -90.0, -104.0, -200.0, -np.inf, np.inf, np.nan,
          0.0, -0.0, 87.3, 88.8, 104.0]
    alpha[1, :len(u1)], logit[1, :len(b1)] = u1, b1
    alpha[3, -1], logit[3, :2] = 2.0 ** -139, [2.0 ** -140, -(2.0 ** -140)]
    dt = np.zeros(HV, F32)
    dt[-1] = 2.0 ** -140
    return alpha, logit, dt, -np.array([0.3, 0.75, 1.5, 3.0], F32)[np.arange(HV) % 4]


def edge_conv():
    """P-edge's conv, history and L2 fixture (spec K§3-§4, §9.2): conv_inputs(5, 43), R = 5 rows, with (channel j of
    head n at n * S + j: q heads n < 16, k heads 16 <= n < 32, v heads 32 <= n < 80; hist[c] the old history, oldest
    first; the other values from default_rng(44))
    - v head 0: taps (0, 0, 0, 1), so s = x: x through [86, 90] at row 0 (above 87.34: __expf(-s)'s halved branch, a
      subnormal e that 1 + e absorbs, h = s), through [-90, -86] at row 1 (below -88.72: e = inf, h = -0), and
      +-[2^-126, 2^-125) on channels 0..7 at row 2 (a normal s whose quotient h = s / 2 is subnormal: spec K§3's other
      `s / d` case);
    - v head 1, subnormal products and sums, by channel mod 3: history and inputs 2^-140 N(0, 1) (subnormal) on the N(0,
      0.5) taps; history, inputs and taps 2^-66 N(0, 1) (normal operands, subnormal products); taps 2^-140 N(0, 1) on
      the N(0, 1) inputs; so s, and h = s / 2, are subnormal;
    - q head 3 and k head 3 at row 0, below 2^-63: zero history, inputs 2^-68 N(0, 1) (subnormal squares in the L2),
      every eighth 2^-128 N(0, 1) (a subnormal h whose product with the L2 scale, about 1000, is normal);
    - NaN and infinities: q head 7 channel 5 at row 1 (that head's L2 rows NaN), v head 2 channel 9 at row 0, the old
      history of v head 2 channel 15; -inf and +inf at row 0 on v head 2 channels 11 and 13, whose tap 3 is 0.5 (s =
      -inf gives h = NaN, s = +inf h = inf);
    - rows R-3..R-1 on v head 3: row R-3+k holds a subnormal word at channel 2k and the NaN word 0x7fa00001 (a payload
      only a word move keeps) at channel 2k + 1, which the history kernel copies (kernels.cu:254-263).
    Returns qkv (R, C), conv_w (C, 4), state (SF), float32."""
    R = 5
    qkv, w, state = conv_inputs(R, 43)
    hist = state[REC:].reshape(C, 3)                                 # a view of the state's history
    rng = np.random.default_rng(44)
    tiny = lambda e, *shape: (2.0 ** e * rng.standard_normal(shape)).astype(F32)   # rounded once, subnormals kept
    head = lambda n: np.arange(n * S, n * S + S)
    v0, v1, v2, v3 = (head(2 * HK + k) for k in range(4))
    w[v0] = [0.0, 0.0, 0.0, 1.0]
    qkv[0, v0], qkv[1, v0] = np.linspace(86.0, 90.0, S), np.linspace(-90.0, -86.0, S)
    qkv[2, v0[:8]] = (-1.0) ** np.arange(8) * (1.0 + np.arange(8) / 8) * 2.0 ** -126
    sub_x, small, sub_w = v1[0::3], v1[1::3], v1[2::3]
    hist[sub_x], qkv[:, sub_x] = tiny(-140, len(sub_x), 3), tiny(-140, R, len(sub_x))
    hist[small], qkv[:, small], w[small] = tiny(-66, len(small), 3), tiny(-66, R, len(small)), tiny(-66, len(small), 4)
    w[sub_w] = tiny(-140, len(sub_w), 4)
    for n in (3, HK + 3):
        ch = head(n)
        hist[ch], qkv[0, ch] = 0.0, tiny(-68, S)
        qkv[0, ch[::8]] = tiny(-128, S // 8)
    qkv[1, head(7)[5]] = qkv[0, v2[9]] = hist[v2[15], 1] = np.nan
    qkv[0, v2[[11, 13]]], w[v2[[11, 13]], 3] = [-np.inf, np.inf], 0.5
    words = qkv.view(U32)
    for k in range(3):
        words[R - 3 + k, v3[2 * k]] = (0x00000001, 0x80000005, 0x007FFFFF)[k]
        words[R - 3 + k, v3[2 * k + 1]] = 0x7FA00001
    return qkv, w, state


# ---- the fp32 emulation (spec T§4.4), the bit-for-bit tests' reference ----------------------------------------------
# The gdn lane's basalt_emu steps, numpy's IEEE float32 arithmetic with subnormals kept as the IEEE-built kernels.cu,
# and the injected exp2 and rsqrt: the GPU's fast::exp2 and fast::rsqrt values, or numpy stand-ins on the host.

LANE_ROWS = np.array([[4 * rg + 32 * m + e for m in range(4) for e in range(4)] for rg in range(8)])
# wl's lane rg of a column's octet: its 16 state rows in chain order, r = 4 m + e (`s[j][4 * m + e]`, kernels.cu:506)


def cuda_nan(x):
    """x with every NaN as sm_120's arithmetic stores it, 0x7fffffff (the kernels' cuda_nan stores; nan.md)."""
    x = np.asarray(x, F32)
    return np.where(np.isnan(x), U32(0x7FFFFFFF), x.view(U32)).view(F32)


def octet(p):
    """wl's butterfly over a column's 8 row-group lanes p (8, ...) (kernels.cu:557-559, :570-572): for x = 1, 2, 4 every
    lane adds its partner's value, FADD own, partner; lane 0's result ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
    is every lane's (each add commutes)."""
    with np.errstate(over="ignore", invalid="ignore"):
        for x in (1, 2, 4):
            p = p + p[np.arange(8) ^ x]
    return p[0]


def rec_emu(h, gate, beta, state0, exp2):
    """gdn_rec_wl_kernel's output oc (the un-normed y) and final state in float32 (kernels.cu:486-585; the SASS rows of
    docs/research/basalt-nvcc/prefill_kernels.md section 2), every value head and column at once, the state held
    lane-major as st[rg, hd, j, r] = state[LANE_ROWS[rg, r], hd, j]. Per token: g = __expf(gate) (expf_ieee); kv =
    lane rg's 16-row FFMA chain from +0 (gdn_kv), then the octet butterfly; delta = fma(-g, kv, v) * beta
    (gdn_delta); s = fma(g, s, k * delta) and o = the lane chain over the updated rows (gdn_update), then the octet
    butterfly, times 0x3DB504F3. Inputs as gdn_prompt_ref's; returns oc (T, HV, S) and the state (S, HV, S), every NaN
    0x7fffffff."""
    h = np.asarray(h, F32).reshape(-1, C)
    T = h.shape[0]
    gate, beta = (np.asarray(a, F32).reshape(T, HV) for a in (gate, beta))
    lanes = lambda x: np.moveaxis(x.reshape(HK, S)[QH][:, LANE_ROWS], 1, 0)[:, :, None]    # (8, HV, 1, 16)
    st = np.moveaxis(np.asarray(state0, F32).reshape(S, HV, S)[LANE_ROWS], 1, -1)        # (8, HV, S, 16)
    oc = np.empty((T, HV, S), F32)
    for t in range(T):
        q, k = lanes(h[t, :HK * S]), lanes(h[t, HK * S:2 * HK * S])
        g = expf_ieee(gate[t], exp2)[:, None]
        delta = gdn_delta(g, octet(gdn_kv(st, k)), h[t, 2 * HK * S:].reshape(HV, S), beta[t][:, None])
        st, o = gdn_update(st, k, q, g, delta)
        with np.errstate(over="ignore", invalid="ignore"):
            oc[t] = octet(o) * RSQRT128
    out = np.empty((S, HV, S), F32)
    out[LANE_ROWS] = np.moveaxis(st, -1, 1)
    return cuda_nan(oc), cuda_nan(out)


def out_norm_emu(oc, z, gamma, exp2, rsqrt):
    """gdn_out_norm_kernel's y16 bits, uint16 (T, HV, S) (kernels.cu:609-623, prefill_kernels.md section 2), every
    (token, value head) row at once, from oc (T, HV, S) float32: each warp's 32 squares summed by warp_sum's xor 16 .. 1
    butterfly (gdn_sq32), the four warp sums left to right (gdn_sum4), then gdn_out's gated norm (fma(ss, 2^-7, eps),
    rsqrt_ieee, the products by gamma and sigm(z) left to right), rounded to FP16 with a NaN as 0x7fff
    (cuda_half_bits). z as gdn_prompt_ref's."""
    oc = np.asarray(oc, F32).reshape(-1, HV, S)
    ss = gdn_sum4(*gdn_sq32(oc.reshape(-1, 32)).reshape(-1, 4).T).reshape(-1, HV, 1)
    return cuda_half_bits(gdn_out(oc, ss, EPS, gamma, np.asarray(z, F32).reshape(oc.shape), exp2, rsqrt))
