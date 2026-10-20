# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp, src/kernels/ple_win_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's metric helpers (le, gt,
# nonfinite and rel_l1, :41-77) and the seven stages its block check compares (:751-759); ple_win_parity's fixture
# scales (:48-56).
"""The n-gram (PLE) block's host references, numpy only (no Metal import), imported as ``from tests.basalt_ple_ref
import ...``.

- ``le``, ``gt``, ``nonfinite`` and ``rel_l1``: ple_parity's NaN-safe metric helpers (ple_parity.cpp:41-77).
  ``rel_l1(a, b)`` takes the reference first and returns ``(rel, mag, nf)``: rel = sum |a - b| / sum |a| (the sum
  floored at 1e-30) in float64, NaN when any pair has a non-finite entry; mag = mean |a| (the finite pairs' sum over
  the length); nf = the count of non-finite pairs.
- R-struct (``r_struct``, spec T§6.2): the block in float64, from the graph comment (basalt
  include/basalt/kernels/ple.hpp:3-13), not from basalt's host f32 reference. The Q2_0 key is decoded by
  tests/contract/gguf_reference.py's ``q2_0``; BF16, F16 and F32 are widened, exactly. For every row it returns the
  capture's seven slices (ple_parity.cpp:171-172: the normed key, value, gate, gated, normalized, conv_out and result),
  the history after the row and the gate's argument s. The window is one pad, P = [history | normalized rows], read
  at rows t, t + 3, t + 6 and t + 9 (ple.hpp's ``padded[hist + t - (kern - 1 - k) * dil]``), so the 12 rows run from
  one initial history. ``r_contract`` (R-contract) is R-struct entered after the projections, fed a raw key and value.
- The synthetic fixture (``synthetic``, spec T§6.3), ``Fixture``: the weights in the encodings the GPU reads (the key
  as type-42 rows, the value as BF16 bits, the conv kernel as F16 in ggml's flat [k + 4 i]) and float32 activations;
  the history is logical, row r oldest first (its row-fastest buffer layout, hist[r + 9 c], is the harness's).
  Seeds: ``default_rng(SYNTHETIC_SEED)`` draws, in order, the key's codes (uniform bytes: four uniform 2-bit codes
  each) and its d = f16(0.02 * U(0.5, 1.5)) per 64-weight block, w_nk, w_nq and w_nc = f32(1 + 0.1 * N(0,1)) (in that
  order), conv1d = f16(0.3 * N(0,1)), emb = f32(0.2 * N(0,1)) (12 x 2560), the history f32(N(0,1)) (9 x 10240) and
  hidden's unit draw z (12 x 10240); the value is bf16(0.05 * N(0,1)) from ``default_rng(VALUE_SEED)``, the
  first (2560, 2560) draw, bit for bit test_ple_value_float64's weights (tests/kernels/test_basalt_mmvf.py). hidden =
  f32(sigma_h * z), with sigma_h by T§6.3's rule 3: R-struct with the unit-scale hidden f32(z), then sigma_h = mean
  |gated + conv_out| / mean |f32(z)|. ``SYNTHETIC_SEED`` = 54 is the first seed from 0 for which property 2 holds
  (seeds 0-53 put some |s[c]| below 0.1).
- ``check_fixture``: the four properties every generator asserts on R-struct before it returns (T§6.3):
  1. hidden is nonzero (every row's every stream: with hidden = 0, s = 0 and gate = 0.5, the key is never observed);
  2. min over rows and streams of |s[c]| >= 0.1;
  3. the block's share of result, sum |gated + conv_out| / sum |result|, is at least 0.3 on every row;
  4. T = 12 rows from one initial history: each row's normalized row enters the history the next rows read."""

import functools
from typing import NamedTuple

import numpy as np

from tests.contract.gguf_reference import q2_0
from tests.kernels.basalt_emu import bf16_bits

F32, F64 = np.float32, np.float64
N, H = 2560, 4                  # NG_N_EMBD, NG_HC (basalt include/basalt/kernels/ngram.hpp:30-41)
D = H * N                       # NG_HC_DIM: a [D] vector is stream-major, element (c, d) at c * N + d
KERN, DIL = 4, 3                # PLE_CONV_KERNEL, NGRAM_SIZE
HIST = (KERN - 1) * DIL         # NG_HIST: 9 history rows, oldest first
ROWS = 12                       # T§6.1: every block-level comparison runs 12 rows from one initial history
EPS = float(F32(1e-6))          # NG_RMS_EPS as float64(float32(1e-6)): the norms' eps and the gate's clamp
KEY_BLOCK = 18                  # a type-42 (Q2_0) block: the f16 d, then 16 bytes of 2-bit codes, for 64 weights
SYNTHETIC_SEED = 54             # the first seed from 0 for which property 2 holds: min |s| 0.1251, min row share 0.628
VALUE_SEED = 790                # test_ple_value_float64's


def le(x, t):
    return bool(np.isfinite(x) and x <= t)       # ple_parity.cpp:48


def gt(x, t):
    return bool(np.isfinite(x) and x > t)        # ple_parity.cpp:49


def nonfinite(a):
    """ple_parity.cpp:52-57: how many entries of ``a`` are not finite."""
    return int(np.count_nonzero(~np.isfinite(a)))


def rel_l1(a, b):
    """ple_parity.cpp:64-77, ``a`` the reference: (rel, mag, nf) as the module docstring says."""
    a, b = np.asarray(a, F64).ravel(), np.asarray(b, F64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    nf = int(np.count_nonzero(~ok))
    m = np.abs(a[ok]).sum()
    mag = m / (a.size or 1)
    if nf:
        return np.nan, mag, nf
    return np.abs(a - b).sum() / (m if m > 1e-30 else 1e-30), mag, 0


class Fixture(NamedTuple):
    key: np.ndarray       # uint8 (D, 720): W_key's type-42 rows, row o = W_key[o, :] in 40 blocks of 64
    value: np.ndarray     # uint16 (N, N): W_value's BF16 bits, row-major
    w_nk: np.ndarray      # float32 (D,): the key norm's gamma
    w_nq: np.ndarray      # float32 (D,): the query norm's
    w_nc: np.ndarray      # float32 (D,): the conv input norm's
    conv1d: np.ndarray    # float16 (KERN * D,): ggml's flat [k + 4 i]
    emb: np.ndarray       # float32 (ROWS, N): the gathered n-gram rows
    hist: np.ndarray      # float32 (HIST, D): the initial history, row r oldest first
    hidden: np.ndarray    # float32 (ROWS, D): the residual stack


class Ref(NamedTuple):
    key: np.ndarray         # (T, D): gnorm(W_key @ emb, w_nk)
    value: np.ndarray       # (T, N): W_value @ emb
    gate: np.ndarray        # (T, H)
    gated: np.ndarray       # (T, D)
    normalized: np.ndarray  # (T, D): gnorm(gated, w_nc), the conv input
    conv_out: np.ndarray    # (T, D): SiLU(conv)
    result: np.ndarray      # (T, D): hidden + gated + conv_out
    hist: np.ndarray        # (T, HIST, D): the history after row t, P[t + 1 .. t + 9]
    s: np.ndarray           # (T, H): the gate's argument


@functools.lru_cache(maxsize=2)
def _key_weights(key):
    """W_key (D, N) in float32 as gguf_reference.q2_0 decodes ``key``, the bytes of the type-42 rows. The decode is a
    scalar loop of about 10 s, so it is cached per key, read-only."""
    w = q2_0(np.frombuffer(key, np.uint8).reshape(-1, KEY_BLOCK)).reshape(D, N)
    w.flags.writeable = False
    return w


def bf16_values(bits):
    """BF16 bits widened to float32, exactly."""
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(F32)


def gnorm(x, w):
    """T§6.2's grouped norm over each of the H streams of N, in float64:
    x[c, d] / sqrt(mean_d(x[c, :]^2) + eps) * w[c, d]."""
    x = np.asarray(x, F64).reshape(-1, H, N)
    y = x / np.sqrt(np.mean(x * x, axis=2, keepdims=True) + EPS) * np.asarray(w, F64).reshape(H, N)
    return y.reshape(-1, D)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x / (1 + np.exp(-x))


def taps(conv1d):
    """w_conv[i, k] = conv1d flat [k + 4 i] (ggml's ne = (kern, hc_dim)), float64."""
    return np.asarray(conv1d, F64).reshape(D, KERN)


def conv(P, w):
    """conv[t, i] = sum_{k=0..3} w[i, k] * P[t + 3k, i] over the padded rows P = [history | normalized rows], tap 0 the
    furthest back (ple.hpp's ``padded[hist + t - (kern - 1 - k) * dil]``)."""
    t = len(P) - HIST
    return sum(w[:, k] * P[DIL * k:DIL * k + t] for k in range(KERN))


def r_contract(fx, key_raw, value):
    """R-struct after the projections: ``key_raw`` (T, D) and ``value`` (T, N) are given (the GPU's, or R-struct's own),
    every other input is ``fx``'s, and the T rows run from ``fx.hist``."""
    t = len(key_raw)
    key = gnorm(key_raw, fx.w_nk)
    query = gnorm(fx.hidden, fx.w_nq)
    value = np.asarray(value, F64).reshape(t, N)
    s = (key * query).reshape(t, H, N).sum(axis=2) / np.sqrt(N)
    gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), EPS)))      # sign(0) = 0
    gated = (value[:, None, :] * gate[:, :, None]).reshape(t, D)
    normalized = gnorm(gated, fx.w_nc)
    P = np.concatenate([np.asarray(fx.hist, F64), normalized])
    conv_out = silu(conv(P, taps(fx.conv1d)))
    result = np.asarray(fx.hidden, F64) + gated + conv_out
    hist = np.stack([P[r + 1:r + 1 + HIST] for r in range(t)])
    return Ref(key, value, gate, gated, normalized, conv_out, result, hist, s)


def r_struct(fx):
    """T§6.2's R-struct of ``fx``, every row."""
    emb = np.asarray(fx.emb, F64)
    key_raw = emb @ _key_weights(fx.key.tobytes()).astype(F64).T
    value = emb @ bf16_values(fx.value).astype(F64).T
    return r_contract(fx, key_raw, value)


def check_fixture(fx, ref):
    """T§6.3's four properties of a fixture, asserted on its R-struct ``ref``; returns min |s| and the smallest row
    share, for the log."""
    chained = (fx.emb.shape == (ROWS, N) and fx.hidden.shape == (ROWS, D) and fx.hist.shape == (HIST, D)
               and ref.hist.shape == (ROWS, HIST, D) and (ref.hist[0, :-1] == fx.hist[1:]).all()
               and (ref.hist[:, -1] == ref.normalized).all() and (ref.hist[-1] == ref.normalized[ROWS - HIST:]).all())
    assert chained, ("T§6.3 property 4, \"T = 12 rows from one initial history\": the fixture has 12 rows and one "
                     "9-row history, and each row's normalized row enters the history the next rows read")
    zero = int(np.count_nonzero(~(np.abs(fx.hidden.reshape(ROWS, H, N)).max(axis=2) > 0)))
    assert zero == 0, (f"T§6.3 property 1, \"hidden is nonzero\": {zero} of {ROWS * H} row streams are zero (with "
                       "hidden = 0, query = 0, s = 0 and gate = 0.5 exactly, so the key is never observed)")
    min_s = float(np.abs(ref.s).min())
    assert min_s >= 0.1, f"T§6.3 property 2, \"min over rows and streams of |s[c]| ≥ 0.1\": {min_s!r}"
    share = np.abs(ref.gated + ref.conv_out).sum(axis=1) / np.abs(ref.result).sum(axis=1)
    assert (share >= 0.3).all(), ("T§6.3 property 3, \"The block's share of result is at least 0.3. The share is "
                                  f"Σ|gated + conv_out| / Σ|result| per row\": {np.round(share, 4).tolist()}")
    return min_s, float(share.min())


def _with_activations(rng, emb_scale, **weights):
    """The fixture of ``weights`` with T§6.3's activations from ``rng``: emb = f32(emb_scale * N(0,1)), the history
    f32(N(0,1)), hidden = f32(sigma_h * z) by rule 3; its four properties asserted."""
    emb = (emb_scale * rng.standard_normal((ROWS, N))).astype(F32)
    hist = rng.standard_normal((HIST, D)).astype(F32)
    z = rng.standard_normal((ROWS, D))
    fx = Fixture(emb=emb, hist=hist, hidden=z.astype(F32), **weights)
    ref = r_struct(fx)
    sigma = np.abs(ref.gated + ref.conv_out).mean() / np.abs(fx.hidden.astype(F64)).mean()
    fx = fx._replace(hidden=(sigma * z).astype(F32))
    check_fixture(fx, r_struct(fx))
    return fx


def synthetic(seed=SYNTHETIC_SEED):
    """T§6.3's synthetic fixture, drawn as the module docstring says."""
    rng = np.random.default_rng(seed)
    blocks = N // 64
    codes = rng.integers(0, 256, (D, blocks, 16), dtype=np.uint8)
    d = (0.02 * rng.uniform(0.5, 1.5, (D, blocks))).astype(np.float16)
    key = np.concatenate([d.view(np.uint8).reshape(D, blocks, 2), codes], axis=2).reshape(D, blocks * KEY_BLOCK)
    value = bf16_bits(np.random.default_rng(VALUE_SEED).standard_normal((N, N)).astype(F32) * F32(0.05))
    w_nk, w_nq, w_nc = ((1 + 0.1 * rng.standard_normal(D)).astype(F32) for _ in range(3))
    conv1d = (0.3 * rng.standard_normal(KERN * D)).astype(np.float16)
    return _with_activations(rng, 0.2, key=key, value=value, w_nk=w_nk, w_nq=w_nq, w_nc=w_nc, conv1d=conv1d)
