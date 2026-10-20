# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_gdn_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its geometry, its layer and segment fixtures, cpu_seg and max_err.
"""The GDN window's host references and fixtures, numpy only (no Metal import), imported as ``from tests.basalt_gdn_ref
import ...``.

- The geometry of dec_gdn_parity (:42-43): n_embd ``N``, ``HK`` q/k heads, ``HV`` value heads of ``S``, ``C`` conv
  channels (q, k, then v), ``ZV`` output columns, ``EPS``; ``MT`` window rows (kDecMaxT); ``SF`` floats of one layer's
  state: ``REC`` of the S x HV x S recurrent state, element (i, hd, j) at (i * HV + hd) * S + j, then the conv history
  [C][3], oldest first; ``NL`` layers per segment state, of which the kernels read layer ``LAYER`` (:415-416).
- ``make_layer`` (:124-145) in its draw order on a numpy Generator: the projections are test_basalt_mmvq.py's
  ``weights`` (random bytes, tamed fp16 scales), imported when it runs; ``COMBOS`` are parity_gdnf's projection types
  (:271) with IQ3_S, which neither the pack nor the repository's mmvq reads, replaced by Q3_K (spec T§5.1). ``randn``
  is basalt's.
- The fixtures, each from its own seeded stream (basalt's mt19937 draws cannot be replayed; only the distributions,
  shapes and layouts carry over, spec T§2.3): ``combo_case(i)``, parity_gdnf's layer and 8-row x for combo i (seed
  43); ``seg_layer()`` and ``segments(trial, nseg)``, parity_seg's layer and its 12 configurations (seed 97).
- The references: ``cpu_seg`` (:309-368) in float64, vectorized per token over heads and columns; ``max_err``
  (:403-407) and ``n_bad``, the count of non-finite values every float64 check asserts is 0 (``std::max`` drops a
  NaN); ``ab64``, the port's float64 gates (spec T§5.5 A1: basalt grades them against nothing); ``conv_tail``, the
  conv history after a window's rows, a bit copy (the one-token calls' history, spec T§5.1).
"""

from typing import NamedTuple

import numpy as np

from monolith.formats.fp import bf16_to_f32
from tests.kernels.basalt_emu import bf16_bits

F32, F64 = np.float32, np.float64
N, HK, HV, S = 2560, 16, 48, 128
C = 2 * HK * S + HV * S                  # 10240: q at [0, 2048), k at [2048, 4096), v at [4096, 10240)
ZV = HV * S                              # 6144
EPS = F32(1e-6)
MT = 8                                   # kDecMaxT, kVerifyMaxT
REC = S * HV * S                         # the recurrent state's floats; the conv history follows
SF = REC + C * 3                         # 817,152
NL, LAYER = 2, 1                         # layers per segment state; the kernels read layer 1 (:415-416)
COMBOS = [(14, 23, 14), (23, 11, 13), (12, 14, 11), (13, 12, 23), (14, 13, 12)]   # (tq, tz, to); :271, 21 -> 11
GDNF_SEED, SEG_SEED = 43, 97
SEG_COUNTS = (1, 2, 5, 8)
SEG_CASES = [(trial, nseg) for trial in range(3) for nseg in SEG_COUNTS]


def randn(rng, shape, sd):
    """basalt's randn (:104-109): normal_distribution<float>(0, sd) draws, float32 standard normals times f32(sd)."""
    return rng.standard_normal(shape, dtype=F32) * F32(sd)


class Layer(NamedTuple):
    """One GDN layer (:117-123): the projection types and GGUF rows (n_out, row bytes), conv_w (C, 4), dt and ssm_a
    (HV,), gamma (S,), the state (SF,), and the gate rows w_alpha and w_beta (HV, N) as bf16 bits."""
    tq: int
    tz: int
    to: int
    wqkv: np.ndarray
    wz: np.ndarray
    wo: np.ndarray
    conv_w: np.ndarray
    dt: np.ndarray
    ssm_a: np.ndarray
    gamma: np.ndarray
    state: np.ndarray
    wa: np.ndarray
    wb: np.ndarray


def make_layer(tq, tz, to, rng):
    """make_layer (:124-145) in its draw order: wqkv (N -> C), wz (N -> ZV), wo (ZV -> N), conv_w N(0, 0.5), dt N(0, 1),
    ssm_a = -exp(N(0, 1)), gamma = 1 + N(0, 0.1), the state N(0, 0.1), then w_alpha and w_beta from one N(0, 0.03),
    alpha first, rounded to bf16 to nearest even."""
    from tests.kernels.test_basalt_mmvq import weights   # here, not above: that module imports the GPU harness
    wqkv, wz, wo = weights(tq, N, C, rng), weights(tz, N, ZV, rng), weights(to, ZV, N, rng)
    conv_w = randn(rng, (C, 4), 0.5)
    dt = randn(rng, HV, 1.0)
    ssm_a = -np.exp(randn(rng, HV, 1.0))
    gamma = F32(1.0) + randn(rng, S, 0.1)
    state = randn(rng, SF, 0.1)
    ab = bf16_bits(randn(rng, (2, HV, N), 0.03))
    return Layer(tq, tz, to, wqkv, wz, wo, conv_w, dt, ssm_a, gamma, state, ab[0], ab[1])


def combo_case(i):
    """parity_gdnf's draws for combo i (:270-276): ``make_layer(*COMBOS[i])``, then x, MT rows of N(0, 1) (the window
    of T tokens reads its first T rows); from default_rng((43, i))."""
    rng = np.random.default_rng((GDNF_SEED, i))
    return make_layer(*COMBOS[i], rng), randn(rng, (MT, N), 1.0)


def seg_layer():
    """parity_seg's layer (:413), ``make_layer(14, 23, 14)`` from default_rng(97); the segments read only its conv_w,
    gamma, w_alpha, w_beta, dt and ssm_a."""
    return make_layer(14, 23, 14, np.random.default_rng(SEG_SEED))


class Segments(NamedTuple):
    """One parity_seg configuration: segment k's rows T[k], kept rows keep[k] and first row row0[k] of the R = sum(T)
    window rows; st0 (nseg, NL, SF), segment k's own state; the window rows qkv (R, C), z (R, ZV) and x (R, N)."""
    T: tuple
    keep: tuple
    row0: tuple
    st0: np.ndarray
    qkv: np.ndarray
    z: np.ndarray
    x: np.ndarray


def segments(trial, nseg, rng=None):
    """parity_seg's configuration (trial, nseg) (:420-438) in its draw order: per segment, T = 1 in trial 0, else 1 +
    a draw mod 8, overwritten by 8 for odd k in trial 2 after its draw; keep = 1 in trial 0, else a draw mod T + 1;
    then every segment's state N(0, 0.1), then qkv, z and x N(0, 1). ``rng`` defaults to default_rng((97, trial,
    nseg)) and is left after x, where basalt draws the commit's recorded inputs next (:505-507)."""
    rng = np.random.default_rng((SEG_SEED, trial, nseg)) if rng is None else rng
    T, keep, row0 = [], [], []
    for k in range(nseg):
        T.append(1 if trial == 0 else int(rng.integers(1, MT + 1)))
        if trial == 2 and k % 2:
            T[k] = MT
        keep.append(1 if trial == 0 else int(rng.integers(0, T[k] + 1)))
        row0.append(sum(T[:k]))
    R = sum(T)
    st0 = randn(rng, (nseg, NL, SF), 0.1)
    qkv = randn(rng, (R, C), 1.0)
    z = randn(rng, (R, ZV), 1.0)
    return Segments(tuple(T), tuple(keep), tuple(row0), st0, qkv, z, randn(rng, (R, N), 1.0))


class CpuSeg(NamedTuple):
    h: np.ndarray       # (n, C) float32: the SiLU'd conv, q and k L2-normed
    y: np.ndarray       # (n, ZV) float32: the gated RMS-normed read-out


def cpu_seg(st0, qkv, z, cw, gamma, gate, beta, n):
    """cpu_seg (:314-368) in float64 over the first n rows of qkv (C), z (ZV), gate and beta (HV), from one layer's
    state ``st0`` (SF floats, never written), conv_w ``cw`` (C x 4) and ``gamma`` (S): per token the conv of
    [history | qkv rows] and SiLU, rounded to float32; the q and k heads L2-normed from those floats; then per value
    head hd (q/k head hd % HK) kv = S^T k on the old state, delta = (v - g kv) b with g = exp(gate), S = g S + k delta,
    o = S^T q on the new state over sqrt(S); y = o * scale * gamma * sigmoid(z), scale = 1 / sqrt(mean(o^2) + EPS),
    left to right. Vectorized per token: only the double rounding of the sums differs from basalt's loops."""
    st0 = np.asarray(st0, F32)
    rows = lambda a, w: np.asarray(a, F32).reshape(-1, w)[:n].astype(F64)
    x = np.concatenate([st0[REC:].reshape(C, 3).T.astype(F64), rows(qkv, C)])     # [history | qkv], (3 + n, C)
    cw = np.asarray(cw, F32).reshape(C, 4).astype(F64)
    eps = F64(EPS)
    h = np.empty((n, C), F32)
    for t in range(n):
        s = np.zeros(C)
        for j in range(4):
            s += x[t + j] * cw[:, j]
        h[t] = s / (1.0 + np.exp(-s))
    qk = h[:, : 2 * HK * S].reshape(n, 2 * HK, S).astype(F64)
    h[:, : 2 * HK * S] = (qk * (1.0 / np.sqrt((qk * qk).sum(-1, keepdims=True) + eps))).reshape(n, -1)
    sd = st0[:REC].reshape(S, HV, S).astype(F64)                                  # [i = key][hd][j = value]
    qh = np.arange(HV) % HK
    g, b = np.exp(rows(gate, HV)), rows(beta, HV)
    zz, gam = rows(z, ZV).reshape(n, HV, S), np.asarray(gamma, F32).astype(F64)
    y = np.empty((n, ZV), F32)
    for t in range(n):
        ht = h[t].astype(F64)
        q, k = ht[: HK * S].reshape(HK, S)[qh], ht[HK * S: 2 * HK * S].reshape(HK, S)[qh]      # (HV, S): [hd][i]
        v = ht[2 * HK * S:].reshape(HV, S)
        delta = (v - g[t][:, None] * np.einsum("ihj,hi->hj", sd, k)) * b[t][:, None]
        sd = g[t][None, :, None] * sd + k.T[:, :, None] * delta[None]
        oc = np.einsum("ihj,hi->hj", sd, q) / np.sqrt(F64(S))
        scale = 1.0 / np.sqrt((oc * oc).sum(1) / S + eps)
        y[t] = (oc * scale[:, None] * gam * (1.0 / (1.0 + np.exp(-zz[t])))).reshape(-1)
    return CpuSeg(h, y)


def max_err(a, b):
    """max_err (:403-407): max |a - b| / (1 + |b|) in float64, a the GPU result, b the reference as the float32 basalt
    stores it. A NaN in ``a`` makes it NaN (basalt's std::max would drop it); ``n_bad`` counts those."""
    a, b = (np.asarray(v, F32).astype(F64) for v in (a, b))
    return float(np.max(np.abs(a - b) / (1.0 + np.abs(b))))


def n_bad(a):
    """The non-finite values in ``a``."""
    return int(np.count_nonzero(~np.isfinite(np.asarray(a))))


def ab64(x, wa, wb, dt, ssm_a):
    """gdn_ab_multi's gate and beta (verify_kernels.cu:113-157) in float64 over the rows of x (T, N): v = the dot of a
    bf16 row with x; gate = softplus(v_alpha + dt) * ssm_a with softplus(u) = u above 20, else log1p(exp(u)); beta =
    sigmoid(v_beta). Returns (gate, beta), (T, HV) each, rounded to float32."""
    x = np.asarray(x, F32).reshape(-1, N).astype(F64)
    va, vb = (x @ bf16_to_f32(w).reshape(HV, N).astype(F64).T for w in (wa, wb))
    u = va + np.asarray(dt, F32).astype(F64)
    sp = np.where(u > 20.0, u, np.log1p(np.exp(np.minimum(u, 20.0))))
    return (sp * np.asarray(ssm_a, F32).astype(F64)).astype(F32), (1.0 / (1.0 + np.exp(-vb))).astype(F32)


def conv_tail(tail, rows):
    """The conv history after the window's ``rows`` (n, C): the last 3 rows of [tail | rows], as [C][3] like ``tail``
    (3 C floats), a bit copy (gdn_conv_commit's rule): with n = 0 the tail itself."""
    t = np.asarray(tail, F32).view(np.uint32).reshape(-1, 3)
    seq = np.concatenate([t.T, np.asarray(rows, F32).view(np.uint32).reshape(-1, t.shape[0])])
    return np.ascontiguousarray(seq[-3:].T).reshape(-1).view(F32)
