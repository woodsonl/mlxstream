# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_gdn_parity.cpp and src/kernels/elementwise_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): cpu_seg's recurrence and gated norm
# (dec_gdn_parity.cpp:339-366) and elementwise_parity's double softplus gate (:78-80).
"""The prompt GDN kernels' host references, numpy only (no Metal import), imported as ``from tests.basalt_prompt_gdn_ref
import ...``. The geometry, ``cpu_seg``, ``max_err``, ``n_bad`` and ``conv_tail`` are the gdn lane's
(tests/basalt_gdn_ref.py) and are used from there, never copied: one float64 module of the GDN arithmetic, each lane
its own driver.

- ``gdn_prompt_ref``: the float64 recurrence and gated norm of dec_gdn_parity's ``cpu_seg`` (:339-366) on the prompt
  layout (spec T§4.3), from an h whose q and k heads are already L2-normed. Per token and value head hd, of q/k head
  ``QH[hd]`` = hd % 16: kv = S^T k on the old state, delta = (v - g kv) b with g = exp(gate), S = g S + k delta, oc =
  S^T q on the new state over sqrt(128); y = oc * scale * gamma * sigmoid(z), scale = 1 / sqrt(mean(oc^2) + EPS), left
  to right, rounded to float32 once. The un-normed oc (gdn_rec_wl_kernel's y) and the final state stay float64,
  unrounded, for W's ``close_to`` (gdn_rec_parity.cu:585-586); y is D-y16's reference (``max_err``).
- ``gates64``: gdn_gates_kernel's gate and beta (kernels.cu:207-216) in float64, elementwise_parity's reference.
- ``conv_l2_64``: D-conv's float64 h, ``cpu_seg``'s conv, SiLU and q/k L2 (dec_gdn_parity.cpp:320-338).
"""

import math

import numpy as np

from tests.basalt_gdn_ref import C, EPS, F32, F64, HK, HV, S, ZV, cpu_seg

QH = np.arange(HV) % HK                  # value head hd reads q/k head hd % HK (`const int qh = hd % HK;`)


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
