# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_gdn_parity.cpp and src/kernels/elementwise_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the arithmetic of cpu_seg's recurrence and
# gated norm (dec_gdn_parity.cpp:339-366) and elementwise_parity's double softplus (:78-80), checked here.
"""Host-only checks of the prompt GDN references (tests/basalt_prompt_gdn_ref.py); no GPU.

- float64: ``gdn_prompt_ref`` on one token from a zero state is spec T§4.3's closed form, computed head by head: kv =
  0, delta = v * beta, the state k[i] * delta[j], oc[j] = (q . k) * delta[j] / sqrt(128) with q and k of key head
  hd % 16, and y the gated norm of oc. ``gates64`` takes each side of the softplus branch (u > 20 gives u, u = 20
  included on the other side, past exp's double overflow too) and beta's sigmoid, per element against Python's math.
"""

import math

import numpy as np

from tests.basalt_gdn_ref import C, EPS, F32, F64, HK, HV, S
from tests.basalt_prompt_gdn_ref import gates64, gdn_prompt_ref


def test_float64_rec_one_token_on_a_zero_state_is_the_closed_form():
    rng = np.random.default_rng(5)
    h = rng.standard_normal((1, C), dtype=F32)
    qk = h[0, :2 * HK * S].reshape(2 * HK, S).astype(F64)
    h[0, :2 * HK * S] = (qk / np.sqrt((qk * qk).sum(-1, keepdims=True))).reshape(-1)    # unit q and k rows
    gate = rng.uniform(-1.0, -0.5, (1, HV)).astype(F32)       # g = exp(gate) in [0.37, 0.61]: a misplaced g shows
    beta = rng.uniform(0.1, 0.9, (1, HV)).astype(F32)
    z = rng.standard_normal((1, HV, S), dtype=F32)
    gamma = (1.0 + 0.1 * rng.standard_normal(S)).astype(F32)
    oc, y, st = gdn_prompt_ref(h, gate, beta, z, gamma, np.zeros((S, HV, S), F32))
    assert oc.shape == y.shape == (1, HV, S) and st.shape == (S, HV, S)
    x = h[0].astype(F64)
    for hd in range(HV):                                        # hd // 3 in place of hd % 16 reads another k
        kh = hd % HK
        q, k = x[kh * S:(kh + 1) * S], x[(HK + kh) * S:(HK + kh + 1) * S]
        delta = x[(2 * HK + hd) * S:(2 * HK + hd + 1) * S] * float(beta[0, hd])
        tol = 1e-12 * np.abs(delta).max()
        assert np.abs(st[:, hd, :] - np.outer(k, delta)).max() <= tol, f"head {hd}: state != k[i] * v[j] * beta"
        want = float(q @ k) * delta / math.sqrt(S)
        assert np.abs(oc[0, hd] - want).max() <= tol, f"head {hd}: oc != (q . k) * delta / sqrt(128), key head {kh}"
        scale = 1.0 / math.sqrt(sum(o * o for o in want) / S + float(EPS))
        ynorm = [want[j] * scale * float(gamma[j]) * (1.0 / (1.0 + math.exp(-float(z[0, hd, j])))) for j in range(S)]
        np.testing.assert_allclose(y[0, hd], ynorm, rtol=2.0 ** -23, atol=0, err_msg=f"head {hd}: y")


def test_float64_gates64_takes_each_side_of_the_softplus_branch():
    us = np.array([-90.0, -30.0, -1.5, 0.0, 7.25, 19.5, 20.0, 20.25, 22.0, 36.5, 100.0, 800.0] * 4)   # u per head
    rng = np.random.default_rng(7)
    exact = np.arange(HV) < HV // 2                             # dt a multiple of 1/8 there: u is us exactly
    dt = np.where(exact, 0.125 * (np.arange(HV) % 5 - 2), rng.normal(0.0, 0.5, HV)).astype(F32)
    alpha = (us - dt).astype(F32).reshape(1, HV)
    ssm_a = (-(np.abs(rng.standard_normal(HV)) + 0.1)).astype(F32)
    logit = (5.0 * rng.standard_normal((1, HV))).astype(F32)
    gate, beta = gates64(alpha, logit, dt, ssm_a)
    u = [float(alpha[0, i]) + float(dt[i]) for i in range(HV)]  # `(double) alpha[i] + (double) dt[i]`
    assert [v for v, e in zip(u, exact) if e] == us[exact].tolist()
    assert any(float(alpha[0, i] + dt[i]) != u[i] for i in range(HV)), "every sum is a float: a float add would pass"
    assert any(v > 20.0 for v in u) and any(v <= 20.0 for v in u) and 20.0 in u, "both sides of u > 20, and u = 20"
    for v in (20.0, 20.25, 22.0):                                # where the two sides differ in double
        assert math.log1p(math.exp(v)) != v
    want = [(v if v > 20.0 else math.log1p(math.exp(v))) * float(ssm_a[i]) for i, v in enumerate(u)]
    np.testing.assert_allclose(gate[0], want, rtol=1e-15, atol=0, err_msg="gate")
    np.testing.assert_allclose(beta[0], [1.0 / (1.0 + math.exp(-float(b))) for b in logit[0]], rtol=1e-15, atol=0,
                               err_msg="beta")
