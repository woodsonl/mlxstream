# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/prefill/gdn_rec_parity.cu, src/kernels/dec_gdn_parity.cpp and
# src/kernels/elementwise_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the
# arithmetic of cpu_seg's recurrence and gated norm (dec_gdn_parity.cpp:339-366), elementwise_parity's double softplus
# (:78-80) and its fixture checks (:82-99), and gdn_rec_parity's hash (mix :373-376), close_to (:559-573) and word
# count (count_diff :404-409), checked here; W's pass criterion (gdn_rec_parity.cu:585-586) and dec_gdn_parity's
# ey < 1e-3 (:551) on the fp32 emulation.
"""Host-only checks of the prompt GDN references and fixtures (tests/basalt_prompt_gdn_ref.py); no GPU.

- float64: ``gdn_prompt_ref`` on one token from a zero state is spec T§4.3's closed form, computed head by head: kv =
  0, delta = v * beta, the state k[i] * delta[j], oc[j] = (q . k) * delta[j] / sqrt(128) with q and k of key head
  hd % 16, and y the gated norm of oc. ``gates64`` takes each side of the softplus branch (u > 20 gives u, u = 20
  included on the other side, past exp's double overflow too) and beta's sigmoid, per element against Python's math.
- ``close_to`` and ``diff_words`` on planted arrays (spec T§4.1-4.2): exact answers, one changed word found, the NaN
  rules of gdn_rec_parity.cu:559-573 and raw-word comparison.
- The fixtures: each deterministic; ``inputs`` basalt's generator (``mix`` against a Python-int transcription of the
  C) with spec T§5.1's distributions; D-gates' crossing the softplus branch on negative ssm_a, as elementwise_parity
  asserts its own; D-conv's non-zero history; P-edge's three reaching every class of spec T§5.3 and K§2-§4, measured
  on the kernels' first float operations or GR's basalt_emu steps, numpy's exp2 standing in for MUFU.EX2 (no class
  rests on its last bits) and the float64 reference for the recurrence.
- The fp32 emulation (spec T§4.4): ``rec_emu`` and ``out_norm_emu``, numpy's exp2 and 1/sqrt standing in for the
  GPU's probes, at T = 1, 2, 17, 257 on W's seeds against the float64 reference: W's gate on oc and the state
  (gdn_rec_parity.cu:585-586) and D-y16's ey < 1e-3 (dec_gdn_parity.cpp:551) on the y16 of the emulated oc.
"""

import math

import numpy as np
import pytest

from tests.basalt_gdn_ref import C, EPS, F32, F64, HK, HV, REC, S, conv_tail, max_err, n_bad
from tests.basalt_prompt_gdn_ref import (M64, SEED_B, SEED_W, TS, U32, close_to, conv_inputs, diff_words, edge_conv,
                                         edge_gates, edge_rec, gates64, gates_inputs, gdn_prompt_ref, inputs, mix,
                                         out_norm_emu, rec_emu)
from tests.kernels.basalt_emu import L2E, ex2n, expf_ieee, gdn_beta, gdn_conv, gdn_gate, gdn_silu, h2f

TINY = 2.0 ** -126


def sub(x):
    """Non-zero values below 2^-126 in magnitude: float32 subnormals, or float64 values in their range."""
    x = np.asarray(x)
    return (x != 0) & (np.abs(x) < TINY)


def normal(x):
    """Values of magnitude 2^-126 or more, infinities included, NaN not."""
    return np.abs(np.asarray(x)) >= TINY


def specials(x):
    """+inf, -inf, NaN, +0 and -0 all among x."""
    x = np.asarray(x)
    zero = x == 0
    return bool(np.isposinf(x).any() and np.isneginf(x).any() and np.isnan(x).any()
                and (zero & ~np.signbit(x)).any() and (zero & np.signbit(x)).any())


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


# ---- close_to and diff_words (spec T§4.1-4.2) -------------------------------------------------------------------

def test_close_to_on_planted_arrays():
    """close_to (gdn_rec_parity.cu:559-573) where the answer is exact: one value off by 1 against a reference of norm 4
    and maximum 2 gives (1/4, 1/2); the reference is read in float64, unrounded; a candidate NaN or infinity gives
    (nan, nan); a NaN reference value makes the RMS NaN and drops out of both maxima; an infinite one makes both NaN;
    an all-zero reference gives sqrt(d2) and max |d|; no values give (0, 0)."""
    ref = np.array([2.0, -2.0, 2.0, -2.0])
    x = ref.astype(F32)
    assert close_to(x, ref) == (0.0, 0.0)
    x[1] = -3.0
    assert close_to(x, ref) == (0.25, 0.5), "`return {r2 > 0 ? std::sqrt(d2 / r2) : ..., rm > 0 ? dm / rm : dm};`"
    assert close_to(np.ones(1, F32), np.array([1.0 + 2.0 ** -30]))[0] > 0, "the reference was rounded to float32"
    for bad in (np.nan, np.inf, -np.inf):
        y = x.copy()
        y[2] = bad
        assert all(math.isnan(v) for v in close_to(y, ref)), "`if (!std::isfinite(x[i])) return {NAN, NAN};`"
    r = ref.copy()
    r[3] = np.nan
    rms, worst = close_to(x, r)
    assert math.isnan(rms) and worst == 0.5, "`dm = std::max(dm, std::fabs(d));` keeps dm against a NaN"
    r[3] = np.inf
    assert all(math.isnan(v) for v in close_to(x, r)), "an infinite reference: inf / inf in both"
    assert close_to(np.array([0.0, 3.0, 0.0, -4.0], F32), np.zeros(4)) == (5.0, 4.0), "`std::sqrt(d2)`, `dm`"
    assert close_to(np.zeros(0, F32), np.zeros(0)) == (0.0, 0.0)


def test_diff_words_on_planted_arrays():
    """diff_words (gdn_rec_parity.cu:475-482, count_diff :404-409) counts differing raw 32-bit words: one planted ulp is
    one word; -0 against +0 differs; two NaNs of one pattern agree, of two patterns differ; an FP16 array counts pairs
    of halves (B-norm's `oc_n * 2` bytes); arrays of two sizes are refused."""
    a = np.linspace(-1.0, 1.0, 64, dtype=F32)
    b = a.copy()
    assert diff_words(a, b) == 0
    b[37] = np.nextafter(b[37], F32(2.0))
    assert diff_words(a, b) == 1, "`d += a[i] != b[i];`"
    z = np.zeros(8, F32)
    assert diff_words(z, -z) == 8, "+0 and -0 are two words"
    n = np.full(8, np.nan, F32)
    m = n.view(U32).copy()
    m[5] ^= 1                                                   # another NaN's payload
    assert diff_words(n, n.copy()) == 0 and diff_words(n, m.view(F32)) == 1
    y = np.arange(8, dtype=np.float16)
    y2 = y.copy()
    y2[4:6] = -y2[4:6]                                          # both halves of word 2
    assert diff_words(y, y2) == 1
    y2[0] = 0.5                                                 # one half of word 0
    assert diff_words(y, y2) == 2
    with pytest.raises(AssertionError):
        diff_words(a, a[:-1])


# ---- the fixtures (spec T§5.2-5.3, K§9.2) --------------------------------------------------------------------------

FIXTURES = {"inputs": lambda: inputs(17, SEED_B), "gates_inputs": lambda: gates_inputs(17, SEED_B),
            "conv_inputs": lambda: conv_inputs(65, SEED_B), "edge_rec": edge_rec, "edge_gates": edge_gates,
            "edge_conv": edge_conv}


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_deterministic(name):
    """Two calls of each fixture give the same bits (spec K§9.2, T§5.2: fixed seeds, no global state)."""
    a, b = FIXTURES[name](), FIXTURES[name]()
    assert [diff_words(x, y) for x, y in zip(a, b)] == [0] * len(a), f"{name}: two calls gave other bits"


def test_fixture_inputs_is_basalts_generator():
    """inputs (gdn_rec_parity.cu:484-505, spec T§5.2): mix is the C finalizer (:373-376) mod 2^64, here against a
    Python-int transcription on keys whose products wrap; another seed gives other values in all six arrays; a width's
    rows are the first rows of a wider one at its seed (element i's key is seed * mult + i); and spec T§5.1's
    distributions: unit q and k rows, gate in [-0.82, -0.02], beta in (0, 1), every value finite and none subnormal."""
    def mix_c(x):
        x ^= x >> 33
        x = x * 0xFF51AFD7ED558CCD & M64
        x ^= x >> 33
        x = x * 0xC4CEB9FE1A85EC53 & M64
        x ^= x >> 33
        return x & 0xFFFFFFFF
    ks = [0, 1, 7, 11 * 0x9E3779B97F4A7C15 & M64, M64 - 5, M64]
    assert mix(np.array(ks, np.uint64)).tolist() == [mix_c(k) for k in ks], "mix is not the C finalizer mod 2^64"
    a, b, c = inputs(5, SEED_W), inputs(2, SEED_W), inputs(5, SEED_W + 1)
    assert all(diff_words(x, y) > 0 for x, y in zip(a, c)), "seed + 1 repeats an array"
    assert [diff_words(x[:2], y) for x, y in zip(a[:4], b[:4])] == [0] * 4, "a width's rows are not a wider one's"
    assert [diff_words(x, y) for x, y in zip(a[4:], b[4:])] == [0] * 2, "gamma or state0 depends on T"
    h, gate, beta = a[:3]
    qk = h[:, :2 * HK * S].reshape(-1, S).astype(F64)
    assert np.abs(np.sqrt((qk * qk).sum(-1)) - 1).max() < 1e-6, "l2_rows: unit q and k rows"
    assert ((gate > -0.8201) & (gate <= F32(-0.02))).all() and ((beta > 0) & (beta < 1)).all()
    assert all(np.isfinite(x).all() and not sub(x).any() for x in a), "a subnormal or non-finite input"


def test_fixture_gates_crosses_the_softplus_branch():
    """D-gates' fixture (gates_inputs, spec K§9.2) at spec T§5.1's widths and seeds (11 + i, 27 + i), asserted as
    elementwise_parity.cpp:82-99 asserts its own: u = alpha + dt (a float sum, as basalt counts it) above 20 somewhere
    (`if (!over) ++bad;`) and past float exp's overflow at 88 (`past_overflow`), at most 20 elsewhere (log1pf's side),
    and ssm_a negative (`if (positive) ++bad;`)."""
    for i, T in enumerate(TS):
        for seed in (SEED_B + i, SEED_W + i):
            alpha, _, dt, ssm_a = gates_inputs(T, seed)
            u = alpha + dt
            assert (u > 20).any() and (u > 88).any(), f"T {T}, seed {seed}: `if (!over) ++bad;`"
            assert (u <= 20).any(), f"T {T}, seed {seed}: no u on log1pf's side"
            assert (ssm_a < 0).all(), f"T {T}, seed {seed}: `if (positive) ++bad;`"


def test_fixture_conv_starts_from_a_nonzero_history():
    """D-conv's fixture (conv_inputs, spec K§9.2: "from a non-zero history"): every history value non-zero, and the
    taps and the state the same at every width (qkv is drawn last)."""
    _, w1, s1 = conv_inputs(1, SEED_B)
    _, w2, s2 = conv_inputs(1000, SEED_B)
    assert (s1[REC:] != 0).all(), "a zero in the conv history"
    assert diff_words(w1, w2) == 0 and diff_words(s1, s2) == 0, "conv_w or the state depends on T"


def test_fixture_edge_rec_reaches_every_class():
    """P-edge's recurrence and norm fixture (edge_rec, spec T§5.3) reaches E1-E6: the decays by __expf's FMUL and ex2n,
    oc and the state by the float64 reference, whose values below 2^-126 are what an IEEE kernel keeps and a flushing
    one stores as 0."""
    h, gate, _, z, gamma, state0 = fx = edge_rec()
    V, e1 = 2 * HK * S, [0, HK, 2 * HK]                         # v of value head hd at V + hd * S
    with np.errstate(all="ignore"):
        g = ex2n(gate * L2E, np.exp2)
        oc = gdn_prompt_ref(*fx)[0]
        st1 = gdn_prompt_ref(*(a[:2] for a in fx[:4]), gamma, state0)[2]      # the state after tokens 0 and 1
        ez = ex2n(z[2, 3, :6] * -L2E, np.exp2)                  # sigm's __expf(-z)
    o1 = oc[0, e1]
    classes = {
        "E1: __expf(gate) halved (gate log2(e) < -126), a subnormal decay":
            (gate[0, e1] * L2E < -126).all() and sub(g[0, e1]).all(),
        "E1: k of key head 0 is 0 at token 0": (h[0, HK * S:(HK + 1) * S] == 0).all(),
        "E1, E5: oc of those rows below 2^-126, its float sum of squares 0":
            sub(o1).any() and (np.abs(o1) < TINY).all() and (np.square(o1.astype(F32)).sum(-1) == 0).all(),
        "E2: v = 0 at token 1, value head 0, columns 0..63, under a normal decay":
            (h[1, V:V + 64] == 0).all() and normal(g[1, 0]),
        "E2: the state through token 1 below 2^-126 there": sub(st1[:, 0, :64]).any(),
        "E3: value head 1's initial state subnormal, its v 0 at token 0":
            sub(state0[:, 1]).any() and (np.abs(state0[:, 1]) < TINY).all() and (h[0, V + S:V + 2 * S] == 0).all(),
        "E4: z = -200, -88, 88, 200, -inf, +inf at token 2, value head 3":
            z[2, 3, :6].tolist() == [-200.0, -88.0, 88.0, 200.0, -np.inf, np.inf],
        "E4: sigm's halved branch, 1 + e above 2^126, e = inf": (z[2, 3, :6] * -L2E < -126).any()
            and (np.isfinite(ez) & (ez + 1 > 2.0 ** 126)).any() and np.isinf(ez).any(),
        "E6: v NaN at token 2, value head 2, column 7": bool(np.isnan(h[2, V + 2 * S + 7])),
    }
    missing = [name for name, hit in classes.items() if not hit]
    assert not missing, f"not reached: {missing}"


def test_fixture_edge_gates_reaches_every_class():
    """P-edge's gates fixture (edge_gates; spec K§2's subnormal sites, §9.2's edges) reaches each class, on the
    kernel's first float operations (FADD u = alpha + dt; FMUL by 0x3FB8AA3B) and GR's basalt_emu steps."""
    alpha, logit, dt, ssm_a = edge_gates()
    with np.errstate(all="ignore"):
        u = alpha + dt
        e = expf_ieee(u, np.exp2)
        eb = ex2n(logit * -L2E, np.exp2)                        # sigm's __expf(-x)
        gate, beta = gdn_gate(alpha, dt, ssm_a, np.exp2), gdn_beta(logit, np.exp2)
    classes = {
        "u in [-104, -87.3]": ((u >= -104) & (u <= -87.3)).any(),
        "__expf(u) halved (u log2(e) < -126), and not beside it":
            (u * L2E < -126).any() and ((u * L2E >= -126) & (u < -87)).any(),
        "a subnormal e that log1pf passes on into a subnormal gate": (sub(e) & sub(gate)).any(),
        "a normal sp times |ssm_a| < 1 into a subnormal gate": (normal(e) & sub(gate)).any(),
        "u > 20, and u = 20 on log1pf's side": (u > 20).any() and (u == 20).any(),
        "u past float exp's overflow (88.72)": (u > 88.73).any(),
        "a subnormal u (alpha + dt)": sub(u).any(),
        "alpha +-inf, NaN, +-0": specials(alpha),
        "beta logits below -87.3: 1 + e above 2^126, a subnormal beta":
            (np.isfinite(eb) & (eb + 1 > 2.0 ** 126) & sub(beta)).any(),
        "beta logits below -88.72: e = inf, beta 0": (np.isinf(eb) & (beta == 0)).any(),
        "subnormal beta logits": sub(logit).any(),
        "beta logits +-inf, NaN, +-0": specials(logit),
        "a NaN gate and beta (cuda_nan's stores)": np.isnan(gate).any() and np.isnan(beta).any(),
        "|ssm_a| on both sides of 1": (np.abs(ssm_a) < 1).any() and (np.abs(ssm_a) > 1).any(),
    }
    missing = [name for name, hit in classes.items() if not hit]
    assert not missing, f"not reached: {missing}"


def test_fixture_edge_conv_reaches_every_class():
    """P-edge's conv, history and L2 fixture (edge_conv; spec K§3-§4's subnormal sites, §9.2's edges) reaches each
    class, on GR's basalt_emu conv sum (the PTX tap order) and SiLU, row t's window the old history then the rows
    (gdn_conv_tiled_kernel's input(), kernels.cu:236-252); the new history by conv_tail's word copy."""
    qkv, w, state = edge_conv()
    hist = state[REC:].reshape(C, 3)
    win = np.concatenate([hist.T, qkv])                         # win[t + k] = input(t - 3 + k)
    taps = np.stack([win[:-3], win[1:-2], win[2:-1], win[3:]])  # tap k's input at each row and channel
    with np.errstate(all="ignore"):
        s = gdn_conv(*taps, *w.T)
        h = gdn_silu(s, np.exp2)
        prod = w.T[:, None, :] * taps
        qk = h[0].reshape(-1, S)[[3, HK + 3]]                   # q head 3 and k head 3 at row 0, before the L2
        r = F32(1) / np.sqrt((qk * qk).sum(-1) + EPS)           # their L2 scale, about 1000
    tail = conv_tail(state[REC:], qkv).view(U32).reshape(C, 3)
    v3, k = (2 * HK + 3) * S, np.arange(3)
    classes = {
        "s above 87.34: __expf(-s) halved, e absorbed by 1 + e, h = s": ((s * -L2E < -126) & (h == s)).any(),
        "s below -88.72: e = inf, h = -0": ((s < -88.72) & (h == 0) & np.signbit(h)).any(),
        "subnormal qkv, history and taps": sub(qkv).any() and sub(hist).any() and sub(w).any(),
        "subnormal products of normal operands": (normal(w.T[:, None, :]) & normal(taps) & sub(prod)).any(),
        "a subnormal s, and h = s / 2": (sub(s) & sub(h)).any(),
        "a normal s with a subnormal quotient h": (normal(s) & sub(h)).any(),
        "q and k heads below 2^-63 at row 0, subnormal squares": (np.abs(qk) < 2.0 ** -63).all() and sub(qk * qk).any(),
        "a subnormal h whose product with the L2 scale is normal": (sub(qk) & normal(qk * r[:, None])).any(),
        "NaN qkv and history values": np.isnan(qkv).any() and np.isnan(hist).any(),
        "a NaN h in a q head (its L2 row NaN)": np.isnan(h[:, :2 * HK * S]).any(),
        "s = -inf (h NaN) and s = +inf (h inf)":
            (np.isneginf(s) & np.isnan(h)).any() and (np.isposinf(s) & np.isposinf(h)).any(),
        "rows R-3..R-1: a subnormal word and 0x7fa00001 in the new history":
            sub(tail[v3 + 2 * k, k].view(F32)).all() and (tail[v3 + 2 * k + 1, k] == 0x7FA00001).all(),
    }
    missing = [name for name, hit in classes.items() if not hit]
    assert not missing, f"not reached: {missing}"


# ---- the fp32 emulation (spec T§4.4) -------------------------------------------------------------------------------

@pytest.mark.parametrize("T", (1, 2, 17, 257))
def test_emulation_rec_and_out_norm_against_float64(T):
    """rec_emu and out_norm_emu with numpy's exp2 and 1/sqrt standing in for the GPU's fast::exp2 and fast::rsqrt, on
    inputs(T, 27 + i) (W's seed for basalt's width i), against gdn_prompt_ref: oc and the state meet W's gate through
    close_to (gdn_rec_parity.cu:585-586), and the y16 of the emulation's own oc, widened to float, meets D-y16's bound
    (dec_gdn_parity.cpp:551, max_err :403-407); no value is non-finite. These bounds hold for any order of the row
    sums: the kernel tests pin the bits."""
    h, gate, beta, z, gamma, state0 = fx = inputs(T, SEED_W + TS.index(T))
    oc_ref, y_ref, st_ref = gdn_prompt_ref(*fx)
    oc, st = rec_emu(h, gate, beta, state0, np.exp2)
    y = h2f(out_norm_emu(oc, z, gamma, np.exp2, lambda x: F32(1) / np.sqrt(x)))
    assert [n_bad(a) for a in (oc, st, y)] == [0, 0, 0], "a non-finite oc, state or y16 value"
    (co_rms, co_worst), (cs_rms, cs_worst) = close_to(oc, oc_ref), close_to(st, st_ref)
    assert (math.isfinite(co_rms) and math.isfinite(cs_rms) and co_rms < 1e-5 and co_worst < 1e-4 and cs_rms < 1e-5
            and cs_worst < 1e-4), ("`std::isfinite(co.rms) && std::isfinite(cs.rms) && co.rms < 1e-5 && co.worst < "
                                   "1e-4 && cs.rms < 1e-5 && cs.worst < 1e-4` (gdn_rec_parity.cu:585-586): co "
                                   f"{co_rms:.3g} {co_worst:.3g}, cs {cs_rms:.3g} {cs_worst:.3g}")
    ey = max_err(y, y_ref)
    assert ey < 1e-3, f"`ey < 1e-3` (dec_gdn_parity.cpp:551): ey {ey:.3g}"
