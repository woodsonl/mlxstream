# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp, src/kernels/ple_win_parity.cpp and
# src/kernels/gr_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's native
# key-norm check's metric and pass criterion (:899-910), ple_win_parity's projected-form input scales (:47-55), and
# gr_parity's scalar activation contract's degenerate norm input and the exactness it relies on (:167-236).
"""The n-gram (PLE) block's Metal port on the GPU, kernel by kernel (kernels/common/basalt_gr_norm.metal and
basalt_ple.metal). Each norm call runs at its call sites' shape, 4 rows of 2560 with 4 gamma rows and eps = f32(1e-6)
(native_ple_postops.cu:352-353, :358), unless stated.

- T-NORM ``test_grouped_norm_float64`` (spec T§7.2): basalt_gr_norm against R-struct's float64 ``gnorm``
  (tests/basalt_ple_ref.py), criterion quoted:
  ``const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;`` (ple_parity.cpp:910), rel =
  rel_l1(reference, GPU) over one call's 4 x 2560 outputs and the non-finite count over the norm's own output (basalt
  counted it over the block's result). The calls: N(0, σ) rows for σ in {1e-3, 1e-2, 1, 1e2} (at 1e-3 the mean square
  is about eps) and a zero row, whose output is 0, each with weights f32(1 + 0.1 N(0, 1)).
- T-NORM ``test_grouped_norm_bitwise_emu``: bit for bit against R-emu, ``basalt_emu.gr_norm`` (the rounding sequence of
  docs/research/basalt-nvcc/ple.md section 1) with the GPU's rsqrtf_ftz (MUFU.RSQ and fast::rsqrt have no bit-exact
  host model), on the calls above plus a row whose squares all lie below 2^-126, each of which .FTZ turns into +0; and
  on G1a, gr_parity's degenerate input (gr_parity.cpp:169-201: GrShapes{2, 1, 2}, so one row of 2 columns and the
  256-thread block; R = 1, eps = 0, gamma in {1.0f, 1.00390625f}). There, "At R=1 and eps=0, xn=gamma exactly"
  (:169), which basalt's exact checks rely on (:233-236): the scale is rsqrtf(1) and the output equals gamma bit for
  bit, provided the GPU's fast::rsqrt(1.0f) is exactly 1.0f. A failure there is a departure from basalt, to record,
  not to tune away.
- T-POST ``test_postops_bitwise_emu`` (spec T§7.3): native_ple_postops (native_ple_postops.cu:352-359) one token at a
  time, launch by launch (the key and query norms, basalt_ple_gate, basalt_ple_broadcast, the conv norm and
  basalt_ple_conv), each launch bit for bit against R-emu (``basalt_emu.gr_norm``, ``ple_gate``, ``ple_broadcast`` and
  ``ple_conv``: ple.md sections 1-4, with the GPU's rsqrtf_ftz, sqrtf_ftz and expf_ftz evaluated on the emulation's
  own arguments) fed the GPU's outputs of the launches before it, so that each comparison is its own kernel's: T§7.3
  "compares all six stages of every row of every fixture above with R-emu, byte for byte" (the query norm is compared
  too). The fixtures: T§7.3's projected form at W1's scales (``projected``), its 12 rows in order, each from the
  history the GPU's earlier rows leave; and T§7.3's edge cases. The gate's are streams of a direct basalt_ple_gate call
  (``GATE_EDGES``): s = +0 and s = -0 (sign 0, so the gate is exactly the sigmoid of 0, 0.5), 0 < |s| < 1e-6 (the
  clamp: the same gate as |s| = 1e-6f), |s| = 1e-6f, and |s| = 2e4 (the sigmoid saturates at 1, and at 0 where
  fast::exp2 overflows). The others are the projected form's first row with inputs planted (``EDGE_ROWS``): a SiLU
  whose fast::exp2 overflows (act = acc * rcp(inf) = -0), subnormal key, value, hidden and history inputs (read as
  zero under the flush) beside fp16-subnormal taps (widened exactly), and gated products below 2^-126 (flushed)."""

import functools

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_ple_ref import (D, EPS, H, HIST, KERN, N, ROWS, Fixture, gnorm, h_oracle, le, nonfinite, r_contract,
                                  rel_l1)
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels import basalt_ple_gpu as ple_gpu

F32, F64 = np.float32, np.float64
SIGMAS = (1e-3, 1e-2, 1.0, 1e2)
NORM_SEED = 910     # default_rng(NORM_SEED) draws, in order: each σ's row then its weights, the zero row's weights,
                    # then the tiny row and its weights
TINY = "squares below 2^-126"


def norm_calls():
    """The standalone norm calls, ``{id: (x, w)}``, each a D-vector (4 rows of 2560) with its own weights: spec T§7.2's
    N(0, σ) rows and zero row, then the tiny row, f32(N(0, 1) 2^-70): normal values whose squares all lie below
    2^-126 (asserted)."""
    rng = np.random.default_rng(NORM_SEED)

    def weights():
        return (1 + 0.1 * rng.standard_normal(D)).astype(F32)

    calls = {}
    for s in SIGMAS:
        x = (s * rng.standard_normal(D)).astype(F32)
        calls[f"sigma={s:g}"] = x, weights()
    calls["zero"] = np.zeros(D, F32), weights()
    tiny = (rng.standard_normal(D) * 2.0 ** -70).astype(F32)
    normal, small = (np.abs(tiny) >= 2.0 ** -126).all(), (np.square(tiny.astype(F64)) < 2.0 ** -127).all()
    assert normal and small, "the tiny row: normal values, squares below 2^-127"
    calls[TINY] = tiny, weights()
    return calls


CALLS = norm_calls()
# G1a: (x, gamma, n_rows, n_cols, gamma_rows, eps) for gr_parity's n_embd = 2, hc = 1 norm at R = 1, eps = 0
G1A = {f"G1a gamma={g}": (np.ones(2, F32), np.full(2, g, F32), 1, 2, 1, 0.0) for g in (1.0, 1.00390625)}
BITWISE = {**{k: (x, w, H, N, H, EPS) for k, (x, w) in CALLS.items()}, **G1A}


def bits(a):
    return np.ascontiguousarray(a, F32).view(np.uint32)


@pytest.mark.parametrize("call", [k for k in CALLS if k != TINY])
def test_grouped_norm_float64(call):
    x, w = CALLS[call]
    actual = ple_gpu.gr_norm(x, w, H, N, H, EPS)
    rel = rel_l1(gnorm(x, w), actual)[0]
    nf = nonfinite(actual)
    print(f"grouped norm [{call}]: rel {rel:.3e}, nf {nf}")
    assert le(rel, 2e-6) and nf == 0, (
        "`const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;` (ple_parity.cpp:910), rel = "
        f"rel_l1(R-struct's gnorm, the GPU) over the call's 4 x 2560 outputs, nf over the norm's output: rel {rel!r}, "
        f"nf {nf}")
    if call == "zero":
        assert (actual == 0).all(), f"spec T§7.2, a zero row: \"the output is 0\": {actual[actual != 0][:8]}"


@pytest.mark.parametrize("call", list(BITWISE))
def test_grouped_norm_bitwise_emu(call):
    x, w, n_rows, n_cols, gamma_rows, eps = BITWISE[call]
    got = ple_gpu.gr_norm(x, w, n_rows, n_cols, gamma_rows, eps)
    want = emu.gr_norm(x, w, n_rows, n_cols, gamma_rows, eps, ple_gpu.rsqrtf_ftz)
    diff = np.flatnonzero(bits(got) != bits(want))
    assert diff.size == 0, (
        f"T-NORM, bit for bit against basalt_emu.gr_norm (ple.md section 1): {diff.size} of {got.size} outputs differ; "
        f"first at {diff[:4].tolist()}: GPU {[hex(v) for v in bits(got).ravel()[diff[:4]]]}, "
        f"R-emu {[hex(v) for v in bits(want).ravel()[diff[:4]]]}")
    if call in G1A:
        one = hex(bits(ple_gpu.rsqrtf_ftz(np.ones(1, F32)))[0])
        out, gamma = [hex(v) for v in bits(got).ravel()], [hex(v) for v in bits(w)]
        print(f"{call}: output {out}, gamma {gamma}, the GPU's rsqrtf_ftz(1.0f) {one}")
        assert out == gamma, (
            f"G1a, \"At R=1 and eps=0, xn=gamma exactly\" (gr_parity.cpp:169): output {out} != gamma {gamma} with the "
            f"GPU's rsqrtf_ftz(1.0f) = {one}; a departure from basalt (gr_parity.cpp:233-236), to record, not to tune "
            "away (spec T§7.2)")


# ---- T-POST: native_ple_postops kernel by kernel against R-emu (spec T§7.3) ----------------------------------------
PROJECTED_SEED = 4      # default_rng(PROJECTED_SEED) draws, in W1's order without its hyper-connection inputs
                        # (ple_win_parity.cpp:50-55): conv1d, hidden's unit draw z, the raw keys, the values, the
                        # history, then w_nk, w_nq and w_nc; the first seed from 0 for which min |s| >= 0.1 holds
                        # (min |s| 0.174)
EDGE_SEED = 1119        # default_rng(EDGE_SEED) draws the planted inputs of the edge row being built
SCALE = np.frombuffer(kernels.basalt_ple_gate_params(), F32)[0]   # the gate's scale as the kernel reads it, 0x3CA1E89B
CLAMP_P = np.uint32(0x38543784).view(F32)                         # fl(SCALE * CLAMP_P) is 1e-6f, the clamp (asserted)
GATE_EDGES = {  # stream: (key[0], query[0]), every other entry 0, so the stream's sum is fl(key[0] * query[0]) exactly
    "s = +0": (0.0, 1.0),
    "s = -0": (-2.0 ** -61, 2.0 ** -61),      # fl(SCALE * -2^-122) is subnormal: .FTZ makes s -0
    "0 < s < 1e-6": (2.0 ** -20, 1.0),
    "-1e-6 < s < 0": (-2.0 ** -20, 1.0),
    "s = 1e-6f": (CLAMP_P, 1.0),
    "s = -1e-6f": (-CLAMP_P, 1.0),
    "s = 2e4": (2.0 ** 10, 2.0 ** 10),        # mag 144: ex2(-208) is +0 and the gate rcp(1) = 1
    "s = -2e4": (-2.0 ** 10, 2.0 ** 10),      # ex2(+208) overflows: the gate is rcp(inf) = +0
}
# The SiLU row's sums, channel c = 0, 1, ...: -acc * log2(e) passes 128 below acc = -88.72 (fast::exp2 overflows)
# and 1 + ex2 passes 2^126 below -87.34 (the reciprocal flushes); either way act = acc * +0 = -0.
SILU_ACC = (-np.inf, -3e38, -1e4, -100.0, -89.0, -88.75, -88.7, -88.5, -88.0, -87.5, -87.0, -80.0, -20.0, 20.0, 80.0,
            88.0, 89.0, 100.0, 3e38, np.inf)
EDGE_ROWS = ("SiLU exp2 overflow", "subnormal inputs", "gated below 2^-126")


@functools.lru_cache(maxsize=1)
def projected():
    """T§7.3's projected form at W1's scales (ple_win_parity.cpp:47-55): per row a raw key f32(0.2 N(0,1)) (D values)
    and a value f32(0.5 N(0,1)) (N), the initial history f32(N(0,1)) (9 x D, row r oldest first), the three norms'
    f32(1 + 0.1 N(0,1)) and conv1d f16(0.3 N(0,1)); hidden = f32(sigma_h z) by T§6.3's rule 3 (R-contract with the
    unit hidden f32(z), then sigma_h = mean |gated + conv_out| / mean |f32(z)|). Asserts T§7.3's min |s| >= 0.1 on
    R-contract. Returns (fx, key, value): a basalt_ple_ref Fixture without key, value and emb (the projections are
    inputs here), the raw keys (12, D) and the values (12, N)."""
    rng = np.random.default_rng(PROJECTED_SEED)
    conv1d = (0.3 * rng.standard_normal(KERN * D)).astype(np.float16)
    z = rng.standard_normal((ROWS, D))
    key = (0.2 * rng.standard_normal((ROWS, D))).astype(F32)
    value = (0.5 * rng.standard_normal((ROWS, N))).astype(F32)
    hist = rng.standard_normal((HIST, D)).astype(F32)
    w_nk, w_nq, w_nc = ((1 + 0.1 * rng.standard_normal(D)).astype(F32) for _ in range(3))
    fx = Fixture(key=None, value=None, w_nk=w_nk, w_nq=w_nq, w_nc=w_nc, conv1d=conv1d, emb=None, hist=hist,
                 hidden=z.astype(F32))
    ref = r_contract(fx, key, value)
    sigma = np.abs(ref.gated + ref.conv_out).mean() / np.abs(fx.hidden.astype(F64)).mean()
    fx = fx._replace(hidden=(sigma * z).astype(F32))
    min_s = float(np.abs(r_contract(fx, key, value).s).min())
    assert min_s >= 0.1, f"T§7.3, \"Each fixture asserts min |s[c]| ≥ 0.1 on R-contract\": {min_s!r}"
    return fx, key, value


def postops_row(key, value, hidden, hist, fx):
    """One token of native_ple_postops (native_ple_postops.cu:352-359) on the GPU, launch by launch, with ``fx``'s
    norms and conv1d, and R-emu on the same inputs, each emulated launch fed the GPU's outputs of the launches before
    it: ``{stage: (GPU, R-emu)}`` for the key norm, the query norm, the gate, the broadcast (gated), the conv norm
    (normalized), and the convolution's conv_out and result. ``hist`` is in basalt's layout (row r of channel c at
    c * 9 + r)."""
    def norm(x, w):
        return ple_gpu.gr_norm(x, w, H, N, H, EPS).ravel(), emu.gr_norm(x, w, H, N, H, EPS, ple_gpu.rsqrtf_ftz).ravel()

    out = {"key": norm(key, fx.w_nk), "query": norm(hidden, fx.w_nq)}
    k, q = out["key"][0], out["query"][0]
    out["gate"] = ple_gpu.gate(k, q), emu.ple_gate(k, q, SCALE, ple_gpu.sqrtf_ftz, gpu.expf_ftz)
    out["gated"] = ple_gpu.broadcast(value, out["gate"][0]), emu.ple_broadcast(value, out["gate"][0])
    gated = out["gated"][0]
    out["normalized"] = norm(gated, fx.w_nc)
    nm = out["normalized"][0]
    got = ple_gpu.conv(hist, nm, fx.conv1d, hidden, gated)
    out["conv_out"], out["result"] = zip(got, emu.ple_conv(hist, nm, fx.conv1d, hidden, gated, gpu.expf_ftz))
    return out


def projected_rows():
    """The projected form's 12 rows in order, each from the history the GPU's earlier rows leave (T§7.3): the initial
    history, then each row's GPU normalized row shifted in. basalt's advance moves words, so the host's shift
    (H-ORACLE's) has its bits."""
    fx, key, value = projected()
    hist, rows = np.ascontiguousarray(fx.hist.T).ravel(), []
    for t in range(ROWS):
        rows.append(postops_row(key[t], value[t], fx.hidden[t], hist, fx))
        hist = h_oracle(hist, rows[-1]["normalized"][0][None], 1)
    return rows


def edge_row(case):
    """The projected form's first row, from the initial history, with ``case``'s inputs planted (``EDGE_ROWS``)."""
    fx, key, value = projected()
    rng = np.random.default_rng(EDGE_SEED)
    k, v, h, taps = key[0].copy(), value[0].copy(), fx.hidden[0].copy(), fx.conv1d.copy()
    hist = np.ascontiguousarray(fx.hist.T).ravel()          # a copy
    if case == "SiLU exp2 overflow":                        # channel c's taps (1, 0, 0, 0): its sum is history row 0
        c = np.arange(len(SILU_ACC))
        taps.reshape(D, KERN)[c] = (1.0, 0.0, 0.0, 0.0)
        hist.reshape(D, HIST)[c, 0] = SILU_ACC
    elif case == "subnormal inputs":                        # every 7th input a float32 subnormal, every 5th tap an
        for a in (k, v, h, hist):                           # fp16 subnormal, of either sign
            n = a[::7].size
            a[::7] = (rng.integers(1, 1 << 23, n, dtype=np.uint32) | rng.integers(0, 2, n, dtype=np.uint32) << 31
                      ).view(F32)
        n = taps[::5].size
        taps[::5] = (rng.integers(1, 1 << 10, n, dtype=np.uint16) | rng.integers(0, 2, n, dtype=np.uint16) << 15
                     ).view(np.float16)
    elif case == "gated below 2^-126":                      # |value| in [2^-126, 2^-123), so |value * gate|
                                                            # straddles 2^-126
        v = (rng.choice((-1.0, 1.0), N) * rng.uniform(1, 8, N) * 2.0 ** -126).astype(F32)
    return [postops_row(k, v, h, hist, fx._replace(conv1d=taps))]


def gate_edges():
    """``GATE_EDGES``' streams through basalt_ple_gate, H streams a call (basalt's launch), and R-emu's ple_gate."""
    keys, queries = np.zeros((len(GATE_EDGES), N), F32), np.zeros((len(GATE_EDGES), N), F32)
    keys[:, 0], queries[:, 0] = np.array(list(GATE_EDGES.values()), F32).T
    return [{"gate": (ple_gpu.gate(keys[i:i + H], queries[i:i + H]),
                      emu.ple_gate(keys[i:i + H], queries[i:i + H], SCALE, ple_gpu.sqrtf_ftz, gpu.expf_ftz))}
            for i in range(0, len(GATE_EDGES), H)]


POSTOPS = {"projected": projected_rows, "gate edges": gate_edges,
           **{case: functools.partial(edge_row, case) for case in EDGE_ROWS}}


@pytest.mark.parametrize("case", list(POSTOPS))
def test_postops_bitwise_emu(case):
    rows = POSTOPS[case]()
    for t, stages in enumerate(rows):
        for stage, (got, want) in stages.items():
            diff = np.flatnonzero(bits(got) != bits(want))
            assert diff.size == 0, (
                "T-POST, \"compares all six stages of every row of every fixture above with R-emu, byte for byte\" "
                f"(spec T§7.3): {case}, row {t}, {stage}: {diff.size} of {got.size} values differ; first at "
                f"{diff[:4].tolist()}: GPU {[hex(v) for v in bits(got)[diff[:4]]]}, "
                f"R-emu {[hex(v) for v in bits(want)[diff[:4]]]}")
    if case == "gate edges":
        assert F32(SCALE * CLAMP_P) == F32(1e-6), "the clamp stream's s = fl(SCALE * CLAMP_P) must be 1e-6f"
        g = dict(zip(GATE_EDGES, (int(b) for b in bits(np.concatenate([r["gate"][0] for r in rows])))))
        print("gate edges:", {name: hex(b) for name, b in g.items()})
        assert g["s = +0"] == g["s = -0"] == 0x3F000000, (
            f"T§7.3, s = +0 and -0: \"sign 0, gate exactly the sigmoid of 0\": {hex(g['s = +0'])}, {hex(g['s = -0'])}")
        assert g["0 < s < 1e-6"] == g["s = 1e-6f"] and g["-1e-6 < s < 0"] == g["s = -1e-6f"], (
            "T§7.3, 0 < |s| < 1e-6: the clamp gives the gate of |s| = 1e-6f, sign for sign")
        assert g["s = 2e4"] == 0x3F800000 and g["s = -2e4"] == 0, (
            "T§7.3, large |s|: the sigmoid saturates at 1, and at +0 where fast::exp2 overflows: "
            f"{hex(g['s = 2e4'])}, {hex(g['s = -2e4'])}")
    elif case == "SiLU exp2 overflow":
        acc = np.array(SILU_ACC)
        act = bits(rows[0]["conv_out"][0][:len(acc)])
        under = np.isfinite(acc) & (acc <= -87.5)
        assert (act[under] == 0x80000000).all(), (
            "T§7.3, \"the SiLU where -acc · log2(e) overflows fast::exp2, so acc / inf\" (and where 1 + ex2 passes "
            f"2^126, so the reciprocal flushes): act = -0 for acc {acc[under].tolist()}: "
            f"{[hex(b) for b in act[under]]}")
    elif case == "gated below 2^-126":
        flushed = int(np.count_nonzero(rows[0]["gated"][0] == 0))
        assert 0 < flushed < D, (
            f"T§7.3, gated products below 2^-126 are flushed and the others kept: {flushed} of {D} are 0")
