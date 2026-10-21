# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_gdn_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its fixtures, pre-fills and pass criteria.
"""dec_gdn_parity's GPU checks (basalt src/kernels/dec_gdn_parity.cpp) on the Metal GDN decode window
(kernels/common/basalt_gdn.metal), through tests/kernels/basalt_gdn_gpu.py, on tests/basalt_gdn_ref.py's fixtures and
float64 references. In every test every output is finite and every sentinel intact.

- G3-G6 (dec_gdn_parity.cpp:288-291), on parity_gdnf's inputs (combo by combo, every window width T = 1..8 of the
  8-row x; qkv and z the harness's native_mmvq of x's native q8_1, as run_ref and run_gdnf share them, :180-187,
  :217-221): dec_gdn_rec's window stores against the per-stage kernels by basalt's ``cmp_f`` (every element's 4
  bytes equal, both sides finite): win_gate and win_beta are gdn_ab_multi's gate and beta, win_h gdn_conv_l2_multi's
  h, win_qkv the qkv rows (the port's copy, spec E§9.3); and by ``cmp_b`` its yq rows are the native q8_1 of
  gdn_step_norm_multi's y on that h, gate and beta, byte for byte, after G6's precondition on that y (``g6_blocks``),
  every half of yq finite; rows T..7 of every store and yq's row T keep their pre-fill.
- G8-G9 (dec_gdn_parity.cpp:293-294), on the same inputs: dec_gdn_rec at every window width T = 1..8 (T from the
  dispatch's params record, decision D16) against the 8-row window, run once per combo before the widths (``w8``,
  :277-278): rows t < T of every output bitwise (invariant 1), by ``cmp_f`` win_h (basalt's h@8) and the stores
  basalt leaves out, win_gate, win_beta and win_qkv, by ``cmp_b`` yq (yq@8); at T = 8, two runs of one window.
- G11 (:296-297): the state's bytes after every dec_gdn_rec dispatch, the T-wide window's and the 8-row window's,
  equal the layer's state (``st_ok``; invariant 3: the window never writes its state).
- W.1, the port's addition (decision D15; flash-next.md's tier-1 rule, the kernel against its own one-token calls):
  each row t of the 8-row window equals dec_gdn_rec run on row t alone (T = 1), bitwise: its gate and beta, which read
  x row t and the layer's constants only, and its win_h and win_qkv over the conv history the rows before t leave
  (``conv_tail``, a bit copy); row 0, run from the layer's own state, in every output, yq included (yq of a row t > 0
  needs the state after the commit of rows < t: PR B, spec-tests-commit.md DCM.8).
- A1, the port's addition (spec T§5.5: no basalt test grades gdn_ab_multi's gate and beta against a reference): on
  parity_gdnf's inputs, combo by combo and for every window width T = 1..8 of the 8-row x, gdn_ab_multi's gate and
  beta within ``max_err < 1e-4`` of the float64 ``ab64``, the bound of the nearest stage (``ec``,
  dec_gdn_parity.cpp:551; flash-next.md's rule for kernels basalt checks only against another GPU kernel).
- S8 (dec_gdn_parity.cpp:501, :551): on parity_seg's configurations (trial x segment count), each segment alone from
  its own layer-1 state (decision D12, spec T§7.2: no ``_seg`` kernel), gdn_conv_l2_multi's h and dec_gdn_rec's
  (the engine's build, GDN_Y_BOUND 0) within ``max_err < 1e-4`` of the float64 ``cpu_seg``'s h, with no non-finite
  value (every 0xFF pre-fill overwritten); and :551's ``d == 0`` as D12 ports it: dec_gdn_rec's h and win_qkv equal
  gdn_conv_l2_multi's h and the qkv rows bitwise (G5 on the segment), its state's bytes unchanged (G11).
- S9 (dec_gdn_parity.cpp:502, :551): on the same segments, the per-stage chain basalt's reference side runs
  (:457-460: gdn_conv_l2_multi's h and gdn_ab_multi's gate and beta into gdn_step_norm_multi), its y within
  ``max_err < 1e-3`` of the float64 ``cpu_seg``'s y, ``cpu_seg`` fed the gate and beta the kernel consumed (:496-499),
  with no non-finite value in h, gate, beta or y.
- A2, the port's addition (spec T§5.5: basalt grades dec_gdn_rec's fp32 y against nothing), as S9: dec_gdn_rec's own
  y (built with GDN_Y_BOUND 1, decision D13) within ``max_err < 1e-3`` of ``cpu_seg``'s y fed the kernel's gate and
  beta, no non-finite value in its h, gate, beta, y or yq; its gate, beta and yq equal the per-stage chain's (G3, G4
  and G6 on the segment), its state's bytes unchanged; and the engine's build, GDN_Y_BOUND 0, run on the same inputs,
  writes the same yq and window stores bit for bit (engineering review E6).
"""

import functools

import numpy as np
import pytest

from tests.basalt_gdn_ref import (COMBOS, C, F32, HV, LAYER, MT, N, REC, SEG_CASES, ZV, ab64, combo_case, conv_tail,
                                  cpu_seg, max_err, n_bad, seg_layer, segments)
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gdn_gpu as gg
from tests.kernels import basalt_gpu as gpu

GDNF_CASES = [(combo, t) for combo in range(len(COMBOS)) for t in range(1, MT + 1)]
# parity_seg's criteria, verbatim (dec_gdn_parity.cpp:501, :502, :551; spec T§9's S8, S9 and S-gate rows)
S8 = '`ec = std::max(ec, max_err(hs.data(), c.h.data(), c.h.size()));` -> `ec < 1e-4`'
S9 = '`ey = std::max(ey, max_err(ys.data(), c.y.data(), c.y.size()));` -> `ey < 1e-3`'
S_GATE = '`d == 0` of `ok = ok && d == 0 && ec < 1e-4 && ey < 1e-3 && es < 1e-3;`'


@functools.lru_cache(maxsize=None)
def gdnf_case(combo):
    """parity_gdnf's inputs for ``combo`` (combo_case, :270-276; run_ref's projections, :180-187): the layer, x's MT
    rows, and qkv (MT, C) and z (MT, ZV), native_mmvq of x's native q8_1 over the MT columns; a window of width T reads
    their first T rows (mmvq's columns are independent: test_basalt_mmvq.py's column invariance)."""
    layer, x = combo_case(combo)
    xq = gpu.q8_1(x)
    return layer, x, gpu.mmvq(layer.tq, layer.wqkv, xq, N, C, MT), gpu.mmvq(layer.tz, layer.wz, xq, N, ZV, MT)


def cmp_f(what, ref, new):
    """cmp_f (dec_gdn_parity.cpp:245-258): the count of elements whose 4 bytes differ plus those not finite on either
    side, and its print of the first."""
    a, b = (np.ascontiguousarray(v, F32).reshape(-1) for v in (ref, new))
    bad = (a.view(np.uint32) != b.view(np.uint32)).astype(int) + (~np.isfinite(a) | ~np.isfinite(b))
    i = np.flatnonzero(bad)
    return int(bad.sum()), (f"{what}[{i[0]}]: ref {a[i[0]]:.9g} new {b[i[0]]:.9g}" if i.size else "")


def cmp_b(ref, new):
    """cmp_b (dec_gdn_parity.cpp:259-265): the count of bytes that differ."""
    return int(np.count_nonzero(np.asarray(ref, np.uint8).reshape(-1) != np.asarray(new, np.uint8).reshape(-1)))


def g6_blocks(y):
    """G6's precondition on the per-stage y (spec T§5.1's G6 note) in its exact form: the count of y's q8_1 blocks on
    which the two sequences G6 compares store different bytes, native_quantize_q8_1's (flushed sum butterfly) and
    dec_gdn_rec's (IEEE sum, q8_1_store_rec), by basalt_emu's transcriptions, which H.4 grades bitwise against both
    helpers (test_basalt_gdn_helpers.py). Its sufficient form, every y 0 or at least 2^-102, fails on every combo and
    width of parity_gdnf's draws (nonzero values below 2^-102 on 0.8-4.6 % of y), so the test asserts the condition
    that form stands for."""
    nat, rec = (q(np.asarray(y, F32)).reshape(-1, 36) for q in (emu.q8_1_native, emu.q8_1_rec))
    return int(np.count_nonzero(np.any(nat != rec, axis=1)))


def yq_halves(yq):
    """The d and sum halves of q8_1 rows (block_q8_1: half2 ds, then the 32 int8 codes), (blocks, 2) float16."""
    return np.ascontiguousarray(np.asarray(yq, np.uint8).reshape(-1, 36)[:, :4]).view(np.float16)


def check_run(where, run):
    """Every test's rule on one dec_gdn_rec run: every sentinel intact, every output finite (yq's d and sum halves)."""
    assert run.ok, f"{where}: dec_gdn_rec wrote past its window's rows (a sentinel byte changed)"
    for what, a in (("win_qkv", run.qkv), ("win_h", run.h), ("win_gate", run.gate), ("win_beta", run.beta),
                    ("yq half", yq_halves(run.yq))):
        assert n_bad(a) == 0, f"{where}: {n_bad(a)} non-finite {what} values, want none"


@functools.lru_cache(maxsize=None)
def gdnf_w8(combo):
    """parity_gdnf's ``w8`` (dec_gdn_parity.cpp:277-278): dec_gdn_rec over all MT rows of ``combo``'s inputs, run once
    per combo before the window widths; a ``Rec``."""
    layer, x, qkv, z = gdnf_case(combo)
    w8 = gg.rec(layer, layer.state, qkv, x, z)
    check_run("the 8-row window", w8)
    return w8


def check_yq(where, y, yq):
    """G6 (dec_gdn_parity.cpp:291) on dec_gdn_rec's yq rows and the per-stage y: the precondition first, so that a draw
    which breaks it fails as a fixture error, not as a kernel mismatch; every d and sum half of yq finite; then yq
    against native_quantize_q8_1 of y, byte for byte."""
    bad = g6_blocks(y)
    assert bad == 0, (f"{where}: fixture error, not a kernel mismatch: G6's precondition fails on {bad} q8_1 blocks of "
                      f"the per-stage y (the native and the rec sequence store different bytes there)")
    halves = yq_halves(yq)
    assert n_bad(halves) == 0, f"{where}: {n_bad(halves)} non-finite yq halves (d, sum), want none"
    d, want = cmp_b(gpu.q8_1(y), yq), '`d += cmp_b("yq", r.yq, n.yq, native_q8_1_bytes(ZV, T));` -> `d == 0`'
    assert d == 0, f"{where}: {d} yq bytes differ; want {want}"


@pytest.mark.parametrize("combo,t", GDNF_CASES)
def test_dec_gdn_rec_matches_the_per_stage_kernels_bitwise(combo, t):
    layer, x, qkv, z = gdnf_case(combo)
    r = gg.rec(layer, layer.state, qkv[:t], x[:t], z[:t])
    gate, beta, ok_ab = gg.ab(layer, x[:t])
    h, ok_h = gg.conv_l2(layer.state, qkv[:t], layer.conv_w)
    y, ok_y = gg.step_norm(layer.state, h, gate, beta, z[:t], layer.gamma)
    assert r.ok and ok_ab and ok_h and ok_y, (f"T={t}: wrote past its window's rows (a sentinel byte changed): "
                                              f"dec_gdn_rec {r.ok}, gdn_ab_multi {ok_ab}, gdn_conv_l2_multi {ok_h}, "
                                              f"gdn_step_norm_multi {ok_y} (True: intact)")
    for what, ref, new, want in (
            ("gate", gate, r.gate, 'd += cmp_f("gate", r.gate, n.gate, (size_t) T * HV, ma);` -> `d == 0'),
            ("beta", beta, r.beta, 'd += cmp_f("beta", r.beta, n.beta, (size_t) T * HV, ma);` -> `d == 0'),
            ("h", h, r.h, 'd += cmp_f("h", r.h, n.h, (size_t) T * C, ma);` -> `d == 0'),
            ("win_qkv", qkv[:t], r.qkv, "win_qkv = the qkv rows (the port's copy, E§9.3)")):
        d, first = cmp_f(what, ref, new)
        assert d == 0, f"T={t}: {d} elements differ or are not finite, first {first}; want `{want}`"
    check_yq(f"T={t}", y, r.yq)


@pytest.mark.parametrize("combo,t", GDNF_CASES)
def test_dec_gdn_rec_columns_do_not_depend_on_the_window_width(combo, t):
    layer, x, qkv, z = gdnf_case(combo)
    w8, n = gdnf_w8(combo), gg.rec(layer, layer.state, qkv[:t], x[:t], z[:t])
    check_run(f"T={t}", n)
    g8 = '`inv += cmp_f("h@8", w8.h, n.h, (size_t) T * C, ma);` -> `inv == 0`'
    for what, also in (("h", ""), ("gate", " on gate too"), ("beta", " on beta too"), ("qkv", " on win_qkv too")):
        d, first = cmp_f(f"{what}@8", getattr(w8, what)[:t], getattr(n, what))
        assert d == 0, (f"T={t}: {d} elements differ from the 8-row window's rows or are not finite, first {first}; "
                        f"want {g8}{also}")
    d = cmp_b(w8.yq[:t], n.yq)
    assert d == 0, (f"T={t}: {d} yq bytes differ from the 8-row window's rows; want "
                    '`inv += cmp_b("yq@8", w8.yq, n.yq, native_q8_1_bytes(ZV, T));` -> `inv == 0`')


@pytest.mark.parametrize("combo,t", GDNF_CASES)
def test_dec_gdn_rec_leaves_the_state_untouched(combo, t):
    layer, x, qkv, z = gdnf_case(combo)
    n = gg.rec(layer, layer.state, qkv[:t], x[:t], z[:t])
    check_run(f"T={t}", n)
    for where, run in (("the 8-row window", gdnf_w8(combo)), (f"the T={t} window", n)):
        assert np.array_equal(run.state.view(np.uint32), layer.state.view(np.uint32)), (
            f"{where} changed its state; want "
            "`const bool st_ok = std::memcmp(st0.data(), st1.data(), st0.size() * 4) == 0;` -> `st_ok`")


@pytest.mark.parametrize("combo", range(len(COMBOS)))
def test_dec_gdn_rec_rows_alone(combo):
    layer, x, qkv, z = gdnf_case(combo)
    w8, bits = gdnf_w8(combo), layer.state.view(np.uint32)
    for t in range(MT):
        hist = conv_tail(layer.state[REC:], qkv[:t]).view(np.uint32)          # the rows before t, committed
        one = gg.rec(layer, np.concatenate([bits[:REC], hist]).view(F32), qkv[t:t + 1], x[t:t + 1], z[t:t + 1])
        check_run(f"row {t} alone", one)
        for what in ("gate", "beta", "h", "qkv"):
            d, first = cmp_f(what, getattr(w8, what)[t:t + 1], getattr(one, what))
            assert d == 0, (f"row {t}: {d} {what} elements of the 8-row window's row differ from the row run alone "
                            f"(T = 1) or are not finite, first {first}; want them bitwise equal (D15), invariant 1: a "
                            f"row's output never depends on the window width T or its position in it")
        if t == 0:
            d = cmp_b(w8.yq[:1], one.yq)
            assert d == 0, (f"row 0: {d} yq bytes of the 8-row window differ from the row run alone (T = 1); want "
                            f"them bitwise equal (D15), invariant 1")


@pytest.mark.parametrize("combo", range(len(COMBOS)))
def test_gdn_ab_multi_matches_the_float64_reference(combo):
    layer, x = combo_case(combo)
    ref = ab64(x, layer.wa, layer.wb, layer.dt, layer.ssm_a)
    for t in range(1, MT + 1):
        gate, beta, ok = gg.ab(layer, x[:t])
        assert ok, f"T={t}: gdn_ab_multi wrote past its window's rows (a sentinel byte changed)"
        for name, got, want in zip(("gate", "beta"), (gate, beta), ref):
            assert n_bad(got) == 0, f"T={t}: {n_bad(got)} non-finite {name} values"
            e = max_err(got, want[:t])
            assert e < 1e-4, f"T={t}: {name}: max_err {e:.3g}, want max_err < 1e-4"


@pytest.mark.parametrize("trial,nseg", SEG_CASES)
def test_gdn_conv_l2_multi_matches_the_float64_reference(trial, nseg):
    layer, seg = seg_layer(), segments(trial, nseg)
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        st, rows = seg.st0[k, LAYER], slice(r0, r0 + t)
        h, ok = gg.conv_l2(st, seg.qkv[rows], layer.conv_w)
        assert ok, f"segment {k} (T={t}): gdn_conv_l2_multi wrote past its window's rows (a sentinel byte changed)"
        assert n_bad(h) == 0, f"segment {k} (T={t}): {n_bad(h)} non-finite h values, want no non-finite value"
        nil = np.zeros((t, HV), F32)                                  # h reads no gate
        e = max_err(h, cpu_seg(st, seg.qkv[rows], seg.z[rows], layer.conv_w, layer.gamma, nil, nil, t).h)
        assert e < 1e-4, f"segment {k} (T={t}): h: max_err {e:.3g}; want {S8}"


@pytest.mark.parametrize("trial,nseg", SEG_CASES)
def test_gdn_step_norm_multi_matches_the_float64_reference(trial, nseg):
    layer, seg = seg_layer(), segments(trial, nseg)
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        st, rows = seg.st0[k, LAYER], slice(r0, r0 + t)
        h, ok_h = gg.conv_l2(st, seg.qkv[rows], layer.conv_w)
        gate, beta, ok_ab = gg.ab(layer, seg.x[rows])
        y, ok = gg.step_norm(st, h, gate, beta, seg.z[rows], layer.gamma)
        assert ok_h and ok_ab and ok, f"segment {k} (T={t}): a per-stage kernel wrote past its window's rows"
        for name, a in (("h", h), ("gate", gate), ("beta", beta), ("y", y)):
            assert n_bad(a) == 0, f"segment {k} (T={t}): {n_bad(a)} non-finite {name} values, want no non-finite value"
        e = max_err(y, cpu_seg(st, seg.qkv[rows], seg.z[rows], layer.conv_w, layer.gamma, gate, beta, t).y)
        assert e < 1e-3, f"segment {k} (T={t}): y: max_err {e:.3g}; want {S9}"


@pytest.mark.parametrize("trial,nseg", SEG_CASES)
def test_dec_gdn_rec_h_matches_the_float64_reference(trial, nseg):
    layer, seg = seg_layer(), segments(trial, nseg)
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        st, rows, where = seg.st0[k, LAYER], slice(r0, r0 + t), f"segment {k} (T={t})"
        r = gg.rec(layer, st, seg.qkv[rows], seg.x[rows], seg.z[rows])          # the engine's build
        h, ok = gg.conv_l2(st, seg.qkv[rows], layer.conv_w)
        assert r.ok and ok, (f"{where}: wrote past its window's rows (a sentinel byte changed): dec_gdn_rec {r.ok}, "
                             f"gdn_conv_l2_multi {ok} (True: intact)")
        assert np.array_equal(r.state.view(np.uint32), st.view(np.uint32)), f"{where}: dec_gdn_rec changed its state"
        assert n_bad(r.h) == 0, f"{where}: {n_bad(r.h)} non-finite h values, want no non-finite value"
        for what, ref, new in (("h", h, r.h), ("win_qkv", seg.qkv[rows], r.qkv)):
            d, first = cmp_f(what, ref, new)
            assert d == 0, f"{where}: {d} elements differ or are not finite, first {first}; want {S_GATE}"
        nil = np.zeros((t, HV), F32)                                  # h reads no gate
        e = max_err(r.h, cpu_seg(st, seg.qkv[rows], seg.z[rows], layer.conv_w, layer.gamma, nil, nil, t).h)
        assert e < 1e-4, f"{where}: h: max_err {e:.3g}; want {S8}"


@pytest.mark.parametrize("trial,nseg", SEG_CASES)
def test_dec_gdn_rec_y_matches_the_float64_reference(trial, nseg):
    layer, seg = seg_layer(), segments(trial, nseg)
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        st, rows, where = seg.st0[k, LAYER], slice(r0, r0 + t), f"segment {k} (T={t})"
        qkv, x, z = seg.qkv[rows], seg.x[rows], seg.z[rows]
        r, e6 = gg.rec(layer, st, qkv, x, z, y=True), gg.rec(layer, st, qkv, x, z)
        gate, beta, ok_ab = gg.ab(layer, x)
        h, ok_h = gg.conv_l2(st, qkv, layer.conv_w)
        y, ok_y = gg.step_norm(st, h, gate, beta, z, layer.gamma)
        assert r.ok and e6.ok and ok_ab and ok_h and ok_y, (
            f"{where}: wrote past its window's rows (a sentinel byte changed): dec_gdn_rec {r.ok} (GDN_Y_BOUND 0: "
            f"{e6.ok}), gdn_ab_multi {ok_ab}, gdn_conv_l2_multi {ok_h}, gdn_step_norm_multi {ok_y} (True: intact)")
        for run in (r, e6):
            same = np.array_equal(run.state.view(np.uint32), st.view(np.uint32))
            assert same, f"{where}: dec_gdn_rec changed its state (GDN_Y_BOUND {int(run is r)})"
        for what in ("yq", "qkv", "h", "gate", "beta"):
            assert getattr(e6, what).tobytes() == getattr(r, what).tobytes(), (
                f"{where}: {what} of the GDN_Y_BOUND 0 build differs from the GDN_Y_BOUND 1 build's; want them bitwise "
                f"equal (engineering review E6)")
        for name, a in (("h", r.h), ("gate", r.gate), ("beta", r.beta), ("y", r.y)):
            assert n_bad(a) == 0, f"{where}: {n_bad(a)} non-finite {name} values, want no non-finite value"
        for what, ref, new in (("gate", gate, r.gate), ("beta", beta, r.beta)):
            d, first = cmp_f(what, ref, new)
            assert d == 0, f"{where}: {d} elements differ or are not finite, first {first}; want {S_GATE}"
        check_yq(where, y, r.yq)
        e = max_err(r.y, cpu_seg(st, qkv, z, layer.conv_w, layer.gamma, r.gate, r.beta, t).y)
        assert e < 1e-3, f"{where}: y: max_err {e:.3g}; want {S9}"
