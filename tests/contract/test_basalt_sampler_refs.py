# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its fixtures, its host-only checks and their pass criteria, the min_p values its fixtures
# pass and its references' host arithmetic.
"""Host-only checks of the sampler family's references (tests/basalt_sampler_ref.py); no GPU.

sampler_parity's host-only checks (spec T§11's rows marked H), each with basalt's criterion verbatim and the numbers
spec T§8 pins:
- M2.1, M4.1, M6.1, M7.1, M8.1: the fixture's host references see the rule its kernel check gates (the top_p cut moves
  with the order, the repeat penalty's sign rule, the sampled penalties, the min_p threshold, the window's tail).
- M9.1 and M10.1: the reference's draws on fixtures 9 and 10 (Philox only: 832 zeros, counts 115/83/58/0).
- M12.O and M17.2, asserted on the port's own seeds: per-row histories change some drafted row's pick, and some
  sampled pick ties with another entry of its list.
- M15.1-2: ``penalty_rows`` lays out each row as the last h tokens of the tail and the window, against the plain
  definition and the old single-row staging.

The reference checks of the port's own:
- ``nv_logf`` (CUDA's logf) of the seven ``min_p`` values sampler_parity's fixtures pass is the correctly rounded
  logarithm's float, so the min_p threshold the references compute is the exact one. The table gives each value's
  correctly rounded logf and the exact logarithm's distance from it in ulps: 0.3f and 0.9f lie within 0.07 and 0.09
  ulp of a rounding midpoint, so a logf with a larger error before rounding may return the neighbour there.
  ``math.log``'s binary64 result, rounded to float32, confirms the table: its error is below 2^-28 float ulps.
- The vectorized top-k list (``mirror_select``: bincount counts, the elementwise penalty, a stable sort) equals
  basalt's O(k x nv) loop (``mirror_select_scalar``), ids and logit bits, on 44 rows that have ties, signed zeros,
  +-inf, NaN, ids outside the vocabulary and fewer candidates than places.
- On the 40 of them with only finite logits, ``mirror_pick`` over that list draws ``sampled_reference``'s token at
  rows 0 and 1's counters: the mirror stands in for basalt's sampled reference on such rows (spec T§3.10).
- ``mirror_prob`` on fixture 10's row (probabilities 0.4 / 0.3 / 0.2 / 0.1, top_p 0.75, min_p 0.3) keeps the three
  that top_p keeps before min_p, then applies the temperature: min_p first would keep two.
- Fixture 16's generator writes row 3's history as T§8.18 lists it, every third slot repeating its edge.
- Every fixture generator, called twice with its seed, gives the same bytes.
"""

import hashlib
import itertools
import math
import types
from dataclasses import replace

import numpy as np
import pytest

from tests.basalt_sampler_ref import (NINF, SamplerParams, bench_rows, coupled_cases, f2u, f32, fixture2_data,
                                      fixture4_data, fixture6_data, fixture7_data, fixture8_data, fixture9_data,
                                      fixture10_data, fixture12_data, fixture13_data, fixture16_data, fixture17_data,
                                      fixture18_data, greedy_pen, mirror_pick, mirror_prob, mirror_select,
                                      mirror_select_scalar, normal_rows, nv_logf, penal, penalty_rows,
                                      philox_uniform, probe_rows, reference_cut, reference_pick, sampled_cut,
                                      sampled_k, sampled_reference, u2f, window_of)

# min_p (float bits), its correctly rounded logf (float bits), the exact logarithm's offset from that float (ulps)
MIN_P_LOGF = [(0x3D4CCCCD, 0xC03FBA14, 0.205), (0x3DCCCCCD, 0xC0135D8E, 0.197), (0x3E4CCCCD, 0xBFCE0210, 0.377),
              (0x3E99999A, 0xBF9A1BC8, 0.435), (0x3ECCCCCD, 0xBF6A9207, -0.278), (0x3F000000, 0xBF317218, 0.032),
              (0x3F666666, 0xBDD7C745, 0.412)]


@pytest.mark.parametrize("x, want, offset", MIN_P_LOGF, ids=[f"{u2f(x):.2g}" for x, _, _ in MIN_P_LOGF])
def test_basalt_sampler_ref_nv_logf_of_min_p_correctly_rounded(x, want, offset):
    exact = math.log(u2f(x))
    assert f2u(float(np.float32(exact))) == want, "the table's logf is not the correctly rounded one"
    ulp = 2.0 ** (math.frexp(u2f(want))[1] - 24)                          # the float's ulp
    assert abs((exact - u2f(want)) / ulp - offset) < 1e-3, "the table's offset"
    got = f2u(float(nv_logf(u2f(x))))
    assert got == want, f"nv_logf({u2f(x)!r}f = {x:#010x}) = {got:#010x}; the correctly rounded logf is {want:#010x}"


def _mirror_cases():
    """The rows of the vectorized-against-scalar check, drawn in order from one RandomState(12).
    - 40 rows of 512 normal logits, every fourth on quarter steps (ties). Each has a window of 0, 1, 16 or 64 ids in
      [-3, 515), so some fall outside the vocabulary, top_k 1, 5, 20 or 64, and penalties repeat 1.1, freq 0.25,
      present 0.5. top_p and min_p are drawn too (seed = case, counter 77), though the list reads none of them.
    - Four rows of 1,000 on quarter steps, at top_k 64 with repeat 2, freq 0.25, present 0.5. They are modelled on
      basalt's fixture 16 (:1365-1457) with fixed ids, without its spread() or its 6.25/6.0 band:
      - row 0: the quarter steps alone;
      - row 1: every value made non-positive, each zero signed by its id's parity (-0 at even ids), and +inf at 3;
      - row 2: -inf everywhere but 1.0 at 400-404, 0.5 at 405-409, NaN at 500 and 501 and +inf at 502. That leaves
        11 candidates for 64 places, so the list ends in (0, -inf) pads;
      - row 3: 9.0 at 0, 501 and 998, which the window [0, 501, 998, 998, 1000, 1077, -5] penalises to 3.75, 3.75
        and 3.5. Those are the values that 600-614 and 615-629 hold. +inf sits at 500, outside the window."""
    rs = np.random.RandomState(12)
    for case in range(40):
        nv = 512
        l = rs.standard_normal(nv).astype(np.float32)
        if case % 4 == 1: l = (np.floor(l * 4) / 4).astype(np.float32)
        win = list(rs.randint(-3, nv + 3, size=int(rs.choice([0, 1, 16, 64]))))
        p = SamplerParams(top_k=int(rs.choice([1, 5, 20, 64])), top_p=float(rs.choice([0.9, 1.0])),
                          min_p=float(rs.choice([0.0, 0.05])), temperature=0.8, seed=case, counter=77,
                          penalty_last_n=len(win), penalty_repeat=1.1, penalty_freq=0.25, penalty_present=0.5)
        yield f"nv512 case {case}", l, win, p, sampled_k(p.top_k, nv)
    inf, nan, nv = np.float32(np.inf), np.float32(np.nan), 1000
    p = SamplerParams(top_k=64, top_p=1.0, temperature=0.8, seed=16, penalty_last_n=64, penalty_repeat=2.0,
                      penalty_freq=0.25, penalty_present=0.5)
    for t in range(4):
        l = (np.floor(rs.standard_normal(nv) * 4) / 4).astype(np.float32)
        if t == 1:
            l = -np.abs(l); l[l == 0] = np.where(np.arange(nv)[l == 0] & 1, np.float32(0.0), np.float32(-0.0)); l[3] = inf
        if t == 2:
            l[:] = NINF; l[[400, 401, 402, 403, 404]] = 1.0; l[[405, 406, 407, 408, 409]] = 0.5; l[[500, 501]] = nan
            l[502] = inf
        if t == 3:
            l[[0, 501, 998]] = 9.0; l[600:615] = 3.75; l[615:630] = 3.5; l[500] = inf
        win = [0, 501, 998, 998, 1000, 1077, -5] if t == 3 else []
        yield f"nv1000 row {t}", l, win, p, 64


def test_basalt_sampler_ref_mirror_select_vectorized_matches_scalar():
    rows = list(_mirror_cases())
    bits = lambda sel: np.asarray(sel.logit, np.float32).view(np.uint32).tolist()
    bad = []
    for name, l, win, p, k in rows:
        got = mirror_select(l, len(l), win, p, k)
        want = mirror_select_scalar(list(l), len(l), win, p, k)
        if got.ids != want.ids or bits(got) != bits(want): bad.append(name)
    assert len(rows) == 44 and not bad, f"vectorized vs scalar: {len(rows)} cases, {len(bad)} mismatches: {bad}"


def test_basalt_sampler_ref_mirror_pick_matches_sampled_reference():
    rows = [case for case in _mirror_cases() if case[0].startswith("nv512")]
    bad = [(name, t) for name, l, win, p, k in rows for t in (0, 1)
           if mirror_pick(mirror_select(l, len(l), win, p, k), k, p, t) != sampled_reference(l, win, p, t)]
    assert len(rows) == 40 and not bad, \
        f"mirror_pick(mirror_select) vs sampled_reference: {2 * len(rows)} picks, {len(bad)} differ: {bad}"


@pytest.mark.parametrize("temperature, want", [(1.0, [4 / 9, 3 / 9, 2 / 9]), (0.5, [16 / 29, 9 / 29, 4 / 29])])
def test_basalt_sampler_ref_mirror_prob_cuts_then_scales(temperature, want):
    p = SamplerParams(top_k=4, top_p=0.75, min_p=0.3, temperature=temperature)
    row = fixture10_data()[0][0]
    sel = mirror_select(row, 8, [], p, 4)
    got = [mirror_prob(sel, 4, p, q) for q in range(8)]
    assert sel.ids == [0, 1, 2, 3] and max(abs(g - w) for g, w in zip(got, want + [0.0] * 5)) < 1e-6, \
        f"mirror_prob over ids {sel.ids}: {got}; top_p keeps three, min_p keeps them, the temperature last: {want}"


@pytest.mark.parametrize("nv", [1000, 248320])
def test_basalt_sampler_ref_fixture16_history(nv):
    h = fixture16_data(nv)[1].reshape(4, 64)
    want = ([0, 501, 998, 998, 1000, 1077, -5] if nv == 1000 else
            [0, 1023, 1024, 1024, 4095, 4096, 4096, 8191, 8192, 8192, 12345, nv // 2 + 1, nv // 2 + 1, nv - 2, nv,
             nv + 77, -5])
    assert (h[:3] == -1).all() and h[3].tolist() == want + [-1] * (64 - len(want)), \
        f"fixture 16 history at nv {nv}: row 3 {h[3].tolist()}, want {want} then -1; rows 0-2 all -1"


GENERATORS = {  # one per random block of sampler_parity (spec T§4), at its cheapest shape
    "logit_bias B": lambda: normal_rows(5, 1, 248320), "fixture 1": lambda: normal_rows(3),
    "fixture 3": lambda: normal_rows(5), "fixture 11": lambda: normal_rows(11), "fixture 12": fixture12_data,
    "fixture 13": fixture13_data, "fixture 14": lambda: normal_rows(14, scale=0.3),
    "fixture 16": lambda: fixture16_data(1000), "fixture 17": lambda: fixture17_data(512),
    "fixture 18": lambda: fixture18_data(512, 64, 1874), "probe": lambda: probe_rows(7), "bench": lambda: bench_rows(4),
    "coupled": coupled_cases,
}


def _digest(x, h):
    """Feeds a generator's output to ``h``: each array's dtype, shape and bytes (NaN compares by its bits), each other
    value's repr; tuples, lists and generators element by element."""
    if isinstance(x, np.ndarray): h.update(f"{x.dtype.str}{x.shape}".encode() + x.tobytes())
    elif isinstance(x, (tuple, list, types.GeneratorType)):
        for v in x: _digest(v, h)
    else: h.update(repr(x).encode())
    return h.hexdigest()


@pytest.mark.parametrize("block", GENERATORS)
def test_basalt_sampler_ref_generators_deterministic(block):
    a, b = (_digest(GENERATORS[block](), hashlib.sha256()) for _ in range(2))
    assert a == b, f"{block}: two calls with the same seed gave other bytes"


# ---- sampler_parity's host-only checks (spec T§11's rows marked H), in its order -----------------------------------

FX2_HEAD = [3.0, 2.549999952316284, 2.5, 2.4499998092651367, 2.3999998569488525, 2.3499999046325684, 2.299999952316284,
            2.25]


def test_basalt_sampler_order_fixture_observable():
    """M2.1 (:833-843): fixture 2's top_p cut depends on the order, temperature after the cut (4 survivors per row) or
    first (3), though its greedy pick does not (0 under both)."""
    l, p = fixture2_data()
    spec, alt = [reference_cut(r, p, False) for r in l], [reference_cut(r, p, True) for r in l]
    cut_spec, cut_alt = sum(spec), sum(alt)
    assert cut_spec != cut_alt, ("`const bool distinguishable = (cut_spec != cut_alt);` ... "
                                 f"`if (!distinguishable) ++bad;`: survivors {cut_spec} under both orders")
    head, picks = [float(x) for x in l[0, :8]], [reference_pick(r, p, tf) for tf in (False, True) for r in l]
    assert (spec, alt, head, picks) == ([4] * 4, [3] * 4, FX2_HEAD, [0] * 8), \
        f"spec T§8.3: cuts per row {spec} / {alt} (want 4 / 3), head {head}, picks {picks} (want 0)"


def test_basalt_sampler_penalty_fixture_observable():
    """M4.1 (:926-930): on fixture 4 A's negative logits the repeat penalty's rule decides the greedy pick: multiplying
    token 0's -1.0 (to -2.0) lets token 1 win, dividing it (to -0.5) would keep token 0."""
    la, hist_a, _, _, p = fixture4_data()
    want_a = [greedy_pen(la[t], hist_a[4 * t:4 * t + 4], p) for t in range(2)]
    alt_a = [greedy_pen(la[t], hist_a[4 * t:4 * t + 4], p, divide_unconditionally=True) for t in range(2)]
    assert want_a[0] != alt_a[0], ("`const bool A_visible = want_a[0] != alt_a[0];` ... `if (!A_visible) ++bad;`: "
                                   f"both rules pick {want_a[0]}")
    assert (want_a, alt_a) == ([1, 1], [0, 0]), f"spec T§8.5: multiply-rule picks {want_a} (want 1), divide-always " \
                                                f"{alt_a} (want 0)"


def test_basalt_sampler_sampled_penalty_observable():
    """M6.1 (:1013-1017): with fixture 6's penalties on, token 0 (9.0, penalised to 2.2 and 2.0) loses the sampled
    picks it wins penalty-free."""
    l, hist, p = fixture6_data()
    clean_p = replace(p, penalty_last_n=0, penalty_repeat=1.0, penalty_freq=0.0, penalty_present=0.0)
    want = [sampled_reference(l[t], hist[4 * t:4 * t + 4], p, t) for t in range(2)]
    clean = [sampled_reference(l[t], hist[4 * t:4 * t + 4], clean_p, t) for t in range(2)]
    assert want[0] != clean[0] or want[1] != clean[1], ("`const bool visible = want[0] != clean[0] || want[1] != "
                                                        f"clean[1];` ... `if (!visible) ++bad;`: {want}, {clean}")
    u = [f"{philox_uniform(p.seed, t):.7g}" for t in range(2)]
    assert (want, clean, u) == ([3, 2], [0, 0], ["0.9025966", "0.6366587"]), \
        f"spec T§8.7: penalised {want} (want [3, 2]), clean {clean} (want [0, 0]), draws {u}"


def test_basalt_sampler_min_p_observable():
    """M7.1 (:1043-1046): fixture 7's survivor count falls with min_p 0, 0.5 and 0.9 (6, 2 and 1 per row: thresholds
    3.307 and 3.895 against 3.2 and 3.5), and min_p 0.9 keeps one per row."""
    l, base = fixture7_data()
    qs = [replace(base, min_p=mp) for mp in (0.0, 0.5, 0.9)]
    c0, c05, c09 = (sum(sampled_cut(r, [], q) for r in l) for q in qs)
    NT3 = len(l)
    assert c0 > c05 and c05 > c09 and c09 >= NT3, \
        f"`const bool visible = c0 > c05 && c05 > c09 && c09 >= NT3;` ... `if (!visible) ++bad;`: {c0}, {c05}, {c09}"
    picks = [[sampled_reference(r, [], q, t) for t, r in enumerate(l)] for q in qs]
    assert ((c0, c05, c09), picks) == ((12, 4, 2), [[2, 2], [1, 1], [0, 0]]), \
        f"spec T§8.8: survivors {(c0, c05, c09)} (want 12, 4, 2), picks {picks} (want [2, 2], [1, 1], [0, 0])"


def test_basalt_sampler_penalty_window_observable():
    """M8.1 (:1087-1091): counting only the last penalty_last_n = 4 of fixture 8's 8 slots changes the greedy pick:
    token 3 falls to 0.4666667 and token 0 keeps 6.0; all 8 would drop token 0 to 0.29999995 and keep token 3."""
    l, hist, p = fixture8_data()
    want, unclamped = greedy_pen(l[0], hist[4:], p), greedy_pen(l[0], hist, p)
    assert want != unclamped, f"`const bool visible = want != unclamped;` ... `if (!visible) ++bad;`: both pick {want}"
    vals = (penal(l[0, 3], 4, p), penal(l[0, 0], 4, p))
    assert (want, unclamped, vals) == (0, 3, (f32(0.4666667), f32(0.29999995))), \
        f"spec T§8.9: clamped pick {want} (want 0), full-history pick {unclamped} (want 3), penalised {vals}"


def test_basalt_sampler_one_penalty_stage_share():
    """M9.1 (:1113-1117): the reference's share of token 0 on fixture 9's 8,000 rows is near 1 / (1 + exp(1.5 / 0.7))
    = 0.105, what one penalties stage gives (a second one after the temperature would give 0.0255): 832 draws,
    Philox only."""
    l, hist, p = fixture9_data()
    want = [sampled_reference(l[t], hist[t:t + 1], p, t) for t in range(len(l))]
    zeros = want.count(0)
    expect, share = 1.0 / (1.0 + math.exp(1.5 / 0.7)), zeros / len(l)
    assert abs(share - expect) < 0.014, ("`const bool near = std::fabs(share - expect) < 0.014;` ... "
                                         f"`if (!near) ++bad;`: share {share:.4f}, expected {expect:.4f}")
    assert zeros == 832, f"spec T§8.10: token 0 drawn {zeros} times of {len(l)} (want 832)"


def test_basalt_sampler_top_p_before_min_p_observable():
    """M10.1 (:1139-1141): top_p before min_p keeps three of fixture 10's 0.4 / 0.3 / 0.2 / 0.1, so token 2 is drawn;
    min_p first would keep two and never draw it. Philox only: counts 115, 83, 58, 0."""
    l, p = fixture10_data()
    want = [sampled_reference(r, [], p, t) for t, r in enumerate(l)]
    twos = want.count(2)
    assert twos != 0, f"`if (twos == 0) ++bad;`: token 2 never drawn in {len(l)} rows"
    counts = [want.count(i) for i in range(4)]
    assert counts == [115, 83, 58, 0], f"spec T§8.11: tokens 0-3 drawn {counts} times (want [115, 83, 58, 0])"


def test_basalt_sampler_per_row_history_observable():
    """M12.O (:1228-1231): with row 0's ``penalty_rows`` history in place of its own, some drafted row (t > 0) of
    fixture 12's 16 (T, H) cases picks otherwise under greedy_pen; asserted on the port's seed (12)."""
    tail, cases = fixture12_data()
    observable = rows_checked = 0
    for T, H, window, l in cases:
        rows = penalty_rows(tail, window, T, H)
        gp = SamplerParams(greedy=True, temperature=0.0, top_k=0, top_p=1.0, penalty_last_n=H, penalty_present=4.0)
        for t in range(T):
            gwant = greedy_pen(l[t], rows[t * H:(t + 1) * H], gp)
            observable += t > 0 and greedy_pen(l[t], rows[:H], gp) != gwant
            rows_checked += 1
    assert rows_checked == 60, f"spec T§8.13: {rows_checked} rows checked (want 60)"
    assert observable != 0, f"`if (observable == 0) ++bad;`: no drafted row of {rows_checked} picks differently"


def test_basalt_penalty_rows_layout():
    """M15.1-2 (:1300-1334): for every (n_tail, T, h) of 7 x 3 x 4, ``penalty_rows`` into slots pre-filled 12345 gives
    each row t the last h tokens of tail + window[0..t], -1 padded in front (M15.1), and row 0 the old single-row
    staging, the consumed tail then the fed-back head last (M15.2): 420 rows."""
    wrong1 = wrong2 = cases = 0
    for n_tail, T, h in itertools.product((0, 1, 5, 63, 64, 65, 300), (1, 3, 8), (1, 4, 64, 100)):
        tail, window = list(range(1000, 1000 + n_tail)), list(range(5000, 5000 + T))
        rows = penalty_rows(tail, window, T, h, np.full(T * h, 12345, np.int32)).tolist()
        for t in range(T):
            seq = tail + window[:t + 1]
            take = min(h, len(seq))
            expect = [-1] * (h - take) + seq[len(seq) - take:]
            wrong1 += rows[t * h:(t + 1) * h] != expect
            cases += 1
        take0 = min(h, n_tail + 1)
        old = [-1] * h
        for j in range(take0 - 1): old[h - take0 + j] = tail[n_tail - (take0 - 1) + j]
        old[h - 1] = window[0]
        wrong2 += rows[:h] != old
        cases += 1
    assert cases == 420, f"spec T§8.16: {cases} rows checked (want 420)"
    assert wrong1 == 0, ("`if (!std::equal(expect.begin(), expect.end(), rows.begin() + (size_t) t * h)) ++wrong;`: "
                         f"{wrong1} of {cases - 84} rows differ")
    assert wrong2 == 0, f"`if (!std::equal(old.begin(), old.end(), rows.begin())) ++wrong;`: {wrong2} of 84 rows differ"


def test_basalt_sampler_sampled_draws_tie_observable():
    """M17.2 (:1532-1534): some of the mirror's 1,632 sampled picks on fixture 17's tie-heavy rows tie with another
    entry of their top_k list, so M17.1's equality gates the tie rule; asserted on the port's seeds (1700 + nv)."""
    T, H = 17, 64
    draws = tied = 0
    for nv in (248320, 512):
        l, hist = fixture17_data(nv)
        config = 0
        for pen in (0, 1):
            base = SamplerParams(temperature=2.5, seed=17, penalty_last_n=H if pen else 0, penalty_repeat=1.25,
                                 penalty_freq=0.25, penalty_present=0.5)
            lists = [mirror_select(l[t], nv, window_of(hist, H, t, base.penalty_last_n), base, sampled_k(64, nv))
                     for t in range(T)]
            # basalt's last axis is the stream (the legacy one, a created one): each launch takes its own counter
            for top_k, top_p, min_p, _ in itertools.product((1, 20, 64), (0.9, 1.0), (0.0, 0.05), range(2)):
                config += 1
                p = replace(base, top_k=top_k, top_p=top_p, min_p=min_p, counter=1000 * config)
                k = sampled_k(top_k, nv)
                for t, sl in enumerate(lists):
                    want = mirror_pick(sl, k, p, t)
                    same = next((sum(sl.logit[j] == sl.logit[i] for j in range(k))
                                 for i in range(k) if sl.ids[i] == want), 0)
                    tied += same > 1
                    draws += 1
    assert draws == 1632, f"spec T§8.19: {draws} draws (want 1632)"
    assert tied != 0, f"`bad += wrong + (tied == 0);`: none of {draws} picks ties with another kept token"
