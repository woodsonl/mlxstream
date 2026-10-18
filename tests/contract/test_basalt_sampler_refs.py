# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): the min_p values its fixtures pass and its references' host arithmetic.
"""Host-only checks of the sampler family's references (tests/basalt_sampler_ref.py); no GPU.

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
import math
import types

import numpy as np
import pytest

from tests.basalt_sampler_ref import (NINF, SamplerParams, bench_rows, coupled_cases, f2u, f32, fixture12_data,
                                      fixture13_data, fixture16_data, fixture17_data, fixture18_data, mirror_pick,
                                      mirror_prob, mirror_select, mirror_select_scalar, normal_rows, nv_logf,
                                      probe_rows, sampled_k, sampled_reference, u2f)

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
    row = np.array([nv_logf(f32(x)) for x in (0.4, 0.3, 0.2, 0.1)] + [-30.0] * 4, np.float32)
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
