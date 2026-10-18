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
"""

import math

import numpy as np
import pytest

from tests.basalt_sampler_ref import (NINF, SamplerParams, f2u, mirror_select, mirror_select_scalar, nv_logf,
                                      sampled_k, u2f)

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
      present 0.5. top_p, min_p and the seed are drawn too, though the list does not read them.
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
