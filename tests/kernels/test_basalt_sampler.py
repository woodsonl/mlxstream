# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its fixtures, pre-fills and pass criteria.
"""sampler_parity's GPU checks (basalt src/kernels/sampler_parity.cpp) on the Metal sampler family, through
tests/kernels/basalt_sampler_gpu.py. Each check is its own test and asserts its own count is 0, with basalt's gate
quoted in the message; the host references are tests/basalt_sampler_ref.py's.

A greedy check takes a path parameter from ``GREEDY``, the greedy paths built so far: ``greedy_cl``, the cluster
argmax as its partials and merge dispatches (basalt's G-CL, sampler_greedy_cluster_kernel), and ``greedy_1b``, the
one-block greedy kernel forced on the same rows (G-1B, sampler_greedy_kernel; F.1: every greedy check also passes on
G-1B).

- M1.1, fixture 1: plain greedy over normal rows is the lowest-index argmax.
- M2.2, fixture 2: greedy over the order fixture picks 0 in every row.
- M3.1-2, fixture 3: greedy consumes no random number (two seeds, one pick) and is the argmax.
- M5.1-3, fixture 5: temperature 0 (or -1) routes to the argmax whatever ``greedy`` says, and so does greedy at
  temperature 1 (the mode rule, ``basalt_sampler_mode``, chooses greedy; a NaN temperature samples); the pick is the
  argmax, never id 0, the row's worst.
- M11.2, fixture 11: a stale history with penalty_last_n = 0 leaves the greedy pick the argmax.
- W.1, invariant 1: each path's window, row for row, equals its rows run alone.
"""

from dataclasses import replace

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_sampler_ref import (SamplerParams, fixture2_data, fixture12_data, normal_rows, penalty_rows,
                                      reference_pick)
from tests.kernels import basalt_sampler_gpu as gpu
from tests.kernels.test_basalt_decode_cluster import fill_logits

GREEDY = ["greedy_cl", "greedy_1b"]


def run(path, logits, p, want, hist=None, hist_len=0):
    """sampler_parity's ``run`` (:104-132): the path's dispatches on ``logits`` into an output prefilled with -1, then
    the number of rows whose pick is not ``want``'s."""
    got = getattr(gpu, path)(logits, p, hist=hist, hist_len=hist_len)
    return int(np.count_nonzero(got != np.asarray(want, np.int32)))


@pytest.mark.parametrize("path", GREEDY)
def test_basalt_sampler_greedy_argmax(path):
    """M1.1 (:799-808): normal(0, 1) rows, greedy with top_k 0 and top_p 1; each pick is reference_pick's."""
    p = SamplerParams(top_k=0, top_p=1.0, temperature=1.0, greedy=True)
    l = normal_rows(3)
    want = [reference_pick(row, p, False) for row in l]
    assert run(path, l, p, want) == 0, 'bad += run("greedy argmax", l, NT, p, want);'


@pytest.mark.parametrize("path", GREEDY)
def test_basalt_sampler_greedy_order_fixture(path):
    """M2.2 (:845): greedy over fixture 2's rows picks reference_pick's 0 (greedy cannot see the order)."""
    l, p = fixture2_data()
    want = [reference_pick(row, p, False) for row in l]
    assert want == [0] * 4
    assert run(path, l, p, want) == 0, 'bad += run("greedy over this fixture", l, NT, p, want);'


@pytest.mark.parametrize("path", GREEDY)
def test_basalt_sampler_greedy_ignores_seed(path):
    """M3.1-2 (:848-878): seeds 1 and 999,999 (StepState's rng words, one compiled pipeline) give the same picks, and
    the seed-1 picks are reference_pick's. Both outputs are prefilled, which basalt's are not."""
    a = SamplerParams(top_k=20, top_p=0.95, temperature=1.0, greedy=True, seed=1)
    b = replace(a, seed=999999)
    l = normal_rows(5)
    wa = np.int32([reference_pick(row, a, False) for row in l])
    ga, gb = getattr(gpu, path)(l, a), getattr(gpu, path)(l, b)
    mismatch, wrong = int(np.count_nonzero(ga != gb)), int(np.count_nonzero(ga != wa))
    assert mismatch == 0, "if (ga[(size_t) t] != gb[(size_t) t]) ++mismatch;"
    assert wrong == 0, "if (ga[(size_t) t] != wa[(size_t) t]) ++wrong;"
    assert mismatch + wrong == 0, "bad += mismatch + wrong;"


TEMPERATURE_ZERO = [                     # (greedy, temperature, gate): fixture 5's three settings, then the port's -1
    (False, 0.0, 'bad += run("T=0 greedy=false is the argmax", l, NT3, p0, want);'),
    (True, 0.0, 'bad += run("T=0 greedy=true  is the argmax", l, NT3, p1, want);'),
    (True, 1.0, 'bad += run("T=1 greedy=true  is the argmax", l, NT3, p2, want);'),
    (False, -1.0, "temperature -1 with greedy false is the argmax (the port's addition, spec T§8.6)"),
]


@pytest.mark.parametrize("path", GREEDY)
@pytest.mark.parametrize("greedy,temperature,gate", TEMPERATURE_ZERO, ids=["M5.1", "M5.2", "M5.3", "T-1"])
def test_basalt_sampler_temperature_zero_is_argmax(path, greedy, temperature, gate):
    """M5.1-3 (:957-984), plus temperature -1: rows of -5 with 3.0 at 100 + t and -9.0 at id 0, the worst; the mode
    rule must choose greedy (mode 0), as basalt's ``p.greedy || p.temperature <= 0.0f`` does, and the pick is
    100 + t. A NaN temperature is not greedy (the host compare is false), so it samples (mode 2)."""
    assert kernels.basalt_sampler_mode(False, float("nan"), 0, False) == 2
    p = SamplerParams(top_k=0, top_p=1.0, temperature=temperature, greedy=greedy)
    assert kernels.basalt_sampler_mode(p.greedy, p.temperature, p.penalty_last_n, False) == 0, gate
    l = np.full((4, 512), -5, np.float32)
    l[np.arange(4), 100 + np.arange(4)] = 3.0
    l[:, 0] = -9.0
    want = 100 + np.arange(4)
    assert run(path, l, p, want) == 0, gate


@pytest.mark.parametrize("path", GREEDY)
def test_basalt_sampler_stale_history_greedy(path):
    """M11.2 (:1145-1169): fixture 11's rows with an 8-slot history left from a penalised request (token 3 first in
    each row) while this greedy request has penalty_last_n = 0: the history is bound and must not change the pick,
    reference_pick's argmax."""
    l = normal_rows(11)
    hist = np.full(32, -1, np.int32)
    hist[::8] = 3
    p = SamplerParams(top_k=20, top_p=0.95, temperature=0.8, seed=5)
    gp = replace(p, greedy=True, top_k=0, top_p=1.0)
    assert kernels.basalt_sampler_mode(gp.greedy, gp.temperature, gp.penalty_last_n, False) == 0   # G-CL: no window
    gwant = [reference_pick(row, gp, False) for row in l]
    assert run(path, l, gp, gwant, hist, 8) == 0, \
        'bad += run("stale history, last_n=0 (greedy)", l, NT, gp, gwant, hist, 8);'


def w1_ties_across():
    """5 rows of decode_cluster_parity's ties-across pattern at 248,320 logits (the maximum in several threadgroups'
    and threads' ranges, NaN, -inf), greedy without a window."""
    rs = np.random.RandomState(20261017)
    return np.stack([fill_logits(rs, 248320, "ties-across") for _ in range(5)]), \
        SamplerParams(greedy=True, temperature=0.0), None, 0


def w1_per_row_history():
    """Fixture 12's window of T = 8 rows and H = 64 slots with M12.G's request, but penalty_last_n 32: each row counts
    the tail of its own penalty_rows history, whose newest token, the row's favourite, the presence penalty
    overturns (on 7 of the 8 rows, row 0's history, a stride of 32 or the row's head would pick otherwise)."""
    tail, cases = fixture12_data()
    T, H, window, l = next(c for c in cases if c[:2] == (8, 64))
    p = SamplerParams(greedy=True, temperature=0.0, top_k=0, top_p=1.0, penalty_last_n=32, penalty_present=4.0)
    return l, p, penalty_rows(tail, window, T, H), H


ROWS_ALONE = {"greedy_cl": w1_ties_across, "greedy_1b": w1_per_row_history}   # W.1's fixture for each path


@pytest.mark.parametrize("path", ROWS_ALONE)
def test_basalt_sampler_rows_alone(path):
    """W.1: the path's multi-row window, row for row, equals each row run alone as a one-row window at its own
    position (gpu.rows_alone, bitwise)."""
    l, p, hist, hist_len = ROWS_ALONE[path]()
    assert gpu.rows_alone(path, l, p, hist=hist, hist_len=hist_len) == [], \
        "invariant 1: a row's output never depends on the window width T or its position in it"
