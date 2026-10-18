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
G-1B). A penalised greedy check runs G-1B, the path basalt's dispatcher takes for a penalty window. The logit_bias
checks route each launch as the dispatcher does (``sample``, by the mode rule): a bias or a penalty window takes G-1B,
a greedy request with neither G-CL, and their ``forced`` cases run G-1B in G-CL's place (F.1).

- LB.1-6, logit_bias (the greedy sides): the bias added in the kernel picks what the host-biased rows pick, bans an id
  at -inf and boosts one by 100, at 512 logits and at the engine's 248,320, and leaves the rows it reads unchanged.
- M1.1, fixture 1: plain greedy over normal rows is the lowest-index argmax.
- M2.2, fixture 2: greedy over the order fixture picks 0 in every row.
- M3.1-2, fixture 3: greedy consumes no random number (two seeds, one pick) and is the argmax.
- M4.2-3, fixture 4, with A1: the repeat penalty multiplies a negative logit, and the presence penalty is a boolean
  (A1's rows are the ones on which that rule decides the pick).
- M5.1-3, fixture 5: temperature 0 (or -1) routes to the argmax whatever ``greedy`` says, and so does greedy at
  temperature 1 (the mode rule, ``basalt_sampler_mode``, chooses greedy; a NaN temperature samples); the pick is the
  argmax, never id 0, the row's worst.
- M8.2, fixture 8: the penalty window is the tail of the history row.
- M11.2, fixture 11: a stale history with penalty_last_n = 0 leaves the greedy pick the argmax.
- M12.G, fixture 12: each row of a window counts its own history row (windows of 1 to 4,096 slots).
- M13.1, fixture 13: history ids outside the vocabulary count nowhere.
- W.1, invariant 1: each path's window, row for row, equals its rows run alone.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from monolith import kernels
from monolith.runtime import _native as nt
from tests.basalt_sampler_ref import (SamplerParams, f32, fixture2_data, fixture4_data, fixture8_data, fixture12_data,
                                      fixture13_data, greedy_pen, normal_rows, penalty_rows, reference_pick, window_of)
from tests.kernels import basalt_sampler_gpu as gpu
from tests.kernels.basalt_gpu import device
from tests.kernels.test_basalt_decode_cluster import fill_logits

GREEDY = ["greedy_cl", "greedy_1b"]


def run(path, logits, p, want, hist=None, hist_len=0):
    """sampler_parity's ``run`` (:104-132): the path's dispatches on ``logits`` into an output prefilled with -1, then
    the number of rows whose pick is not ``want``'s."""
    got = getattr(gpu, path)(logits, p, hist=hist, hist_len=hist_len)
    return int(np.count_nonzero(got != np.asarray(want, np.int32)))


def sample(logits, p, *, hist=None, hist_len=0, bias=None, forced=False, lbuf=None):
    """sampler_parity's DeviceRows ``sample`` (:493-501) on the path basalt's dispatcher takes for ``p``
    (sampler.cu:988-1020), as the mode rule routes it: G-CL for a greedy request without a penalty window and a bias,
    G-1B for one with either; ``forced``, G-1B in G-CL's place (F.1). ``lbuf``: the rows held in one buffer."""
    mode = kernels.basalt_sampler_mode(p.greedy, p.temperature, p.penalty_last_n, bias is not None)
    assert mode != 2, f"a sampled request, and no sampled path here: {p}"
    if mode == 0 and not forced:
        return gpu.greedy_cl(logits, p, hist=hist, hist_len=hist_len, lbuf=lbuf)
    return gpu.greedy_1b(logits, p, hist=hist, hist_len=hist_len, bias=bias, lbuf=lbuf)


# ---- logit_bias (:719-797) ------------------------------------------------------------------------------------------

def bias_fixture_a():
    """logit_bias fixture A (:723-738, exact data): 4 rows of 512 logits ((7 v + 3 t) % 17) / 4, quarter steps 0 to
    4, but 50 at row 0's id 0; the bias, -inf at id 0, 20 at id 7, -9 at id 16 and 0 elsewhere; ``adjusted``, the
    rows plus the bias, one float32 add each (row 0's id 0 becomes -inf); the history, 7 in all 4 slots of every row.
    Returns (l, bias, adjusted, hist)."""
    l = ((np.arange(512) * 7 + np.arange(4)[:, None] * 3) % 17).astype(np.float32) / np.float32(4)
    l[0, 0] = 50
    bias = np.zeros(512, np.float32)
    bias[[0, 7, 16]] = -np.inf, 20, -9
    return l, bias, l + bias, np.full(16, 7, np.int32)


def bias_request(greedy, penalty):
    """A fixture-A case's request (:739-742): temperature 0 when greedy (else 0.8), seed 71, top_k 16; with the
    penalty, last_n 4, freq 3 and repeat 1.1 (present 0: token 7, counted 4 times, falls to (l + 20) / 1.1 - 12)."""
    p = SamplerParams(greedy=greedy, temperature=0.0 if greedy else 0.8, seed=71, top_k=16)
    return replace(p, penalty_last_n=4, penalty_freq=3.0, penalty_repeat=1.1) if penalty else p


# (greedy, penalty, forced): fixture A's greedy cases, and the unpenalised one with G-1B forced for its launches
# without the bias (the others take G-1B anyway)
BIAS_CASES = [(True, False, False), (True, False, True), (True, True, False)]
BIAS_IDS = ["greedy", "greedy-F", "greedy-penalty"]


def bias_launches(greedy, penalty, forced):
    """One fixture-A case's launches in basalt's order (:743-752), with their picks: ``want``, the request without
    the bias over the rows the host biased; ``got``, with it over the rows held in one buffer, ``raw`` (DeviceRows);
    ``again``, without it over ``raw``; ``plain``, without it over a fresh upload. Then ``raw``'s bits read back.
    Every launch binds the 4-slot history rows, whether or not the request counts them, as DeviceRows does."""
    l, bias, adjusted, hist = bias_fixture_a()
    p = bias_request(greedy, penalty)
    raw = nt.Buffer(device(), l.tobytes())

    def go(rows, **kw):
        return sample(rows, p, hist=hist, hist_len=4, forced=forced, **kw)

    want, got = go(adjusted), go(l, bias=bias, lbuf=raw)
    again, plain = go(l, lbuf=raw), go(l)
    return want, got, again, plain, np.frombuffer(raw.read(0, l.nbytes), np.uint32)


@pytest.mark.parametrize("greedy,penalty,forced", BIAS_CASES, ids=BIAS_IDS)
def test_basalt_sampler_bias_matches_prebiased(greedy, penalty, forced):
    """LB.1 (:743-749): fixture A's request with the bias added in the kernel picks what it picks over the rows the
    host biased (the same float32 add), and never the banned id 0. The port's host reference agrees: the argmax of
    the host-biased rows after the penalty over each row's window (greedy_pen; spec T§8.1)."""
    want, got = bias_launches(greedy, penalty, forced)[:2]
    _, _, adjusted, hist = bias_fixture_a()
    p = bias_request(greedy, penalty)
    host = [greedy_pen(adjusted[t], window_of(hist, 4, t, p.penalty_last_n), p) for t in range(4)]
    assert not (np.any(got != want) or 0 in got), \
        "if (got != want || std::find(got.begin(), got.end(), 0) != got.end()) {"
    assert list(got) == host, f"the host reference over the host-biased rows picks {host}, the kernel {list(got)}"


@pytest.mark.parametrize("greedy,penalty,forced", BIAS_CASES, ids=BIAS_IDS)
def test_basalt_sampler_bias_absent_unchanged(greedy, penalty, forced):
    """LB.2 (:750-752): after the biased launch over ``raw``, the request without the bias picks over ``raw`` what it
    picks over a fresh upload of the same rows."""
    again, plain = bias_launches(greedy, penalty, forced)[2:4]
    assert np.array_equal(again, plain), ('if (raw.sample(p, nullptr) != plain.sample(p, nullptr)) { '
                                          'std::printf("logit_bias: absent bias changed the pick\\n"); ++bad; }')


def test_basalt_sampler_bias_greedy_host_argmax():
    """LB.3 (:754-763): the greedy request at temperature 0 with the bias, over fixture A's rows without a history
    (G-1B: a bias), picks each row's host argmax of the biased rows, the lowest id on a tie: 7 in every row, 20
    above the rest."""
    l, bias, adjusted, _ = bias_fixture_a()
    p = SamplerParams(greedy=True, temperature=0.0)
    got = sample(l, p, bias=bias)
    best = [greedy_pen(row, [], p) for row in adjusted]
    assert best == [7] * 4, f"spec T§8.1: the host argmax is {best}, 7 in every row"
    wrong = int(np.count_nonzero(got != np.int32(best)))
    assert wrong == 0, ('if (got[(size_t) t] != best) { std::printf("logit_bias: greedy row %d is %d, host argmax '
                        '%d\\n", t, got[(size_t) t], best); ++bad; }')


@pytest.mark.parametrize("greedy,penalty,forced", BIAS_CASES, ids=BIAS_IDS)
def test_basalt_sampler_bias_leaves_logits(greedy, penalty, forced):
    """LB.4 (:764-766): ``raw`` reads back as uploaded, bit for bit, after the case's launches over it, with the bias
    and without: no kernel wrote the bias, or anything else, into the logits it reads."""
    back = bias_launches(greedy, penalty, forced)[4]
    unchanged = np.array_equal(back, bias_fixture_a()[0].view(np.uint32).ravel())
    assert unchanged, 'if (unchanged != l) { std::printf("logit_bias: the logits were modified\\n"); ++bad; }'


ONLY = 109266                                     # fixture B's one permitted, or boosted, id (:770)


def bias_fixture_b():
    """logit_bias fixture B (:768-783) at the engine's vocabulary, 248,320: ``large``, one row of 100 but -50 at
    ``ONLY``, with ``bans``, -inf but 0 at ``ONLY``; ``spread``, one row of normal(0, 1) draws (RandomState(5),
    basalt's ``zeros``), with ``boost``, 0 but 100 at ``ONLY``. Returns (large, bans, spread, boost)."""
    V = 248320
    large, bans, boost = np.full((1, V), 100, np.float32), np.full(V, -np.inf, np.float32), np.zeros(V, np.float32)
    large[0, ONLY], bans[ONLY], boost[ONLY] = -50, 0, 100
    return large, bans, normal_rows(5, rows=1, nv=V), boost


def bias_b_requests(greedy):
    """Fixture B's requests on one side (:784-788): top_k (0, 1, 20, 64) x top_p (1, 0.5) x min_p (0, 0.3) x seed
    (1, 2, 3), at temperature 0 when greedy, else 1: 48 rewrites of one record."""
    return [SamplerParams(greedy=greedy, temperature=0.0 if greedy else 1.0, top_k=k, top_p=top_p, min_p=min_p,
                          seed=seed)
            for k, top_p, min_p, seed in itertools.product((0, 1, 20, 64), (1.0, 0.5), (0.0, 0.3), (1, 2, 3))]


BIAS_B_CASES = [(True, False)]                     # (greedy, forced): fixture B's greedy side, G-1B (a bias)


@pytest.mark.parametrize("greedy,forced", BIAS_B_CASES, ids=["greedy"])
def test_basalt_sampler_bias_ban_at_vocab(greedy, forced):
    """LB.5 (:789-790): with every id but ``ONLY`` banned (100 + -inf), every setting picks ``ONLY``: the row's lowest
    logit (-50), and after the bias its only finite one."""
    large, bans, _, _ = bias_fixture_b()
    lost = sum(list(sample(large, p, bias=bans, forced=forced)) != [ONLY] for p in bias_b_requests(greedy))
    assert lost == 0, ('if (many.sample(p, nullptr) != std::vector<int>{only}) { std::printf("logit_bias: the only '
                       'permitted token lost (k=%d greedy=%d)\\n", k, (int) greedy); ++bad; }')


@pytest.mark.parametrize("greedy,forced", BIAS_B_CASES, ids=["greedy"])
def test_basalt_sampler_bias_boost_at_vocab(greedy, forced):
    """LB.6 (:791-792): ``ONLY`` boosted by 100 over normal(0, 1) rows, a margin of about 95, is every setting's
    pick."""
    _, _, spread, boost = bias_fixture_b()
    lost = sum(list(sample(spread, p, bias=boost, forced=forced)) != [ONLY] for p in bias_b_requests(greedy))
    assert lost == 0, ('if (spread.sample(p, nullptr) != std::vector<int>{only}) { std::printf("logit_bias: the '
                       'boosted token lost (k=%d greedy=%d)\\n", k, (int) greedy); ++bad; }')


# ---- the fixtures of main (:799-1276) --------------------------------------------------------------------------------

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


def test_basalt_sampler_penalty_repeat_on_negatives():
    """M4.2 (:911-931): fixture 4 A, every logit negative, token 0 (-1.0) in each row's history and token 1 (-1.2)
    not: the repeat penalty 2 multiplies token 0 to -2.0 and token 1 wins, where dividing it (-0.5) would keep token
    0; the run gates only when that difference shows (M4.1). G-1B: a penalty window."""
    la, hist_a, _, _, p = fixture4_data()
    want_a = [greedy_pen(la[t], hist_a[4 * t:4 * t + 4], p) for t in range(2)]
    alt_a = [greedy_pen(la[t], hist_a[4 * t:4 * t + 4], p, divide_unconditionally=True) for t in range(2)]
    assert want_a[0] != alt_a[0], "const bool A_visible = want_a[0] != alt_a[0]; ... if (!A_visible) ++bad;"
    assert run("greedy_1b", la, p, want_a, hist_a, 4) == 0, \
        'else bad += run("penalties: repeat on negatives", la, NT2, p, want_a, hist_a, 4);'


def test_basalt_sampler_penalty_presence_boolean():
    """M4.3 (:933-954) and A1 (spec T§10.2): fixture 4 B, token 0 at 5.0 counted twice and token 1 at 3.4, with
    present 1.5 and no freq: token 0 falls to 5.0 / 2 - 1.5 = 1.0 and token 1 wins. It would win as well under
    presence times count (-0.5) or no presence term (2.5), so M4.3 cannot see its name. A1 runs the rows with token 1
    at 0.0 and at 1.5, where the boolean rule picks (0, 1), presence times count (1, 1) and no presence term (0, 0).
    G-1B: a penalty window."""
    _, _, lb, hist_b, p = fixture4_data()
    q = replace(p, penalty_present=1.5, penalty_freq=0.0)
    want_b = [greedy_pen(lb[t], hist_b[4 * t:4 * t + 4], q) for t in range(2)]
    assert run("greedy_1b", lb, q, want_b, hist_b, 4) == 0, \
        'bad += run("penalties: presence is boolean", lb, NT2, q, want_b, hist_b, 4);'
    rows = lb.copy()
    rows[:, 1] = 0.0, 1.5
    want = [greedy_pen(rows[t], hist_b[4 * t:4 * t + 4], q) for t in range(2)]           # the boolean rule
    counted = [int(rows[t, 1] > f32(5.0) / f32(2.0) - f32(2.0) * f32(1.5)) for t in range(2)]
    absent = [int(rows[t, 1] > f32(5.0) / f32(2.0)) for t in range(2)]
    assert (want, counted, absent) == ([0, 1], [1, 1], [0, 0]), \
        f"spec T§10.2 A1: the boolean rule picks {want}, presence times count {counted}, no presence term {absent}"
    assert run("greedy_1b", rows, q, want, hist_b, 4) == 0, "A1: picks (0, 1) on the two presence rows"


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


def test_basalt_sampler_penalty_window_tail():
    """M8.2 (:1061-1093): fixture 8's 8-slot history with penalty_last_n 4, the file's one window shorter than its
    history: only the last four slots count, so token 3 (6.5, in all four) falls below token 0 (6.0, only in the
    first four) and the pick is 0; all eight would pick 3, and the run gates only when that shows (M8.1). G-1B: a
    penalty window."""
    l, hist, p = fixture8_data()
    want, unclamped = greedy_pen(l[0], window_of(hist, 8, 0, p.penalty_last_n), p), greedy_pen(l[0], hist, p)
    assert want != unclamped, "const bool visible = want != unclamped; ... if (!visible) ++bad;"
    assert run("greedy_1b", l, p, [want], hist, 8) == 0, \
        'else bad += run("penalty window: tail only", {l.begin(), l.end()}, 1, p, {want}, hist, 8);'


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


FX12_CASES = [(T, H) for T in (1, 2, 4, 8) for H in (1, 64, 1024, 4096)]


@pytest.mark.parametrize("T,H", FX12_CASES, ids=[f"T{T}-H{H}" for T, H in FX12_CASES])
def test_basalt_sampler_per_row_history_greedy(T, H):
    """M12.G (:1171-1223): fixture 12's window of T rows, each with its own penalty_rows history of H slots, all of
    them its window (penalty_last_n = H = hist_len), its newest token a favourite at 5.0 that the presence penalty 4
    overturns: each row's pick is greedy_pen over its own history row. G-1B: a penalty window."""
    tail, cases = fixture12_data()
    window, l = next(c[2:] for c in cases if c[:2] == (T, H))
    rows = penalty_rows(tail, window, T, H)
    gp = SamplerParams(greedy=True, temperature=0.0, top_k=0, top_p=1.0, penalty_last_n=H, penalty_present=4.0)
    gwant = [greedy_pen(l[t], rows[t * H:(t + 1) * H], gp) for t in range(T)]
    assert run("greedy_1b", l, gp, gwant, rows, H) == 0, \
        f'bad += run(name, l, T, gp, gwant, rows_int, H); (name "per-row history T={T} H={H} greedy")'


def test_basalt_sampler_history_ids_out_of_vocab_greedy():
    """M13.1 (:1234-1274): fixture 13's 16-slot histories hold ids outside the vocabulary in every third slot (512,
    1512, 0x7fffffff, -5, 1 << 20) beside drawn ids, one of them each row's favourite at 4.0: no window counts them
    and they set no bitmap bit, so each pick is the greedy reference over the drawn ids alone (presence 3). G-1B: a
    penalty window; a bitmap store past the scratch's end would be a shader-validation report."""
    l, hist, valid = fixture13_data()
    gp = SamplerParams(greedy=True, temperature=0.0, top_k=0, top_p=1.0, penalty_last_n=16, penalty_present=3.0)
    gwant = [greedy_pen(l[t], valid[16 * t:16 * (t + 1)], gp) for t in range(4)]
    assert run("greedy_1b", l, gp, gwant, hist, 16) == 0, \
        'bad += run("out-of-vocab history ids (greedy)", l, NT, gp, gwant, hist, H);'


# ---- W.1 (invariant 1) -----------------------------------------------------------------------------------------------

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
