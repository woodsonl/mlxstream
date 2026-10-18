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
a greedy request with neither G-CL, a sampled one S-SPLIT, and their ``forced`` cases run G-1B in G-CL's place (F.1).
A sampled check takes its path from ``SAMPLED``, the sampled paths built so far: ``split``, the split part and merge
(S-SPLIT, sampler_split_part_kernel then sampler_split_merge_kernel).

- LB.1-6, logit_bias (both sides): the bias added in the kernel picks what the host-biased rows pick (the sampled
  penalised case also shows the order: bias, then penalties), bans an id at -inf and boosts one by 100, at 512 logits
  and at the engine's 248,320 (61 split blocks, 60 of them empty after the ban), and leaves the rows it reads
  unchanged.
- M1.1, fixture 1: plain greedy over normal rows is the lowest-index argmax.
- M2.2, fixture 2: greedy over the order fixture picks 0 in every row.
- M3.1-2, fixture 3: greedy consumes no random number (two seeds, one pick) and is the argmax.
- M4.2-3, fixture 4, with A1: the repeat penalty multiplies a negative logit, and the presence penalty is a boolean
  (A1's rows are the ones on which that rule decides the pick, on G-1B and the sampled paths).
- M5.1-3, fixture 5: temperature 0 (or -1) routes to the argmax whatever ``greedy`` says, and so does greedy at
  temperature 1 (the mode rule, ``basalt_sampler_mode``, chooses greedy; a NaN temperature samples); the pick is the
  argmax, never id 0, the row's worst.
- M6.2, fixture 6: the penalties in the sampled chain turn its picks from [0, 0] to sampled_reference's [3, 2].
- M7.2-4, fixture 7: min_p 0, 0.5 and 0.9 through the sampled chain pick sampled_reference's token.
- M8.2, fixture 8: the penalty window is the tail of the history row.
- M11.1-2, fixture 11: a stale history with penalty_last_n = 0 leaves the sampled pick the no-history reference's and
  the greedy pick the argmax.
- M12.G and M12.S, fixture 12: each row of a window counts its own history row (windows of 1 to 4,096 slots), greedy
  and sampled.
- M13.1-2, fixture 13: history ids outside the vocabulary count nowhere, greedy and sampled.
- M14.1-4, fixture 14: top_k 64 as given, and 0, 100 and -3 meaning 64, pick the 64-wide reference's token.
- M16.1-2, fixture 16: under NaN arithmetic the pick is the list's k-th entry, so top_k 0 to 64 read the whole list
  (ties, signed zeros, NaN, +inf, the empty rounds' sentinel, penalised ties) at 61, 25, 64 and 1 split blocks.
- M17.1, fixture 17: tie-heavy rows through the whole chain on two queues, against the mirror.
- M18.2, fixture 18 (a): the split path on penalised rows at the engine's vocabulary, against the mirror.
- A2-A4, A7 and A8, the port's additions (spec T§10.2): the sampled kernels' penalty window is the tail; the
  temperature comes after the filters; min_keep 3 and 10 clamp the top_p cut; subnormal logits are ordered exactly;
  min_p above 1 keeps nothing and picks the list's head.
- TP.1, the port's addition (spec S§19.3): the drafter's greedy tail, ``top_prob`` (the cluster partials, then
  bs_greedy_top_prob), picks the host argmax, mapped, with the float64 softmax probability of the pick within
  q5k_head_parity's 1e-5.
- PL.1, the port's building-block check of the split path's first stage (``split_part``, bs_split_part alone): each
  4,096-logit block's list in the scratch equals mirror_select's over the block, ids and value bits.
- W.1, invariant 1: each path's window, row for row, equals its rows run alone (the top-prob path's probabilities
  bit for bit too); the split's window is also the mirror's (mirror_select, then mirror_pick), row for row.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from monolith import kernels
from monolith.runtime import _native as nt
from tests.basalt_sampler_ref import (SamplerParams, f32, fixture2_data, fixture4_data, fixture6_data, fixture7_data,
                                      fixture8_data, fixture12_data, fixture13_data, fixture16_data, fixture17_data,
                                      fixture18_data, greedy_pen, mirror_pick, mirror_prob, mirror_select, normal_rows,
                                      penal_vec, penalty_rows, reference_pick, sampled_k, sampled_reference, select_vec,
                                      window_of)
from tests.kernels import basalt_sampler_gpu as gpu
from tests.kernels.basalt_gpu import device
from tests.kernels.test_basalt_decode_cluster import PATTERNS, VOCABS, fill_logits, host_argmax

GREEDY = ["greedy_cl", "greedy_1b"]
SAMPLED = ["split"]


def run(path, logits, p, want, hist=None, hist_len=0):
    """sampler_parity's ``run`` (:104-132): the path's dispatches on ``logits`` into an output prefilled with -1, then
    the number of rows whose pick is not ``want``'s."""
    got = getattr(gpu, path)(logits, p, hist=hist, hist_len=hist_len)
    return int(np.count_nonzero(got != np.asarray(want, np.int32)))


def sample(logits, p, *, hist=None, hist_len=0, bias=None, forced=False, lbuf=None):
    """sampler_parity's DeviceRows ``sample`` (:493-501) on the path basalt's dispatcher takes for ``p``
    (sampler.cu:988-1020), as the mode rule routes it: G-CL for a greedy request without a penalty window and a bias,
    G-1B for one with either, S-SPLIT for a sampled request (every sampled shape here fits it: at most 64 rows and
    64 blocks); ``forced``, G-1B in G-CL's place (F.1). ``lbuf``: the rows held in one buffer."""
    mode = kernels.basalt_sampler_mode(p.greedy, p.temperature, p.penalty_last_n, bias is not None)
    if mode == 2:
        return gpu.split(logits, p, hist=hist, hist_len=hist_len, bias=bias, lbuf=lbuf)
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
# without the bias (the others take G-1B anyway); then its sampled cases, S-SPLIT
BIAS_CASES = [(True, False, False), (True, False, True), (True, True, False), (False, False, False),
              (False, True, False)]
BIAS_IDS = ["greedy", "greedy-F", "greedy-penalty", "sampled", "sampled-penalty"]


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
    host biased (the same float32 add), and never the banned id 0. The port's host reference agrees (spec T§8.1):
    over the host-biased rows with the penalty over each row's window, the argmax (greedy_pen), or the sampled
    chain's pick (mirror_pick over mirror_select's 16). On the sampled penalised case that pick, [7, 116, 7, 7],
    shows the order: the bias added after the penalties would pick 7 in every row."""
    want, got = bias_launches(greedy, penalty, forced)[:2]
    l, bias, adjusted, hist = bias_fixture_a()
    p = bias_request(greedy, penalty)
    windows = [window_of(hist, 4, t, p.penalty_last_n) for t in range(4)]
    k = sampled_k(p.top_k, 512)
    if greedy:
        host = [greedy_pen(adjusted[t], windows[t], p) for t in range(4)]
    else:
        host = [mirror_pick(mirror_select(adjusted[t], 512, windows[t], p, k), k, p, t) for t in range(4)]
        after = [mirror_pick(select_vec(penal_vec(l[t], windows[t], p) + bias, k), k, p, t) for t in range(4)]
        assert not penalty or host != after, f"spec T§8.1: bias then penalties picks {host}, the reverse {after}"
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


BIAS_B_CASES = [(True, False), (False, False)]     # (greedy, forced): fixture B's greedy side, G-1B (a bias); sampled
BIAS_B_IDS = ["greedy", "sampled"]                 # side, S-SPLIT over 61 blocks, 60 of them without a finite logit


@pytest.mark.parametrize("greedy,forced", BIAS_B_CASES, ids=BIAS_B_IDS)
def test_basalt_sampler_bias_ban_at_vocab(greedy, forced):
    """LB.5 (:789-790): with every id but ``ONLY`` banned (100 + -inf), every setting picks ``ONLY``: the row's lowest
    logit (-50), and after the bias its only finite one."""
    large, bans, _, _ = bias_fixture_b()
    lost = sum(list(sample(large, p, bias=bans, forced=forced)) != [ONLY] for p in bias_b_requests(greedy))
    assert lost == 0, ('if (many.sample(p, nullptr) != std::vector<int>{only}) { std::printf("logit_bias: the only '
                       'permitted token lost (k=%d greedy=%d)\\n", k, (int) greedy); ++bad; }')


@pytest.mark.parametrize("greedy,forced", BIAS_B_CASES, ids=BIAS_B_IDS)
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


@pytest.mark.parametrize("path", ["greedy_1b"] + SAMPLED)
def test_basalt_sampler_penalty_presence_boolean(path):
    """M4.3 (:933-954) and A1 (spec T§10.2): fixture 4 B, token 0 at 5.0 counted twice and token 1 at 3.4, with
    present 1.5 and no freq: token 0 falls to 5.0 / 2 - 1.5 = 1.0 and token 1 wins. It would win as well under
    presence times count (-0.5) or no presence term (2.5), so M4.3 cannot see its name. A1 runs the rows with token 1
    at 0.0 and at 1.5, where the boolean rule picks (0, 1), presence times count (1, 1) and no presence term (0, 0):
    on G-1B (a penalty window), and on the sampled paths with top_k 1, whose one-entry list is the penalised argmax,
    so the pick is the rule's whatever the draw (mirror_pick). M4.3 is G-1B's."""
    _, _, lb, hist_b, p = fixture4_data()
    q = replace(p, penalty_present=1.5, penalty_freq=0.0)
    if path == "greedy_1b":
        want_b = [greedy_pen(lb[t], hist_b[4 * t:4 * t + 4], q) for t in range(2)]
        assert run(path, lb, q, want_b, hist_b, 4) == 0, \
            'bad += run("penalties: presence is boolean", lb, NT2, q, want_b, hist_b, 4);'
    rows = lb.copy()
    rows[:, 1] = 0.0, 1.5
    want = [greedy_pen(rows[t], hist_b[4 * t:4 * t + 4], q) for t in range(2)]           # the boolean rule
    counted = [int(rows[t, 1] > f32(5.0) / f32(2.0) - f32(2.0) * f32(1.5)) for t in range(2)]
    absent = [int(rows[t, 1] > f32(5.0) / f32(2.0)) for t in range(2)]
    assert (want, counted, absent) == ([0, 1], [1, 1], [0, 0]), \
        f"spec T§10.2 A1: the boolean rule picks {want}, presence times count {counted}, no presence term {absent}"
    if path in SAMPLED:
        q = replace(q, greedy=False, top_k=1)
        mirror = [mirror_pick(mirror_select(rows[t], 8, hist_b[4 * t:4 * t + 4], q, 1), 1, q, t) for t in range(2)]
        assert mirror == want, f"the sampled chain's top_k 1 picks {mirror}, the boolean rule {want}"
    assert run(path, rows, q, want, hist_b, 4) == 0, "A1: picks (0, 1) on the two presence rows"


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


@pytest.mark.parametrize("path", SAMPLED)
def test_basalt_sampler_sampled_penalties(path):
    """M6.2 (:986-1018): fixture 6's two rows (9.0, 4.5, 4.4, 4.3, then -8) through the sampled chain with top_k 5,
    top_p 0.9 and temperature 0.8 at seed 9, under repeat 3, freq 0.2 and present 0.6 over 4-slot windows that count
    token 0 once and twice: it falls to 2.2 and 2.0, and the picks are sampled_reference's [3, 2], where penalty-free
    rows pick [0, 0] (M6.1)."""
    l, hist, p = fixture6_data()
    want = [sampled_reference(l[t], hist[4 * t:4 * t + 4], p, t) for t in range(2)]
    assert want == [3, 2], f"spec T§8.7: sampled_reference picks {want}"
    assert run(path, l, p, want, hist, 4) == 0, \
        'else bad += run("sampled chain: penalties + top_k/p", l, NT2, p, want, hist, 4);'


MIN_P_PICKS = {0.0: [2, 2], 0.5: [1, 1], 0.9: [0, 0]}     # fixture 7's sampled_reference picks (spec T§8.8)


@pytest.mark.parametrize("path", SAMPLED)
@pytest.mark.parametrize("mp", MIN_P_PICKS)
def test_basalt_sampler_min_p_chain(path, mp):
    """M7.2-4 (:1021-1058): fixture 7's two rows of 8 (4.0, 3.5, 3.2, 3.1, 2.0, then -8) through the sampled chain
    with top_k 6, top_p 1 and temperature 0.9 at seed 77 (counter 0), at min_p 0, 0.5 and 0.9, which keep 6, 2 and
    1 per row (M7.1): each row's pick is sampled_reference's, at its own counter t."""
    l, base = fixture7_data()
    q = replace(base, min_p=mp)
    want = [sampled_reference(l[t], [], q, t) for t in range(2)]
    assert want == MIN_P_PICKS[mp], f"spec T§8.8: sampled_reference picks {want} at min_p {mp}"
    assert run(path, l, q, want) == 0, f'bad += run(name, l, NT3, q, want); (name "sampled chain: min_p={mp:.1f}")'


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


def fixture11():
    """Fixture 11 (:1145-1155): normal(0, 1) rows (seed 11) with an 8-slot history each, left from a penalised request
    (token 3 first, -1 after it), and the sampled request, top_k 20, top_p 0.95, temperature 0.8 at seed 5, whose
    penalty_last_n is 0. Returns (l, hist, p)."""
    hist = np.full(32, -1, np.int32)
    hist[::8] = 3
    return normal_rows(11), hist, SamplerParams(top_k=20, top_p=0.95, temperature=0.8, seed=5)


@pytest.mark.parametrize("path", SAMPLED)
def test_basalt_sampler_stale_history_sampled(path):
    """M11.1 (:1145-1161): fixture 11's sampled request over its rows with the stale history bound (hist_len 8,
    penalty_last_n 0): no window is counted, so each pick is sampled_reference's without a history."""
    l, hist, p = fixture11()
    want = [sampled_reference(row, [], p, t) for t, row in enumerate(l)]
    assert run(path, l, p, want, hist, 8) == 0, \
        'bad += run("stale history, last_n=0 (sampled)", l, NT, p, want, hist, 8);'


@pytest.mark.parametrize("path", GREEDY)
def test_basalt_sampler_stale_history_greedy(path):
    """M11.2 (:1145-1169): fixture 11's rows with an 8-slot history left from a penalised request (token 3 first in
    each row) while this greedy request has penalty_last_n = 0: the history is bound and must not change the pick,
    reference_pick's argmax."""
    l, hist, p = fixture11()
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


@pytest.mark.parametrize("path", SAMPLED)
@pytest.mark.parametrize("T,H", FX12_CASES, ids=[f"T{T}-H{H}" for T, H in FX12_CASES])
def test_basalt_sampler_per_row_history_sampled(path, T, H):
    """M12.S (:1171-1225): fixture 12's window of T rows and H-slot penalty_rows histories, sampled with top_k 20,
    top_p 0.9 and temperature 0.7 at seed 1000 + H and counter 77, under penalty_last_n = H, presence 4 and repeat
    1.1: each row's pick is sampled_reference over its own history row."""
    tail, cases = fixture12_data()
    window, l = next(c[2:] for c in cases if c[:2] == (T, H))
    rows = penalty_rows(tail, window, T, H)
    sp2 = SamplerParams(top_k=20, top_p=0.9, temperature=0.7, seed=1000 + H, counter=77, penalty_last_n=H,
                        penalty_present=4.0, penalty_repeat=1.1)
    swant = [sampled_reference(l[t], rows[t * H:(t + 1) * H], sp2, t) for t in range(T)]
    assert run(path, l, sp2, swant, rows, H) == 0, \
        f'bad += run(name, l, T, sp2, swant, rows_int, H); (name "per-row history T={T} H={H} sampled")'


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


@pytest.mark.parametrize("path", SAMPLED)
def test_basalt_sampler_history_ids_out_of_vocab_sampled(path):
    """M13.2 (:1234-1275): fixture 13's rows and junk-laden histories, sampled with top_k 20, top_p 0.95 and
    temperature 0.8 at seed 13 under presence 3 over all 16 slots: each pick is sampled_reference's over the drawn ids
    alone."""
    l, hist, valid = fixture13_data()
    sp2 = SamplerParams(greedy=False, temperature=0.8, top_k=20, top_p=0.95, seed=13, penalty_last_n=16,
                        penalty_present=3.0)
    swant = [sampled_reference(l[t], valid[16 * t:16 * (t + 1)], sp2, t) for t in range(4)]
    assert run(path, l, sp2, swant, hist, 16) == 0, \
        'bad += run("out-of-vocab history ids (sampled)", l, NT, sp2, swant, hist, H);'


TOP_K_CONTRACT = [                    # (top_k, gate): fixture 14's four runs, each against the top_k = 64 reference
    (64, 'bad += run("sampled top_k=64", l, NT, p64, want);'),
    (0, 'bad += run("sampled top_k=0 means 64", l, NT, p0, want);'),
    (100, 'bad += run("sampled top_k=100 means 64", l, NT, p100, want);'),
    (-3, 'bad += run("sampled top_k=-3 means 64", l, NT, pneg, want);'),
]


@pytest.mark.parametrize("path", SAMPLED)
@pytest.mark.parametrize("top_k,gate", TOP_K_CONTRACT, ids=["M14.1", "M14.2", "M14.3", "M14.4"])
def test_basalt_sampler_top_k_contract(path, top_k, gate):
    """M14.1-4 (:1278-1298): fixture 14's flat rows (normal(0, 1) * 0.3, 4 x 512) sampled with top_p 1, temperature
    1.5 and seed 14: top_k 64 as given, and 0, 100 and -3 meaning 64 (sampled_k), so every row's pick is the top_k = 64
    sampled_reference's, every row written. The port's observability addition (spec T§8.15), on its own seed: the
    63-wide reference and an uncapped 100-wide list (mirror_pick) give other picks on these rows, so the width 64
    decides them."""
    l = normal_rows(14, scale=0.3)
    p64 = SamplerParams(top_k=64, top_p=1.0, temperature=1.5, seed=14)
    want = [sampled_reference(l[t], [], p64, t) for t in range(4)]
    w63 = [sampled_reference(l[t], [], replace(p64, top_k=63), t) for t in range(4)]
    w100 = [mirror_pick(select_vec(l[t], 100), 100, p64, t) for t in range(4)]
    assert w63 != want and w100 != want, \
        f"spec T§8.15: the 64-wide list decides the picks ({want}; 63 wide {w63}, 100 wide {w100})"
    assert run(path, l, replace(p64, top_k=top_k), want) == 0, gate


FX16_BASE = SamplerParams(top_p=1.0, min_p=0.0, temperature=0.8, seed=16, penalty_last_n=64, penalty_repeat=2.0,
                          penalty_freq=0.25, penalty_present=0.5)


def fx16_requests():
    """Fixture 16's 67 launches (:1426-1431), rewrites of one record: launch i asks top_k i, but 100 at i = 65 and -3
    at 66 (0, 100 and -3 mean 64), top_p 0.5 at odd i, else 1 (the tail's two branches, under NaN arithmetic both),
    counter i."""
    return [replace(FX16_BASE, top_k=100 if i == 65 else -3 if i == 66 else i, top_p=0.5 if i & 1 else 1.0, counter=i)
            for i in range(67)]


def fx16_lists(l, hist, nv):
    """Each row's 64-entry list under fixture 16's penalties over its window (mirror_select)."""
    return [mirror_select(l[t], nv, window_of(hist, 64, t, 64), FX16_BASE, sampled_k(64, nv)) for t in range(4)]


FX16_CASES = [("split", nv) for nv in (248320, 100003, 262144, 1000)]       # (path, nv): 61, 25, 64 and 1 blocks


@pytest.mark.parametrize("path,nv", FX16_CASES, ids=[f"{path}-{nv}" for path, nv in FX16_CASES])
def test_basalt_sampler_top_k_list_under_ties(path, nv):
    """M16.1 (:1365-1449): fixture 16's four rows (ties across blocks, SIMD-groups and lanes, signed zeros, NaN, a row
    of 11 candidates, penalised ties on window ids at the 1,024 and 4,096 edges, ids outside the vocabulary) hold a
    +inf, so the tail's sums are NaN, no cut fires, no draw lands, and the pick is the list's last kept entry,
    sel_ids[k - 1]: the 67 launches over the rows uploaded once read each row's list position by position, against
    mirror_select's. Row 2's positions past its 11 candidates are the empty round's id 0, which the merge's sink makes
    of the part's sentinel."""
    l, hist = fixture16_data(nv)
    lists = fx16_lists(l, hist, nv)
    rows = nt.Buffer(device(), l.tobytes())
    wrong = []
    for p in fx16_requests():
        k = sampled_k(p.top_k, nv)
        got = getattr(gpu, path)(l, p, hist=hist, hist_len=64, lbuf=rows)
        wrong += [(p.top_k, t, int(got[t]), lists[t].ids[k - 1]) for t in range(4) if got[t] != lists[t].ids[k - 1]]
    assert not wrong, f"if (got[(size_t) t] != want) {{ ... ++wrong; }} ((top_k, row, got, want): {wrong[:8]})"


@pytest.mark.parametrize("nv", (248320, 100003, 262144, 262145, 1000))
def test_basalt_sampler_top_k_list_reaches_sentinel(nv):
    """M16.2 (:1453-1456), structural: row 2 of fixture 16 holds 11 candidates, so its list is the empty round's
    (id 0, -inf) from position 11 on, and 56 of the 67 launches read it there (k > 11): M16.1 gates the sentinel."""
    l, hist = fixture16_data(nv)
    ids, logit = fx16_lists(l, hist, nv)[2]
    sentinels = sum(sampled_k(p.top_k, nv) > 11 for p in fx16_requests())
    assert ids[11:] == [0] * 53 and np.all(np.float32(logit[11:]) == -np.inf) and np.isfinite(logit[1:11]).all(), \
        "fixture 16's row 2 has 11 candidates"
    assert sentinels == 56, "bad += wrong + (sentinels == 0);"


@pytest.mark.parametrize("path", SAMPLED)
@pytest.mark.parametrize("pen", (0, 1))
@pytest.mark.parametrize("nv", (248320, 512))
def test_basalt_sampler_sampled_draws_under_ties(path, nv, pen):
    """M17.1 (:1459-1526): fixture 17's 17 tie-heavy rows (half steps at 248,320, whole steps at 512) and 64-slot
    histories (half on the row's head) through the whole chain at temperature 2.5, seed 17: top_k 1, 20 and 64 x
    top_p 0.9 and 1 x min_p 0 and 0.05, each on two ``nt.Queue`` objects in turn (basalt's legacy and created
    streams), with counter 1000 times the launch's number; ``pen`` 1 counts the windows (repeat 1.25, freq 0.25,
    present 0.5), ``pen`` 0 binds them with penalty_last_n 0. Every pick is mirror_pick's over the row's 64-entry
    list. Under shader validation this is T§9's two-queue test of the 17-row scratch."""
    T, H = 17, 64
    l, hist = fixture17_data(nv)
    base = SamplerParams(temperature=2.5, seed=17, penalty_last_n=H if pen else 0, penalty_repeat=1.25,
                         penalty_freq=0.25, penalty_present=0.5)
    lists = [mirror_select(l[t], nv, window_of(hist, H, t, base.penalty_last_n), base, sampled_k(64, nv))
             for t in range(T)]
    rows = nt.Buffer(device(), l.tobytes())
    queues = nt.Queue(device()), nt.Queue(device())
    settings = itertools.product((1, 20, 64), (0.9, 1.0), (0.0, 0.05), queues)
    wrong = []
    for config, (top_k, top_p, min_p, queue) in enumerate(settings, start=24 * pen + 1):     # basalt's ++config
        p = replace(base, top_k=top_k, top_p=top_p, min_p=min_p, counter=1000 * config)
        k = sampled_k(top_k, nv)
        got = getattr(gpu, path)(l, p, hist=hist, hist_len=H, lbuf=rows, queue=queue)
        wrong += [(config, t) for t in range(T) if got[t] != mirror_pick(lists[t], k, p, t)]
    assert not wrong, f"if (got[(size_t) t] != want) {{ ... ++wrong; }} ((config, row): {wrong[:8]})"


def test_basalt_sampler_split_after_replay():
    """M18.2 (:1537-1596): fixture 18 (a)'s three rows of 248,320 half steps and 64-slot windows, the request top_k
    20, top_p 0.9, temperature 2.5, seed 18, counter 77 under repeat 1.25, freq 0.25 and present 0.5, on the split
    path (basalt's stream after its capture): every pick is mirror_pick's. The recorded one-block launch it follows is
    the program tier's M18.1."""
    T, nv, H = 3, 248320, 64
    l, hist = fixture18_data(nv, T, 1800)
    p = SamplerParams(top_k=20, top_p=0.9, temperature=2.5, seed=18, counter=77, penalty_last_n=H, penalty_repeat=1.25,
                      penalty_freq=0.25, penalty_present=0.5)
    k = sampled_k(p.top_k, nv)
    want = [mirror_pick(mirror_select(l[t], nv, window_of(hist, H, t, H), p, k), k, p, t) for t in range(T)]
    got = gpu.split(l, p, hist=hist, hist_len=H)
    wrong = int(np.count_nonzero(got != np.int32(want)))
    assert wrong == 0, 'compare("same stream, uncaptured", rows.sample(p, cs), lists, k, p);'


# ---- the port's additions on the sampled paths (spec T§10.2) --------------------------------------------------------

@pytest.mark.parametrize("path", SAMPLED)
def test_basalt_sampler_sampled_penalty_window_tail(path):
    """A2: fixture 8's row and 8-slot history copied to 16 rows, sampled (top_k 8, top_p 1, temperature 1, seed 8,
    counter 0) under fixture 8's penalties: penalty_last_n 4 counts the history's last four slots, so token 3 falls
    and every pick is sampled_reference's 0, where the whole window would pick 3 on 9 of the 16 rows."""
    l8, h8, p8 = fixture8_data()
    l, hist = np.tile(l8, (16, 1)), np.tile(h8, 16)
    p = replace(p8, greedy=False, top_k=8, seed=8, counter=0)
    want = [sampled_reference(l[t], window_of(hist, 8, t, 4), p, t) for t in range(16)]
    full = [sampled_reference(l[t], hist[8 * t:8 * t + 8], p, t) for t in range(16)]
    assert want == [0] * 16 and full.count(3) == 9, f"spec T§10.2 A2: the tail picks {want}, the whole window {full}"
    assert run(path, l, p, want, hist, 8) == 0, \
        "A2: the 16 tail-window picks equal sampled_reference (all 0); the full window would differ"


@pytest.mark.parametrize("path", SAMPLED)
def test_basalt_sampler_temperature_after_filters(path):
    """A3: fixture 2's row (3.0, then 2.55 down to 2.25 at ids 1-7, -1000 elsewhere) on 256 rows, sampled with its
    top_k 0, top_p 0.5 and temperature 0.5 at seed 2, counter 0, as four windows of 64 rows at positions 0, 64, 128 and
    192 (row t's counter is t; the split takes 64 rows): top_p cuts the list at 4 before the temperature sharpens it,
    so tokens 0-3 are drawn 122, 48, 54 and 32 times, where the temperature first would cut at 3 and never draw token
    3 (138, 74, 44, 0). Every pick is sampled_reference's."""
    l2, p2 = fixture2_data()
    l = np.tile(l2[0], (256, 1))
    p = replace(p2, greedy=False, seed=2)
    want = [sampled_reference(l[t], [], p, t) for t in range(256)]
    first = [sampled_reference(l[t] * (f32(1) / f32(p.temperature)), [], replace(p, temperature=1.0), t)
             for t in range(256)]
    counts = [[w.count(i) for i in range(4)] for w in (want, first)]
    assert counts == [[122, 48, 54, 32], [138, 74, 44, 0]], f"spec T§10.2 A3: token counts {counts}"
    got = np.concatenate([getattr(gpu, path)(l[w:w + 64], p, position=w) for w in range(0, 256, 64)])
    wrong = int(np.count_nonzero(got != np.int32(want)))
    assert wrong == 0, "A3: token 3 drawn; picks equal sampled_reference (counts 122/48/54/32 at seed 2)"


@pytest.mark.parametrize("path", SAMPLED)
@pytest.mark.parametrize("min_keep", (3, 10))
def test_basalt_sampler_min_keep(path, min_keep):
    """A4: fixture 7's rows with top_p 0.1, whose cut keeps 1 (4.0's probability alone passes it), at min_keep 3
    (keeps 3) and 10 (more than k = 6: keeps 6), the kept entries those mirror_prob gives a probability: each pick
    is the mirror's, and the clamp decides them (min_keep 1 picks 0 in both rows)."""
    l, base = fixture7_data()
    p = replace(base, top_p=0.1, min_keep=min_keep)
    sels = [mirror_select(l[t], 8, [], p, 6) for t in range(2)]
    kept = [sum(mirror_prob(sel, 6, p, q) > 0 for q in sel.ids) for sel in sels]
    want = [mirror_pick(sels[t], 6, p, t) for t in range(2)]
    one = [mirror_pick(sels[t], 6, replace(p, min_keep=1), t) for t in range(2)]
    assert kept == [min(min_keep, 6)] * 2 and one == [0, 0] and want != one, \
        f"spec T§10.2 A4: min_keep {min_keep} keeps {kept} and picks {want}; min_keep 1 picks {one}"
    assert run(path, l, p, want) == 0, "A4: min_keep 3 and 10 keep 3 and 6; picks equal the mirror"


A7_BITS = (0x00000001, 0x00000002, 0x007FFFFF, 0x80000001, 0x80000002, 0x807FFFFF, 0x00000000, 0x80000000)


def a7_rows(nv, T=64):
    """A7's 64 rows: -inf, but 8 slots holding a per-row permutation of the distinct subnormals 2^-149, 2^-148 and
    0x007fffff of both signs and the two zeros. The slots are ids 1-8 at nv = 512 (id 0 is left to the sentinel) and
    1 + 7 * 4,096 j at 248,320, in 8 of the 61 blocks, so the merge orders them across blocks."""
    rs = np.random.RandomState([20261021, nv])
    slots = 1 + np.arange(8) * (1 if nv <= kernels.BS_BLOCK else 7 * kernels.BS_BLOCK)
    l = np.full((T, nv), -np.inf, np.float32)
    for t in range(T):
        l[t, slots] = np.uint32(A7_BITS).view(np.float32)[rs.permutation(8)]
    return l


@pytest.mark.parametrize("path", SAMPLED)
@pytest.mark.parametrize("top_p", (1.0, 0.5))
@pytest.mark.parametrize("nv", (512, 248320))
def test_basalt_sampler_subnormal_order(path, nv, top_p):
    """A7: a7_rows' subnormal-led rows, no penalties and no bias, top_k 0 (64), temperature 1, seed 21: the list is
    the 8 values in their order (+0 and -0 tie, the lower id first), then sentinels; every exp of a difference is 1.0,
    so each kept entry is drawn with the same probability, at top_p 1 all 8 and at 0.5 the first 4 (the lower-id
    zero last). Every pick is the mirror's (numpy's compares order subnormals exactly); the draws reach every kept
    position, and a compare that read the subnormals as zeros, as this GPU's do, would pick otherwise."""
    l = a7_rows(nv)
    p = SamplerParams(top_k=0, top_p=top_p, temperature=1.0, seed=21)
    k = sampled_k(p.top_k, nv)
    sels = [mirror_select(row, nv, [], p, k) for row in l]
    want = [mirror_pick(sel, k, p, t) for t, sel in enumerate(sels)]
    flushed = np.where(l == -np.inf, l, np.copysign(np.float32(0), l))
    flush = [mirror_pick(mirror_select(row, nv, [], p, k), k, p, t) for t, row in enumerate(flushed)]
    reached = {sel.ids.index(w) for sel, w in zip(sels, want)}
    assert reached == set(range(8 if top_p == 1.0 else 4)) and flush != want, \
        f"the draws reach list positions {sorted(reached)}; a flushing compare picks the same in " \
        f"{sum(a == b for a, b in zip(flush, want))} of {len(want)} rows"
    assert run(path, l, p, want) == 0, "A7: subnormal-led rows equal the mirror"


@pytest.mark.parametrize("path", SAMPLED)
def test_basalt_sampler_empty_keep(path):
    """A8: fixture 14's rows and request (top_k 64, top_p 1, temperature 1.5, seed 14) at min_p 2, which basalt's
    callers never pass and its record takes: the threshold sel[0] + logf(2) lies above the list's head, the min_p cut
    keeps nothing, and the tail picks sel_ids[0], each row's argmax (mirror_pick), where min_p 0 draws other tokens."""
    l = normal_rows(14, scale=0.3)
    p = SamplerParams(top_k=64, top_p=1.0, temperature=1.5, seed=14, min_p=2.0)
    k = sampled_k(p.top_k, 512)
    sels = [mirror_select(row, 512, [], p, k) for row in l]
    want = [mirror_pick(sel, k, p, t) for t, sel in enumerate(sels)]
    drawn = [mirror_pick(sel, k, replace(p, min_p=0.0), t) for t, sel in enumerate(sels)]
    assert want == [int(np.argmax(row)) for row in l] and drawn != want, \
        f"spec T§10.2 A8: min_p 2 picks {want}, min_p 0 {drawn}"
    assert run(path, l, p, want) == 0, "A8: `min_p = 2.0`: pick `sel_ids[0]`"


# ---- TP.1: the drafter's greedy tail (spec S§19.3) ------------------------------------------------------------------

def softmax_top(row, pick):
    """The float64 softmax probability of ``pick``, the row's maximum: 1 / sum_i exp(l[i] - l[pick])."""
    r = np.asarray(row, np.float64)
    return 1.0 / float(np.sum(np.exp(r - r[pick])))


@pytest.mark.parametrize("nv", VOCABS)
def test_basalt_sampler_greedy_top_prob_vs_host(nv):
    """TP.1: sample_greedy_top_prob (sampler.cu:930-972), which basalt runs only as q5k_head_parity's reference
    (q5k_head_parity.cpp:111, :130), against the host on a window of one row per decode_cluster_parity argmax pattern
    (ties, -0 against +0, subnormals, NaN, +-inf, empty rows) at each of its vocabularies, without the map and with
    it (map[i] = 3 i + 7: never i, and never -1, which the prefill reads). Each id is the host argmax (the lowest
    index of the largest non-NaN value above -inf, else 0), mapped, so an empty row gives map[0]. On a row of finite
    logits the probability is within 1e-5 relative of the float64 softmax of the pick (q5k_head_parity's bound, a
    non-finite probability counting as a failure); on a row with a NaN or an infinity only the id is checked. The
    map is applied last, so the probabilities' bits are the same with it and without."""
    rs = np.random.RandomState([20261017, nv])
    l = np.stack([fill_logits(rs, nv, pattern) for pattern in PATTERNS])
    id_map = np.arange(nv, dtype=np.int32) * 3 + 7
    p = SamplerParams(greedy=True, temperature=0.0)
    ids, bits = gpu.top_prob(l, p)
    mapped, mapped_bits = gpu.top_prob(l, p, id_map=id_map)
    hb = np.int32([host_argmax(row) for row in l])
    id_bad = int(np.count_nonzero(ids != hb)) + int(np.count_nonzero(mapped != id_map[hb]))
    prob = bits.view(np.float32)
    finite = [t for t in range(len(l)) if np.isfinite(l[t]).all()]
    rel = {PATTERNS[t]: abs(float(prob[t]) - softmax_top(l[t], hb[t])) / max(1e-30, softmax_top(l[t], hb[t]))
           for t in finite}
    nonfinite = [PATTERNS[t] for t in finite if not np.isfinite(prob[t])]
    over = {k: r for k, r in rel.items() if not r <= 1e-5}
    max_rel = max((r for r in rel.values() if r == r), default=float("nan"))
    assert id_bad == 0, "if (logit_bad || id_bad || max_rel > 1e-5) fail = 1; (id_bad: the host argmax, mapped)"
    assert not nonfinite, f"non-finite probabilities on finite rows count as failures: {nonfinite}"
    assert not over, f"if (logit_bad || id_bad || max_rel > 1e-5) fail = 1; ({over}, max_rel {max_rel:.2e})"
    assert np.array_equal(bits, mapped_bits), "the map is applied last: the probabilities do not depend on it"


def test_basalt_sampler_greedy_top_prob_geometry():
    """The probability's sums are basalt's only at 1,024 threads (spec S§7, S§16.3 (c)): the pipeline must run them
    (``gpu.top_prob`` refuses one that cannot), and a dispatch of 512 threads per row returns at once, leaving every
    pick and probability its prefill instead of a probability over half the row's sums."""
    part, prob = gpu.pipeline("bs_greedy_partial"), gpu.pipeline("bs_greedy_top_prob")
    assert prob.max_threads_per_threadgroup >= gpu.TOP_PROB_THREADS
    l = normal_rows(7)
    rows = len(l) + 1
    ids, bits = gpu.launch([(part, (gpu.AM_CTAS, rows), gpu.threads(part)), (prob, (rows,), 512)], l,
                           SamplerParams(greedy=True, temperature=0.0), mode=0, scratch={8: rows * gpu.AM_CTAS * 8})
    assert np.all(ids == -1) and np.all(bits == 0xFFFFFFFF), "a 512-thread dispatch wrote a pick or a probability"


# ---- PL.1: the split path's block lists (the port's building-block check) -------------------------------------------

def block_lists(l, hist, hist_len, p):
    """The lists bs_split_part stores for the rows ``l`` (biased already): mirror_select's two stages as the kernel
    splits them, the penalty over the row's whole window (penal_vec), then the selection over each 4,096-logit block's
    slice (select_vec), the block's first id added; padded with the part's sentinel (V, -inf), which the merge's sink
    turns into mirror_select's (0, -inf). (T, n_blocks, k, 2) uint32: (id, value bits)."""
    T, V = l.shape
    n = kernels.BS_BLOCK
    nb, k = -(-V // n), sampled_k(p.top_k, V)
    want = np.empty((T, nb, k, 2), np.uint32)
    for t in range(T):
        s = penal_vec(l[t], window_of(hist, hist_len, t, p.penalty_last_n), p)
        for b in range(nb):
            ids, lg = select_vec(s[b * n:(b + 1) * n], k)
            lg = np.float32(lg)
            want[t, b, :, 0] = np.where(lg == -np.inf, V, np.int64(ids) + b * n)
            want[t, b, :, 1] = lg.view(np.uint32)
    return want


def pl1_fixture16(nv, top_k):
    """Fixture 16's rows and 64-slot histories at nv (spec T§8.18: ties across lanes, SIMD-groups and blocks, signed
    zeros, NaN, +inf, a row of 11 candidates whose other blocks are empty, window ids on the 1,024 and 4,096 edges,
    repeated and outside the vocabulary), its base request at ``top_k``, and a fifth row without a window,
    decode_cluster's subnormal pattern. No bias. Returns (l, bias, hist, hist_len, p)."""
    l, hist = fixture16_data(nv)
    l = np.vstack([l, fill_logits(np.random.RandomState([20261018, nv]), nv, "denormal")])
    p = SamplerParams(top_k=top_k, top_p=1.0, min_p=0.0, temperature=0.8, seed=16, penalty_last_n=64,
                      penalty_repeat=2.0, penalty_freq=0.25, penalty_present=0.5)
    return l, None, np.concatenate([hist, np.full(64, -1, np.int32)]), 64, p


def pl1_patterns(nv, top_k):
    """decode_cluster's 12 argmax patterns at nv, a row each, with a sparse bias (-inf at five drawn ids, 8 at five,
    the subnormal 1e-42 at five, 0 elsewhere) and 16 history slots per row whose last 12 are its window
    (penalty_last_n 12; repeat 1.5, freq 0.25, present 0.5): the row's six largest biased ids, the first two counted
    twice and the last three before the window; nv, nv + 77, -5, -1 and 0x7fffffff, outside the vocabulary; and two
    drawn ids. Returns (l, bias, hist, hist_len, p)."""
    rs = np.random.RandomState([20261019, nv])
    l = np.stack([fill_logits(rs, nv, pattern) for pattern in PATTERNS])
    bias = np.zeros(nv, np.float32)
    for x in (-np.inf, 8.0, 1e-42):
        bias[rs.randint(nv, size=5)] = x
    hist = []
    with np.errstate(invalid="ignore"):
        biased = l + bias
    for row in biased:
        top = np.argsort(-np.where(np.isnan(row), -np.inf, row), kind="stable")[np.arange(6) % nv]
        window = np.concatenate([top[[0, 0, 1, 1, 2]], [nv, nv + 77, -5, -1, 0x7FFFFFFF], rs.randint(nv, size=2)])
        rs.shuffle(window)
        hist.append(np.concatenate([top[3:], [-1], window]))
    p = SamplerParams(top_k=top_k, penalty_last_n=12, penalty_repeat=1.5, penalty_freq=0.25, penalty_present=0.5)
    return l, bias, np.concatenate(hist).astype(np.int32), 16, p


PL1_CASES = ([("fixture16", nv, top_k) for nv in (248320, 100003, 262144, 1000) for top_k in (0, 1, 11)]
             + [("patterns", nv, 0) for nv in VOCABS])


@pytest.mark.parametrize("fixture,nv,top_k", PL1_CASES, ids=[f"{f}-{nv}-top_k{k}" for f, nv, k in PL1_CASES])
def test_basalt_sampler_split_part_lists(fixture, nv, top_k):
    """PL.1: every block's list in cand, read back after bs_split_part alone (gpu.split_part), equals mirror_select's
    over the block (block_lists), ids and value bits, its sentinels included: on fixture 16 at 61, 25, 64 and 1
    blocks with k 64, 1 and 11 (its sparse row's candidates), and on decode_cluster's patterns at each of its
    vocabularies with a bias and a window (k = min(64, nv); partial last blocks, empty SIMD-group slices, lanes past
    the vocabulary). Fixture 16's sparse row leaves empty blocks, all sentinels, wherever it has more than one block
    or k is above 11; every fixture's window changes some block's first 64 pairs."""
    l, bias, hist, hist_len, p = (pl1_fixture16 if fixture == "fixture16" else pl1_patterns)(nv, top_k)
    got = gpu.split_part(l, p, hist=hist, hist_len=hist_len, bias=bias)
    with np.errstate(invalid="ignore"):
        rows = l if bias is None else l + bias
    want = block_lists(rows, hist, hist_len, p)
    top64 = replace(p, top_k=0)
    assert not np.array_equal(block_lists(rows, hist, hist_len, top64), block_lists(rows, None, 0, top64)), \
        "the fixture's window changes some block's first 64"
    bad = [(t, b) for t in range(len(l)) for b in range(want.shape[1]) if not np.array_equal(got[t, b], want[t, b])]
    assert not bad, f"each block's cand list equals mirror_select (T§3.6): (row, block) {bad[:8]} of {len(bad)} differ"


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


def w1_finite_rows():
    """5 rows at 248,320 of decode_cluster_parity's random, ties and signed-zero patterns, all finite, so the
    top-prob path's probabilities are sums of 248,320 terms, compared bit for bit."""
    rs = np.random.RandomState(20261018)
    l = np.stack([fill_logits(rs, 248320, pattern) for pattern in ("random", "ties", "signed-zero", "random", "ties")])
    return l, SamplerParams(greedy=True, temperature=0.0), None, 0


def w1_sampled():
    """8 rows of normal(0, 3) at 248,320 logits (61 blocks), each with its own 16-slot window (its three largest ids,
    the first twice, among drawn ones), sampled with top_k 40, top_p 0.6, min_p 0.25 and temperature 0.8 under
    repeat 1.3, freq 0.1 and present 0.2, at a seed and a counter with high words. top_p keeps 11 to 18 of each row's
    40 and min_p then fewer in four rows (3 of row 1's 11), and each stage decides picks: without top_p 4 rows would
    pick otherwise, without min_p 2, without the penalties all 8. So every step of the tail runs, after the merge of
    61 lists, with each row's own window and its draw at counter + t."""
    rs = np.random.RandomState(20261020)
    V = 248320
    l = rs.standard_normal((8, V)).astype(np.float32) * np.float32(3)
    hist = rs.randint(V, size=(8, 16)).astype(np.int32)
    hist[:, :4] = np.argsort(-l, axis=1, kind="stable")[:, [0, 0, 1, 2]]
    p = SamplerParams(top_k=40, top_p=0.6, min_p=0.25, temperature=0.8, penalty_last_n=16, penalty_repeat=1.3,
                      penalty_freq=0.1, penalty_present=0.2, seed=(3 << 32) + 77, counter=(1 << 32) + 5)
    return l, p, hist.ravel(), 16


ROWS_ALONE = {"greedy_cl": w1_ties_across, "greedy_1b": w1_per_row_history,     # W.1's fixture for each path
              "top_prob": w1_finite_rows, "split": w1_sampled}


@pytest.mark.parametrize("path", ROWS_ALONE)
def test_basalt_sampler_rows_alone(path):
    """W.1: the path's multi-row window, row for row, equals each row run alone as a one-row window at its own
    position (gpu.rows_alone, bitwise). A sampled path's window must also be the mirror's, mirror_pick over
    mirror_select's list with the row's window at counter + t, so that the check cannot pass on a tail that is wrong
    in every row alike."""
    l, p, hist, hist_len = ROWS_ALONE[path]()
    assert gpu.rows_alone(path, l, p, hist=hist, hist_len=hist_len) == [], \
        "invariant 1: a row's output never depends on the window width T or its position in it"
    if path in SAMPLED:
        T, V = l.shape
        k = sampled_k(p.top_k, V)

        def mirror(q, windowed=True):
            return [mirror_pick(mirror_select(l[t], V, window_of(hist, hist_len, t, q.penalty_last_n) if windowed
                                              else [], q, k), k, q, t) for t in range(T)]

        want = mirror(p)
        assert want != mirror(replace(p, top_p=1.0)) and want != mirror(replace(p, min_p=0.0)) \
            and want != mirror(p, windowed=False), "the fixture's top_p, min_p and penalties each decide a pick"
        got = list(getattr(gpu, path)(l, p, hist=hist, hist_len=hist_len))
        assert got == want, f"the window's picks {got} are mirror_pick's {want} (T§3.6)"
