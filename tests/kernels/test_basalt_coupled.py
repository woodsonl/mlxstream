# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its fixtures, pre-fills and pass criteria.
"""sampler_parity's coupled_parity (basalt src/kernels/sampler_parity.cpp:573-653) and the port's additions for the
coupled draft (spec T§7, T§10.2 A8 and A10) on the Metal sampler family, through tests/kernels/basalt_sampler_gpu.py's
``coupled_stage`` and ``coupled_draft``, over coupled_parity's 420 cases (``coupled_cases``: seven draft vocabularies,
subset index i the token id i + 25 of nv + 50 ids, a ring of 256 history slots and 8 drafts, a draft index j and a
step record per case; the request's image uploaded to the round's copy, as basalt uploads it, with no stage). The
host's draft (``mirror``) is mirror_select's list over the case's row penalised over its window, mirror_pick's draw
from it with the verify row's counter cell + 1, mapped through the subset, and kernel_qprob's probability of the pick.

- C.1: the draft's token id (bs_coupled_merge's) is the verify sampler's pick over the row the draft penalised, run as
  coupled_parity runs it (S-SPLIT, one row, the request without penalties drawing with counter cell + 1), mapped
  through the subset, and the ring's slot cap + j holds it. Its part-list half: bs_coupled_part's block lists (16
  SIMD-groups of 8 logits per lane) over that row equal bs_split_part's (4 of 32 per lane) and the mirror's
  (block_lists, mirror_select's per block), ids and value bits; only the first k of each block's 64 entries are
  written.
- C.2: the row read back after the draft's kernels is the host's penal over draft j's window ring[cap + j - h,
  cap + j), counted over token ids and mapped through the subset's inverse map, bit for bit, at basalt's 1,024
  threads and at 32; a case without penalties leaves it untouched, and the ring keeps every slot but the draft's own.
- C.3: the draft's id and out_prob bits are the host's, over every case and at step records of -1, -2 and 2^31 - 1,
  whose counters need the cell sign-extended before the add.
- C.4: greedy, temperature 0 and -0 take the merge's greedy branch: the list's head (the sentinel's 0 for an empty
  row), mapped, with probability 1.0; a NaN temperature and a positive subnormal one sample (setp.gtu on the bits).
- C.5: a 100-slot window holding -1, ids outside [0, id_vocab) and ids without a subset slot: the penalize kernel
  skips them and counts the rest, at 1,024 and 32 threads, and the draft is the host's.
- C.6: bs_coupled_stage copies the staged request's 72 bytes, its padding and pointer slot included, and the staged
  history's last h = coupled_hist_len(penalty_last_n, cap) slots (coupled_hist_base's, before cap) into the ring,
  nothing else, at basalt's 256 threads and at 32, at coupled_parity's cap and the engine's 4,096.
- A8 (``test_basalt_sampler_empty_keep``; its sampled and probed paths are test_basalt_sampler.py's): min_p 2 keeps
  nothing, and the draft is the list's head with probability 1.0.
"""

import functools
import struct
from dataclasses import replace

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_sampler_ref import (COUPLED_HIST_CAP, COUPLED_VOCABS, coupled_cases, coupled_draft_counter,
                                      coupled_hist_base, coupled_hist_len, coupled_hist_start, mirror_pick,
                                      mirror_select, penal_vec, raw32, sampled_k)
from tests.kernels import basalt_sampler_gpu as gpu
from tests.kernels.test_basalt_sampler import block_lists, kernel_qprob

CAP = gpu.CPL_CAP
ONE = 0x3F800000                       # 1.0f: out_prob of the greedy branch and of a tail that keeps nothing


@functools.lru_cache(maxsize=None)
def cases():
    """coupled_parity's 420 cases, drawn once, by vocabulary."""
    by = {}
    for c in coupled_cases():
        by.setdefault(c.nv, []).append(c)
    return by


def maps(nv):
    """coupled_parity's subset (:587-591): sub[i] = i + 25 for the nv slots, and its inverse over id_vocab = nv + 50
    ids, -1 off the subset."""
    sub = np.arange(nv, dtype=np.int32) + 25
    inv = np.full(nv + 50, -1, np.int32)
    inv[sub] = np.arange(nv, dtype=np.int32)
    return sub, inv


def draft(case, threads=1024):
    """coupled_parity's draft call (:633-635) on ``case``, with the subset's maps."""
    sub, inv = maps(case.nv)
    return gpu.coupled_draft(case.logits, case.p, j=case.j, ring=case.ring, cell=case.cell, sub_to_id=sub,
                             id_to_sub=inv, threads=threads)


def window(case):
    """Draft j's counted window as subset slots: the ring's [cap + j - h, cap + j), h = coupled_hist_len(penalty_last_n,
    cap), its token ids inside [0, id_vocab) through the inverse map (-1 off the subset, which penal_vec drops)."""
    _, inv = maps(case.nv)
    h = coupled_hist_len(case.p.penalty_last_n, CAP)
    w = case.ring[coupled_hist_start(CAP, case.j, h):CAP + case.j]
    return inv[w[(w >= 0) & (w < len(inv))]]


def listed(case):
    """mirror_select's list for the case: its logits penalised over ``window``, k = sampled_k(top_k, nv) entries."""
    return mirror_select(case.logits, case.nv, window(case), case.p, sampled_k(case.p.top_k, case.nv))


def mirror(case):
    """The case's draft on the host: mirror_pick's draw from its list (``listed``) at row 0 with the verify row's
    counter, coupled_draft_counter(cell) (cell + 1, not the image's counter), mapped through the subset (+ 25), and the
    pick's probability, kernel_qprob's bits: (float) ex[pi] in the kernel's order (the pick's id is in the kept list
    once: the fixture's lists hold no sentinel), 1.0 when nothing is kept. Returns (id, probability bits, the list)."""
    p, sel = replace(case.p, counter=coupled_draft_counter(case.cell)), listed(case)
    pick = mirror_pick(sel, len(sel.ids), p, 0)
    return pick + 25, kernel_qprob(sel, len(sel.ids), p, pick), sel


def appended(case, token):
    """The ring after draft j: the case's, with ``token`` in draft j's slot cap + j."""
    ring = case.ring.copy()
    ring[CAP + case.j] = token
    return ring


@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_matches_verify_sampler(nv):
    """C.1 (:644-653): for each case, the draft's token id is the verify sampler's pick over the row the draft
    penalised (read back), run as coupled_parity runs it (``q = p; q.counter = (uint64_t) cell + 1;
    q.penalty_last_n = 0``, :636-639: S-SPLIT, one row), mapped through the subset (vid + 25), and the ring's slot
    cap + j holds it. The image's counter is 0, so a merge that drew with it rather than the step record's fails.
    Its part-list half: bs_coupled_part's list of every block over that row equals bs_split_part's with the verify
    request, and mirror_select's over the block (block_lists), ids and value bits, the sentinels (nv, -inf) included;
    the entries past k keep their prefill (the stride is 64, k is the request's)."""
    fail, bad = [], []
    for case in cases()[nv]:
        r = draft(case)
        q = replace(case.p, counter=coupled_draft_counter(case.cell), penalty_last_n=0)
        k, row = sampled_k(q.top_k, nv), r.logits.view(np.float32)[None]
        vid = int(gpu.split(row, q)[0])
        if not (r.id == vid + 25 and r.ring[CAP + case.j] == r.id):
            fail.append(case.c)
        split = gpu.split_part(row, q)[0]
        if not (np.array_equal(r.cand[:, :k], split) and np.array_equal(split, block_lists(row, None, 0, q)[0])
                and np.all(r.cand[:, k:] == 0xFFFFFFFF)):
            bad.append(case.c)
    assert not fail, f"const bool ok = cid == vid + 25 && r2[(size_t) (cap + j)] == cid;  (cases {fail} of 60 fail)"
    assert not bad, f"bs_split_part and bs_coupled_part give equal lists (the mirror's): cases {bad} of 60 differ"


@pytest.mark.parametrize("threads", [1024, 32])
@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_penalties_vs_host(nv, threads):
    """C.2: for each case, the row read back after the draft's kernels at ``threads`` per threadgroup equals penal_vec
    of the case's logits over its window (``window``), every one of the nv logits bit for bit, and the ring is
    unchanged but for the merge's append of the draft at cap + j (C.1's). The fixture's penalties (a third of the
    cases: last_n 64, repeat 1.1, freq 0.05, present 0.1) change some logits, so the check sees a kernel that applies
    none."""
    bad, changed = [], 0
    for case in cases()[nv]:
        r = draft(case, threads)
        want = penal_vec(case.logits, window(case), case.p).view(np.uint32)
        changed += int(np.count_nonzero(want != case.logits.view(np.uint32)))
        if not (np.array_equal(r.logits, want) and np.array_equal(r.ring, appended(case, r.id))):
            bad.append(case.c)
    assert changed, "the fixture's penalties change some logit"
    assert not bad, (f"penalised logits read back equal host penal over the ring window through inv (bitwise): "
                     f"cases {bad} of 60 differ")


CELLS = (-1, -2, 2**31 - 1)            # step records whose counter needs the cell sign-extended before the add


def wrong_counters(cell):
    """The counters a merge that zero-extended the cell, or added in 32 bits, would draw with, where they are not
    coupled_draft_counter(cell)'s."""
    return {cell % 2**32 + 1, (cell + 1) % 2**32, ((cell + 1 + 2**31) % 2**32 - 2**31) % 2**64} - {
        coupled_draft_counter(cell)}


@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_pick_and_prob_vs_mirror(nv):
    """C.3: for each case, and for its first 12 with each of CELLS as the step record (-1: draft_first at position 0,
    counter 0), the draft's token id and out_prob bits are ``mirror``'s, and the ring's slot cap + j holds the id. At
    each of CELLS, each of its wrong counters changes the mirror's pick on some case."""
    bad, seen = [], set()
    for case in cases()[nv] + [case._replace(cell=cell) for case in cases()[nv][:12] for cell in CELLS]:
        r = draft(case)
        want_id, want_prob, sel = mirror(case)
        if (r.id, r.prob, r.ring[CAP + case.j]) != (want_id, want_prob, want_id):
            bad.append((case.c, case.cell))
        if case.cell in CELLS:
            k = sampled_k(case.p.top_k, nv)
            seen |= {(case.cell, w) for w in wrong_counters(case.cell)
                     if mirror_pick(sel, k, replace(case.p, counter=w), 0) + 25 != want_id}
    assert seen == {(c, w) for c in CELLS for w in wrong_counters(c)}, f"a wrong counter changes no pick: {seen}"
    assert not bad, ("id and `out_prob` equal `mirror_select`/`mirror_pick` on the penalised logits (`(float) ex[pi]`): "
                     f"(case, cell) {bad} differ")


SUBNORMAL = float(np.uint32(1).view(np.float32))   # the least positive float32, 2^-149: above 0 on its bits


@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_greedy_branch(nv):
    """C.4 (sampler.cu:805-806): over every case, greedy at temperature 0.6, and temperature 0 or -0 with greedy
    false, take the greedy branch: the id is the list's head (mirror_select's sel_ids[0]) mapped, the probability
    exactly 1.0 and the ring's slot cap + j the id, where the case's own request draws another token on some case.
    So does a row of -inf, whose list is empty: its head is the sentinel's id stored as 0, mapped to 25. The branch is
    setp.gtu on the temperature's bits, so on the first 12 cases a NaN temperature samples (inv_t 0: a uniform draw
    over the kept list, the mirror's id and probability, not 1.0 on some case), and so does a positive subnormal one,
    which this GPU's float compare reads as 0 (inv_t +inf: every exponential NaN, the draw falling to the last kept
    entry; the mirror's id and a NaN probability, payloads aside)."""
    head, sampled, drawn, nan_draws = [], [], 0, 0
    rows = cases()[nv]
    empty = rows[0]._replace(logits=np.full(nv, -np.inf, np.float32))
    for case in rows + [empty]:
        want = listed(case).ids[0] + 25
        for kw in ({"greedy": True}, {"temperature": 0.0}, {"temperature": -0.0}):
            r = draft(case._replace(p=replace(case.p, **kw)))
            if (r.id, r.prob, r.ring[CAP + case.j]) != (want, ONE, want):
                head.append((case.c, kw))
        drawn += case is not empty and mirror(case)[0] != want
    with np.errstate(over="ignore", invalid="ignore"):           # 1 / 2^-149 overflows; 0 * inf is NaN
        for case in rows[:12]:
            for t in (float("nan"), SUBNORMAL):
                s = case._replace(p=replace(case.p, temperature=t))
                r, (want_id, want_prob, _) = draft(s), mirror(s)
                if t != t:                                       # NaN: inv_t 0, a finite probability
                    ok = (r.id, r.prob) == (want_id, want_prob)
                    nan_draws += want_prob != ONE
                else:                                            # 2^-149: inv_t +inf, a NaN probability
                    ok = r.id == want_id and np.isnan(np.uint32([r.prob, want_prob]).view(np.float32)).all()
                if not (ok and r.ring[CAP + case.j] == r.id):
                    sampled.append((case.c, t))
    assert drawn and nan_draws, f"the greedy branch is observable: {drawn} sampled, {nan_draws} NaN-temperature draws"
    assert not head, f"`greedy = true` and `temperature = 0`: id `sel_ids[0]` mapped, prob 1.0: {head} differ"
    assert not sampled, f"a NaN or positive subnormal temperature samples (setp.gtu): (case, temperature) {sampled}"


def skip_case(case):
    """``case`` with penalties over a 100-slot window, last_n 100 (it reaches the ring's -1 slots below 176), whose
    every third filled slot holds an id the penalize kernel must skip: a negative one, one at or past id_vocab, or a
    token id without a subset slot (inv -1: ids 0, 3 and 24, slots of the row were they taken as their own, and
    nv + 25 and nv + 49). At 32 threads a thread's entries are 32 apart, so one that meets such an id has a counted
    one 32 later (3 does not divide 32)."""
    nv = case.nv
    junk = np.int32([-2, 0, nv + 50, 3, -2**31, nv + 25, 2**31 - 1, 24, nv + 49])
    ring = case.ring.copy()
    slots = np.arange(176, CAP + case.j, 3)
    ring[slots] = junk[np.arange(len(slots)) % len(junk)]
    return case._replace(ring=ring, p=replace(case.p, penalty_last_n=100, penalty_repeat=1.1, penalty_freq=0.05,
                                             penalty_present=0.1))


@pytest.mark.parametrize("threads", [1024, 32])
@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_window_skips(nv, threads):
    """C.5 (sampler.cu:719-722): over every case made a ``skip_case``, at ``threads`` per threadgroup (basalt's 1,024,
    and 32, where a thread meets a skipped id before a counted one), the row read back is penal_vec over the window's
    countable ids alone (``window``) bit for bit, the draft is ``mirror``'s (id and out_prob) and the ring is
    unchanged but for its slot cap + j. The skipped ids matter: taken as their own slots, the ids without a subset
    slot would penalise other logits in every case."""
    bad, visible = [], 0
    _, inv = maps(nv)
    for case in map(skip_case, cases()[nv]):
        r = draft(case, threads)
        want = penal_vec(case.logits, window(case), case.p).view(np.uint32)
        w = case.ring[coupled_hist_start(CAP, case.j, 100):CAP + case.j]
        w = w[(w >= 0) & (w < len(inv))]
        own = np.where(inv[w] >= 0, inv[w], w)                  # an id without a slot taken as its own slot
        visible += not np.array_equal(penal_vec(case.logits, own, case.p).view(np.uint32), want)
        want_id, want_prob, _ = mirror(case)
        if not (np.array_equal(r.logits, want) and (r.id, r.prob) == (want_id, want_prob)
                and np.array_equal(r.ring, appended(case, want_id))):
            bad.append(case.c)
    assert visible == 60, f"ids without a slot taken as their own penalise another row in {visible} of 60 cases"
    assert not bad, f"windows with -1, ids outside `[0, id_vocab)` and outside the subset are skipped: cases {bad} differ"


STAGE_CASES = ([(CAP, n, tg) for n in (0, -5, 1, 64, CAP - 1, CAP, CAP + 1, 2**31 - 1) for tg in (256, 32)]
               + [(COUPLED_HIST_CAP, n, 256) for n in (64, COUPLED_HIST_CAP, COUPLED_HIST_CAP + 904)])


@pytest.mark.parametrize("cap,last_n,threads", STAGE_CASES, ids=[f"cap{c}-last_n{n}-{t}" for c, n, t in STAGE_CASES])
def test_basalt_coupled_stage(cap, last_n, threads):
    """C.6: params and ring base copied, h = coupled_hist_len(penalty_last_n, cap). The staged image has penalty_last_n
    ``last_n``, junk in its padding and pointer slot and, as bits, a subnormal top_p, a signalling NaN min_p and a -0
    temperature; the round's copy starts as 0xFF bytes (penalty_last_n -1, so a stage that read h from the copy would
    stage nothing). The staged history holds coupled_hist_base's h slots (the last h of a drawn tail and the next
    token, -1 padded in front when the tail is short) before cap and distinct sentinels below them; the ring starts
    as drawn words. After the stage the copy is the image byte for byte, and the ring is the staged history in
    [cap - h, cap) and its own words everywhere else."""
    rs = np.random.RandomState([20261021, cap, last_n & 0xFFFFFFFF, threads])
    w = [int(x) for x in raw32(rs, 8)]
    image = bytearray(kernels.basalt_coupled_params(
        top_k=w[0] % 200 - 100, top_p=1e-40, temperature=-0.0, min_keep=3, penalty_last_n=last_n, penalty_repeat=1.1,
        penalty_freq=0.05, penalty_present=0.1, seed=w[1] << 32 | w[2], counter=w[3] << 32 | w[4], greedy=True,
        pad0=w[5] | 1, pad1=rs.bytes(7), logit_bias=w[6] << 32 | w[7]))
    struct.pack_into("<I", image, 8, 0x7FA00001)                    # min_p: a signalling NaN's bits
    h = coupled_hist_len(last_n, cap)
    mhist = -1000 - np.arange(cap, dtype=np.int32)                  # below cap - h: slots the stage must not copy
    mhist[cap - h:] = coupled_hist_base(rs.randint(0, 248320, size=min(h, 100)).astype(np.int32), 7, h)
    ring = raw32(rs, cap + gpu.CPL_MAX_T).astype(np.uint32).view(np.int32)
    assert h == 0 or not np.array_equal(ring[cap - h:cap], mhist[cap - h:]), "the ring starts unlike the staged base"
    copy, got = gpu.coupled_stage(image, mhist, ring, cap, threads=threads)
    want = ring.copy()
    want[cap - h:cap] = mhist[cap - h:]
    crit = "params and ring base copied, h = coupled_hist_len(penalty_last_n, cap)"
    assert copy == bytes(image), f"{crit}: the round's copy {copy.hex()} is not the staged image {bytes(image).hex()}"
    assert np.array_equal(got, want), \
        f"{crit}: the ring is not the staged history in [{cap - h}, {cap}) and its own words elsewhere"


@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_sampler_empty_keep(nv):
    """A8, the coupled draft's case (spec T§10.2; the sampled paths' and the probe's are test_basalt_sampler.py's): at
    min_p 2, which basalt's callers never pass and the image carries as given, the threshold sel[0] + logf(2) lies
    above the list's head, the min_p cut keeps nothing, and the merge's tail picks sel_ids[0] with out_prob 1.0
    (sampler.cu:425-429): over every case the id is the list's head mapped, the mirror's, the probability exactly 1.0
    and the ring's slot cap + j the id, where the case's own min_p draws another token on some case."""
    bad, drawn = [], 0
    for case in cases()[nv]:
        e = case._replace(p=replace(case.p, min_p=2.0))
        r, (want_id, want_prob, sel) = draft(e), mirror(e)
        assert (want_id, want_prob) == (sel.ids[0] + 25, ONE), f"min_p 2 keeps nothing in the mirror (case {case.c})"
        drawn += mirror(case)[0] != want_id
        if (r.id, r.prob, r.ring[CAP + case.j]) != (want_id, ONE, want_id):
            bad.append(case.c)
    assert drawn, "the case's own min_p draws another token than the head on some case"
    assert not bad, f"A8: `min_p = 2.0`: pick `sel_ids[0]`, prob 1.0 (cases {bad} of 60 differ)"
