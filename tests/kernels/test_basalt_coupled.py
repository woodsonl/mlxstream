# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its fixtures, pre-fills and pass criteria.
"""sampler_parity's coupled_parity (basalt src/kernels/sampler_parity.cpp:573-653) and the port's additions for the
coupled draft (spec T§7, T§10.2 A10) on the Metal sampler family, through tests/kernels/basalt_sampler_gpu.py's
``coupled_stage`` and ``coupled_draft``, over coupled_parity's 420 cases (``coupled_cases``: seven draft vocabularies,
subset index i the token id i + 25 of nv + 50 ids, a ring of 256 history slots and 8 drafts, a draft index j and a
step record per case; the request's image uploaded to the round's copy, as basalt uploads it, with no stage).

- C.1, its part-list half (the merge's half comes with bs_coupled_merge): bs_coupled_part's block lists (16
  SIMD-groups of 8 logits per lane) over the row bs_coupled_penalize penalised equal bs_split_part's (4 of 32 per
  lane), the verify sampler's first stage run on that row without penalties as coupled_parity's verify call runs it,
  and the mirror's (block_lists, mirror_select's per block), ids and value bits; only the first k of each block's 64
  entries are written.
- C.2: the row read back after the draft's kernels is the host's penal over draft j's window ring[cap + j - h,
  cap + j), counted over token ids and mapped through the subset's inverse map, bit for bit, at basalt's 1,024
  threads and at 32; a case without penalties leaves it untouched, and nothing writes the ring.
- C.6: bs_coupled_stage copies the staged request's 72 bytes, its padding and pointer slot included, and the staged
  history's last h = coupled_hist_len(penalty_last_n, cap) slots (coupled_hist_base's, before cap) into the ring,
  nothing else, at basalt's 256 threads and at 32, at coupled_parity's cap and the engine's 4,096.
"""

import functools
import struct
from dataclasses import replace

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_sampler_ref import (COUPLED_HIST_CAP, COUPLED_VOCABS, coupled_cases, coupled_hist_base,
                                      coupled_hist_len, coupled_hist_start, penal_vec, raw32, sampled_k)
from tests.kernels import basalt_sampler_gpu as gpu
from tests.kernels.test_basalt_sampler import block_lists

CAP = gpu.CPL_CAP


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


@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_matches_verify_sampler(nv):
    """C.1 (:644-653), its part-list half: for each case, bs_coupled_part's list of every block over the penalised row
    (read back after the draft's kernels) equals bs_split_part's over that row with the verify call's request (``q =
    p; q.penalty_last_n = 0``, :636-639), and mirror_select's over the block (block_lists), ids and value bits, the
    sentinels (nv, -inf) included; the entries past k keep their prefill (the stride is 64, k is the request's)."""
    bad = []
    for case in cases()[nv]:
        r = draft(case)
        q = replace(case.p, penalty_last_n=0)
        k, row = sampled_k(q.top_k, nv), r.logits.view(np.float32)[None]
        split = gpu.split_part(row, q)[0]
        if not (np.array_equal(r.cand[:, :k], split) and np.array_equal(split, block_lists(row, None, 0, q)[0])
                and np.all(r.cand[:, k:] == 0xFFFFFFFF)):
            bad.append(case.c)
    assert not bad, f"bs_split_part and bs_coupled_part give equal lists (the mirror's): cases {bad} of 60 differ"


@pytest.mark.parametrize("threads", [1024, 32])
@pytest.mark.parametrize("nv", COUPLED_VOCABS)
def test_basalt_coupled_penalties_vs_host(nv, threads):
    """C.2: for each case, the row read back after the draft's kernels at ``threads`` per threadgroup equals penal_vec
    of the case's logits over its window (``window``), every one of the nv logits bit for bit, and the ring is
    unchanged. The fixture's penalties (a third of the cases: last_n 64, repeat 1.1, freq 0.05, present 0.1) change
    some logits, so the check sees a kernel that applies none."""
    bad, changed = [], 0
    for case in cases()[nv]:
        r = draft(case, threads)
        want = penal_vec(case.logits, window(case), case.p).view(np.uint32)
        changed += int(np.count_nonzero(want != case.logits.view(np.uint32)))
        if not (np.array_equal(r.logits, want) and np.array_equal(r.ring, case.ring)):
            bad.append(case.c)
    assert changed, "the fixture's penalties change some logit"
    assert not bad, (f"penalised logits read back equal host penal over the ring window through inv (bitwise): "
                     f"cases {bad} of 60 differ")


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
    assert copy == bytes(image), f"the round's copy is the staged image, 72 bytes: {copy.hex()} != {bytes(image).hex()}"
    assert np.array_equal(got, want), \
        f"the ring is the staged history in [cap - h, cap) = [{cap - h}, {cap}) and unchanged elsewhere"
