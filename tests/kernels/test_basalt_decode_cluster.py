# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/decode_cluster_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its argmax fixtures, pre-fills and pass criterion.
"""decode_cluster_parity's argmax half (basalt src/kernels/decode_cluster_parity.cpp:174-243, :412-421) on the Metal
sampler family: the cluster argmax, here its partials and merge dispatches (``greedy_cl``,
tests/kernels/basalt_sampler_gpu.py), against the one-block greedy kernel (``greedy_1b``) and the host's answer, the
lowest index of the largest non-NaN value above -inf, else 0.

DCP.1: for each vocabulary of basalt's list, each of its 12 argmax patterns (``fill_logits``, :175-212, on the port's
own seeded draws: normal(0, 3), then the pattern's overwrites) and T in (1, 5), even rows take the pattern and odd rows
the random one (:216); the cluster output starts as 0xA5 bytes, the one-block output as 0x5A (:220-221). The one-block
side runs as basalt's ``greedy_one_block`` does (:45-62): a one-slot history of -1 per row with penalty_last_n 1, a
window that counts nothing. basalt skips without clusters (:390-393); the port has none to lack, so it never skips.
numpy's float32 compares keep subnormals, as the host's do: the denormal pattern's largest values are subnormals,
which this GPU's own compares would see as zero. Each denormal case of more than one logit is asserted to hold a row
whose pick a compare that flushed subnormals would miss; a case whose draws put each denormal row's largest value at
id 0, where the flushed row's tie also lands, is drawn again (``case_rows``).
"""

from dataclasses import replace

import numpy as np
import pytest

from tests.basalt_sampler_ref import SamplerParams
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_sampler_gpu as gpu

F32 = np.float32
NAN, INF = F32(np.nan), F32(np.inf)
VOCABS = (248320, 248321, 248319, 32771, 24576, 8193, 1000, 33, 1)
PATTERNS = ("random", "ties", "all-equal", "all--inf", "all-nan", "nan+max", "+inf", "signed-zero", "max-last",
            "max-first", "denormal", "ties-across")
GREEDY = SamplerParams(greedy=True, temperature=0.0)
NEUTRAL = replace(GREEDY, penalty_last_n=1)          # greedy_one_block's window (:57-61), over a history of -1


def fill_logits(rs, nv, pattern):
    """basalt's ``fill_logits`` (:183-212) on RandomState ``rs``: normal(0, 3) values, then the pattern's overwrites in
    its order, ``at()`` being uniform_int_distribution<int>(0, nv - 1)."""
    l = rs.standard_normal(nv).astype(F32) * F32(3)
    at = lambda n=None: rs.randint(nv, size=n)
    if pattern == "ties":
        for _ in range(1 + at() % 9):
            l[at()] = 100
    elif pattern == "all-equal":
        l[:] = 0.25
    elif pattern == "all--inf":
        l[:] = -INF
    elif pattern == "all-nan":
        l[:] = NAN
    elif pattern == "nan+max":
        l[at(nv) % 3 == 0] = NAN
        l[at()] = 50
        l[at()] = 50
    elif pattern == "+inf":
        for _ in range(4):
            l[at()] = INF
    elif pattern == "signed-zero":
        l = -np.abs(l) - F32(1)
        for i in range(6):
            l[at()] = 0.0 if i & 1 else -0.0
    elif pattern == "max-last":
        l[nv - 1] = 1e30
    elif pattern == "max-first":
        l[0] = 1e30
        if nv > 1:
            l[nv - 1] = 1e30
    elif pattern == "denormal":                       # (float) (at() % 5) * 1e-42f - 1e-30f * (v & 1)
        l = (at(nv) % 5).astype(F32) * F32(1e-42) - F32(1e-30) * (np.arange(nv) & 1).astype(F32)
    elif pattern == "ties-across":                    # 7 in several threadgroups' and threads' ranges, plus NaN, -inf
        v = 0
        while v < nv:
            l[v] = 7
            v += 1 + at() % 9000
        l[3::11] = NAN
        l[5::13] = -INF
    return l


def host_argmax(r):
    """The host's answer (:229-236): ``if (r[v] > -kInf && (hb < 0 || r[v] > r[hb])) hb = v;``, else 0. ``r > -inf``
    is false for NaN and -inf; equal values (-0 and +0 among them) go to the lowest index."""
    valid = r > -INF
    return int(np.flatnonzero(valid & (r == r[valid].max()))[0]) if valid.any() else 0


def test_basalt_decode_cluster_host_orders_subnormals():
    """The host side keeps subnormals: without that the denormal pattern would pass on a flushing compare."""
    assert host_argmax(F32([0.0, 1e-45, 3e-45, 2e-45, -0.0])) == 2
    assert host_argmax(F32([-np.inf, np.nan, -0.0, 0.0])) == 2


def case_rows(nv, pattern, T):
    """The case's T rows (:216): even rows the pattern, odd rows the random one, on RandomState([20261002, nv, pattern
    index, T]). A denormal case in which a compare that read subnormals as zero would pick what the host picks in every
    row (each denormal row's largest value at id 0) is drawn again with a fifth seed word, the draw's number."""
    for draw in range(8):
        rs = np.random.RandomState([20261002, nv, PATTERNS.index(pattern), T] + ([draw] if draw else []))
        l = np.stack([fill_logits(rs, nv, pattern if t % 2 == 0 else "random") for t in range(T)])
        if pattern != "denormal" or nv == 1 or any(host_argmax(emu.ftz(r)) != host_argmax(r) for r in l):
            break
    return l


@pytest.mark.parametrize("T", (1, 5))
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("nv", VOCABS)
def test_basalt_decode_cluster_argmax(nv, pattern, T):
    """DCP.1, ``run_argmax_case`` (:214-243): every row's cluster pick equals its one-block pick and the host's. A
    denormal case of more than one logit has a row whose pick a compare that flushed subnormals would miss."""
    l = case_rows(nv, pattern, T)
    if pattern == "denormal" and nv > 1:
        assert any(host_argmax(emu.ftz(r)) != host_argmax(r) for r in l), "a flushing compare would pass this case"
    a = gpu.greedy_cl(l, GREEDY, fill=0xA5)
    b = gpu.greedy_1b(l, NEUTRAL, hist=np.full(T, -1, np.int32), hist_len=1, fill=0x5A)
    hb = [host_argmax(r) for r in l]
    bad = sum(int(a[t] != b[t] or a[t] != hb[t]) for t in range(T))
    assert bad == 0, "if (a[(size_t) t] != b[(size_t) t] || a[(size_t) t] != hb) ++bad;"
