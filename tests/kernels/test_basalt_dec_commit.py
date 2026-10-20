# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_commit_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its geometry, its GDN fixture and set_commit's record.
"""dec_commit_parity's GDN half (basalt src/kernels/dec_commit_parity.cpp) on the Metal commit
(kernels/common/basalt_gdn_commit.metal's ``gdn_commit_layers``), through tests/kernels/basalt_gdn_gpu.py's
``gdn_commit``, which binds a sentinel tail past the state and checks it and the ``used`` word after every dispatch.

- The geometry is dec_commit_parity's (:29-31): ``HK`` q/k heads and ``HV`` value heads of ``S``, ``C`` conv channels
  (q, k, then v), ``MT`` window rows (7, below kVerifyMaxT's 8), ``GDN_FLOATS`` floats per layer: ``REC`` of the
  S x HV x S recurrent state, element (i, hd, j) at (i * HV + hd) * S + j, then the conv history [C][3].
- ``commit_record`` is set_commit's record (:147-153), the engine's ``commit`` (D19); ``make_gdn`` the GDN fixture's
  draws (:57-74) on a numpy Generator (basalt's mt19937 draws cannot be replayed, and every check here compares the
  GPU with itself or with an exact host copy). The conv history a commit of n rows must leave is
  tests/basalt_gdn_ref.py's ``conv_tail`` of the history and the layer's first n qkv rows, a word copy
  (gdn_conv_commit_n, verify_kernels.cu:88-101).
- DCM.5, the keep count's edges (none in basalt): ``commit[0]`` of -1, 0 and -2^31 stores nothing but ``used`` = 0,
  and 8 and 100 give the bits of 7, the window's rows, with ``used`` = 7 (basalt clamps to kVerifyMaxT = 8 and would
  read row 7 past this window). The 7-row commit must store: every thread's rows of the state change and every
  layer's history is the word copy, so the equalities cannot hold for a commit that stores nothing.
"""

from typing import NamedTuple

import numpy as np
import pytest

from tests.basalt_gdn_ref import conv_tail, randn
from tests.kernels import basalt_gdn_gpu as gdn_gpu

U32 = np.uint32
S, HK, HV, MT = 128, 16, 48, 7       # dec_commit_parity.cpp:29
C = 2 * HK * S + HV * S              # 10240: q at [0, 2048), k at [2048, 4096), v at [4096, 10240)
REC = S * HV * S                     # a layer's recurrent state; its conv history follows
GDN_FLOATS = REC + C * 3             # 817152 (:30)
NL = 3                               # parity's GDN layers (make_gdn(rng, 3), :159)

CRIT5 = ("`commit[0] in {-1, 0}` leaves state and history equal to `state0` bitwise; `commit[0] in {8, 100}` gives "
         "the bits of `commit[0] = 7` (the window's static row count) with no shader-validation report")


def commit_record(n_keep, p0, mt=MT):
    """set_commit's record (dec_commit_parity.cpp:147-153), the engine's ``commit`` (D19), as 2 + mt int32 words: word
    0 n_keep, word 1 n_keep - 1 (wrapped as int32), word 2 + t p0 + t for t < n_keep, else -1. The GDN commit reads
    word 0 only. The qsa lane's indexer commit test imports this helper for the words from 2 on (D20); the program
    tier builds its 8-row record with mt=8."""
    words = [n_keep, n_keep - 1] + [p0 + t if t < n_keep else -1 for t in range(mt)]
    return np.array(words, np.int64).astype(np.int32)


class Gdn(NamedTuple):
    """The buffers of make_gdn (dec_commit_parity.cpp:57-74) the commit reads, float32: nl layers of state and conv
    history (``state0``), and per layer MT rows of ``h`` and ``qkv`` (C each) and of ``gate`` and ``beta`` (HV each)."""
    state0: np.ndarray
    h: np.ndarray
    qkv: np.ndarray
    gate: np.ndarray
    beta: np.ndarray


def make_gdn(rng, nl):
    """make_gdn (dec_commit_parity.cpp:57-74) in its draw order, without z, y and gamma, which follow and which the
    commit never reads: state0 N(0, 0.1), h N(0, 0.1), qkv N(0, 1), gate N(0, 0.3), beta N(0, 0.3)."""
    return Gdn(randn(rng, nl * GDN_FLOATS, 0.1), randn(rng, nl * MT * C, 0.1), randn(rng, nl * MT * C, 1.0),
               randn(rng, nl * MT * HV, 0.3), randn(rng, nl * MT * HV, 0.3))


@pytest.fixture(scope="module")
def gdn():
    """parity's GDN fixture (dec_commit_parity.cpp:156-159): make_gdn(rng, 3) on seed 7."""
    return make_gdn(np.random.default_rng(7), NL)


def assert_bits(what, got, want, crit):
    """``got`` and ``want`` hold the same 32-bit words, NaN payloads included; else name the first differing index."""
    g, w = (np.ascontiguousarray(a).reshape(-1).view(U32) for a in (got, want))
    assert g.shape == w.shape, f"{what}: {g.shape} words against {w.shape} ({crit})"
    bad = np.flatnonzero(g != w)
    assert bad.size == 0, (f"{what}: {bad.size} words differ, the first at {bad[0]}: got {g[bad[0]]:#010x}, want "
                           f"{w[bad[0]]:#010x} ({crit})")


def layer(a, l):
    return a[l * GDN_FLOATS:(l + 1) * GDN_FLOATS]


# ---- DCM.5: the keep count's edges ----------------------------------------------------------------------------------

def test_gdn_commit_clamps_the_keep_count_to_the_window(gdn):
    def commit(n):
        return gdn_gpu.gdn_commit(gdn.state0, gdn.h, gdn.qkv, gdn.gate, gdn.beta, commit_record(n, 0), c=C, h_v=HV,
                                  mt=MT, n_layers=NL)

    for n in (-1, 0, -2 ** 31):
        st, used = commit(n)
        assert used == 0, f"commit[0] = {n}: used {used}, want 0 ({CRIT5})"
        assert_bits(f"commit[0] = {n}: the state and history", st, gdn.state0, CRIT5)
    full, used = commit(MT)
    assert used == MT, f"commit[0] = {MT}: used {used} ({CRIT5})"
    for l in range(NL):
        st, st0 = layer(full, l), layer(gdn.state0, l)
        moved = (st[:REC].view(U32) != st0[:REC].view(U32)).reshape(4, S // 4, HV, S).any(axis=1)
        assert moved.all(), (f"layer {l}: {np.count_nonzero(~moved)} threads' 32 rows (row group, head, column) kept "
                             f"state0's bits after {MT} rows, so the clamp's equalities have no power ({CRIT5})")
        assert_bits(f"layer {l}: the history after {MT} rows", st[REC:],
                    conv_tail(st0[REC:], gdn.qkv[l * MT * C:(l + 1) * MT * C]), CRIT5)
    for n in (MT + 1, 100):
        st, used = commit(n)
        assert used == MT, f"commit[0] = {n}: used {used}, want {MT} ({CRIT5})"
        assert_bits(f"commit[0] = {n} against {MT}: the state and history", st, full, CRIT5)
