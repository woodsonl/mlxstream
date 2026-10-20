# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_commit_parity.cpp and src/kernels/dec_gdn_parity.cpp:370-407,
# :504-551 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): dec_commit_parity's geometry, its GDN
# fixture, set_commit's record, the threadgroup counts and the GDN half's criterion; dec_gdn_parity's cpu_commit,
# parity_seg's commit inputs and the commit's es < 1e-3 bound, graded by max_err.
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
- DCM.1, parity's GDN criterion (:163-178, ``if (ra != rb)``): for n_keep 0..7 and p0 in {0, 1, 5, 6, 1001}, each run
  from a fresh state0, the 3 layers' state and history at threadgroup counts {1, 7, 24, 32, 170} (basalt's, :171), 816
  (one item per threadgroup) and 1000 (idle threadgroups) equal the 1-threadgroup run at p0 = 0 bit for bit. p0 reaches
  no GDN word (the commit reads word 0 only), so every count also repeats 5 times on the same kept rows.
- DCM.2, the commit against its own one-row calls (the contract's tier-1 substitute for basalt's per-stage reference,
  which the port does not build): a commit of n = 1..7 rows equals n one-row commits in turn, the k-th fed a window
  whose row 0 is row k and whose rows 1..6 are NaN in all four buffers, so a row past n that is read shows in the
  state or the history; and the 3-layer launch equals each layer committed alone (n_layers = 1), as basalt's
  reference commits layer by layer (:75-82).
- DCM.4, the commit's accuracy, on the nearest basalt bound: dec_gdn_parity's ``es < 1e-3`` (dec_gdn_parity.cpp:551),
  which grades the segment commit basalt ties bit for bit to gdn_commit_layers. ``[dcp]``, parity's fixture: for
  n_keep 0..7, each of the 3 layers (GDN_FLOATS words) holds no non-finite value, counted before the maximum, and is
  within ``max_err < 1e-3`` of ``cpu_commit``, dec_gdn_parity's float64 reference (:370-401), taken in one pass over
  the kept rows; its history equals the word copy bit for bit, and at n_keep 0 the layer keeps state0's bits.
  ``[seg-<trial>-<nseg>]``, S11, the bound on the inputs it grades in basalt (:504-551): ``seg_cases``' commits of
  parity_seg's 12 configurations, one segment at a time (D12: no _seg kernel), each segment's 2 layers committed with
  keep[k] of its T[k] rows, its 8 window rows NaN past T[k]; the same checks per layer.
- DCM.5, the keep count's edges (none in basalt): ``commit[0]`` of -1, 0 and -2^31 stores nothing but ``used`` = 0,
  and 8 and 100 give the bits of 7, the window's rows, with ``used`` = 7 (basalt clamps to kVerifyMaxT = 8 and would
  read row 7 past this window). The 7-row commit must store: every thread's rows of the state change and every
  layer's history is the word copy, so the equalities cannot hold for a commit that stores nothing.
"""

from typing import NamedTuple

import numpy as np
import pytest

from tests.basalt_gdn_ref import SEG_CASES, SEG_SEED, conv_tail, max_err, n_bad, randn, segments
from tests.kernels import basalt_gdn_gpu as gdn_gpu

U32 = np.uint32
S, HK, HV, MT = 128, 16, 48, 7       # dec_commit_parity.cpp:29
C = 2 * HK * S + HV * S              # 10240: q at [0, 2048), k at [2048, 4096), v at [4096, 10240)
REC = S * HV * S                     # a layer's recurrent state; its conv history follows
GDN_FLOATS = REC + C * 3             # 817152 (:30)
NL = 3                               # parity's GDN layers (make_gdn(rng, 3), :159)
P0S = (0, 1, 5, 6, 1001)             # parity's p0 (:164)
TGS = (1, 7, 24, 32, 170, 816, 1000) # parity's blocks (:171), then one item per threadgroup (3 x 272) and idle ones
SEG_MT, CNL = 8, 2                   # parity_seg's window rows (kDecMaxT, the contract's MT) and commit layers (:505)

CRIT1 = ("for `n_keep = 0..7` × `p0 in {0, 1, 5, 6, 1001}` (3 layers, fresh `state0` each time), state + history bytes "
         "(`GDN_FLOATS` per layer) at threadgroup counts `{1, 7, 24, 32, 170, 816, 1000}` equal the 1-threadgroup "
         "result, and every run repeats bit for bit")
CRIT2 = ("a commit of `n = 1..7` rows equals `n` successive one-row commits, the `k`-th fed row `k` (rows 1..6 NaN), "
         "byte for byte in state and history; the 3-layer launch equals each layer committed alone (`n_layers = 1`)")
CRIT4 = ("state against the numpy float64 `cpu_commit` from `state0`: `max_err < 1e-3`, with zero non-finite values "
         "counted first; `n_keep = 0` leaves `state0` bit for bit; the history equals the numpy copy `last three of "
         "[hist | qkv_0 .. qkv_{n-1}]` bitwise")
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


def cpu_commit(st0, h, qkv, gate, beta, n):
    """cpu_commit (dec_gdn_parity.cpp:370-401) in float64 for every n_keep = 0..n in one pass: n + 1 layers of
    GDN_FLOATS float32 words, the k-th basalt's ``cpu_commit(st0, h, qkv, gate, beta, k)``. ``st0`` is one layer's
    state and history, never written; ``h`` and ``qkv`` its window rows (C each), ``gate`` and ``beta`` (HV each), of
    which the first n are read. Per kept row t and value head hd (q/k head hd % HK), on the state before the row: kv =
    S^T k, delta = (v - g kv) b with g = exp(gate) in double, S = g S + k delta; vectorized over heads and columns, so
    only the double rounding of the sums differs from basalt's loops. Layer k is the state after k rows rounded to
    float32, then the history ``conv_tail`` copies from st0's and the first k qkv rows; layer 0 is st0 itself."""
    st0 = np.asarray(st0, np.float32)
    f64 = lambda a, w: np.asarray(a, np.float32).reshape(-1, w)[:n].astype(np.float64)
    hr, g, b = f64(h, C), np.exp(f64(gate, HV)), f64(beta, HV)
    qh = np.arange(HV) % HK
    sd = st0[:REC].reshape(S, HV, S).astype(np.float64)                          # [i = key][hd][j = value]
    out = [st0]
    for t in range(n):
        k, v = hr[t, HK * S:2 * HK * S].reshape(HK, S)[qh], hr[t, 2 * HK * S:].reshape(HV, S)   # [hd][i], [hd][j]
        delta = (v - g[t][:, None] * np.einsum("ihj,hi->hj", sd, k)) * b[t][:, None]
        sd = g[t][None, :, None] * sd + k.T[:, :, None] * delta[None]
        hist = conv_tail(st0[REC:], np.reshape(qkv, (-1, C))[:t + 1])
        out.append(np.concatenate([sd.astype(np.float32).reshape(-1), hist]))
    return out


def seg_cases(trial, nseg):
    """The commits of parity_seg's configuration (trial, nseg) (dec_gdn_parity.cpp:504-531), one per segment (D12):
    tests/basalt_gdn_ref.py's ``segments`` on its default stream gives the T[k], keep[k] and st0[k] that S8 and S9
    grade; the same stream then draws the recorded inputs in :505-507's order, hL N(0, 0.1), qL N(0, 1), gL and bL
    N(0, 0.3), CNL layer-major blocks of R = sum(T) rows. Yields per segment k: keep[k], st0[k]'s CNL layers (flat)
    and its window, h and qkv (CNL, SEG_MT, C), gate and beta (CNL, SEG_MT, HV): each layer's rows row0[k] ..
    row0[k] + T[k] - 1, then NaN. basalt offsets its pointers by row0[k] with h_stride R * C; the harness binds at
    offset 0, so the window is repacked to the dense strides at mt 8 (h_stride 8 C, gb_stride 8 HV)."""
    rng = np.random.default_rng((SEG_SEED, trial, nseg))
    seg = segments(trial, nseg, rng)
    R = sum(seg.T)
    rec = [randn(rng, (CNL, R, w), sd) for w, sd in ((C, 0.1), (C, 1.0), (HV, 0.3), (HV, 0.3))]   # hL, qL, gL, bL
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        win = [np.full((CNL, SEG_MT, a.shape[2]), np.nan, np.float32) for a in rec]
        for w, a in zip(win, rec):
            w[:, :t] = a[:, r0:r0 + t]
        yield seg.keep[k], seg.st0[k].reshape(-1), win


# ---- DCM.1: the threadgroup count changes no bit --------------------------------------------------------------------

def test_gdn_commit_is_bitwise_the_same_at_every_threadgroup_count(gdn):
    for n in range(MT + 1):
        ref = None                                    # the first run, p0 = 0 at 1 threadgroup
        for p0 in P0S:
            for tgs in TGS:
                st, _ = gdn_gpu.gdn_commit(gdn.state0, gdn.h, gdn.qkv, gdn.gate, gdn.beta, commit_record(n, p0), c=C,
                                           h_v=HV, mt=MT, n_layers=NL, threadgroups=tgs)
                if ref is None:
                    ref = st
                    continue
                assert_bits(f"n_keep {n}, p0 {p0}, {tgs} threadgroups against 1 at p0 0: the state and history", st,
                            ref, CRIT1)


# ---- DCM.2: n rows against n one-row commits, and each layer alone --------------------------------------------------

def test_gdn_commit_of_n_rows_equals_n_one_row_commits(gdn):
    win = [a.reshape(NL, MT, -1) for a in (gdn.h, gdn.qkv, gdn.gate, gdn.beta)]     # per layer, MT rows
    st = gdn.state0
    for n in range(1, MT + 1):
        rows = [np.full_like(w, np.nan) for w in win]                 # the window of row n - 1: rows 1..6 NaN
        for r, w in zip(rows, win):
            r[:, 0] = w[:, n - 1]
        st, _ = gdn_gpu.gdn_commit(st, *rows, commit_record(1, 0), c=C, h_v=HV, mt=MT, n_layers=NL)
        full, _ = gdn_gpu.gdn_commit(gdn.state0, gdn.h, gdn.qkv, gdn.gate, gdn.beta, commit_record(n, 0), c=C, h_v=HV,
                                     mt=MT, n_layers=NL)
        assert_bits(f"{n} rows in one commit against {n} one-row commits: the state and history", full, st, CRIT2)
        for l in range(NL):
            alone, _ = gdn_gpu.gdn_commit(layer(gdn.state0, l), *(w[l] for w in win), commit_record(n, 0), c=C,
                                          h_v=HV, mt=MT)
            assert_bits(f"layer {l} committed alone against the {NL}-layer launch, {n} rows", alone, layer(full, l),
                        CRIT2)


# ---- DCM.4: the state against the float64 cpu_commit, the history against its word copy ----------------------------

@pytest.mark.parametrize("case", ["dcp", *SEG_CASES], ids=["dcp", *(f"seg-{t}-{n}" for t, n in SEG_CASES)])
def test_gdn_commit_state_meets_the_double_reference_and_copies_the_history(case, gdn):
    # each commit: (what, n_keep, its layers' state0, its window (layers, rows, width) x 4, mt, each layer's reference)
    if case == "dcp":
        win = [a.reshape(NL, MT, -1) for a in (gdn.h, gdn.qkv, gdn.gate, gdn.beta)]     # per layer, MT rows
        refs = [cpu_commit(layer(gdn.state0, l), *(w[l] for w in win), MT) for l in range(NL)]
        commits = [(f"dcp, n_keep {n}", n, gdn.state0, win, MT, [r[n] for r in refs]) for n in range(MT + 1)]
    else:
        trial, nseg = case
        commits = ((f"seg-{trial}-{nseg}, segment {k}, keep {n}", n, st0, win, SEG_MT,
                    [cpu_commit(layer(st0, l), *(w[l] for w in win), n)[n] for l in range(CNL)])
                   for k, (n, st0, win) in enumerate(seg_cases(trial, nseg)))
    for what, n, st0, win, mt, refs in commits:
        st, _ = gdn_gpu.gdn_commit(st0, *win, commit_record(n, 0, mt), c=C, h_v=HV, mt=mt, n_layers=len(refs))
        for l, ref in enumerate(refs):
            got, at = layer(st, l), f"{what}, layer {l}"
            assert n_bad(got) == 0, f"{at}: {n_bad(got)} non-finite values in the state and history ({CRIT4})"
            e = max_err(got, ref)
            assert e < 1e-3, f"{at}: max_err {e:.3g} against cpu_commit, want max_err < 1e-3 ({CRIT4})"
            assert_bits(f"{at}: the history", got[REC:], ref[REC:], CRIT4)
            if n == 0:
                assert_bits(f"{at}: the state and history", got, layer(st0, l), CRIT4)


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
