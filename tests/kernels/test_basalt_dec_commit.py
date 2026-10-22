# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_commit_parity.cpp, src/kernels/dec_gdn_parity.cpp:370-407,
# :504-551 and src/kernels/cuda/verify_kernels.cu:296-330 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
# third_party/NOTICE): dec_commit_parity's geometry, its GDN fixture, set_commit's record, the threadgroup counts and
# the GDN half's criterion; dec_gdn_parity's cpu_commit, parity_seg's commit inputs and the commit's es < 1e-3 bound,
# graded by max_err; commit_pre_run's operation order, which commit_f32 transcribes.
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
  keep[k] of its T[k] rows, its 8 window rows NaN past T[k]; the same checks per layer. Each segment is committed
  again as basalt dispatches it (:519-523: every recorded array from row0[k]'s row, h_stride R C, gb_stride R HV,
  mt T[k]), which must equal the dense run bit for bit, as basalt ties that commit to the one es grades (:535).
- DCM.5, the keep count's edges (none in basalt): ``commit[0]`` of -1, 0 and -2^31 stores nothing but ``used`` = 0,
  and 8 and 100 give the bits of 7, the window's rows, with ``used`` = 7 (basalt clamps to kVerifyMaxT = 8 and would
  read row 7 past this window). The 7-row commit must store: every thread's rows of the state change and every
  layer's history is the word copy, so the equalities cannot hold for a commit that stores nothing.
- DCM.7, subnormals (contract invariant 5; none in basalt, whose fixture keeps the decay in about [0.3, 3]): first the
  decay probe, ``probe_expf_ieee``, which runs the commit's own ``expf_ieee`` text: for gates in [-100, -88] (the
  fixture's and a grid) the decay is a nonzero subnormal, and every probed decay is within 1e-4 relative, plus
  2^-148, of float64's exp. Then one layer, ``subnormal_layer``, with v = 0 and two head groups: (a) a subnormal decay
  on a state and keys of order 0.1 to 1, (b) ordinary gates on a state of order 1e-39 with keys of order 1. A commit of
  n in {1, MT} rows equals ``commit_f32``, commit_pre_run's order in numpy float32 on the probed decays, bit for bit,
  and the same oracle flushing every input and result differs from it in value: group (a) at n = 1 (its state is
  +-0 on either build from the second row on), group (b) at n = MT.
- DCM.8, a window after a commit continues the longer window (contract invariant 1 across the commit; none in basalt,
  whose commit and window recurrences do the same per-token arithmetic, verify_kernels.cu:296-330 against
  dec_gdn_rec.cu:248-259 and gdn_step_norm_body's update, verify_kernels.cu:417 and :423, so it ties the commit to the
  window whose state it continues): on ``window_case``, tests/basalt_gdn_ref.py's make_layer and MT rows of qkv, z
  and x, window A runs rows 0..6 from the layer's state0; for n = 1..6, the commit of n rows (mt 7) of window A's own
  h, gate and beta and the raw qkv rows leaves the state and history from which window B runs rows n..6, and window
  B's row j of every output equals window A's row n + j bit for bit: h, gate, beta and the fp32 y, and on ``[rec]``
  (dec_gdn_rec built with its y, GDN_Y_BOUND 1, D13) the q8_1 rows yq and win_qkv too; ``[stage]`` runs the
  per-stage chain, gdn_ab_multi and gdn_conv_l2_multi into gdn_step_norm_multi. Every sentinel holds and every output
  is finite.
- DCM.9, NaNs and special words (contract invariant 5's NaN rule; none in basalt, which stores the state's float,
  verify_kernels.cu:327-330): parity's fixture with a quarter of each layer's conv history and qkv rows replaced at
  random by ``SPECIALS`` (subnormals, +-0, +-inf and the NaN payloads 0x7FA00001 and 0xFFC00000), committed for n =
  1..MT rows with and without a NaN (this GPU's 0x7fc00000) in row 0's v at ``NAN_AT``. The update spreads it down that
  column only, whose 128 state words must be stored as sm_120's 0x7fffffff (cuda_nan); every other word equals the
  NaN-free run's, whose state is finite; and each layer's history is conv_tail's word copy, every special word in each
  of its three slots (n = 1 and 2 keep words of the initial history).
"""

from typing import NamedTuple

import numpy as np
import pytest

from monolith import kernels
from tests.basalt_gdn_ref import N, SEG_CASES, SEG_SEED, ZV, conv_tail, make_layer, max_err, n_bad, randn, segments
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gdn_gpu as gdn_gpu
from tests.kernels import basalt_gpu as gpu

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
CRIT7 = ("decay probe: for gates in `[-100, -88]`, `0 < g < 2^-126` and `|g - exp(gate)| <= 1e-4 * exp(gate) + "
         "2^-148` (float64); `v = 0` with two head groups, (a) gates in `[-100, -88]` with `state0` and `k` of order "
         "0.1 to 1 and (b) gates randn 0.3 with `state0` randn scaled by `1e-39` and `k` of order 1; state bitwise "
         "equal to a numpy float32 + `basalt_emu.fma32` transcription with the probed `g`; power: the same oracle with "
         "`basalt_emu.ftz` on every input and result differs from it in each group")
CRIT8 = ("window A over rows `0..6` from `state0`, commit `n = 1..6`, window B over rows `n..6` from the committed "
         "state and history: `y_B[j] == y_A[n + j]` bitwise, for the per-stage float `y` and for `dec_gdn_rec`'s "
         "outputs")
CRIT9 = ("A NaN in row 0's `v` of one (head, col) makes that column's 128 state words `0x7FFFFFFF`, and every other "
         "word equals the NaN-free run's. The qkv rows and the initial history hold subnormals, +-0, +-inf and the NaN "
         "payloads `0x7FA00001` and `0xFFC00000`, and the history must equal the numpy word copy")


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
    N(0, 0.3), CNL layer-major blocks of R = sum(T) rows. Yields per segment k: keep[k], st0[k]'s CNL layers (flat),
    its window repacked to the dense strides at mt 8, h and qkv (CNL, SEG_MT, C), gate and beta (CNL, SEG_MT, HV):
    each layer's rows row0[k] .. row0[k] + T[k] - 1, then NaN, so a read past them shows; and basalt's own layout
    (:519-521) as (rows, T[k], R): ``rows`` the four recorded arrays flat from row0[k]'s row, so layer l's rows start
    l * R rows on, which basalt's commit reads with h_stride R * C and gb_stride R * HV (past a segment's rows lie
    the next segment's, not NaN)."""
    rng = np.random.default_rng((SEG_SEED, trial, nseg))
    seg = segments(trial, nseg, rng)
    R = sum(seg.T)
    rec = [randn(rng, (CNL, R, w), sd) for w, sd in ((C, 0.1), (C, 1.0), (HV, 0.3), (HV, 0.3))]   # hL, qL, gL, bL
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        win = [np.full((CNL, SEG_MT, a.shape[2]), np.nan, np.float32) for a in rec]
        for w, a in zip(win, rec):
            w[:, :t] = a[:, r0:r0 + t]
        yield seg.keep[k], seg.st0[k].reshape(-1), win, ([a.reshape(-1)[r0 * a.shape[2]:] for a in rec], t, R)


PROBE = """
kernel void probe_expf_ieee(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = as_type<uint>(expf_ieee(x[i]));
}
"""


def probe_expf_ieee(x):
    """The port's decay, basalt_ieee.metal's ``expf_ieee`` (the commit's __expf, verify_kernels.cu:315), on each
    element of the float32 array ``x``: the inline kernel ``PROBE`` appended to basalt_gdn_commit_source()
    (test_basalt_platform.py's pattern), so it runs the very text the commit compiles. fast::exp2's bits have no host
    model, so DCM.7's oracle takes the decay from here. Returns float32 in x's shape."""
    x = np.ascontiguousarray(x, np.float32)
    pso = gpu.pipeline(kernels.basalt_gdn_commit_source() + PROBE, "probe_expf_ieee")
    out = gpu.run(pso, [x.tobytes(), x.nbytes], np.uint32(x.size).tobytes(), 2, -(-x.size // 256), 256, fill=0xFF)
    return np.frombuffer(out[1], np.float32).reshape(x.shape)


def commit_f32(st0, h, gate, beta, n, g, ftz=False):
    """DCM.7's oracle: one layer's state after its first n rows, commit_pre_run (verify_kernels.cu:296-330) in the
    order nvcc compiles it (verify_gdn.md section 7), in numpy float32, every operation rounded once with subnormals
    kept (basalt_emu.fma32 for each FFMA). Per row t: the decay gt = g(gate[t]), ``g`` the port's decay on a float32
    array (probe_expf_ieee); per row group rg of 32 key rows and column, kv = fma(s_r, k_r, kv) from +0, r ascending;
    kv_col = ((kv_0 + kv_1) + kv_2) + kv_3; delta = fma(-gt, kv_col, v) b; per row kd = k delta, s = fma(gt, s, kd).
    ``ftz``: the same with basalt_emu.ftz on every input and every result (fma_ftz, mul_ftz, add_ftz), a flushing
    build. ``st0``: the layer, of which the state is read; ``h`` its rows (C each), ``gate`` and ``beta`` (HV each).
    Returns the S x HV x S state, flat float32."""
    fma, mul, add = (emu.fma_ftz, emu.mul_ftz, emu.add_ftz) if ftz else (emu.fma32, np.multiply, np.add)
    s = np.asarray(st0, np.float32)[:REC].reshape(4, 32, HV, S)                # [rg][r][hd][j]: key row 32 rg + r
    hr, br = np.reshape(h, (-1, C)), np.reshape(beta, (-1, HV))
    gd = np.asarray(g(np.reshape(gate, (-1, HV))[:n]), np.float32)[:, :, None]  # [t][hd][1]
    qh = np.arange(HV) % HK
    for t in range(n):
        k = hr[t, HK * S:2 * HK * S].reshape(HK, 4, 32)[qh].transpose(1, 2, 0)[..., None]   # [rg][r][hd][1]
        v = hr[t, 2 * HK * S:].reshape(HV, S)
        kv = np.zeros((4, HV, S), np.float32)
        for r in range(32):
            kv = fma(s[:, r], k[:, r], kv)
        delta = mul(fma(emu.sign_flip(gd[t]), add(add(add(kv[0], kv[1]), kv[2]), kv[3]), v), br[t][:, None])
        s = fma(gd[t], s, mul(k, delta))
    return s.reshape(-1)


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
    # each commit: (what, n_keep, its layers' state0, its window (layers, rows, width) x 4, mt, each layer's reference,
    # seg_cases' basalt layout or None)
    if case == "dcp":
        win = [a.reshape(NL, MT, -1) for a in (gdn.h, gdn.qkv, gdn.gate, gdn.beta)]     # per layer, MT rows
        refs = [cpu_commit(layer(gdn.state0, l), *(w[l] for w in win), MT) for l in range(NL)]
        commits = [(f"dcp, n_keep {n}", n, gdn.state0, win, MT, [r[n] for r in refs], None) for n in range(MT + 1)]
    else:
        trial, nseg = case
        commits = ((f"seg-{trial}-{nseg}, segment {k}, keep {n}", n, st0, win, SEG_MT,
                    [cpu_commit(layer(st0, l), *(w[l] for w in win), n)[n] for l in range(CNL)], bas)
                   for k, (n, st0, win, bas) in enumerate(seg_cases(trial, nseg)))
    for what, n, st0, win, mt, refs, bas in commits:
        st, _ = gdn_gpu.gdn_commit(st0, *win, commit_record(n, 0, mt), c=C, h_v=HV, mt=mt, n_layers=len(refs))
        for l, ref in enumerate(refs):
            got, at = layer(st, l), f"{what}, layer {l}"
            assert n_bad(got) == 0, f"{at}: {n_bad(got)} non-finite values in the state and history ({CRIT4})"
            e = max_err(got, ref)
            assert e < 1e-3, f"{at}: max_err {e:.3g} against cpu_commit, want max_err < 1e-3 ({CRIT4})"
            assert_bits(f"{at}: the history", got[REC:], ref[REC:], CRIT4)
            if n == 0:
                assert_bits(f"{at}: the state and history", got, layer(st0, l), CRIT4)
        if bas is not None:      # the segment's commit as basalt dispatches it (:519-523), mt T[k] (keep[k] <= T[k])
            rows, t, r = bas
            strided, _ = gdn_gpu.gdn_commit(st0, *rows, commit_record(n, 0, t), c=C, h_v=HV, mt=t, n_layers=CNL,
                                            strides={"h_stride": r * C, "gb_stride": r * HV})
            assert_bits(f"{what}: basalt's layout (from row0[k], h_stride R C, gb_stride R HV, R {r}) against the "
                        "dense one", strided, st, "basalt ties its per-segment commit to the one es grades bitwise, "
                        ":535")


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


# ---- DCM.7: subnormals, as the IEEE build keeps them ----------------------------------------------------------------

@pytest.fixture(scope="module")
def subnormal_layer():
    """DCM.7's layer (C section 6), on its own seed: the even value heads are group (a), gates U(-100, -88) (a
    subnormal decay) with state0 and keys of magnitude U(0.1, 1), random signs; the odd ones group (b), gates N(0, 0.3)
    with state0 N(0, 1) times 1e-39 (a subnormal state) and keys N(0, 1). HK is even, so a value head's q/k head
    hd % HK keeps its parity and each group has keys of its own. v = 0, q (never read) 0, beta N(0, 0.3); the conv
    history and qkv 0."""
    rng = np.random.default_rng(70)
    mag = lambda shape: (rng.uniform(0.1, 1.0, shape) * rng.choice((-1.0, 1.0), shape)).astype(np.float32)
    a, ka = np.arange(HV) % 2 == 0, np.arange(HK) % 2 == 0
    state0 = np.zeros(GDN_FLOATS, np.float32)
    state0[:REC] = np.where(a[:, None], mag((S, HV, S)), randn(rng, (S, HV, S), 1e-39)).reshape(-1)
    h = np.zeros((MT, C), np.float32)
    h[:, HK * S:2 * HK * S] = np.where(ka[:, None], mag((MT, HK, S)), randn(rng, (MT, HK, S), 1.0)).reshape(MT, -1)
    gate = np.where(a, rng.uniform(-100, -88, (MT, HV)), randn(rng, (MT, HV), 0.3)).astype(np.float32)
    return Gdn(state0, h.reshape(-1), np.zeros(MT * C, np.float32), gate.reshape(-1), randn(rng, MT * HV, 0.3))


@pytest.mark.parametrize("n", [1, MT])
def test_gdn_commit_keeps_subnormals_like_the_ieee_build(n, subnormal_layer):
    lay = subnormal_layer
    # the probe, before the oracle takes its bits: the fixture's gates and a grid over [-100, -88]
    x = np.concatenate([lay.gate, np.linspace(-100, -88, 4097, dtype=np.float32)])
    g, want = probe_expf_ieee(x).astype(np.float64), np.exp(x.astype(np.float64))
    bad = np.flatnonzero((x >= -100) & (x <= -88) & ~((g > 0) & (g < 2.0 ** -126)))
    assert bad.size == 0, f"expf_ieee({x[bad[0]]}) = {g[bad[0]]!r}, want a nonzero subnormal ({CRIT7})"
    bad = np.flatnonzero(~(np.abs(g - want) <= 1e-4 * want + 2.0 ** -148))
    assert bad.size == 0, f"expf_ieee({x[bad[0]]}) = {g[bad[0]]!r} against exp's {want[bad[0]]!r} ({CRIT7})"
    st, _ = gdn_gpu.gdn_commit(*lay, commit_record(n, 0), c=C, h_v=HV, mt=MT)
    ieee = commit_f32(lay.state0, lay.h, lay.gate, lay.beta, n, probe_expf_ieee)
    assert_bits(f"{n} rows: the state against commit_f32", st[:REC], ieee, CRIT7)
    # the power: group (a) at n = 1 (from its second row on, its state is +-0 on either build), group (b) at n = MT
    grp = 0 if n == 1 else 1
    ftz = commit_f32(lay.state0, lay.h, lay.gate, lay.beta, n, probe_expf_ieee, ftz=True)
    i, f = (o.reshape(S, HV, S)[:, grp::2] for o in (ieee, ftz))
    assert np.any(i != f), (f"group ({'ab'[grp]}), {n} rows: the flushing oracle's state equals the IEEE one in "
                            f"value, so the fixture cannot tell the builds apart ({CRIT7})")


# ---- DCM.8: a window after a commit continues the longer window -----------------------------------------------------

@pytest.fixture(scope="module")
def window_case():
    """DCM.8's layer and window rows, on their own seed: tests/basalt_gdn_ref.py's make_layer (parity_seg's projection
    types; the window runs none of its projections), whose state is state0, then MT rows each of qkv, z and x,
    N(0, 1)."""
    rng = np.random.default_rng(80)
    return make_layer(14, 23, 14, rng), randn(rng, (MT, C), 1.0), randn(rng, (MT, ZV), 1.0), randn(rng, (MT, N), 1.0)


def window(path, lay, state, qkv, z, x):
    """A GDN window over the rows of qkv, z and x from ``state`` (one layer's state and history) on ``path``: "rec",
    dec_gdn_rec built with its fp32 y (GDN_Y_BOUND 1, D13), or "stage", the per-stage chain, gdn_ab_multi and
    gdn_conv_l2_multi into gdn_step_norm_multi. Asserts every sentinel intact and every output finite (yq: its blocks'
    d and sum halves). Returns {name: the output's rows}: h, gate, beta and y, and on rec the q8_1 rows yq and the
    qkv rows win_qkv."""
    if path == "rec":
        r = gdn_gpu.rec(lay, state, qkv, x, z, y=True)
        out, ok = {"h": r.h, "gate": r.gate, "beta": r.beta, "y": r.y, "yq": r.yq, "win_qkv": r.qkv}, r.ok
    else:
        gate, beta, ok_ab = gdn_gpu.ab(lay, x)
        h, ok_h = gdn_gpu.conv_l2(state, qkv, lay.conv_w)
        y, ok_y = gdn_gpu.step_norm(state, h, gate, beta, z, lay.gamma)
        out, ok = {"h": h, "gate": gate, "beta": beta, "y": y}, ok_ab and ok_h and ok_y
    at = f"{path}, a window of {len(qkv)} rows"
    assert ok, f"{at}: a store past the window's rows (a sentinel byte changed) ({CRIT8})"
    for name, a in out.items():
        v = np.ascontiguousarray(a.reshape(-1, kernels.Q8_1_BLOCK_BYTES)[:, :4]).view(np.float16) if name == "yq" else a
        assert n_bad(v) == 0, f"{at}: {n_bad(v)} non-finite {name} values, want none ({CRIT8})"
    return out


@pytest.mark.parametrize("path", ["rec", "stage"])
def test_gdn_window_after_a_commit_continues_the_longer_window(path, window_case):
    lay, qkv, z, x = window_case
    a = window(path, lay, lay.state, qkv, z, x)                       # window A: rows 0..6 from state0
    for n in range(1, MT):
        st, _ = gdn_gpu.gdn_commit(lay.state, a["h"], qkv, a["gate"], a["beta"], commit_record(n, 0), c=C, h_v=HV,
                                   mt=MT)                             # window A's stored rows and the raw qkv rows
        b = window(path, lay, st, qkv[n:], z[n:], x[n:])              # window B: rows n..6 from the committed layer
        for name, rows in b.items():
            assert_bits(f"{path}, {n} rows committed: window B's {name} rows 0..{MT - 1 - n} against window A's rows "
                        f"{n}..{MT - 1}", rows, a[name][n:], CRIT8)


# ---- DCM.9: a NaN the update makes is stored as sm_120's, the history's words as they came --------------------------

SPECIALS = np.array([0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF,   # +-(the smallest and the largest subnormal)
                     0x00000000, 0x80000000, 0x7F800000, 0xFF800000,   # +-0, +-inf
                     0x7FA00001, 0xFFC00000], U32)                     # a signaling NaN's payload, a negative quiet NaN
NAN_AT = (1, 37, 77)     # DCM.9's NaN in row 0's v: layer, value head (q/k head 5), column (block 2, lane 13)


def test_gdn_commit_stores_cudas_nan_and_copies_specials(gdn):
    rng = np.random.default_rng(90)

    def special(w):      # a quarter of the uint32 words w, drawn at random, become SPECIALS words, in place
        m = rng.random(w.shape) < 0.25
        w[m] = rng.choice(SPECIALS, np.count_nonzero(m))

    st0 = gdn.state0.view(U32).reshape(NL, GDN_FLOATS).copy()   # words throughout: no float operation quiets a NaN
    special(st0[:, REC:])
    qkv = gdn.qkv.view(U32).copy()
    special(qkv)
    l, hd, j = NAN_AT
    h = gdn.h.view(U32).copy()
    h[l * MT * C + 2 * HK * S + hd * S + j] = 0x7FC00000          # this GPU's NaN, not sm_120's
    col = l * GDN_FLOATS + (np.arange(S) * HV + hd) * S + j       # that column's 128 state words
    st0, qkv, h = (a.reshape(-1).view(np.float32) for a in (st0, qkv, h))
    for n in range(1, MT + 1):
        free, _ = gdn_gpu.gdn_commit(st0, gdn.h, qkv, gdn.gate, gdn.beta, commit_record(n, 0), c=C, h_v=HV, mt=MT,
                                     n_layers=NL)
        st, _ = gdn_gpu.gdn_commit(st0, h, qkv, gdn.gate, gdn.beta, commit_record(n, 0), c=C, h_v=HV, mt=MT,
                                   n_layers=NL)
        bad = n_bad(free.reshape(NL, GDN_FLOATS)[:, :REC])
        assert bad == 0, (f"{n} rows without the NaN: {bad} non-finite state values, so the NaN's column proves "
                          f"nothing ({CRIT9})")
        want = free.view(U32).copy()
        want[col] = 0x7FFFFFFF
        assert_bits(f"{n} rows, a NaN in row 0's v at layer {l}, head {hd}, column {j}: the state and history against "
                    f"the NaN-free run's with that column 0x7fffffff", st, want, CRIT9)
        for k in range(NL):
            hist = conv_tail(layer(st0, k)[REC:], qkv[k * MT * C:(k * MT + n) * C])
            slots = hist.view(U32).reshape(C, 3)
            assert all(np.isin(SPECIALS, slots[:, s]).all() for s in range(3)), (
                f"{n} rows, layer {k}: a slot of the expected history lacks a SPECIALS word, so the copy's check has "
                f"no power ({CRIT9})")
            assert_bits(f"{n} rows, layer {k}: the history against its word copy", layer(st, k)[REC:], hist, CRIT9)
