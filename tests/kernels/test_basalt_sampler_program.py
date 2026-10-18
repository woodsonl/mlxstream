# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp and src/kernels/decode_cluster_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures, pre-fills and pass criteria.
"""basalt's sampler as an emitted program (the ``basalt_sample`` op kind, the emitter's ``_basalt_sample``): a
hand-built graph, ``logits`` (and ``history``) in, ``basalt_sample``, ``token`` out, emitted by ``emit_program`` at
the engine's t = 8 or at the test's T, its ICB encoded once by ``Engine`` and replayed, with ``Engine.buffers``
written between runs: the request's record through the packer (the build's structure kept), the counter base in
``ctr_base``, the seed and rows in StepState. Every replay starts with ``token`` and ``qprob`` as 0xFF bytes; the
grid's rows past ``t_this_step`` must keep them, and ``qprob`` everywhere (no request sets PROBE). The cluster pair
writes ``token`` before the gated dispatches, so each sampled check first asserts that its data tells the sampled
pick from the cluster's.

- M18.1, fixture 18 (a) (sampler_parity.cpp:1537-1593, basalt's captured graph): the three penalised rows of 248,320
  logits, recorded once and replayed twice on each sampled path the build chooses (the split at t = 64, the
  one-block kernel at t = 70), every pick mirror_pick's.
- DC.1 with A6, the draw counter (:1336-1363): 32 one-row steps of a T = 1 program, the advance moving the position,
  against one 32-row step of a T = 32 program, twice, at the counter 2^32 + 7 (ctr_base's two words); the picks
  vary and are basalt's host list. Also on the one-block path, forced on the same shapes (F.2).
- DCP.2, decode_cluster_parity's graph case, argmax side (:245-306): one program of 5 rows at basalt's vocabulary
  (built at t = 5 and at t = 8), replayed 6 times over fresh logits written into the same buffer, against the
  one-block greedy kernel on the same logits.
- The greedy-only build (attr ``greedy_only``, the cluster pair alone), the port's: its picks equal the full build's in
  mode 0 and the host argmax, at t = 1 and 8.
- MG.1, the port's: one penalised request in each mode, the record's ``mode`` deciding which dispatches run: mode 0
  the cluster pick, mode 1 the one-block greedy pick, mode 2 the sampled pick; the scratch of a dispatch its mode
  gates off keeps its 0xFF bytes. On the split and on the one-block path (forced).
"""

import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler import emit_program
from monolith.core import BlockDomain, DType, Graph, OpClass, T
from monolith.core.profile import Profile
from monolith.core.step_state import StepStateLayout
from monolith.runtime import Engine
from tests.basalt_sampler_ref import (M, SamplerParams, fixture18_data, mirror_pick, mirror_select, penal_vec,
                                      sampled_k, window_of)
from tests.kernels import basalt_sampler_gpu as gpu
from tests.kernels.basalt_gpu import device
from tests.kernels.test_basalt_decode_cluster import NEUTRAL, fill_logits, host_argmax

PROF = Profile.from_dict("p", {"gpu_cores": 40, "nominal_gbps": 400.0, "engine": {
    "family": "Apple9", "lane_order": "interleaved16", "threadgroups_per_core": 2}})
V = 248320                                     # basalt's vocabulary: 61 blocks of 4,096 logits
CACHE = {}                                     # the pipelines, shared by the programs of one StepState layout
SPLIT = ["bs_greedy_partial", "bs_greedy_merge", "bs_greedy_one_block", "bs_split_part", "bs_split_merge"]
ONE_BLOCK = SPLIT[:3] + ["bs_sample_one_block"]


def program(t, vocab=V, *, hist_len=0, advance=False, forced=False, greedy_only=False):
    """The graph emitted at ``t`` (StepState for t_max = max(8, t)) and instantiated: ``history`` (T, hist_len) when
    hist_len > 0; ``advance``, the step's advance on ``token`` as the program's tail; ``forced``, the one-block path
    whatever the shape (F.2: the build's path choice overridden, the handler's dispatches otherwise);
    ``greedy_only``, the op's greedy-only build."""
    g = Graph()
    ins = [g.input("logits", (T, vocab), DType.F32), g.input("ctr_base", (2,), DType.U32),
           g.input("nq", (1,), DType.U32)]
    if hist_len:
        ins.append(g.input("history", (T, hist_len), DType.I32))
    token, qprob = g.value("token", (T,), DType.I32), g.value("qprob", (T,), DType.F32)
    g.op("basalt_sample", ins, [token, qprob], domain=BlockDomain("span", 1), klass=OpClass.REDUCE, hist_len=hist_len,
         **(dict(greedy_only=True) if greedy_only else {}))
    with pytest.MonkeyPatch.context() as mp:
        if forced:
            mp.setattr(kernels, "basalt_sampler_path", lambda t_max, vocab: "one_block")
        prog = emit_program(g, pack=[], profile=PROF, t=t, layout=StepStateLayout(t_max=max(8, t)),
                            tail="advance" if advance else None, token=token)
    return Engine(prog, device(), pipeline_cache=CACHE)


def functions(eng):
    return [eng.program.kernels[o.kernel].function for o in eng.program.ops]


def binding(eng, function, slot):
    """The buffer that ``function``'s dispatch binds at ``slot``."""
    o = next(o for o in eng.program.ops if eng.program.kernels[o.kernel].function == function)
    return next(name for index, name, _ in o.bindings if index == slot)


def state(eng, **fields):
    """StepState's ``fields`` set in the live buffer, every other field (the ring's counters among them) kept."""
    st = eng.state()
    st.update(fields)
    eng.buffers[eng.program.step_state].write(eng.program.layout.pack(st), 0)


def request(eng, p, mode=None):
    """Request ``p`` (SamplerParams) as its writer sets it: the op's record repacked by the packer in ``mode`` (the
    mode rule's when None) on the build's structure (vocab, rows, blocks, history width, asserted kept), the counter
    base in ``ctr_base``, low word first, and the seed in StepState."""
    name = binding(eng, "bs_greedy_partial", 7)
    build = eng.program.buffers[name].init
    vocab, rows, _, hist_len = struct.unpack_from("<IIII", build)
    if mode is None:
        mode = kernels.basalt_sampler_mode(p.greedy, p.temperature, p.penalty_last_n, False)
    rec = gpu.record(p, vocab, rows, hist_len, mode)
    assert rec[:16] == build[:16], "the request's record changed the program's structure"
    eng.buffers[name].write(rec, 0)
    eng.buffers["ctr_base"].write(struct.pack("<II", p.counter & M, p.counter >> 32 & M), 0)
    state(eng, rng_lo=p.seed & M, rng_hi=p.seed >> 32 & M)


def step(eng, rows):
    """One replay of the encoded step, ``token`` and ``qprob`` refilled with 0xFF bytes first. Returns ``token``'s
    first ``rows`` rows and the run's report; asserts every later row of the buffer -1 and ``qprob`` untouched. A
    greedy-only program binds no ``qprob`` (no dispatch writes it), so it has no such buffer."""
    probe = "qprob" in eng.buffers
    for name in ("token", "qprob") if probe else ("token",):
        eng.buffers[name].fill(0xFF)
    r = eng.run(1, steps_per_cb=1, in_flight=1)
    assert r.steps == 1 and not r.done, r
    tok = np.frombuffer(eng.read("token"), np.int32)
    assert np.all(tok[rows:] == -1), "a row past t_this_step was written"
    if probe:
        assert np.all(np.frombuffer(eng.read("qprob"), np.uint32) == 0xFFFFFFFF), "qprob was written without the probe"
    return tok[:rows].copy(), r


# ---- M18.1: fixture 18 (a), the captured graph (sampler_parity.cpp:1537-1593) ----------------------------------------

@pytest.mark.parametrize("t,path", [(64, SPLIT), (70, ONE_BLOCK)], ids=["split-t64", "one_block-t70"])
def test_basalt_sampler_replayed_one_block(t, path):
    """M18.1 (:1586-1593): fixture 18 (a)'s three rows of 248,320 half steps and 64-slot windows, the request top_k
    20, top_p 0.9, temperature 2.5, seed 18, counter 77 under repeat 1.25, freq 0.25 and present 0.5, in a program
    built at t (basalt's capture; 64, the split's widest window, and 70, past it, which takes the one-block kernel),
    recorded once and replayed twice, the scratch left as the first replay left it: every replayed pick is
    mirror_pick's."""
    rows, H = 3, 64
    l, hist = fixture18_data(V, rows, 1800)
    p = SamplerParams(top_k=20, top_p=0.9, temperature=2.5, seed=18, counter=77, penalty_last_n=H, penalty_repeat=1.25,
                      penalty_freq=0.25, penalty_present=0.5)
    k = sampled_k(p.top_k, V)
    want = np.int32([mirror_pick(mirror_select(l[r], V, window_of(hist, H, r, H), p, k), k, p, r) for r in range(rows)])
    assert any(want[r] != host_argmax(l[r]) for r in range(rows)), "the mirror picks the cluster's argmax in every row"
    eng = program(t, hist_len=H)
    assert functions(eng) == path
    eng.buffers["logits"].write(l.tobytes(), 0)
    eng.buffers["history"].write(hist.tobytes(), 0)
    request(eng, p)
    state(eng, t_this_step=rows)
    wrong = [(replay, r) for replay in range(2) for r in np.flatnonzero(step(eng, rows)[0] != want)]
    assert not wrong, f'compare("captured graph", got, lists, k, p); ((replay, row): {wrong})'


# ---- DC.1 + A6: the draw counter (:1336-1363) ------------------------------------------------------------------------

# basalt's host list for the draw-counter rows (spec T§8.17): floor(16 u) of row t's draw at seed 123, counter
# 2^32 + 7 + t
DC_HOST = [12, 12, 11, 11, 1, 6, 5, 2, 15, 10, 15, 5, 5, 12, 5, 7,
           15, 13, 9, 13, 11, 7, 0, 9, 15, 5, 8, 12, 4, 15, 2, 13]


@pytest.mark.parametrize("forced", [False, True], ids=["split", "one_block-F"])
def test_basalt_sampler_draw_counter_segmentation(forced):
    """DC.1 (:1336-1361) with A6: 16 logits of 0, top_k 16, top_p 1, temperature 1, seed 123 and the counter
    2^32 + 7 in ``ctr_base`` (a high word the position cannot carry). ``singles``: a T = 1 program with the advance,
    32 steps, each one replay that publishes its row to the ring and moves the position by one (row 0 draws at
    ctr_base + position); ``batch`` and ``repeated``: one T = 32 program at position 0, two replays. The three agree,
    the picks vary, and the batch is basalt's host list (A6: a kernel that dropped the counter's high word agrees with
    itself). ``forced``: both programs on the one-block path (F.2)."""
    n, nv = 32, 16
    p = SamplerParams(top_k=nv, top_p=1.0, seed=123, counter=(1 << 32) + 7)
    sel = mirror_select(np.zeros(nv, np.float32), nv, [], p, nv)
    assert [mirror_pick(sel, nv, p, t) for t in range(n)] == DC_HOST, "spec T§8.17: the host reference's list"
    for t in (1, n):
        eng = program(t, nv, advance=t == 1, forced=forced)
        assert functions(eng) == (ONE_BLOCK if forced else SPLIT) + (["advance"] if t == 1 else [])
        assert not forced or not any(".bs.cand" in name for name in eng.program.buffers)
        eng.buffers["logits"].write(np.zeros((t, nv), np.float32).tobytes(), 0)
        request(eng, p)
        if t == 1:
            singles = []
            for i in range(n):
                _, r = step(eng, 1)
                assert len(r.tokens) == 1 and eng.state()["position"] == i + 1, (i, r.tokens, eng.state()["position"])
                singles += r.tokens
        else:
            batch, repeated = (step(eng, n)[0].tolist() for _ in range(2))
    varies = any(b != batch[0] for b in batch)
    valid = batch == singles and batch == repeated and varies
    assert valid, ("const bool valid = batch == singles && batch == repeated && varies; ... bad += !valid; "
                   f"(batch {batch}, singles {singles}, repeated {repeated})")
    assert batch == DC_HOST, f"A6: DC.1's batch against the host list (batch {batch})"


# ---- DCP.2: decode_cluster_parity's graph case, argmax side (:245-306) -----------------------------------------------

@pytest.mark.parametrize("t", (5, 8))
def test_basalt_decode_cluster_graph_replay(t):
    """DCP.2 (:245-306): nq = 5 rows of 248,320 logits in a program built at t (5, the window's own; 8, the engine's)
    with the build's record (mode 0: the cluster pair alone), encoded once and replayed 6 times, each replay over
    fresh logits written into the same buffer (fill_logits' random pattern, then ties-across, alternately; the
    port's own seeded draws), against the one-block greedy kernel (``greedy_1b`` through basalt's neutral window,
    :45-62) on the same logits."""
    nq, rs = 5, np.random.RandomState(20261002)
    eng = program(t)
    assert functions(eng) == SPLIT
    state(eng, t_this_step=nq)
    fails = []
    for rep in range(6):
        l = np.stack([fill_logits(rs, V, "ties-across" if rep % 2 else "random") for _ in range(nq)])
        eng.buffers["logits"].write(l.tobytes(), 0)
        u = step(eng, nq)[0]
        v = gpu.greedy_1b(l, NEUTRAL, hist=np.full(nq, -1, np.int32), hist_len=1, fill=0x5A)
        if d2 := int(np.count_nonzero(u != v)):
            fails.append((rep, d2))
    assert not fails, f"for (size_t i = 0; i < u.size(); ++i) d2 += u[i] != v[i]; if (d1 || d2) ++fails; ({fails})"


@pytest.mark.parametrize("t", (1, 8))
def test_basalt_sampler_greedy_only_program(t):
    """The greedy-only build (attr greedy_only: the cluster pair alone) against the full build in mode 0, both at
    t: over two replays of fresh logits (fill_logits' random pattern, then ties-across) each pick is the full
    program's and the host argmax (the lowest index of the largest non-NaN value above -inf); the full program's
    qprob keeps its 0xFF bytes, and the greedy-only one has no qprob buffer."""
    rs = np.random.RandomState([20261018, t])
    only, full = program(t, greedy_only=True), program(t)
    assert functions(only) == SPLIT[:2] and "qprob" not in only.buffers and "qprob" in full.buffers
    for eng in (only, full):
        state(eng, t_this_step=t)
    for rep in range(2):
        l = np.stack([fill_logits(rs, V, "ties-across" if rep else "random") for _ in range(t)])
        picks = []
        for eng in (only, full):
            eng.buffers["logits"].write(l.tobytes(), 0)
            picks.append(step(eng, t)[0].tolist())
        want = [host_argmax(row) for row in l]
        assert picks[0] == picks[1] == want, f"replay {rep}: greedy_only {picks[0]}, full {picks[1]}, host {want}"


# ---- MG.1: the mode word gates the dispatches ------------------------------------------------------------------------

@pytest.mark.parametrize("forced", [False, True], ids=["split", "one_block-F"])
def test_basalt_sampler_mode_selects_path(forced):
    """MG.1: 8 rows of 248,320 normal logits, each with 20.0 at one id, the only one in its 4-slot history, and 19.0
    at four others; the request top_k 4, top_p 1, temperature 1 under repeat 2 (20 falls to 10, out of the list).
    Its record written in each mode: mode 0 picks the 20.0 (the cluster pair alone), mode 1 the lowest of the 19.0s
    (the one-block greedy kernel's penalised argmax), mode 2 the draw among the four 19.0s (the sampled path,
    mirror_pick's). A scratch whose dispatches the mode gates off keeps its 0xFF bytes: ``cand`` (the split's
    lists) in modes 0 and 1, the bitmap (the one-block kernels') in mode 0, and on the split in mode 2."""
    rows, H = 8, 4
    rs = np.random.RandomState(20261018)
    l = rs.standard_normal((rows, V)).astype(np.float32)
    hist = np.full(rows * H, -1, np.int32)
    for r in range(rows):
        ids = rs.choice(V, 5, replace=False)
        l[r, ids[0]], l[r, ids[1:]], hist[r * H] = 20.0, 19.0, ids[0]
    p = SamplerParams(top_k=4, top_p=1.0, temperature=1.0, seed=181, counter=5, penalty_last_n=H, penalty_repeat=2.0)
    win = [window_of(hist, H, r, H) for r in range(rows)]
    want = [[host_argmax(l[r]) for r in range(rows)], [host_argmax(penal_vec(l[r], win[r], p)) for r in range(rows)],
            [mirror_pick(mirror_select(l[r], V, win[r], p, 4), 4, p, r) for r in range(rows)]]
    assert all(want[0][r] != want[1][r] for r in range(rows)) and \
        any(want[2][r] not in (want[0][r], want[1][r]) for r in range(rows)), "each mode's pick tells its path apart"
    eng = program(8, hist_len=H, forced=forced)
    assert functions(eng) == (ONE_BLOCK if forced else SPLIT)
    eng.buffers["logits"].write(l.tobytes(), 0)
    eng.buffers["history"].write(hist.tobytes(), 0)
    gated = {binding(eng, "bs_greedy_one_block", 10): (0,) if forced else (0, 2)}
    if not forced:
        gated[binding(eng, "bs_split_part", 9)] = (0, 1)
    for mode in (0, 1, 2):
        request(eng, p, mode)
        for name in gated:
            eng.buffers[name].fill(0xFF)
        got = step(eng, rows)[0].tolist()
        assert got == want[mode], f"mode {mode} returns its path's pick: got {got}, want {want[mode]}"
        for name, modes in gated.items():
            if mode in modes:
                assert set(eng.read(name)) == {0xFF}, f"mode {mode} ran a dispatch that writes {name}"
