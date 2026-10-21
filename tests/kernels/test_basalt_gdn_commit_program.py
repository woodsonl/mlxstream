"""basalt's GDN commit as an emitted program (the ``basalt_gdn_commit`` op kind, the emitter's ``_gdn_commit``): a
hand-built graph of no weights emitted by ``emit_program`` at t = 1 without a tail, its ICB encoded once by ``Engine``
and replayed, with ``Engine.buffers`` written between runs. The op alone on the GPU is
tests/kernels/test_basalt_dec_commit.py; its dispatch and refusals are tests/contract/test_basalt_gdn_commit_ops.py.

- PG.1: two GDN layers at Flash-Next's shapes (C = 10240, HV = 48) with the contract's MT = 8 window rows (D18):
  ``gdn.{i}.state`` and the four ``gdn.{i}.win_*`` stores graph states, each layer's own draws in make_gdn's
  distributions (dec_commit_parity.cpp:57-74), written once; the host-written record ``commit`` (2 + MT int32 words,
  D19) the one graph input both ops read; ``gdn.{i}.commit`` each op's output. The program is replayed for the records
  n_keep = 3, 0, 8 and 100, each written into ``commit`` before its run, with every ``gdn.{i}.commit`` refilled with
  0xFF bytes (invariant 4: the count is the record buffer's, never the encoded params record's; D15). The state
  carries from run to run, as a session's does. After each run each layer's state and history equal the single-kernel
  harness (tests/kernels/basalt_gdn_gpu.py ``gdn_commit``) on the same bytes, the state that run started from, bit
  for bit, and are finite; word 0 of ``gdn.{i}.commit`` is the clamped n (3, 0, 8, 8) and its other words keep
  their 0xFF.
- PG.2: the commit program between window programs of one session (contract invariant 1 across the commit; the host
  loop of spec-engine.md §9.4): tests/kernels/test_basalt_gdn_program.py's GDN block programs (``block``: x's native
  q8_1, the pair projection, ``basalt_gdn_window``, the out projection, over parity_gdnf's combo 0 in a synthetic
  .basalt file) at widths 7, 6, 4 and 1 and a one-layer commit program, all emitted with the window programs' profile
  (tests/contract/test_basalt_windows.py ``PROFILE``), their engines sharing the weights, ``gdn.0.state`` and the four
  stores by name (``Engine(buffers=...)``); the layer's state written once and carried from run to run. For n = 1, 3
  and 6: window(7) on x's rows 0..6, the commit of record n, then window(7 - n) on rows n..6, each run started after
  the one before it returned (spec-engine.md Q8), every activation and store 0xFF before each window and
  ``gdn.0.commit`` before the commit. Window(7 - n)'s q8_1 rows (``gdn.0.yq``) and the four stores' rows 0..6 - n
  equal window(7)'s rows n..6 bit for bit, and those are finite (yq: its blocks' d and sum halves); word 0 of
  ``gdn.0.commit`` is n and its other words keep their 0xFF. This is tests/kernels/test_basalt_dec_commit.py's DCM.8
  through the emitted programs, the pair projection's qkv rows (``gdn.0.win_qkv``) included.
"""

import numpy as np

from monolith import kernels
from monolith.compiler import emit_program
from monolith.core import DType, Graph
from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.basalt_gdn_ref import n_bad, randn
from tests.contract.test_basalt_gdn_commit_ops import C, HV, MT, STATE, STORES, commit_op, spec
from tests.contract.test_basalt_windows import PROFILE
from tests.kernels import basalt_gdn_gpu as gdn_gpu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_dec_commit import assert_bits, commit_record
from tests.kernels.test_basalt_gdn_program import ACTS, block, case, write_layer
from tests.kernels.test_basalt_sampler_program import PROF

NL = 2                           # GDN layers in the program
RECORDS = (3, 0, 8, 100)         # n_keep: some rows, none, the window's MT rows, past them (clamped to MT)
ROWS = ((C, 0.1), (C, 1.0), (HV, 0.3), (HV, 0.3))     # win_h, win_qkv, win_gate, win_beta: (width, sd), make_gdn's
CRIT1 = ("Each layer's state must equal `basalt_gdn_gpu.gdn_commit` on the same bytes, bitwise, and `gdn.{i}.commit` "
         "must hold the clamped n")
WIDTHS = (7, 6, 4, 1)            # PG.2's window programs: window(7), then window(7 - n) for n in KEEPS
KEEPS = (1, 3, 6)
WIN = ["gdn.0.state", *(f"gdn.0.{s}" for s in STORES)]     # what the commit program shares with the window programs
YQ = "gdn.0.yq"                  # the window's q8_1 rows (an activation of T rows)
CRIT2 = ("For n in {1, 3, 6}: window(7), then commit(n), then window(7 - n). The second window's q8_1 rows and "
         "`win_*` rows must equal the first's rows n..6 bytewise")


def test_gdn_commit_program_replays_each_record_as_the_harness():
    """PG.1: one encoding of the two-layer commit program replayed for n_keep 3, 0, 8 and 100."""
    g = Graph("commit")
    made = {"commit": g.input("commit", (2 + MT,), DType.I32)}
    for i in range(NL):
        commit_op(g, spec(i), made)
    eng = Engine(emit_program(g, pack=[], profile=PROF, t=1, tail=None), gpu.device())
    rng = np.random.default_rng(20261020)
    rows = []
    for i in range(NL):
        eng.buffers[f"gdn.{i}.state"].write(randn(rng, STATE, 0.1).tobytes(), 0)
        rows.append([randn(rng, (MT, w), sd) for w, sd in ROWS])
        for name, a in zip(STORES, rows[i]):
            eng.buffers[f"gdn.{i}.{name}"].write(a.tobytes(), 0)
    for n in RECORDS:
        rec = commit_record(n, 0, mt=MT)
        eng.buffers["commit"].write(rec.tobytes(), 0)
        before = [np.frombuffer(eng.read(f"gdn.{i}.state"), np.float32) for i in range(NL)]
        for i in range(NL):
            eng.buffers[f"gdn.{i}.commit"].fill(0xFF)
        assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
        for i in range(NL):
            want, used = gdn_gpu.gdn_commit(before[i], *rows[i], rec, c=C, h_v=HV, mt=MT)
            got = np.frombuffer(eng.read(f"gdn.{i}.state"), np.float32)
            assert_bits(f"n_keep {n}: layer {i}'s state and history against the harness", got, want, CRIT1)
            assert np.isfinite(got).all(), f"n_keep {n}: layer {i}: {np.count_nonzero(~np.isfinite(got))} non-finite"
            out = np.frombuffer(eng.read(f"gdn.{i}.commit"), np.uint32)
            assert out[0] == used == min(n, MT) and (out[1:] == 0xFFFFFFFF).all(), \
                f"n_keep {n}: layer {i}'s gdn.{i}.commit holds {[hex(w) for w in out]} ({CRIT1})"


def test_gdn_window_after_the_commit_program_continues_the_longer_window(tmp_path):
    """PG.2: window(7), the commit of n rows, then window(7 - n), for n = 1, 3 and 6, on five engines of one session."""
    layer, x = case()
    src = BasaltSource(write_layer(tmp_path / "gdn.basalt", layer))
    graphs = {t: Graph("step") for t in WIDTHS}
    for g in graphs.values():
        block(g, src)
    progs = {t: emit_program(g, pack=src, profile=PROFILE, t=t, tail=None) for t, g in graphs.items()}
    dev, cache = gpu.device(), {}
    wins = {7: Engine(progs[7], dev, pipeline_cache=cache)}
    shared = {n: b for n, b in wins[7].buffers.items() if progs[7].buffers[n].role in ("weights", "state")}
    wins.update({t: Engine(progs[t], dev, buffers=shared, pipeline_cache=cache) for t in WIDTHS[1:]})
    g = Graph("commit")
    commit_op(g, spec(0), {"commit": g.input("commit", (2 + MT,), DType.I32)})
    com = Engine(emit_program(g, pack=[], profile=PROFILE, t=1, tail=None), dev, buffers=shared, pipeline_cache=cache)
    assert all(e.buffers[n] is shared[n] for e in (*wins.values(), com) for n in WIN), f"{WIN} not shared ({CRIT2})"
    shared[WIN[0]].write(layer.state.tobytes(), 0)

    def window(t, xs):
        """The t-wide program on the rows xs: its q8_1 rows (t) and the four stores (MT rows each), as float32 rows."""
        eng = wins[t]
        eng.buffers["x"].write(xs.tobytes(), 0)
        for name in (*ACTS, *WIN[1:]):
            eng.buffers[name].fill(0xFF)
        assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
        return {name: np.frombuffer(eng.read(name), np.float32).reshape(t if name == YQ else MT, -1)
                for name in (YQ, *WIN[1:])}

    for n in KEEPS:
        a = window(7, x[:7])
        com.buffers["commit"].write(commit_record(n, 0, mt=MT).tobytes(), 0)
        com.buffers["gdn.0.commit"].fill(0xFF)
        assert com.run(1, steps_per_cb=1, in_flight=1).steps == 1
        used = np.frombuffer(com.read("gdn.0.commit"), np.uint32)
        assert used[0] == n and (used[1:] == 0xFFFFFFFF).all(), \
            f"n {n}: gdn.0.commit holds {[hex(w) for w in used]}, want word 0 {n} and the others 0xFF ({CRIT2})"
        b = window(7 - n, x[n:7])
        for name, rows in a.items():
            want = rows[n:7]
            v = (np.ascontiguousarray(want.view(np.uint8).reshape(-1, kernels.Q8_1_BLOCK_BYTES)[:, :4])
                 .view(np.float16) if name == YQ else want)
            assert n_bad(v) == 0, f"n {n}: {n_bad(v)} non-finite values in window(7)'s {name} rows {n}..6 ({CRIT2})"
            assert_bits(f"n {n}: window({7 - n})'s {name} rows 0..{6 - n} against window(7)'s rows {n}..6",
                        b[name][:7 - n], want, CRIT2)
