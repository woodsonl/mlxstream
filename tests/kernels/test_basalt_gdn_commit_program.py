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
"""

import numpy as np

from monolith.compiler import emit_program
from monolith.core import DType, Graph
from monolith.runtime import Engine
from tests.basalt_gdn_ref import randn
from tests.contract.test_basalt_gdn_commit_ops import C, HV, MT, STATE, STORES, commit_op, spec
from tests.kernels import basalt_gdn_gpu as gdn_gpu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_dec_commit import assert_bits, commit_record
from tests.kernels.test_basalt_sampler_program import PROF

NL = 2                           # GDN layers in the program
RECORDS = (3, 0, 8, 100)         # n_keep: some rows, none, the window's MT rows, past them (clamped to MT)
ROWS = ((C, 0.1), (C, 1.0), (HV, 0.3), (HV, 0.3))     # win_h, win_qkv, win_gate, win_beta: (width, sd), make_gdn's
CRIT1 = ("Each layer's state must equal `basalt_gdn_gpu.gdn_commit` on the same bytes, bitwise, and `gdn.{i}.commit` "
         "must hold the clamped n")


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
