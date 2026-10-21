"""Host-only checks of basalt's GDN commit; no GPU.

The op (monolith/ops/basalt_gdn.py) as compiler/emit_basalt.py's ``_gdn_commit`` emits it, in a program of no weights;
the GPU side is tests/kernels/test_basalt_dec_commit.py.

- CT.1: ``basalt_gdn_commit`` registered (MAP over heads, its binding the kind on both families), its handler, and the
  coverage guard.
- CT.2: at Flash-Next's layer (C = 10240, HV = 48, MT = 8), one dispatch of ``gdn_commit_layers`` with the builder's
  source and no macros, grid (272, 1, 1) of 128 threads; input i at slot i and ``used`` at 6, each at byte 0 of its own
  buffer, the program's own params record at 7 (``kernels.basalt_gdn_commit_params``) and no StepState at 15;
  ``writes`` [0, 6]; the state and the stores states of their sizes, the record ``commit`` a host-written 40-byte arena
  input bound as itself, never as StepState.
- The packer (monolith/kernels.py ``basalt_gdn_commit_params`` and ``basalt_gdn_commit_grid``): the record's 48 bytes at
  the contract's shapes and at basalt's multi-layer strides, the threadgroup count, and the refusals, each a valid
  input with exactly one defect refused by that defect's own check (its message).
"""

import re
import struct

import pytest

from monolith import kernels
from monolith.compiler import check_coverage, emit_program
from monolith.compiler.emit import HANDLERS
from monolith.core import BlockDomain, DType, Graph, OpClass
from monolith.ops import OPS
from tests.contract.test_basalt_ops import spliced
from tests.contract.test_basalt_windows import PROFILE

C, HV, MT = 10240, 48, 8          # Flash-Next's GDN layer (h_k = 16) and the contract's window rows (D18)
STATE = 817152                    # a layer's 128 x 48 x 128 state then its 10240 x 3 history (D17; dcp's GDN_FLOATS)
RECORD = "<12I"                   # GdnCommitParams (basalt_gdn_commit.metal)
STORES = ("win_h", "win_qkv", "win_gate", "win_beta")      # the window's stores in the commit's slot order (1 to 4)


def commit_op(g):
    """``basalt_gdn_commit`` of layer 0 at the contract's shapes: ``gdn.0.state`` and its four stores (graph states),
    the record ``commit`` (a graph input), in slot order, then ``gdn.0.commit``, the n used."""
    ins = [g.state("gdn.0.state", (STATE,), DType.F32),
           *[g.state(f"gdn.0.{s}", (MT, w), DType.F32) for s, w in zip(STORES, (C, C, HV, HV))],
           g.input("commit", (2 + MT,), DType.I32)]
    used = g.value("gdn.0.commit", (1,), DType.U32)
    g.op("basalt_gdn_commit", ins, [used], domain=BlockDomain("heads", 1), klass=OpClass.MAP, updates=[ins[0].name])
    return [*ins, used]


# ---- CT.1 and CT.2: registration, and the dispatch against the builders --------------------------------------------

def test_registered_and_covered():
    od = OPS.get("basalt_gdn_commit")
    assert od.klass is OpClass.MAP and od.domain_kind == "heads"
    assert od.binding_for("apple9").kernel == od.binding_for("apple10").kernel == "basalt_gdn_commit"
    assert HANDLERS["basalt_gdn_commit"].__name__ == "_gdn_commit"
    g = Graph("commit")
    commit_op(g)
    check_coverage(g, PROFILE)


def test_commit_dispatch_is_the_builders():
    """One dispatch of ``gdn_commit_layers``: the builder's source, no macros, grid (272, 1, 1) of 128 threads (one
    threadgroup per item); the state, win_h, win_qkv, win_gate, win_beta, the record and used at slots 0 to 6, each at
    byte 0 of its own buffer, the record ``basalt_gdn_commit_params`` of the shapes at 7, nothing at 15 (the record
    carries the count, no StepState); the commit writes slots 0 and 6. The state and the stores are states of their
    sizes; the record ``commit`` is a 40-byte arena buffer of its own name, as no StepState field is named commit."""
    g = Graph("commit")
    vals = commit_op(g)
    prog = emit_program(g, pack=[], profile=PROFILE, t=1, tail=None)
    (o,) = prog.ops
    k = prog.kernels[o.kernel]
    assert (k.function, k.macros, k.source) == ("gdn_commit_layers", {}, spliced(kernels.basalt_gdn_commit_source()))
    assert (o.name, o.grid, o.threadgroup, o.threadgroup_memory) == ("basalt_gdn_commit", (272, 1, 1), (128, 1, 1), [])
    assert [b[0] for b in o.bindings] == list(range(8)) and o.meta["writes"] == [0, 6]
    assert [(name, off) for _, name, off in o.bindings[:7]] == [(v.name, 0) for v in vals]
    _, prm, off = o.bindings[7]
    rec = prog.buffers[prm]
    assert (prm, off, rec.role, len(rec.init)) == ("params.S1.basalt_gdn_commit.0", 0, "params", 48)
    assert struct.unpack_from(RECORD, rec.init) == struct.unpack(RECORD, kernels.basalt_gdn_commit_params(C, HV, MT))
    sizes = {"gdn.0.state": ("state", 4 * STATE), "gdn.0.win_h": ("state", 4 * MT * C),
             "gdn.0.win_qkv": ("state", 4 * MT * C), "gdn.0.win_gate": ("state", 4 * MT * HV),
             "gdn.0.win_beta": ("state", 4 * MT * HV), "commit": ("arena", 40), "gdn.0.commit": ("arena", 16)}
    assert {n: (prog.buffers[n].role, prog.buffers[n].nbytes) for n in sizes} == sizes
    assert "commit" not in prog.layout.offsets and prog.step_state not in {name for _, name, _ in o.bindings}


def test_basalt_gdn_commit_params_record_and_grid():
    """The record's fields in GdnCommitParams' order (state_stride, h_stride, gb_stride, C, h_k, h_v, n_layers,
    conv_blocks, mt, three zero words): the dense defaults at the contract's shapes and at the smallest layer (one key
    and one value head, one row); dec_commit_parity.cpp:83-89's three layers at its MT = 7, strides given; and
    dec_gdn_parity.cpp:519-521's form, two layers whose rows sit in a wider window (20 rows here), packed as given. The
    threadgroup count is one per item, 272 a layer."""
    rec = kernels.basalt_gdn_commit_params(C, HV, MT)
    assert len(rec) == 48 and struct.unpack(RECORD, rec) == (817152, 81920, 384, 10240, 16, 48, 1, 80, 8, 0, 0, 0)
    small = kernels.basalt_gdn_commit_params(384, 1, 1)
    assert struct.unpack(RECORD, small) == (17536, 384, 1, 384, 1, 1, 1, 3, 1, 0, 0, 0)
    dcp = kernels.basalt_gdn_commit_params(C, HV, 7, n_layers=3, state_stride=STATE, h_stride=7 * C, gb_stride=7 * HV)
    assert struct.unpack(RECORD, dcp) == (STATE, 71680, 336, C, 16, HV, 3, 80, 7, 0, 0, 0)
    dgp = kernels.basalt_gdn_commit_params(C, HV, MT, n_layers=2, state_stride=STATE, h_stride=20 * C,
                                           gb_stride=20 * HV)
    assert struct.unpack(RECORD, dgp) == (STATE, 204800, 960, C, 16, HV, 2, 80, MT, 0, 0, 0)
    assert kernels.basalt_gdn_commit_grid(C, HV) == 272 and kernels.basalt_gdn_commit_grid(C, HV, 36) == 9792


def _packer(bad, good=None):
    """The packer at the contract's shapes with the keywords ``good`` (valid: none, or each one step inside its
    limit), or with ``bad`` instead (the one defect)."""
    return lambda defect: kernels.basalt_gdn_commit_params(
        **{"c": C, "h_v": HV, "mt": MT, **(bad if defect else (good or {}))})


# (id, build, why): build(False) must pass and build(True) raise a ValueError whose message holds why, literally
REFUSALS = [
    ("packer h_v of 0", _packer({"h_v": 0}), "h_v must be at least 1, got 0"),
    ("packer C - 128 h_v of 0", _packer({"c": 128 * HV}),
     "C - 128 * h_v must be a positive multiple of 256, got 0 (C 6144, h_v 48)"),
    ("packer C - 128 h_v not whole 256s", _packer({"c": C + 128}),
     "C - 128 * h_v must be a positive multiple of 256, got 4224 (C 10368, h_v 48)"),
    ("packer mt of 0", _packer({"mt": 0}, {"mt": 1}), "mt must be 1 to 8, got 0"),
    ("packer mt of 9", _packer({"mt": 9}, {"mt": 8}), "mt must be 1 to 8, got 9"),
    ("packer n_layers of 0", _packer({"n_layers": 0}, {"n_layers": 1}), "n_layers must be at least 1, got 0"),
    ("packer state_stride short", _packer({"state_stride": STATE - 1}, {"state_stride": STATE}),
     "state_stride 817151 is below its dense value 817152"),
    ("packer h_stride short", _packer({"h_stride": 81919}, {"h_stride": 81920}),
     "h_stride 81919 is below its dense value 81920"),
    ("packer gb_stride short", _packer({"gb_stride": 383}, {"gb_stride": 384}),
     "gb_stride 383 is below its dense value 384"),
    # 5256 layers of 817,152 floats end at 4,294,950,912 < 2^32; 5257 end at 4,295,768,064
    ("packer n_layers past the uint index", _packer({"n_layers": 5257}, {"n_layers": 5256}),
     "n_layers 5257 times the widest stride 817152 reaches 2^32 floats"),
    ("packer h_stride past the uint index", _packer({"h_stride": 2 ** 32}, {"h_stride": 2 ** 32 - 1}),
     "n_layers 1 times the widest stride 4294967296 reaches 2^32 floats"),
    ("packer gb_stride past the uint index", _packer({"gb_stride": 2 ** 32}, {"gb_stride": 2 ** 32 - 1}),
     "n_layers 1 times the widest stride 4294967296 reaches 2^32 floats"),
]


@pytest.mark.parametrize("build, why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(build, why):
    """CT.4: each case passes without its defect, then its defect is refused by its own check (the message)."""
    build(False)
    with pytest.raises(ValueError, match=re.escape(why)):
        build(True)
