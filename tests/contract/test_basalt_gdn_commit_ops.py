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
- CT.3: a commit program of Flash-Next's 36 GDN layers, all reading the one record, has a barrier before its first
  dispatch only: each layer's dispatch writes its own state and output.
- CT.4: the refusals, each a valid commit (or record) with exactly one defect refused by that defect's own check (its
  message): the handler's (a dynamic-T program, the input count, the roles, ``updates``, each dtype, the shapes, and
  MT = 9 and C - 128 HV, which reach the packer's checks) and the packer's.
- The packer (monolith/kernels.py ``basalt_gdn_commit_params`` and ``basalt_gdn_commit_grid``): the record's 48 bytes at
  the contract's shapes and at basalt's multi-layer strides, and the threadgroup count.
"""

import re
import struct

import pytest

from monolith import kernels
from monolith.compiler import check_coverage, emit_program
from monolith.compiler.emit import HANDLERS
from monolith.core import BlockDomain, DType, Graph, OpClass
from monolith.ops import OPS
from tests.contract.test_basalt_gdn_ops import declare
from tests.contract.test_basalt_ops import spliced
from tests.contract.test_basalt_windows import PROFILE

C, HV, MT = 10240, 48, 8          # Flash-Next's GDN layer (h_k = 16) and the contract's window rows (D18)
STATE = 817152                    # a layer's 128 x 48 x 128 state then its 10240 x 3 history (D17; dcp's GDN_FLOATS)
RECORD = "<12I"                   # GdnCommitParams (basalt_gdn_commit.metal)
STORES = ("win_h", "win_qkv", "win_gate", "win_beta")      # the window's stores in the commit's slot order (1 to 4)


def spec(i=0, mt=MT, c=C):
    """Layer i's commit as test_basalt_gdn_ops ``declare`` takes it, ``{name: (Graph constructor, shape, dtype)}`` in
    slot order: ``gdn.{i}.state``, dense for c channels and HV value heads, and its four stores of mt rows (graph
    states), the record ``commit`` of 2 + mt words (a graph input), then ``gdn.{i}.commit``, the n used."""
    return {f"gdn.{i}.state": ("state", (128 * 128 * HV + 3 * c,), "f32"),
            **{f"gdn.{i}.{s}": ("state", (mt, w), "f32") for s, w in zip(STORES, (c, c, HV, HV))},
            "commit": ("input", (2 + mt,), "i32"), f"gdn.{i}.commit": ("value", (1,), "u32")}


def commit_op(g, sp=None, made=None, updates=True):
    """``basalt_gdn_commit`` over the values of ``sp`` (``spec()`` unless given; a None entry left out; ``made``: values
    already in the graph, by name), updating its state unless ``updates`` is False (the attr left out)."""
    *ins, used = declare(g, spec() if sp is None else sp, made)
    g.op("basalt_gdn_commit", ins, [used], domain=BlockDomain("heads", 1), klass=OpClass.MAP,
         **({"updates": [ins[0].name]} if updates else {}))
    return [*ins, used]


# ---- CT.1 to CT.3: registration, the dispatch against the builders, the barriers -----------------------------------

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


def test_layers_commit_behind_one_barrier():
    """CT.3: Flash-Next's 36 GDN layers committed by one program, every layer's op reading the one record: the barrier
    pass sets ``barrier_before`` on the first dispatch (every program's first op waits) and on none of the 35 after
    it, since each layer's dispatch writes only its own state and output."""
    g = Graph("commit")
    made = {"commit": g.input("commit", (2 + MT,), DType.I32)}
    for i in range(36):
        commit_op(g, spec(i), made)
    prog = emit_program(g, pack=[], profile=PROFILE, t=1, tail=None)
    assert [o.bindings[0][1] for o in prog.ops] == [f"gdn.{i}.state" for i in range(36)]
    assert [o.barrier_before for o in prog.ops] == [True] + [False] * 35


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


def emit_commit(mt=MT, c=C, values=None, updates=True, dynamic_t=False):
    """Layer 0's commit over ``spec(0, mt, c)`` with ``values`` replacing its entries, emitted alone."""
    g = Graph("commit")
    commit_op(g, {**spec(0, mt, c), **(values or {})}, updates=updates)
    return emit_program(g, pack=[], profile=PROFILE, t=1, tail=None, dynamic_t=dynamic_t)


def _handler(bad):
    """The commit emitted at the contract's shapes (valid), or with ``bad``, keywords of ``emit_commit`` (the one
    defect)."""
    return lambda defect: emit_commit(**(bad if defect else {}))


def _packer(bad, good=None):
    """The packer at the contract's shapes with the keywords ``good`` (valid: none, or each one step inside its
    limit), or with ``bad`` instead (the one defect)."""
    return lambda defect: kernels.basalt_gdn_commit_params(
        **{"c": C, "h_v": HV, "mt": MT, **(bad if defect else (good or {}))})


VALUES = spec()
BAD_TYPE = {"f32": "bf16", "i32": "u32", "u32": "i32"}

# (id, build, why): build(False) must pass and build(True) raise a ValueError whose message holds why, literally
REFUSALS = [
    # the handler (compiler/emit_basalt.py ``_gdn_commit``) in its checks' order; MT and C reach the packer's checks
    ("handler dynamic-T", _handler({"dynamic_t": True}),
     "the commit's count is word 0 of its host-written record, not StepState's; a dynamic-T program cannot run it"),
    ("handler no record", _handler({"values": {"commit": None}}),
     "5 inputs and 1 outputs; the commit reads 6 (the state, win_h, win_qkv, win_gate, win_beta, the record)"),
    *[(f"handler {name} not a graph state", _handler({"values": {name: ("value", VALUES[name][1], "f32")}}),
       f"{name!r} is not a graph state; the state and the window stores are a session's buffers")
      for name in [f"gdn.0.{s}" for s in STORES]],
    ("handler record not a graph input", _handler({"values": {"commit": ("value", (2 + MT,), "i32")}}),
     "'commit' is not a graph input; the record is the host's, written once it has chosen n_keep"),
    ("handler updates absent", _handler({"updates": False}),
     "updates names []; the commit writes exactly the state 'gdn.0.state'"),
    *[(f"handler {name} as {BAD_TYPE[ty]}", _handler({"values": {name: (kind, shape, BAD_TYPE[ty])}}),
       f"{name!r} is {BAD_TYPE[ty]} {list(shape)}, not {ty} {list(shape)}")
      for name, (kind, shape, ty) in VALUES.items()],
    ("handler win_qkv rows not win_h's", _handler({"values": {"gdn.0.win_qkv": ("state", (MT - 1, C), "f32")}}),
     "'gdn.0.win_qkv' is f32 [7, 10240], not f32 [8, 10240] (the window's qkv rows; MT and C from 'gdn.0.win_h')"),
    ("handler win_gate width not win_beta's", _handler({"values": {"gdn.0.win_gate": ("state", (MT, HV - 1), "f32")}}),
     "'gdn.0.win_beta' is f32 [8, 48], not f32 [8, 47] (each value head's beta per row; MT from 'gdn.0.win_h', HV "
     "from 'gdn.0.win_gate')"),
    ("handler state without its history", _handler({"values": {"gdn.0.state": ("state", (128 * 128 * HV,), "f32")}}),
     "'gdn.0.state' is f32 [786432], not f32 [817152] (the recurrent state, then the conv history; HV from "
     "'gdn.0.win_gate', C from 'gdn.0.win_h')"),
    ("handler record a word short", _handler({"values": {"commit": ("input", (1 + MT,), "i32")}}),
     "'commit' is i32 [9], not i32 [10] (the record: n_keep, n_keep - 1, then a position per window row)"),
    ("handler MT of 9", _handler({"mt": 9}), "mt must be 1 to 8, got 9"),
    ("handler C - 128 HV not whole 256s", _handler({"c": C + 128}),
     "C - 128 * h_v must be a positive multiple of 256, got 4224 (C 10368, h_v 48)"),
    # the packer alone
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
