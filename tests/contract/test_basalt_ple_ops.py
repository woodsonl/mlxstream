"""basalt's n-gram block on the decode path (monolith/ops/basalt_ple.py) as compiler/emit_basalt.py emits it: the window
op ``basalt_ple_block`` and the commit op ``basalt_ple_commit``, over the pack's six tensors declared as the op contract
gives them. The GPU side is tests/kernels/test_basalt_ple_program.py.

- REG ``test_registered_and_covered``: both kinds are ``OpClass.MAP``, each bound to the kernel of its own name, and
  the coverage guard passes on the window graph and on the commit graph."""

from monolith.compiler.coverage import check_coverage
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.core.shapes import T
from monolith.ops import OPS
from tests.basalt_ple_ref import D, HIST, KERN, N, T_MAX
from tests.contract.test_basalt_ops import op
from tests.contract.test_basalt_windows import PROFILE

F32 = DType.F32
KEY, VALUE, TAPS = "gguf/blk.1.ple_key.weight", "dense/blk.1.ple_value.weight", "dense/blk.1.ple_conv1d.weight"
NORMS = tuple(f"dense/blk.1.ple_norm_{n}.weight" for n in ("key", "query", "conv"))


# ---- graph builders ----------------------------------------------------------------------------------------------

def states(g):
    """``hist`` and ``snap``: one size in every program, the windows' and the commit's, so they are shared by name."""
    return g.state("ngram.hist", (D, HIST), F32), g.state("ngram.snap", (T_MAX, D, HIST), F32)


def block(g):
    """The window: ``basalt_ple_block`` over the T rows of ``ngram.rows`` and ``ngram.hidden``, with the key, the
    value, the three norms and the taps declared as they bind (``gguf/`` a weight, ``dense/`` a constant)."""
    hist, snap = states(g)
    ins = [g.input("ngram.rows", (T, N), F32), g.input("ngram.hidden", (T, D), F32), g.weight(KEY, (D, N), "q2_0"),
           g.const(VALUE, (N, N), DType.BF16), *(g.const(n, (D,), F32) for n in NORMS),
           g.const(TAPS, (D, KERN), DType.F16), hist, snap]
    outs = [g.value("ngram.result", (T, D), F32), g.value("ngram.normalized", (T, D), F32)]
    return op(g, "basalt_ple_block", ins, outs, updates=[hist.name, snap.name])


def commit(g):
    """The commit: ``basalt_ple_commit`` restores ``hist`` from the snapshot that word 1 of the record ``commit``
    names, into a placeholder output."""
    hist, snap = states(g)
    record = g.input("commit", (2 + T_MAX,), DType.I32)
    return op(g, "basalt_ple_commit", [hist, snap, record], [g.value("ngram.commit", (1,), DType.U32)],
              updates=[hist.name])


# ---- registration ------------------------------------------------------------------------------------------------

def test_registered_and_covered():
    for build in (block, commit):
        g = Graph("step")
        kind = build(g).kind
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
        assert [o.kind for o in g.ops] == [kind]
        check_coverage(g, PROFILE)
