"""basalt's n-gram block on the decode path (monolith/ops/basalt_ple.py) as compiler/emit_basalt.py emits it: the window
op ``basalt_ple_block`` and the commit op ``basalt_ple_commit``, over the pack's six tensors declared as the op contract
gives them, from a synthetic .basalt file that holds them under the pack's names and types (``write_file``). The GPU
side is tests/kernels/test_basalt_ple_program.py.

- REG ``test_registered_and_covered``: both kinds are ``OpClass.MAP``, each bound to the kernel of its own name, and
  the coverage guard passes on the window graph and on the commit graph.
- DISP ``test_dispatches_are_the_builders``: at T = 1 .. 8 the window is 11 T dispatches in the plan's order
  (``STAGES``), each with its builder's kernel, grid, threads and params record (the records' fields pinned,
  ``LAUNCHES``), its slots and ``writes=``, row r's row bindings at r times their row bytes; no StepState.
- X-PROJ ``test_projections_are_main_ops``: every row's q8_1, key and value dispatches are what main's handlers emit for
  a one-column graph on the same tensors, the row's input at r times its row bytes.
- BUF ``test_buffers``: ``hist`` and ``snap`` are states of one size at every width, the row I/O T rows, the seven
  scratch buffers of the plan's sizes, and the six named values in buffers of their own.
- BAR ``test_barriers``: every two dispatches in conflict at buffer granularity, write-after-read included, have a
  barrier between them."""

import json
import struct

import pytest

from monolith import kernels
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import basalt_windows
from monolith.core import DType, Graph
from monolith.core.ir import OpClass
from monolith.core.shapes import T
from monolith.ops import OPS
from monolith.packs.basalt import BasaltSource
from tests.basalt_ple_ref import D, HIST, KERN, N, T_MAX, TENSORS
from tests.contract.test_basalt_ops import emit, gemv, op, q8_1, spliced, weight
from tests.contract.test_basalt_reader import write_basalt
from tests.contract.test_basalt_windows import PROFILE

F32 = DType.F32
KEY, VALUE, TAPS = "gguf/blk.1.ple_key.weight", "dense/blk.1.ple_value.weight", "dense/blk.1.ple_conv1d.weight"
NORMS = tuple(f"dense/blk.1.ple_norm_{n}.weight" for n in ("key", "query", "conv"))


def write_file(path, data):
    """A .basalt file holding the pack's six n-gram tensors as the pack ships them (tests.basalt_ple_ref.TENSORS; the
    pack's dense.index rows 45-50): the key in the gguf section as ggml type 42, with its shape-only dense row, and
    the other five as dense rows of their kinds (5 F16, 2 F32, 4 BF16), each at the next multiple of 64. ``data`` maps
    each tensor's name to its bytes."""
    dense, rows = bytearray(), []
    for name, t in TENSORS.items():
        assert len(data[name]) == t.nbytes, (name, len(data[name]))
        ne0, ne1 = (*t.ne, 0)[:2]
        if t.dense is None:                                                  # the key: bytes in gguf, a shape only
            key = name
            rows.append(f"{name} 0 0 0 0 {len(dense)} 0 {ne0} {ne1} 8 0 32 0 0 0 0 0 0 0")
            continue
        dense += bytes(-len(dense) % 64)
        at = len(dense)
        rows.append(f"{name} 0 {t.kind} {at} {t.nbytes} {at} {t.nbytes} {ne0} {ne1} 0 0 1 0 0 0 0 0 0 0")
        dense += data[name]
    index = f"# dense index v3\n# align 64 pool {len(dense)} tensors {len(rows)}\n" + "\n".join(rows) + "\n"
    t = TENSORS[key]
    meta = {"format": "basalt", "version": 1,
            "tensors": {key: {"type_id": t.type_id, "shape": list(t.ne), "offset": 0, "bytes": t.nbytes}}}
    path.write_bytes(write_basalt({"meta": json.dumps(meta).encode(), "gguf": data[key], "dense": bytes(dense),
                                   "dense.index": index.encode()}))
    return path


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """The six tensors as zero bytes (the planner and the emitter never read them)."""
    return BasaltSource(write_file(tmp_path_factory.mktemp("ple") / "ple.basalt",
                                   {name: bytes(t.nbytes) for name, t in TENSORS.items()}))


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


def window(src, t):
    """The window's graph and its program at T = t."""
    g = Graph("step")
    block(g)
    return g, emit(g, src, t)


# ---- registration ------------------------------------------------------------------------------------------------

def test_registered_and_covered():
    for build in (block, commit):
        g = Graph("step")
        kind = build(g).kind
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
        assert [o.kind for o in g.ops] == [kind]
        check_coverage(g, PROFILE)


# ---- the window's dispatches -------------------------------------------------------------------------------------

STAGES = [  # the plan's eleven dispatches a row (§3): stage, function, the values in slots 0, 1, ..., writes=
    ("q8_1", "native_quantize_q8_1", ("rows", "ple.q8_1"), [1]),
    ("key", "basalt_mmvq", (KEY, "ple.q8_1", "ple.key"), [2]),
    ("value", "basalt_mmvf", ("rows", VALUE, "ple.value"), [2]),
    ("norm_key", "basalt_gr_norm", ("ple.key", NORMS[0], "ple.key_norm"), [2]),
    ("norm_query", "basalt_gr_norm", ("hidden", NORMS[1], "normalized"), [2]),
    ("gate", "basalt_ple_gate", ("ple.key_norm", "normalized", "ple.gate"), [2]),
    ("broadcast", "basalt_ple_broadcast", ("ple.value", "ple.gate", "ple.gated"), [2]),
    ("norm_conv", "basalt_gr_norm", ("ple.gated", NORMS[2], "normalized"), [2]),
    ("conv", "basalt_ple_conv", ("hist", "normalized", TAPS, "hidden", "ple.gated", "ple.conv", "result"), [5, 6]),
    ("advance", "basalt_ple_advance", ("hist", "normalized"), [0]),
    ("snapshot", "basalt_ple_snapshot", ("snap", "hist"), [0]),
]
LAUNCHES = {  # the plan's launches at every width (§3): macros, (threadgroups, threads), params fields or None
    "native_quantize_q8_1": ({}, (10, 256), ("<4I", (2560, 0, 0, 0))),
    "basalt_mmvq": ({"MMVQ_TYPE": "42", "NCOLS": "1", "ROWS": "4", "NW": "4"}, (2560, 128),
                    ("<4I", (2560, 10240, 0, 0))),
    "basalt_mmvf": ({"BLOCK": "256", "NT": "1"}, (2560, 256), ("<8I", (2560, 2560, 1, 2560, 2560, 0, 0, 0))),
    "basalt_gr_norm": ({"NORM_BLOCK": "1024"}, (4, 1024), ("<4I", (2560, 4, 0x358637BD, 0))),    # eps f32(1e-6)
    "basalt_ple_gate": ({}, (4, 512), ("<4I", (0x3CA1E89B, 0, 0, 0))),                          # 1 / sqrt(2560)
    "basalt_ple_broadcast": ({}, (40, 256), None),
    "basalt_ple_conv": ({}, (40, 256), None),
    "basalt_ple_advance": ({}, (40, 256), None),
    "basalt_ple_snapshot": ({}, (64, 256), None),
}
ROW_BYTES = {"rows": 10240, "hidden": 40960, "result": 40960, "normalized": 40960, "snap": 368640}
SCRATCH = {"ple.q8_1": 2880, "ple.key": 40960, "ple.key_norm": 40960, "ple.value": 10240, "ple.gate": 16,
           "ple.gated": 40960, "ple.conv": 40960}
OWN = ("rows", "hidden", "result", "normalized", "hist", "snap")      # the values that must not share a buffer


def builders():
    """Each kernel's (source, macros, (threadgroups, threads), params record or None) from monolith/kernels.py's
    builders: the quantizer, the key and the value at one column whatever the width, the norms over 4 rows."""
    mq, mf = kernels.basalt_mmvq_macros(42, N, 1), kernels.basalt_mmvf_macros(N, N, 1)
    mn, ple = kernels.basalt_gr_norm_macros(N), kernels.basalt_ple_source()
    return {
        "native_quantize_q8_1": (kernels.basalt_q8_1_source(), {}, kernels.basalt_q8_1_grid(N),
                                 kernels.basalt_q8_1_params(N)),
        "basalt_mmvq": (kernels.basalt_mmvq_source(), mq, kernels.basalt_mmvq_grid(mq, D),
                        kernels.basalt_mmvq_params(N, D)),
        "basalt_mmvf": (kernels.basalt_mmvf_source(), mf, kernels.basalt_mmvf_grid(mf, N),
                        kernels.basalt_mmvf_params(N, N, 1)),
        "basalt_gr_norm": (kernels.basalt_gr_norm_source(), mn, kernels.basalt_gr_norm_grid(4, mn),
                           kernels.basalt_gr_norm_params(N, 4, kernels.PLE_RMS_EPS)),
        "basalt_ple_gate": (ple, {}, kernels.PLE_GRIDS["gate"], kernels.basalt_ple_gate_params()),
        **{f"basalt_ple_{s}": (ple, {}, kernels.PLE_GRIDS[s], None)
           for s in ("broadcast", "conv", "advance", "snapshot")},
    }


def rows_of(prog, t):
    """``(r, STAGES entry)`` of each dispatch in order, after checking the names: 11 T, row by row in STAGES' order."""
    assert [o.name for o in prog.ops] == [f"basalt_ple_block:{r}:{s[0]}" for r in range(t) for s in STAGES]
    return [(r, s) for r in range(t) for s in STAGES]


def scratch_of(prog, t):
    """The program's scratch buffers by role: ``ws.T<t>.ple.<role>.<n>`` as ``ple.<role>``."""
    return {"ple." + b.split(".")[3]: b for b in prog.buffers if b.startswith(f"ws.T{t}.ple.")}


@pytest.mark.parametrize("t", range(1, 9))
def test_dispatches_are_the_builders(src, t):
    """DISP, the window: 11 T dispatches in STAGES' order, each with its builder's kernel (function, macros, source),
    grid, threads and params record (made once: five records, nine pipelines), the builders' values the plan's
    (``LAUNCHES``, the records' fields by struct.unpack_from), each slot bound to its value (a weight at its window, a
    scratch buffer at 0, row r of a row binding at r times its row bytes), the plan's ``writes=``; no StepState and
    no threadgroup memory."""
    g, prog = window(src, t)
    _, bound = basalt_windows(src, g.values.values(), t)
    built = builders()
    assert set(built) == set(LAUNCHES)
    for fn, (macros, grid, fields) in LAUNCHES.items():
        _, m, gr, prm = built[fn]
        assert (m, gr, prm is None) == (macros, grid, fields is None), fn
        assert fields is None or struct.unpack_from(fields[0], prm) == fields[1], fn
    scratch = scratch_of(prog, t)

    def at(v, r):
        return (scratch[v], 0) if v in scratch else bound[v] if v in bound else (f"ngram.{v}", r * ROW_BYTES.get(v, 0))

    for o, (r, (stage, fn, values, writes)) in zip(prog.ops, rows_of(prog, t)):
        source, macros, (groups, threads), prm = built[fn]
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, macros, spliced(source)), o.name
        assert (o.grid, o.threadgroup, o.threadgroup_memory) == ((groups, 1, 1), (threads, 1, 1), []), o.name
        want = [(i, *at(v, r)) for i, v in enumerate(values)]
        if prm is not None:
            slot, record, off = o.bindings[-1]
            assert (slot, off, prog.buffers[record].role, prog.buffers[record].init) == (len(values), 0, "params",
                                                                                         prm), o.name
            want.append(o.bindings[-1])
        assert o.bindings == want, o.name
        assert o.meta["writes"] == writes, o.name
    assert prog.step_state not in {b for o in prog.ops for _, b, _ in o.bindings}
    assert sum(b.role == "params" for b in prog.buffers.values()) == 5 and len(prog.kernels) == 9


@pytest.mark.parametrize("t", range(1, 9))
def test_projections_are_main_ops(src, t):
    """X-PROJ: every row's q8_1, key and value dispatches are the dispatches main's ``_q8_1``, ``_mmvq`` and ``_mmvf``
    emit for a one-column graph of the same three ops on the same tensors (basalt's per-row call): the same pipeline
    (its key, source, macros and function), grid, threads, written slots and params bytes, each weight at the same
    bytes of the file, the row's input at r times its row bytes where main binds row 0, every other slot a buffer of
    the same role and size at 0."""
    _, prog = window(src, t)
    g = Graph("step")
    x = g.input("ngram.rows", (T, N), F32)
    gemv(g, "basalt_mmvq", "key", q8_1(g, "xq", [x]), weight(g, KEY, src))
    gemv(g, "basalt_mmvf", "value", x, weight(g, VALUE, src))
    main = emit(g, src, 1)
    mains = dict(zip(("q8_1", "key", "value"), main.ops))
    assert [o.name.split(":")[0] for o in main.ops] == ["basalt_q8_1", "basalt_mmvq", "basalt_mmvf"]

    def reaches(p, b, off):     # a binding as what it reads or writes: a window's bytes of the file, or a buffer
        spec = p.buffers[b]
        return (spec.role, spec.file, spec.file_offset + off) if spec.role == "weights" else (spec.role, spec.nbytes,
                                                                                              spec.init, off)

    ops = {o.name: o for o in prog.ops}
    for r in range(t):
        for stage, m in mains.items():
            o = ops[f"basalt_ple_block:{r}:{stage}"]
            assert (o.kernel, prog.kernels[o.kernel]) == (m.kernel, main.kernels[m.kernel]), o.name
            assert (o.grid, o.threadgroup, o.threadgroup_memory, o.meta["writes"]) == (
                m.grid, m.threadgroup, m.threadgroup_memory, m.meta["writes"]), o.name
            assert [i for i, _, _ in o.bindings] == [i for i, _, _ in m.bindings], o.name
            for (_, b, off), (_, mb, moff) in zip(o.bindings, m.bindings):
                if mb == "ngram.rows":
                    assert (b, off) == (mb, moff + 4 * N * r), o.name
                else:
                    assert reaches(prog, b, off) == reaches(main, mb, moff), (o.name, b, mb)


@pytest.mark.parametrize("t", range(1, 9))
def test_buffers(src, t):
    """BUF, the window: ``hist`` (368,640 B) and ``snap`` (2,949,120 B) are states of one size at every width, so the
    programs share them by name; the rows are T rows of 10,240 B and the hidden rows and the outputs T rows of 40,960 B,
    host- or GPU-written arenas; the scratch is the plan's seven buffers at its sizes, one each, which every row reuses;
    and rows, hidden, result, normalized, hist and snap are bound to six buffers of their own."""
    _, prog = window(src, t)
    sizes = {n: (b.role, b.nbytes) for n, b in prog.buffers.items()}
    assert {v: sizes[f"ngram.{v}"] for v in OWN} == {
        "rows": ("arena", t * 10240), "hidden": ("arena", t * 40960), "result": ("arena", t * 40960),
        "normalized": ("arena", t * 40960), "hist": ("state", 368640), "snap": ("state", 2949120)}
    assert {v: sizes[b] for v, b in scratch_of(prog, t).items()} == {v: ("arena", n) for v, n in SCRATCH.items()}
    assert len([b for b in prog.buffers if b.startswith("ws.")]) == len(SCRATCH)
    bound = {v: set() for v in OWN}
    for o, (_, (_, _, values, _)) in zip(prog.ops, rows_of(prog, t)):
        for (_, b, _), v in zip(o.bindings, values):
            if v in bound:
                bound[v].add(b)
    assert all(len(b) == 1 for b in bound.values()) and len(set().union(*bound.values())) == len(OWN), bound


@pytest.mark.parametrize("t", range(1, 9))
def test_barriers(src, t):
    """BAR: two dispatches conflict when one writes a buffer the other reads or writes (buffer granularity, the reads
    and writes the plan's ``writes=`` give each dispatch's bindings), write-after-read included; every such pair has a
    barrier between them, on the later one or on a dispatch between them (a barrier makes its dispatch, and every one
    after it, wait for every one before it: monolith/compiler/barriers.py)."""
    _, prog = window(src, t)
    access = []
    for o, (_, (_, _, _, writes)) in zip(prog.ops, rows_of(prog, t)):
        w = {b for i, b, _ in o.bindings if i in writes}
        access.append(({b for _, b, _ in o.bindings} - w, w))
    last, war = -1, 0                                                   # the latest barrier at or before dispatch j
    for j, (rj, wj) in enumerate(access):
        last = j if prog.ops[j].barrier_before else last
        for i, (ri, wi) in enumerate(access[:j]):
            if wj & (ri | wi) or rj & wi:
                assert last > i, (prog.ops[i].name, prog.ops[j].name)
                war += not (wj & wi or rj & wi)                         # only a write after a read
    assert prog.ops[0].barrier_before and war, war
