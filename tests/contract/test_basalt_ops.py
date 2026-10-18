"""basalt's op family (monolith/ops/basalt.py) as compiler/emit_basalt.py emits it over a synthetic .basalt file bound
through its windows (compiler/emit.py ``basalt_windows``): each op's dispatch (function, macros, grid, threadgroup,
params and written slots) equal to monolith/kernels.py's builders, and each refusal, a valid input with exactly one
defect refused by that defect's own check (its message). The GPU side is tests/kernels/test_basalt_ops_program.py."""

import json

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler import emit_program
from monolith.compiler.coverage import check_coverage
from monolith.compiler.emit import basalt_windows
from monolith.core import DType, Graph
from monolith.core.ir import BlockDomain, OpClass
from monolith.core.shapes import T
from monolith.core.step_state import StepStateLayout
from monolith.formats.gguf import GGML_TYPES, ggml_row_bytes
from monolith.ops import OPS
from monolith.packs import PackFile
from monolith.packs.basalt import BasaltSource
from monolith.packs.packer import ALIGN
from tests.contract.test_basalt_reader import write_basalt
from tests.contract.test_basalt_windows import PROFILE

F32, U8 = DType.F32, DType.U8
KINDS = ("basalt_q8_1", "basalt_mmvq", "basalt_mmvf", "basalt_dense_tc", "basalt_dense_tc_pair")
Q8_1_FUNCTIONS = {"native": "native_quantize_q8_1", "swiglu": "native_swiglu_quantize_q8_1", "ieee": "quantize_q8_1"}
QK = kernels.Q8_1_BLOCK_BYTES

# The tensors (name: (ggml type id, n_out, n_in)): every native_mmvq type at basalt's 2560 inputs, Q8_0 at 640 (the
# shared expert's down, fed by the SwiGLU quantizer), every dense_tc type in every summation order at basalt's defaults
# (170 multiprocessors, an aligned weight; "tile" at 2721 rows where a row is whole 16-byte units, so the one order's
# 170 tiles are passed), the pair's types, and Q4_K at 2048 x 336, which 170 multiprocessors run in the one order and
# 20 (PROFILE's cores) in the tile order. BF16 rows (name: (n_out, n_in)) live in the dense section, as basalt's do.
DTC_IN = {"one": 2048, "tile": 512, "s2": 22528, "s1": 23040}
MMVQ = {f"mmvq.{t}": (t, 70, 2560) for t in kernels.MMVQ_TRAITS}
MMVQ["mmvq.down"] = (8, 70, 640)
DTC = {f"dtc.{t}.{o}": (t, 2721 if o == "tile" and DTC_IN[o] // 256 * sb % 16 == 0 else 16, DTC_IN[o])
       for t, sb in kernels.DENSE_TC_SB.items() for o in DTC_IN}
PAIRS = [(14, 23), (12, 11), (42, 13)]                  # (w0, w1) types at 2560 inputs, 35 and 18 rows
PAIR = {f"pair.{a}.{b}.{m}": (ty, (35, 18)[m], 2560) for a, b in PAIRS for m, ty in enumerate((a, b))}
GGUF = {**MMVQ, **DTC, **PAIR, "order.q4_k": (12, 336, 2048)}
BF16 = {"mmvf.q": (128, 2560), "mmvf.gate": (1, 2560)}


def gguf_bytes(type_id, n_out, n_in):
    return n_out * ggml_row_bytes(type_id, n_in)


def write_file(path, gguf, bf16):
    """A .basalt file with ``gguf`` ({name: (type id, n_out, n_in, bytes)}) in its gguf section, each tensor at the next
    multiple of 4096 (basalt's kTensorAlign), and the BF16 matrices ``bf16`` ({name: (n_out, n_in, bytes)}) as dense
    rows of kind 4 at multiples of 64 (its arena's alignment), so that tensors bind at offsets inside their windows."""
    blob, tensors = bytearray(), {}
    for name, (t, n_out, n_in, data) in gguf.items():
        blob += bytes(-len(blob) % 4096)
        tensors[name] = {"type_id": t, "shape": [n_in, n_out], "offset": len(blob), "bytes": len(data)}
        blob += data
    dense, rows = bytearray(), []
    for name, (n_out, n_in, data) in bf16.items():
        dense += bytes(-len(dense) % 64)
        rows.append(f"{name} 0 4 {len(dense)} {len(data)} {len(dense)} {len(data)} {n_in} {n_out} 0 0 1 0 0 0 0 0 0 0")
        dense += data
    index = f"# dense index v3\n# align 64 pool {len(dense)} tensors {len(rows)}\n" + "\n".join(rows) + "\n"
    meta = {"format": "basalt", "version": 1, "tensors": tensors}
    path.write_bytes(write_basalt({"meta": json.dumps(meta).encode(), "gguf": bytes(blob), "dense": bytes(dense),
                                   "dense.index": index.encode()}))
    return path


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """The tensors above as random bytes (the planner and the emitter never read them)."""
    rng = np.random.default_rng(3)
    noise = lambda n: rng.integers(0, 256, n, dtype=np.uint8).tobytes()     # noqa: E731
    gguf = {n: (t, o, i, noise(gguf_bytes(t, o, i))) for n, (t, o, i) in GGUF.items()}
    bf16 = {n: (o, i, noise(2 * o * i)) for n, (o, i) in BF16.items()}
    return BasaltSource(write_file(tmp_path_factory.mktemp("ops") / "ops.basalt", gguf, bf16))


# ---- graph builders (the GPU test uses them too) -------------------------------------------------------------------

def weight(g, name, src=None, shape=None, fmt=None):
    """The value that declares a gguf tensor or a BF16 dense row exactly (``gguf/`` a weight, ``dense/`` a constant),
    or, with ``shape`` and ``fmt``, a weight of that format (a pack entry)."""
    if shape is not None:
        return g.weight(name, shape, fmt)
    e = src.entries[name]
    if name.startswith("dense/"):
        return g.const(name, e.shape, DType.BF16)
    return g.weight(name, e.shape, GGML_TYPES[e.type_id][0])


def op(g, kind, ins, outs, **attrs):
    return g.op(kind, ins, outs, domain=BlockDomain("rows", 1), klass=OpClass.MAP, **attrs)


def q8_1(g, name, xs, kind="native", rows=T, width=None):
    """``basalt_q8_1`` of ``xs`` (``[x]``, or ``[gate, up]`` for swiglu) into q8_1 rows named ``name``."""
    n_in = xs[0].shape[1]
    y = g.value(name, (rows, n_in // 32 * QK if width is None else width), U8)
    op(g, "basalt_q8_1", xs, [y], quantizer=kind)
    return y


def gemv(g, kind, name, x, *ws, rows=T, widths=None):
    """``kind`` (basalt_mmvq, basalt_dense_tc, basalt_dense_tc_pair, basalt_mmvf) of ``x`` by ``ws`` into F32 outputs
    named ``name`` (the pair: ``name.0``, ``name.1``), each as wide as its weight's rows unless ``widths`` says."""
    widths = widths or [w.shape[0] for w in ws]
    names = [name] if len(ws) == 1 else [f"{name}.{m}" for m in range(len(ws))]
    ys = [g.value(n, (rows, wd), F32) for n, wd in zip(names, widths)]
    op(g, kind, [x, *ws], ys)
    return ys


def emit(g, src, t=1, packs=(), **kw):
    return emit_program(g, pack=[src, *packs], profile=PROFILE, t=t, tail=None, **kw)


def specs(prog, kind):
    return [o for o in prog.ops if o.name.startswith(kind + ":")]


def kernel_of(prog, o):
    return prog.kernels[o.kernel]


def spliced(source):
    """A builder's source as the emitter registers it: the StepState struct after the prelude (``_Ctx.kernel``)."""
    return source.replace(kernels.PRELUDE, kernels.PRELUDE + StepStateLayout().to_msl() + "\n", 1)


def all_ops(g, src):
    """Every kind over every tensor: one native q8_1 per input width, the SwiGLU into the Q8_0 down, the IEEE rows on
    their own; returns {output name: (kind, weights)}."""
    widths = sorted({e[2] for e in GGUF.values()})
    xs = {n: g.input(f"x{n}", (T, n), F32) for n in widths}
    xq = {n: q8_1(g, f"xq{n}", [xs[n]]) for n in widths}
    gate, up = g.input("gate", (T, 640), F32), g.input("up", (T, 640), F32)
    hq = q8_1(g, "hq", [gate, up], "swiglu")
    q8_1(g, "xq_ieee", [xs[2560]], "ieee")
    out = {}
    for name, (t, n_out, n_in) in {**MMVQ, **DTC, "order.q4_k": GGUF["order.q4_k"]}.items():
        kind = "basalt_mmvq" if name.startswith("mmvq.") else "basalt_dense_tc"
        w = weight(g, f"gguf/{name}", src)
        gemv(g, kind, f"y.{name}", hq if name == "mmvq.down" else xq[n_in], w)
        out[f"y.{name}"] = (kind, [w])
    for a, b in PAIRS:
        ws = [weight(g, f"gguf/pair.{a}.{b}.{m}", src) for m in (0, 1)]
        gemv(g, "basalt_dense_tc_pair", f"y.pair.{a}.{b}", xq[2560], *ws)
        out[f"y.pair.{a}.{b}"] = ("basalt_dense_tc_pair", ws)
    for name in BF16:
        w = weight(g, f"dense/{name}", src)
        gemv(g, "basalt_mmvf", f"y.{name}", xs[2560], w)
        out[f"y.{name}"] = ("basalt_mmvf", [w])
    return out


# ---- registration, and each dispatch against the builders ----------------------------------------------------------

def test_registered_and_covered(src):
    for kind in KINDS:
        od = OPS.get(kind)
        assert od.klass == OpClass.MAP and od.binding_for(PROFILE.key).kernel == kind
    g = Graph("step")
    all_ops(g, src)
    assert {o.kind for o in g.ops} == set(KINDS)
    check_coverage(g, PROFILE)


def test_grid_builders_are_the_launchers():
    """The two grid builders this PR adds, against the harness's own formulas (tests/kernels/basalt_gpu.py)."""
    for n in (32, 256, 2560, 8 * 23040):
        assert kernels.basalt_q8_1_grid(n) == (-(-n // 256), 256)
    for n_in, n_out, t in ((2560, 128, 1), (2560, 1, 5), (640, 7, 8)):
        m = kernels.basalt_mmvf_macros(n_in, n_out, t)
        assert kernels.basalt_mmvf_grid(m, n_out) == (n_out, kernels.basalt_mmvf_block(n_in))


def expected(gop, src, t):
    """(function, macros, source, params, grid, threadgroup) of ``gop``'s dispatch from the builders."""
    if gop.kind == "basalt_q8_1":
        n = t * gop.inputs[0].shape[1]
        return (Q8_1_FUNCTIONS[gop.attrs["quantizer"]], {}, kernels.basalt_q8_1_source(), kernels.basalt_q8_1_params(n),
                -(-n // 256), 256)
    (t0, *rest), n_in = [(src.entries[w.name].type_id, src.entries[w.name].shape[0]) for w in gop.inputs[1:]], \
        src.entries[gop.inputs[1].name].shape[1]
    tid, n_out = t0
    if gop.kind == "basalt_mmvq":
        m = kernels.basalt_mmvq_macros(tid, n_in, t)
        return ("basalt_mmvq", m, kernels.basalt_mmvq_source(), kernels.basalt_mmvq_params(n_in, n_out),
                *kernels.basalt_mmvq_grid(m, n_out))
    if gop.kind == "basalt_dense_tc":
        m = kernels.basalt_dense_tc_macros(tid, n_in, n_out)
        return ("basalt_dense_tc", m, kernels.basalt_dense_tc_source(), kernels.basalt_dense_tc_params(n_in, n_out, t),
                *kernels.basalt_dense_tc_grid(m, n_out))
    if gop.kind == "basalt_dense_tc_pair":
        (t1, n_out1), = rest
        m = kernels.basalt_dense_tc_pair_macros(tid, t1, n_in)
        return ("basalt_dense_tc_pair", m, kernels.basalt_dense_tc_source(),
                kernels.basalt_dense_tc_params(n_in, n_out, t, n_out1), *kernels.basalt_dense_tc_pair_grid(n_out, n_out1))
    m = kernels.basalt_mmvf_macros(n_in, n_out, t)
    return ("basalt_mmvf", m, kernels.basalt_mmvf_source(), kernels.basalt_mmvf_params(n_in, n_out, t),
            n_out, kernels.basalt_mmvf_block(n_in))


def slot_values(gop):
    """The values in the kernel's buffer slots 0, 1, ...: weights first for the q8_1 GEMVs, as the kernels take them."""
    ins, outs = list(gop.inputs), list(gop.outputs)
    if gop.kind in ("basalt_mmvq", "basalt_dense_tc", "basalt_dense_tc_pair"):
        ins = ins[1:] + ins[:1]
    return ins + outs


@pytest.mark.parametrize("t", range(1, 9))
def test_dispatches_are_the_builders(src, t):
    """Every op of ``all_ops`` at T = t: one dispatch whose kernel, macros, grid, threadgroup and params record are the
    builders' (the q8_1 and BF16 grids as the harness writes them), binding each weight at the window and offset the
    planner gives it and each activation at 0 of its own buffer, writing exactly its outputs' slots, the params at
    the slot after them; no StepState and no threadgroup memory."""
    g = Graph("step")
    all_ops(g, src)
    prog = emit(g, src, t)
    _, bound = basalt_windows(src, g.values.values(), t)
    assert len(prog.ops) == len(g.ops)
    writer = {name: o for o in prog.ops for i, name, _ in o.bindings if i in o.meta["writes"]}
    for gop in g.ops:
        o = writer[gop.outputs[0].name]
        fn, macros, source, params, grid, tg = expected(gop, src, t)
        k = prog.kernels[o.kernel]
        assert (k.function, k.macros, k.source) == (fn, macros, spliced(source)), gop
        assert (o.grid, o.threadgroup, o.threadgroup_memory) == ((grid, 1, 1), (tg, 1, 1), []), gop
        vals = slot_values(gop)
        assert o.bindings[:-1] == [(i, *bound.get(v.name, (v.name, 0))) for i, v in enumerate(vals)], gop
        slot, prm, off = o.bindings[-1]
        assert (slot, off, prog.buffers[prm].role, prog.buffers[prm].init) == (len(vals), 0, "params", params), gop
        assert o.meta["writes"] == list(range(len(gop.inputs), len(vals))), gop


def test_dense_tc_runs_basalts_orders(src):
    """Each dtc.<type>.<order> tensor runs in the order it names, at basalt's 170 multiprocessors and aligned weights."""
    g = Graph("step")
    all_ops(g, src)
    prog = emit(g, src, 8)
    seen = set()
    for o in [o for o in specs(prog, "basalt_dense_tc") if "/dtc." in o.name]:
        _, ty, order = o.name.split("/")[1].split(".")
        assert prog.kernels[o.kernel].macros["DTC_ORDER"] == str(kernels.DENSE_TC_ORDERS[order]), o.name
        seen.add((int(ty), order))
    assert seen == {(t, o) for t in kernels.DENSE_TC_SB for o in DTC_IN}


def test_q8_1_rows_reach_their_readers_behind_a_barrier(src):
    """The barrier pass orders each GEMV behind the quantizer whose rows it reads (``writes`` declares them)."""
    g = Graph("step")
    x = g.input("x", (T, 2560), F32)
    xq = q8_1(g, "xq", [x])
    gemv(g, "basalt_mmvq", "y0", xq, weight(g, "gguf/mmvq.8", src))
    gemv(g, "basalt_dense_tc", "y1", xq, weight(g, "gguf/pair.14.23.0", src))
    gemv(g, "basalt_mmvf", "y2", x, weight(g, "dense/mmvf.q", src))
    prog = emit(g, src, 2)
    assert [o.barrier_before for o in prog.ops] == [True, True, False, False]
