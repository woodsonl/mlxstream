"""Host-only checks of basalt's GDN decode window; no GPU.

The op (monolith/ops/basalt_gdn.py) as compiler/emit_basalt.py's ``_gdn_window`` emits it, over a pack of zeros whose
aux entries hold the layer's weights, each exactly its bytes (tests/contract/test_basalt_ops.py ``pack``); the GPU side
is tests/kernels/test_basalt_gdn_program.py.

- C.1: ``basalt_gdn_window`` registered (MAP over heads, its binding the kind on both families), its handler, and the
  coverage guard.
- C.2: at every width T = 1 .. 8, at basalt's geometry (16 key heads, 48 value heads) with the six weights as weights
  and at 2 and 4 heads with them as constants: one dispatch of ``dec_gdn_rec`` with the builder's source and no
  macros, grid (HV, 1, 1), threadgroup 512 and no threadgroup memory; input i at slot i (an activation or state at 0
  of its own buffer, a weight at its pack entry), yq at 14, the program's own params record at 16
  (``kernels.basalt_gdn_params``) and slot 15 unbound; ``writes`` [10 .. 14]; the state and the stores states of
  their sizes, the stores 8 rows at every width.
- C.3: the barrier pass orders the window behind the pair projection that writes its qkv and z, and the GEMV that
  reads yq behind the window.
- C.4: each refusal, a valid window with exactly one defect refused by that defect's own check (its message, which
  names the value).
- C.5's packer half: ``kernels.basalt_gdn_params``' record, and each of its refusals alone.

C.5's reference half: the references and fixtures of tests/basalt_gdn_ref.py.

- Each fixture generator, called twice with its seed, gives the same bytes; the default stream of ``segments`` is
  default_rng((97, trial, nseg)), which a caller drawing past x (the commit's recorded inputs) passes itself; every
  configuration keeps parity_seg's layout rules (dec_gdn_parity.cpp:422-429).
- ``cpu_seg`` is finite on a segment of eight rows fed ``ab64``'s gate and beta, and on its first two rows equals
  basalt's loops (``cpu_seg_loops``) for three value heads whose q/k head is hd % 16, not hd // 3; ``ab64`` equals
  the plain dot of the bf16 rows there.
- ``conv_tail`` is the bit copy of the last three rows of [history | window rows] for 0 to 6 rows, subnormals, -0,
  infinities and NaN payloads included.
"""

import hashlib
import math
import re
import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler import check_coverage, emit_program
from monolith.compiler.emit import HANDLERS
from monolith.compiler.emit_basalt import GGML_IDS
from monolith.core import BlockDomain, DType, Graph, OpClass, StepStateLayout, T
from monolith.formats.gguf import ggml_row_bytes
from monolith.ops import OPS
from tests.basalt_gdn_ref import (C, COMBOS, EPS, HK, HV, LAYER, MT, N, NL, S, SEG_CASES, SEG_SEED, SF, ZV, ab64,
                                  combo_case, conv_tail, cpu_seg, cpu_seg_loops, max_err, n_bad, seg_layer, segments)
from tests.contract.test_basalt_ops import gemv, pack, q8_1, spliced
from tests.contract.test_basalt_windows import PROFILE


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_gdn_ref_fixtures_give_the_same_bytes_for_one_seed():
    (layer, x), (again, x2) = combo_case(1), combo_case(1)
    assert layer[:3] == COMBOS[1] and x.shape == (MT, N)
    assert digest(*layer[3:], x) == digest(*again[3:], x2), "combo_case(1) drew different bytes"
    case = segments(2, 5)
    assert digest(*case[3:]) == digest(*segments(2, 5)[3:]) and case[:3] == segments(2, 5)[:3]
    explicit = segments(2, 5, np.random.default_rng((SEG_SEED, 2, 5)))
    assert digest(*explicit[3:]) == digest(*case[3:]), "segments' default stream is not default_rng((97, trial, nseg))"
    for trial, nseg in SEG_CASES:
        T, keep, row0, st0, qkv, z, x = segments(trial, nseg)
        R = sum(T)
        assert len(T) == len(keep) == len(row0) == nseg and all(1 <= t <= MT for t in T)
        assert all(0 <= kk <= t for kk, t in zip(keep, T)) and list(row0) == [sum(T[:k]) for k in range(nseg)]
        if trial == 0:
            assert set(T) == set(keep) == {1}
        if trial == 2:
            assert all(T[k] == MT for k in range(1, nseg, 2))
        assert st0.shape == (nseg, NL, SF) and qkv.shape == (R, C) and z.shape == (R, ZV) and x.shape == (R, N)


def test_cpu_seg_is_finite_on_one_segment_and_follows_basalts_loops():
    layer, case, k, heads = seg_layer(), segments(2, 2), 1, (0, 17, 40)
    rows = slice(case.row0[k], case.row0[k] + case.T[k])
    assert case.T[k] == MT
    st0, qkv, z, x = case.st0[k, LAYER], case.qkv[rows], case.z[rows], case.x[rows]
    gate, beta = ab64(x, layer.wa, layer.wb, layer.dt, layer.ssm_a)
    ref = cpu_seg(st0, qkv, z, layer.conv_w, layer.gamma, gate, beta, MT)
    assert ref.h.shape == (MT, C) and ref.y.shape == (MT, ZV)
    assert n_bad(ref.h) == n_bad(ref.y) == n_bad(gate) == n_bad(beta) == 0, "cpu_seg or ab64 is not finite"
    h, y = cpu_seg_loops(st0, qkv[:2], z[:2], layer.conv_w, layer.gamma, gate, beta, 2, heads)
    assert max_err(ref.h[:2], h) < 1e-6, max_err(ref.h[:2], h)
    for (t, hd), want in y.items():
        err = max_err(ref.y[t, hd * S: (hd + 1) * S], want)
        assert err < 1e-6, (t, hd, err)
    for hd in heads:
        wa, wb = ((w[hd].astype(np.uint32) << 16).view(np.float32).tolist() for w in (layer.wa, layer.wb))
        va = vb = 0.0
        for i, xi in enumerate(x[0].tolist()):
            va += wa[i] * xi
            vb += wb[i] * xi
        u = va + float(layer.dt[hd])
        want = ((u if u > 20.0 else math.log1p(math.exp(u))) * float(layer.ssm_a[hd]), 1.0 / (1.0 + math.exp(-vb)))
        assert max_err([gate[0, hd], beta[0, hd]], want) < 1e-6, (hd, gate[0, hd], beta[0, hd], want)


@pytest.mark.parametrize("n", range(7))
def test_conv_tail_copies_the_last_three_rows_bit_for_bit(n):
    ch = np.arange(C, dtype=np.uint32)
    tail = np.stack([(j << 16) | ch for j in range(3)], axis=1)                    # [C][3], oldest first: subnormals
    rows = np.stack([((3 + r) << 16) | ch for r in range(6)])                      # window row r, channel c
    specials = np.array([0x80000000, 0x00000001, 0x7FA00001, 0xFFC00000, 0x7F800000, 0xFF800000], np.uint32)
    tail[:6, 1], tail[100:106, 2], rows[0, 7:13], rows[5, 9:15] = specials, specials, specials, specials
    want = np.stack([tail[:, n + j] if n + j < 3 else rows[n + j - 3] for j in range(3)], axis=1)
    got = conv_tail(tail.reshape(-1).view(np.float32), rows[:n].view(np.float32))
    assert got.dtype == np.float32 and got.shape == (3 * C,)
    bad = np.flatnonzero(got.view(np.uint32) != want.reshape(-1))
    assert bad.size == 0, f"n = {n}: {bad.size} words differ, the first at {bad[:1]}"


# ---- the op's values, its builders and the packs -------------------------------------------------------------------

QK = kernels.Q8_1_BLOCK_BYTES
SMALL = (2, 4)                                   # (h_k, HV): C = 1024, ZV = 512, the refusals' and the chain's layer
SC, SZ = 2 * S * SMALL[0] + S * SMALL[1], S * SMALL[1]
STATE = "gdn.0.state"
STORES = [f"gdn.0.win_{s}" for s in ("qkv", "h", "gate", "beta")]


def spec(hk=HK, hv=HV, n=N, c=None, decl="weight"):
    """The window's values in kernel-slot order, then yq: ``{name: (Graph constructor, shape, dtype or ggml type)}``
    for hk key heads, hv value heads, n_embd n and c channels (2 * 128 * hk + 128 * hv unless given); the six weights
    declared as ``decl`` (``weight``, as gguf tensors, or ``const``, as dense rows)."""
    c = 2 * S * hk + S * hv if c is None else c
    zv = S * hv
    return {STATE: ("state", (S * zv + 3 * c,), "f32"), "qkv": ("input", (T, c), "f32"),
            "conv": (decl, (c, 4), "f32"), "x": ("input", (T, n), "f32"), "alpha": (decl, (hv, n), "bf16"),
            "beta": (decl, (hv, n), "bf16"), "dt": (decl, (hv,), "f32"), "a": (decl, (hv,), "f32"),
            "z": ("input", (T, zv), "f32"), "gamma": (decl, (S,), "f32"),
            STORES[0]: ("state", (MT, c), "f32"), STORES[1]: ("state", (MT, c), "f32"),
            STORES[2]: ("state", (MT, hv), "f32"), STORES[3]: ("state", (MT, hv), "f32"),
            "gdn.0.yq": ("value", (T, zv // 32 * QK), "u8")}


def declare(g, sp, made=None):
    """The values of ``sp`` in its order, a None entry left out; ``made``: values already in the graph, by name."""
    made, vals = made or {}, []
    for name, entry in sp.items():
        if name in made:
            vals.append(made[name])
        elif entry is not None:
            kind, shape, ty = entry
            vals.append(g.weight(name, shape, ty) if kind == "weight" else
                        getattr(g, kind)(name, shape, DType.parse(ty)))
    return vals


def window(g, vals, updates=None, eps=float(EPS)):
    """``basalt_gdn_window`` of ``vals`` (the inputs in slot order, then yq), updating the stores that are graph states
    unless ``updates`` says; ``eps=None`` leaves the attr out."""
    *ins, yq = vals
    ups = [v.name for v in ins[10:] if v.is_state] if updates is None else updates
    g.op("basalt_gdn_window", ins, [yq], domain=BlockDomain("heads", 1), klass=OpClass.MAP, updates=ups,
         **({} if eps is None else {"eps": eps}))
    return yq


def nbytes(v):
    """A weight's ggml rows, or a constant's values, in bytes."""
    if v.is_weight:
        return ggml_row_bytes(GGML_IDS[v.format], v.shape[-1]) * math.prod(v.shape[:-1])
    return math.prod(v.shape) * v.dtype.itemsize


def packs(directory, vals, at=None, short=(), slab=()):
    """Packs of zeros for the weights and constants among ``vals``: aux entries of exactly their bytes (four fewer for
    those in ``short``) at 64-byte steps plus ``at[name]`` bytes, and those in ``slab`` as bf16 block-lane-major slabs
    of a second pack."""
    at, entries, slabs, off = at or {}, {}, {}, 0
    for v in vals:
        if v.is_weight or v.is_const:
            nb = nbytes(v) - (4 if v.name in short else 0)
            if v.name in slab:
                slabs[v.name] = (0, nb)
                continue
            entries[v.name] = (off + at.get(v.name, 0), nb)
            off += -(-(nb + 16) // 64) * 64
    directory.mkdir(parents=True, exist_ok=True)
    return [pack(directory / "aux", entries)] + ([pack(directory / "slabs", slabs, slab=True)] if slabs else [])


def emit(g, pks, t=2, **kw):
    return emit_program(g, pack=pks, profile=PROFILE, t=t, tail=None, **kw)


def emit_window(directory, geometry=SMALL, values=None, attrs=None, at=None, short=(), slab=(), **kw):
    """One window over ``spec(*geometry)`` with ``values`` replacing its entries (None drops one), its weights in
    ``packs``, emitted at T = 2 unless ``kw`` says."""
    g = Graph("step")
    vals = declare(g, {**spec(*geometry), **(values or {})})
    window(g, vals, **(attrs or {}))
    return emit(g, packs(directory, vals, at, short, slab), **kw)


# ---- C.1 to C.3: registration, the dispatch against the builders, the barriers -----------------------------------

def test_registered_and_covered():
    od = OPS.get("basalt_gdn_window")
    assert od.klass is OpClass.MAP and od.domain_kind == "heads"
    assert od.binding_for("apple9").kernel == od.binding_for("apple10").kernel == "basalt_gdn_window"
    assert HANDLERS["basalt_gdn_window"].__name__ == "_gdn_window"
    g = Graph("step")
    window(g, declare(g, spec()))
    check_coverage(g, PROFILE)


@pytest.mark.parametrize("t", range(1, 9))
@pytest.mark.parametrize("hk, hv, decl", [(HK, HV, "weight"), (*SMALL, "const")], ids=["basalt", "small-const"])
def test_window_dispatch_is_the_builders(tmp_path, hk, hv, decl, t):
    """One dispatch of ``dec_gdn_rec`` at T = t: the builder's source, no macros (GDN_Y_BOUND 0: the engine binds no
    y), grid (HV, 1, 1) of 512 threads; input i at slot i, yq at 14, the record at 16 and nothing at 15 (T is the
    program's width, no StepState); the record ``basalt_gdn_params`` of the shapes; the window writes slots 10 to 14;
    the state and the stores are states of their sizes, the stores 8 rows at every width."""
    g = Graph("step")
    vals = declare(g, spec(hk, hv, decl=decl))
    window(g, vals)
    (pk,) = packs(tmp_path, vals)
    prog = emit(g, [pk], t)
    (o,) = prog.ops
    c, zv = 2 * S * hk + S * hv, S * hv
    k = prog.kernels[o.kernel]
    assert (k.function, k.macros, k.source) == ("dec_gdn_rec", {}, spliced(kernels.basalt_gdn_source()))
    assert (o.name, o.grid, o.threadgroup, o.threadgroup_memory) == (f"basalt_gdn_window:{STATE}", (hv, 1, 1),
                                                                      (512, 1, 1), [])
    assert [b[0] for b in o.bindings] == [*range(15), 16] and o.meta["writes"] == [10, 11, 12, 13, 14]
    for (_, name, off), v in zip(o.bindings, vals):
        if v.is_weight or v.is_const:
            b = prog.buffers[name]
            assert (b.role, b.file, b.file_offset + off) == ("weights", str(pk.dir / "weights.pack"),
                                                             pk.aux[v.name]["offset"]), v.name
        else:
            assert (name, off) == (v.name, 0), v.name
    sizes = {STATE: ("state", (S * zv + 3 * c) * 4), "gdn.0.yq": ("arena", t * zv // 32 * QK),
             **{s: ("state", MT * w * 4) for s, w in zip(STORES, (c, c, hv, hv))}}
    assert {n: (prog.buffers[n].role, prog.buffers[n].nbytes) for n in sizes} == sizes
    _, prm, off = o.bindings[-1]
    rec = prog.buffers[prm]
    assert (prm, off, rec.role) == (f"params.S{t}.basalt_gdn_window.0", 0, "params")
    assert rec.init == kernels.basalt_gdn_params(t, c, N, hk, hv, float(EPS))
    assert struct.unpack_from("<IIIIIfII", rec.init) == (t, c, N, hk, hv, float(EPS), 0, 0)


def test_the_window_waits_for_the_pair_and_its_reader_for_the_window(tmp_path):
    """The GDN block, basalt's path: the native q8_1 rows of x, the pair projection of qkv and z, the window, then
    dense_tc on yq; each op reads the one before it, so the barrier pass puts each behind a barrier."""
    hk, hv = SMALL
    g = Graph("step")
    x = g.input("x", (T, N), DType.F32)
    w = {name: g.weight(name, shape, "q4_k") for name, shape in (("w.qkv", (SC, N)), ("w.z", (SZ, N)),
                                                                  ("w.out", (N, SZ)))}
    qkv, z = gemv(g, "basalt_dense_tc_pair", "pair", q8_1(g, "xq", [x]), w["w.qkv"], w["w.z"])
    vals = declare(g, spec(hk, hv), {"x": x, "qkv": qkv, "z": z})
    gemv(g, "basalt_dense_tc", "out", window(g, vals), w["w.out"])
    prog = emit(g, packs(tmp_path, [*w.values(), *vals]))
    assert [o.name.split(":")[0] for o in prog.ops] == ["basalt_q8_1", "basalt_dense_tc_pair", "basalt_gdn_window",
                                                         "basalt_dense_tc"]
    assert [o.barrier_before for o in prog.ops] == [True, True, True, True]


# ---- C.4: refusals; each case's valid window emits, and the window with its one defect is refused by its own check -

T16 = StepStateLayout(t_max=16)
VALUES = spec(*SMALL)
BAD_TYPE = {"f32": "bf16", "bf16": "f16", "u8": "i32"}
CONSTS = {"qkv": ("const", (2, SC), "f32"), "x": ("const", (2, N), "f32"), "z": ("const", (2, SZ), "f32")}
ALIGNED = [("qkv", 1, 4), ("conv", 2, 4), ("x", 3, 16), ("alpha", 4, 16), ("beta", 5, 16), ("dt", 6, 4), ("a", 7, 4),
           ("z", 8, 4), ("gamma", 9, 4)]                        # (value, its kernel slot, the alignment it needs)


def type_message(name, ty):
    """The message of the check that refuses ``name`` declared ``BAD_TYPE[ty]`` instead of ``ty``."""
    bad = BAD_TYPE[ty]
    if name in ("conv", "alpha", "beta"):
        return re.escape(f"the weight {name!r} ({bad} ") + ".*" + re.escape(f"of ggml type [{ty!r}]")
    if name in ("dt", "a", "gamma"):
        return re.escape(f"the vector {name!r} ({bad} ") + ".*" + re.escape(f"of ggml type {ty}")
    return re.escape(f"{name!r} is {bad} ") + r"(\[[^\]]*\]), not " + re.escape(f"{ty} ") + r"\1"


SHAPES = [                                  # (value, a shape of the wrong size or rank, the message of its own check)
    (STATE, (S * SZ,), r"'gdn.0.state' is f32 \[65536\], not f32 \[68608\] \(the recurrent state, then the conv"),
    ("qkv", (T, SC + 4), r"'qkv' is f32 \[2, 1028\], not f32 \[2, 1024\] \(the window's q, k and v projections\)"),
    ("conv", (SC, 3), "the conv weight 'conv' holds 3 taps a channel; basalt's conv has 4"),
    ("x", (T, N + 4), r"'x' is f32 \[2, 2564\], not f32 \[2, 2560\] \(the gates' input rows\)"),
    ("alpha", (5, N), r"the gate rows 'alpha' are \[5, 2560\], not \[4, 2560\]: one row of n_embd per value head"),
    ("beta", (4, N + 256), r"the gate rows 'beta' are \[4, 2816\], not \[4, 2560\]"),
    ("dt", (4, 1), r"the vector 'dt' \(f32 \[4, 1\]\) is not a weight or constant \[4\]"),
    ("a", (5,), r"the vector 'a' \(f32 \[5\]\) is not a weight or constant \[4\]"),
    ("z", (T, SZ + 4), r"'z' is f32 \[2, 516\], not f32 \[2, 512\] \(the output gate's rows\)"),
    ("gamma", (64,), r"the vector 'gamma' \(f32 \[64\]\) is not a weight or constant \[128\]"),
    *[(s, (7, w), rf"'{re.escape(s)}' is f32 \[7, {w}\], not f32 \[8, {w}\] \(the window's 8 rows, a commit's replay\)")
      for s, w in zip(STORES, (SC, SC, 4, 4))],
    ("gdn.0.yq", (T, SZ // 32 * QK - QK), r"'gdn.0.yq' is u8 \[2, 540\], not u8 \[2, 576\] \(q8_1 rows of the window"),
]

REFUSALS = [
    ("dynamic-T", {}, {"dynamic_t": True}, "a dynamic-T program cannot run them"),
    ("T = 9", {"t": 8, "layout": T16}, {"t": 9, "layout": T16}, "the window's width must be 1 to 8, got 9"),
    ("n_embd 2048", {}, {"geometry": (*SMALL, 2048)}, "n_embd must be 2560, got 2048"),
    ("C not 256 h_k + 128 HV", {}, {"geometry": (*SMALL, N, SC + 128)},
     r"C must be 2 \* 128 \* h_k \+ 128 \* h_v = 1024, got 1152"),
    ("no key head", {}, {"geometry": (0, 4)}, "h_k must be positive, got 0"),
    ("HV not a multiple of h_k", {}, {"geometry": (2, 3)}, "h_v must be a multiple of h_k, got h_v 3 and h_k 2"),
    ("13 inputs", {}, {"values": {STORES[3]: None}}, r"13 inputs and 1 outputs; the window reads 14 \(the state"),
    *[(f"{name} as {BAD_TYPE[ty]}", {}, {"values": {name: (kind, shape, BAD_TYPE[ty])}}, type_message(name, ty))
      for name, (kind, shape, ty) in VALUES.items()],
    *[(f"{name} of shape {list(shape)}", {}, {"values": {name: (VALUES[name][0], shape, VALUES[name][2])}}, why)
      for name, shape, why in SHAPES],
    *[(f"{name} not a graph state", {}, {"values": {name: ("value", VALUES[name][1], "f32")}},
       f"{re.escape(repr(name))} is not a graph state") for name in (STATE, *STORES)],
    ("the state in updates", {}, {"attrs": {"updates": [STATE, *STORES]}},
     "updates names 'gdn.0.state'; the window reads the state and never writes it"),
    ("a store missing from updates", {}, {"attrs": {"updates": STORES[:3]}},
     r"updates names \[.*\]; the window writes exactly its stores"),
    ("no eps", {}, {"attrs": {"eps": None}}, "needs attr eps, the norms' epsilon"),
    *[(f"{name} off {align} bytes", {"values": CONSTS, "at": {name: align}},
       {"values": CONSTS, "at": {name: align // 2}},
       rf"{re.escape(repr(name))} binds at byte \d+ of pack\.0; slot {slot} needs {align}-byte alignment")
      for name, slot, align in ALIGNED],
    ("dt's pack entry a value short", {}, {"short": ("dt",)},
     r"the vector 'dt' is 16 bytes of f32 \[4\], but its pack entry holds 12"),
    ("alpha a pack slab", {}, {"slab": ("alpha",)}, "the weight 'alpha' is a block-lane-major pack slab"),
    ("gamma a pack slab", {}, {"slab": ("gamma",)}, "the vector 'gamma' is a block-lane-major pack slab"),
]


@pytest.mark.parametrize("good, bad, why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(tmp_path, good, bad, why):
    emit_window(tmp_path / "good", **good)
    with pytest.raises(ValueError, match=why):
        emit_window(tmp_path / "bad", **bad)


# ---- C.5's packer half ---------------------------------------------------------------------------------------------

def test_gdn_params_record_and_refusals():
    """``kernels.basalt_gdn_params``: the 32-byte record at every width; each refusal alone on a valid record: what
    host dec_gdn_rec exits on (dec_gdn_rec.cu:355-367), no key head refused before the modulo, and the port's C."""
    for t in range(1, 9):
        rec = kernels.basalt_gdn_params(t, C, N, HK, HV, float(EPS))
        assert len(rec) == 32 and struct.unpack("<IIIIIfII", rec) == (t, C, N, HK, HV, float(EPS), 0, 0)
    good = dict(t=8, c=C, n_embd=N, h_k=HK, h_v=HV, eps=float(EPS))
    for change, why in [({"t": 0}, "width must be 1 to 8, got 0"), ({"t": 9}, "width must be 1 to 8, got 9"),
                        ({"n_embd": 2048}, "n_embd must be 2560, got 2048"),
                        ({"h_k": 0}, "h_k must be positive, got 0"), ({"h_k": -1}, "h_k must be positive, got -1"),
                        ({"h_v": 40}, "h_v must be a multiple of h_k, got h_v 40 and h_k 16"),
                        ({"c": C + 128}, r"C must be 2 \* 128 \* h_k \+ 128 \* h_v = 10240, got 10368")]:
        with pytest.raises(ValueError, match=why):
            kernels.basalt_gdn_params(**{**good, **change})
