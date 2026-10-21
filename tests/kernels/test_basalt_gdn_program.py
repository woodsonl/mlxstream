# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_gdn_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its out-projection checks and pass criteria.
"""basalt's GDN decode window in compiled programs on the GPU (the ``basalt_gdn_window`` op kind, the emitter's
``_gdn_window``): one hand-built static program per window width T = 1 .. 8 (basalt captures one graph per width),
each the GDN block on the engine's path A (verify.cpp:1224-1245, spec E§10 items 1-4): ``basalt_q8_1`` native on x,
``basalt_dense_tc_pair`` (qkv, z), the window, ``basalt_dense_tc`` on its yq rows (out). The weights are parity_gdnf's
combo 0 (tests/basalt_gdn_ref.py ``combo_case(0)``) in a synthetic .basalt file laid out as basalt's converter writes
one: the projections GGUF tensors, the conv weight, dt, ssm_a and gamma F32 and the gate rows BF16 dense rows, the
vectors 1-D. The file is bound through its windows, and again through exact per-tensor copies
(``BasaltSource(copy=True)``), so that shader validation bounds every read by its own tensor. The eight engines share
the weights and the states by name (the layer's state, written once, and the four window stores); before each run every
activation and store is 0xFF bytes. The 8-wide program runs once more first, as basalt's ``w8`` (:277-278).

- G7 / P.1 (dec_gdn_parity.cpp:292): out of the program against the per-stage chain run by the single-kernel harness
  (tests/kernels/basalt_gpu.py, basalt_gdn_gpu.py) on the same x and bytes: native q8_1, dense_tc_pair,
  gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi, native q8_1 of its y, dense_tc (basalt's run_ref, :176-193,
  with the engine's projections), by basalt's ``cmp_f`` (every element's 4 bytes equal, both sides finite), after G6's
  precondition on the chain's y (``g6_blocks``, a fixture error); qkv and z first (:286-287), so that a difference
  names its stage.
- G10 (:295): rows t < T of the T-wide program's out against the 8-wide program's, by ``cmp_f``.
- P.2: the program's yq rows and four window stores bitwise dec_gdn_rec's (``basalt_gdn_gpu.rec``) on the program's
  own qkv and z, the harness's run with every output finite and every sentinel intact; rows T..7 of every store keep
  their 0xFF; the state's bytes unchanged.
- P.3: the eight programs hold the same states, of the same sizes, and their engines one buffer per state (and
  weight) name.
"""

import functools
import json

import numpy as np
import pytest

from monolith.compiler import emit_program
from monolith.core import DType, Graph, T
from monolith.formats.gguf import GGML_TYPES
from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.basalt_gdn_ref import C, F32, MT, N, ZV, combo_case
from tests.contract.test_basalt_gdn_ops import STATE, STORES, declare, spec, window
from tests.contract.test_basalt_ops import gemv, q8_1
from tests.contract.test_basalt_reader import write_basalt
from tests.contract.test_basalt_windows import PROFILE
from tests.kernels import basalt_gdn_gpu as gg
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_dec_gdn import check_run, cmp_f, g6_blocks

WIDTHS = range(1, MT + 1)
PIPELINES = {}                                   # one pipeline cache for every engine of the module
ACTS = ("xq", "pair.0", "pair.1", "gdn.0.yq", "out")
GGUF = {"attn_qkv.weight": ("tq", "wqkv", N, C), "attn_gate.weight": ("tz", "wz", N, ZV),
        "ssm_out.weight": ("to", "wo", ZV, N)}   # verify.cpp:1182-1187: name -> (type, Layer field, n_in, n_out)
DENSE = {"conv": ("ssm_conv1d.weight", 2, "conv_w"), "alpha": ("ssm_alpha.weight", 4, "wa"),
         "beta": ("ssm_beta.weight", 4, "wb"), "dt": ("ssm_dt.bias", 2, "dt"), "a": ("ssm_a", 2, "ssm_a"),
         "gamma": ("ssm_norm.weight", 2, "gamma")}   # spec's key -> (the dense row, its kind: 2 F32, 4 BF16, field)
G1 = '`d += cmp_f("qkv", r.qkv, n.qkv, (size_t) T * C, ma);` -> `d == 0`'
G2 = '`d += cmp_f("z", r.z, n.z, (size_t) T * ZV, ma);` -> `d == 0`'
G7 = '`d += cmp_f("out", r.bo, n.bo, (size_t) T * N, ma);` -> `d == 0`'
G10 = '`inv += cmp_f("out@8", w8.bo, n.bo, (size_t) T * N, ma);` -> `inv == 0`'


@functools.lru_cache(maxsize=None)
def case():
    """parity_gdnf's combo 0 (:270-276): the layer and x's MT rows."""
    return combo_case(0)


def write_layer(path, layer):
    """``layer``'s weights as a .basalt file (``write_basalt``) in tests/contract/test_basalt_ops.py ``write_file``'s
    layout: the projections GGUF tensors in ``gguf``, each at the next multiple of 4096 (basalt's kTensorAlign); the
    GDN weights ``dense`` rows at multiples of 64, F32 (kind 2) and BF16 (kind 4) as basalt's converter stores them
    (tools/convert/model_sections.py:44-58), a vector a row of ne1 0 (1-D); ``dense.index`` after ``dense``, so that
    the last row's page ends inside the file."""
    blob, tensors = bytearray(), {}
    for name, (ty, field, n_in, n_out) in GGUF.items():
        blob += bytes(-len(blob) % 4096)
        w = getattr(layer, field)
        tensors[name] = {"type_id": getattr(layer, ty), "shape": [n_in, n_out], "offset": len(blob), "bytes": w.nbytes}
        blob += w.tobytes()
    pool, rows = bytearray(), []
    for name, kind, field in DENSE.values():
        pool += bytes(-len(pool) % 64)
        a = getattr(layer, field)
        ne0, ne1 = a.shape[-1], a.shape[0] if a.ndim == 2 else 0
        rows.append(f"{name} 0 {kind} {len(pool)} {a.nbytes} {len(pool)} {a.nbytes} {ne0} {ne1} 0 0 1 0 0 0 0 0 0 0")
        pool += a.tobytes()
    index = f"# dense index v3\n# align 64 pool {len(pool)} tensors {len(rows)}\n" + "\n".join(rows) + "\n"
    meta = {"format": "basalt", "version": 1, "tensors": tensors}
    path.write_bytes(write_basalt({"meta": json.dumps(meta).encode(), "gguf": bytes(blob), "dense": bytes(pool),
                                   "dense.index": index.encode()}))
    return path


def tensor(g, src, name):
    """The value that declares the file's tensor ``name`` exactly: a ``gguf/`` tensor a weight of its ggml type, a
    ``dense/`` row a constant of its dtype."""
    e = src.entries[name]
    ty = GGML_TYPES[e.type_id][0]
    return g.weight(name, e.shape, ty) if name.startswith("gguf/") else g.const(name, e.shape, DType.parse(ty))


def block(g, src):
    """The GDN block over ``src``: x's native q8_1 rows, the pair projection (qkv ``pair.0``, z ``pair.1``), the window
    (tests/contract/test_basalt_gdn_ops.py ``spec``: ``gdn.0.state``, the stores, ``gdn.0.yq``), the out projection."""
    x = g.input("x", (T, N), DType.F32)
    qkv, z = gemv(g, "basalt_dense_tc_pair", "pair", q8_1(g, "xq", [x]), tensor(g, src, "gguf/attn_qkv.weight"),
                  tensor(g, src, "gguf/attn_gate.weight"))
    made = {"x": x, "qkv": qkv, "z": z, **{k: tensor(g, src, f"dense/{name}") for k, (name, _, _) in DENSE.items()}}
    gemv(g, "basalt_dense_tc", "out", window(g, declare(g, spec(), made)), tensor(g, src, "gguf/ssm_out.weight"))


@pytest.fixture(scope="module", params=[False, True], ids=["windows", "exact-copies"])
def runs(request, tmp_path_factory):
    """The eight programs and engines; the bytes of every activation, store and the state after the 8-wide program's
    first run (``w8``) and after the T-wide program's run on x's first T rows (``wide[T]``)."""
    layer, x = case()
    src = BasaltSource(write_layer(tmp_path_factory.mktemp("gdn") / "gdn.basalt", layer), copy=request.param)
    graphs = {t: Graph("step") for t in WIDTHS}
    for g in graphs.values():
        block(g, src)
    progs = {t: emit_program(g, pack=src, profile=PROFILE, t=t, tail=None) for t, g in graphs.items()}
    dev = gpu.device()
    engines = {8: Engine(progs[8], dev, pipeline_cache=PIPELINES)}
    shared = {n: b for n, b in engines[8].buffers.items() if progs[8].buffers[n].role in ("weights", "state")}
    for t in WIDTHS[:-1]:
        engines[t] = Engine(progs[t], dev, buffers=shared, pipeline_cache=PIPELINES)
    shared[STATE].write(layer.state.tobytes(), 0)

    def run(t):
        eng = engines[t]
        eng.buffers["x"].write(x[:t].tobytes(), 0)
        for name in (*ACTS, *STORES):
            eng.buffers[name].fill(0xFF)
        assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
        return {name: np.frombuffer(eng.read(name), np.uint8) for name in (*ACTS, *STORES, STATE)}

    w8 = run(8)
    return {"layer": layer, "x": x, "progs": progs, "engines": engines, "w8": w8, "wide": {t: run(t) for t in WIDTHS}}


def rows(got, t, name):
    """``name``'s bytes in ``got`` (one run of the T-wide program) as float32 rows: an activation's T, a store's MT."""
    a = got[name].view(F32)
    return a.reshape(MT if name in STORES else t, -1)


@functools.lru_cache(maxsize=None)
def chain(t):
    """The per-stage chain over combo 0's first t rows of x: qkv and z, the dense_tc_pair of x's native q8_1; y,
    gdn_step_norm_multi on gdn_conv_l2_multi's h and gdn_ab_multi's gate and beta; out, dense_tc of y's native q8_1.
    The harness's GEMV outputs start as 0xFE bytes, the program's as 0xFF."""
    layer, x = case()
    pair = gpu.dense_tc_pair(layer.tq, layer.wqkv, C, layer.tz, layer.wz, ZV, gpu.q8_1(x[:t]), N, (t,), fill=0xFE)
    qkv, z = (a[:t] for a in pair[t])
    h, ok_h = gg.conv_l2(layer.state, qkv, layer.conv_w)
    gate, beta, ok_ab = gg.ab(layer, x[:t])
    y, ok_y = gg.step_norm(layer.state, h, gate, beta, z, layer.gamma)
    assert ok_h and ok_ab and ok_y, (f"T={t}: wrote past its window's rows (a sentinel byte changed): "
                                     f"gdn_conv_l2_multi {ok_h}, gdn_ab_multi {ok_ab}, gdn_step_norm_multi {ok_y}")
    return qkv, z, y, gpu.dense_tc(layer.to, layer.wo, gpu.q8_1(y), ZV, N, (t,), fill=0xFE)[t][:t]


@pytest.mark.parametrize("t", WIDTHS)
def test_gdn_block_program_matches_the_per_stage_chain_bitwise(runs, t):
    qkv, z, y, out = chain(t)
    bad = g6_blocks(y)
    assert bad == 0, (f"T={t}: fixture error, not a kernel mismatch: G6's precondition fails on {bad} q8_1 blocks of "
                      f"the chain's y (the native and the rec sequence store different bytes there)")
    got = runs["wide"][t]
    for what, ref, name, want in (("qkv", qkv, "pair.0", G1), ("z", z, "pair.1", G2), ("out", out, "out", G7)):
        d, first = cmp_f(what, ref, rows(got, t, name))
        assert d == 0, f"T={t}: {d} {what} elements differ or are not finite, first {first}; want {want}"


@pytest.mark.parametrize("t", WIDTHS)
def test_gdn_block_program_columns_do_not_depend_on_the_window_width(runs, t):
    d, first = cmp_f("out@8", rows(runs["w8"], 8, "out")[:t], rows(runs["wide"][t], t, "out"))
    assert d == 0, (f"T={t}: {d} out elements differ from the 8-wide program's rows or are not finite, first {first}; "
                    f"want {G10}")


@pytest.mark.parametrize("t", WIDTHS)
def test_gdn_window_program_matches_the_harness(runs, t):
    layer, x, got = runs["layer"], runs["x"], runs["wide"][t]
    r = gg.rec(layer, layer.state, rows(got, t, "pair.0"), x[:t], rows(got, t, "pair.1"))
    check_run(f"T={t}: the harness's dec_gdn_rec", r)
    for name, ref in zip(("gdn.0.yq", *STORES), (r.yq, r.qkv, r.h, r.gate, r.beta)):
        n = ref.nbytes                                                  # rows t < T
        assert got[name][:n].tobytes() == ref.tobytes(), f"T={t}: {name}: the program's rows differ from dec_gdn_rec's"
        assert np.all(got[name][n:] == 0xFF), f"T={t}: {name}: the program wrote a row at or past T (a 0xFF changed)"
    assert got[STATE].tobytes() == layer.state.tobytes(), f"T={t}: the program changed {STATE}"


def test_gdn_programs_share_their_states(runs):
    progs, engines = runs["progs"], runs["engines"]
    want = {n: b.nbytes for n, b in progs[8].buffers.items() if b.role == "state"}
    assert sorted(want) == sorted([STATE, *STORES]), sorted(want)
    for t, p in progs.items():
        assert {n: b.nbytes for n, b in p.buffers.items() if b.role == "state"} == want, t
        shared = [n for n, b in p.buffers.items() if b.role in ("state", "weights")]
        assert all(engines[t].buffers[n] is engines[8].buffers[n] for n in shared), t
