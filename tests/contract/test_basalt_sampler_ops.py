"""basalt's sampler as an op kind (monolith/ops/basalt_sample.py, the emitter's ``_basalt_sample``): its registration
and the coverage guard; the programs the handler emits at basalt's vocabulary, static and dynamic, without and with
the bias and history inputs (the kernels in order, their macros, StepState at slot 15, the slots, the grids, the
scratch sizes, the one params record's bytes and the barriers that order the overwrites of ``token``); the greedy-only
build (attr ``greedy_only``); the sampled path the build chooses; the request packer's refusals, ``min_p = 2``
accepted, and the mode table. No GPU."""

import math
import struct

import pytest

from monolith import kernels
from monolith.compiler import check_coverage, emit_program
from monolith.compiler.emit import HANDLERS
from monolith.core import BlockDomain, DType, Graph, OpClass, T
from monolith.core.profile import Profile
from monolith.core.step_state import StepStateLayout
from monolith.ops import OPS

PROF = Profile.from_dict("p", {"gpu_cores": 40, "nominal_gbps": 400.0,
                               "engine": {"family": "Apple9", "lane_order": "interleaved16", "threadgroups_per_core": 2}})
V = 248320                                     # basalt's vocabulary: 61 blocks of 4,096 logits
H = 64                                         # a history row's slots
RECORD = "<IIIIIIifffiifffI"                   # BsSampleParams (basalt_sampler.metal)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def graph(vocab=V, *, bias=False, hist_len=0, rows=T, logits_dtype=DType.F32, history_cols=None, greedy_only=False):
    g = Graph()
    ins = [g.input("logits", (rows, vocab), logits_dtype), g.input("ctr_base", (2,), DType.U32),
           g.input("nq", (1,), DType.U32)]
    if bias:
        ins.append(g.input("bias", (vocab,), DType.F32))
    if hist_len or history_cols:
        ins.append(g.input("history", (rows, history_cols or hist_len), DType.I32))
    token, qprob = g.value("token", (rows,), DType.I32), g.value("qprob", (rows,), DType.F32)
    attrs = dict(greedy_only=True) if greedy_only else {}
    g.op("basalt_sample", ins, [token, qprob], domain=BlockDomain("span", 1), klass=OpClass.REDUCE, bias=bias,
         hist_len=hist_len, **attrs)
    return g


def emit(g, **kw):
    return emit_program(g, pack=[], profile=PROF, tail=None, **kw)


def slots(o):
    return {b[0]: (b[1], b[2]) for b in o.bindings}


def functions(prog):
    return [prog.kernels[o.kernel].function for o in prog.ops]


def test_basalt_sample_registered_and_covered():
    od = OPS.get("basalt_sample")
    assert od.klass is OpClass.REDUCE and od.domain_kind == "span"
    assert od.binding_for("apple9").kernel == "bs_greedy_partial" and od.binding_for("apple10").kernel == "bs_greedy_partial"
    assert HANDLERS["basalt_sample"].__name__ == "_basalt_sample"     # registered and emittable
    check_coverage(graph(), PROF)


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("inputs", [False, True], ids=["plain", "bias_history"])
def test_basalt_sample_program_at_basalt_vocabulary(dynamic, inputs):
    hist_len = H if inputs else 0
    prog = emit(graph(bias=inputs, hist_len=hist_len), **(dict(dynamic_t=True) if dynamic else dict(t=8)))
    ops = prog.ops
    assert functions(prog) == ["bs_greedy_partial", "bs_greedy_merge", "bs_greedy_one_block", "bs_split_part",
                               "bs_split_merge"]
    assert [o.name for o in ops] == ["basalt_sample", "bs_greedy_merge", "bs_greedy_one_block", "bs_split_part",
                                     "bs_split_merge"]
    for o in ops:                                                     # STEP_STATE alone: no T_SRC, static or not
        assert prog.kernels[o.kernel].macros == {"STEP_STATE": "1", "BS_BIAS_BOUND": "1" if inputs else "0"}
        assert "struct StepState" in prog.kernels[o.kernel].source
        assert [b for b in o.bindings if b[0] == 15] == [(15, prog.step_state, 0)]
    assert [o.grid for o in ops] == [(8, 8, 1), (8, 1, 1), (8, 1, 1), (61, 8, 1), (8, 1, 1)]
    assert [o.threadgroup for o in ops] == [(1024, 1, 1), (32, 1, 1), (1024, 1, 1), (128, 1, 1), (32, 1, 1)]
    assert [sorted(slots(o)) for o in ops] == [[0, 7, 8, 15], [5, 7, 8, 15], [0, 3, 4, 5, 7, 10, 15],
                                              [0, 3, 4, 7, 9, 15], [1, 2, 5, 6, 7, 9, 15]]
    assert [o.meta["writes"] for o in ops] == [[8], [5], [5, 10], [9], [5, 6]]
    s = [slots(o) for o in ops]
    assert s[0][0] == s[2][0] == s[3][0] == ("logits", 0) and s[1][5] == s[2][5] == s[4][5] == ("token", 0)
    assert s[4][6] == ("qprob", 0) and s[4][1] == ("ctr_base", 0) and s[4][2] == ("nq", 0)
    for o in (s[2], s[3]):                                            # absent inputs: a 16-byte buffer, never read
        if inputs:
            assert o[3] == ("bias", 0) and o[4] == ("history", 0)
        else:
            assert prog.buffers[o[3][0]].nbytes == prog.buffers[o[4][0]].nbytes == 16
    # the one record every dispatch reads, at build: the program's structure, mode 0 (greedy), SamplerParams' defaults
    names = {o[7][0] for o in s}
    assert len(names) == 1
    name = names.pop()
    assert prog.buffers[name].role == "params" and name.split(".")[2] == "basalt_sample"   # not set_sampling's "sample"
    assert struct.unpack_from(RECORD, prog.buffers[name].init) == (
        V, 8, 61, hist_len, 0, 0, 20, f32(0.95), 0.0, 1.0, 1, 0, 1.0, 0.0, 0.0, 0)
    assert len(prog.buffers[name].init) == 64
    # the scratch: 8 (key, id) pairs per row, 64 entries per block and row, a ceil(V / 32)-word bitmap per row
    assert s[0][8] == s[1][8] and prog.buffers[s[0][8][0]].nbytes == 8 * 8 * 8
    assert s[3][9] == s[4][9] and prog.buffers[s[3][9][0]].nbytes == 8 * 61 * 64 * 8 == 249856
    assert prog.buffers[s[2][10][0]].nbytes == (8 * 7760 * 4 if inputs else 16)
    # the merge waits for the partials, and each overwrite of token for the one before it
    assert ops[1].barrier_before and ops[2].barrier_before and ops[4].barrier_before


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_basalt_sample_greedy_only(dynamic):
    """Attr greedy_only: the cluster pair alone, as the full program emits it (grids, slots, writes, the record at mode
    0), and none of the gated dispatches' scratch; bias and history are refused (mode 0 takes neither)."""
    kw = dict(dynamic_t=True) if dynamic else dict(t=8)
    prog, full = emit(graph(greedy_only=True), **kw), emit(graph(), **kw)
    assert functions(prog) == ["bs_greedy_partial", "bs_greedy_merge"]
    for o, f in zip(prog.ops, full.ops):
        assert (o.grid, o.threadgroup, o.meta["writes"], slots(o)) == (f.grid, f.threadgroup, f.meta["writes"], slots(f))
    assert prog.ops[1].barrier_before
    rec = slots(prog.ops[0])[7][0]
    assert struct.unpack_from(RECORD, prog.buffers[rec].init) == (
        V, 8, 61, 0, 0, 0, 20, f32(0.95), 0.0, 1.0, 1, 0, 1.0, 0.0, 0.0, 0)
    assert not any(tag in name for name in prog.buffers for tag in (".bs.bits", ".bs.none", ".bs.cand"))
    for bad in (dict(bias=True), dict(hist_len=H)):
        with pytest.raises(ValueError, match="greedy_only"):
            emit(graph(greedy_only=True, **bad), **kw)


def test_basalt_sample_path_choice():
    assert kernels.basalt_sampler_path(64, 262144) == "split" and kernels.basalt_sampler_path(1, 1) == "split"
    assert kernels.basalt_sampler_path(64, 262145) == "one_block" and kernels.basalt_sampler_path(65, V) == "one_block"
    assert kernels.basalt_sampler_path(70, V) == "one_block"
    wide = emit(graph(262144), t=8)
    assert functions(wide)[3:] == ["bs_split_part", "bs_split_merge"] and wide.ops[3].grid == (64, 8, 1)
    for prog, rows in ((emit(graph(262145), t=8), 8), (emit(graph(V), t=70, layout=StepStateLayout(t_max=70)), 70)):
        assert functions(prog) == ["bs_greedy_partial", "bs_greedy_merge", "bs_greedy_one_block", "bs_sample_one_block"]
        o = prog.ops[3]
        assert o.grid == (rows, 1, 1) and o.threadgroup == (1024, 1, 1) and o.meta["writes"] == [5, 6, 10]
        assert sorted(slots(o)) == [0, 1, 2, 3, 4, 5, 6, 7, 10, 15] and slots(o)[10] == slots(prog.ops[2])[10]
        assert prog.ops[0].grid == (8, rows, 1) and o.barrier_before
        assert not any(b[0] == 9 for op in prog.ops for b in op.bindings) and not any(".bs.cand" in n for n in prog.buffers)


def test_basalt_sample_handler_refuses_mismatched_graphs():
    with pytest.raises(ValueError, match="basalt_sample"):
        emit(graph(logits_dtype=DType.BF16), t=8)                     # the kernels read float32 bits
    with pytest.raises(ValueError, match="basalt_sample"):
        emit(graph(hist_len=H, history_cols=32), t=8)                 # a history narrower than hist_len
    with pytest.raises(ValueError, match="basalt_sample"):
        emit(graph(rows=4), t=8)                                      # rows not from T: the kernels read t_this_step
    g = graph()
    g.ops[0].attrs["bias"] = True                                     # a bias without its input
    with pytest.raises(ValueError, match="basalt_sample"):
        emit(g, t=8)


def test_basalt_sampler_params_refusals_and_min_p():
    with pytest.raises(ValueError):
        kernels.basalt_sampler_params(V, 8, 61, 0, penalty_last_n=1)     # a window without a history (sampler.cu:983)
    for vocab, rows in ((0, 8), (-1, 8), (V, 0), (V, -1)):              # nothing to sample (sampler.cu:981)
        with pytest.raises(ValueError):
            kernels.basalt_sampler_params(vocab, rows, 61, 0)
    for mode in (-1, 3):
        with pytest.raises(ValueError):
            kernels.basalt_sampler_params(V, 8, 61, 0, mode=mode)
    kernels.basalt_sampler_params(V, 8, 61, H, penalty_last_n=H)           # with a history
    kernels.basalt_sampler_params(V, 8, 61, 0, penalty_last_n=-5)          # no window: basalt tests last_n > 0
    rec = kernels.basalt_sampler_params(V, 8, 61, 0, mode=2, min_p=2.0, bias=True, probe=True)
    assert struct.unpack_from(RECORD, rec)[4:10] == (2, 3, 20, f32(0.95), 2.0, 1.0)   # min_p above 1 is basalt's to accept


def test_basalt_coupled_node_refusals():
    """basalt_coupled_node refuses what coupled_draft_sample exits on (sampler.cu:1057-1060): no logits, or more blocks
    than the merge's 64 lists; 64 blocks exactly is the widest node."""
    for nv in (0, -1, 64 * 4096 + 1):
        with pytest.raises(ValueError, match="basalt_coupled"):
            kernels.basalt_coupled_node(nv, nv, 256, 0)
    assert struct.unpack("<8I", kernels.basalt_coupled_node(64 * 4096, 106299, 256, 3, sub_to_id=True)) == (
        64 * 4096, 106299, 256, 3, 2, 0, 0, 0)


def test_basalt_sampler_mode_table():
    """The mode rule against basalt's verify window (verify.cpp:2496-2498, :2659; sampler.cu:991)."""
    nan, inf = float("nan"), math.inf
    table = [((True, 1.0, 0, False), 0), ((False, 0.0, 0, False), 0), ((False, -0.0, 0, False), 0),
             ((False, -1.0, 0, False), 0), ((False, -inf, 0, False), 0),
             ((False, 1e-50, 0, False), 0),            # the record's float32 is +0
             ((False, 1e-40, 0, False), 2),            # a float32 subnormal: above 0 on basalt's host
             ((False, nan, 0, False), 0),              # a NaN temperature is not sampled: the captured greedy pick
             ((False, nan, H, False), 2),              # with a window the probe's branch runs, and samples it
             ((False, nan, 0, True), 2),               # (not greedy, and NaN <= 0 is false); with a bias too
             ((True, nan, H, False), 1),               # greedy with a window: the one-block greedy kernel
             ((True, nan, 0, False), 0), ((False, 1.0, 0, False), 2), ((False, inf, 0, False), 2),
             ((True, 1.0, H, False), 1), ((True, 1.0, 0, True), 1), ((False, 0.0, H, True), 1),
             ((False, 0.0, -5, False), 0),             # penalty_last_n <= 0: no window
             ((False, 1.0, H, True), 2)]               # sampled: the penalties and the bias are the chain's
    assert [kernels.basalt_sampler_mode(*args) for args, _ in table] == [mode for _, mode in table]
