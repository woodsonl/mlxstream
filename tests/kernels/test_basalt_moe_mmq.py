# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) tests/cuda/prefill_mmq_kq_test.cpp and tests/cuda/prefill_cpu_share_test.cpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures (tests/kernels/
# basalt_moe_fixture.py) and their checks on the routed experts' MMQ.
"""basalt's MoE MMQ for the routed experts on Q2_0 (src/prefill/moe_mmq.cu, src/prefill/mmq/) and its Q8_0 instance:
the tests that run on the host (the emulator's launch walk, the fixture, and the emulator's own gates before any Metal
exists), then the Metal port's (kernels/common/basalt_mmq.metal).

- test_mmq_walk_facts: ``emu.pick_j`` is mul_mat_q_switch_J's tile width (mmq.cuh:867-889) where the first width
  reaching the fewest column tiles is not 128, and ``emu.mmq_walk`` gives tile 0's pieces, in the fixup's order, for
  six launches at basalt's 170 multiprocessors.
- test_mmq_q2_streamk_pin: the chains are the reference GPU's: in each product some launch of the fixtures has other
  chains at this Mac's 40 GPU cores (the analogue of tests/contract/test_basalt_ops.py's
  test_dense_tc_order_is_basalts_not_this_gpus).
- test_moe_fixture_dequant_matches_gguf_reference: the fixture's vectorized dequantizer equals
  tests/contract/gguf_reference.py's per-value transcription of dequantize_row_q2_0 bit for bit.
- test_moe_fixture_configs: every configuration builds (and so meets its coverage assertions), the blobs sit where
  cpu_share :259-264 puts them with 0xFF everywhere else, and the configurations' plans (``emu.moe_plan``, basalt's host
  walk prefill.cpp:1840-1996) equal an independent stable sort, with and without CPU-held experts.
- test_mmq_q2_emulator_float64: the host emulator ``emu.mmq_moe``, which the Metal chain must equal bit for bit,
  meets G1-G3 (prefill_mmq_kq_test.cpp:269, :277-278, :375-377) against the float64 reference.
- test_mmq_q2_batch_independence_emulator: MK-4's bound (:433) measured on the emulator first (L13).
- test_mmq_reciprocal_division_counts: how often the port's reciprocal-and-multiply form of the quantizer's and
  SwiGLU's divisions differs from the correctly rounded quotient (contract invariant 5's count).
- test_mmq_launch_and_pieces (M3, spec-kernels-mmq.md 10): the kernels' launch helpers mmq_launch and mmq_piece_mask,
  run on the GPU by a probe kernel, equal emu.mmq_launches and the pieces of emu.mmq_walk on every tile of every launch
  of scripts/mmq_pieces.py's sweep and of the fixtures' plans.
- test_mmq_quantize_edges (M1): kernel 1, basalt_mmq_quantize, alone: Xq's bytes equal emu.mmq_quantize's on every
  slot through the plan's src, the edge groups (NaN, subnormal, d_inv = +inf, +-inf, roundf's ties) hold their literal
  codes and scales, and slots at or past bounds[n] keep the 0xFF fill, with and without experts held on the CPU.
- test_mmq_swiglu_quantize_fused (M2): kernel 3, basalt_mmq_swiglu_quantize (mmq::swiglu fused with the quantizer on
  H), alone: Hq's bytes equal the separate path's (emu.mmq_swiglu's stored H, then emu.mmq_quantize, with the GPU's
  expf_ftz) on g and u's edge values, each pair as a whole group and as one value among plain ones, and on plain rows;
  the uniform groups' NaN, infinite, zero and finite h give their literal bytes, the padding blocks are zero, and slots
  at or past bounds[n] keep the 0xFF fill, with and without experts held on the CPU.
- test_mmq_swiglu_edges (SG-1, spec-tests-moe.md 7.2): kernel 3 on the edge pairs where flushing decides h, each alone
  in its group: Hq's bytes equal the emulator's, and the emulator without the flushes (a port that keeps subnormals)
  differs in exactly the first two edge groups.
- test_mmq_plan_matches_host_walk: the plan kernels basalt_mmq_plan_count, _scan and _fill (basalt's host expert walk,
  prefill.cpp:1840-1932, :1993-1996; integers, so bitwise) equal emu.moe_plan word for word, slot, src, n,
  rows_local, order, bounds and wpre, at four chunk sizes on route-like ids and on ids repeating an expert inside a
  token, on a stale mmq_hist, with experts held on the CPU through cpu_mask, and on every fixture; every word past
  them keeps the 0xFF fill.
- test_mmq_plan_routed_id_out_of_range: a routed id outside [0, n_expert) in the chunk's partial last range sets
  StepState's error to BASALT_ERR_EXPERT_ID with done, and slot, src and plan keep their 0xFF fill (basalt refuses at
  prefill.cpp:1843, before its first write); a step that arrives with done set stores nothing at all.
- test_mmq_q2_emulation_bit_for_bit (E1-E5, spec-tests-moe.md 5.4): the chain, the plan kernels then kernels 1-4
  (basalt_mmq: every launch's mul_mat_q<Q2_0> with its stream-k fixup folded per output element, in one dispatch a
  product) in one command buffer, on each fixture configuration: Xq's bytes, GU's bits, Hq's bytes and D's bits equal
  the host emulator's (emu.mmq_moe's steps at basalt's 170 multiprocessors, with the GPU's expf_ftz), slots past
  rows_local keep the 0xFF fill, and the launches cover split and unsplit ones, tiles of three or more pieces and an
  empty column tile.
- test_mmq_down_padded_steps_signed_zero (M5): kernel 4 alone on crafted Hq and down weights whose chains are -0
  before the down product's padded sub-blocks: the scales mmq_dA finds past each row (the next row's, the next launch
  slot's row 0, the zeroed tail past a group's last slot) decide the zero's sign as basalt's group buffer does.
- test_mmq_chain_routed_id_out_of_range: a routed id out of range through the whole chain stores nothing, not even
  over an earlier chunk's plan, which kernels 1-4 would otherwise run on.
- test_mmq_q2_float64_reference: the Metal chain meets G1-G3 (prefill_mmq_kq_test.cpp:269, :277-278, :375-377) against
  the float64 reference.
- test_mmq_q2_batch_independence: MK-4 (:433) on the Metal chain, every run through one program, with I1: rows whose
  tile chains are their 256-token run's are bitwise equal to it.
- test_mmq_order_is_basalts_not_this_gpus (M4): MMQ_SM_COUNT is basalt's 170 unless asked, and it reaches the
  arithmetic: built at this Mac's 40 cores, kernel 2 gives other bits, the emulator's at 40.
- test_mmq_nan_reaches_d: a NaN group, a +inf and a -inf activation make their slots' GU and D NaN, each stored as
  sm_120's 0x7fffffff, and the chain's bytes equal the emulator's; on the Q8_0 instance their rows' y, equal to the
  Q2_0 instance's.
- test_mmq_q8_0_equals_q2_0_tiles: the Q8_0 instance (basalt_prompt_mmq's pair: kernel 1 on rows by position, kernel 2
  with MMQ_TYPE 8 and one expert, draft_kv's proj, prefill.cpp:971-980) on a Q8_0 matrix holding a Q2_0 matrix's values
  gives y bit-equal to the Q2_0 instance's (kernel 2 with the gate/up product's macros) on the same Xq rows, and its Xq
  equals the emulator's (spec-engine.md 7.2).
- test_mmq_q8_0_float64_reference: the Q8_0 instance at draft_kv's shapes meets G1, G2 and G3's plain bounds
  (prefill_mmq_kq_test.cpp:269, :277-278, :375-377) against the float64 product with ggml's dequantized weights."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu
from tests.kernels import basalt_moe_fixture as fx
from tests.kernels import basalt_prompt_gpu as gpu
from tests.kernels.basalt_cpu_emu import unpack64_q2_0

F32 = np.float32
NSM = kernels.DENSE_TC_SM_COUNT       # basalt's reference GPU, which the port pins (never this Mac's)
THIS_MAC = 40                         # the GPU cores of the M3 Max these tests run on


def test_mmq_walk_facts():
    """pick_j (mul_mat_q_switch_J with ncols_opt = maxr) where the first width reaching the fewest column tiles is not
    128 (200 rows: 112, 129: 80, 257: 96), and mmq_walk's tile 0 at basalt's 170 multiprocessors in six launches of
    one-row experts (J = 8, ntx = 1): the full groups unsplit (gate/up 160 tiles, down 320 and 160), 150 gate/up tiles
    cut at block 32, one expert's 10 gate/up tiles cut at every iteration and its 20 down tiles at blocks 4 and 8."""
    assert (emu.pick_j(200), emu.pick_j(129), emu.pick_j(257)) == (112, 80, 96)
    table = [("gate_up", 16, 160, emu.mmq_fixup((0, 40))), ("down", 16, 320, emu.mmq_fixup((0, 10))),
             ("down", 8, 160, emu.mmq_fixup((0, 10))), ("gate_up", 15, 170, emu.mmq_fixup((0, 32), (32, 40))),
             ("gate_up", 1, 170, emu.mmq_fixup(*[(4 * q, 4 * q + 4) for q in range(10)])),
             ("down", 1, 170, emu.mmq_fixup((0, 4), (4, 8), (8, 10)))]
    for product, ngx, grid, tile0 in table:
        nty, bpne = fx.PRODUCTS[product]
        (la,) = emu.mmq_launches(np.arange(ngx + 1), nty, NSM)
        assert (la.J, la.ntx, la.ntiles, la.grid) == (8, 1, nty * ngx, grid), (product, ngx)
        assert emu.mmq_walk(la.ntiles, bpne, la.grid)[0] == tile0, (product, ngx)


def test_mmq_q2_streamk_pin():
    """E5's pin: the stream-k grid follows the multiprocessor count (mmq.cuh:830-833), so in each product some launch
    of the fixtures (configurations x groups) has other chains at this Mac's 40 cores than at basalt's 170, while
    the full groups run unsplit at both."""
    for product in fx.PRODUCTS:
        differ = []
        for name in fx.CONFIGS:
            cfg = fx.config(name)
            for la, here in zip(fx.launches(cfg, product), fx.launches(cfg, product, THIS_MAC)):
                if la.tiles != here.tiles:
                    differ.append((name, la.ngx, la.grid, here.grid))
                elif la.ngx == emu.MMQ_GROUP:
                    assert la.grid == la.ntiles == here.grid, (name, product, la.j0)
        print(product, "launches whose chains differ at", THIS_MAC, "cores:", differ)
        assert differ, product


def test_moe_fixture_dequant_matches_gguf_reference():
    """The fixture's dequant_q2_0 equals gguf_reference.dequantize(42, rows) (tests/contract/gguf_reference.py's q2_0,
    one value at a time) bit for bit, compared as uint32, on 512 blocks of random bytes whose scales hold every fp16
    class: both zeros, subnormals of both signs, normals, both infinities, quiet and signalling NaNs of both signs."""
    rng = np.random.default_rng(0)
    b = rng.integers(0, 256, (512, fx.QB), dtype=np.uint8)
    planted = [0x0000, 0x8000, 0x0001, 0x83FF, 0x7C00, 0xFC00, 0x7E00, 0xFE01, 0x7C01, 0xFD55]
    b[:len(planted), :2] = np.array(planted, "<u2").view(np.uint8).reshape(-1, 2)
    h = b[:, :2].copy().view("<u2")[:, 0]
    exp, man = h & 0x7C00, h & 0x3FF
    sub, inf, nan = (exp == 0) & (man != 0), (exp == 0x7C00) & (man == 0), (exp == 0x7C00) & (man != 0)
    assert {0, 0x8000} <= set(h.tolist()) and all(len(set(h[c] >> 15)) == 2 for c in (sub, inf, nan))
    rows = b.reshape(8, 64 * fx.QB)
    with np.errstate(invalid="ignore"):                    # the NaN scales
        got, want = fx.dequant_q2_0(rows, 8, 64 * fx.QK), gguf_reference.dequantize(42, rows)
    assert np.isnan(want).any() and np.array_equal(got.view(np.uint32), want.view(np.uint32))


def _sorted_rows(ids, n_expert, cpu=()):
    """The routed rows by a stable sort, apart from moe_plan's walk: the GPU's experts in id order, then the CPU's in id
    order, each expert's pairs in pair order (prefill_cpu_share_test's sort_rows, :126-145, with the CPU block last)."""
    flat = np.asarray(ids).ravel()
    key = np.where(np.isin(flat, list(cpu)), flat + n_expert, flat)
    perm = np.argsort(key, kind="stable")
    slot = np.empty(flat.size, np.int64)
    slot[perm] = np.arange(flat.size)
    order = np.array(sorted(set(flat.tolist()) - set(cpu)), np.int64)
    local = int(np.count_nonzero(key < n_expert))
    bounds = np.append(np.searchsorted(key[perm], order), local)
    return slot.reshape(np.shape(ids)), perm // np.shape(ids)[1], order, local, bounds


def test_moe_fixture_configs():
    """Every configuration builds, which asserts its coverage (basalt_moe_fixture._coverage: share with tile width 8 and
    a split short last group, kq with 61 experts, tile width 128 on the hot group and its 13-expert last group split,
    wide with ntx > 1 and an empty column tile). Its blobs sit at e * (BYTES + 64) + 2 * (e % 2), hold the generator's
    bytes (scales in [0.004, 0.024] at bytes 0-1 of every block) and every other byte is 0xFF. Its plan equals the
    stable sort; so do moe_plan's on random ids with experts repeated inside a token, with and without CPU-held experts
    (one of them without rows), and an id out of range raises basalt's error (prefill.cpp:1843)."""
    for name, (T, hot, unused, outliers, maxt, seed) in fx.CONFIGS.items():
        cfg = fx.config(name)
        assert cfg.x.shape == (T, fx.N) and not cfg.x[5].any() and cfg.ids.shape == (T, fx.K)
        assert all(len(set(r)) == fx.K for r in cfg.ids.tolist()) and cfg.ids.max() < fx.E - unused
        e = np.arange(fx.E)
        assert np.array_equal(cfg.offsets, e * (fx.BYTES + fx.SLACK) + 2 * (e % 2))
        rest = np.ones(cfg.buf.size, bool)
        rng = np.random.default_rng(seed)
        for i in range(fx.E):
            assert np.array_equal(fx.blob(cfg, i), fx.make_blob(rng)), (name, i)
            rest[cfg.offsets[i]: cfg.offsets[i] + fx.BYTES] = False
        assert np.all(cfg.buf[rest] == 0xFF) and np.count_nonzero(rest) == fx.E * fx.SLACK, name
        d = np.stack([fx.blob(cfg, i) for i in range(0, fx.E, 21)]).reshape(-1, fx.QB)[:, :2].copy().view("<f2")
        assert d.min() >= np.float16(0.004) and d.max() <= np.float16(0.024), name
        slot, src, order, local, bounds = _sorted_rows(cfg.ids, fx.E)
        p = cfg.plan
        assert (p.n, p.rows_local) == (len(order), T * fx.K) and np.array_equal(p.order, order), name
        assert np.array_equal(p.slot, slot) and np.array_equal(p.src, src) and np.array_equal(p.bounds, bounds), name
        assert np.array_equal(p.wpre, np.append(0, np.cumsum((np.diff(bounds) + 7) // 8))), name
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 64, (37, 10))
    for cpu in ((), (3, 17, 40, 70)):                      # 70 routes no row
        slot, src, order, local, bounds = _sorted_rows(ids, 80, cpu)
        p = emu.moe_plan(ids, 80, cpu)
        assert (p.n, p.rows_local) == (len(order), local) and np.array_equal(p.order, order), cpu
        assert np.array_equal(p.slot, slot) and np.array_equal(p.src, src) and np.array_equal(p.bounds, bounds), cpu
    assert any(len(set(r)) < 10 for r in ids.tolist()) and local < ids.size
    for bad in (64, -1):
        with pytest.raises(ValueError, match="prefill: routed id out of range"):
            emu.moe_plan(np.where(np.arange(20).reshape(2, 10) == 13, bad, 0), 64)


def _host_expf(x):
    """The host's stand-in for basalt_fp.metal's expf_ftz in the tests that run without the GPU: exp in float64,
    rounded once to float32, then flushed. fast::exp2, which the kernel's expf_ftz uses for MUFU.EX2, has no bit-exact
    host model (basalt_emu.swiglu_q8_1); the bitwise tests take the GPU's values (basalt_gpu.expf_ftz)."""
    with np.errstate(over="ignore", under="ignore"):
        return emu.ftz(np.exp(np.asarray(x, F32).astype(np.float64)).astype(F32))


@functools.lru_cache(maxsize=None)
def _emulated(name):
    """basalt_emu.mmq_moe on a configuration at basalt's 170 multiprocessors with _host_expf (GU does not depend on
    it); cached, as G1-G3, MK-4's full run and the division counts read the same run."""
    cfg = fx.config(name)
    return emu.mmq_moe(cfg.x, cfg.plan, cfg.buf, cfg.offsets, expf=_host_expf, nsm=NSM)


def _float64_gates(cfg, dm):
    """G1-G3 on Dm ``dm`` (sorted rows) against the fixture's float64 reference, in basalt's order: G1, basalt's throw
    at a non-finite output (``if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");``,
    prefill_mmq_kq_test.cpp:269, prefill_cpu_share_test.cpp:224), counted before compare; G2, the zero token's ten rows
    exactly zero (``else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero``, :277-278); G3,
    ``const double lim_rms = outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;`` (:375), failing if
    ``em.rms > lim_rms || em.worst > lim_worst`` (:376-377)."""
    bad = np.count_nonzero(~np.isfinite(dm))
    nonzero = [k for k in range(fx.K) if np.any(dm[cfg.plan.slot[5, k]] != 0)]
    em = fx.compare(dm, fx.reference(cfg.name), cfg.plan.slot)
    lim_rms, lim_worst = (3e-2, 0.15) if cfg.outliers else (2e-2, 0.1)
    print(f"\n{cfg.name}: rel RMS {em.rms:.3e} (bound {lim_rms})  worst row {em.worst:.3e} (bound {lim_worst})  "
          f"max rel {em.rmax:.3e}  non-finite {bad}")
    assert bad == 0, f"{cfg.name}: {bad} non-finite outputs"
    assert not nonzero, f"{cfg.name}: the zero token's rows for slots {nonzero} are not zero (worst = 1e30)"
    assert not (em.rms > lim_rms or em.worst > lim_worst), (cfg.name, em.rms, em.worst)


@pytest.mark.parametrize("name", ["share-plain", "share-outliers", "kq-plain", "kq-outliers"])
def test_mmq_q2_emulator_float64(name):
    """nearest bound; no basalt test launches this kernel: G3 is prefill_mmq_kq_test.cpp:375's
    ``const double lim_rms = outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;``, which basalt gates on its
    K-quant MMQ and never on Q2_0 (the contract's MMQ-on-Q2_0 row, docs/design/flash-next.md:451). The host emulator
    basalt_emu.mmq_moe, which the Metal chain must equal bit for bit (E1-E4), meets G1-G3 as
    test_mmq_q2_float64_reference states them (_float64_gates), before any Metal exists."""
    _float64_gates(fx.config(name), _emulated(name).d)


@pytest.mark.parametrize("kind", ["plain", "outliers"])
def test_mmq_q2_batch_independence_emulator(kind):
    """nearest bound; no basalt test launches this kernel: MK-4's ``if (worst_alone > 0.1 * em.worst)``
    (prefill_mmq_kq_test.cpp:433), which basalt runs only on its K-quant pairs (:525-527), measured on Q2_0 with the
    host emulator before any Metal (L13; eng-review D24): the fixture's mk4 on the kq configuration, tokens 0..7 alone
    and last of three against the 256-token run, basalt_emu.mmq_moe as the runner. Also counts the small runs'
    non-finite values, which MK-4 lets through: 0. The factor 0.1 is basalt's and is never loosened here; a failure
    goes to the owner with the measured ratio."""
    cfg = fx.config(f"kq-{kind}")

    def run(x, ids):                                       # the configuration's own run is the cached one
        r = _emulated(cfg.name) if x is cfg.x else emu.mmq_moe(x, emu.moe_plan(ids, fx.E), cfg.buf, cfg.offsets,
                                                                expf=_host_expf, nsm=NSM)
        return r.d

    m = fx.mk4(cfg, run)
    print(f"\n{cfg.name}: worst_alone {m.worst_alone:.3e}  em.worst {m.em.worst:.3e}  worst_alone / em.worst "
          f"{m.worst_alone / m.em.worst:.3e} (fails above 0.1); {m.differ} of 160 rows differ; {m.nonfinite} non-finite")
    assert m.nonfinite == 0, (cfg.name, m.nonfinite)
    assert not m.worst_alone > 0.1 * m.em.worst, (cfg.name, m.worst_alone, m.em.worst)


def test_mmq_reciprocal_division_counts():
    """Contract invariant 5's count (docs/design/flash-next.md:130-134; no basalt test: basalt has no quotient to
    compare) for the PR's two divisions that nvcc compiles as MUFU.RCP and a multiply, which the port maps to rcp_ftz
    (the correctly rounded reciprocal, flushed) and mul_ftz. The quantizer's ``const float d_inv = 127.0f / amax;``
    (quantize.cu:51) on the kq fixtures' X, per 32-value group: mul_ftz(127, rcp_ftz(amax)) against
    ftz(fl(127 / ftz(amax))), with the codes mmq_code(mul_ftz(x, d_inv)) that change (:53-56), each by one. SwiGLU's
    ``h[i] = g / (1.0f + __expf(-g)) * u;`` (moe_mmq.cu:92) on the emulator's GU: mul_ftz(g, rcp_ftz(add_ftz(1, e)))
    against ftz(fl(g / add_ftz(1, e))), before and after the ``* u``. Both sides take the same e (the host's exp,
    _host_expf), so the counts measure the mapping alone and need no GPU. ``d = 1.0f / d_inv`` (:57) is a reciprocal,
    which rcp_ftz rounds correctly: no count."""
    for name in ("kq-plain", "kq-outliers"):
        v = fx.config(name).x.reshape(-1, 32)
        amax = np.fmax.reduce(emu.ftz(np.abs(v)), axis=1)
        port = emu.mul_ftz(F32(127.0), emu.rcp_ftz(amax))
        with np.errstate(divide="ignore"):
            exact = emu.ftz(F32(127.0) / emu.ftz(amax))
        cp, ce = (emu.mmq_code(emu.mul_ftz(v, d[:, None])).astype(np.int64) for d in (port, exact))
        nd = np.count_nonzero(port.view(np.uint32) != exact.view(np.uint32))
        nc = np.count_nonzero(cp != ce)
        gu = _emulated(name).gu
        g, u = gu[:, :fx.FF], gu[:, fx.FF:]
        t = emu.add_ftz(F32(1.0), _host_expf(-g))
        qp, qe = emu.mul_ftz(g, emu.rcp_ftz(t)), emu.ftz(g / t)
        nq = np.count_nonzero(qp.view(np.uint32) != qe.view(np.uint32))
        nh = np.count_nonzero(emu.mul_ftz(qp, u).view(np.uint32) != emu.mul_ftz(qe, u).view(np.uint32))
        print(f"\n{name}: quantizer d_inv: {nd} of {amax.size} groups ({nd / amax.size:.3%}) differ from fl(127 / amax), "
              f"{nc} of {v.size} codes ({nc / v.size:.4%}) change; SwiGLU: {nq} of {g.size} quotients "
              f"({nq / g.size:.3%}) differ from fl(g / (1 + e)), {nh} after * u ({nh / g.size:.3%})")
        assert np.all(np.abs(cp - ce) <= 1), name


# M3's probe (spec-kernels-mmq.md 10): item i = (first, n, g, t) runs the helpers as basalt_mmq runs them per
# threadgroup, mmq_launch on group g of the n launch slots whose n + 1 bounds start at bounds[first], then
# mmq_piece_mask on tile t of that launch, with the product's macros.
MMQ_PROBE = """
kernel void mmq_launch_probe(device const int* bounds [[buffer(0)]], device const int4* item [[buffer(1)]],
                             device int* out [[buffer(2)]], constant uint& count [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const int4 a = item[i];
    const MmqLaunch L = mmq_launch(bounds + a.x, a.y, a.z, NTY);
    device int* o = out + 7 * i;
    o[0] = L.ngx; o[1] = L.maxr; o[2] = L.J; o[3] = L.ntx; o[4] = L.ntiles; o[5] = L.grid;
    o[6] = int(mmq_piece_mask(a.w, L.ntiles, BPNE, BPI, L.grid));
}
"""
SWEEP_MAXR = list(range(1, 257)) + [384, 512, 1000, 2048]       # scripts/mmq_pieces.py's launches()


def _sweep_plans():
    """(bounds, g) for every launch shape of the sweep, ngx 1-16 experts with maxr in SWEEP_MAXR, plus maxr 16,384 at
    ngx 16 and 65,536 at ngx 1. Shape c is group g = c % 3 of a plan of 16 g + ngx launch slots (more after a full
    group when c % 4 != 0): the other groups' experts hold more rows than maxr, the group's own a uniform count in
    [1, maxr] with maxr at a random slot, so a helper that reads outside its group or misses its largest expert errs."""
    rng = np.random.default_rng(3)
    shapes = [(ngx, m) for ngx in range(1, 17) for m in SWEEP_MAXR] + [(16, 16384), (1, 65536)]
    for c, (ngx, maxr) in enumerate(shapes):
        g = c % 3
        cnt = maxr + 1 + rng.integers(0, 8, 16 * g + ngx + (c % 4 if ngx == 16 else 0))
        cnt[16 * g: 16 * g + ngx] = rng.integers(1, maxr + 1, ngx)
        cnt[16 * g + rng.integers(0, ngx)] = maxr
        yield np.append(0, np.cumsum(cnt)), g


def _piece_masks(tiles, bpi):
    """Per tile of an emu.mmq_walk launch, bit q set where a piece of its K range starts at k iteration q > 0, after
    asserting that the walk's expression is mmq_fixup's fold of its pieces, each from an iteration boundary (so the
    starts fix the expression the kernel forms, spec-kernels-mmq.md 9.2)."""
    out = np.empty(len(tiles), np.int32)
    for t, expr in tiles.items():
        p = sorted(emu.mmq_pieces(expr))
        assert expr == emu.mmq_fixup(*p) and all(lo % bpi == 0 for lo, _ in p), (t, expr)
        out[t] = sum(1 << (lo // bpi) for lo, _ in p[1:])
    return out


@pytest.mark.parametrize("product,n,nty,bpne,bpi", [("gate_up", 0, 10, 40, 4), ("down", 0, 20, 10, 4),
                                                     ("q8_0", 2560, 20, 80, 8), ("q8_0", 512, 4, 80, 8)],
                         ids=["gate_up", "down", "q8_0-2560", "q8_0-512"])
def test_mmq_launch_and_pieces(product, n, nty, bpne, bpi):
    """M3 (spec-kernels-mmq.md 10): basalt_mmq.metal's mmq_launch and mmq_piece_mask (9.3), compiled from
    basalt_mmq_source() with the product's macros and run on the GPU by a probe kernel, equal the host's transcription
    of basalt's launch rules (emu.mmq_launches: MMQ_GROUP experts a launch, prefill.cpp:2243-2248; mul_mat_q_switch_J's
    tile width, mmq.cuh:866-889; ntx, the tile count and the stream-k grid at basalt's 170 multiprocessors, :801-833)
    and the pieces of the literal walk of mul_mat_q and its fixup (emu.mmq_walk, :487-629, :654-698, :765), on every
    tile of every launch: scripts/mmq_pieces.py's sweep (_sweep_plans: ngx 1-16, maxr 1-256, 384, 512, 1000, 2048),
    maxr 16,384 at ngx 16 and 65,536 at ngx 1 (x * grid past 2^32: the 64-bit products), and every group of the five
    fixtures' plans. The Q8_0 instance runs the same helpers with bpne 80 and bpi 8 at draft_kv's widths, 2560 and 512
    rows."""
    macros = kernels.basalt_mmq_macros(product, n=n, k=2560 if n else 0)
    assert (macros["NTY"], macros["BPNE"], macros["BPI"]) == (str(nty), str(bpne), str(bpi))
    plans = list(_sweep_plans()) + [(fx.config(name).plan.bounds, g) for name in fx.CONFIGS
                                    for g in range(-(-fx.config(name).plan.n // emu.MMQ_GROUP))]
    bounds, items, want, masks, first, wide = [], [], [], {}, 0, 0
    for b, g in plans:
        la = emu.mmq_launches(b, nty, NSM)[g]
        if (la.ntiles, la.grid) not in masks:
            masks[la.ntiles, la.grid] = _piece_masks(emu.mmq_walk(la.ntiles, bpne, la.grid, bpi), bpi)
        t = np.arange(la.ntiles)
        items.append(np.stack([np.full_like(t, first), np.full_like(t, len(b) - 1), np.full_like(t, g), t], axis=1))
        want.append(np.column_stack([np.tile([la.ngx, la.maxr, la.J, la.ntx, la.ntiles, la.grid], (la.ntiles, 1)),
                                     masks[la.ntiles, la.grid]]))
        bounds.append(b)
        first += len(b)
        wide = max(wide, la.ntiles * bpne * la.grid)
    items, want = np.concatenate(items).astype(np.int32), np.concatenate(want).astype(np.int32)
    pso = gpu.prompt_pipeline(kernels.basalt_mmq_source() + MMQ_PROBE, "mmq_launch_probe", 256, macros)
    out = basalt_gpu.run(pso, [np.concatenate(bounds).astype(np.int32).tobytes(), items.tobytes(), want.nbytes],
                         np.uint32(len(items)).tobytes(), 3, -(-len(items) // 256), 256, fill=0xFF)[2]
    got = np.frombuffer(out, np.int32).reshape(want.shape)
    bad = np.flatnonzero((got != want).any(axis=1))
    print(f"\n{product} (NTY {nty}): {len(plans)} launches, {len(items)} tiles ({np.count_nonzero(want[:, 6])} split), "
          f"{len(masks)} (ntiles, grid) pairs; {len(bad)} tiles differ")
    assert wide >= 2 ** 32, wide
    assert not len(bad), [(items[i].tolist(), got[i].tolist(), want[i].tolist()) for i in bad[:5]]


def test_mmq_quantize_edges():
    """M1 (spec-kernels-mmq.md 10): kernel 1, basalt_mmq_quantize (quantize_mmq_q8_1<D4>, quantize.cu:10-84), alone at
    bound B = 4 on two tokens whose 20 slots read them through the plan's src = [1, 0, 0, 1, ...]. Xq's bytes equal
    basalt_emu.mmq_quantize's (spec-tests-moe.md 5.4 step 1) on every slot, slots p >= bounds[n] keep the 0xFF fill, and
    the edge groups, each token's at other lanes, SIMD-groups and threadgroups, hold their literal bytes: all NaN (d4
    0x7fffffff, codes 0); one NaN among normals (its code 0); all subnormal (flushed: d4 +0, codes 0); tiny normals,
    amax < 127 / FLT_MAX, so d_inv = +inf (codes 0xFF for positive values and 0x00 for negative ones, F2I's saturation,
    0 for the zeros and for a flushed positive subnormal; d4 +0); a +inf, a -inf (d_inv 0: d4 +inf, codes 0); amax 127,
    where d_inv = fl(127 fl(1 / 127)) is 1 exactly, holding +-0.5, +-2.5, +-126.5 and 0.49999997 (roundf's ties away
    from zero and its add rounded toward zero: codes +-1, +-3, +-127, 0). A second plan holds two experts on the CPU
    (rows_local 18 < 10 R): their last block of slots keeps the fill too, as no GPU kernel reads it."""
    rng = np.random.default_rng(4)
    B, T = 4, 2
    bits = lambda b: np.asarray(b, np.uint32).view(F32)
    sign = lambda: rng.integers(0, 2, 32, dtype=np.uint32) << np.uint32(31)
    nan = bits(sign() | 0x7F800000 | rng.integers(1, 1 << 23, 32, dtype=np.uint32))     # quiet and signalling
    one_nan = rng.standard_normal(32, dtype=F32)
    one_nan[13] = bits(0xFFC00001)                                                        # a quiet NaN
    sub = bits(sign() | rng.integers(1, 1 << 23, 32, dtype=np.uint32))
    tiny = (rng.uniform(1, 8, 32) * 2.0 ** -126 * rng.choice([-1, 1], 32)).astype(F32)
    tiny[[3, 11, 17, 20]] = F32(2.0 ** -122), F32(0.0), F32(-0.0), bits(5)              # amax; zeros; a subnormal
    pinf, ninf = rng.standard_normal((2, 32), dtype=F32)
    pinf[7], ninf[30] = np.inf, -np.inf
    ties = rng.uniform(-126, 126, 32).astype(F32)
    at = [29, 0, 4, 9, 14, 19, 24, 31]                   # 127 in the group's last thread: amax crosses the butterfly
    ties[at] = 127, 0.5, -0.5, 2.5, -2.5, 126.5, -126.5, bits(0x3EFFFFFF)
    zeros = dict.fromkeys(range(32), 0)
    groups = [(nan, 0x7FFFFFFF, zeros), (one_nan, None, {13: 0}), (sub, 0, zeros),
              (tiny, 0, dict(enumerate(np.where(tiny >= 2.0 ** -126, -1, 0)))), (pinf, 0x7F800000, zeros),
              (ninf, 0x7F800000, zeros), (ties, 0x3F800000, dict(zip(at, (127, 1, -1, 3, -3, 127, -127, 0))))]
    where = [[0, 5, 10, 15, 22, 47, 79], [41, 46, 51, 56, 63, 8, 40]]   # 32-value group gi of each token's row: lanes
    x = rng.standard_normal((B, fx.N), dtype=F32)                      # 8 (gi % 4) .., block gi // 4, threadgroup gi // 16
    for t in range(T):
        for gi, (v, _, _) in zip(where[t], groups):
            x[t, 32 * gi: 32 * gi + 32] = v
    ids = np.array([[e for e in range(20) if e % 4 in (1, 2)], [e for e in range(20) if e % 4 in (0, 3)]])
    want = emu.mmq_quantize(x[:T])
    for cpu, local in (((), 10 * T), ((5, 12), 18)):
        plan = emu.moe_plan(ids, 512, cpu)
        assert plan.bounds[-1] == plan.rows_local == local
        got = gpu.mmq_quantize(x, plan, B=B)
        bad = np.argwhere(got[:local] != want[plan.src[:local]])
        assert not len(bad), f"cpu {cpu}: {len(bad)} bytes differ from the emulator, first (slot, block, byte) {bad[0]}"
        assert np.all(got[local:] == 0xFF), f"cpu {cpu}: a slot at or past bounds[n] = {local} was written"
        if not cpu:
            assert plan.src.tolist() == [1, 0, 0, 1] * 5
            for p in range(local):
                for gi, (_, d4, codes) in zip(where[plan.src[p]], groups):
                    blk, g = got[p, gi // 4], gi % 4
                    c = blk[16 + 32 * g: 48 + 32 * g].view(np.int8)
                    assert d4 is None or blk[:16].view(np.uint32)[g] == d4, (p, gi, hex(blk[:16].view(np.uint32)[g]))
                    assert all(c[i] == code for i, code in codes.items()), (p, gi, c.tolist())


def _groups(hq):
    """Hq's bytes [rows, blocks, 144] per 32-value group of a row, four a block: the d4 words, uint32 [rows, groups],
    and the codes, int8 [rows, groups, 32] (block_q8_1_mmq: d4[4], then its 128 codes; group 4 b + g holds values
    128 b + 32 g .. +31)."""
    hq = np.asarray(hq, np.uint8)
    return (np.ascontiguousarray(hq[..., :16]).view(np.uint32).reshape(len(hq), -1),
            np.ascontiguousarray(hq[..., 16:]).view(np.int8).reshape(len(hq), -1, 32))


def test_mmq_swiglu_quantize_fused():
    """M2 (spec-kernels-mmq.md 10): kernel 3, basalt_mmq_swiglu_quantize, alone at bound B = 4 on two tokens' 20 slots.
    The fused kernel (mmq::swiglu, moe_mmq.cu:85-93, with quantize_mmq_q8_1<D4> on its H; H never stored) gives the
    Hq bytes of the separate path, basalt_emu.mmq_swiglu's stored H (``h[i] = g / (1.0f + __expf(-g)) * u;``, :92, in
    fast math, through cuda_nan) then basalt_emu.mmq_quantize, with the GPU's expf_ftz. GU's rows: g in {+-0,
    +-subnormal, +-1e-30, +-20, +-100, +-inf, NaN} against u in {+-0, +-1, +-subnormal, +-inf, NaN}, the 117 pairs
    each as a whole 32-value group (rows 0-5: its own |h| the group's amax) and as one value among plain ones (rows
    6-11), then plain rows (g, u ~ N(0, 3)); groups 16-19 of a row are columns 512-639, the second threadgroup's
    values below K. The uniform groups' bytes are also checked literally: 43 pairs give a NaN h (g NaN or -inf, u NaN,
    0 x inf), so an all-NaN group whose d4 is cuda_nan(1 / NaN) = 0x7fffffff; 14 an infinite one (d4 +inf); 50 a zero
    one (+-0 or flushed: d4 +0); 10 a finite one (codes +-127); all with codes 0 but the finite ones. Padding blocks
    5-7 (pad512(640), quantize.cu:39's zeros) are zero bytes (d4 +0, codes 0); slots at or past bounds[n] keep the
    0xFF fill, with and without experts held on the CPU (bounds[n] 18 < 10 R)."""
    rng = np.random.default_rng(5)
    B, T = 4, 2
    bits = lambda b: np.asarray(b, np.uint32).view(F32)
    g_edge = np.concatenate([bits([0, 0x80000000, 0x007FFFFF, 0x80000001]),
                             np.array([1e-30, -1e-30, 20, -20, 100, -100, np.inf, -np.inf], F32), bits([0x7FC00001])])
    u_edge = np.concatenate([bits([0, 0x80000000]), np.array([1, -1], F32), bits([0x00400000, 0x80000001]),
                             np.array([np.inf, -np.inf], F32), bits([0xFFA00000])])
    gp, up = (a.ravel() for a in np.meshgrid(g_edge, u_edge, indexing="ij"))
    gu = rng.normal(0.0, 3.0, (10 * B, 2 * fx.FF)).astype(F32)
    i = np.arange(len(gp))
    r, j = np.divmod(i, 20)
    cols = 32 * j[:, None] + np.arange(32)
    gu[r[:, None], cols], gu[r[:, None], fx.FF + cols] = gp[:, None], up[:, None]      # rows 0-5: whole groups
    at = 32 * j + 7 * i % 32
    gu[6 + r, at], gu[6 + r, fx.FF + at] = gp, up                                      # rows 6-11: one value each
    want = emu.mmq_quantize(emu.mmq_swiglu(gu, basalt_gpu.expf_ftz))
    ids = np.array([[e for e in range(20) if e % 4 in (1, 2)], [e for e in range(20) if e % 4 in (0, 3)]])
    for cpu, local in (((), 10 * T), ((5, 12), 18)):
        plan = emu.moe_plan(ids, 512, cpu)
        assert plan.bounds[-1] == plan.rows_local == local
        got = gpu.mmq_swiglu_quantize(gu, plan, B=B)
        bad = np.argwhere(got[:local] != want[:local])
        assert not len(bad), (f"cpu {cpu}: {len(bad)} bytes differ from the separate path, first (slot, block, "
                              f"byte) {bad[0]}")
        assert np.all(got[local:] == 0xFF), f"cpu {cpu}: a slot at or past bounds[n] = {local} was written"
        assert not got[:local, 5:].any(), f"cpu {cpu}: a padding block is not zero"
    d4, codes = _groups(got)
    d4, codes = d4[:6, :20].reshape(-1)[:len(gp)], codes[:6, :20].reshape(-1, 32)[:len(gp)]
    nan, inf, zero = d4 == 0x7FFFFFFF, d4 == 0x7F800000, d4 == 0
    fin = ~(nan | inf | zero)
    assert (nan.sum(), inf.sum(), zero.sum(), fin.sum()) == (43, 14, 50, 10)
    assert not codes[~fin].any() and np.all(np.abs(codes[fin].astype(int)) == 127)
    assert np.all(np.isfinite(d4[fin].view(F32)) & (d4[fin].view(F32) > 0))


def test_mmq_swiglu_edges():
    """SG-1 (spec-tests-moe.md 7.2): kernel 3 alone on the GU rows whose exponent edges the chain's GU never reaches,
    4 rows x 1280 (g = GU[r][f], u = GU[r][640 + f]): rows 0-2 N(0, 3) (seed 23), row 3 an edge pair at column 32 j of
    groups j = 0..7, every other g and u 0 (h +0, so each edge group's amax is the edge's own |h|). One token's ten
    slots, six of them experts held on the CPU, so bounds[n] = 4: the kernel forms those four rows, at bound B = 1, and
    the other six slots keep the 0xFF fill. Every Hq byte, padding blocks included, equals the separate path's
    (basalt_emu.mmq_swiglu then mmq_quantize, with the GPU's expf_ftz), whose H holds the table's flushing column: -88
    (1e30) -0, as t = 1 + e > 2^126 makes rcp_ftz +0 (div.approx.ftz's 0 for such a divisor); 2^-126 (2^100) +0, the
    quotient 2^-127 flushed; -87 (1) a negative normal near -1.4e-36; -inf (1) NaN; +inf (2) +inf; NaN (1) NaN; 0
    (+inf) NaN; 40 (1e38) +inf. And the same emulator with rcp_ftz and mul_ftz replaced by IEEE float32 (a port that
    keeps subnormals: -5.3e-7 and 2^-27 there) differs from it in exactly the first two edge groups."""
    rng = np.random.default_rng(23)
    gu = np.zeros((10, 2 * fx.FF), F32)
    gu[:3] = rng.normal(0.0, 3.0, (3, 2 * fx.FF))
    gu[3, 32 * np.arange(8)] = [-88.0, 2.0 ** -126, -87.0, -np.inf, np.inf, np.nan, 0.0, 40.0]
    gu[3, fx.FF + 32 * np.arange(8)] = [1e30, 2.0 ** 100, 1.0, 1.0, 2.0, 1.0, np.inf, 1e38]
    plan = emu.moe_plan(np.arange(10)[None], 512, range(4, 10))
    assert plan.bounds[-1] == plan.rows_local == 4
    got = gpu.mmq_swiglu_quantize(gu, plan, B=1)
    h = emu.mmq_swiglu(gu[:4], basalt_gpu.expf_ftz)
    want = emu.mmq_quantize(h)
    bad = np.argwhere(got[:4] != want)
    assert not len(bad), f"{len(bad)} bytes differ from the emulator, first (row, block, byte) {bad[0]}"
    assert np.all(got[4:] == 0xFF), "a slot at or past bounds[n] = 4 was written"
    edge = h[3, 32 * np.arange(8)]
    assert edge.view(np.uint32)[[0, 1, 3, 4, 5, 6, 7]].tolist() == [0x80000000, 0, 0x7FFFFFFF, 0x7F800000, 0x7FFFFFFF,
                                                                     0x7FFFFFFF, 0x7F800000], edge
    assert -1.5e-36 < edge[2] < -1.3e-36, edge[2]
    g, u = gu[:4, :fx.FF], gu[:4, fx.FF:]
    with np.errstate(all="ignore"):
        keep = (F32(1.0) / emu.add_ftz(basalt_gpu.expf_ftz(-g), F32(1.0))) * g * u
    assert abs(keep[3, 0] / F32(-5.3e-7) - 1) < 0.01 and keep[3, 32] == F32(2.0 ** -27), keep[3, :64:32]
    (da, ca), (db, cb) = _groups(want), _groups(emu.mmq_quantize(emu.cuda_nan(keep)))
    differ = set(zip(*(a.tolist() for a in np.nonzero((da != db) | (ca != cb).any(axis=-1)))))
    assert differ == {(3, 0), (3, 1)}, sorted(differ)


def _plan_ids(rng, R):
    """R tokens of ten distinct routed ids, as route_kernel gives them: slot 0 is expert 7 in three tokens of four (so
    mostly more than 8 of a full range's 16 tokens route to it), the other slots uniform on [0, 448) without 7 or a
    repeat, and token R // 2's slot 9 is expert 500, its only row: experts 448-511 but 500 hold none."""
    ids = np.argsort(rng.random((R, 447)), axis=1)[:, :10]
    ids += ids >= 7
    ids[rng.random(R) < 0.75, 0] = 7
    ids[R // 2, 9] = 500
    return ids


def _check_plan(got, ids, B, cpu=(), n_expert=512, what=""):
    """gpu.mmq_plan's ``got`` against emu.moe_plan(ids, n_expert, cpu), basalt's host walk, word for word: slot's R
    rows, src's 10 R rows, the plan's n, rows_local, order[n], bounds[n + 1] and wpre[n + 1] (gpu.plan_words), every
    other word still the 0xFF fill (-1); StepState's error and done 0. Returns the walk's plan."""
    want = emu.moe_plan(ids, n_expert, cpu)
    R = len(ids)
    slot, src = np.full((B, 10), -1, np.int32), np.full(10 * B, -1, np.int32)
    slot[:R], src[:10 * R] = want.slot, want.src
    assert (got.state["error"], got.state["done"]) == (0, 0), (what, got.state["error"], got.state["done"])
    for name, a, b in (("plan", got.plan, gpu.plan_words(want)), ("slot", got.slot.ravel(), slot.ravel()),
                       ("src", got.src, src)):
        bad = np.flatnonzero(a != b)
        assert not len(bad), f"{what}: {len(bad)} {name} words differ, first {bad[0]}: {a[bad[0]]} for {b[bad[0]]}"
    return want


@pytest.mark.parametrize("R", [1, 17, 256, 4096], ids=lambda r: f"R={r}")
def test_mmq_plan_matches_host_walk(R):
    """No basalt test (basalt's walk is host code; integers, so bitwise): the plan kernels basalt_mmq_plan_count,
    _scan and _fill, in one command buffer at bound B = 4096, equal emu.moe_plan, the transcription of basalt's host
    walk (prefill.cpp:1840-1932, :1993-1996), word for word: slot, src, n, rows_local, order, bounds and wpre, and every
    word past them keeps the 0xFF fill; id rows past R hold 512 and -1 and the scratch starts as 0xFF, so a kernel that
    reads past the chunk's pairs or ranges raises a false error 5. Ids on R tokens: route-like (_plan_ids: an expert
    on more than 8 of the first range's 16 tokens, one with a single row, at least 63 with none) and repeating experts
    inside a token (a token of ten 3s on ten consecutive rows; every src row below 10 R written). At R = 17 also the
    stale scratch: after an R = 256 plan on the same un-cleared mmq_hist, the 17 tokens' plan is still theirs. At R =
    256 also residency (L10): cpu_mask nonzero (1, 2^31, 2^32 - 1, 2, 3) for four experts with rows and one without,
    the walk's plan with those experts' rows the last block in id order (prefill.cpp:1907-1908): the mechanics of the
    input, not the CPU-share rule; and every fixture configuration at its own token count, N_EXPERT 64 (cpu_mask 64
    words)."""
    B = 4096
    rng = np.random.default_rng(9 + R)
    ids = _plan_ids(rng, R)
    cnt = np.bincount(ids.ravel(), minlength=512)
    assert cnt[500] == 1 and np.count_nonzero(cnt == 0) >= 63 and (R < 16 or np.count_nonzero(ids[:16] == 7) > 8)
    want = _check_plan(gpu.mmq_plan(ids, B=B), ids, B, what="route-like")
    assert want.rows_local == 10 * R
    rep = rng.integers(0, 40, (R, 10))
    rep[0] = 3
    got = gpu.mmq_plan(rep, B=B)
    _check_plan(got, rep, B, what="repeats")
    assert np.all(got.src[:10 * R] >= 0) and np.array_equal(np.diff(got.slot[0]), np.ones(9)), got.slot[0]
    if R == 17:
        ids256 = _plan_ids(rng, 256)
        first = gpu.mmq_plan(ids256, B=B)
        _check_plan(first, ids256, B, what="stale scratch's first plan")
        _check_plan(gpu.mmq_plan(ids, B=B, hist=first.hist), ids, B, what="stale scratch")
    if R == 256:
        routed = np.setdiff1d(ids, [7, 500])
        cpu = [7, 500, *rng.choice(routed, 2, replace=False), 470]
        mask = np.zeros(512, np.uint32)
        mask[cpu] = [1, 1 << 31, 2 ** 32 - 1, 2, 3]
        want = _check_plan(gpu.mmq_plan(ids, B=B, cpu_mask=mask), ids, B, cpu, what="cpu_mask")
        assert want.rows_local < 10 * R and 470 not in ids
        for name in fx.CONFIGS:
            cfg = fx.config(name)
            _check_plan(gpu.mmq_plan(cfg.ids, B=B, n_expert=fx.E), cfg.ids, B, n_expert=fx.E, what=name)


@pytest.mark.parametrize("n_expert,bad", [(512, (512, -1)), (512, (512,)), (512, (-1,)), (64, (64,))],
                         ids=["512,-1", "512", "-1", "64-of-64"])
def test_mmq_plan_routed_id_out_of_range(n_expert, bad):
    """R 7.5, eng-review D23: the plan kernels alone at R = 17 (bound B = 256), token 16, alone in the chunk's partial
    last range, holding ``bad`` among valid ids: 512 and -1 together and each alone at basalt's 512 experts, and 64 at
    N_EXPERT 64 (the fixtures' count; cpu_mask 64 words). basalt refuses ``if (e < 0 || e >= m.g->n_expert) { err =
    "prefill: routed id out of range"; return false; }`` (prefill.cpp:1843) inside its counting loop, before its first
    off, slot or src write (:1897-1916): StepState takes error BASALT_ERR_EXPERT_ID and done, its other words unchanged,
    and slot, src and plan keep their 0xFF fill (mmq_hist, which plan_count writes first, is scratch and not compared).
    A step that arrives with done set (any later layer of the refused step) stores nothing at all, mmq_hist included."""
    ids = _plan_ids(np.random.default_rng(8), 17) % n_expert
    ids[16, 3: 3 + len(bad)] = bad
    layout = StepStateLayout()
    for done in (0, 1):
        got = gpu.mmq_plan(ids, B=256, n_expert=n_expert, done=done)
        want = layout.unpack(layout.pack({"t_this_step": 17, "done": done}))
        if not done:
            want.update(error=kernels.BASALT_ERR_EXPERT_ID, done=1)
        assert got.state == want, (done, got.state["error"], got.state["done"])
        assert all(np.all(a == -1) for a in (got.slot, got.src, got.plan)), done
        assert not done or got.hist == b"\xff" * len(got.hist)


@pytest.mark.parametrize("name", list(fx.CONFIGS))
def test_mmq_q2_emulation_bit_for_bit(name):
    """E1-E5 (spec-tests-moe.md 5.4; no basalt test: the tier-1 gates against the host emulator at basalt's 170
    multiprocessors): the chain gpu.moe_mmq, the plan kernels on the configuration's ids and then kernels 1-4 in one
    command buffer, on its x, compiled at the multiple of 256 rows that holds its tokens: share's 12 at 256 (so 2440
    slots must stay unwritten), kq's 256 and wide's 512 at R = B; x's rows past R are NaN. First the plan's words, slot
    and src equal emu.moe_plan's (_check_plan), so a later mismatch is the kernels'. E1: Xq's 32-value groups, d4 and
    codes (so every byte), equal the emulator's slot for slot; E2: GU's bits over all rows_local x 1280; E3: Hq's
    groups, padding blocks 5-7 included; E4: D's bits over rows_local x 2560. Each reports its first mismatch, and
    slots at or past rows_local keep the 0xFF fill in all four. Xq and GU do not depend on expf, so they are the cached
    host-expf emulation's (_emulated); Hq and D are emulated from that GU with the GPU's expf_ftz (basalt_gpu.expf_ftz,
    spec-tests-moe.md 5.4 step 3). E5: the configuration's launches at 170 include an unsplit one in each product, a
    split gate/up launch with a tile of three or more pieces and a split down launch (but in share-outliers, whose last
    group's 8 experts make 160 down tiles, unsplit), and in wide two column tiles with an empty one; the tile widths
    run 8 to 128 (basalt_moe_fixture._coverage)."""
    cfg = fx.config(name)
    p, want = cfg.plan, _emulated(name)
    hq = emu.mmq_quantize(emu.mmq_swiglu(want.gu, basalt_gpu.expf_ftz))
    d = emu.mmq_product(hq, p, cfg.buf, cfg.offsets, "down", NSM)
    B, local = 256 * -(-cfg.T // 256), p.rows_local
    x = np.full((B, fx.N), np.nan, F32)
    x[:cfg.T] = cfg.x
    got = gpu.moe_mmq(x, cfg.ids, cfg.buf, cfg.offsets, B=B, n_expert=fx.E)
    _check_plan(got, cfg.ids, B, n_expert=fx.E, what=name)
    for e, a, w in (("E1 Xq", got.xq, want.xq), ("E3 Hq", got.hq, hq)):
        (d4, codes), (wd4, wcodes) = _groups(a[:local]), _groups(w)
        bad = np.argwhere((d4 != wd4) | (codes != wcodes).any(axis=-1))
        assert not len(bad), (f"{e}: {len(bad)} groups differ from the emulator, first (slot, group) "
                              f"{tuple(bad[0])}: d4 {d4[tuple(bad[0])]:#010x} for {wd4[tuple(bad[0])]:#010x}")
    for e, a, w in (("E2 GU", got.gu, want.gu.view(np.uint32)), ("E4 D", got.d, d.view(np.uint32))):
        bad = np.argwhere(a[:local] != w)
        assert not len(bad), (f"{e}: {len(bad)} elements differ from the emulator, first (slot, column) "
                              f"{tuple(bad[0])}: {a[tuple(bad[0])]:#010x} for {w[tuple(bad[0])]:#010x}")
    for e, a in (("Xq", got.xq), ("GU", got.gu), ("Hq", got.hq), ("D", got.d)):
        assert np.all(a[local:] == np.iinfo(a.dtype).max), f"{e}: a slot at or past rows_local = {local} was written"
    la = {pr: fx.launches(cfg, pr) for pr in fx.PRODUCTS}
    pieces = {pr: [max(len(emu.mmq_pieces(t)) for t in g.tiles.values()) for g in ls] for pr, ls in la.items()}
    assert 1 in pieces["gate_up"] and 1 in pieces["down"] and max(pieces["gate_up"]) >= 3, (name, pieces)
    assert max(pieces["down"]) > 1 or (name == "share-outliers" and la["down"][-1].ngx == 8), (name, pieces)
    cnt = np.diff(p.bounds)
    assert not name.startswith("wide") or any(g.ntx > 1 and any(jt * g.J >= c for c in cnt[g.j0: g.j0 + g.ngx]
                                                                for jt in range(g.ntx)) for g in la["gate_up"]), name
    print(f"\n{name}: B {B}, {local} slots; launches (ngx, J, ntx, grid, a tile's most pieces) at {NSM}: "
          + "; ".join(f"{pr} {[(g.ngx, g.J, g.ntx, g.grid, n) for g, n in zip(la[pr], pieces[pr])]}" for pr in la))


def test_mmq_down_padded_steps_signed_zero():
    """M5 (spec-kernels-mmq.md 10, 9.6): kernel 4 alone (basalt_mmq with the down product's macros) at bound B = 2, on
    17 one-row experts: two tokens route experts 3 j + 1 for j = 0..19 (table ids that are not launch slots), the last
    three held on the CPU, so group 0's 16 experts run unsplit at 170 and group 1's one expert splits (a split tile's
    fixup starts +0 + P, so it ends +0). Hq: groups 0-19 d4 2^-120 and codes -1, blocks 5-7 as the quantizer pads them
    (d4 +0, codes 0). The down weights: codes 2 (w = +1) and d = +-2^-14, so every real step adds +-2^-129, which
    flushes, and block 9 positive leaves every chain -0 before the last iteration's padded sub-blocks 20-23. Those run
    with c = 0 and dB = +0, so fma_ftz(+0, mul_ftz(dA, 0), -0) (mmq_vec_dot.cuh:87) keeps -0 only where dA is negative:
    the chain stays -0 only when both scales mmq_dA finds are negative, blocks 0 and 1 of the next row, of the next
    launch slot's row 0 past row 2559, or the zeroed tail (+0) past group 0's last slot (prefill.cpp:608-611,
    :2249-2251). The blobs sit as the fixture's, 0xFF after each and before odd ones, so a read past a matrix meets a
    NaN scale. Gate: D's bits equal the emulator's (emu.mmq_product) and those signs' expectation: -0 under a next row
    of two negative scales, +0 under two positive ones or mixed ones in either order; rows 2559 of slots 0-14 by the
    next slot's row 0, -0 and +0 both occurring; slot 15's row 2559 +0 (the tail) though order[16]'s row 0 is
    negative; slot 16 +0 throughout; the CPU's slots 17-19 keep the 0xFF fill. So a port that skips the padded steps,
    reads row i's own scales or any past a matrix, reads +0 for every row 2559 or the next group's expert, or adds +0
    onto a one-piece tile fails."""
    rng = np.random.default_rng(12)
    experts = 1 + 3 * np.arange(20)
    plan = emu.moe_plan(experts.reshape(2, 10), fx.E, experts[17:])
    assert (plan.n, plan.rows_local) == (17, 17) and np.array_equal(plan.order, experts[:17])
    g0, g1 = emu.mmq_launches(plan.bounds, fx.PRODUCTS["down"][0], NSM)
    assert (g0.ngx, g0.ntiles, g0.grid, g1.ngx, g1.ntiles, g1.grid) == (16, 320, 320, 1, 20, 170)
    neg = rng.random((17, fx.N, 10)) < 0.5                 # each down block's scale negative, per slot, row and block
    neg[..., 9] = False
    neg[1:17, 0, :2] = (np.arange(1, 17) % 2 == 1)[:, None]   # row 0 of slot j: two negative scales for odd j
    neg[16, 0, :2] = True                                  # order[16]'s row 0, behind group 0's last slot
    blk = np.full((17, fx.N, 10, fx.QB), 0xAA, np.uint8)   # codes 2: w = +1
    blk[..., :2] = np.where(neg, 0x8400, 0x0400).astype("<u2")[..., None].view(np.uint8)   # fp16 -+2^-14
    at = np.arange(17) * (fx.BYTES + fx.SLACK) + 2 * (np.arange(17) % 2)
    buf = np.full(17 * (fx.BYTES + fx.SLACK), 0xFF, np.uint8)
    for j in range(17):
        buf[at[j] + fx.DOWN_OFF: at[j] + fx.BYTES] = blk[j].ravel()
    offsets = np.zeros(fx.E, np.int64)
    offsets[plan.order] = at
    hq = np.zeros((20, 8, kernels.MMQ_BLOCK_BYTES), np.uint8)
    hq[:, :5, :16] = np.full((20, 5, 4), 2.0 ** -120, F32).view(np.uint8)
    hq[:, :5, 16:] = 0xFF                                  # codes -1
    both = neg[..., 0] & neg[..., 1]
    nxt = np.zeros((17, fx.N), bool)                       # the padded steps' scales all negative: -0 kept
    nxt[:16, :-1] = both[:16, 1:]                          # row i + 1 of the slot's own matrix
    nxt[:15, -1] = both[1:16, 0]                           # past row 2559: the next launch slot's row 0
    want = np.where(nxt, np.uint32(0x80000000), np.uint32(0))   # slot 15's row 2559 the tail; slot 16 split
    mixed = neg[:16, 1:, 0] != neg[:16, 1:, 1]
    assert (nxt[:16, :-1].any() and (~neg[:16, 1:, 0] & ~neg[:16, 1:, 1]).any() and (mixed & neg[:16, 1:, 0]).any()
            and (mixed & neg[:16, 1:, 1]).any() and (both[:16, :-1] != both[:16, 1:]).any())
    assert nxt[:15, -1].any() and not nxt[:15, -1].all() and both[16, 0] and not nxt[15, -1]
    emu_d = emu.mmq_product(hq[:17], plan, buf, offsets, "down", NSM).view(np.uint32)
    bad = np.argwhere(emu_d != want)
    assert not len(bad), f"emulator: {len(bad)} elements differ from the signs' expectation, first {tuple(bad[0])}"
    got = gpu.mmq(hq, plan, buf, offsets, B=2, product="down")
    bad = np.argwhere(got[:17] != want)
    assert not len(bad), (f"{len(bad)} D elements differ, first (slot, row) {tuple(bad[0])}: "
                          f"{got[tuple(bad[0])]:#010x} for {want[tuple(bad[0])]:#010x}")
    assert np.all(got[17:] == 0xFFFFFFFF), "a CPU slot (17-19) was written"
    print(f"\n-0 elements: {np.count_nonzero(want)} of {want.size}; rows 2559 of slots 0-14 -0: "
          f"{np.flatnonzero(nxt[:15, -1]).tolist()}")


def test_mmq_chain_routed_id_out_of_range():
    """eng-review D23: the plan test's chunk (R = 17 at bound B = 256, token 16 holding 512 and -1 among valid ids, at
    basalt's 512 experts) through the chain gpu.moe_mmq, the plan kernels then kernels 1-4 in one command buffer. basalt
    refuses inside its counting loop (``if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range";
    return false; }``, prefill.cpp:1843), before any write: StepState takes error BASALT_ERR_EXPERT_ID and done, its
    other words unchanged, and slot, src, plan, Xq, GU, Hq and D keep their 0xFF fill (mmq_hist, which plan_count writes
    first, is scratch and not compared). Kernels 1-4 return at done, which 0xFF plan words cannot show (n = -1 stops
    them at their own bounds), so the step runs again with slot, src and plan holding an earlier chunk's plan (the
    share-plain fixture's 12 tokens, whose blobs the table holds): the refused step leaves those words as they were
    and Xq, GU, Hq and D their fill; and once more arriving with done set (a later layer of the refused step), which
    stores nothing at all, mmq_hist included."""
    cfg = fx.config("share-plain")
    ids = _plan_ids(np.random.default_rng(8), 17)
    ids[16, 3:5] = 512, -1
    x = np.zeros((256, fx.N), F32)
    x[:cfg.T] = cfg.x
    layout = StepStateLayout()
    for prior, done in ((None, 0), (cfg.plan, 0), (cfg.plan, 1)):
        got = gpu.moe_mmq(x, ids, cfg.buf, cfg.offsets, B=256, prior=prior, done=done)
        want = layout.unpack(layout.pack({"t_this_step": 17, "done": done}))
        if not done:
            want.update(error=kernels.BASALT_ERR_EXPERT_ID, done=1)
        what = f"prior {'share-plain' if prior else None}, done {done}"
        assert got.state == want, (what, got.state["error"], got.state["done"])
        slot, src = np.full((256, 10), -1, np.int32), np.full(2560, -1, np.int32)
        plan = np.full(kernels.MMQ_PLAN_WORDS, -1, np.int32)
        if prior is not None:
            slot[:cfg.T], src[:10 * cfg.T], plan = prior.slot, prior.src, gpu.plan_words(prior)
        for e, a, w in (("slot", got.slot, slot), ("src", got.src, src), ("plan", got.plan, plan)):
            assert np.array_equal(a, w), f"{what}: {e} written"
        for e, a in (("Xq", got.xq), ("GU", got.gu), ("Hq", got.hq), ("D", got.d)):
            assert np.all(a == np.iinfo(a.dtype).max), f"{what}: {e} written"
        assert not done or got.hist == b"\xff" * len(got.hist), f"{what}: mmq_hist written"


@pytest.mark.parametrize("name", ["share-plain", "share-outliers", "kq-plain", "kq-outliers"])
def test_mmq_q2_float64_reference(name):
    """nearest bound; no basalt test launches this kernel: G3 is prefill_mmq_kq_test.cpp:375's
    ``const double lim_rms = outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;``, which basalt gates on its
    K-quant MMQ and never on Q2_0 (the contract's MMQ-on-Q2_0 row, docs/design/flash-next.md:451). The Metal chain,
    gpu.moe_mmq at bound B = 256 on the configuration (x's rows past its tokens NaN), meets G1-G3 (_float64_gates)
    against the fixture's float64 reference (spec-tests-moe.md 5.3); G1 and G2 are basalt's own checks
    (prefill_mmq_kq_test.cpp:269, :277-278; prefill_cpu_share_test.cpp:224, :231-232). G1 counts over the chain's
    rows_local x 2560 outputs, whose bytes start as 0xFF (NaN), so an output the chain never writes counts."""
    cfg = fx.config(name)
    x = np.full((256, fx.N), np.nan, F32)
    x[:cfg.T] = cfg.x
    got = gpu.moe_mmq(x, cfg.ids, cfg.buf, cfg.offsets, B=256, n_expert=fx.E)
    _float64_gates(cfg, got.d[:cfg.plan.rows_local].view(F32))


def _chains(plan, row):
    """I1's chains (spec-tests-moe.md 5.6) of sorted row ``row`` of ``plan`` at basalt's 170 multiprocessors: its
    expert's 10 gate/up tiles it = 0..9 and the 20 down tiles of its column tile jt, each emu.mmq_walk's expression (its
    pieces and the fixup's order), and the expert's group, its place zt in the group and the tile width J."""
    z = int(np.searchsorted(plan.bounds, row, side="right")) - 1          # the row's launch slot
    out = []
    for nty, bpne in fx.PRODUCTS.values():
        la = emu.mmq_launches(plan.bounds, nty, NSM)[z // emu.MMQ_GROUP]
        tiles = emu.mmq_walk(la.ntiles, bpne, la.grid)
        zt, jt = z % emu.MMQ_GROUP, (row - plan.bounds[z]) // la.J
        out += [tiles[(it * la.ngx + zt) * la.ntx + jt] for it in range(nty)]
    return out, (z // emu.MMQ_GROUP, z % emu.MMQ_GROUP, la.J)


@pytest.mark.parametrize("kind", ["plain", "outliers"])
def test_mmq_q2_batch_independence(kind):
    """nearest bound; no basalt test launches this kernel: MK-4's ``if (worst_alone > 0.1 * em.worst)``
    (prefill_mmq_kq_test.cpp:433), which basalt runs only on its K-quant pairs (:525-527), measured on Q2_0 with the
    host emulator first (test_mmq_q2_batch_independence_emulator; L13, eng-review D24) and run here on the Metal chain
    through the same fx.mk4: the kq configuration's 256 tokens, then tokens 0..7 each alone and last of three
    (:383-411), worst_alone as :416-423. Every run goes through one program, gpu.moe_mmq_program at bound B = 256: the
    same buffers and dispatches, the row count and the ids written into their buffers (x's rows past R NaN), so the
    plan, the groups, J and the stream-k pieces come from the ids on the GPU, never from the encode (spec-tests-moe.md
    5.6's addition 3). Its additions 1 and 2: I1, every row whose 30 tile chains (_chains) are its 256-token run's has
    GU, Hq and D rows bitwise equal to that run's, and such rows exist, among them rows whose expert changed its group,
    its place zt in the group and its tile width J (128 to 8); the small runs' non-finite values, which MK-4 lets
    through, are 0. The factor 0.1 is basalt's and is never loosened here; a failure goes to the owner with the
    measured ratio."""
    cfg = fx.config(f"kq-{kind}")
    B, runs = 256, []
    chain = gpu.moe_mmq_program(cfg.buf, cfg.offsets, B=B, n_expert=fx.E)

    def run(x, ids):
        xb = np.full((B, fx.N), np.nan, F32)
        xb[:len(x)] = x
        got, rows = chain(xb, ids), fx.K * len(ids)
        runs.append((np.asarray(ids), {e: getattr(got, e)[:rows].copy() for e in ("gu", "hq", "d")}))
        return runs[-1][1]["d"].view(F32)

    m = fx.mk4(cfg, run)
    print(f"\n{cfg.name}: worst_alone {m.worst_alone:.3e}  em.worst {m.em.worst:.3e}  worst_alone / em.worst "
          f"{m.worst_alone / m.em.worst:.3e} (fails above 0.1); {m.differ} of 160 rows differ; {m.nonfinite} non-finite")
    assert m.nonfinite == 0, (cfg.name, m.nonfinite)
    assert len(runs) == 17, len(runs)
    big, matched, changed = runs[0][1], 0, set()
    for i, (ids1, got) in enumerate(runs[1:]):
        t = i // 2                                         # mk4's order: t = 0..7, alone, then last of three
        assert len(ids1) == (1, 3)[i % 2] and np.array_equal(ids1[-1], cfg.ids[t]), i
        p1 = emu.moe_plan(ids1, fx.E)
        for k in range(fx.K):
            r1, r0 = p1.slot[-1, k], cfg.plan.slot[t, k]
            (c1, w1), (c0, w0) = _chains(p1, r1), _chains(cfg.plan, r0)
            if c1 != c0:                                   # MK-4's bound only
                continue
            matched += 1
            changed |= {what for what, on in (("group", w0[0] != w1[0]), ("zt", w0[1] != w1[1]),
                                              ("J 128 to 8", (w0[2], w1[2]) == (128, 8))) if on}
            for e in ("gu", "hq", "d"):
                assert np.array_equal(got[e][r1], big[e][r0]), f"I1: token {t} of {len(ids1)}, slot {k}: {e} differs"
    print(f"I1: {matched} of 160 rows have their 256-token run's 30 chains and equal GU, Hq and D rows; among them "
          f"rows whose expert changed: {sorted(changed)}")
    assert matched and changed == {"group", "zt", "J 128 to 8"}, (matched, changed)
    assert not m.worst_alone > 0.1 * m.em.worst, (cfg.name, m.worst_alone, m.em.worst)


def test_mmq_order_is_basalts_not_this_gpus():
    """M4 (spec-kernels-mmq.md 10): MMQ_SM_COUNT, the multiprocessors whose stream-k pieces the kernels reproduce
    (mmq.cuh:830-833), is basalt's 170 (kernels.DENSE_TC_SM_COUNT) in every product's macros unless a caller asks
    otherwise, never this Mac's 40 (tests/contract/test_basalt_ops.py's test_dense_tc_order_is_basalts_not_this_gpus),
    and it reaches the arithmetic: on share-plain, whose last gate/up launch (4 experts, 40 tiles) splits at 170 and
    runs unsplit at 40 (test_mmq_q2_streamk_pin), kernel 2 built at 170 equals the emulator at 170 bit for bit and built
    at 40 differs from it in some element, where it equals the emulator at 40 (so the difference is basalt's rule at 40
    cores, not a broken build)."""
    for product, kw in (("gate_up", {}), ("down", {}), ("q8_0", {"n": 2560, "k": 2560})):
        assert kernels.basalt_mmq_macros(product, **kw)["MMQ_SM_COUNT"] == str(NSM) == "170", product
        assert kernels.basalt_mmq_macros(product, sm_count=THIS_MAC, **kw)["MMQ_SM_COUNT"] == "40", product
    cfg = fx.config("share-plain")
    p, want = cfg.plan, _emulated("share-plain")
    assert fx.launches(cfg, "gate_up")[-1].tiles != fx.launches(cfg, "gate_up", THIS_MAC)[-1].tiles
    B = 256
    xq = np.full((10 * B, fx.N // 128, kernels.MMQ_BLOCK_BYTES), 0xFF, np.uint8)
    xq[:p.rows_local] = want.xq
    got = {n: gpu.mmq(xq, p, cfg.buf, cfg.offsets, B=B, sm_count=n)[:p.rows_local] for n in (NSM, THIS_MAC)}
    bad = np.argwhere(got[NSM] != want.gu.view(np.uint32))
    assert not len(bad), f"the 170 build: {len(bad)} GU elements differ from the emulator at 170, first {tuple(bad[0])}"
    differ = np.count_nonzero(got[THIS_MAC] != want.gu.view(np.uint32))
    print(f"\nthe 40 build: {differ} of {got[THIS_MAC].size} GU elements differ from the emulator at 170")
    assert differ, "MMQ_SM_COUNT 40 gives the 170 build's bits: the macro does not reach the stream-k pieces"
    here = emu.mmq_product(want.xq, p, cfg.buf, cfg.offsets, "gate_up", THIS_MAC).view(np.uint32)
    assert np.array_equal(got[THIS_MAC], here), "the 40 build differs from the emulator at 40"


@pytest.mark.parametrize("path", ["moe", "q8_0"])
def test_mmq_nan_reaches_d(path):
    """eng-review D25's stored-NaN row (no basalt test: basalt's tests feed no non-finite activation): every NaN the
    chain stores in GU and D, and the Q8_0 instance in y, goes through cuda_nan (mmq.cuh:216's write-back; sm_120
    stores 0x7fffffff, this GPU's arithmetic NaN is 0x7fc00000). The share-plain configuration's x with three tokens
    poisoned, each alone: token 0 an all-NaN 32-value group, whose d4 is cuda_nan(1 / NaN), so every step fma_ftz(NaN,
    .., sum) of each output; token 1 a +inf and token 2 a -inf element, whose group's amax +inf gives d_inv = 0, codes 0
    and d4 = rcp_ftz(0) = +inf, so fma_ftz(+inf, +-0, sum) (mmq_vec_dot.cuh:87) makes each output NaN. ``moe``: through
    gpu.moe_mmq (the plan kernels and kernels 1-4 at bound B = 256), Xq, GU, Hq and D equal emu.mmq_moe's bytes (with
    the GPU's expf_ftz), the poisoned tokens' 30 slots hold 0x7fffffff in every GU and D element, and every other slot's
    elements are finite. ``q8_0``: the same three rows poisoned among 192 (tile width 96, 8 or 9 pieces a tile) through
    _q8_0_and_q2_0 (Xq equal to emu.mmq_quantize's), the poisoned rows hold 0x7fffffff in every y element, the other
    rows are finite, and y equals the Q2_0 instance's bit for bit (test_mmq_q8_0_equals_q2_0_tiles's weights)."""
    cfg = fx.config("share-plain")
    if path == "q8_0":
        x = np.full((256, fx.N), np.nan, F32)
        x[:192] = np.random.default_rng(9).standard_normal((192, fx.N), dtype=F32)
    else:
        x = cfg.x.copy()
    x[0, 224:256] = np.nan                                 # group 7, block 1
    x[1, 1000], x[2, 2047] = np.inf, -np.inf               # groups 31 and 63, the second and fourth 512-value blocks
    if path == "q8_0":
        _, y8, y2 = _q8_0_and_q2_0(x, 192, 1)
        bad = np.argwhere(y8 != y2)
        assert not len(bad), f"y: {len(bad)} words differ from the Q2_0 instance's, first {tuple(bad[0])}"
        assert np.all(y8[:3] == 0x7FFFFFFF), "y: a poisoned row's element is not 0x7fffffff"
        assert np.all(np.isfinite(y8[3:].view(F32))), "y: a non-finite element outside the poisoned rows"
        return
    with np.errstate(invalid="ignore"):
        want = emu.mmq_moe(x, cfg.plan, cfg.buf, cfg.offsets, expf=basalt_gpu.expf_ftz, nsm=NSM)
    xb = np.full((256, fx.N), np.nan, F32)
    xb[:cfg.T] = x
    got = gpu.moe_mmq(xb, cfg.ids, cfg.buf, cfg.offsets, B=256, n_expert=fx.E)
    local = cfg.plan.rows_local
    for e in ("xq", "gu", "hq", "d"):
        a = getattr(got, e)[:local]
        bad = np.argwhere(a != getattr(want, e).view(a.dtype))
        assert not len(bad), f"{e}: {len(bad)} words differ from the emulator, first {tuple(bad[0])}"
    poisoned = np.isin(cfg.plan.src[:local], [0, 1, 2])
    assert np.count_nonzero(poisoned) == 3 * fx.K
    for e in ("gu", "d"):
        a = getattr(got, e)[:local]
        assert np.all(a[poisoned] == 0x7FFFFFFF), f"{e}: a poisoned slot's element is not 0x7fffffff"
        assert np.all(np.isfinite(a[~poisoned].view(F32))), f"{e}: a non-finite element outside the poisoned slots"


def _dense(x, w, R, m, B=256):
    """gpu.mmq_dense, basalt_prompt_mmq's pair (draft_kv's proj, prefill.cpp:971-980: mmq::quantize without ids, then
    mul_mat_q<Q8_0> with p.n = 1 and p.bounds = {0, rows}), of the float32 rows ``x`` [B m, K] with the Q8_0 matrix
    ``w``, compiled at bound ``B`` and run with ``R`` rows of ``m``, so R m dense rows. Asserts that Xq's bytes equal
    emu.mmq_quantize's (kernel 1's rows by position, quantize.cu:29 without ids) and that Xq and y keep their 0xFF fill
    past R m; returns Xq's bytes [R m, pad512(K) / 128, 144] and y's bits [R m, N]."""
    rows = R * m
    xq, y = gpu.mmq_dense(x, w, B=B, rows=R, rows_per_t=m)
    with np.errstate(invalid="ignore"):
        bad = np.argwhere(xq[:rows] != emu.mmq_quantize(x[:rows]))
    assert not len(bad), f"Xq: {len(bad)} bytes differ from emu.mmq_quantize, first {tuple(bad[0])}"
    assert np.all(xq[rows:] == 0xFF) and np.all(y[rows:] == 0xFFFFFFFF), "a row past R m written"
    return xq[:rows], y[:rows]


@functools.lru_cache(maxsize=None)
def _q2_0_as_q8_0():
    """spec-engine.md 7.2's weights: a Q2_0 matrix at the one shape a Q2_0 instance shares with a dense matrix, the
    gate/up product's [1280, 2560] (fx.fill_q2_0), followed by 64 bytes of 0xFF (a read past it is a validation report,
    else an fp16 NaN scale); and the Q8_0 matrix [1280, 80 * 34] of the same values, each 64-value block as two Q8_0
    blocks (ggml_common.h:163-167: fp16 d, then qs[32]) with its d's bits and the int8 codes q - 1 in {-1, 0, 1, 2}
    (unpack64_q2_0's code q), so equal d on each pair of adjacent blocks. Cached and read-only."""
    w2 = fx.fill_q2_0(np.random.default_rng(80), 2 * fx.FF, fx.N).reshape(-1, fx.QB)
    w8 = np.empty((len(w2), 2, kernels.Q8_0_BLOCK_BYTES), np.uint8)
    w8[..., :2] = w2[:, None, :2]
    w8[..., 2:] = (unpack64_q2_0(w2[:, 2:]).astype(np.int8) - 1).reshape(-1, 2, 32).view(np.uint8)
    out = np.append(w2, np.full(64, 0xFF, np.uint8)), w8.reshape(2 * fx.FF, -1)
    for a in out:
        a.flags.writeable = False
    return out


def _q8_0_and_q2_0(x, R, m, B=256):
    """The Q8_0 instance and the Q2_0 one on the same rows (spec-engine.md 7.2), with _q2_0_as_q8_0's weights: _dense
    of ``x`` [B m, 2560] with the Q8_0 matrix (R rows of m), then kernel 2 with the gate/up product's macros (MMQ_TYPE
    42; gpu.mmq) on the same Xq rows with the Q2_0 matrix as expert 0's blob, through a plan of one launch slot holding
    the R m rows, n 1, bounds {0, R m}, wpre {0, ceil(R m / 8)} (moe_plan's form of p.n = 1 and p.bounds = {0, rows};
    kernel 2 reads no slot or src). Returns Xq's bytes and both y's bits [R m, 1280], the Q8_0 instance's first."""
    w2, w8 = _q2_0_as_q8_0()
    rows = R * m
    xq, y8 = _dense(x, w8, R, m, B)
    xq2 = np.full((10 * B,) + xq.shape[1:], 0xFF, np.uint8)
    xq2[:rows] = xq
    plan = SimpleNamespace(slot=np.zeros((R, 10), np.int32), n=1, rows_local=rows, order=[0], bounds=[0, rows],
                           wpre=[0, -(-rows // 8)])
    return xq, y8, gpu.mmq(xq2, plan, w2, [0], B=B)[:rows]


def test_mmq_q8_0_equals_q2_0_tiles():
    """spec-engine.md 7.2's bitwise check (no basalt test: basalt never sets its MMQ instances side by side): the Q8_0
    tile load (mmq_load_tiles.cuh:77-124) leaves the x tile that load_tiles_q2_0 (:11-75) leaves for the same values,
    and everything after it is one code, the integer dot and the fp32 epilogue (mmq_vec_dot.cuh:14-92), the stream-k
    pieces and the fixup (mmq.cuh:390-864), whose piece starts are the same function of (tiles, grid) for Q8_0's 80
    blocks of 8 an iteration as for Q2_0's 40 of 4 (scripts/draft_mmq_tiles.py). So _q2_0_as_q8_0's Q8_0 matrix, which
    holds its Q2_0 matrix's values, gives y bit-equal to the Q2_0 instance's on the same Xq rows (_q8_0_and_q2_0): 768
    rows (R 192 of 4, fc_hidden's ROWS_PER_T: tile width 128, six column tiles, split at 170 multiprocessors) and one
    row (width 8, a one-column window, every k iteration its own piece); every y is finite."""
    rng = np.random.default_rng(8)
    for R, m in ((192, 4), (1, 1)):
        x = np.full((256 * m, fx.N), np.nan, F32)
        x[:R * m] = rng.standard_normal((R * m, fx.N), dtype=F32)
        _, y8, y2 = _q8_0_and_q2_0(x, R, m)
        bad = np.argwhere(y8 != y2)
        assert not len(bad), f"R {R} m {m}: {len(bad)} of {y8.size} y elements differ from the Q2_0 instance's, " \
                             f"first {tuple(bad[0])}"
        assert np.all(np.isfinite(y8.view(F32))), f"R {R} m {m}: a non-finite y"


# draft_kv's dense Q8_0 products (spec-engine.md 7.2; prefill.cpp:1008, :1012, :1021-1022), "N x K @ rows": (N, R, m,
# B), R rows of m activation rows each at bound B: fc_embedding on nb rows, fc_hidden on nb rows of hc = 4 streams,
# attn_k and attn_v on nb rows, at nb = 192 and the last batch's 64 of a 256-row chunk, and one row. fc_hidden runs at
# draft_kv's own bound, B = 192 at m.T = 256 (spec-engine.md 4), so R = B and every row of its buffers is live; the
# others at B = 256, so rows past R m stay to be checked.
Q8_0_SHAPES = {"2560x2560@192": (2560, 192, 1, 256), "2560x2560@768": (2560, 192, 4, 192),
               "512x2560@192": (512, 192, 1, 256), "512x2560@64": (512, 64, 1, 256), "512x2560@1": (512, 1, 1, 256)}


@pytest.mark.parametrize("name", list(Q8_0_SHAPES))
def test_mmq_q8_0_float64_reference(name):
    """nearest bound; no basalt test launches this kernel: G3 is prefill_mmq_kq_test.cpp:375's ``const double lim_rms =
    outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;`` at its plain values, failing if ``em.rms > lim_rms ||
    em.worst > lim_worst`` (:376-377; eng-review D20). basalt runs mul_mat_q<Q8_0> only as the grouped down product of
    its K-quant pairs Q4_K / Q8_0 and Q5_K / Q8_0 at K = 640 (:525-527, :213-214), gated through the whole chain's D,
    never this dense launch, draft_kv's proj (prefill.cpp:971-980). Here at draft_kv's shapes (Q8_0_SHAPES), each at
    its bound B (@768 at R = B = 192, the others at B = 256 with fewer rows): the matrix [N, 2560] by fill_matrix at
    Q8_0 (:74-95: every byte uniform, then each block's own d from U(0.0008, 0.0032), float32 rounded to fp16), N(0, 1)
    activations with row 5 all zero where the rows reach it (make_x, :112-119) and the rows past R m NaN. Through
    _dense (Xq's bytes equal emu.mmq_quantize's, Xq and y keep their 0xFF fill past R m), y meets G1 (no non-finite
    output, counted; :269), G2 (the zero row exactly zero; :277-278) and G3 against the float64 product of the
    activations with the matrix as ggml dequantizes it (gguf_reference), by fx.compare (basalt's compare, :259-282)."""
    n, R, m, B = Q8_0_SHAPES[name]
    rows, nb = R * m, n * fx.N // 32
    rng = np.random.default_rng(n + rows)
    w = rng.integers(0, 256, (nb, kernels.Q8_0_BLOCK_BYTES), dtype=np.uint8)
    lo = F32(0.0008)
    d = lo + (F32(4.0) * lo - lo) * rng.random(nb, dtype=np.float32)
    w[:, :2] = d.astype("<f2").view(np.uint8).reshape(nb, 2)
    w = w.reshape(n, -1)
    x = np.full((B * m, fx.N), np.nan, F32)
    x[:rows] = rng.standard_normal((rows, fx.N), dtype=F32)
    zero = rows > 5
    if zero:
        x[5] = 0.0
    y = _dense(x, w, R, m, B)[1].view(F32)
    ref = (x[:rows].astype(np.float64) @ gguf_reference.dequantize(8, w).astype(np.float64).T).astype(F32)
    bad = np.count_nonzero(~np.isfinite(y))
    em = fx.compare(y, ref, np.arange(rows))
    print(f"\n{name}: rel RMS {em.rms:.3e} (bound 2e-2)  worst row {em.worst:.3e} (bound 0.1)  max rel {em.rmax:.3e}  "
          f"non-finite {bad}")
    assert bad == 0, f"{name}: {bad} non-finite outputs"
    assert not zero or not np.any(y[5] != 0), f"{name}: the zero row is not zero (worst = 1e30)"
    assert not (em.rms > 2e-2 or em.worst > 0.1), (name, em.rms, em.worst)
