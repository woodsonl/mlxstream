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
- test_mmq_q2_emulation_bit_for_bit (E1-E2, spec-tests-moe.md 5.4): kernel 1, then kernel 2 (basalt_mmq: every
  launch's mul_mat_q<Q2_0> with its stream-k fixup folded per output element, in one dispatch) on each fixture
  configuration: Xq's bytes and GU's bits equal the host emulator's (emu.mmq_moe at basalt's 170 multiprocessors), and
  slots past rows_local keep the 0xFF fill."""

import functools

import numpy as np
import pytest

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu
from tests.kernels import basalt_moe_fixture as fx
from tests.kernels import basalt_prompt_gpu as gpu

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
    """E1-E2 (spec-tests-moe.md 5.4; no basalt test: the tier-1 gates against the host emulator at basalt's 170
    multiprocessors): kernel 1, then kernel 2 (basalt_mmq: mul_mat_q<Q2_0> and its stream-k fixup per output element,
    every launch of the plan in one dispatch), on the configuration's x through its plan (emu.moe_plan's), compiled at
    the multiple of 256 rows that holds its tokens: share's 12 at 256 (so 2440 slots must stay unwritten), kq's 256 and
    wide's 512 at R = B; x's rows past R are NaN. E1: Xq's 32-value groups, d4 and codes (so every byte), equal the
    emulator's slot for slot; E2: GU's bits equal the emulator's over all rows_local x 1280, kernel 2 reading kernel 1's
    Xq. Each reports its first mismatch, and slots at or past rows_local keep the 0xFF fill in both. Xq and GU do not
    depend on expf, so the cached host-expf emulation (_emulated) serves here; Hq and Dm do, and need the GPU's
    expf_ftz. The configurations hold tile widths 8 to 128, split and unsplit gate/up launches and wide's launch of two
    column tiles with an empty one (basalt_moe_fixture._coverage)."""
    cfg = fx.config(name)
    p, want = cfg.plan, _emulated(name)
    B, local = 256 * -(-cfg.T // 256), p.rows_local
    x = np.full((B, fx.N), np.nan, F32)
    x[:cfg.T] = cfg.x
    xq = gpu.mmq_quantize(x, p, B=B)
    (d4, codes), (wd4, wcodes) = _groups(xq[:local]), _groups(want.xq)
    bad = np.argwhere((d4 != wd4) | (codes != wcodes).any(axis=-1))
    assert not len(bad), (f"E1: {len(bad)} Xq groups differ from the emulator, first (slot, group) {tuple(bad[0])}: "
                          f"d4 {d4[tuple(bad[0])]:#010x} for {wd4[tuple(bad[0])]:#010x}")
    assert np.all(xq[local:] == 0xFF), f"E1: a slot at or past rows_local = {local} was written"
    gu, want_gu = gpu.mmq(xq, p, cfg.buf, cfg.offsets, B=B), want.gu.view(np.uint32)
    bad = np.argwhere(gu[:local] != want_gu)
    assert not len(bad), (f"E2: {len(bad)} GU elements differ from the emulator, first (slot, column) "
                          f"{tuple(bad[0])}: {gu[tuple(bad[0])]:#010x} for {want_gu[tuple(bad[0])]:#010x}")
    assert np.all(gu[local:] == 0xFFFFFFFF), f"E2: a slot at or past rows_local = {local} was written"
    print(f"\n{name}: B {B}, {local} slots, gate/up launches (ngx, J, ntx, grid) "
          f"{[(la.ngx, la.J, la.ntx, la.grid) for la in emu.mmq_launches(p.bounds, 10, NSM)]}")
