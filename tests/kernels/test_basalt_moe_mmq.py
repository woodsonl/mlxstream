# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) tests/cuda/prefill_mmq_kq_test.cpp and tests/cuda/prefill_cpu_share_test.cpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures (tests/kernels/
# basalt_moe_fixture.py) and their checks on the routed experts' MMQ.
"""basalt's MoE MMQ for the routed experts on Q2_0 (src/prefill/moe_mmq.cu, src/prefill/mmq/), the tests that run on
the host: the emulator's launch walk, the fixture, and the emulator's own gates before any Metal exists.

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
  SwiGLU's divisions differs from the correctly rounded quotient (contract invariant 5's count)."""

import functools

import numpy as np
import pytest

from monolith import kernels
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_moe_fixture as fx

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
    """Nearest bound; no basalt test launches this kernel: G3 is prefill_mmq_kq_test.cpp:375's
    ``const double lim_rms = outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;``, which basalt gates on its
    K-quant MMQ and never on Q2_0 (the contract's MMQ-on-Q2_0 row, docs/design/flash-next.md:451). The host emulator
    basalt_emu.mmq_moe, which the Metal chain must equal bit for bit (E1-E4), meets G1-G3 as
    test_mmq_q2_float64_reference states them (_float64_gates), before any Metal exists."""
    _float64_gates(fx.config(name), _emulated(name).d)


@pytest.mark.parametrize("kind", ["plain", "outliers"])
def test_mmq_q2_batch_independence_emulator(kind):
    """Nearest bound; no basalt test launches this kernel: MK-4's ``if (worst_alone > 0.1 * em.worst)``
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
