# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) tests/cuda/prefill_mmq_kq_test.cpp and tests/cuda/prefill_cpu_share_test.cpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures (tests/kernels/
# basalt_moe_fixture.py) and their checks on the routed experts' MMQ.
"""basalt's MoE MMQ for the routed experts on Q2_0 (src/prefill/moe_mmq.cu, src/prefill/mmq/), the tests that run on
the host: the emulator's launch walk and the fixture.

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
  walk prefill.cpp:1840-1996) equal an independent stable sort, with and without CPU-held experts."""

import numpy as np
import pytest

from monolith import kernels
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_moe_fixture as fx

NSM = kernels.DENSE_TC_SM_COUNT       # basalt's reference GPU, which the port pins (never this Mac's)
THIS_MAC = 40                         # the GPU cores of the M3 Max these tests run on


def _out(*pieces):
    """A tile's output when its k range splits into ``pieces`` (k order), in the fixup's order (mmq.cuh:654-698, :765):
    P_0 alone, else ((+0 + P_{m-1}) + ... + P_0) + P_m."""
    *rest, last = [("P", lo, hi) for lo, hi in pieces]
    s = ("zero",)
    for p in reversed(rest):
        s = ("add", s, p)
    return ("add", s, last) if rest else last


def test_mmq_walk_facts():
    """pick_j (mul_mat_q_switch_J with ncols_opt = maxr) where the first width reaching the fewest column tiles is not
    128 (200 rows: 112, 129: 80, 257: 96), and mmq_walk's tile 0 at basalt's 170 multiprocessors in six launches of
    one-row experts (J = 8, ntx = 1): the full groups unsplit (gate/up 160 tiles, down 320 and 160), 150 gate/up tiles
    cut at block 32, one expert's 10 gate/up tiles cut at every iteration and its 20 down tiles at blocks 4 and 8."""
    assert (emu.pick_j(200), emu.pick_j(129), emu.pick_j(257)) == (112, 80, 96)
    table = [("gate_up", 16, 160, _out((0, 40))), ("down", 16, 320, _out((0, 10))), ("down", 8, 160, _out((0, 10))),
             ("gate_up", 15, 170, _out((0, 32), (32, 40))),
             ("gate_up", 1, 170, _out(*[(4 * q, 4 * q + 4) for q in range(10)])),
             ("down", 1, 170, _out((0, 4), (4, 8), (8, 10)))]
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
