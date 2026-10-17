# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/q5k_head_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its weights and activations, row counts, maps, planted rows and pass criteria.
"""basalt's Q5_K head on Metal (kernels/common/basalt_q5k_head.metal).

q5k_head_parity, ported (basalt src/kernels/q5k_head_parity.cpp, ctest ``q5k_head_parity 20000 5``: 20,000 rows):
random Q5_K rows (random bytes, then d and dmin = fp16(|N(0, 1)| * 0.01), :33-43) and N(0, 1) activations quantized
by native_quantize_q8_1, n_in 2560. numpy's generator stands for basalt's mt19937 + normal_distribution: the same
distributions, other draws. The timing loops are not ported.
- The head against native_mmvq (basalt_mmvq.metal, Q5_K) bit for bit, 1 to 8 columns (:64-92).
- The contract's check at the head's full shape (docs/design/flash-next.md, kernels without a reference test):
  2560 -> 248,320 rows, bit for bit against native_mmvq.
- A float64 reference for the Q5_K dot (pr-b-scope-basalt.md section 11's new test) under the nearest basalt bound,
  mmvq_multi_parity.cpp:461's rule for native_mmvq's types."""

import numpy as np
import pytest

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels.test_basalt_mmvq import reference

F32 = np.float32
N_IN, ROWS, MT = 2560, 20000, 8      # basalt's n_in, ctest's row count, its 8 columns
QCOL = N_IN // 32 * 36               # q8_1 bytes a column


def weights(rows, rng, n_in=N_IN):
    """q5k_head_parity.cpp:33-43: random bytes, then each block's d and dmin fp16(|N(0, 1)| * 0.01)."""
    w = rng.integers(0, 256, (rows * n_in // 256, 176), dtype=np.uint8)
    dm = (np.abs(rng.standard_normal((w.shape[0], 2))).astype(F32) * F32(0.01)).astype(np.float16)
    w[:, :4] = dm.view(np.uint8).reshape(-1, 4)
    return w.reshape(rows, -1)


def activations(ncols, rng):
    """N(0, 1) columns quantized by native_quantize_q8_1 (:47-59)."""
    return gpu.q8_1(rng.standard_normal(ncols * N_IN).astype(F32), "native")


def column(xq, j):
    return xq[j * QCOL: (j + 1) * QCOL]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)                 # basalt's seed, std::mt19937 rng(5) (:32)
    w = weights(ROWS, rng)
    return w, activations(MT, rng), rng


def bad_bits(a, b):
    return int(np.count_nonzero(np.asarray(a, F32).view(np.uint32) != np.asarray(b, F32).view(np.uint32)))


def test_head_against_native_mmvq(data):
    w, xq, _ = data
    fail = 0
    for nc in range(1, MT + 1):
        a = gpu.mmvq(13, w, xq[: nc * QCOL], N_IN, ROWS, nc, fill=0xFF)        # basalt's pre-fills (:65-66)
        b = gpu.q5k_head(w, xq[: nc * QCOL], N_IN, ROWS, nc, fill=0xEE)
        bad = bad_bits(a, b)
        # basalt: ``for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;`` (:74) and
        # ``if (bad) fail = 1;`` (:91)
        fail |= bad != 0
        assert np.all(np.isfinite(b)), nc
    assert not fail


def test_head_full_shape():
    """flash-next.md: q5k_head_mmvq at the head's full shape, 2560 -> 248,320, bit for bit against native_mmvq (basalt's
    test default, q5k_head_parity.cpp:28), one column and eight."""
    rows = 248320
    rng = np.random.default_rng(28)
    w = weights(rows, rng)
    xq = activations(MT, rng)
    for nc in (1, MT):
        a = gpu.mmvq(13, w, xq[: nc * QCOL], N_IN, rows, nc, fill=0xFF)
        b = gpu.q5k_head(w, xq[: nc * QCOL], N_IN, rows, nc, fill=0xEE)
        assert bad_bits(a, b) == 0, nc


@pytest.mark.parametrize("n_in", [256, 2560, 16384])
def test_head_shapes_and_grids(n_in):
    """Other row lengths (q5k_head_supported's range: 1 to 64 blocks) and grids, each column against native_mmvq."""
    rng = np.random.default_rng(n_in)
    rows = 333
    w = weights(rows, rng, n_in)
    ncols = 1 if n_in == 16384 else 4                 # basalt's 96 KiB staging rule allows 2 columns at 16384
    xq = gpu.q8_1(rng.standard_normal(ncols * n_in).astype(F32), "native")
    a = gpu.mmvq(13, w, xq, n_in, rows, ncols)
    for sm_count, per_sm in ((170, 2), (40, 1), (1, 1)):
        b = gpu.q5k_head(w, xq, n_in, rows, ncols, sm_count=sm_count, per_sm=per_sm)
        assert bad_bits(a, b) == 0, (sm_count, per_sm)


def test_float64_reference(data):
    """The Q5_K dot against the float64 product on the same bytes, under mmvq_multi_parity.cpp:461's rule: worst
    |y - ref| <= 1e-4 * max |ref|, every column."""
    w, xq, _ = data
    rows = 2000
    y = gpu.q5k_head(w[:rows], xq, N_IN, rows, MT)
    for j in range(MT):
        ref = reference(13, w[:rows], column(xq, j), N_IN, rows)
        worst, scale = np.max(np.abs(y[j].astype(np.float64) - ref)), np.max(np.abs(ref))
        # basalt: ``const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);`` (mmvq_multi_parity.cpp:461)
        assert np.all(np.isfinite(y[j])) and worst <= 1e-4 * (scale + 1e-30), (j, worst, scale)


def test_head_emulation_bit_for_bit(data):
    """The head against the numpy emulation of nvcc's sequence (basalt_emu.mmvq, Q5_K: the head's per-value
    sequence is native_mmvq's, q5k_head.md section 1)."""
    w, xq, _ = data
    rows = 96
    got = gpu.q5k_head(w[:rows], xq[: 3 * QCOL], N_IN, rows, 3)
    assert bad_bits(got, emu.mmvq(13, w[:rows], xq[: 3 * QCOL], N_IN, rows, 3)) == 0


# ---- the fused top-1 (q5k_head_parity.cpp:93-211) ------------------------------------------------------------------

def host_top(a, mapped):
    """The host argmax over the GPU logits (the lowest row on ties: basalt's ``a[i] > a[best]`` scan and the sampler's
    greedy rule, flash-next.md "Greedy pick") and the double softmax sum relative to it (:171-179)."""
    rows = np.flatnonzero(mapped)
    best = int(rows[np.argmax(a[rows])])                # np.argmax: the first maximum
    s = float(np.sum(np.exp(a[rows].astype(np.float64) - np.float64(a[best]))))
    return best, s


def test_top1(data):
    """:93-146 at ctest's row counts ({20000, 106299, 40525, 1000}, the larger skipped above 20,000, :105-107), each of
    8 columns: the logits bit for bit against native_mmvq, the id against the host argmax over the GPU logits (basalt:
    sample_greedy_top_prob on the GPU, sampler.cu:930, not on main), the probability against a host double softmax
    over them (the comparison basalt's holes section makes, :171-185)."""
    w, xq, _ = data
    logit_bad = id_bad = 0
    max_rel = 0.0
    for rr in (ROWS, 106299, 40525, 1000):
        if rr > ROWS:
            continue
        for col in range(MT):
            xc = column(xq, col)
            a = gpu.mmvq(13, w[:rr], xc, N_IN, rr, 1)[0]
            b, b_id, b_p, _ = gpu.q5k_top1(w[:rr], xc, rr)
            best, s = host_top(a, np.ones(rr, bool))
            logit_bad += bad_bits(a, b)
            id_bad += b_id != best
            max_rel = max(max_rel, abs(float(b_p) - 1.0 / s) * s)
    # basalt: ``if (logit_bad || id_bad || max_rel > 1e-5) fail = 1;`` (:145)
    assert logit_bad == 0 and id_bad == 0 and max_rel <= 1e-5, (logit_bad, id_bad, max_rel)


def holes_map(rr):
    """:157-158: every fifth row (i % 5 == 2) and the last 2048 rows without a token, the others id 3i + 7."""
    i = np.arange(rr)
    return np.where((i % 5 == 2) | (i >= rr - 2048), -1, 3 * i + 7).astype(np.int32)


def test_top1_holes(data):
    """:147-211 at min(rows, 106299) = 20,000 rows, 8 columns: masked rows -inf, the others bit for bit native_mmvq's;
    the id against map[host argmax], the probability against the host double softmax; then the mask kernel after a
    plain head the same way (:186-193)."""
    w, xq, _ = data
    rr = min(ROWS, 106299)
    mp = holes_map(rr)
    mapped = mp >= 0
    bad = 0
    max_rel = 0.0
    for col in range(MT):
        xc = column(xq, col)
        a = gpu.mmvq(13, w[:rr], xc, N_IN, rr, 1)[0]
        b, b_id, b_p, _ = gpu.q5k_top1(w[:rr], xc, rr, mp)
        bad += int(np.count_nonzero(~np.isneginf(b[~mapped]))) + bad_bits(a[mapped], b[mapped])
        best, s = host_top(a, mapped)
        bad += b_id != mp[best]
        max_rel = max(max_rel, abs(float(b_p) - 1.0 / s) * s)
        c = gpu.q5k_head_masked(w[:rr], xc, N_IN, rr, mp)
        bad += int(np.count_nonzero(~np.isneginf(c[~mapped]))) + bad_bits(a[mapped], c[mapped])
    # basalt: ``if (bad || max_rel > 1e-5) fail = 1;`` (:210)
    assert bad == 0 and max_rel <= 1e-5, (bad, max_rel)


@pytest.mark.parametrize("rr,grid", [(20000, {}), (1000, {}), (20000, {"sm_count": 40, "per_sm": 1}),
                                     (5000, {"sm_count": 3, "per_sm": 1})])
def test_top1_emulation_bit_for_bit(data, rr, grid):
    """The partials, the id and the probability bit for bit against the numpy emulation of the take and merge
    (basalt_emu.q5k_top1) on the kernel's logits, with the GPU's expf_ftz; with and without the holes map; at basalt's
    grid and others (the probability's last bits follow the grid, mtp.cpp:1007-1008)."""
    w, xq, _ = data
    blocks = kernels.basalt_q5k_head_grid(rr, **grid)
    for mp in (None, holes_map(rr)):
        y, b_id, b_p, part = gpu.q5k_top1(w[:rr], column(xq, 3), rr, mp, **grid)
        mapped = np.ones(rr, bool) if mp is None else mp >= 0
        e_part, e_row, e_p = emu.q5k_top1(y, mapped, blocks, gpu.expf_ftz)
        assert bad_bits(part[:, 0], e_part[0]) == 0 and bad_bits(part[:, 2], e_part[2]) == 0
        assert np.array_equal(part[:, 1].view(np.int32), e_part[1].astype(np.int32))
        e_row = e_row if e_row < rr else 0             # no row taken (every row masked): row 0 (q5k_head.cu:330)
        assert b_id == (e_row if mp is None else mp[e_row]) and bad_bits(b_p, e_p) == 0, (b_id, e_row, b_p, e_p)
