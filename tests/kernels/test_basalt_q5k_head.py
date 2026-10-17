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
