# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/gr_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its write fixture (tests/basalt_hc_ref.py's gr_fixture).
"""basalt's hyper-connection write, ``native_gr_post``, on Metal (kernels/common/basalt_gr_post.metal). Basalt gates
none of its values; these are the contract's tests (docs/design/flash-next.md, the no-reference table).

- HG-EMU: bit for bit against the host emulation of the sequence nvcc compiles (``gr_post_emulation``,
  docs/research/basalt-nvcc/native_gr_postops.md section 1), its exponential the GPU's own ``expf_ftz``, at the real
  shape (n_embd 2560, hc 4) at T = 1 and 8 and at a ragged one (n_embd 100, T = 3: the last threadgroup passes the
  last row), with injections planted in every flush and saturation of the weight (one column of every token reads
  the weight out) and residual/block-output pairs whose sum is subnormal or cancels; each planted weight's exact
  value (1, +0 or 2) checked apart from the exponential; in basalt's in-place call and in the program's separate
  output, which give the same bytes.

Every test checks the sentinels (the output's rows T..7 keep their 0xFF bytes, the block output's and injections'
buffers their input bytes) and counts the output's non-finite values."""

import numpy as np
import pytest

from tests import basalt_hc_ref as ref
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32, U32 = np.float32, np.uint32
HC = 4
FILL = 0xFFFFFFFF                  # gpu.gr_post's 0xFF bytes: the rows past T, and the separate output


def write(R, bo, inj, T, n_embd, in_place=True):
    """``gpu.gr_post`` with the sentinels checked: the output's rows T..7 keep their 0xFF bytes, and the block
    output's and injections' buffers hold what was written (rows < T the inputs, the rest 0xFF). The 8 output rows."""
    out, bo_after, inj_after = gpu.gr_post(R, bo, inj, T, n_embd, HC, in_place=in_place)
    assert np.all(out[T:].view(U32) == FILL), "a store into the rows past the window"
    for name, a, after in (("block_out", bo, bo_after), ("inject", inj, inj_after)):
        want = np.full(after.shape, FILL, U32)
        want[:T] = np.ascontiguousarray(a, F32)[:T].view(U32)
        assert np.array_equal(after.view(U32), want), f"the kernel changed {name}"
    return out


# HG-EMU's injections with the weight each must give exactly (None: a NaN stream), ordered so that the ragged case's
# 12 slots hold every kind (native_gr_postops.md section 1): x = inj / 4 is +0 (+-0, +-subnormal: w = 1); 1 + e is
# above 2^126, its reciprocal flushed (-350, -352: w = +0); e overflows (-356, -1e30, -inf: w = +0); 1 + e rounds to
# 1 (70, 100: w = 2); e is flushed or 0 (351, 1e30, +inf: w = 2).
PLANTS = ((0.0, 1), (-350.0, 0), (70.0, 2), (np.nan, None), (-0.0, 1), (-356.0, 0), (351.0, 2), (np.inf, 2),
          (1e-40, 1), (-1e30, 0), (100.0, 2), (-np.inf, 0), (-1e-40, 1), (-352.0, 0), (1e30, 2))
# Token 0's stream 0 (injection +0, so w = 1 exactly): (block_out, residual, the stored bits) in columns 0..7.
PAIRS = ((1.5 * 2.0 ** -126, -(2.0 ** -126), 0x00000000),     # sum 2^-127, subnormal: flushed to +0
         (-1.5 * 2.0 ** -126, 2.0 ** -126, 0x80000000),       # -2^-127: flushed to -0
         (0.75, -0.75, 0x00000000),                           # cancels: +0
         (-0.0, -0.0, 0x80000000),                            # -0 + -0
         (0.0, -0.0, 0x00000000),                             # +0 + -0
         (0.5, 1e-40, 0x3F000000),                            # a subnormal residual is read as +0
         (-1e-40, 3.0, 0x40400000),                           # so is a subnormal block output
         (2.0, None, 0x7FFFFFFF))                             # a NaN residual (0xffc00001): sm_120's NaN
NAN_R = 0xFFC00001
W_COL = len(PAIRS)                 # block output 1 and residual +0 in every stream: the output is the weight itself


def planted(n_embd, T, plant):
    """gr_fixture's residuals (stream scales 1, 4, 16, 64) and block outputs with uniform injections in [-100, 100];
    ``plant``: PLANTS in the injections' first slots, PAIRS in token 0's stream 0, and column W_COL of every token
    reading out each stream's weight (``fma(1, w, +0) = w``: a weight of +0 and an unflushed reciprocal's 2^-125 store
    the same bits against a residual of order 1). Also the planted slot count."""
    R, bo, _ = ref.gr_fixture(n_embd, HC, T)
    inj = np.random.default_rng(5).uniform(-100.0, 100.0, (T, HC)).astype(F32)
    k = min(len(PLANTS), T * HC) if plant else 0
    inj.reshape(-1)[:k] = [v for v, _ in PLANTS[:k]]
    if plant:
        for d, (b, r, _) in enumerate(PAIRS):
            bo[0, d] = b
            R[0, 0, d] = np.nan if r is None else r
        R.view(U32)[0, 0, len(PAIRS) - 1] = NAN_R
        bo[:, W_COL], R[:, :, W_COL] = 1.0, 0.0
    return R, bo, inj, k


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "separate_output"])
@pytest.mark.parametrize("n_embd, T, plant", [(2560, 1, False), (2560, 8, True), (100, 3, True)])
def test_matches_emulation_bit_for_bit(n_embd, T, plant, in_place):
    """HG-EMU. Every element of rows < T has the emulation's bits; each planted injection's stream has the bits of
    its exact weight (``fma(w, bo, R)``, ``.FTZ``, with no exponential; at column W_COL the weight's own bits); token
    0's stream 0 the PAIRS' bits; the non-finite outputs are exactly the planted NaN stream and residual. The separate
    output equals basalt's in-place call byte for byte, all 8 rows."""
    R, bo, inj, k = planted(n_embd, T, plant)
    out = write(R, bo, inj, T, n_embd, in_place)
    got = out[:T]
    want = ref.gr_post_emulation(R, bo, inj, HC, gpu.expf_ftz)
    bad = np.argwhere(got.view(U32) != want.view(U32))
    nonfinite = np.count_nonzero(~np.isfinite(got))
    planted_nan = (n_embd + 1) if plant else 0               # the NaN injection's stream, and the NaN residual
    print(f"HG-EMU n_embd {n_embd} T {T} {'in place' if in_place else 'separate output'}: {len(bad)} of {got.size} "
          f"elements differ; {nonfinite} non-finite ({planted_nan} planted)")
    if len(bad):
        t, c, d = bad[0]
        print(f"  first at (t, c, d) = ({t}, {c}, {d}): inject {inj[t, c]!r}, R {R[t, c, d]!r}, bo {bo[t, d]!r}: "
              f"got {got[t, c, d].view(U32):#010x}, want {want[t, c, d].view(U32):#010x}")
    assert len(bad) == 0, f"{len(bad)} elements differ from the emulation"
    for j in range(k):
        t, c = divmod(j, HC)
        w = PLANTS[j][1]
        exact = np.full(n_embd, 0x7FFFFFFF, U32) if w is None else \
            ref.cuda_nan(emu.fma_ftz(F32(w), bo[t], R[t, c])).view(U32)
        assert np.array_equal(got[t, c].view(U32), exact), f"inject {PLANTS[j][0]} does not give w = {w}"
    if plant:
        assert [int(v) for v in got[0, 0, : len(PAIRS)].view(U32)] == [p[2] for p in PAIRS]
    assert nonfinite == planted_nan, "a non-finite output nothing planted"
    if not in_place:
        assert np.array_equal(out.view(U32), write(R, bo, inj, T, n_embd, True).view(U32)), \
            "the separate output's bytes differ from basalt's in-place call"
