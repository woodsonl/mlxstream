# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp and src/kernels/gr_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's native key-norm check's metric
# and pass criterion (:899-910), and gr_parity's scalar activation contract's degenerate norm input and the exactness
# it relies on (:167-236).
"""The n-gram (PLE) block's Metal port on the GPU, kernel by kernel (kernels/common/basalt_gr_norm.metal). Each call
runs at its call sites' shape, 4 rows of 2560 with 4 gamma rows and eps = f32(1e-6) (native_ple_postops.cu:352-353,
:358), unless stated.

- T-NORM ``test_grouped_norm_float64`` (spec T§7.2): basalt_gr_norm against R-struct's float64 ``gnorm``
  (tests/basalt_ple_ref.py), criterion quoted:
  ``const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;`` (ple_parity.cpp:910), rel =
  rel_l1(reference, GPU) over one call's 4 x 2560 outputs and the non-finite count over the norm's own output (basalt
  counted it over the block's result). The calls: N(0, σ) rows for σ in {1e-3, 1e-2, 1, 1e2} (at 1e-3 the mean square
  is about eps) and a zero row, whose output is 0, each with weights f32(1 + 0.1 N(0, 1)).
- T-NORM ``test_grouped_norm_bitwise_emu``: bit for bit against R-emu, ``basalt_emu.gr_norm`` (the rounding sequence of
  docs/research/basalt-nvcc/ple.md section 1) with the GPU's rsqrtf_ftz (MUFU.RSQ and fast::rsqrt have no bit-exact
  host model), on the calls above plus a row whose squares all lie below 2^-126, each of which .FTZ turns into +0; and
  on G1a, gr_parity's degenerate input (gr_parity.cpp:169-201: GrShapes{2, 1, 2}, so one row of 2 columns and the
  256-thread block; R = 1, eps = 0, gamma in {1.0f, 1.00390625f}). There, "At R=1 and eps=0, xn=gamma exactly"
  (:169), which basalt's exact checks rely on (:233-236): the scale is rsqrtf(1) and the output equals gamma bit for
  bit, provided the GPU's fast::rsqrt(1.0f) is exactly 1.0f. A failure there is a departure from basalt, to record,
  not to tune away."""

import numpy as np
import pytest

from tests.basalt_ple_ref import D, EPS, H, N, gnorm, le, nonfinite, rel_l1
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_ple_gpu as ple_gpu

F32, F64 = np.float32, np.float64
SIGMAS = (1e-3, 1e-2, 1.0, 1e2)
NORM_SEED = 910     # default_rng(NORM_SEED) draws, in order: each σ's row then its weights, the zero row's weights,
                    # then the tiny row and its weights
TINY = "squares below 2^-126"


def norm_calls():
    """The standalone norm calls, ``{id: (x, w)}``, each a D-vector (4 rows of 2560) with its own weights: spec T§7.2's
    N(0, σ) rows and zero row, then the tiny row, f32(N(0, 1) 2^-70): normal values whose squares all lie below
    2^-126 (asserted)."""
    rng = np.random.default_rng(NORM_SEED)

    def weights():
        return (1 + 0.1 * rng.standard_normal(D)).astype(F32)

    calls = {}
    for s in SIGMAS:
        x = (s * rng.standard_normal(D)).astype(F32)
        calls[f"sigma={s:g}"] = x, weights()
    calls["zero"] = np.zeros(D, F32), weights()
    tiny = (rng.standard_normal(D) * 2.0 ** -70).astype(F32)
    normal, small = (np.abs(tiny) >= 2.0 ** -126).all(), (np.square(tiny.astype(F64)) < 2.0 ** -127).all()
    assert normal and small, "the tiny row: normal values, squares below 2^-127"
    calls[TINY] = tiny, weights()
    return calls


CALLS = norm_calls()
# G1a: (x, gamma, n_rows, n_cols, gamma_rows, eps) for gr_parity's n_embd = 2, hc = 1 norm at R = 1, eps = 0
G1A = {f"G1a gamma={g}": (np.ones(2, F32), np.full(2, g, F32), 1, 2, 1, 0.0) for g in (1.0, 1.00390625)}
BITWISE = {**{k: (x, w, H, N, H, EPS) for k, (x, w) in CALLS.items()}, **G1A}


def bits(a):
    return np.ascontiguousarray(a, F32).view(np.uint32)


@pytest.mark.parametrize("call", [k for k in CALLS if k != TINY])
def test_grouped_norm_float64(call):
    x, w = CALLS[call]
    actual = ple_gpu.gr_norm(x, w, H, N, H, EPS)
    rel = rel_l1(gnorm(x, w), actual)[0]
    nf = nonfinite(actual)
    print(f"grouped norm [{call}]: rel {rel:.3e}, nf {nf}")
    assert le(rel, 2e-6) and nf == 0, (
        "`const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;` (ple_parity.cpp:910), rel = "
        f"rel_l1(R-struct's gnorm, the GPU) over the call's 4 x 2560 outputs, nf over the norm's output: rel {rel!r}, "
        f"nf {nf}")
    if call == "zero":
        assert (actual == 0).all(), f"spec T§7.2, a zero row: \"the output is 0\": {actual[actual != 0][:8]}"


@pytest.mark.parametrize("call", list(BITWISE))
def test_grouped_norm_bitwise_emu(call):
    x, w, n_rows, n_cols, gamma_rows, eps = BITWISE[call]
    got = ple_gpu.gr_norm(x, w, n_rows, n_cols, gamma_rows, eps)
    want = emu.gr_norm(x, w, n_rows, n_cols, gamma_rows, eps, ple_gpu.rsqrtf_ftz)
    diff = np.flatnonzero(bits(got) != bits(want))
    assert diff.size == 0, (
        f"T-NORM, bit for bit against basalt_emu.gr_norm (ple.md section 1): {diff.size} of {got.size} outputs differ; "
        f"first at {diff[:4].tolist()}: GPU {[hex(v) for v in bits(got).ravel()[diff[:4]]]}, "
        f"R-emu {[hex(v) for v in bits(want).ravel()[diff[:4]]]}")
    if call in G1A:
        one = hex(bits(ple_gpu.rsqrtf_ftz(np.ones(1, F32)))[0])
        out, gamma = [hex(v) for v in bits(got).ravel()], [hex(v) for v in bits(w)]
        print(f"{call}: output {out}, gamma {gamma}, the GPU's rsqrtf_ftz(1.0f) {one}")
        assert out == gamma, (
            f"G1a, \"At R=1 and eps=0, xn=gamma exactly\" (gr_parity.cpp:169): output {out} != gamma {gamma} with the "
            f"GPU's rsqrtf_ftz(1.0f) = {one}; a departure from basalt (gr_parity.cpp:233-236), to record, not to tune "
            "away (spec T§7.2)")
