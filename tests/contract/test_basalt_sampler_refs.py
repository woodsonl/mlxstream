# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): the min_p values its fixtures pass and its references' host arithmetic.
"""Host-only checks of the sampler family's references (tests/basalt_sampler_ref.py); no GPU.

- ``nv_logf`` (CUDA's logf) of the seven ``min_p`` values sampler_parity's fixtures pass is the correctly rounded
  logarithm's float, so the min_p threshold the references compute is the exact one. The table gives each value's
  correctly rounded logf and the exact logarithm's distance from it in ulps: 0.3f and 0.9f lie within 0.07 and 0.09
  ulp of a rounding midpoint, so a logf with a larger error before rounding may return the neighbour there.
  ``math.log``'s binary64 result, rounded to float32, confirms the table: its error is below 2^-28 float ulps.
"""

import math

import numpy as np
import pytest

from tests.basalt_sampler_ref import f2u, nv_logf, u2f

# min_p (float bits), its correctly rounded logf (float bits), the exact logarithm's offset from that float (ulps)
MIN_P_LOGF = [(0x3D4CCCCD, 0xC03FBA14, 0.205), (0x3DCCCCCD, 0xC0135D8E, 0.197), (0x3E4CCCCD, 0xBFCE0210, 0.377),
              (0x3E99999A, 0xBF9A1BC8, 0.435), (0x3ECCCCCD, 0xBF6A9207, -0.278), (0x3F000000, 0xBF317218, 0.032),
              (0x3F666666, 0xBDD7C745, 0.412)]


@pytest.mark.parametrize("x, want, offset", MIN_P_LOGF, ids=[f"{u2f(x):.2g}" for x, _, _ in MIN_P_LOGF])
def test_basalt_sampler_ref_nv_logf_of_min_p_correctly_rounded(x, want, offset):
    exact = math.log(u2f(x))
    assert f2u(float(np.float32(exact))) == want, "the table's logf is not the correctly rounded one"
    ulp = 2.0 ** (math.frexp(u2f(want))[1] - 24)                          # the float's ulp
    assert abs((exact - u2f(want)) / ulp - offset) < 1e-3, "the table's offset"
    got = f2u(float(nv_logf(u2f(x))))
    assert got == want, f"nv_logf({u2f(x)!r}f = {x:#010x}) = {got:#010x}; the correctly rounded logf is {want:#010x}"
