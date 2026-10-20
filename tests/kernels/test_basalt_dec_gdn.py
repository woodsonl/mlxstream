# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_gdn_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): its fixtures, pre-fills and pass criteria.
"""dec_gdn_parity's GPU checks (basalt src/kernels/dec_gdn_parity.cpp) on the Metal GDN decode window
(kernels/common/basalt_gdn.metal), through tests/kernels/basalt_gdn_gpu.py, on tests/basalt_gdn_ref.py's fixtures and
float64 references. In every test every output is finite and every sentinel intact.

- A1, the port's addition (spec T§5.5: no basalt test grades gdn_ab_multi's gate and beta against a reference): on
  parity_gdnf's inputs, combo by combo and for every window width T = 1..8 of the 8-row x, gdn_ab_multi's gate and
  beta within ``max_err < 1e-4`` of the float64 ``ab64``, the bound of the nearest stage (``ec``,
  dec_gdn_parity.cpp:551; flash-next.md's rule for kernels basalt checks only against another GPU kernel).
"""

import pytest

from tests.basalt_gdn_ref import COMBOS, MT, ab64, combo_case, max_err, n_bad
from tests.kernels import basalt_gdn_gpu as gg


@pytest.mark.parametrize("combo", range(len(COMBOS)))
def test_gdn_ab_multi_matches_the_float64_reference(combo):
    layer, x = combo_case(combo)
    ref = ab64(x, layer.wa, layer.wb, layer.dt, layer.ssm_a)
    for t in range(1, MT + 1):
        gate, beta, ok = gg.ab(layer, x[:t])
        assert ok, f"T={t}: gdn_ab_multi wrote past its window's rows (a sentinel byte changed)"
        for name, got, want in zip(("gate", "beta"), (gate, beta), ref):
            assert n_bad(got) == 0, f"T={t}: {n_bad(got)} non-finite {name} values"
            e = max_err(got, want[:t])
            assert e < 1e-4, f"T={t}: {name}: max_err {e:.3g}, want max_err < 1e-4"
