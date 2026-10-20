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
- S8, its per-stage half (dec_gdn_parity.cpp:501, :551): on parity_seg's configurations (trial x segment count), each
  segment alone from its own layer-1 state (decision D12, spec T§7.2: no ``_seg`` kernel), gdn_conv_l2_multi's h
  within ``max_err < 1e-4`` of the float64 ``cpu_seg``'s h, with no non-finite value (every 0xFF pre-fill overwritten).
- S9, its per-stage half (dec_gdn_parity.cpp:502, :551): on the same segments, the per-stage chain basalt's reference
  side runs (:457-460: gdn_conv_l2_multi's h and gdn_ab_multi's gate and beta into gdn_step_norm_multi), its y within
  ``max_err < 1e-3`` of the float64 ``cpu_seg``'s y, ``cpu_seg`` fed the gate and beta the kernel consumed (:496-499),
  with no non-finite value in h, gate, beta or y.
"""

import numpy as np
import pytest

from tests.basalt_gdn_ref import (COMBOS, F32, HV, LAYER, MT, SEG_CASES, ab64, combo_case, cpu_seg, max_err, n_bad,
                                  seg_layer, segments)
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


@pytest.mark.parametrize("trial,nseg", SEG_CASES)
def test_gdn_conv_l2_multi_matches_the_float64_reference(trial, nseg):
    layer, seg = seg_layer(), segments(trial, nseg)
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        st, rows = seg.st0[k, LAYER], slice(r0, r0 + t)
        h, ok = gg.conv_l2(st, seg.qkv[rows], layer.conv_w)
        assert ok, f"segment {k} (T={t}): gdn_conv_l2_multi wrote past its window's rows (a sentinel byte changed)"
        assert n_bad(h) == 0, f"segment {k} (T={t}): {n_bad(h)} non-finite h values, want no non-finite value"
        nil = np.zeros((t, HV), F32)                                  # h reads no gate
        e = max_err(h, cpu_seg(st, seg.qkv[rows], seg.z[rows], layer.conv_w, layer.gamma, nil, nil, t).h)
        assert e < 1e-4, f"segment {k} (T={t}): h: max_err {e:.3g}, want max_err < 1e-4"


@pytest.mark.parametrize("trial,nseg", SEG_CASES)
def test_gdn_step_norm_multi_matches_the_float64_reference(trial, nseg):
    layer, seg = seg_layer(), segments(trial, nseg)
    for k, (t, r0) in enumerate(zip(seg.T, seg.row0)):
        st, rows = seg.st0[k, LAYER], slice(r0, r0 + t)
        h, ok_h = gg.conv_l2(st, seg.qkv[rows], layer.conv_w)
        gate, beta, ok_ab = gg.ab(layer, seg.x[rows])
        y, ok = gg.step_norm(st, h, gate, beta, seg.z[rows], layer.gamma)
        assert ok_h and ok_ab and ok, f"segment {k} (T={t}): a per-stage kernel wrote past its window's rows"
        for name, a in (("h", h), ("gate", gate), ("beta", beta), ("y", y)):
            assert n_bad(a) == 0, f"segment {k} (T={t}): {n_bad(a)} non-finite {name} values, want no non-finite value"
        e = max_err(y, cpu_seg(st, seg.qkv[rows], seg.z[rows], layer.conv_w, layer.gamma, gate, beta, t).y)
        assert e < 1e-3, f"segment {k} (T={t}): y: max_err {e:.3g}, want max_err < 1e-3"
