# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp, src/kernels/ple_win_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's checks on its block oracle (the
# conv input and the conv tap order, :984-1025) and their pass criteria; the fixture's properties (spec T§6.3).
"""Host-only checks of the n-gram block's references (tests/basalt_ple_ref.py); no GPU. Each runs the 12 rows of the
synthetic fixture (seed ``SYNTHETIC_SEED``) through R-struct, from one initial history (spec T§6.1).

- FX.2 ``test_fixture_properties``: the fixture's four properties (T§6.3: hidden nonzero, min |s| >= 0.1, the
  block's share of result >= 0.3 on every row, 12 rows from one history), and the same seed gives the same bytes.
- C13 ``test_conv_input_is_normalized_not_gated`` (ple_parity.cpp:984-989): on every row, convolving gated instead of
  normalized would be visible, criterion quoted: ``if (!(rel_norm_gated > 0.05)) ++bad;`` (:988).
- C14 ``test_tap_order_rivals_are_visible`` (:992-1025): on every row, R-struct's conv_out against the reversed taps
  (basalt's rival, P[t + 3 (3 - k)]) and against the transposed kernel (conv1d read as flat [k * 10240 + i], the bug
  of basalt's first reference, :1012-1016), each with ``if (!gt(rel, 0.05)) ++bad;`` (:1024); here on the synthetic
  taps."""

import numpy as np
import pytest

from tests.basalt_ple_ref import (D, DIL, KERN, ROWS, Fixture, check_fixture, conv, gt, r_struct, rel_l1, silu,
                                  synthetic, taps)


@pytest.fixture(scope="module")
def synth():
    fx = synthetic()
    return fx, r_struct(fx)


def test_fixture_properties(synth):
    fx, ref = synth
    min_s, share = check_fixture(fx, ref)          # the four properties, each asserted with T§6.3's words
    again = synthetic()
    differ = [name for name, a, b in zip(Fixture._fields, fx, again)
              if (a.dtype, a.shape, a.tobytes()) != (b.dtype, b.shape, b.tobytes())]
    assert not differ, f"FX.2, \"the same seed gives the same bytes\": two calls differ in {differ}"
    print(f"synthetic fixture: min |s| {min_s:.4f}, min row share {share:.4f}")


def test_conv_input_is_normalized_not_gated(synth):
    _, ref = synth
    rels = [rel_l1(ref.gated[t], ref.normalized[t])[0] for t in range(ROWS)]
    bad = sum(not (rel > 0.05) for rel in rels)
    assert bad == 0, ("`if (!(rel_norm_gated > 0.05)) ++bad;` (ple_parity.cpp:988), rel_norm_gated = "
                      f"rel_l1(gated, normalized): {bad} of {ROWS} rows fail; per row {np.round(rels, 4).tolist()}")


def test_tap_order_rivals_are_visible(synth):
    fx, ref = synth
    P = np.concatenate([fx.hist, ref.normalized])      # R-struct's padded rows: [history | normalized rows]
    w = taps(fx.conv1d)
    rivals = {  # each as basalt accumulates its rival, tap kk = 0..3 in order
        # basalt's rival (:1003-1016): tap kk reads row (K - 1 - kk) * dil, 9, 6, 3, 0; the source reads 0, 3, 6, 9
        "reversed taps, P[t + 3 (3 - k)]": sum(w[:, k] * P[DIL * (KERN - 1 - k):][:ROWS] for k in range(KERN)),
        "transposed kernel, conv1d flat [k * 10240 + i]": conv(P, np.asarray(fx.conv1d, np.float64).reshape(KERN, D).T),
    }
    bad = {}
    for name, rival in rivals.items():
        out = silu(rival)
        rels = [rel_l1(ref.conv_out[t], out[t])[0] for t in range(ROWS)]
        bad[name] = (sum(not gt(rel, 0.05) for rel in rels), np.round(rels, 4).tolist())
    assert all(n == 0 for n, _ in bad.values()), \
        f"`if (!gt(rel, 0.05)) ++bad;` (ple_parity.cpp:1024), rel = rel_l1(conv_out, rival) per row: {bad}"
