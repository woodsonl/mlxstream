"""Host-only checks of basalt's GDN decode window; no GPU. C.5's reference half: the references and fixtures of
tests/basalt_gdn_ref.py.

- Each fixture generator, called twice with its seed, gives the same bytes; the default stream of ``segments`` is
  default_rng((97, trial, nseg)), which a caller drawing past x (the commit's recorded inputs) passes itself; every
  configuration keeps parity_seg's layout rules (dec_gdn_parity.cpp:422-429).
- ``cpu_seg`` is finite on a segment of eight rows fed ``ab64``'s gate and beta, and on its first two rows equals
  basalt's loops (``cpu_seg_loops``) for three value heads whose q/k head is hd % 16, not hd // 3; ``ab64`` equals
  the plain dot of the bf16 rows there.
- ``conv_tail`` is the bit copy of the last three rows of [history | window rows] for 0 to 6 rows, subnormals, -0,
  infinities and NaN payloads included.
"""

import hashlib
import math

import numpy as np
import pytest

from tests.basalt_gdn_ref import (C, COMBOS, LAYER, MT, N, NL, S, SEG_CASES, SEG_SEED, SF, ZV, ab64, combo_case,
                                  conv_tail, cpu_seg, cpu_seg_loops, max_err, n_bad, seg_layer, segments)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_gdn_ref_fixtures_give_the_same_bytes_for_one_seed():
    (layer, x), (again, x2) = combo_case(1), combo_case(1)
    assert layer[:3] == COMBOS[1] and x.shape == (MT, N)
    assert digest(*layer[3:], x) == digest(*again[3:], x2), "combo_case(1) drew different bytes"
    case = segments(2, 5)
    assert digest(*case[3:]) == digest(*segments(2, 5)[3:]) and case[:3] == segments(2, 5)[:3]
    explicit = segments(2, 5, np.random.default_rng((SEG_SEED, 2, 5)))
    assert digest(*explicit[3:]) == digest(*case[3:]), "segments' default stream is not default_rng((97, trial, nseg))"
    for trial, nseg in SEG_CASES:
        T, keep, row0, st0, qkv, z, x = segments(trial, nseg)
        R = sum(T)
        assert len(T) == len(keep) == len(row0) == nseg and all(1 <= t <= MT for t in T)
        assert all(0 <= kk <= t for kk, t in zip(keep, T)) and list(row0) == [sum(T[:k]) for k in range(nseg)]
        if trial == 0:
            assert set(T) == set(keep) == {1}
        if trial == 2:
            assert all(T[k] == MT for k in range(1, nseg, 2))
        assert st0.shape == (nseg, NL, SF) and qkv.shape == (R, C) and z.shape == (R, ZV) and x.shape == (R, N)


def test_cpu_seg_is_finite_on_one_segment_and_follows_basalts_loops():
    layer, case, k, heads = seg_layer(), segments(2, 2), 1, (0, 17, 40)
    rows = slice(case.row0[k], case.row0[k] + case.T[k])
    assert case.T[k] == MT
    st0, qkv, z, x = case.st0[k, LAYER], case.qkv[rows], case.z[rows], case.x[rows]
    gate, beta = ab64(x, layer.wa, layer.wb, layer.dt, layer.ssm_a)
    ref = cpu_seg(st0, qkv, z, layer.conv_w, layer.gamma, gate, beta, MT)
    assert ref.h.shape == (MT, C) and ref.y.shape == (MT, ZV)
    assert n_bad(ref.h) == n_bad(ref.y) == n_bad(gate) == n_bad(beta) == 0, "cpu_seg or ab64 is not finite"
    h, y = cpu_seg_loops(st0, qkv[:2], z[:2], layer.conv_w, layer.gamma, gate, beta, 2, heads)
    assert max_err(ref.h[:2], h) < 1e-6, max_err(ref.h[:2], h)
    for (t, hd), want in y.items():
        err = max_err(ref.y[t, hd * S: (hd + 1) * S], want)
        assert err < 1e-6, (t, hd, err)
    for hd in heads:
        wa, wb = ((w[hd].astype(np.uint32) << 16).view(np.float32).tolist() for w in (layer.wa, layer.wb))
        va = vb = 0.0
        for i, xi in enumerate(x[0].tolist()):
            va += wa[i] * xi
            vb += wb[i] * xi
        u = va + float(layer.dt[hd])
        want = ((u if u > 20.0 else math.log1p(math.exp(u))) * float(layer.ssm_a[hd]), 1.0 / (1.0 + math.exp(-vb)))
        assert max_err([gate[0, hd], beta[0, hd]], want) < 1e-6, (hd, gate[0, hd], beta[0, hd], want)


@pytest.mark.parametrize("n", range(7))
def test_conv_tail_copies_the_last_three_rows_bit_for_bit(n):
    ch = np.arange(C, dtype=np.uint32)
    tail = np.stack([(j << 16) | ch for j in range(3)], axis=1)                    # [C][3], oldest first: subnormals
    rows = np.stack([((3 + r) << 16) | ch for r in range(6)])                      # window row r, channel c
    specials = np.array([0x80000000, 0x00000001, 0x7FA00001, 0xFFC00000, 0x7F800000, 0xFF800000], np.uint32)
    tail[:6, 1], tail[100:106, 2], rows[0, 7:13], rows[5, 9:15] = specials, specials, specials, specials
    want = np.stack([tail[:, n + j] if n + j < 3 else rows[n + j - 3] for j in range(3)], axis=1)
    got = conv_tail(tail.reshape(-1).view(np.float32), rows[:n].view(np.float32))
    assert got.dtype == np.float32 and got.shape == (3 * C,)
    bad = np.flatnonzero(got.view(np.uint32) != want.reshape(-1))
    assert bad.size == 0, f"n = {n}: {bad.size} words differ, the first at {bad[:1]}"
