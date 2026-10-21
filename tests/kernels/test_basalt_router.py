# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/router_top10_parity.cpp and src/kernels/cuda/native_router.cu @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): router_top10_parity's fixtures
# (tests/basalt_hc_ref.py's router_cases) and its criteria (:83-107), verbatim, on route's port.
"""basalt's router top-10 on Metal (kernels/common/basalt_router.metal): one kernel, ``basalt_router_top10``, for
``route`` (``native_router_top10``, ``native_router_top10_multi``) and the fused read's ``hc_top10``, which give the
same bits. Basalt's router_top10_parity runs its generic ``router_top10``; the contract applies its rule to this kernel
(docs/design/flash-next.md, the router's no-reference row).

- HR-REF: router_top10_parity's rule, ids exact and weights ``rel <= 1e-5`` (router_top10_parity.cpp:86-90), against
  its double host reference (``router_reference``) on its four fixtures, each one dispatch of its 64 tokens, behind a
  guard that refuses near-ties among the reference's ranks 0..10 (``near_tie_guard``).

Every test checks the sentinels (the ids' and weights' row past the window keeps its 0xFF bytes, ``gpu.top10``) and
counts the weights' non-finite values."""

import numpy as np

from tests import basalt_hc_ref as ref
from tests.kernels import basalt_gpu as gpu

NT = 64                            # router_top10_parity.cpp:121, one dispatch per case
TOL = 1e-5                         # run_case's tol (:129, :134, :141, :146)


def test_router_top10_parity():
    cases = ref.router_cases()
    refs = {}
    for name, logits in cases.items():             # the guard, before any dispatch
        r_ids, r_w, _, p11 = ref.router_reference(logits)
        assert ref.near_tie_guard(p11), f"{name}: a near-tie among the reference's ranks 0..10: redraw the fixture"
        refs[name] = r_ids, r_w
    bad = []
    for name, logits in cases.items():
        ids, w = gpu.top10(logits, NT)
        r_ids, r_w = refs[name]
        rel = ref.rel_w(w, r_w)
        id_bad = np.count_nonzero(ids != r_ids)
        w_bad = np.count_nonzero(~(rel <= TOL))
        nonfinite = np.count_nonzero(~np.isfinite(w))
        worst_sum = np.abs(np.cumsum(w.astype(np.float64), axis=1)[:, -1] - 1.0).max()    # its own sum (:92-101)
        print(f"  {name:<26} ids {'*** WRONG ***' if id_bad else 'exact'} ({id_bad} bad)   weights worst rel "
              f"{rel.max():.3e} ({w_bad} over tol)   |sum-1| {worst_sum:.1e}   non-finite {nonfinite}")
        if id_bad:
            bad.append(f"{name}: {id_bad} ids differ: if (h_ids[i] != r_ids[i]) ++id_bad; (router_top10_parity.cpp:86)")
        if w_bad:
            bad.append(f"{name}: {w_bad} weights: if (!(rel <= tol)) ++w_bad; tol = 1e-5 (router_top10_parity.cpp:90)")
        if nonfinite:
            bad.append(f"{name}: {nonfinite} non-finite weights (must be 0)")
    assert not bad, "\n".join(bad)
