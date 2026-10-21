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
- HR-EMU: bit for bit against ``top10_emulation`` (``route``, native_router.cu:46-100, with the GPU's own
  ``expf_ftz``) on the four fixtures, the planted tokens (``planted_rows``, one dispatch) and 12 random windows of 8
  tokens; the planted tokens' literal outputs, and every regular token's selected sum at least 2^-9 (so the 2^-14
  floor fires only on the degenerate tokens, whose weights show it: 0xff800000, not +0).
- HR-WIN: invariant 1 on the kernel alone, bitwise against its own one-token calls (docs/design/flash-next.md:86-91,
  :185-186): every token of "random normal" with the planted tokens in rows 1, 5, ..., 49, in the windows of tokens
  0..n-1 (n = 1..8 and 64), of widths 2..8 at row 56 and reversed, hc_fused_parity's window rule
  (hc_fused_parity.cpp:328, its message :333).

Every test checks the sentinels (the ids' and weights' row past the window keeps its 0xFF bytes, ``gpu.top10``) and
counts the weights' non-finite values."""

import numpy as np

from tests import basalt_hc_ref as ref
from tests.kernels import basalt_emu as emu
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


DEGENERATE = ("P7nan", "P7inf", "P7ninf")            # every p NaN: ids 0..9, every weight 0xff800000


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


def random_windows():
    """The 12 random windows of 8 tokens, ``N(0, s^2)`` logits for s = 0.5, 2 and 6, four per scale, in that order:
    float64 normals of ``default_rng(20261018)`` times s, rounded to float32."""
    rng = np.random.default_rng(20261018)
    return {f"N(0, {s}^2) window {j}": (rng.standard_normal((8, 512)) * s).astype(np.float32)
            for s in (0.5, 2.0, 6.0) for j in range(4)}


def test_matches_route_emulation():
    planted = ref.planted_rows()
    names = list(planted)
    fixtures = {**ref.router_cases(), "planted rows": np.stack(list(planted.values())), **random_windows()}
    bad, out = [], {}
    for name, logits in fixtures.items():
        n = len(logits)
        ids, w = gpu.top10(logits, n)
        want_ids, want_w, e, S, _, tot = ref.top10_emulation(logits, gpu.expf_ftz)
        off = np.flatnonzero((ids != want_ids).any(1) | (bits(w) != bits(want_w)).any(1))
        regular = ~np.isin(names, DEGENERATE) if name == "planted rows" else np.ones(n, bool)
        nonfinite = np.count_nonzero(~np.isfinite(w[regular]))
        print(f"  {name:<26} {len(off)} of {n} tokens differ from the emulation   non-finite {nonfinite}   "
              f"min tot {tot[regular].min():.6e}")
        if len(off):
            k = off[0]
            bad.append(f"{name}: {len(off)} tokens differ from route's emulation; the first, token {k}: ids "
                       f"{ids[k].tolist()} against {want_ids[k].tolist()}, weights {[hex(b) for b in bits(w[k])]} "
                       f"against {[hex(b) for b in bits(want_w[k])]}")
        if nonfinite:
            bad.append(f"{name}: {nonfinite} non-finite weights of regular tokens (must be 0)")
        if not (tot[regular] >= 2.0 ** -9).all():                          # HR-CLAMP's GPU half
            bad.append(f"{name}: a regular token's selected sum {tot[regular].min()!r} is below 2^-9")
        out[name] = ids, w, e, S
    ids, w, e, S = out["planted rows"]
    row = {nm: k for k, nm in enumerate(names)}
    first = lambda nm, m: ids[row[nm]][:m].tolist()                       # noqa: E731
    literal = [
        ("P1", first("P1", 7) == [3, 8, 40, 72, 300, 200, 201], "ids begin 3, 8, 40, 72, 300, 200, 201"),
        ("P2", ids[row["P2"]][5:].tolist() == np.sort(31 * np.arange(12) + 7)[:5].tolist(),
         "ranks 5..9 are the five lowest of the twelve ids 31 j + 7"),
        ("P3", first("P3", 10) == list(range(10)) and (bits(w[row["P3"]]) == 0x3DCCCCCD).all(),
         "ids 0..9, weights 0x3dcccccd"),
        *((nm, first(nm, 10) == list(range(10)) and (bits(w[row[nm]]) == 0xFF800000).all(),
           "ids 0..9, weights 0xff800000 (the floor fires)") for nm in DEGENERATE),
        ("D", first("D", 4) == [3, 7, 40, 300], "ids begin 3, 7, 40, 300"),
        ("E", first("E", 10) == [17, 33, 400, 0, 1, 2, 3, 4, 5, 6]
         and bits(w[row["E"]]).tolist() == [0x3EAAAAAB] * 3 + [0] * 7,
         "ids 17, 33, 400, 0..6, weights 0x3eaaaaab x 3 and +0 x 7"),
        ("F", ids[row["F"]][9] == 1 and bits(w[row["F"]])[9] == 0, "the tenth id 1, its weight +0"),
    ]
    k = row["F"]                                  # F's precondition: e normal, e * r subnormal before the flush
    ek, rk = float(e[k, 1]), float(emu.rcp_ftz(S[k]))
    literal.append(("F", 2.0 ** -126 <= ek < np.inf and ek * rk < 2.0 ** -126,
                    f"expert 1's e = {ek!r} normal and e * r = {ek * rk!r} below 2^-126"))
    bad += [f"planted {nm}: {what}: ids {ids[row[nm]].tolist()}, weights {[hex(b) for b in bits(w[row[nm]])]}"
            for nm, ok, what in literal if not ok]
    assert not bad, "\n".join(bad)


def test_a_token_does_not_depend_on_the_window():
    planted = ref.planted_rows()
    W = ref.router_cases()["random normal"].copy()
    at = 1 + 4 * np.arange(len(planted))                      # rows 1, 5, ..., 49
    W[at] = np.stack(list(planted.values()))
    one = [gpu.top10(W[k:k + 1], 1) for k in range(NT)]       # every token alone: its T = 1 value
    one_ids, one_w = np.concatenate([i for i, _ in one]), np.concatenate([w for _, w in one])
    runs = {f"tokens 0..{n - 1}": np.arange(n) for n in (*range(1, 9), NT)}
    runs.update({f"rows 56..{55 + n}": 56 + np.arange(n) for n in range(2, 9)})
    runs["W reversed"] = np.arange(NT)[::-1]                  # token k in row 63 - k
    bad, off = 0, []
    for name, rows in runs.items():
        ids, w = gpu.top10(W[rows], len(rows))
        d = np.count_nonzero(ids != one_ids[rows]) + np.count_nonzero(bits(w) != bits(one_w[rows]))   # :328
        bad += d
        if d:
            off.append(f"{name}: {d} values")
    regular = ~np.isin(np.arange(NT), at[[list(planted).index(nm) for nm in DEGENERATE]])
    nonfinite = np.count_nonzero(~np.isfinite(one_w[regular]))
    msg = f"{bad} values of a token differ between window sizes (must be 0)"         # hc_fused_parity.cpp:333
    print(f"  {msg}   ({len(runs)} windows against {NT} one-token dispatches; non-finite {nonfinite})")
    assert not bad, (f"{msg}: " + "; ".join(off) + "; if (std::memcmp(&by_t[T][ot + i], &by_t[t_hi][oh + i], 4) != 0) "
                     "++bad; (hc_fused_parity.cpp:328)")
    assert not nonfinite, f"{nonfinite} non-finite weights of regular tokens (must be 0)"
