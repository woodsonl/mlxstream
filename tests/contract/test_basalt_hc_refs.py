"""The hyper-connection unit's own test code and kernel sources, GPU-free.

- HF-COMPILE: each of the unit's kernel files, its comments removed, has ``#pragma STDC FP_CONTRACT OFF`` ahead of its
  first function (docs/design/flash-next.md, invariant 5: "Each Metal source starts with ``#pragma STDC FP_CONTRACT
  OFF``"), and no ``atomic`` (no threadgroup of a dispatch waits on or hands work to another, the property
  hc_fused_parity's co-residency check establishes), no ``simd_sum``, ``simd_max``, ``simd_min`` or ``simd_prefix``
  (their reduction order is undocumented: every reduction is a shuffle loop in basalt's offset order), no ``fast::``
  or ``precise::`` (each approximate operation goes through its one shared helper, as basalt_fp.metal's, so kernels
  basalt checks bitwise against each other stay bitwise equal) and no ``while`` (every GPU loop bounded: a ``for`` of
  a fixed trip count).
- HREF-SELF: the host references of tests/basalt_hc_ref.py on inputs whose answers follow by hand from basalt's
  formulas, since a misreading the kernel and its reference share would pass every comparison between them.
- HR-CLAMP, its host half: router_top10_parity's assertion that the router's 2^-14 floor cannot fire for 512 experts
  and top 10 (router_top10_parity.cpp:148-160), and the port's kernel's own bound, a regular token's selected sum at
  least 2^-9; its GPU half is tests/kernels/test_basalt_router.py's HR-EMU (the degenerate tokens' weights)."""

import re
from pathlib import Path

import numpy as np
import pytest

from tests.basalt_hc_ref import RENORM_CLAMP, gr_fixture, gr_write_double, rel_terms, router_cases, router_reference

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("kernels/common/basalt_gr_post.metal", "kernels/common/basalt_router.metal")    # the unit's kernel files
PRAGMA = re.compile(r"^[ \t]*#pragma STDC FP_CONTRACT OFF[ \t]*$", re.M)
BANNED = ("atomic", "simd_sum", "simd_max", "simd_min", "simd_prefix", "fast::", "precise::", "while")


@pytest.mark.parametrize("path", SOURCES)
def test_kernel_sources_keep_the_port_rules(path):
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", (ROOT / path).read_text(), flags=re.S)
    pragma, body = PRAGMA.search(code), code.find("{")
    assert pragma and pragma.start() < body, f"{path}: #pragma STDC FP_CONTRACT OFF does not precede its first function"
    found = [word for word in BANNED if word in code]
    assert not found, f"{path} uses {found}"


def test_references_on_known_values():
    # gr_write_double with a zero injection is R + bo exactly: its weight 2 / (1 + exp(-0 / hc)) is 1.
    R, bo, inj = gr_fixture(inj_sd=0.0)
    want, bo_w = gr_write_double(R, bo, inj, 4)
    bo64 = np.broadcast_to(bo.astype(np.float64)[:, None, :], R.shape)
    assert np.array_equal(bo_w, bo64) and np.array_equal(want, R.astype(np.float64) + bo64)
    # rel_terms rounds nothing to binary32. On this float32 triple basalt's double rule gives 1.000000007e-6, a failure
    # of `const bool plain = worst_plain < 1e-6;` (gr_parity.cpp:645); the denominator |R| + |bo| rounded in binary32
    # (1.875) would give 9.999999747e-7, a pass.
    r, b, got = np.float32(0.9375), np.float32(-0.93749994), np.float32(1.9346046e-06)
    rel = rel_terms(np.float64(r) + np.float64(b), got, r, b)
    assert rel > 1e-6 and abs(rel - 1.000000007e-6) < 1e-15, rel
    # router_reference on "all equal (ties)": every p is 2^-9, the stable order keeps index order, so ids 0..9, and each
    # weight is float32(2^-9 / (10 * 2^-9)), float32(0.1).
    cases = router_cases()
    ids, w, _, _ = router_reference(cases["all equal (ties)"])
    assert (ids == np.arange(10)).all() and (w.view(np.uint32) == 0x3DCCCCCD).all(), (ids, w)
    # On "dominant expert" token t's rank 0 is expert t % 512; the other 511 p are equal, so ranks 1..9 are the nine
    # lowest other ids.
    ids, _, _, _ = router_reference(cases["dominant expert"])
    for t, row in enumerate(ids.tolist()):
        assert row == [t % 512] + [e for e in range(10) if e != t % 512][:9], (t, row)


def test_router_floor_cannot_fire_on_finite_logits():
    K, NE = 10, 512                                   # router_top10_parity.cpp:119-120
    assert K / NE > RENORM_CLAMP, (
        "const double min_possible = (double) K / (double) NE; if (min_possible <= RENORM_CLAMP) { "
        'std::fprintf(stderr, "the clamp CAN trigger, so it needs a test\\n"); ++bad; } '
        "(router_top10_parity.cpp:153, :157-160)")
    # The kernel's own bound: with no NaN, no +inf and a logit above -inf, the maximum's e is 1 and every e is at most
    # 1, so S <= 512 and the largest p, mul_ftz(1, rcp_ftz(S)), is at least 2^-9; the selected sum contains it.
    assert 2.0 ** -9 > RENORM_CLAMP, "a regular token's selected sum (at least 2^-9) could reach the floor"
