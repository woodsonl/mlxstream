# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/expert.cpp:143-148, :336-370, :503-556 and
# src/kernels/cpu/expert_multi_test.cpp:56-72 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE):
# the two kernels' operation order and its per-token independence check.
"""basalt's two CPU Q2_0 kernels on ARM NEON (runtime/src/basalt_cpu.cpp), against the numpy emulation of the
AVX-512 semantics they transcribe (tests/kernels/basalt_cpu_emu.py) and, for the row dot, against an independent
float64 reference.

Comparison policy, the same everywhere: **float32 bits** where the result is not a NaN, **NaN-ness** where either
side is. The two platforms' default NaN differs (x86 sets the sign bit of the ``0 * inf`` indefinite, AArch64 does
not), nothing downstream reads those bits, and ``abs``, MAXPS, the ``>= 0`` compare and the truncation all ignore
the payload. Where a case pins a value rather than a match - the all-zero chunk's ``+0.0``, the band's -127 codes,
the untouched output rows - the test asserts that value directly, because the kernel and the emulator share one
reading of the specification and a case that drifts into a harmless regime would otherwise still pass.

The one compiler-dependent value in either kernel is ``amax`` when a chunk holds a NaN: ``_mm512_reduce_max_ps``
is header-defined, and GCC's first two levels pass the high half to MAXPS as SRC1 where Clang's pass the low half.
The port writes GCC's order; case A6 asserts it at the positions where the two actually differ and prints Clang's
value beside it, so a flip is a one-line change in both the kernel and the emulator.

Cases (the plan's section 7). A, ``act_quant_q8_1``: A1 the two caller widths, A2 an all-zero chunk, A3 +-0 and the
maximum planted in each of the 32 positions, A4 the scale band's four regimes, A5 +-inf, A6 the NaN sweep over both
reduction orders, A7 the half-away-from-zero rounding and the ``0.49999997f`` tie, A8 the saturation trap, A9 the
contraction trap. B, ``q2_0_gguf_rows_multi``: B1 both pack shapes, B2 nt = 1..8 token independence, B3 the row
ranges and the absolute-index write, B4 stride padding, B5 the unread tail, B6 all 65,536 fp16 block scales, B7
adversarial activations, B8 the independent float64 check, B9 determinism.

Fixtures are the repository's: ``random_rows`` (test_basalt_experts, basalt's own Q2_0 row fixture - random code
bytes with a finite fp16 scale per block) for the bitwise cases, and ``quantize_q2_0`` (test_basalt_iq_parity,
tools/gguf_writer.py's quantizer) for B8, which needs rows that mean something.

NO KERNEL, BINDING OR REFERENCE CHANGE WAS NEEDED: every case below passed against runtime/src/basalt_cpu.cpp,
its binding in runtime/src/native.mm and tests/kernels/basalt_cpu_emu.py as the preceding change wrote them. The
suite was checked to bite by mutating the kernel: the saturating FCVTZS in place of the integer indefinite fails
A4, A5, A6 and A8; Clang's operand order in ``reduce_max_ps`` fails A6 alone; a reversed unpack shift fails every
B case; and the same reversed shift applied to the emulator as well - a misreading the kernel and its reference
share - fails B8 alone, which is what B8 is for.

Run with ``-s`` for the A6 Clang-order table, the A8 pre-clamp margin and the B7 cancellation residue.
"""

import numpy as np
import pytest

from monolith.runtime import _native as _nt
from tests.contract import gguf_reference
from tests.kernels import basalt_cpu_emu as emu
from tests.kernels.basalt_emu import fma32
from tests.kernels.test_basalt_experts import random_rows
from tests.kernels.test_basalt_iq_parity import quantize_q2_0

F32 = np.float32
H, MAXC, QKA, QK, BB, MAXT = emu.H, emu.MAXC, emu.QKA, emu.QK, emu.BB, emu.MAXT
TWO_M128 = F32(2.0) ** -128                  # s <= this gives inv = +inf (plan section 1, correction 1)
TWO_M149 = F32(2.0) ** -149                  # the smallest positive float32


# ---- the comparison policy and the two kernels through their bindings --------------------------------------------

def recip(s):
    """``1.f / s`` in float32, overflowing to +inf quietly as basalt's step 6 does."""
    with np.errstate(divide="ignore", over="ignore"):
        return (F32(1.0) / F32(s)).astype(F32)


def same_f32(got, want, what):
    """float32 bits where neither side is a NaN, NaN-ness where either is."""
    got = np.ascontiguousarray(got, F32).reshape(-1)
    want = np.ascontiguousarray(want, F32).reshape(-1)
    gn, wn = np.isnan(got), np.isnan(want)
    bad = np.flatnonzero(gn != wn)
    assert bad.size == 0, f"{what}: NaN-ness differs at {bad[:8]} (got {got[bad[:8]]}, want {want[bad[:8]]})"
    gb, wb = got[~gn].view(np.uint32), want[~wn].view(np.uint32)
    bad = np.flatnonzero(gb != wb)
    assert bad.size == 0, f"{what}: bits differ at {bad[:8]} (got {got[~gn][bad[:8]]}, want {want[~wn][bad[:8]]})"


def kernel_act(x):
    """``act_quant_q8_1`` through the binding, trimmed to what it writes: (q[:n], scale, sum, hx, nchunks)."""
    x = np.ascontiguousarray(x, F32)
    q, scale = np.zeros(H, np.int8), np.zeros(MAXC, F32)
    total, hx = np.zeros(MAXC, np.int32), np.zeros(MAXC, F32)
    nchunks = _nt.basalt_act_quant_q8_1(x, q, scale, total, hx)
    return q[: x.size], scale[:nchunks], total[:nchunks], hx[:nchunks], nchunks


def check_act(x, order="gcc"):
    """kernel == emulator on every output. Returns the kernel's (q, scale, sum, hx, nchunks)."""
    got, want = kernel_act(x), emu.act_quant_q8_1(x, order)
    assert got[4] == want[4], f"nchunks {got[4]} != {want[4]}"
    assert np.array_equal(got[0], want[0]), f"codes differ at {np.flatnonzero(got[0] != want[0])[:8]}"
    assert np.array_equal(got[2], want[2]), f"sum differs at {np.flatnonzero(got[2] != want[2])[:8]}"
    same_f32(got[1], want[1], "scale")
    same_f32(got[3], want[3], "hx")
    return got


def act_arrays(xs):
    """``len(xs)`` tokens' quantized activations, in the (nt, H) / (nt, MAXC) arrays the row kernel's binding takes.
    Planting ``scale`` and ``hx`` by hand is what makes the adversarial cases of B7 reachable."""
    q, scale, hx = np.zeros((len(xs), H), np.int8), np.zeros((len(xs), MAXC), F32), np.zeros((len(xs), MAXC), F32)
    for t, x in enumerate(xs):
        qt, st, _, ht, nchunks = kernel_act(x)
        q[t, : qt.size], scale[t, :nchunks], hx[t, :nchunks] = qt, st, ht
    return q, scale, hx


def prefilled(ntok, nrows, fill=0xFF):
    """An output array pre-filled with ``fill`` bytes, as basalt's tests pre-fill theirs (basalt_gpu.py:21-28)."""
    return np.full((ntok, nrows, 4), fill, np.uint8).view(F32).reshape(ntok, nrows)


def kernel_rows(w, row_bytes, nblocks, act, nrows, r0, r1, fill=0xFF):
    """``q2_0_gguf_rows_multi`` through the binding, on a caller-owned pre-filled ``out``."""
    q, scale, hx = act
    out = prefilled(q.shape[0], nrows, fill)
    _nt.basalt_q2_0_gguf_rows_multi(np.ascontiguousarray(w, np.uint8), row_bytes, nblocks,
                                    np.ascontiguousarray(q, np.int8), np.ascontiguousarray(scale, F32),
                                    np.ascontiguousarray(hx, F32), out, r0, r1)
    return out


def check_rows(w, row_bytes, nblocks, act, nrows, r0=0, r1=None, what=""):
    """kernel == emulator over the whole output, the pre-filled rows included. Returns the kernel's ``out``."""
    r1 = nrows if r1 is None else r1
    got = kernel_rows(w, row_bytes, nblocks, act, nrows, r0, r1)
    want = emu.q2_0_gguf_rows_multi(w, row_bytes, nblocks, list(zip(*act)), prefilled(act[0].shape[0], nrows), r0, r1)
    same_f32(got, want, what or f"rows({row_bytes}, {nblocks}, {r0}:{r1})")
    return got


# ---- A. act_quant_q8_1 -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2560, 640])
def test_a1_normal_activations(n):
    """A1. N(0, 1) at the two widths the callers pass: the FFN input (2560) and the SwiGLU vector (640)."""
    x = np.random.default_rng(7).standard_normal(n).astype(F32)
    q, scale, total, hx, nchunks = check_act(x)
    assert nchunks == n // QKA and np.isfinite(scale).all() and np.abs(q).max() == 127


def test_a2_all_zero_chunk():
    """A2. An all-zero chunk: ``amax > 0`` is false, so s = 0, inv = 0, every code 0, sum 0 and hx exactly +0.0
    (the sign bit matters: a -0.0 would be a different float32)."""
    q, scale, total, hx, _ = check_act(np.zeros(QKA, F32))
    assert np.array_equal(q, np.zeros(QKA, np.int8)) and total[0] == 0
    assert scale.view(np.uint32)[0] == 0 and hx.view(np.uint32)[0] == 0, "scale and hx must be +0.0, not -0.0"


def test_a3_signed_zeros_and_every_amax_position():
    """A3. +-0 mixes (the sign clear of step 2 leaves +0.0, so s = 0 and every code 0), then the maximum planted in
    each of the 32 positions in turn: the fixed ``maxps(|x0|, |x1|)`` pairing of expert.cpp:352 feeds element j and
    element 16+j to one lane, and the four-level tree must reach every lane. With no NaN present both are
    order-independent (maximum is associative and commutative there), which is why A6 is the case that pins the
    operand order."""
    z = np.where(np.arange(QKA) % 3 == 0, F32(-0.0), F32(0.0)).astype(F32)
    q, scale, total, hx, _ = check_act(z)
    assert np.array_equal(q, np.zeros(QKA, np.int8)) and total[0] == 0
    assert scale.view(np.uint32)[0] == 0 and hx.view(np.uint32)[0] == 0

    base = (F32(0.001) * np.arange(1, QKA + 1)).astype(F32)          # distinct, so the maximum is unique
    for p in range(QKA):
        for sign in (F32(1.0), F32(-1.0)):
            x = base.copy()
            x[p] = sign * F32(9.5)
            _, scale, _, _, _ = check_act(x)
            assert scale[0] == F32(9.5) / F32(127.0), (p, sign, scale[0])


def test_a4_the_scale_band():
    """A4. The four regimes of ``s = amax / 127`` and ``inv = 1 / s``, at the boundaries the plan measured:

    (a) ``amax = 127 * 2^-128`` puts s exactly on 2^-128, where ``1.f / s`` is already +inf (the specification's
        "s < 2^-128" at 01-cpu-kernels.md:155 is off by this one value; basalt's plain ``1.f / s`` governs). Then
        ``0 * inf = NaN`` and ``x * inf = +-inf``, every one of which VCVTTPS2DQ turns into the integer indefinite:
        **every code -127, the zeros included**, sum -4064 and hx = s * (float) -4064.
    (b) one ulp of s above that, ``inv`` is the finite 3.402822e38 and the codes are ordinary - and subnormal
        arithmetic is live, so this case also fails loudly if FTZ is ever switched on.
    (c) a subnormal s well inside the finite-``inv`` band (``amax = 127 * 2^-127``).
    (d) ``amax = 2^-149`` makes ``fl(amax / 127) = 0``, a regime the specification does not name: the ``s > 0.f``
        guard gives inv = 0, so every code is 0 with scale +0.0 - while ``amax = 127 * 2^-149`` keeps s nonzero
        (2^-149) and therefore lands back in (a)'s band, every code -127."""
    x = np.zeros(QKA, F32)
    x[5], x[9], x[17] = F32(127.0) * TWO_M128, -F32(127.0) * TWO_M128 / F32(2.0), TWO_M128
    q, scale, total, hx, _ = check_act(x)                            # (a)
    assert scale[0] == TWO_M128 and recip(scale[0]) == F32(np.inf)
    assert np.array_equal(q, np.full(QKA, -127, np.int8)), "every code, zeros included, must be -127"
    assert total[0] == -4064 and hx[0] == (scale[0] * F32(-4064.0)).astype(F32)

    s_b = np.nextafter(TWO_M128, F32(1.0))                           # (b)
    x = np.zeros(QKA, F32)
    x[3], x[21] = F32(s_b * F32(127.0)), -F32(s_b * F32(127.0)) / F32(2.0)
    q, scale, _, _, _ = check_act(x)
    assert scale[0] == s_b and recip(scale[0]) == F32(3.402822e38) and np.isfinite(recip(scale[0]))
    assert q[3] == 127 and q[21] == -64 and q[0] == 0, "an ordinary chunk, zeros staying 0"

    x = np.zeros(QKA, F32)                                           # (c)
    x[11], x[30] = F32(127.0) * (F32(2.0) ** -127), -F32(2.0) ** -127
    q, scale, _, _, _ = check_act(x)
    assert scale[0] == F32(2.0) ** -127 and (scale.view(np.uint32)[0] & 0x7F800000) == 0, "s must be subnormal"
    assert np.isfinite(recip(scale[0])) and q[11] == 127 and q[30] == -1

    x = np.zeros(QKA, F32)                                           # (d)
    x[1] = TWO_M149
    q, scale, total, hx, _ = check_act(x)
    assert scale.view(np.uint32)[0] == 0 and np.array_equal(q, np.zeros(QKA, np.int8))
    assert total[0] == 0 and hx.view(np.uint32)[0] == 0

    x = np.zeros(QKA, F32)                                           # (d), the neighbour that is not this regime
    x[1] = F32(127.0) * TWO_M149
    q, scale, total, _, _ = check_act(x)
    assert scale[0] == TWO_M149 and np.array_equal(q, np.full(QKA, -127, np.int8)) and total[0] == -4064


def test_a5_infinite_elements():
    """A5. An infinite element makes ``amax = inf``, so s = inf and inv = 0: the infinite lanes compute
    ``inf * 0 = NaN`` and clamp to -127, while every finite lane gets t = +-0 and code 0 - including the -0.0
    element, whose ``t = -0.0`` still takes the ordered ``t >= 0`` branch and adds +0.5."""
    x = np.zeros(QKA, F32)
    x[0], x[4], x[6], x[8] = F32(np.inf), F32(-np.inf), F32(-0.0), F32(1.0)
    q, scale, total, _, _ = check_act(x)
    assert scale[0] == F32(np.inf) and q[0] == -127 and q[4] == -127
    assert q[6] == 0 and q[8] == 0 and total[0] == -254


def test_a6_nan_sweep_over_both_reduction_orders():
    """A6. A NaN swept over all 32 positions of a ramp peaking at element 31, checked bit for bit against the
    emulator in **GCC's** operand order, with Clang's value printed beside it.

    The divergence is not only NaN-ness. MAXPS returns SRC2 on an unordered compare, so the two orders swallow
    different lanes: at position 19 GCC's tree ends on the NaN (and ``amax > 0.f`` is then false, so s = 0 and every
    finite element's code is 0) where Clang's keeps 0.32; at 31 it is the other way round; at 23 both are finite and
    different (0.31 against 0.32). A flat chunk with the NaN at 16 and the true maximum at 4 shows the same thing
    without any NaN surviving: GCC's tree swallows the 9.0, Clang's keeps it. A test that planted a NaN "somewhere
    in the upper half" would pass under either order and prove nothing."""
    ramp = (F32(0.01) * np.arange(1, QKA + 1)).astype(F32)           # 0.01 .. 0.32, the maximum at element 31
    table = []
    for p in range(QKA):
        x = ramp.copy()
        x[p] = F32(np.nan)
        check_act(x, "gcc")                                          # the port's order, bit for bit
        g, c = (emu.act_quant_q8_1(x, o)[1][0] for o in ("gcc", "clang"))
        table.append((p, float(g), float(c)))
    print("\nA6 scale by NaN position (GCC's order is the port's):")
    print("   " + ", ".join(f"{p}: {g:.8g}/{c:.8g}" for p, g, c in table if g != c))

    s = {p: (g, c) for p, g, c in table}
    assert s[19] == (0.0, float(ramp[31] / F32(127.0))), "19: GCC ends on the NaN (s = 0), Clang keeps 0.32"
    assert s[31] == (float(ramp[30] / F32(127.0)), 0.0), "31: the other way round"
    assert s[23] == (float(ramp[30] / F32(127.0)), float(ramp[31] / F32(127.0))), "23: two different finite maxima"

    flat = np.full(QKA, F32(0.1), F32)
    flat[4], flat[16] = F32(9.0), F32(np.nan)
    check_act(flat, "gcc")
    g, c = (emu.act_quant_q8_1(flat, o)[1][0] for o in ("gcc", "clang"))
    print(f"   flat chunk, NaN at 16, maximum 9.0 at 4: GCC {float(g):.8g}, Clang {float(c):.8g}")
    assert g == F32(0.1) / F32(127.0) and c == F32(9.0) / F32(127.0), "GCC swallows the maximum, Clang keeps it"


def test_a7_rounding_half_away_from_zero_and_the_tie():
    """A7. ``r = t + copysign(0.5, t)`` truncated is lround's rule, round half AWAY from zero: with amax = 127 the
    scale is exactly 1 and every t is exact, so +-2.5 must give +-3. And the second rounding is part of the
    contract: ``t = 0.49999997f`` makes the exact sum ``1 - 2^-25``, a tie that rounds to the even 1.0f, so the code
    is 1 where ``lround(t)`` would give 0 (01-cpu-kernels.md:158)."""
    x = np.zeros(QKA, F32)
    x[0], x[1], x[2], x[3], x[4] = F32(127.0), F32(2.5), F32(-2.5), F32(0.5), F32(-0.5)
    q, scale, _, _, _ = check_act(x)
    assert scale[0] == F32(1.0)
    assert (q[1], q[2], q[3], q[4]) == (3, -3, 1, -1), "half away from zero, both signs"

    amax, xt = F32(np.e), F32(0.01070189569145441)                   # t = 0.49999997f exactly (plan section 7, A7)
    x = np.zeros(QKA, F32)
    x[0], x[7], x[9] = amax, xt, -xt
    q, scale, _, _, _ = check_act(x)
    inv = recip(scale[0])
    assert F32(xt * inv) == F32(0.49999997), f"t is {F32(xt * inv):.9g}, not the 0.49999997f this case needs"
    assert F32(F32(xt * inv) + F32(0.5)) == F32(1.0) and np.round(F32(xt * inv)) == 0
    assert (q[7], q[9]) == (1, -1), "the second rounding carries t over the integer; lround would give 0"


def test_a8_the_saturation_trap():
    """A8. The clamp's reachable side. VCVTTPS2DQ returns the integer indefinite 0x80000000 for a NaN, for +-inf and
    for an out-of-range value, which the clamp turns into **-127**; FCVTZS instead saturates (NaN -> 0,
    +inf -> INT32_MAX), which would give 0 and +127 here. Both NaN and +inf are reachable: a chunk in the band of
    A4(a) has inv = +inf, so its zeros compute ``0 * inf = NaN`` and its nonzeros ``+-inf``.

    The clamp's high side is not reachable at all: ``|x| <= amax`` and the rounding is monotone, so r is largest at
    x = amax, and over every positive finite float32 amax with a finite inv ``trunc(r)`` never exceeds 127 (the
    largest is 127.50003051757812, at amax = 3.7322113794423347e-37). This asserts that at the measured worst case
    and over a random sweep, so the +127 arm is shown inert rather than tested. A finite ``r >= 2^31`` - the third
    FCVTZS saturation case - is unreachable through this kernel for the same reason, and is checked on the
    reference's own VCVTTPS2DQ instead."""
    x = np.zeros(QKA, F32)
    x[2] = F32(127.0) * TWO_M128                                     # inv = +inf: zeros -> NaN, x[2] -> +inf
    x[3] = -x[2]                                                     # -> -inf
    q, _, _, _, _ = check_act(x)
    assert q[0] == -127 and q[2] == -127 and q[3] == -127, "NaN, +inf and -inf must all clamp to -127"

    def pre_clamp(chunk):
        """The chunk's steps 5-10 without basalt's clamp, in the reference's own arithmetic."""
        a = np.abs(np.asarray(chunk, F32))
        amax = emu.reduce_max_ps(emu.maxps(a[None, :16], a[None, 16:]))
        s = np.where(amax > 0, (amax / F32(127.0)).astype(F32), F32(0)).astype(F32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            inv = np.where(s > 0, (F32(1.0) / s).astype(F32), F32(0)).astype(F32)
            t = (np.asarray(chunk, F32) * inv).astype(F32)
            r = (t + np.where(t >= 0, F32(0.5), F32(-0.5)).astype(F32)).astype(F32)
        return r, emu.cvttps_epi32(r)

    rng = np.random.default_rng(3)
    worst, worst_amax = -np.inf, None
    sweep = [F32(3.7322113794423347e-37)] + [F32(v) for v in rng.standard_normal(64) * 10 ** rng.uniform(-30, 30, 64)]
    for amax in sweep:
        if not (amax > 0) or not np.isfinite(recip(F32(np.abs(amax) / F32(127.0)))):
            continue
        amax = F32(np.abs(amax))
        chunk = (amax * np.linspace(-1.0, 1.0, QKA)).astype(F32)
        chunk[0], chunk[QKA - 1] = amax, -amax
        r, v = pre_clamp(chunk)
        assert v.max() <= 127, f"the +127 clamp fired at amax = {float(amax)!r}: trunc(r) = {v.max()}"
        check_act(chunk)
        if r.max() > worst:
            worst, worst_amax = float(r.max()), float(amax)
    print(f"\nA8 largest pre-clamp r over the sweep: {worst!r} at amax = {worst_amax!r} (the clamp's high side is "
          f"inert)")
    assert worst == 127.50003051757812, "the tight-band worst case must be in the sweep"

    big = np.asarray([2.0 ** 31, -(2.0 ** 31) - 2.0 ** 8, np.inf, -np.inf, np.nan], F32)   # not reachable through
    assert np.array_equal(emu.cvttps_epi32(big), np.full(5, emu.INDEF, np.int64))          # the kernel; see above
    assert np.array_equal(np.clip(emu.cvttps_epi32(big), -127, 127), np.full(5, -127, np.int64))


def test_a9_the_contraction_trap():
    """A9. Step 7's ``_mm512_mul_ps`` and step 9's ``_mm512_add_ps`` round twice, which GCC 12.2/13.2/14.3/15.2 and
    Clang 21 all emit as a separate vmulps and vaddps. A contracted ``fma(x, inv, +-0.5)`` skips the rounding of t
    and changes the code, so this input fails the moment the translation unit loses ``-ffp-contract=off`` or the two
    statements are merged into one expression."""
    amax, xt = F32(np.e), F32(0.01070189569145441)
    x = np.zeros(QKA, F32)
    x[0], x[6], x[11] = amax, xt, -xt
    q, scale, _, _, _ = check_act(x)
    inv = recip(scale[0])
    two = np.asarray([F32(F32(xt * inv) + F32(0.5)), F32(F32(-xt * inv) + F32(-0.5))], F32)
    fused = np.asarray([fma32(xt, inv, F32(0.5)), fma32(-xt, inv, F32(-0.5))], F32)
    assert not np.array_equal(np.trunc(two.astype(np.float64)), np.trunc(fused.astype(np.float64))), \
        f"this input no longer separates the two roundings from the FMA: {two} against {fused}"
    assert (q[6], q[11]) == (1, -1), f"contracted codes would be {np.trunc(fused.astype(np.float64)).astype(int)}"


# ---- B. q2_0_gguf_rows_multi -------------------------------------------------------------------------------------

SHAPES = [(2560, 40, 720), (640, 10, 180)]                           # (n, nblocks, row_bytes): gate/up, then down


@pytest.mark.parametrize("n, nblocks, row_bytes", SHAPES)
def test_b1_both_pack_shapes(n, nblocks, row_bytes):
    """B1. The two shapes this pack has (native_expert.cpp:49-53 at 2560/640): 40 blocks in 5 full groups of 8, and
    10 blocks in one full group and a ragged one whose mask keeps only ``corr`` lanes 0-3."""
    rng = np.random.default_rng(101)
    nrows = 17
    w = random_rows(rng, nrows, n)
    act = act_arrays([rng.standard_normal(n).astype(F32)])
    out = check_rows(w, row_bytes, nblocks, act, nrows, what=f"B1 {n}")
    assert np.isfinite(out).all()


def test_b2_every_token_equals_its_single_token_run():
    """B2. The kernel's own promise (pool.hpp:54-55, "Every token's output is bitwise the single-token job's"),
    in the shape of expert_multi_test.cpp:56-72: nt = 1..8 against each token's own nt = 1 run, bit for bit."""
    rng = np.random.default_rng(23)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 13
    w = random_rows(rng, nrows, n)
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32) * (1 + t) for t in range(MAXT)])
    single = [kernel_rows(w, row_bytes, nblocks, (q[t: t + 1], scale[t: t + 1], hx[t: t + 1]), nrows, 0, nrows)[0]
              for t in range(MAXT)]
    for ntok in range(1, MAXT + 1):
        got = check_rows(w, row_bytes, nblocks, (q[:ntok], scale[:ntok], hx[:ntok]), nrows, what=f"B2 nt={ntok}")
        for t in range(ntok):
            assert np.array_equal(got[t].view(np.uint32), single[t].view(np.uint32)), (ntok, t)


def test_b3_row_ranges_and_the_absolute_index():
    """B3. ``out[t]`` is indexed by the ABSOLUTE row r, and rows outside [r0, r1) are not touched - which is what
    lets the pool hand row slices of one buffer to different workers (expert.hpp:123-126, pool.cpp:418-431). The
    output is pre-filled with 0xFF bytes and checked raw, so an untouched row is 0xFFFFFFFF, not merely a NaN."""
    rng = np.random.default_rng(55)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 11
    w = random_rows(rng, nrows, n)
    act = act_arrays([rng.standard_normal(n).astype(F32), rng.standard_normal(n).astype(F32)])
    full = check_rows(w, row_bytes, nblocks, act, nrows, what="B3 full")
    for r0, r1 in ((0, nrows), (3, 7), (nrows - 1, nrows), (5, 5)):
        got = check_rows(w, row_bytes, nblocks, act, nrows, r0, r1, what=f"B3 {r0}:{r1}")
        inside = np.zeros(nrows, bool)
        inside[r0:r1] = True
        assert np.all(got.view(np.uint32)[:, ~inside] == 0xFFFFFFFF), f"{r0}:{r1} wrote outside its range"
        assert np.array_equal(got.view(np.uint32)[:, inside], full.view(np.uint32)[:, inside]), \
            f"{r0}:{r1} did not write out[t][r] at the absolute r"


def test_b4_stride_padding_is_never_read():
    """B4. ``row_bytes`` is a stride, not a length: with 64 bytes of garbage after each row's 10 blocks the result
    must equal the same blocks packed at ``row_bytes = 18 * nblocks``."""
    rng = np.random.default_rng(77)
    n, nblocks, nrows = 640, 10, 9
    tight = random_rows(rng, nrows, n).reshape(nrows, BB * nblocks)
    pad = 64
    padded = np.concatenate([tight, rng.integers(0, 256, (nrows, pad), dtype=np.uint8)], axis=1)
    act = act_arrays([rng.standard_normal(n).astype(F32)])
    a = check_rows(tight.reshape(-1), BB * nblocks, nblocks, act, nrows, what="B4 tight")
    b = check_rows(padded.reshape(-1), BB * nblocks + pad, nblocks, act, nrows, what="B4 padded")
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), "the stride padding changed the result"


def test_b5_the_unread_tail():
    """B5. The read set of section 3: ``q[0, 64 * nblocks)``, ``scale[0, 2 * nblocks)`` and ``hx[0, 2 * nblocks)``,
    and nothing else - never ``sum``, never ``nchunks``, and nothing past the width (``_mm512_maskz_loadu_ps`` does
    not touch its masked-off elements). At nblocks = 10 the tail is poisoned with NaNs in scale and hx and with
    alternating +-127 in q; the result must not move a bit."""
    rng = np.random.default_rng(91)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 7
    w = random_rows(rng, nrows, n)
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32)])
    base = check_rows(w, row_bytes, nblocks, (q, scale, hx), nrows, what="B5 clean")
    q[0, QK * nblocks:] = np.where(np.arange(H - QK * nblocks) % 2, 127, -127)
    scale[0, 2 * nblocks:] = F32(np.nan)
    hx[0, 2 * nblocks:] = F32(np.nan)
    poisoned = check_rows(w, row_bytes, nblocks, (q, scale, hx), nrows, what="B5 poisoned")
    assert np.array_equal(base.view(np.uint32), poisoned.view(np.uint32)), "the kernel read past its width"


def test_b6_every_fp16_block_scale():
    """B6. All 65,536 half bit patterns as a block's ``d`` against one fixed activation: +-0, every subnormal, 65504,
    +-inf and every NaN. ``_mm256_cvtph_ps`` and ``vcvt_f32_f16`` are both exact for every half, subnormals included
    (MXCSR.DAZ is ignored, and FPCR's FZ16 is off); NaN payloads may differ between the two, which is why the
    comparison is NaN-ness where either side is a NaN."""
    rng = np.random.default_rng(13)
    nrows, nblocks, row_bytes = 1 << 16, 1, BB
    w = np.empty((nrows, row_bytes), np.uint8)
    w[:, :2] = np.arange(nrows, dtype=np.uint16).view(np.uint8).reshape(-1, 2)
    w[:, 2:] = rng.integers(0, 256, (nrows, 16), dtype=np.uint8)
    act = act_arrays([rng.standard_normal(QK).astype(F32)])
    out = check_rows(w.reshape(-1), row_bytes, nblocks, act, nrows, what="B6 halves")
    halves = np.frombuffer(np.arange(nrows, dtype=np.uint16).tobytes(), np.float16)
    assert np.isnan(out[0][np.isnan(halves)]).all(), "a NaN scale must give a NaN row"
    assert np.isfinite(out[0][: 0x7C00]).all(), "a finite scale must give a finite row"
    assert np.isnan(out[0][[0x7C00, 0xFC00]]).all(), "an infinite scale reaches inf * 0 = NaN on a zero code"
    assert out[0][[0, 0x8000]].view(np.uint32).tolist() == [0, 0], "+-0 as a scale leaves res at +0.0"


def test_b7_adversarial_activations():
    """B7. ``scale`` and ``hx`` are planted directly, which the binding allows and the quantizer cannot produce:
    subnormal, zero and huge values; a row whose ``P = RN(d * s)`` overflows to +-inf (and whose lanes then compute
    ``inf * 0 = NaN``); blocks carrying all four codes 0..3; and nblocks = 9, a ragged second group where the mask
    keeps only ``corr`` lanes 0-1.

    The cancelling case is the sharpest of them: with every code byte 0x55 every code is 1, so every weight
    dequantizes to ``(1 - 1) * d = 0`` and the exact result is zero - ``R(acc)`` and ``R(corr)`` are then the same
    sum reached through different lane structures (acc's lane L carries weights 4L..4L+3, corr's carries chunk
    16g+L), and what is left is pure rounding residue, where the tree shape and the FMA order are visible."""
    rng = np.random.default_rng(5)
    n, nblocks, row_bytes, nrows = 640, 10, 180, 12
    w = random_rows(rng, nrows, n).reshape(nrows, row_bytes)
    for b in range(nblocks):                                         # all four codes 0..3 in every block
        w[0, b * BB + 2: (b + 1) * BB] = np.tile(np.asarray([0x1B, 0xE4, 0x4E, 0xB1], np.uint8), 4)
        w[1, b * BB + 2: (b + 1) * BB] = 0x1B
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32)])
    scale[0, 0], hx[0, 0] = TWO_M149, -TWO_M149                      # subnormal
    scale[0, 1], hx[0, 1] = F32(0.0), F32(-0.0)                      # zero, and a -0.0 hx
    scale[0, 2], hx[0, 2] = F32(3.0e38), F32(-3.0e38)                # P = d * s overflows for any sane d
    scale[0, 19], hx[0, 19] = F32(np.inf), F32(1.0)                  # the ragged group's last kept lane
    out = check_rows(w.reshape(-1), row_bytes, nblocks, (q, scale, hx), nrows, what="B7 planted")
    assert np.isnan(out).any() or np.isinf(out).any(), "the overflow regime must be reached"

    for nb in (9, 2, 1):                                             # ragged groups: 1, 2 and 8+1 blocks
        rb = BB * nb
        ww = random_rows(rng, nrows, QK * nb)
        check_rows(ww, rb, nb, act_arrays([rng.standard_normal(QK * nb).astype(F32)]), nrows, what=f"B7 nb={nb}")

    nblocks, row_bytes, n = 40, 720, 2560                            # the cancelling row set
    w = random_rows(rng, nrows, n).reshape(nrows, row_bytes)
    for b in range(nblocks):
        w[:, b * BB + 2: (b + 1) * BB] = 0x55                        # every code 1: (code - 1) * d = 0
    act = act_arrays([rng.standard_normal(n).astype(F32)])
    out = check_rows(w.reshape(-1), row_bytes, nblocks, act, nrows, what="B7 cancellation")
    print(f"\nB7 R(acc) - R(corr) where the exact sum is zero: max |res| = {np.abs(out).max():.3e}")
    assert np.abs(out).max() < 1e-5, "the correction no longer cancels the dot; check the lane mapping"


@pytest.mark.parametrize("n, nblocks, row_bytes", SHAPES)
def test_b8_against_a_float64_reference(n, nblocks, row_bytes):
    """B8. The independent check. The emulation and the kernel share one reading of section 3's unpack and lane
    mapping; this builds ``sum (code - 1) * d * q * s`` in float64 from llama.cpp's own numpy dequantizer
    (tests/contract/gguf_reference.dequantize, written apart from the C reference the format plugins follow) and
    compares against it, which is the only case that catches a misreading of either.

    The bound is on ``sum |terms|``, not on ``|ref|``: a row whose terms cancel has a large relative error for a
    correct kernel. A misread unpack or lane mapping is an O(1) error, so 1e-6 of the row's own magnitude - about
    twenty times the worst observed - still catches it."""
    rng = np.random.default_rng(42)
    nrows = 48
    values = (rng.standard_normal((nrows, n)) * 0.02 * (0.5 + (np.arange(nrows)[:, None] % 7) / 7.0)).astype(F32)
    w = quantize_q2_0(values)                                        # real weights, not random bytes
    q, scale, hx = act_arrays([rng.standard_normal(n).astype(F32)])
    got = check_rows(w, row_bytes, nblocks, (q, scale, hx), nrows, what=f"B8 {n}")[0].astype(np.float64)

    weights = gguf_reference.dequantize(42, w.reshape(nrows, row_bytes)).astype(np.float64)
    x = (q[0, :n].astype(np.float64).reshape(-1, QKA) * scale[0, : n // QKA, None].astype(np.float64)).reshape(-1)
    ref = weights @ x
    magnitude = np.abs(weights) @ np.abs(x)
    rel = np.abs(got - ref) / magnitude
    aggregate = np.abs(got - ref).sum() / np.abs(ref).sum()
    print(f"\nB8 n={n}: max |res - ref| / sum|terms| = {rel.max():.2e}, aggregate L1 = {aggregate:.2e}")
    assert rel.max() < 1e-6 and aggregate < 1e-6, (rel.max(), aggregate)


def test_b9_determinism():
    """B9. Repeated runs bit-identical, the numerics contract's last gate (CLAUDE.md)."""
    rng = np.random.default_rng(2026)
    n, nblocks, row_bytes, nrows = 2560, 40, 720, 9
    w = random_rows(rng, nrows, n)
    act = act_arrays([rng.standard_normal(n).astype(F32) for _ in range(3)])
    first = kernel_rows(w, row_bytes, nblocks, act, nrows, 0, nrows)
    for _ in range(4):
        again = kernel_rows(w, row_bytes, nblocks, act, nrows, 0, nrows)
        assert np.array_equal(first.view(np.uint32), again.view(np.uint32)), "the row dot is not deterministic"
    x = rng.standard_normal(n).astype(F32)
    one = kernel_act(x)
    for _ in range(4):
        other = kernel_act(x)
        assert np.array_equal(one[0], other[0]) and np.array_equal(one[2], other[2])
        assert np.array_equal(one[1].view(np.uint32), other[1].view(np.uint32))
        assert np.array_equal(one[3].view(np.uint32), other[3].view(np.uint32))
