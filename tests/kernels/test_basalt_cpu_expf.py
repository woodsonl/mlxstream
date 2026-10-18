# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/pool.cpp:434 @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE): the CPU expert pool's SwiGLU and its order of operations.
"""``expf_glibc`` and the SwiGLU of ``runtime/src/basalt_cpu_expf.cpp``, the one piece of arithmetic basalt's CPU
expert pool performs itself.

**What is pinned and what is only bounded.** The function must return glibc's bits, because basalt runs on Linux
with glibc and a token's output has to be the same number whichever tier produced it. Six glibc values have been
recorded so far, on a Xeon Platinum 8375C running glibc 2.39
(``.context/basalt-port/expf-ground-truth/raw/probe_run1.txt``); group A asserts those six, bit for bit, and they
are the only inputs at which this suite knows glibc's answer. The fuller corpus is still being produced. Everywhere
else the suite does what can be done without it: group B holds the transcription to the 0.502 ulp Arm records for
this algorithm, against a float64 reference, over the SwiGLU's reachable range; group C pins the specials and the
three thresholds against that same reference; group D pins the SwiGLU's order of operations. A transcription that
passed B, C and D could still differ from glibc somewhere - only the corpus can close that - but it could not
differ by much, and not in the shape of the usual transcription mistake.

The float64 reference is ``np.exp`` in double, which carries 29 bits more than the float32 result and is itself
accurate to well under 2^-29 of a float32 ulp. Its own error is therefore invisible at the 10^-3 ulp the group B
assertions turn on. Where a case asserts the *correctly rounded* value it is written ``np.float32(np.exp(x64))``:
one rounding, from the double to the float.

Cases. **A**, the recorded glibc bits: A1 the six values. **B**, the error bound: B1 a stratified sample of
[-40, 40] - the SwiGLU's reachable range, with most of its mass near zero - B2 the measured worst case and the
window around it, and B3 the two arguments a contracted multiply-add would move. **C**, the specials: C1 both
zeros, C2 both infinities, C3 NaN (quiet and signalling), C4 the overflow threshold and the largest argument with
a finite result, C5 the two underflow thresholds and the ``may_uflow`` band, C6 the subnormal results. **D**, the
SwiGLU: D1 its order against basalt's expression, D2 that the order is load-bearing, D3 its specials.

The suite bites, measured by mutating the kernel and rebuilding: ``poly`` in place of ``poly_scaled`` fails A1, B1,
B2, B3, C4 and C6; the platform's ``expf`` fails B1, B2, B3 and C6 - it exceeds the 0.502 ulp this algorithm holds
to, and differs from the transcription at 1,930,130 of the 2,218,786,818 float32 arguments in [-40, 40] (measured
here), though not at any of the recorded six, so A1 alone would miss it; ``std::fma`` at ``z - kd`` and
``z + SHIFT`` fails B3 and nothing else, which is what B3 is for; and the SwiGLU as ``g * (1 / (1 + e)) * u``
fails D1.

Two mutations fail nothing, and both are findings rather than gaps. ``round``/``lround`` for the rounding - Arm's
other branch, the one glibc compiles on aarch64 - passes, because no float32 argument makes ``InvLn2N * x`` an
exact tie (swept here over all 2^32), so the two branches are the same function. Dropping the
``WANT_ERRNO_UFLOW`` band passes, because the main path computes 0x00000001 across the whole band anyway (C5
asserts exactly that); the band is transcribed because glibc has it, and ``errno``, which nothing here reads, is
the only thing it changes.

Run with ``-s`` for the measured worst ulp and its argument.
"""

import numpy as np
import pytest

from monolith.runtime import _native as _nt

F32, U32 = np.float32, np.uint32

# probe_run1.txt:"parent expf(...)" and "child expf(...)": glibc 2.39, Xeon Platinum 8375C, with the ifunc dispatch
# intact and again with GLIBC_TUNABLES disabling AVX2 and FMA. The two agree on these six, which is why no variant
# is named; the sweep that would extend that agreement ran in the parent alone, so it holds for these six only.
GLIBC_SIX = (
    (0x3F000000, 0x3FD3094C),      # 0.5
    (0xBF800000, 0x3EBC5AB2),      # -1
    (0x3A83126F, 0x3F8020C9),      # 0.001
    (0xC2AF0000, 0x006CB2BC),      # -87.5, a subnormal result
    (0x42B10000, 0x7F4CDCC4),      # 88.5, near overflow
    (0xC2C80000, 0x0000001B),      # -100, a tiny subnormal
)

# the thresholds expf.c branches on, as float32 bit patterns (runtime/src/basalt_cpu_expf.cpp)
OFLOW = 0x42B17217       # 0x1.62e42ep6f, log(0x1p128); x above it takes the overflow return
UFLOW = 0xC2CFF1B4       # -0x1.9fe368p6f, log(0x1p-150); x below it takes the underflow return
MAY_UFLOW = 0xC2CE8ECF   # -0x1.9d1d9ep6f, log(0x1p-149); x below it returns (0x1.4p-75f)^2


def f32(bits):
    """The float32 array of these bit patterns. Bits in, bits out: nothing converts through a double, so a
    signalling NaN reaches the kernel as one."""
    return np.asarray(bits, dtype=U32).view(F32)


def bits_of(x):
    return np.asarray(x, dtype=F32).view(U32)


def expf(x):
    out = np.empty_like(np.asarray(x, dtype=F32))
    _nt.basalt_expf(np.asarray(x, dtype=F32), out)
    return out


def swiglu(g, u):
    g, u = np.asarray(g, dtype=F32), np.asarray(u, dtype=F32)
    out = np.empty_like(g)
    _nt.basalt_swiglu(g, u, out)
    return out


def ulp_error(x):
    """|expf(x) - exp(x)| in units of the float32 ulp at the exact value, which is 2^-149 below 2^-126."""
    ref = np.exp(np.asarray(x, dtype=F32).astype(np.float64))
    exponent = np.maximum(np.floor(np.log2(np.abs(ref))), -126.0)
    ulp = np.ldexp(1.0, (exponent - 23).astype(np.int64))
    return np.abs(expf(x).astype(np.float64) - ref) / ulp


def correctly_rounded(x):
    """exp(x) in double, rounded once to float32 - the value an infinitely precise expf would return. The cast is
    allowed to overflow to +inf and to flush to zero: those are the right answers at the two ends."""
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(np.asarray(x, dtype=F32).astype(np.float64)).astype(F32)


# ---- A: the recorded glibc bits ----------------------------------------------------------------------------------

def test_a1_recorded_glibc_values():
    """A1. The six values recorded from glibc 2.39, bit for bit. These are the only inputs at which this suite
    knows glibc's answer; everything below bounds the error instead of pinning the bits."""
    arg, want = zip(*GLIBC_SIX)
    got = bits_of(expf(f32(arg)))
    for a, w, g in zip(arg, want, got):
        assert g == w, f"expf({a:#010x}) = {g:#010x}, glibc 2.39 gives {w:#010x} (probe_run1.txt)"


# ---- B: the error bound against a float64 reference --------------------------------------------------------------

def test_b1_ulp_bound_over_the_reachable_range(capsys):
    """B1. Arm records this algorithm's worst case as 0.502 ulp under round to nearest, and the transcription has
    to stay inside that. The sample is the SwiGLU's reachable range: uniform over [-40, 40], a normal sample with
    most of its mass near zero, the exact zero neighbourhood, and the whole underflow region, where the result is
    subnormal and the ulp is 2^-149."""
    rng = np.random.default_rng(20261018)
    sample = np.concatenate([
        rng.uniform(-40.0, 40.0, 1 << 20).astype(F32),                        # the range, flat
        rng.normal(0.0, 2.0, 1 << 20).astype(F32),                            # where the mass is
        rng.normal(0.0, 0.01, 1 << 16).astype(F32),                           # and nearer still
        f32(np.arange(0xC2AE0000, 0xC2D10000, dtype=U32)),                    # -87 .. -104.5, subnormal results
        f32(np.arange(0x42B00000, OFLOW + 1, dtype=U32)),                     # 88 .. 88.72, up to the threshold
    ])
    err = ulp_error(sample)
    worst = int(np.argmax(err))
    with capsys.disabled():
        print(f"\n  B1: {sample.size} arguments, worst {err[worst]:.6f} ulp at "
              f"{bits_of(sample)[worst]:#010x} ({float(sample[worst])!r})")
    assert err.max() <= 0.502, f"worst {err.max():.6f} ulp at {bits_of(sample)[worst]:#010x}"


def test_b2_the_measured_worst_case():
    """B2. The worst argument of an exhaustive sweep of every float32 whose result is finite and nonzero (measured
    here, 4,294,967,296 inputs): 0.501637 ulp, above a half and so not the correctly rounded value, which is what
    "not correctly rounded" means for this algorithm. The window around it is swept here so a transcription that
    drifts has to beat the bound somewhere in the neighbourhood too."""
    x = f32([0xBBE7328F])
    assert bits_of(expf(x))[0] == 0x3F7E333C
    assert bits_of(correctly_rounded(x))[0] == 0x3F7E333B, "the correctly rounded value is one ulp below"
    assert 0.5 < ulp_error(x)[0] <= 0.502
    window = f32(np.arange(0xBBE70000, 0xBBE80000, dtype=U32))
    assert ulp_error(window).max() <= 0.502


def test_b3_the_two_arguments_contraction_would_move():
    """B3. The build flag's witness. Contracting the two mul-adds around the rounding - `z + SHIFT` and `z - kd`,
    which a compiler reaches once `z = InvLn2N * xd` is substituted - changes the result at exactly two of the
    4,294,967,296 float32 inputs (measured here, both orders swept). These are those two. The expected values are
    this port's own unfused arithmetic, since glibc's are not among the recorded six, so the case guards
    -ffp-contract=off (runtime/CMakeLists.txt) rather than the bits: fused, the two would read 0x56FC9F1C and
    0x11FA2993. Neither the error bound nor the recorded values can see this, which is why it is a case."""
    assert list(bits_of(expf(f32([0x4202422F, 0xC27C65D9])))) == [0x56FC9F1B, 0x11FA2992]


# ---- C: the specials ---------------------------------------------------------------------------------------------

def test_c1_zeros():
    """C1. Both zeros take the main path and give exactly 1."""
    assert list(bits_of(expf(f32([0x00000000, 0x80000000])))) == [0x3F800000, 0x3F800000]


def test_c2_infinities():
    """C2. +inf passes through `x + x`; -inf is the one early return expf.c makes by bit pattern, before the NaN
    test, and gives +0 rather than an underflow."""
    assert list(bits_of(expf(f32([0x7F800000, 0xFF800000])))) == [0x7F800000, 0x00000000]


def test_c3_nan():
    """C3. A NaN reaches `x + x`, which quiets a signalling NaN and leaves a quiet one alone, payload included.
    glibc's own NaN bits are not in the recorded corpus, so this pins the port's rule, not glibc's."""
    got = bits_of(expf(f32([0x7FC00001, 0xFFC00001, 0x7F800001])))
    assert list(got) == [0x7FC00001, 0xFFC00001, 0x7FC00001]


def test_c4_overflow_boundary():
    """C4. The literal `0x1.62e42ep6f` is itself the largest argument with a finite result, and the next float up
    takes the overflow return. Both sides agree with the reference: above the literal the true value is past
    FLT_MAX, which rounds to +inf."""
    assert OFLOW == bits_of(F32(float.fromhex("0x1.62e42ep6")))
    assert bits_of(expf(f32([OFLOW])))[0] == 0x7F7FFF84          # finite, just under FLT_MAX
    assert bits_of(correctly_rounded(f32([OFLOW])))[0] == 0x7F7FFF84
    assert bits_of(expf(f32([OFLOW + 1])))[0] == 0x7F800000      # the overflow return
    assert np.isinf(correctly_rounded(f32([OFLOW + 1])))[0]
    assert bits_of(expf(f32([0x7F7FFFFF])))[0] == 0x7F800000     # the largest finite float


def test_c5_underflow_boundaries_and_the_band():
    """C5. The two underflow thresholds. `x < -0x1.9fe368p6f` returns (0x1p-95f)^2 = +0; the band above it,
    `x < -0x1.9d1d9ep6f`, returns (0x1.4p-75f)^2 = 0x00000001, which is WANT_ERRNO_UFLOW - a glibc choice Arm's own
    default build leaves off. Both agree with the correctly rounded value across the band, which is why the
    constant is allowed to stand in for the computation."""
    assert UFLOW == bits_of(F32(-float.fromhex("0x1.9fe368p6")))
    assert MAY_UFLOW == bits_of(F32(-float.fromhex("0x1.9d1d9ep6")))
    assert bits_of(expf(f32([UFLOW])))[0] == 0x00000001          # the comparison is strict: this one computes
    assert bits_of(expf(f32([UFLOW + 1])))[0] == 0x00000000      # and one float further down underflows
    band = f32(np.arange(MAY_UFLOW, UFLOW + 1, dtype=U32))       # the whole may_uflow band, 7,397 arguments
    assert (bits_of(expf(band)) == 0x00000001).all()
    assert (bits_of(expf(band)) == bits_of(correctly_rounded(band))).all()
    below = f32(np.arange(UFLOW + 1, UFLOW + 0x4000, dtype=U32))
    assert (bits_of(expf(below)) == 0).all() and (bits_of(correctly_rounded(below)) == 0).all()


def test_c6_subnormal_results():
    """C6. Every argument whose result is subnormal: from -87.3365478515625, the first whose result falls below
    2^-126, to the underflow threshold, the last with a result above zero - 2,180,453 of them, swept. The bound
    holds there too, where the ulp is 2^-149 and the double has 20-odd bits rounded away rather than one."""
    x = f32(np.arange(0xC2AE8000, UFLOW + 1, dtype=U32))
    out = bits_of(expf(x))
    assert (out > 0).all(), "nothing underflows to zero above the threshold"
    first = int(np.argmax(out < 0x00800000))
    assert bits_of(x)[first] == 0xC2AEAC50 and out[first] == 0x007FFFE6
    assert (out[first:] < 0x00800000).all(), "the results are monotone: subnormal from there down"
    assert ulp_error(x[first:]).max() <= 0.502


# ---- D: the SwiGLU -----------------------------------------------------------------------------------------------

def _swiglu_sample():
    rng = np.random.default_rng(1434)
    g = np.concatenate([
        rng.uniform(-40.0, 40.0, 1 << 18).astype(F32),
        rng.normal(0.0, 2.0, 1 << 18).astype(F32),
        f32([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, OFLOW, UFLOW, MAY_UFLOW]),
    ])
    u = np.concatenate([
        rng.normal(0.0, 1.0, 1 << 19).astype(F32),
        f32([0x3F800000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00002, 0x00000001, 0x7F7FFFFF, 0x00000000]),
    ])
    return g, u


def test_d1_swiglu_is_basalts_expression():
    """D1. `(g / (1.f + std::exp(-g))) * u` (pool.cpp:434): the divide, then the add, then the multiply, each in
    float32. The order is part of the result, so the check is bitwise against that expression and nothing looser."""
    g, u = _swiglu_sample()
    with np.errstate(invalid="ignore"):                       # -inf / inf is a NaN on both sides, not an error
        want = (g / (F32(1.0) + expf(-g))) * u
    got = swiglu(g, u)
    bad = np.flatnonzero(bits_of(got) != bits_of(want))
    assert bad.size == 0, (f"{bad.size} of {g.size} differ, first at g={bits_of(g)[bad[0]]:#010x} "
                           f"u={bits_of(u)[bad[0]]:#010x}: {bits_of(got)[bad[0]]:#010x} vs "
                           f"{bits_of(want)[bad[0]]:#010x}")


def test_d2_the_order_is_load_bearing():
    """D2. The witness for D1: the same algebra with the reciprocal taken first, which is what a port reaching for
    a sigmoid would write, differs on this sample. Without this, D1 would pass for a reordered kernel."""
    g, u = _swiglu_sample()
    with np.errstate(invalid="ignore"):
        reciprocal_first = (g * (F32(1.0) / (F32(1.0) + expf(-g)))) * u
    assert (bits_of(swiglu(g, u)) != bits_of(reciprocal_first)).any()


def test_d3_swiglu_specials():
    """D3. The specials basalt's expression itself produces: +inf passes through, -inf gives -inf/inf = NaN, and a
    NaN on either side propagates. Pinned as values, since a port that quietly guarded them would still pass D1."""
    g = f32([0x7F800000, 0xFF800000, 0x00000000, 0x80000000, 0x7FC00001])
    u = f32([0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000])
    got = bits_of(swiglu(g, u))
    assert got[0] == 0x7F800000                                  # inf / (1 + 0) * 1
    assert (got[1] & 0x7FFFFFFF) > 0x7F800000                    # -inf / inf = NaN
    assert got[2] == 0x00000000 and got[3] == 0x80000000         # +-0 keep their sign
    assert (got[4] & 0x7FFFFFFF) > 0x7F800000


# ---- the binding's own checks ------------------------------------------------------------------------------------

def test_binding_refuses_mismatched_lengths_and_dtypes():
    x = np.zeros(8, F32)
    with pytest.raises(ValueError, match="as many floats"):
        _nt.basalt_expf(x, np.zeros(7, F32))
    with pytest.raises(ValueError, match="same number of floats"):
        _nt.basalt_swiglu(x, x, np.zeros(9, F32))
    with pytest.raises(TypeError):
        _nt.basalt_expf(np.zeros(8, np.float64), np.zeros(8, np.float64))
    with pytest.raises(TypeError):
        _nt.basalt_expf(np.zeros((2, 4), F32), np.zeros((2, 4), F32))
    with pytest.raises(TypeError):
        _nt.basalt_expf(np.zeros(16, F32)[::2], np.zeros(8, F32))          # not contiguous
