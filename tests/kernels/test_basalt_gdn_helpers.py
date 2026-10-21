"""The GDN window's shared helpers on Metal (kernels/common/basalt_ieee.metal's ``__expf`` and ``rsqrtf`` block,
basalt_ieee_f64.metal, basalt_libdevice.metal, basalt_q8_1.metal and basalt_gdn.metal), each through an inline kernel
appended to ``basalt_gdn_source()`` (test_basalt_platform.py's pattern), one thread per element, against
tests/kernels/basalt_emu.py, which follows the SASS (docs/research/basalt-nvcc/dec_gdn.md, verify_gdn.md). Every
comparison is bitwise; a NaN result compares by class (the kernels' stores canonicalize NaN).

- H.1: ``sign_flip`` (by its bits, NaN payloads included), ``expf_ieee`` (``__expf``), ``expf_ieee(sign_flip(x))``
  against the SASS's folded constant (``__expf(-x)``: FMUL by 0xBFB8AA3B) and ``rsqrt_ieee`` (``rsqrtf``), the GPU's
  own ``fast::exp2`` and ``fast::rsqrt`` values injected for MUFU.EX2 and MUFU.RSQ: x uniform in [-110, 90], 256 ulps
  either side of where x * log2(e) crosses -126 (x near -87.34, both signs), zeros, subnormals of both signs, the
  smallest normal, the largest finite values, infinities and NaNs; for rsqrt also [2^-126, 2^128). The rows must
  exercise what they claim: subnormal results of the halved branch (which a plain square would flush) and a finite
  rsqrt of every positive subnormal (which a plain prescale would flush to an infinity). The results must also be
  within 1e-4 of float64's exp and 1/sqrt (relative; plus 2^-149 for exp), which catches an error both sides share.
- H.2: ``fma_ieee`` (``fma.rn.f32`` with subnormals) against ``basalt_emu.fma32``: random normals; subnormal operands
  beside products of 2^-149 to 2^-85; results that cancel into the subnormal range (c = -RN(a * b) moved by up to 4
  ulps) or round into it or up to 2^-126; products below 2^-126 of operands at least 2^-100 beside a c of 2^-100 to
  2^-97 (the fast path's premise: such a product cannot move the result); operands at the 2^-100 guard's edges with
  results on both sides of it; exact binary32 midpoint products of 2^-30 to 2^31 beside a nonzero c below 2^-100;
  every triple of zeros, ones, a subnormal, the edges, infinities and NaN. Fixed rows with their bits: engineering
  review E3's two (a subnormal product of normal operands on a zero addend, and operands of at least 2^-100 cancelling
  to 2^-146, which D6's literal guard returns as the hardware's +0), the two exact products that round up to 2^-126,
  the signs of zero products on +-0 addends (D10), and three exact results just off a binary32 midpoint (above one
  near 1; below a subnormal one; below one near 2^-100, of a product under the guard) and their negations. The
  hardware's own ``fma`` must differ from IEEE on many rows, so that the slow path is what the rows check; and
  rounding to nearest in binary64, then binary32 (the slow path with ``f64_fma`` for ``f64_fma_odd``) must give the
  wrong neighbour on the six and on many midpoint rows, so that its rounding to odd is checked too.
- H.3: ``log1pf_nv`` (libdevice's ``__nv_log1pf``, dec_gdn.md section 6a) against ``basalt_emu.log1pf_nv``, which
  runs the whole sequence with IEEE subnormals and FADD.RZ through ``Fraction``: [2^-126, 2^-60] of both signs, some
  below 2^-125 (FMUL p, m, p subnormal); subnormals of both signs, which come back as they are; 64 ulps either side of
  0.75 * 2^n - 1 for n = -2..29, which holds every float32 x whose round-to-nearest x + 1 steps k where FADD.RZ's does
  not; [-1, e^20] by bits and by value, [-1, 1] by value; zeros, -1, x < -1, the largest finite values, infinities and
  NaNs. Fixed rows with their bits: 0x3EFFFFFF, the one float32 whose result FADD.RZ changes (a round-to-nearest sum
  gives 0x3ECF9920), the smallest subnormals (which this GPU's m + s would flush to +0), +-0, -1, +inf, and one x per
  Horner step of the first five whose result changes when that step's FFMA is split into FMUL + FADD. The results
  must also be within 1 ulp of float64's log1p rounded to float32 wherever x >= -1, which catches an error both
  transcriptions share.
- H.4: ``q8_1_store_rec`` (dec_gdn_rec's q8_1 rows: native's d and codes, the block sum by unflushed adds) against
  ``basalt_emu.q8_1_rec``, which follows the SASS (dec_gdn.md section 6c: the max, the sum and the amax == 0 test
  unflushed), one thread per value and a SIMD-group per block: N(0, 1) blocks, values spread over e^-80 to e^80, whole
  blocks scaled from 2^-150 to 2^125, all-subnormal and subnormal-or-zero blocks (with NaN lanes too), values below
  2^-120 (d flushes to +0), lanes that cancel to subnormal sums of either sign, codes at rounding ties (an amax whose
  correctly rounded amax / 127 is not native's d, values within 2 ulps of (k + 1/2) d), NaN lanes of every payload
  (signaling too) and infinite ones (inf - inf too), all-NaN, zero and overflowing blocks. Fixed blocks with their
  bytes: a lone -1e-40, whose IEEE sum stores ds.y 0x8000 where the flushed one stores 0x0000, and two normals that
  cancel to -2^-148. Also ``q8_1_store_native``'s bytes on every block whose values are each 0 or at least 2^-102
  (G6's precondition), and its d and codes on every block (a flushed max changes no bit); the two ds.y must differ on
  many blocks, and only in the sign of a zero, and the tie blocks' codes must depend on the quotient's rounding.
- H.5: the GDN window's first steps (basalt_gdn.metal: ``gdn_conv``, ``gdn_silu``, ``gdn_sum4``, ``gdn_ab_dot``,
  ``gdn_beta``, ``gdn_gate``; the L2 scale as the kernels write it at the call site) against ``basalt_emu``'s twins,
  which follow the SASS rows (dec_gdn.md section 1, verify_gdn.md sections 2-5), the GPU's ``fast::exp2`` and
  ``fast::rsqrt`` injected; ``gdn_ab_dot`` one SIMD-group per row of 2560 (the window's n_embd). Classes: random
  model-like rows; conv products below 2^-126 of operands near 2^-63, subnormal taps or weights beside large ones, and
  tap 0 cancelling tap 1's rounded product into the subnormal range; SiLU and beta arguments that are subnormal or
  tiny, in (-88.73, -87.33) (a subnormal beta) and at the edges where ``__expf(-x)`` halves or overflows; partials,
  and adjacent dot elements, that cancel below 2^-126; L2 operands with a subnormal y or a zero or subnormal
  ``ss + eps``; gate arguments with ``u = v + dt`` in [-104, -87] (a subnormal ``__expf(u)``) against ``ssm_a`` on both
  sides of 1, and u at 20; whole rows of subnormal bf16 weights against activations near 2^100, tiny weights against
  tiny activations, and zero weights (a dot of +0: the chains start from +0); infinities and NaNs. Each kernel also
  stores the step in this GPU's plain operations, which must differ from IEEE's on many rows of each class built for
  a subnormal path; the other order of the conv's taps 0 and 1 must differ on many random rows, and the random rows
  must be near float64.
- H.6: the recurrence and output steps (basalt_gdn.metal: the decay ``expf_ieee(gate)``, ``gdn_kv``, kv_col by
  ``gdn_sum4``, ``gdn_delta``, ``gdn_update`` (the state update and o), ``gdn_oc``, ``gdn_sq32``, ss by ``gdn_sum4``,
  ``gdn_out``) against ``basalt_emu``'s twins, which follow the SASS rows (dec_gdn.md sections 1-2, verify_gdn.md
  section 6), the GPU's ``fast::exp2`` and ``fast::rsqrt`` injected: one head per threadgroup of 512 in
  gdn_step_norm_multi's layout, k and q in threadgroup memory, one token, every intermediate compared in the window's
  order. Classes: random model-like heads (L2-normed k and q, gates of a few units, beta in (0, 1), eps 1e-6); whole
  columns of state and v scaled by 2^-140 to 1 (a decaying state: tiny partial sums and deltas, subnormal updated
  states, subnormal oc whose y the norm's scale of about 2^10 makes normal); gates in (-103.2, -87.4), a subnormal
  decay, beside zero key rows, zero v and a +0 or subnormal beta, so that g s reaches the state; z in (-88.72, -87.34),
  a subnormal sigmoid; a state and v near 2^-61 with eps 0 (subnormal squares, ss * 2^-7 subnormal: the norm's rsqrt
  takes its scaled branch); infinities, NaNs and signed zeros. Each step's plain form on the same operands must differ
  from IEEE's on many elements of each class built for it, and the random heads' decay, updated state, oc and y must
  be within 1e-5 of float64's, relative to the sum of the absolute terms.
"""

import numpy as np

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

U32, F32, F64 = np.uint32, np.float32, np.float64


def run_kernel(src, name, inputs, cols, dtype=U32):
    """The inline kernel ``name`` (``src``, appended to basalt_gdn_source()), one thread per row of ``inputs``
    (equal-length arrays at buffers 0, 1, ...); then the output, ``cols`` values of ``dtype`` per row, starting as 0xFF
    bytes, and the row count at the next slot. Returns the output as (rows, cols)."""
    n = len(inputs[0])
    pso = gpu.pipeline(kernels.basalt_gdn_source() + src, name)
    bufs = [np.ascontiguousarray(x).tobytes() for x in inputs] + [cols * n * np.dtype(dtype).itemsize]
    out = gpu.run(pso, bufs, np.uint32(n).tobytes(), len(inputs) + 1, -(-n // 256), 256, fill=0xFF)
    return np.frombuffer(out[-1], dtype).reshape(n, cols)


def nan32(b):
    return (b & U32(0x7FFFFFFF)) > U32(0x7F800000)


def differ(got, want):
    """Rows where float32 bits ``got`` are not ``want``; a NaN matches a NaN of any payload."""
    return (nan32(got) != nan32(want)) | (~nan32(got) & (got != want))


def assert_bits(name, got, want, *x):
    """``got`` (float32 bits) is ``want`` bit for bit; a NaN matches a NaN of any payload. ``x``: the operands."""
    want = np.asarray(want, F32).view(U32)
    bad = np.flatnonzero(differ(got, want))
    assert bad.size == 0, (f"{name}: {bad.size} of {len(got)} differ; first x="
                           + ", ".join(f"{v.view(U32)[bad[0]]:#010x}" for v in x)
                           + f": got {got[bad[0]]:#010x}, want {want[bad[0]]:#010x}")


# ---- H.1: sign_flip, expf_ieee, rsqrt_ieee -------------------------------------------------------------------------

H1 = """
kernel void h1_exp_rsqrt(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = x[i];
  device uint* o = out + 4 * i;
  o[0] = as_type<uint>(sign_flip(v));
  o[1] = as_type<uint>(expf_ieee(v));
  o[2] = as_type<uint>(expf_ieee(sign_flip(v)));
  o[3] = as_type<uint>(rsqrt_ieee(v));
}

// the stand-ins for MUFU.EX2 and MUFU.RSQ alone: the values the emulation injects
kernel void h1_fast(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[2 * i] = as_type<uint>(fast::exp2(x[i]));
  out[2 * i + 1] = as_type<uint>(fast::rsqrt(x[i]));
}
"""
NL2E = U32(0xBFB8AA3B).view(F32)          # -log2(e) rounded: the constant nvcc folds __expf(-x)'s negation into


def fast_exp2(e):
    return run_kernel(H1, "h1_fast", [np.asarray(e, F32)], 2).view(F32)[:, 0]


def fast_rsqrt(x):
    return run_kernel(H1, "h1_fast", [np.asarray(x, F32)], 2).view(F32)[:, 1]


def h1_args(rng):
    """H.1's arguments (float32) by class."""
    x0 = F32(-126.0 / F64(emu.L2E))                         # fl(x * log2(e)) crosses -126 near here
    edge = (x0.view(U32) + np.arange(-256, 257)).astype(U32).view(F32)
    sub = rng.integers(1, 1 << 23, 1 << 12, dtype=U32)
    normal = (rng.integers(1, 255, 1 << 12, dtype=U32) << U32(23)) | rng.integers(0, 1 << 23, 1 << 12, dtype=U32)
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,   # 2^-126
                        0x7F000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,                    # 2^127, max
                        0x7FC00000, 0xFFC00001, 0x7F800001, 0xFFFFFFFF], U32)                          # NaNs
    return {"uniform": rng.uniform(-110, 90, 1 << 16).astype(F32), "edge": np.r_[edge, -edge],
            "subnormal": np.r_[sub, sub | U32(0x80000000)].view(F32), "normal": normal.view(F32),
            "special": special.view(F32)}


def test_sign_flip_expf_ieee_rsqrt_ieee_match_the_emulation():
    args = h1_args(np.random.default_rng(31))
    x = np.concatenate(list(args.values()))
    xb = x.view(U32)
    got = run_kernel(H1, "h1_exp_rsqrt", [x], 4)
    assert np.array_equal(got[:, 0], emu.sign_flip(x).view(U32)), "sign_flip: not the bits with the sign flipped"
    with np.errstate(over="ignore", invalid="ignore"):
        neg = emu.ex2n(x * NL2E, fast_exp2)                 # __expf(-x): FMUL e, x, 0xBFB8AA3B, then ex2n
        e = x * emu.L2E
    assert_bits("expf_ieee", got[:, 1], emu.expf_ieee(x, fast_exp2), x)
    assert_bits("expf_ieee(sign_flip(x))", got[:, 2], neg, x)
    assert_bits("rsqrt_ieee", got[:, 3], emu.rsqrt_ieee(x, fast_rsqrt), x)
    # the rows exercise what they claim: both sides of the halving threshold, subnormal results of the halved branch
    # (a plain r * r flushes them), and a finite rsqrt of every positive subnormal (a plain x * 2^24 flushes to 0)
    edge_e = args["edge"][:513] * emu.L2E
    assert np.any(edge_e < -126) and np.any(edge_e >= -126)
    res = got[:, 1]
    assert np.count_nonzero((e < -126) & ((res & U32(0x7F800000)) == 0) & ((res & U32(0x7FFFFFFF)) != 0)) > 1000
    ps = ((xb & U32(0x7F800000)) == 0) & (xb != 0) & (xb < U32(0x80000000))
    assert np.count_nonzero(ps) > 1000 and np.all(np.isfinite(got[ps, 3].view(F32)))
    # near float64: an error both sides share (a constant, the threshold, a scale) cannot hide
    with np.errstate(invalid="ignore"):                     # the signaling-NaN rows raise the flag in the cast
        r, xf = got.view(F32).astype(F64), x.astype(F64)
    for col, sign in [(1, 1.0), (2, -1.0)]:
        m = (sign * xf >= -110) & (sign * xf <= 88)
        want = np.exp(sign * xf[m])
        assert np.all(np.abs(r[m, col] - want) <= 1e-4 * want + 2.0 ** -149), col
    p = (xb > 0) & (xb < U32(0x7F800000))
    want = 1 / np.sqrt(xf[p])
    assert np.all(np.abs(r[p, 3] - want) <= 1e-4 * want)


# ---- H.2: fma_ieee -------------------------------------------------------------------------------------------------

H2 = """
kernel void h2_fma(device const uint* a [[buffer(0)]], device const uint* b [[buffer(1)]],
                   device const uint* c [[buffer(2)]], device uint* out [[buffer(3)]],
                   constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float x = as_type<float>(a[i]), y = as_type<float>(b[i]), z = as_type<float>(c[i]);
  out[2 * i] = as_type<uint>(fma_ieee(x, y, z));
  out[2 * i + 1] = as_type<uint>(fma(x, y, z));             // the hardware's alone
}
"""
SIGN = U32(0x80000000)
EDGE = np.array([0x0D7FFFFF, 0x0D800000, 0x0D800001], U32)   # just below 2^-100, 2^-100, just above
# (a, b, c, IEEE's bits), checked exactly with Fraction
FIXED = [(0x1C800000, 0x1C800000, 0x00000000, 0x00000200),   # E3: 2^-70 * 2^-70 + 0 = 2^-140, a subnormal
         (0x3F800001, 0x0D800001, 0x8D800002, 0x00000008),   # E3: operands >= 2^-100 cancelling to 2^-146
         (0x1EF80000, 0x21042108, 0x00000000, 0x00800000),   # 31 2^-70 * 1082401 2^-81 = 2^-126 - 2^-151: up
         (0x1FFFF000, 0x20000800, 0x00000000, 0x00800000),   # 4095 2^-75 * 4097 2^-75 = 2^-126 - 2^-150: the tie
         (0xBF800000, 0x00000000, 0x00000000, 0x00000000),   # D10: -1 * 0 + 0 = +0
         (0x3F800000, 0x80000000, 0x00000000, 0x00000000),   # 1 * -0 + 0 = +0
         (0xBF800000, 0x00000000, 0x80000000, 0x80000000),   # -1 * 0 + -0 = -0
         (0x80000000, 0x3F800000, 0x80000000, 0x80000000),   # -0 * 1 + -0 = -0
         (0xBF800000, 0x80000000, 0x80000000, 0x00000000),   # -1 * -0 + -0 = +0
         (0x00000000, 0x00000000, 0x80000000, 0x00000000),   # 0 * 0 + -0 = +0
         # double rounding (rounding to nearest in binary64, then binary32) gives the other neighbour on these six:
         (0x3F800800, 0x3F800800, 0x00000001, 0x3F801001),   # (1 + 2^-12)^2, a midpoint, + 2^-149: up (0x3F801000)
         (0xBF800800, 0x3F800800, 0x80000001, 0xBF801001),
         (0x19FFFFFE, 0x1A000001, 0x00000201, 0x00000201),   # 2^-150 (1 - 2^-46) + 513 2^-149: down (0x00000202)
         (0x99FFFFFE, 0x1A000001, 0x80000201, 0x80000201),
         (0x3F800001, 0x017FFFFE, 0x0D800001, 0x0D800001),   # 2^-124 (1 - 2^-46) + 2^-100 (1 + 2^-23): a product
         (0xBF800001, 0x017FFFFE, 0x8D800001, 0x8D800001)]   # below the guard, down (0x0D800002)


def floats(rng, exp):
    """Random signs and fractions on the biased exponents ``exp`` (float32 bits)."""
    exp = np.asarray(exp)
    return (rng.integers(0, 1 << 32, exp.shape, dtype=U32) & U32(0x807FFFFF)) | (exp.astype(U32) << U32(23))


def f32(v):
    """float64 values as float32 bits, rounded to nearest even."""
    return np.asarray(v, F64).astype(F32).view(U32)


def h2_rows(rng, n=4096):
    """H.2's operand triples (float32 bits) by class."""
    k = np.arange(n)
    sign = lambda: rng.integers(0, 2, n).astype(U32) << U32(31)
    swap = lambda x, y: (np.where(k % 2 == 0, x, y), np.where(k % 2 == 0, y, x))
    rows = {}
    ea, eb = rng.integers(64, 191, n), rng.integers(64, 191, n)
    ec = np.clip(ea + eb - 127 + rng.integers(-30, 31, n), 1, 254)                   # within 30 binades of a * b
    rows["normal"] = (floats(rng, ea), floats(rng, eb), floats(rng, ec))
    # a subnormal or lowest-binade a (or b) times 1 to 2^40; c +-0, subnormal or near the product
    lo, hi = floats(rng, rng.choice([0, 0, 1, 2], n)), floats(rng, rng.integers(127, 168, n))
    c = np.select([k % 3 == 0, k % 3 == 1], [sign(), floats(rng, np.zeros(n, int))],
                  floats(rng, rng.integers(1, 42, n)))
    rows["subnormal"] = (*swap(lo, hi), c)
    # products of 2^-126 to 2^-57 less their own rounding, moved by up to 4 ulps
    ea, pe = rng.integers(64, 128, n), rng.integers(1, 71, n)
    a, b = floats(rng, ea), floats(rng, pe + 127 - ea)
    p = f32(a.view(F32).astype(F64) * b.view(F32).astype(F64))
    rows["cancel"] = (a, b, (p.astype(np.int64) + rng.integers(-4, 5, n)).astype(U32) ^ SIGN)
    # c = +-0, b = RN(target / a): products within about 2^-146 of +-2^-126, or at 2^-127 to 2^-150
    u = np.where(k % 2 == 0, 0, rng.integers(1, 25, n))
    target = np.ldexp(1 + rng.integers(-40, 41, n) * 2.0 ** -26, -126 - u) * np.where(k % 4 < 2, 1.0, -1.0)
    a = floats(rng, rng.integers(77, 101, n))
    rows["round"] = (a, f32(target / a.view(F32).astype(F64)), sign())
    # products below 2^-126 of operands at least 2^-100, beside a c of 2^-100 to 2^-97: IEEE's result is c
    ea = rng.integers(27, 64, n)
    rows["tiny_product"] = (floats(rng, ea), floats(rng, rng.integers(27, 127 - ea)),
                            floats(rng, rng.integers(27, 30, n)))
    # one of a, b at the guard's edge, the other in [0.5, 2), c at the edge or +-0: results on both sides of 2^-100
    e, m = rng.choice(EDGE, n) | sign(), floats(rng, rng.integers(126, 128, n))
    rows["edge"] = (*swap(e, m), np.where(k % 4 == 0, sign(), rng.choice(EDGE, n) | sign()))
    # binary32 midpoints: a * b of 2^-30 to 2^31 with one bit, the 25th significant, below the 24 kept (odd parts of na
    # and 26 - na bits whose product has 25), beside a nonzero c below 2^-100 (the slow path) and below half a binary64
    # ulp of a * b: rounding to nearest twice ties to the even neighbour, the wrong one where c points to the odd one
    na = rng.integers(2, 25, 4 * n)
    oa, ob = rng.integers(1 << (na - 1), 1 << na) | 1, rng.integers(1 << (25 - na), 1 << (26 - na)) | 1
    keep = oa * ob < 1 << 25
    ma, mb = (oa << (24 - na))[keep][:n], (ob << (na - 2))[keep][:n]
    assert len(ma) == n
    ea = rng.integers(97, 158, n)
    eb = 254 - ea + rng.integers(-30, 31, n)
    rows["midpoint"] = (sign() | (ea.astype(U32) << U32(23)) | (ma - (1 << 23)).astype(U32),
                        (eb.astype(U32) << U32(23)) | (mb - (1 << 23)).astype(U32),
                        sign() | rng.integers(1, 0x0D800000, n, dtype=U32))
    v = np.array([0, 0x80000000, 0x3F800000, 0xBF800000, 0x00080000, 0x80080000, 0x0D800000, 0x8D7FFFFF,
                  0x7F800000, 0xFF800000, 0x7FC00000], U32)   # +-0, +-1, +-2^-130, the edges, +-inf, NaN
    rows["specials"] = tuple(g.ravel() for g in np.meshgrid(v, v, v, indexing="ij"))
    return rows


def test_fma_ieee_matches_the_emulation():
    rows = h2_rows(np.random.default_rng(32))
    fixed = np.array(FIXED, U32)
    a, b, c = (np.concatenate([fixed[:, i]] + [r[i] for r in rows.values()]) for i in range(3))
    got = run_kernel(H2, "h2_fma", [a, b, c], 2)
    with np.errstate(invalid="ignore"):                     # the signaling-NaN rows raise the flag in the cast
        want = emu.fma32(a.view(F32), b.view(F32), c.view(F32)).view(U32)
    nf = len(FIXED)
    assert np.array_equal(want[:nf], fixed[:, 3]), "basalt_emu.fma32 misses a fixed row's bits"
    assert np.array_equal(got[:nf, 0], fixed[:, 3]), f"fma_ieee on the fixed rows: {[hex(g) for g in got[:nf, 0]]}"
    assert_bits("fma_ieee", got[:, 0], want.view(F32), a, b, c)
    # the rows exercise what they claim: D6's literal guard returns the hardware's +0 on E3's rows; the hardware
    # differs from IEEE on many rows, many results are subnormal, and some round up to 2^-126 from below
    hw = got[:, 1]
    assert np.all(hw[:2] == 0), f"the hardware's fma on E3's rows: {hw[0]:#010x}, {hw[1]:#010x}"
    assert np.count_nonzero(differ(hw, want)) > 1000
    mag = want & U32(0x7FFFFFFF)
    assert np.count_nonzero((mag != 0) & (mag < U32(0x00800000))) > 1000
    at, off = {}, nf
    for name, r in rows.items():
        at[name], off = slice(off, off + len(r[0])), off + len(r[0])
    with np.errstate(invalid="ignore"):
        prod = np.abs(a.view(F32).astype(F64) * b.view(F32).astype(F64))      # exact
    sl = at["round"]
    assert np.count_nonzero((prod[sl] < 2.0 ** -126) & (mag[sl] == U32(0x00800000))) > 0
    # the fast path's premise: a product below 2^-126 beside a c of at least 2^-100 does not move it
    sl = at["tiny_product"]
    assert np.all(prod[sl] < 2.0 ** -126)
    assert np.array_equal(hw[sl], c[sl]) and np.array_equal(want[sl], c[sl])
    edge = mag[at["edge"]]
    assert np.count_nonzero(edge < U32(0x0D800000)) > 100 and np.count_nonzero(edge >= U32(0x0D800000)) > 100
    # double rounding gives the wrong neighbour on the six fixed traps and on many midpoint rows, each of which the
    # guard sends to the slow path (c nonzero and below 2^-100): what they check is the slow path's rounding to odd
    twice = lambda s: f32(a.view(F32)[s].astype(F64) * b.view(F32)[s].astype(F64) + c.view(F32)[s].astype(F64))
    sl, mc = at["midpoint"], c[at["midpoint"]] & U32(0x7FFFFFFF)
    assert np.all(twice(slice(nf - 6, nf)) != want[nf - 6:nf])
    assert np.all((mc != 0) & (mc < U32(0x0D800000))) and np.count_nonzero(twice(sl) != want[sl]) > 1000


# ---- H.3: log1pf_nv ------------------------------------------------------------------------------------------------

H3 = """
kernel void h3_log1p(device const uint* x [[buffer(0)]], device uint* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = as_type<uint>(log1pf_nv(as_type<float>(x[i])));
}
"""
E20 = np.exp(F64(20)).astype(F32)          # e^20: the softplus's x = __expf(u), u <= 20, stays at or below it
# (x, log1pf's bits); NaN results compare by class (CUDA's FFMA gives 0x7FFFFFFF)
H3_FIXED = [(0x3EFFFFFF, 0x3ECF991F),      # 0.5 - 2^-25: RN(x + 1) = 1.5 steps k, RZ does not; RN gives 0x3ECF9920
            (0x00000001, 0x00000001),      # 2^-149 as is: this GPU's m + s would flush it to +0
            (0x80000001, 0x80000001),
            (0x00000000, 0x00000000),
            (0x80000000, 0x80000000),      # -0: the special block's FSEL
            (0xBF800000, 0xFF800000),      # -1: the sequence's -inf
            (0x7F800000, 0x7F800000),      # +inf: fma(x, inf, inf)
            (0xFF800000, 0x7FFFFFFF),      # -inf, x < -1 and NaN: fma(x, inf, inf), NaN
            (0xC0000000, 0x7FFFFFFF),
            (0x7FC00000, 0x7FFFFFFF),
            # one per Horner step from the first: that step's FFMA split into FMUL + FADD gives
            (0x3E5D5456, 0x3E486147),      # 0x3E486146
            (0x3E5BA6F5, 0x3E46FFF8),      # 0x3E46FFF9
            (0x3E3A2BA5, 0x3E2B0DC6),      # 0x3E2B0DC5
            (0x3E0187DD, 0x3DF3F06B),      # 0x3DF3F06A
            (0x3D9BC76F, 0x3D962384)]      # 0x3D962383


def h3_args(rng, n=1 << 12):
    """H.3's arguments (float32 bits) by class."""
    tiny = np.r_[rng.integers(0x00800000, F32(2.0 ** -60).view(U32) + 1, n, dtype=U32),
                 rng.integers(0x00800000, 0x01000000, n // 4, dtype=U32)]      # [2^-126, 2^-125): m * p subnormal
    sub = rng.integers(1, 1 << 23, n, dtype=U32)
    x0 = (0.75 * 2.0 ** np.arange(-2, 30) - 1).astype(F32).view(U32)          # where k steps: x + 1 = 0.75 * 2^n
    top = int(E20.view(U32))
    domain = [rng.integers(0, top + 1, 1 << 16, dtype=U32),                     # [0, e^20] by bits
              rng.integers(0x80000000, 0xBF800001, 1 << 15, dtype=U32),         # [-1, -0] by bits
              rng.uniform(-1, float(E20), 1 << 14).astype(F32).view(U32),
              rng.uniform(-1, 1, 1 << 14).astype(F32).view(U32)]
    special = np.array([0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000, 0xBF7FFFFF, 0xBF800001,
                        0x41A00000, top, 0x7F7FFFFF, 0xFF7FFFFF, 0xFF800001, 0xFFC00001, 0x7F800001, 0xFFFFFFFF], U32)
    return {"tiny": np.r_[tiny, tiny | SIGN], "subnormal": np.r_[sub, sub | SIGN],
            "rz": (x0[:, None].astype(np.int64) + np.arange(-64, 65)).astype(U32).ravel(),
            "domain": np.concatenate(domain), "special": special}


def test_log1pf_nv_matches_the_emulation():
    args = h3_args(np.random.default_rng(33))
    fixed = np.array(H3_FIXED, U32)
    xb = np.concatenate([fixed[:, 0]] + list(args.values()))
    x = xb.view(F32)
    got = run_kernel(H3, "h3_log1p", [xb], 1)[:, 0]
    with np.errstate(invalid="ignore"):                     # the signaling-NaN rows raise the flag in the cast
        want = emu.log1pf_nv(x)
    nf = len(H3_FIXED)
    assert not np.any(differ(want.view(U32)[:nf], fixed[:, 1])), "basalt_emu.log1pf_nv misses a fixed row's bits"
    assert not np.any(differ(got[:nf], fixed[:, 1])), f"log1pf_nv on the fixed rows: {[hex(g) for g in got[:nf]]}"
    assert_bits("log1pf_nv", got, want, x)
    # the rows exercise what they claim: subnormals come back as they are and nonzero, tiny normals as they are, some
    # below 2^-125; the rz rows hold all five float32 x whose round-to-nearest x + 1 gives another k than FADD.RZ's
    # (n = 0, 1, 2 and 25: x + 1 is exact for x in [-1, -0.5] and, n = 3..24, where x and x + 1 share a binade;
    # from n = 26, ulp(x) >= 4 and both sums are x)
    at, off = {}, nf
    for name, r in args.items():
        at[name], off = slice(off, off + len(r)), off + len(r)
    sl = at["subnormal"]
    assert np.array_equal(got[sl], xb[sl]) and np.all((got[sl] & U32(0x7FFFFFFF)) != 0)
    sl = at["tiny"]
    assert np.array_equal(got[sl], xb[sl])
    assert np.count_nonzero((xb[sl] & U32(0x7FFFFFFF)) < U32(0x01000000)) > 1000
    xr = x[at["rz"]]
    s = xr.astype(F64) + 1.0                                # exact: 2^-26 <= ulp(x) and |x| < 2^29
    rn = s.astype(F32)
    rz = np.where(np.abs(rn.astype(F64)) > np.abs(s), np.nextafter(rn, F32(0)), rn)
    k = lambda t: (t.view(U32) - U32(0x3F400000)) & U32(0xFF800000)
    assert np.count_nonzero(k(rn) != k(rz)) == 5
    # near float64: within 1 ulp of log1p rounded to float32 (on ordered keys), wherever x >= -1
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = ~np.isnan(x) & (x >= -1)
        ref = np.log1p(x[ok].astype(F64)).astype(F32).view(U32)
    key = lambda b: np.where(b >= SIGN, -(b & U32(0x7FFFFFFF)).astype(np.int64), b.astype(np.int64))
    assert np.max(np.abs(key(got[ok]) - key(ref))) <= 1, "log1pf_nv: more than 1 ulp from float64's log1p"


# ---- H.4: q8_1_store_rec -------------------------------------------------------------------------------------------

H4 = """
kernel void h4_q8_1(device const uint* x [[buffer(0)]], device uchar* rec [[buffer(1)]],
                    device uchar* native [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;                                       // n is a multiple of 32: whole SIMD-groups return
  const float v = as_type<float>(x[i]);
  q8_1_store_rec(v, rec, i);
  q8_1_store_native(v, native, i);
}
"""
NANS = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF], U32)
# ({lane: bits}, the other lanes +0; q8_1_store_rec's ds.x, ds.y and {lane: code}, the other codes 0); the flushed sum
# of q8_1_store_native is +0 on both, its ds.y 0x0000
H4_FIXED = [({0: 0x800116C2}, 0x0000, 0x8000, {}),                  # -1e-40 alone: the IEEE sum's half is -0
            ({0: 0x81000001, 16: 0x01000000}, 0x0000, 0x8000,      # -(2^-125 + 2^-148) + 2^-125 = -2^-148; d flushes
             {0: -127, 16: 127})]                                   # to +0, so the quotients are +-inf, clamped


def h4_run(xb):
    """q8_1_store_rec's and q8_1_store_native's blocks (rows of 36 bytes, starting as 0xFF) of the float32 bits ``xb``,
    a multiple of 32 of them: one thread per value, a SIMD-group per block."""
    n = len(xb)
    assert n % 32 == 0
    pso = gpu.pipeline(kernels.basalt_gdn_source() + H4, "h4_q8_1")
    out = gpu.run(pso, [np.ascontiguousarray(xb, U32).tobytes(), 36 * n // 32, 36 * n // 32], np.uint32(n).tobytes(),
                  3, -(-n // 256), 256, fill=0xFF)
    return [np.frombuffer(b, np.uint8).reshape(-1, 36) for b in out[1:]]


def h4_blocks(rng, nb=256):
    """H.4's blocks (rows of 32 float32 bits) by class."""
    sub = lambda *s: rng.integers(1, 1 << 23, s, dtype=U32) | (rng.integers(0, 2, s, dtype=U32) << U32(31))
    zero = lambda *s: rng.integers(0, 2, s, dtype=U32) << U32(31)
    normal = lambda *s: rng.standard_normal(s).astype(F32).view(U32)
    rows = np.arange(nb)
    b = {"random": (rng.standard_normal((2048, 32)) * 3).astype(F32).view(U32),
         "wide": (rng.standard_normal((nb, 32)) * np.exp(rng.uniform(-80, 80, (nb, 32)))).astype(F32).view(U32),
         "scaled": np.ldexp(rng.standard_normal((4 * nb, 32)), rng.integers(-150, 126, (4 * nb, 1))).astype(F32)
         .view(U32),                                                # whole blocks from the subnormals to 2^127
         "subnormal": sub(nb, 32),
         "sub_or_zero": np.where(rng.random((nb, 32)) < rng.choice([0.03, 0.2, 0.5], (nb, 1)), sub(nb, 32),
                                 zero(nb, 32)),
         "sub_nan": np.where(rng.random((nb, 32)) < 0.1, rng.choice(NANS, (nb, 32)),
                             np.where(rng.random((nb, 32)) < 0.5, sub(nb, 32), zero(nb, 32))),
         "flush_d": floats(rng, rng.integers(0, 7, (nb, 32)))}     # below 2^-120: amax * fl(1/127) flushes to +0
    # lane j + 16 cancels lane j to (k_j + e_j) 2^-149 with |e_j| <= 4: subnormal sums of either sign
    k = rng.integers(1 << 20, (1 << 24) - 4, (nb, 16)) * rng.choice([-1, 1], (nb, 16))
    m = np.concatenate([k, -(k + rng.integers(-4, 5, (nb, 16)))], axis=1)
    b["cancel"] = np.ldexp(m.astype(F64), -149).astype(F32).view(U32)
    # codes at rounding ties: lane 0 an amax whose fl(amax / 127) is not native's d = fl(amax * fl(1/127)), the others
    # within 2 ulps of (k + 1/2) d, where the quotient's rounding decides the code
    a = np.abs(rng.standard_normal(1 << 14)).astype(F32)
    a = a[a / F32(127.0) != emu.mul_ftz(a, emu.RCP127)][:nb]
    near = ((rng.integers(-127, 127, (nb, 31)) + 0.5) * emu.mul_ftz(a, emu.RCP127)[:, None]).astype(F32).view(U32)
    near = (near.astype(np.int64) + rng.integers(-2, 3, (nb, 31))).astype(U32)
    b["tie"] = np.c_[(a * rng.choice([-1, 1], nb).astype(F32)).view(U32), near]
    x = normal(nb, 32)
    x[rows, rng.integers(0, 32, nb)] = rng.choice(NANS, nb)
    x[rows[::2], rng.integers(0, 32, nb // 2)] = rng.choice(NANS, nb // 2)
    b["nan"] = np.r_[x, np.repeat(NANS, 32).reshape(-1, 32)]       # one or two NaN lanes; all NaN
    x = normal(nb, 32)
    x[rows, rng.integers(0, 32, nb)] = rng.choice([0x7F800000, 0xFF800000], nb)
    x[rows[::4], 7], x[rows[::4], 23] = 0x7F800000, 0xFF800000      # inf - inf: a NaN sum
    b["inf"] = x
    b["zero"] = np.r_[np.zeros((1, 32), U32), np.full((1, 32), SIGN), zero(6, 32)]
    b["overflow"] = np.r_[(rng.standard_normal((nb, 32)) * 1e5).astype(F32).view(U32),          # sums past 65504
                          (rng.uniform(1e38, 3.4e38, (nb, 32)) * rng.choice([-1, 1], (nb, 32))).astype(F32)
                          .view(U32)]                                                         # past FLT_MAX
    return b


def test_q8_1_store_rec_matches_the_emulation():
    blocks = h4_blocks(np.random.default_rng(34))
    fixed = np.zeros((len(H4_FIXED), 32), U32)
    lit = np.zeros((len(H4_FIXED), 36), np.uint8)
    for r, (lanes, dsx, dsy, codes) in enumerate(H4_FIXED):
        fixed[r, list(lanes)] = list(lanes.values())
        lit[r, :4] = np.array([dsx, dsy], np.uint16).view(np.uint8)
        lit[r, [4 + j for j in codes]] = np.array(list(codes.values()), np.int8).view(np.uint8)
    xb = np.concatenate([fixed] + list(blocks.values()))
    names = np.repeat(["fixed"] + list(blocks), [len(fixed)] + [len(v) for v in blocks.values()])
    rec, nat = h4_run(xb.ravel())
    with np.errstate(invalid="ignore"):                     # the signaling-NaN lanes raise the flag in the casts
        want = emu.q8_1_rec(xb.view(F32).ravel()).reshape(-1, 36)
    nf = len(H4_FIXED)
    assert np.array_equal(want[:nf], lit), "basalt_emu.q8_1_rec misses a fixed block's bytes"
    assert np.array_equal(rec[:nf], lit), f"q8_1_store_rec on the fixed blocks: {[r.tobytes().hex() for r in rec[:nf]]}"
    bad = np.flatnonzero(np.any(rec != want, axis=1))
    assert bad.size == 0, (f"q8_1_store_rec: {bad.size} of {len(rec)} blocks differ; first block {bad[0]} "
                           f"({names[bad[0]]}): got {rec[bad[0]].tobytes().hex()}, want {want[bad[0]].tobytes().hex()}")
    # G6's precondition (every value 0 or at least 2^-102): q8_1_store_native stores the same bytes, so dec_gdn_parity's
    # comparison of dec_gdn_rec's yq with the native q8_1 of the per-stage y holds
    mag = xb & U32(0x7FFFFFFF)
    g6 = np.all((mag == 0) | ((mag >= U32(0x0C800000)) & (mag <= U32(0x7F800000))), axis=1)
    assert np.array_equal(rec[g6], nat[g6]), "q8_1_store_rec and q8_1_store_native differ under G6's precondition"
    # the d and the codes are native's on every block (the SASS's unflushed max and zero test change no bit), and the
    # sum's half differs only in the sign of a zero (dec_gdn.md section 6c); the rows exercise what they claim: many
    # blocks meet the precondition, many have a ds.y the flushed sum changes, the fixed ones among them, and many have
    # a subnormal largest |x| (the SASS's codes are then NaN quotients, 0) or a normal one whose d flushes
    keep = np.r_[0:2, 4:36]
    assert np.array_equal(rec[:, keep], nat[:, keep]), "q8_1_store_rec's d or codes are not native's"
    dsy = lambda q: q[:, 2:4].copy().view(np.uint16)[:, 0]
    diff = dsy(rec) != dsy(nat)
    assert np.all((dsy(rec) | dsy(nat))[diff] == 0x8000), "a ds.y differs in more than the sign of a zero"
    assert np.count_nonzero(g6) > 2000 and np.count_nonzero(diff) > 200 and np.all(diff[:nf])
    top = np.max(np.where(nan32(xb), U32(0), mag), axis=1)
    assert np.count_nonzero((top > 0) & (top < U32(0x00800000))) > 100
    assert np.count_nonzero((top >= U32(0x00800000)) & (top < U32(0x03800000))) > 100     # below 2^-120
    # and the tie blocks' codes depend on the tail: the correctly rounded quotient gives others on many lanes
    v = xb[names == "tie"].view(F32)
    d = emu.mul_ftz(np.abs(v[:, :1]), emu.RCP127)
    assert np.count_nonzero(emu.roundf(v / d) != emu.roundf(emu.mul_ftz(v, emu.rcp_ftz(d)))) > 100


# ---- H.5: gdn_conv, gdn_silu, gdn_sum4, the L2 scale, gdn_ab_dot, gdn_beta, gdn_gate --------------------------------

H5 = """
// each step, then its plain form: this GPU's own operations, which read and write subnormals as zero
kernel void h5_conv(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  float v[8];
  for (uint k = 0; k < 8; ++k) v[k] = as_type<float>(a[8 * i + k]);
  out[2 * i] = as_type<uint>(gdn_conv(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]));
  out[2 * i + 1] = as_type<uint>(fma(v[3], v[7], fma(v[2], v[6], fma(v[0], v[4], v[1] * v[5]))));
}

kernel void h5_unary(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float s = as_type<float>(a[i]);
  device uint* o = out + 4 * i;
  o[0] = as_type<uint>(gdn_silu(s));
  o[1] = as_type<uint>(s / (1.0f + expf_neg(s)));
  o[2] = as_type<uint>(gdn_beta(s));
  o[3] = as_type<uint>(1.0f / (1.0f + expf_neg(s)));
}

kernel void h5_sum4(device const uint4* a [[buffer(0)]], device uint* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float4 p = as_type<float4>(a[i]);
  out[2 * i] = as_type<uint>(gdn_sum4(p.x, p.y, p.z, p.w));
  out[2 * i + 1] = as_type<uint>(((p.x + p.y) + p.z) + p.w);
}

// the L2 scale, which the kernels write at its call sites (dec_gdn_rec.cu:228-229, verify_kernels.cu:53)
kernel void h5_l2(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                  constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float y = as_type<float>(a[3 * i]), ss = as_type<float>(a[3 * i + 1]), eps = as_type<float>(a[3 * i + 2]);
  out[2 * i] = as_type<uint>(ieee_mul(y, rsqrt_ieee(ieee_add(ss, eps))));
  out[2 * i + 1] = as_type<uint>(y * fast::rsqrt(ss + eps));
}

kernel void h5_gate(device const uint* a [[buffer(0)]], device uint* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = as_type<float>(a[3 * i]), dt = as_type<float>(a[3 * i + 1]), sa = as_type<float>(a[3 * i + 2]);
  out[2 * i] = as_type<uint>(gdn_gate(v, dt, sa));
  const float u = v + dt;
  out[2 * i + 1] = as_type<uint>((u > 20.0f ? u : log1pf_nv(expf_ftz(u))) * sa);
}

// one SIMD-group per row of 8 * n8 bf16 weights and as many activations: lane 0 stores the dot and the plain chain's
kernel void h5_dot(device const uint4* w [[buffer(0)]], device const float4* x [[buffer(1)]],
                   device uint* out [[buffer(2)]], constant uint2& p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
  const uint row = i / 32, lane = i % 32, n8 = p.y;
  if (row >= p.x) return;                                   // whole SIMD-groups return
  const float v = gdn_ab_dot(w + row * n8, x + 2 * row * n8, lane, n8);
  device const ushort* wr = (device const ushort*) (w + row * n8);
  device const uint* xr = (device const uint*) (x + 2 * row * n8);
  float u = 0.0f;
  for (uint j = lane; j < n8; j += 32)
    for (uint e = 0; e < 8; ++e) u = fma(as_type<float>(uint(wr[8 * j + e]) << 16), as_type<float>(xr[8 * j + e]), u);
  u = warp_sum(u);
  if (lane == 0) {
    out[2 * row] = as_type<uint>(v);
    out[2 * row + 1] = as_type<uint>(u);
  }
}
"""
H5_N = 2560                # n_embd, the one width the window takes: 320 words of 8 bf16 weights, 10 per lane
H5_SPECIAL = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF, 0x00000001], U32)


def stack(classes):
    """The classes' rows concatenated, and each class's slice of them."""
    at, off = {}, 0
    for name, r in classes.items():
        at[name], off = slice(off, off + len(r)), off + len(r)
    return np.concatenate(list(classes.values())), at


def subnormal32(b):
    """Nonzero float32 bits below 2^-126 (subnormal)."""
    m = b & U32(0x7FFFFFFF)
    return (m != 0) & (m < U32(0x00800000))


def exercised(plain, at, least):
    """``plain`` (the rows where the plain form differs from IEEE's) holds more than ``least[c]`` rows of class c."""
    n = {c: int(np.count_nonzero(plain[at[c]])) for c in least}
    assert all(n[c] > least[c] for c in least), f"the plain form differs on too few rows: {n}, want more than {least}"


def h5_conv_rows(rng, n=2048):
    """H.5's conv operands by class: rows of v0, v1, v2, x, w0, w1, w2, w3 (float32 bits)."""
    k = np.arange(n)[:, None]
    big, sub = floats(rng, rng.integers(147, 187, (n, 4))), floats(rng, np.zeros((n, 4), int))
    # v0 = -v1 and w0 = w1 moved by up to 4 ulps: tap 0 cancels tap 1's rounded product (2^-126 to 2^-61) down to the
    # subnormal range; the other two taps zero, or products of 2^-154 to 2^-132
    ea, pe = rng.integers(64, 128, n), rng.integers(1, 67, n)
    a, b = floats(rng, ea), floats(rng, pe + 127 - ea)
    w0 = (b.astype(np.int64) + rng.integers(-4, 5, n)).astype(U32)
    tail = np.where(k % 2 == 0, floats(rng, np.zeros((n, 4), int)) & SIGN, floats(rng, rng.integers(50, 61, (n, 4))))
    special = rng.standard_normal((n // 8, 8)).astype(F32).view(U32)
    special[np.arange(n // 8), rng.integers(0, 8, n // 8)] = rng.choice(H5_SPECIAL, n // 8)
    return {"random": np.c_[rng.standard_normal((n, 4)) * 2, rng.standard_normal((n, 4)) * 0.3].astype(F32).view(U32),
            "tiny": floats(rng, rng.integers(52, 66, (n, 8))),        # 2^-75 to 2^-62: products below 2^-123
            "subnormal": np.c_[np.where(k % 2 == 0, big, sub), np.where(k % 2 == 0, sub, big)],   # beside 2^20 to 2^60
            "cancel": np.c_[a ^ SIGN, a, tail[:, :2], w0, b, tail[:, 2:]],
            "special": special}


def test_gdn_conv_matches_the_emulation():
    rows, at = stack(h5_conv_rows(np.random.default_rng(35)))
    v = rows.view(F32)
    got = run_kernel(H5, "h5_conv", [rows], 2)
    with np.errstate(invalid="ignore"):                     # the signaling-NaN rows raise the flag in the casts
        want = emu.gdn_conv(*v.T)
    assert_bits("gdn_conv", got[:, 0], want, *v.T)
    # the rows exercise what they claim: the plain fma chain differs on the small classes, and the other order of taps
    # 0 and 1 (tap 0's product rounded alone) on the random rows (dec_gdn.md section 6d)
    exercised(differ(got[:, 1], want.view(U32)), at, {"tiny": 1000, "subnormal": 1500, "cancel": 300})
    r = v[at["random"]]
    other = emu.fma32(r[:, 3], r[:, 7], emu.fma32(r[:, 2], r[:, 6], emu.fma32(r[:, 1], r[:, 5], r[:, 0] * r[:, 4])))
    assert np.count_nonzero(other != want[at["random"]]) > 150, "taps 0 and 1 in the other order: too few rows differ"
    # near float64 on the random rows: an error both sides share cannot hide
    t = r[:, :4].astype(F64) * r[:, 4:].astype(F64)
    s = got[at["random"], 0].view(F32).astype(F64)
    assert np.all(np.abs(s - t.sum(1)) <= 2.0 ** -21 * np.abs(t).sum(1))


def h5_unary_args(rng, n=1 << 12):
    """H.5's SiLU and beta arguments (float32 bits) by class."""
    sign = rng.integers(0, 2, n, dtype=U32) << U32(31)
    # where -x * log2(e) crosses 126 (beta's divisor passes 2^126), -126 (__expf(-x) halves) and 128 (it overflows)
    cross = (np.array([-126.0, 126.0, -128.0]) / F64(emu.L2E)).astype(F32).view(U32)
    edge = (cross[:, None].astype(np.int64) + np.arange(-128, 129)).astype(U32).ravel()
    special = np.array([0, SIGN, 1, 0x80000001, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
                        0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001], U32)
    return {"uniform": rng.uniform(-110, 110, n).astype(F32).view(U32),
            "subnormal": rng.integers(1, 1 << 23, n, dtype=U32) | sign,
            "tiny": rng.integers(0x00800000, 0x0D800000, n, dtype=U32) | sign,     # [2^-126, 2^-100)
            "beta_sub": rng.uniform(-88.73, -87.33, n).astype(F32).view(U32),     # beta below 2^-126
            "edge": edge, "special": special}


def test_gdn_silu_and_gdn_beta_match_the_emulation():
    args, at = stack(h5_unary_args(np.random.default_rng(36)))
    x = args.view(F32)
    got = run_kernel(H5, "h5_unary", [args], 4)
    with np.errstate(invalid="ignore"):
        silu, beta = emu.gdn_silu(x, fast_exp2), emu.gdn_beta(x, fast_exp2)
    assert_bits("gdn_silu", got[:, 0], silu, x)
    assert_bits("gdn_beta", got[:, 2], beta, x)
    exercised(differ(got[:, 1], silu.view(U32)), at, {"subnormal": 3000, "tiny": 50})
    exercised(differ(got[:, 3], beta.view(U32)), at, {"beta_sub": 3000, "edge": 100})
    assert np.count_nonzero(subnormal32(got[at["beta_sub"], 2])) > 3000, "too few subnormal betas"
    # near float64 (where float32's __expf(-x) does not overflow)
    with np.errstate(invalid="ignore"):
        xf, r = x.astype(F64), got.view(F32).astype(F64)
    with np.errstate(over="ignore", invalid="ignore"):
        for col, lo, ref in [(0, -88.0, xf / (1 + np.exp(-xf))), (2, -88.7, 1 / (1 + np.exp(-xf)))]:
            m = np.zeros(len(x), bool)
            m[at["uniform"]] = True
            m &= xf >= lo
            assert np.all(np.abs(r[m, col] - ref[m]) <= 1e-5 * np.abs(ref[m]) + 2.0 ** -149), col


def h5_sum4_rows(rng, n=2048):
    """H.5's gdn_sum4 partials by class: rows of p0..p3 (float32 bits)."""
    # p0 + p1 = -e 2^-149 (|e| <= 4) exactly, p2 and p3 below 2^-129: every partial sum subnormal or zero
    k = rng.integers(1 << 20, (1 << 24) - 4, n) * rng.choice([-1, 1], n)
    m = np.c_[k, -(k + rng.integers(-4, 5, n)), rng.integers(-(1 << 20), 1 << 20, (n, 2))]
    special = np.array([[0x7F800000, 0xFF800000, 0, 0], [0x7FC00000, 0x3F800000, 0, 0], [0x7F7FFFFF, 0x7F7FFFFF, 0, 0],
                        [SIGN, SIGN, SIGN, SIGN], [0x3F800000, 0xBF800000, SIGN, SIGN],
                        [0x00800001, 0x80800000, 0, 0]], U32)               # the last: 2^-149, which a flush loses
    return {"random": (rng.standard_normal((n, 4)) * np.exp(rng.uniform(-3, 3, (n, 1)))).astype(F32).view(U32),
            "cancel": np.ldexp(m.astype(F64), -149).astype(F32).view(U32),
            "subnormal": floats(rng, np.zeros((n, 4), int)), "special": special}


def test_gdn_sum4_matches_the_emulation():
    rows, at = stack(h5_sum4_rows(np.random.default_rng(37)))
    p = rows.view(F32)
    got = run_kernel(H5, "h5_sum4", [rows], 2)
    with np.errstate(invalid="ignore"):
        want = emu.gdn_sum4(*p.T)
    assert_bits("gdn_sum4", got[:, 0], want, *p.T)
    assert got[at["special"].stop - 1, 0] == 1, "gdn_sum4(2^-126 + 2^-149, -2^-126, 0, 0) is not 2^-149"
    exercised(differ(got[:, 1], want.view(U32)), at, {"cancel": 1500, "subnormal": 1500})
    r = p[at["random"]].astype(F64)
    s = got[at["random"], 0].view(F32).astype(F64)
    assert np.all(np.abs(s - r.sum(1)) <= 2.0 ** -21 * np.abs(r).sum(1))


def h5_l2_rows(rng, n=2048):
    """H.5's L2-scale operands by class: rows of y, ss, eps (float32 bits)."""
    sign = rng.integers(0, 2, n, dtype=U32) << U32(31)
    one, eps = 0x3F800000, int(F32(1e-6).view(U32))
    # y subnormal or below 2^-112 times rsqrt(ss + 1e-6) of 31 to 1000: products a flushed y loses
    y = np.where(np.arange(n) % 2 == 0, rng.integers(1, 1 << 23, n, dtype=U32),
                 rng.integers(0x00800000, 0x07800000, n, dtype=U32)) | sign
    ss = np.exp(rng.uniform(np.log(1e-9), np.log(1e-3), n)).astype(F32).view(U32)
    # ss + eps zero or subnormal (eps 0 or subnormal; the kernels' is 1e-6): rsqrt_ieee's scaled branch
    lo = np.where(rng.random((n, 2)) < 0.2, U32(0), rng.integers(1, 1 << 23, (n, 2), dtype=U32))
    special = np.array([[0x7FC00000, one, eps], [0x7F800000, one, eps], [one, 0x7F800000, eps], [one, 0xBF800000, eps],
                        [0, 0, 0], [one, 0, 0], [SIGN, one, eps]], U32)
    return {"random": np.c_[(rng.standard_normal(n) * 0.3).astype(F32).view(U32),
                            (rng.chisquare(128, n) * 0.09).astype(F32).view(U32), np.full(n, eps, U32)],
            "small_y": np.c_[y, ss, np.full(n, eps, U32)],
            "small_ss": np.c_[rng.standard_normal(n).astype(F32).view(U32), lo],
            "special": special}


def test_the_l2_scale_matches_the_emulation():
    rows, at = stack(h5_l2_rows(np.random.default_rng(38)))
    y, ss, eps = rows.view(F32).T
    got = run_kernel(H5, "h5_l2", [rows], 2)
    with np.errstate(invalid="ignore"):
        want = emu.gdn_l2_scale(y, ss, eps, fast_rsqrt)
    assert_bits("the L2 scale", got[:, 0], want, y, ss, eps)
    exercised(differ(got[:, 1], want.view(U32)), at, {"small_y": 500, "small_ss": 1000})
    sl = at["random"]
    ref = y[sl].astype(F64) / np.sqrt(ss[sl].astype(F64) + eps[sl].astype(F64))
    assert np.all(np.abs(got[sl, 0].view(F32).astype(F64) - ref) <= 1e-5 * np.abs(ref))


def h5_gate_rows(rng, n=2048):
    """H.5's gate operands by class: rows of v, dt, sa (float32 bits)."""
    dt = rng.standard_normal(n)
    sa = rng.choice([-1, 1], n) * np.exp2(rng.uniform(-10, 30, n))      # |sa| on both sides of 1
    v20 = (np.int64(F32(20.0).view(U32)) + np.arange(-64, 65)).astype(U32).view(F32)
    edge = np.c_[np.r_[v20, 19.5, 10.0], np.r_[np.zeros(len(v20)), 0.5, 10.0], np.full(len(v20) + 2, -2.0)]
    special = np.array([[np.nan, 0, -1], [0, np.nan, -1], [1, 0, np.nan], [np.inf, 0, -1], [-np.inf, 0, -1],
                        [1, 0, np.inf], [-np.inf, 0, np.inf], [np.inf, -np.inf, -1], [0, 0, 0], [-0.0, -0.0, -1]])
    u = rng.uniform(-104, -87, n)
    rows = {"random": np.c_[rng.standard_normal(n) * 1.5, dt, -np.exp(rng.uniform(-1, 3, n))],
            "subnormal_exp": np.c_[u - dt, dt, sa],                  # u = v + dt in [-104, -87]: __expf(u) subnormal
            "large": np.c_[rng.uniform(20, 100, n // 4), dt[:n // 4], sa[:n // 4]],
            "edge": edge, "special": special}
    with np.errstate(invalid="ignore"):
        return {c: r.astype(F32).view(U32) for c, r in rows.items()}


def test_gdn_gate_matches_the_emulation():
    rows, at = stack(h5_gate_rows(np.random.default_rng(39)))
    v, dt, sa = rows.view(F32).T
    got = run_kernel(H5, "h5_gate", [rows], 2)
    with np.errstate(invalid="ignore"):
        want = emu.gdn_gate(v, dt, sa, fast_exp2)
    assert_bits("gdn_gate", got[:, 0], want, v, dt, sa)
    exercised(differ(got[:, 1], want.view(U32)), at, {"subnormal_exp": 1000})
    g = got[at["subnormal_exp"], 0]
    assert np.count_nonzero(subnormal32(g)) > 200 and np.count_nonzero((g & U32(0x7F800000)) != 0) > 200
    sl = at["random"]
    u = v[sl].astype(F64) + dt[sl].astype(F64)
    ref = np.where(u > 20, u, np.log1p(np.exp(u))) * sa[sl].astype(F64)
    assert np.all(np.abs(got[sl, 0].view(F32).astype(F64) - ref) <= 1e-5 * np.abs(ref))


def bf16(x):
    """float32 values as bf16 bits, truncated."""
    return (np.asarray(x, F32).view(U32) >> U32(16)).astype(np.uint16)


def h5_dot_rows(rng, r=128, n=H5_N):
    """H.5's gdn_ab_dot rows by class: (bf16 weight bits, activations), n of each per row."""
    sgn = lambda *s: rng.choice([-1.0, 1.0], s)
    # adjacent elements cancel to -e_i 2^-149 (|e_i| <= 4; weights 1): subnormal chains and butterflies
    k = rng.integers(1, (1 << 24) - 4, (r, n // 2)) * sgn(r, n // 2)
    m = np.stack([k, -(k + rng.integers(-4, 5, (r, n // 2)))], axis=2).reshape(r, n)
    # zero weights, every product -0 in the first two rows: the chains start from +0 (the first FFMA's RZ)
    zw = np.repeat(np.array([0, 0x8000, 0, 0], np.uint16)[:, None], n, axis=1)
    zw[3] = rng.choice(np.array([0, 0x8000], np.uint16), n)
    zx = np.abs(rng.standard_normal((4, n))) * np.array([-1, 1, 1, 1])[:, None]
    zx[3] *= sgn(n)
    sw, sx = bf16(rng.standard_normal((8, n)) * 0.03), rng.standard_normal((8, n))
    col = rng.integers(0, n, 8)
    sx[np.arange(4), col[:4]] = [np.inf, -np.inf, np.nan, np.inf]
    sw[np.arange(4, 8), col[4:]] = [0x7F80, 0xFF80, 0x7FC0, 0x7F80]          # bf16 inf, -inf, NaN
    sub = (rng.integers(1, 128, (r, n)) | (rng.integers(0, 2, (r, n)) << 15)).astype(np.uint16)   # bf16 subnormals
    return {"random": (bf16(rng.standard_normal((r, n)) * 0.03), rng.standard_normal((r, n))),
            "subnormal_w": (sub, sgn(r, n) * np.exp2(rng.uniform(90, 110, (r, n)))),
            "tiny": (bf16(sgn(r, n) * np.exp2(rng.uniform(-75, -60, (r, n)))),
                     sgn(r, n) * np.exp2(rng.uniform(-80, -65, (r, n)))),        # products of 2^-155 to 2^-125
            "cancel": (np.full((r, n), 0x3F80, np.uint16), np.ldexp(m.astype(F64), -149)),
            "zero": (zw, zx), "special": (sw, sx)}


def h5_dot_run(w, x):
    """gdn_ab_dot of each row of bf16 weight bits ``w`` against the same row of ``x``, and the plain chain's (float32
    bits, rows of 2): one SIMD-group per row."""
    rows, n = w.shape
    pso = gpu.pipeline(kernels.basalt_gdn_source() + H5, "h5_dot")
    out = gpu.run(pso, [np.ascontiguousarray(w, np.uint16).tobytes(), np.ascontiguousarray(x, F32).tobytes(), 8 * rows],
                  np.array([rows, n // 8], U32).tobytes(), 3, -(-rows // 8), 256, fill=0xFF)
    return np.frombuffer(out[2], U32).reshape(rows, 2)


def test_gdn_ab_dot_matches_the_emulation():
    classes = h5_dot_rows(np.random.default_rng(40))
    w, at = stack({c: r[0] for c, r in classes.items()})
    x = np.concatenate([np.asarray(r[1], F32) for r in classes.values()])
    names = np.repeat(list(classes), [len(r[0]) for r in classes.values()])
    got = h5_dot_run(w, x)
    with np.errstate(invalid="ignore"):
        want = emu.gdn_ab_dot(w, x).view(U32)
    bad = np.flatnonzero(differ(got[:, 0], want))
    assert bad.size == 0, (f"gdn_ab_dot: {bad.size} of {len(w)} rows differ; first row {bad[0]} ({names[bad[0]]}): "
                           f"got {got[bad[0], 0]:#010x}, want {want[bad[0]]:#010x}")
    assert np.all(got[at["zero"], 0] == 0), f"zero weights: {[hex(g) for g in got[at['zero'], 0]]}, not +0"
    exercised(differ(got[:, 1], want), at, {"subnormal_w": 64, "tiny": 64, "cancel": 64})
    sl = at["random"]
    t = ((w[sl].astype(U32) << U32(16)).view(F32).astype(F64)) * x[sl].astype(F64)
    s = got[sl, 0].view(F32).astype(F64)
    assert np.all(np.abs(s - t.sum(1)) <= 1e-5 * np.abs(t).sum(1))


# ---- H.6: the decay, gdn_kv, gdn_delta, gdn_update, gdn_oc, gdn_sq32, gdn_out --------------------------------------

H6 = """
// one head per threadgroup of 512 threads in gdn_step_norm_multi's layout (col = tid % 128, rg = tid / 128; k and q in
// threadgroup memory), one token: each step as the kernels call it, its plain form (this GPU's own operations) beside
// it on the same operands. Every thread stores a record of H6_W words: the (step, plain) pairs of the decay, kv,
// kv_col, delta, o, oc, its SIMD-group's sum of squares, ss and y, then its 32 updated state rows and their plain forms
kernel void h6_step(device const uint* st [[buffer(0)]], device const uint* vec [[buffer(1)]],
                    device const uint* sc [[buffer(2)]], device uint* out [[buffer(3)]],
                    uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float sk[128], sq[128], red[4][128], wsum[4];
  const uint col = tid % 128, rg = tid / 128;
  device const uint* hv = vec + 640 * head;                 // k, q, v, z, gamma: 128 each
  if (tid < 128) {
    sk[tid] = as_type<float>(hv[tid]);
    sq[tid] = as_type<float>(hv[128 + tid]);
  }
  const float v = as_type<float>(hv[256 + col]), z = as_type<float>(hv[384 + col]), gam = as_type<float>(hv[512 + col]);
  const float gate = as_type<float>(sc[4 * head]), b = as_type<float>(sc[4 * head + 1]);
  const float eps = as_type<float>(sc[4 * head + 2]);
  float s[32], s0[32];
  for (uint r = 0; r < 32; ++r) s[r] = s0[r] = as_type<float>(st[16384 * head + (32 * rg + r) * 128 + col]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float g = expf_ieee(gate), g_p = expf_ftz(gate);
  const float kv = gdn_kv(s, sk + 32 * rg);
  float kv_p = 0.0f;
  for (uint r = 0; r < 32; ++r) kv_p = fma(s0[r], sk[32 * rg + r], kv_p);
  red[rg][col] = kv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float kv_col = gdn_sum4(red[0][col], red[1][col], red[2][col], red[3][col]);
  const float kvc_p = ((red[0][col] + red[1][col]) + red[2][col]) + red[3][col];
  const float delta = gdn_delta(g, kv_col, v, b), delta_p = fma(sign_flip(g), kv_col, v) * b;
  const float o = gdn_update(s, sk + 32 * rg, sq + 32 * rg, g, delta);
  device uint* w = out + H6_W * (512 * head + tid);
  float o_p = 0.0f;
  for (uint r = 0; r < 32; ++r) {
    w[18 + r] = as_type<uint>(s[r]);
    w[50 + r] = as_type<uint>(fma(g, s0[r], sk[32 * rg + r] * delta));
    o_p = fma(s[r], sq[32 * rg + r], o_p);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);         // every thread has read red's kv partials
  red[rg][col] = o;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float oc = gdn_oc(red[0][col], red[1][col], red[2][col], red[3][col]);
  const float oc_p = (((red[0][col] + red[1][col]) + red[2][col]) + red[3][col]) * as_type<float>(0x3DB504F3u);
  const float sq2 = gdn_sq32(oc), sq2_p = warp_sum(oc * oc);
  if (tid < 128 && tid % 32 == 0) wsum[tid / 32] = sq2;    // the four column groups of rg 0, in order
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ss = gdn_sum4(wsum[0], wsum[1], wsum[2], wsum[3]);
  const float ss_p = ((wsum[0] + wsum[1]) + wsum[2]) + wsum[3];
  const float y = gdn_out(oc, ss, eps, gam, z);
  const float y_p = ((oc * fast::rsqrt(fma(ss, 0.0078125f, eps))) * gam) * (1.0f / (1.0f + expf_neg(z)));
  const float f[18] = {g, g_p, kv, kv_p, kv_col, kvc_p, delta, delta_p, o, o_p, oc, oc_p, sq2, sq2_p, ss, ss_p, y, y_p};
  for (uint i = 0; i < 18; ++i) w[i] = as_type<uint>(f[i]);
}
"""
H6_W = 82                  # a thread's record: 9 (step, plain) pairs, its 32 updated state rows, their plain forms


def h6_run(st, vec, sc):
    """h6_step over the heads (float32 state (heads, 128, 128) rows by columns, vec (heads, 5, 128): k, q, v, z, gamma,
    sc (heads, 4): gate, beta, eps, 0): one threadgroup of 512 per head. The records as (heads, 4, 128, H6_W) uint32,
    [head, rg, col] for thread 128 rg + col."""
    n = len(st)
    pso = gpu.pipeline(kernels.basalt_gdn_source() + f"constant constexpr uint H6_W = {H6_W};\n" + H6, "h6_step")
    assert pso.max_threads_per_threadgroup >= 512, pso.max_threads_per_threadgroup
    out = gpu.run(pso, [np.ascontiguousarray(a, F32).tobytes() for a in (st, vec, sc)] + [4 * H6_W * 512 * n], None, 0,
                  n, 512, fill=0xFF)
    return np.frombuffer(out[3], U32).reshape(n, 4, 128, H6_W)


def h6_heads(rng, n=16):
    """H.6's heads by class: float64 (state (n, 128, 128) rows by columns, vec (n, 5, 128): k, q, v, z, gamma, sc
    (n, 4): gate, beta, eps, 0)."""
    def base():
        kq = rng.standard_normal((2, n, 128))
        kq /= np.linalg.norm(kq, axis=2, keepdims=True)              # L2-normed, as the window's q and k
        vec = np.stack([kq[0], kq[1], rng.standard_normal((n, 128)), rng.standard_normal((n, 128)) * 2,
                        1 + 0.2 * rng.standard_normal((n, 128))], axis=1)
        sc = np.c_[-np.exp(rng.uniform(-4, 2.5, n)), 1 / (1 + np.exp(-rng.standard_normal(n))), np.full(n, 1e-6),
                   np.zeros(n)]
        return rng.standard_normal((n, 128, 128)) * 0.3, vec, sc

    heads = {"random": base()}
    # whole columns, their state and v, scaled by 2^-e (e up to 140) and a quarter of the v zero: a state decayed to
    # 2^-140, tiny partial sums and deltas, subnormal oc whose y the norm's scale makes normal
    st, vec, sc = base()
    scale = np.exp2(-rng.uniform(0, 140, (n, 1, 128)))
    st *= scale
    vec[:, 2] *= scale[:, 0] * (rng.random((n, 128)) < 0.75)
    heads["decaying"] = (st, vec, sc)
    # gates in (-103.2, -87.4): a subnormal decay g; then g s reaches the state where kd is about 0: a third of the
    # key rows zero, half the v zero, a beta of +0 or a subnormal one on two heads in three
    st, vec, sc = base()
    vec[:, 0] *= rng.random((n, 128)) < 2 / 3
    vec[:, 2] *= rng.random((n, 128)) < 0.5
    sc[:, 0] = rng.uniform(-103.2, -87.4, n)
    third = np.arange(n) % 3
    sc[:, 1] = np.select([third == 1, third == 2], [0.0, np.exp2(-rng.uniform(127, 149, n))], sc[:, 1])
    heads["subnormal_decay"] = (st, vec, sc)
    # z in (-88.72, -87.34): a subnormal sigmoid (1 + e passes 2^126) on three columns in four
    st, vec, sc = base()
    vec[:, 3] = np.where(rng.random((n, 128)) < 0.75, rng.uniform(-88.72, -87.34, (n, 128)), vec[:, 3])
    heads["small_out"] = (st, vec, sc)
    # the state and v scaled by 2^-58 to 2^-64 and eps 0: |oc| near 2^-66, subnormal squares, ss * 2^-7 subnormal (the
    # norm's rsqrt takes its scaled branch)
    st, vec, sc = base()
    scale = np.exp2(-rng.uniform(58, 64, (n, 1, 1)))
    st *= scale
    vec[:, 2] *= scale[:, 0]
    sc[:, 2] = 0
    heads["small_ss"] = (st, vec, sc)
    # infinities, NaNs and zeros planted one kind per head
    st, vec, sc = base()
    st[0, 5, 7], st[1, 40, 3], vec[2, 0, 17], vec[3, 1, 100], vec[4, 2, 9] = np.nan, np.inf, np.inf, np.nan, -np.inf
    sc[5:10, 0] = [np.nan, -np.inf, np.inf, -200.0, 0.0]           # g NaN, +0, inf, +0 (halved: 2^-144.3 squared), 1
    sc[10:12, 1] = [np.nan, np.inf]
    vec[12, 3, :8] = [np.nan, np.inf, -np.inf, 0.0, -0.0, 88.8, -88.8, 1e30]
    vec[13, 4, :4] = [np.inf, np.nan, 0.0, -0.0]
    st[14], vec[14, 2] = 0.0, 0.0                                   # a zero head: the chains' +0 from RZ
    st[14, ::2], vec[14, 2, ::2] = -0.0, -0.0
    st[15], vec[15, 2], sc[15, 2] = 0.0, 0.0, 0.0                   # and with eps 0: rsqrt_ieee(+0) = inf, y NaN
    heads["special"] = (st, vec, sc)
    return heads


def test_gdn_recurrence_and_output_steps_match_the_emulation():
    classes = h6_heads(np.random.default_rng(41))
    st, at = stack({c: r[0].astype(F32) for c, r in classes.items()})
    vec = np.concatenate([r[1] for r in classes.values()]).astype(F32)
    sc = np.concatenate([r[2] for r in classes.values()]).astype(F32)
    names = np.repeat(list(classes), [len(r[0]) for r in classes.values()])
    got = h6_run(st, vec, sc)
    n = len(st)
    k, q, v, z, gam = vec.transpose(1, 0, 2)
    gate, b, eps = sc[:, 0], sc[:, 1], sc[:, 2]
    s4 = st.reshape(n, 4, 32, 128).transpose(0, 1, 3, 2)          # [head, rg, col, r]: thread (rg, col)'s rows
    k4, q4 = k.reshape(n, 4, 1, 32), q.reshape(n, 4, 1, 32)
    with np.errstate(over="ignore", invalid="ignore"):
        g = emu.expf_ieee(gate, fast_exp2)
        kv = emu.gdn_kv(s4, k4)
        kv_col = emu.gdn_sum4(*kv.transpose(1, 0, 2))
        delta = emu.gdn_delta(g[:, None], kv_col, v, b[:, None])
        s1, o = emu.gdn_update(s4, k4, q4, g[:, None, None], delta[:, None, :])
        oc = emu.gdn_oc(*o.transpose(1, 0, 2))
        sq2 = emu.gdn_sq32(oc.reshape(-1, 32)).reshape(n, 4)       # lane = col % 32 of column group col // 32
        ss = emu.gdn_sum4(*sq2.T)
        y = emu.gdn_out(oc.ravel(), np.repeat(ss, 128), np.repeat(eps, 128), gam.ravel(), z.ravel(), fast_exp2,
                        fast_rsqrt).reshape(n, 128)
    # (step, its field pair, the threads that hold it: every thread, rg 0's columns, rg 0's SIMD-groups, one per head)
    every, cols, groups, one = np.s_[:, :, :], np.s_[:, 0, :], np.s_[:, 0, ::32], np.s_[:, 0, 0]
    steps = [("decay", 0, one, g), ("kv", 1, every, kv), ("kv_col", 2, cols, kv_col), ("delta", 3, cols, delta),
             ("update", None, every, s1), ("o", 4, every, o), ("oc", 5, cols, oc), ("sq32", 6, groups, sq2),
             ("ss", 7, one, ss), ("y", 8, cols, y)]
    plain = {}
    for name, f, sel, want in steps:                             # in the window's order: the first failure names it
        ieee, plain_bits = (got[..., 18:50], got[..., 50:82]) if f is None else (got[sel + (2 * f,)],
                                                                                 got[sel + (2 * f + 1,)])
        want = np.asarray(want, F32).view(U32)
        bad = np.argwhere(differ(ieee, want))
        assert len(bad) == 0, (f"{name}: {len(bad)} of {want.size} differ; first at {tuple(bad[0])} "
                               f"({names[bad[0][0]]}): got {ieee[tuple(bad[0])]:#010x}, want "
                               f"{want[tuple(bad[0])]:#010x}")
        plain[name] = differ(plain_bits, want)
    # the classes exercise what they claim: the plain step differs on many of their elements, and the subnormal values
    # the IEEE path keeps are there (subnormal updated states; subnormal oc whose y is normal; ss * 2^-7 subnormal)
    least = {"decay": {"subnormal_decay": 15}, "kv": {"decaying": 600}, "kv_col": {"decaying": 120},
             "delta": {"decaying": 120, "subnormal_decay": 350},
             "update": {"decaying": 15000, "subnormal_decay": 100000},
             "o": {"decaying": 600, "subnormal_decay": 3000}, "oc": {"decaying": 120, "subnormal_decay": 700},
             "sq32": {"small_ss": 60}, "ss": {"small_ss": 8},
             "y": {"decaying": 120, "subnormal_decay": 700, "small_out": 700, "small_ss": 1000}}
    for name, m in plain.items():
        if name in least:
            exercised(m, at, least[name])
    sub_s = subnormal32(got[..., 18:50])
    assert np.count_nonzero(sub_s[at["decaying"]]) > 15000 and np.count_nonzero(sub_s[at["subnormal_decay"]]) > 100000
    normal_y = subnormal32(got[:, 0, :, 10]) & ((got[:, 0, :, 16] & U32(0x7F800000)) != 0)
    assert np.count_nonzero(normal_y[at["decaying"]]) > 50, "too few subnormal oc with a normal y"
    assert np.all(got[at["small_ss"], 0, 0, 14].view(F32) < F32(2.0 ** -119)), "small_ss: an ss * 2^-7 not subnormal"
    # near float64 on the random heads: an error both sides share (a constant, an operand order, a scale) cannot hide;
    # each tolerance is relative to the sum of the absolute terms
    sl = at["random"]
    s64, k64, q64, v64, z64, gam64 = (a[sl].astype(F64) for a in (st, k, q, v, z, gam))
    g64, b64, eps64 = np.exp(gate[sl].astype(F64))[:, None], b[sl].astype(F64)[:, None], eps[sl].astype(F64)[:, None]
    kv64, kva = np.einsum("hrc,hr->hc", s64, k64), np.einsum("hrc,hr->hc", np.abs(s64), np.abs(k64))
    d64, da = (v64 - g64 * kv64) * b64, (np.abs(v64) + g64 * kva) * b64
    s164 = g64[:, :, None] * s64 + k64[:, :, None] * d64[:, None, :]
    s1a = g64[:, :, None] * np.abs(s64) + np.abs(k64)[:, :, None] * da[:, None, :]
    oc64 = np.einsum("hrc,hr->hc", s164, q64) / np.sqrt(128)
    oca = np.einsum("hrc,hr->hc", s1a, np.abs(q64)) / np.sqrt(128)
    scale = 1 / np.sqrt((oc64 ** 2).sum(1, keepdims=True) / 128 + eps64)
    sig = 1 / (1 + np.exp(-z64))
    r = got[sl].view(F32).astype(F64)
    gs = r[..., 18:50].transpose(0, 1, 3, 2).reshape(-1, 128, 128)          # [head, row, col]
    assert np.all(np.abs(r[:, 0, 0, 0] - g64[:, 0]) <= 1e-5 * g64[:, 0]), "the decay is not near float64's"
    assert np.all(np.abs(gs - s164) <= 1e-5 * s1a), "the updated state is not near float64's"
    assert np.all(np.abs(r[:, 0, :, 10] - oc64) <= 1e-5 * oca), "oc is not near float64's"
    ya = oca * scale * np.abs(gam64) * sig
    assert np.all(np.abs(r[:, 0, :, 16] - oc64 * scale * gam64 * sig) <= 1e-5 * ya), "y is not near float64's"
