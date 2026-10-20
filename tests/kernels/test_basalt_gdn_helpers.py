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
  results on both sides of it; every triple of zeros, ones, a subnormal, the edges, infinities and NaN. Fixed rows
  with their bits: engineering review E3's two (a subnormal product of normal operands on a zero addend, and operands
  of at least 2^-100 cancelling to 2^-146, which D6's literal guard returns as the hardware's +0), the two exact
  products that round up to 2^-126, and the signs of zero products on +-0 addends (D10). The hardware's own ``fma``
  must differ from IEEE on many rows, so that the slow path is what the rows check.
- H.3: ``log1pf_nv`` (libdevice's ``__nv_log1pf``, dec_gdn.md section 6a) against ``basalt_emu.log1pf_nv``, which
  runs the whole sequence with IEEE subnormals and FADD.RZ through ``Fraction``: [2^-126, 2^-60] of both signs, some
  below 2^-125 (FMUL p, m, p subnormal); subnormals of both signs, which come back as they are; 64 ulps either side of
  0.75 * 2^n - 1 for n = -2..29, which holds every float32 x whose round-to-nearest x + 1 steps k where FADD.RZ's does
  not; [-1, e^20] by bits and by value, [-1, 1] by value; zeros, -1, x < -1, the largest finite values, infinities and
  NaNs. Fixed rows with their bits: 0x3EFFFFFF, the one float32 whose result FADD.RZ changes (a round-to-nearest sum
  gives 0x3ECF9920), the smallest subnormals (which this GPU's m + s would flush to +0), +-0, -1 and +inf. The results
  must also be within 1 ulp of float64's log1p rounded to float32 wherever x >= -1, which catches an error both
  transcriptions share.
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
         (0x00000000, 0x00000000, 0x80000000, 0x00000000)]   # 0 * 0 + -0 = +0


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
            (0x7FC00000, 0x7FFFFFFF)]


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
