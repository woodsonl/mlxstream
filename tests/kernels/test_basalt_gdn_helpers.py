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


def assert_bits(name, got, want, x):
    """``got`` (float32 bits) is ``want`` bit for bit; a NaN matches a NaN of any payload."""
    want = np.asarray(want, F32).view(U32)
    bad = np.flatnonzero((nan32(got) != nan32(want)) | (~nan32(got) & (got != want)))
    assert bad.size == 0, (f"{name}: {bad.size} of {len(got)} differ; first x={x.view(U32)[bad[0]]:#010x}: "
                           f"got {got[bad[0]]:#010x}, want {want[bad[0]]:#010x}")


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
