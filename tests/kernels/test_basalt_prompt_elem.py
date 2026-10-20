# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/qsa_parity.cpp, src/kernels/rope_parity.cpp and
# src/kernels/shared_expert_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their
# pass criteria (qsa_parity.cpp:504-505 and :611, rope_parity.cpp:400 and :429, shared_expert_parity.cpp:372), used as
# the nearest bounds for kernels no basalt test launches, and their fixtures' forms.
"""basalt's prompt-path attention row kernels (src/prefill/kernels.cu:726-817: rms_rows, rope<false>, split_q,
gate_attn and kv_append's fp16 branch) on Metal (kernels/common/basalt_prompt_qsa.metal), and the libdevice functions
the rotation's angle calls (kernels/common/basalt_libdevice.metal), against host references. Every probe output starts
as 0xFF bytes, so an unwritten word fails its comparison.

- RO-5: ``nv_powf(0x3F1AB32B, pair)``, pairs 0..31 (this pack's theta_scale, ``powf(1e7f, -2.0f / 64.0f)``), from a
  probe kernel and from the host twin (basalt_emu.nv_powf), bitwise equal to the 32 words
  docs/research/basalt-nvcc/prefill_kernels.md section 4 records for nvcc 13.0's ``__nv_powf``; and in the same
  launch every pair at theta_scale values across the rope handler's domain [2^(-125/31), 1], bitwise equal to the twin.
- RO-3h, the slow path: ``nv_trig_slow``, the Payne-Hanek reduction of ``__nv_cosf`` and ``__nv_sinf`` in soft binary64,
  from a probe kernel, bitwise equal in r and j to the host twin (basalt_emu.nv_trig_slow) at every angle of KE 8.2's
  three Payne-Hanek chunks that takes it and across the function's domain; and its binary64 conversion
  ``f64_from_u63`` against Python's correctly rounded int-to-float.
- RO-3h: the rotation's angle ``float(pos) * nv_powf(0x3F1AB32B, pair)`` and its ``nv_cosf`` and ``nv_sinf`` from a
  probe kernel, bitwise equal to the host twins (basalt_emu.nv_powf, basalt_emu.nv_trig) at positions 0..16383 and
  the three Payne-Hanek chunks, pairs 0..31; and both functions at NaN, +-inf, +-0, subnormals and across the float
  range, both signs.
"""

from fractions import Fraction

import numpy as np

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32, U32 = np.float32, np.uint32


def probe(src, name, *inputs, words=1):
    """The probe kernel ``name`` (``src``, appended to basalt_sampler_source(), which composes the helpers
    basalt_libdevice.metal needs ahead of it), one thread per element of the equal-length ``inputs`` (buffers 0, 1,
    ...): an output of ``words`` 32-bit words each, starting as 0xFF bytes, then the element count at the next slot."""
    n = len(inputs[0])
    pso = gpu.pipeline(kernels.basalt_sampler_source() + src, name)
    bufs = [np.ascontiguousarray(v).tobytes() for v in inputs] + [4 * words * n]
    out = gpu.run(pso, bufs, U32(n).tobytes(), len(inputs) + 1, -(-n // 256), 256, fill=0xFF)
    return np.frombuffer(out[-1], U32)


# ---- RO-5: nv_powf ---------------------------------------------------------------------------------------------------

POWF = """
kernel void probe_nv_powf(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                          device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = as_type<uint>(nv_powf(x[i], y[i]));
}
"""

# nv_powf(0x3F1AB32B, pair), pairs 0..31: nvcc 13.0's __nv_powf (prefill_kernels.md section 4; KE 8.2 (a))
POWF_VECTOR = [0x3F800000, 0x3F1AB32B, 0x3EBAF81A, 0x3E61F835, 0x3E088D77, 0x3DA50956, 0x3D47763E, 0x3CF11175,
               0x3C91AD38, 0x3C301051, 0x3BD4CA14, 0x3B80967D, 0x3B1B690C, 0x3ABBD3EB, 0x3A6301E0, 0x3A092E01,
               0x39A5CB5E, 0x394860BF, 0x38F22CE1, 0x3892587D, 0x3830DF50, 0x37D5C43F, 0x37812DAA, 0x371C1FC2,
               0x36BCB0BF, 0x36640CC4, 0x3609CF49, 0x35A68E4A, 0x35494C54, 0x34F34999, 0x3493048C, 0x3431AF42]
THETA_SCALE = 0x3F1AB32B
LOWEST_THETA_SCALE = 0x3D7A56E8          # the least float32 x with x^31 >= 2^-125: the domain's bottom, 2^(-125/31)


def test_nv_powf_vector():
    """RO-5 (KE 8.2 (a)): the 32 recorded words pin the transcription and, at this theta_scale, the reciprocal's
    mapping (no pair's result depends on rcp.approx.ftz's last bits there). Elsewhere in the domain the reciprocal is
    invariant 5's mapping on both sides, so the GPU must equal the twin bit for bit: at the domain's lowest float32
    (pair 31's power just above 2^-126), the binade split of m (0x3F3504F3), the powers of two (m = 1), 1 - 2^-24, 1,
    and random values."""
    lo = LOWEST_THETA_SCALE
    power31 = [Fraction(float(U32(w).view(F32))) ** 31 for w in (lo - 1, lo)]
    assert power31[0] < Fraction(1, 2 ** 125) <= power31[1]
    edges = [THETA_SCALE, lo, lo + 1, 0x3D800000, 0x3E000000, 0x3E800000, 0x3F000000, 0x3F3504F2, 0x3F3504F3,
             0x3F3504F4, 0x3EB504F3, 0x3E3504F3, 0x3F7FFFFF, 0x3F800000]
    xs = np.r_[np.array(edges, U32), np.random.default_rng(751).integers(lo, 0x3F800000, 512, U32, endpoint=True)]
    pairs = np.arange(32, dtype=F32)
    x, y = np.repeat(xs.view(F32), 32), np.tile(pairs, len(xs))
    got = probe(POWF, "probe_nv_powf", x, y)
    want = np.array(POWF_VECTOR, U32)
    assert np.array_equal(got[:32], want), [f"{p}: {g:#010x} != {w:#010x}" for p, (g, w) in
                                            enumerate(zip(got[:32], want)) if g != w]
    assert np.array_equal(emu.nv_powf(U32(THETA_SCALE).view(F32), pairs).view(U32), want)
    twin = emu.nv_powf(x, y).view(U32)
    bad = np.flatnonzero(got != twin)
    assert bad.size == 0, (f"{bad.size} of {len(x)} differ; first x={x.view(U32)[bad[0]]:#010x} y={y[bad[0]]}: "
                           f"got {got[bad[0]]:#010x}, twin {twin[bad[0]]:#010x}")
    bottom = twin[32 + 31].view(F32)                                 # pair 31 at the lowest theta_scale
    assert 2.0 ** -126 <= bottom < 2.0 ** -124


# ---- RO-3h, the slow path: nv_trig_slow ------------------------------------------------------------------------------

TRIG_SLOW = """
kernel void probe_nv_trig_slow(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                               constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  int j;
  out[2 * i] = as_type<uint>(nv_trig_slow(x[i], j));
  out[2 * i + 1] = uint(j);
}

kernel void probe_f64_from_u63(device const ulong* v [[buffer(0)]], device ulong* out [[buffer(1)]],
                               constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[i] = f64_from_u63(v[i]);
}
"""

SLOW_CHUNKS = (105600, 174720, 261888)    # KE 8.2's Payne-Hanek chunks of 256 positions; the last ends at 262,143
SLOW_MIN = 0x47CE4780                     # 105615.0f: an |x| below it (setp.ltu) stays off the slow path


def test_rope_angle_slow_path_bit_for_bit():
    """RO-3h, the slow path (KE 8.2 (b)): nv_trig_slow from a probe kernel, r and j bitwise equal to the host twin's,
    at every angle th = fl32(float(pos) * nv_powf(0x3F1AB32B, pair)) of at least 105615 for pos in [p0, p0 + 256),
    p0 = 105600, 174720 and 261888 (the last chunk of the trained context, rope_scaling.hpp:72), pairs 0..31: pair 0
    from position 105615, pair 1 from 174,774, pair 2 never (95,728 at 262,143). In the same launch, the function's
    whole domain, finite |x| >= 105615, of which the rope's angles (below 2^18) take only the window's word offset
    k = 0 and bit offsets 15 and 16: both signs of every binade from 2^16 to 2^127 at four fixed and four random
    significands (k = 0..3, and the bit offset 0 at 2^33, 2^65 and 2^97), the domain's bottom and random words.
    Its conversion f64_from_u63 is checked alone, since the bits above seldom show its rounding: against Python's
    int-to-float (nearest even) at 0, every power of two, ties both ways, 2^63 - 1 (the carry to 2^63) and random
    integers of every length."""
    ts = U32(THETA_SCALE).view(F32)
    pos = (np.array(SLOW_CHUNKS)[:, None] + np.arange(256)).astype(F32)
    th = pos[:, :, None] * emu.nv_powf(ts, np.arange(32, dtype=F32))       # (chunk, row, pair)
    slow = th.view(U32) >= SLOW_MIN
    assert slow.sum(axis=1)[:, :3].tolist() == [[241, 0, 0], [256, 202, 0], [256, 256, 0]] and slow.sum() == 1211
    rng = np.random.default_rng(161)
    sig = np.r_[np.array([0, 1, 0x400000, 0x7FFFFF], U32), rng.integers(0, 1 << 23, 4, U32)]
    sweep = ((np.arange(143, 255, dtype=U32) << 23)[:, None] | sig).reshape(-1)
    sweep = np.r_[U32(SLOW_MIN), sweep[sweep >= SLOW_MIN], rng.integers(SLOW_MIN, 0x7F800000, 1024, U32)]
    x = np.r_[th[slow], np.r_[sweep, sweep | 0x80000000].view(F32)]
    got = probe(TRIG_SLOW, "probe_nv_trig_slow", x, words=2).reshape(-1, 2)
    want = np.array([[F32(r).view(U32), j & 0xFFFFFFFF] for r, j in map(emu.nv_trig_slow, x)], U32)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert bad.size == 0, (f"{bad.size} of {len(x)} differ; first x={x.view(U32)[bad[0]]:#010x}: got (r, j) = "
                           f"({got[bad[0], 0]:#010x}, {got[bad[0], 1]:#x}), twin ({want[bad[0], 0]:#010x}, "
                           f"{want[bad[0], 1]:#x})")
    ints = [0, (1 << 53) + 1, (1 << 53) + 3, (1 << 62) + (1 << 9), (1 << 62) + 3 * (1 << 9),
            (1 << 62) + (1 << 9) + 1, (1 << 63) - 1] + [1 << k for k in range(63)]
    ints += [int(v) >> (63 - k) | 1 << (k - 1) for k in range(1, 64) for v in rng.integers(0, 1 << 63, 8, np.uint64)]
    got = probe(TRIG_SLOW, "probe_f64_from_u63", np.array(ints, np.uint64), words=2).view(np.uint64)
    want = np.array([float(v) for v in ints]).view(np.uint64)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (f"{bad.size} of {len(ints)} differ; first n={ints[bad[0]]:#x}: got {got[bad[0]]:#018x}, "
                           f"want {want[bad[0]]:#018x}")


# ---- RO-3h: nv_cosf and nv_sinf of the rotation's angle --------------------------------------------------------------

TRIG = f"#define THETA_SCALE_BITS {THETA_SCALE:#x}u\n" + """
kernel void probe_rope_angle(device const int* pos [[buffer(0)]], device const int* pair [[buffer(1)]],
                             device uint* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float th = float(pos[i]) * nv_powf(as_type<float>(THETA_SCALE_BITS), float(pair[i]));
  out[3 * i] = as_type<uint>(th);
  out[3 * i + 1] = as_type<uint>(nv_cosf(th));
  out[3 * i + 2] = as_type<uint>(nv_sinf(th));
}

kernel void probe_nv_trig(device const float* x [[buffer(0)]], device uint* out [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  out[2 * i] = as_type<uint>(cuda_nan(nv_cosf(x[i])));
  out[2 * i + 1] = as_type<uint>(cuda_nan(nv_sinf(x[i])));
}
"""


def test_rope_angle_bit_for_bit():
    """RO-3h (KE 8.2): the rotation's angle th = float(pos) * nv_powf(0x3F1AB32B, float(pair)) and its nv_cosf and
    nv_sinf from a probe kernel, bitwise equal to the host twins (th = fl32(float(pos) * basalt_emu.nv_powf(...)), then
    basalt_emu.nv_trig) at positions 0..16383 (one maximal chunk) and KE 8.2's three Payne-Hanek chunks of 256 (p0 =
    105600, 174720, 261888: the 1,211 slow angles of the slow-path test), pairs 0..31. Then both functions at the
    arguments no angle takes, from a second probe that stores them through cuda_nan, as a kernel does: NaNs of several
    payloads and +-inf (KE 5.2.2's two cases: a NaN stays on the fast path, an infinity gives NaN; CUDA's 0x7fffffff
    for both functions), +-0 and subnormals (cos 1, sin the argument: the port addition), and the whole float range in
    both signs, every binade at four fixed and four random significands and random words, so that the negative
    quadrants and the tiny normals (r * r subnormal) run too."""
    ts = U32(THETA_SCALE).view(F32)
    pos = np.r_[np.arange(16384), (np.array(SLOW_CHUNKS)[:, None] + np.arange(256)).reshape(-1)].astype(np.int32)
    pos, pair = np.repeat(pos, 32), np.tile(np.arange(32, dtype=np.int32), len(pos))
    got = probe(TRIG, "probe_rope_angle", pos, pair, words=3).reshape(-1, 3)
    th = pos.astype(F32) * emu.nv_powf(ts, pair.astype(F32))
    assert int((th.view(U32) >= SLOW_MIN).sum()) == 1211
    want = np.stack([th.view(U32), emu.nv_trig(th, 1).view(U32), emu.nv_trig(th, 0).view(U32)], axis=1)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert bad.size == 0, (f"{bad.size} of {len(pos)} differ; first pos={pos[bad[0]]} pair={pair[bad[0]]}: got "
                           f"(th, c, s) = {[f'{w:#010x}' for w in got[bad[0]]]}, twin "
                           f"{[f'{w:#010x}' for w in want[bad[0]]]}")
    rng = np.random.default_rng(162)
    special = np.array([0x7FC00000, 0xFFC00001, 0x7FA5A5A5, 0x7F800001, 0x7F800000, 0xFF800000,
                        0, 0x80000000, 1, 0x80000001, 0x00400000, 0x007FFFFF, 0x807FFFFF], U32)
    sig = np.r_[np.array([0, 1, 0x400000, 0x7FFFFF], U32), rng.integers(0, 1 << 23, 4, U32)]
    sweep = ((np.arange(255, dtype=U32) << 23)[:, None] | sig).reshape(-1)          # every binade, zeros included
    sweep = np.r_[sweep, rng.integers(0, 0x7F800000, 2048, U32)]
    x = np.r_[special, sweep, sweep | 0x80000000].view(F32)
    got = probe(TRIG, "probe_nv_trig", x, words=2).reshape(-1, 2)
    want = np.stack([emu.nv_trig(x, 1).view(U32), emu.nv_trig(x, 0).view(U32)], axis=1)
    assert (want[:6] == 0x7FFFFFFF).all()                                 # NaNs and infinities: NaN, both functions
    assert (want[6:13, 0] == 0x3F800000).all() and (want[6:13, 1] == special[6:]).all()   # zeros, subnormals
    bad = np.flatnonzero((got != want).any(axis=1))
    assert bad.size == 0, (f"{bad.size} of {len(x)} differ; first x={x.view(U32)[bad[0]]:#010x}: got (c, s) = "
                           f"({got[bad[0], 0]:#010x}, {got[bad[0], 1]:#010x}), twin ({want[bad[0], 0]:#010x}, "
                           f"{want[bad[0], 1]:#010x})")
