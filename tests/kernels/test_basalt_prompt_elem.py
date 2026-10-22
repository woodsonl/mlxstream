# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/qsa_parity.cpp, src/kernels/rope_parity.cpp and
# src/kernels/shared_expert_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their
# pass criteria (qsa_parity.cpp:504-505 and :611, rope_parity.cpp:400 and :429, shared_expert_parity.cpp:372), used as
# the nearest bounds for kernels no basalt test launches, and their fixtures' forms.
"""basalt's prompt-path attention row kernels (src/prefill/kernels.cu:726-817: rms_rows, rope<false>, split_q,
gate_attn and kv_append's fp16 branch) on Metal (kernels/common/basalt_prompt_qsa.metal), and the libdevice functions
the rotation's angle calls (kernels/common/basalt_libdevice.metal), against host references. Every probe output starts
as 0xFF bytes, so an unwritten word fails its comparison. Each kernel compiles once per macro set at the bound B = 256
(ROWS_MAX) and runs at R tokens (StepState.t_this_step, basalt_prompt_gpu.run_rows) into an output of B + 1 rows of
0xFF bytes: the rows from token R on must keep them, no live word may be non-finite (but where a test feeds a
non-finite input) and the inputs must keep their bytes (TA 3.0).

- RMS-1 to RMS-5, ``prompt_rms_rows`` (kernels.cu:726-734) at basalt's call shapes (prefill.cpp:1610, :1642, :1645;
  draft_kv's :1006, :1010, :1023): against float64 at qsa_parity's bound (RMS-1); one token alone equal to its rows in
  a 256-token run and two runs equal, bitwise, and a run with StepState.done storing nothing (RMS-2, RMS-3); every
  word equal to the host twin basalt_emu.rms_rows_emu with this GPU's fast::rsqrt, on random rows, TA 3.1's IEEE rows
  (a)-(d) and hc's tie row, each of which a flushing kernel fails (RMS-4); the stored words of rows with an infinite
  or NaN element (RMS-5).
- RO-1 to RO-4, ``prompt_rope`` (kernels.cu:738-758, rope_kernel<false> at scaling None) at basalt's call shapes
  (prefill.cpp:1611, :1643, :1646; draft_kv's :1024): against float64 at rope_parity's bound to position 16383, the
  worst at 32512..32767 printed, not gated (RO-1); every word of every head equal to the host twin
  basalt_emu.rope_emu with the (c, s) the kernel writes on a probe head (a = 1, b = 0), on random rows, subnormal
  and tiny rows whose products or results fall into the subnormals, and NaN and infinite entries (RO-2); those (c, s)
  bitwise equal to the host twins of nv_cosf and nv_sinf of fl32(float(pos) nv_powf(0x3F1AB32B, pair)) at positions
  0..16383 and KE 8.2's three Payne-Hanek chunks (RO-3, the kernel's own angle); one token alone equal to its rows in
  a 256-token run, two runs equal and a StepState.done run storing nothing (RO-4).

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
from functools import partial

import numpy as np
import pytest

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu
from tests.kernels import basalt_prompt_gpu as pgpu

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


# ---- RMS-1 to RMS-5: prompt_rms_rows ---------------------------------------------------------------------------------

B = 256                                   # every pipeline's bound ROWS_MAX (TA 3.0)
EPS = F32(1e-6)                           # basalt's EPS (prefill.cpp:57) at every call
EPS_BITS = int(EPS.view(U32))             # the handler's EPS_BITS, 0x358637BD
RSQRT = partial(pgpu.hc_probe, "rsqrt")   # this GPU's fast::rsqrt: the twin's one approximate operation
# basalt's calls, all with ld = cols (prefill.cpp:1610, :1642, :1645; draft_kv's :1006, :1010, :1023): name -> (heads,
# cols, row counts R); a draft_kv batch of nb rows is R = nb tokens
RMS_SHAPES = {"A1": (2, 256, (1, 7, B)), "A5": (24, 256, (1, 7, B)), "A7": (4, 128, (1, 7, B)),
              "D3": (1, 2560, (1, 192)), "D6": (1, 10240, (1, 192)), "D15": (2, 256, (1, 192))}
RMS_GATE = ("nearest bound; no basalt test launches this kernel: qsa_parity.cpp:611 "
            'if (!(worst <= 1e-6)) { std::printf("    *** over 1e-6 ***\\n"); ++g_bad; }')


def rms_fixture(heads, cols):
    """TA 3.1's inputs from default_rng(611): x [B, heads * cols] with kernel row t heads + h ~ N(0, 1) 4^((t heads + h)
    % 4) (qsa_parity.cpp:548's cell magnitudes 1, 4, 16, 64), w [cols] ~ 1 + 0.1 N(0, 1) (:558), float32."""
    rng = np.random.default_rng(611)
    x = rng.standard_normal((B * heads, cols)) * 4.0 ** (np.arange(B * heads)[:, None] % 4)
    return x.astype(F32).reshape(B, heads * cols), (1 + 0.1 * rng.standard_normal(cols)).astype(F32)


def rms_rows(x, w, heads, R, done=0, finite=True):
    """prompt_rms_rows on x [B, heads * cols] and w [cols] (COLS, HEADS, EPS_BITS 1e-6f) at bound B with R tokens in
    StepState (and ``done``), B * heads threadgroups of 256, into y of B + 1 token rows of 0xFF bytes. Asserts that x
    and w kept their bytes, that every row from token R on (every row, with ``done``) kept the fill and, with
    ``finite``, that no live word is non-finite. Returns the live rows' words, uint32 [R, heads * cols]."""
    cols = len(w)
    pso = pgpu.prompt_pipeline(kernels.basalt_prompt_qsa_source(), "prompt_rms_rows", B,
                               {"COLS": str(cols), "HEADS": str(heads), "EPS_BITS": f"{EPS_BITS:#x}u"})
    ins = [np.ascontiguousarray(x, F32).tobytes(), np.ascontiguousarray(w, F32).tobytes()]
    out = pgpu.run_rows(pso, [*ins, 4 * (B + 1) * heads * cols], None, 0, (B * heads, 1, 1), (256, 1, 1), R, 0,
                        fill=0xFF, done=done)
    assert out[:2] == ins, f"{heads} x {cols}: an input's bytes changed"
    y = np.frombuffer(out[2], U32).reshape(B + 1, heads * cols)
    n = 0 if done else R
    assert (y[n:] == 0xFFFFFFFF).all(), f"{heads} x {cols} at {R} rows (done {done}): a row from token {n} on stored"
    bad = np.count_nonzero(~np.isfinite(y[:n].view(F32)))
    assert not (finite and bad), f"{heads} x {cols} at {R} rows: {bad} live words not finite"
    return y[:n]


def rms_double(x, w):
    """TA 3.1's float64 reference on the float32 inputs, per kernel row r: w[c] r[c] / sqrt(sum_c r[c]^2 / cols + eps),
    eps the float64 of 1e-6f. Returns x's shape."""
    r = np.asarray(x, np.float64).reshape(-1, len(w))
    ref = np.asarray(w, np.float64) * r / np.sqrt((r * r).sum(axis=1, keepdims=True) / len(w) + np.float64(EPS))
    return ref.reshape(np.shape(x))


def rms_twin(x, w, flush=False):
    """basalt_emu.rms_rows_emu of x's kernel rows with this GPU's fast::rsqrt (``flush``: its flushing form), x's
    shape as uint32 words."""
    return emu.rms_rows_emu(np.asarray(x, F32).reshape(-1, len(w)), w, EPS, RSQRT, flush).reshape(np.shape(x))


@pytest.mark.parametrize("name,R", [(n, R) for n, (_, _, counts) in RMS_SHAPES.items() for R in counts])
def test_rms_rows_vs_double(name, R):
    """RMS-1 (TA 3.1). Nearest bound; no basalt test launches this kernel: qsa_parity.cpp:611, ``if (!(worst <= 1e-6))
    { std::printf("    *** over 1e-6 ***\\n"); ++g_bad; }``, gates the table indexer's pooled keys
    (``indexer_key_append``: the key norm, the rotation and the pooling; docs/design/flash-next.md:503) against
    float64, its worst taken as rel_terms against the rotation's terms (:600-608). Here the bound takes TA 3.1's
    per-element measure, the qsa lane's for decode's norm: worst = max |got - ref| / max(|ref|, 1e-30) over the live
    rows, at every call shape (A1, A5, A7 at T in {1, 7, 256}; draft_kv's D3, D6, D15 at nb in {1, 192}). Prints the
    worst per shape."""
    heads, cols, _ = RMS_SHAPES[name]
    x, w = rms_fixture(heads, cols)
    got = rms_rows(x, w, heads, R).view(F32).astype(np.float64)
    ref = rms_double(x, w)[:R]
    worst = float((np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)).max())
    print(f"RMS-1 {name} ({R * heads} rows of {cols}): worst {worst:.3e}")
    assert worst <= 1e-6, f"{RMS_GATE} ({name}, {R} rows): worst {worst:.3e}"


@pytest.mark.parametrize("name", ["A1", "A5", "A7"])
def test_rms_rows_row_invariance(name):
    """RMS-2 (TA 3.1): one threadgroup per row and nothing shared between rows (the grid of host :1003), so token t
    alone (its rows copied to token 0, R = 1) equals token t's rows of the 256-token run bitwise, t in {0, 1, 255}."""
    heads, cols, _ = RMS_SHAPES[name]
    x, w = rms_fixture(heads, cols)
    full = rms_rows(x, w, heads, B)
    for t in (0, 1, B - 1):
        xt = x.copy()
        xt[0] = x[t]
        assert np.array_equal(rms_rows(xt, w, heads, 1)[0], full[t]), f"{name}: token {t} alone differs from its rows"


@pytest.mark.parametrize("name", ["A1", "A5", "A7"])
def test_rms_rows_repeat(name):
    """RMS-3 (TA 3.1): two runs on the same inputs store the same bits; and a run with StepState.done = 1 stores
    nothing (every row keeps the fill: the guard ahead of the first barrier)."""
    heads, cols, _ = RMS_SHAPES[name]
    x, w = rms_fixture(heads, cols)
    assert np.array_equal(rms_rows(x, w, heads, B), rms_rows(x, w, heads, B)), f"{name}: two runs differ"
    rms_rows(x, w, heads, B, done=1)


@pytest.mark.parametrize("name", list(RMS_SHAPES))
def test_rms_rows_emulation_bit_for_bit(name):
    """RMS-4 (TA 3.1, TA 3.0 (iii)): every live word at each of the shape's row counts equal to the host twin,
    basalt_emu.rms_rows_emu, whose rsqrt is this GPU's fast::rsqrt of the argument fl(fl(S / cols) + eps) the twin
    computes. A1's kernel rows 0-3 are TA 3.1's IEEE rows at cols 256: (a) r ~ 1e-20 N(0, 1) with entries 1e-42 and
    -1e-42 (subnormal inputs; a subnormal mean square, absorbed by eps; s ~ 1000, so s r is subnormal there), (b) r ~
    1e4 N(0, 1) with entries 1e-35 and -3e-35 (s ~ 1e-4, s r subnormal), (c) a row of ones under w[9] = 1e-39 (a
    subnormal operand of the second product, s r near 1), (d) a row of +0 and -0 under a negated w[10] (every output
    the zero with the sign of r w; s finite). D3's token 0 (cols 2560) is test_basalt_prompt_hc.py's E6 tie row: r[0]
    = 2^-70, r[256] = 1 + 2^-12, the rest 0, so thread 0's subnormal first square breaks the tie of its second fma.
    RMS-1's 1e-30 floor passes a flushed subnormal, so this is their only test: the flushing twin (every operation's
    inputs and result flushed) must differ from the IEEE twin in each of rows (a), (b), (c) and the tie row."""
    heads, cols, counts = RMS_SHAPES[name]
    x, w = rms_fixture(heads, cols)
    rows, planted = x.reshape(-1, cols), []           # the kernel's rows, a view of x
    if name == "A1":
        rng = np.random.default_rng(733)
        rows[0] = rng.standard_normal(cols) * 1e-20                           # (a)
        rows[0, 5:7] = (1e-42, -1e-42)
        rows[1] = rng.standard_normal(cols) * 1e4                             # (b)
        rows[1, 5:7] = (1e-35, -3e-35)
        rows[2], w[9] = 1, 1e-39                                              # (c)
        rows[3], w[10] = 0, -w[10]                                            # (d)
        rows[3, 1::2] = -0.0
        planted = [0, 1, 2]
        zero = np.where(np.signbit(rows[3]) ^ np.signbit(w), 0x80000000, 0)
        assert np.array_equal(rms_twin(x, w).reshape(-1, cols)[3], zero), "(d): the twin's row is not r w's zeros"
    if name == "D3":
        rows[0] = 0                                                           # hc's E6
        rows[0, 0], rows[0, 256] = 2.0 ** -70, 1 + 2.0 ** -12
        planted = [0]
    want = rms_twin(x, w)
    for R in counts:
        bad = np.argwhere(rms_rows(x, w, heads, R) != want[:R])
        assert not len(bad), f"{name} at {R} rows: {len(bad)} words differ from the twin, the first at {bad[0]}"
    flushed, want = rms_twin(x, w, flush=True).reshape(-1, cols), want.reshape(-1, cols)
    for k in planted:
        assert (flushed[k] != want[k]).any(), f"{name} row {k}: the flushing twin's words equal the IEEE twin's"


def test_rms_rows_nonfinite():
    """RMS-5 (TA 3.1), at A1's shape and 256 tokens: kernel row 3 with r[7] = +inf has ss = inf and s =
    fast::rsqrt(inf) = +0, so it stores 0 * inf = NaN as cuda_nan's 0x7fffffff at column 7 and the zero with the sign
    of r w everywhere else; row 4 with r[9] the NaN payload 0x7fa00001 stores 0x7fffffff in every column; every other
    row stores the words of a run without the two entries."""
    heads, cols, _ = RMS_SHAPES["A1"]
    x, w = rms_fixture(heads, cols)
    clean = rms_rows(x, w, heads, B).reshape(-1, cols)
    rows = x.reshape(-1, cols)
    rows[3, 7] = np.inf
    rows.view(U32)[4, 9] = 0x7FA00001
    got = rms_rows(x, w, heads, B, finite=False).reshape(-1, cols)
    want = np.where(np.signbit(rows[3]) ^ np.signbit(w), 0x80000000, 0).astype(U32)
    want[7] = 0x7FFFFFFF
    assert np.array_equal(got[3], want), f"+inf row: {np.flatnonzero(got[3] != want).size} words differ"
    assert (got[4] == 0x7FFFFFFF).all(), f"NaN row: {np.unique(got[4])}"
    keep = np.ones(len(got), bool)
    keep[[3, 4]] = False
    assert np.array_equal(got[keep], clean[keep]), "a row without a non-finite element changed"


# ---- RO-1 to RO-4: prompt_rope ---------------------------------------------------------------------------------------

# basalt's calls, all with ld = heads * dim (prefill.cpp:1611, :1643, :1646; draft_kv's :1024): name -> (heads, dim,
# row counts R); a draft_kv batch of nb tokens is R = nb
ROPE_SHAPES = {"A2": (2, 256, (1, 7, B)), "A6": (24, 256, (1, 7, B)), "A8": (4, 128, (1, 7, B)),
               "D16": (2, 256, (1, 192))}
ROPE_GATE = "nearest bound; no basalt test launches this kernel: rope_parity.cpp:400 if (!(rel <= 3e-3)) ++agree_bad;"


def rope_fixture(heads, dim, n=B):
    """TA 3.2's inputs from default_rng(400): x [n, heads * dim] ~ N(0, 1) (rope_parity.cpp:353-367's rows), float32."""
    return np.random.default_rng(400).standard_normal((n, heads * dim)).astype(F32)


def rope(x, heads, dim, R, p0, done=0, finite=True):
    """prompt_rope on x [B, heads * dim] (HEADS, DIM, THETA_SCALE_BITS 0x3F1AB32B) at bound B with R tokens from
    position p0 in StepState (and ``done``), B * heads threadgroups of 32, into y of B + 1 token rows of 0xFF bytes.
    Asserts that x kept its bytes, that every row from token R on (every row, with ``done``) kept the fill and, with
    ``finite``, that no live word is non-finite. Returns the live rows' words, uint32 [R, heads, dim]."""
    pso = pgpu.prompt_pipeline(kernels.basalt_prompt_qsa_source(), "prompt_rope", B,
                               {"HEADS": str(heads), "DIM": str(dim), "THETA_SCALE_BITS": f"{THETA_SCALE:#x}u"})
    xb = np.ascontiguousarray(x, F32).tobytes()
    out = pgpu.run_rows(pso, [xb, 4 * (B + 1) * heads * dim], None, 0, (B * heads, 1, 1), (32, 1, 1), R, p0,
                        fill=0xFF, done=done)
    assert out[0] == xb, f"{heads} x {dim}: x's bytes changed"
    y = np.frombuffer(out[1], U32).reshape(B + 1, heads, dim)
    n = 0 if done else R
    assert (y[n:] == 0xFFFFFFFF).all(), f"{heads} x {dim} at {R} rows from {p0} (done {done}): token {n} on stored"
    bad = np.count_nonzero(~np.isfinite(y[:n].view(F32)))
    assert not (finite and bad), f"{heads} x {dim} at {R} rows from {p0}: {bad} live words not finite"
    return y[:n]


def test_rope_kernel_angle_bit_for_bit():
    """RO-3 (TA 3.2, KE 8.2), the kernel's own angle: at A2's shape head 0 of every token is a probe row (a = 1, b = 0
    in all 32 pairs), on which prompt_rope stores y[pair] = fma(c, 1, -(0 s)) = c and y[pair + 32] = fma(c, 0, 1 s) =
    s exactly; both equal, bitwise, the host twins nv_cosf(th) and nv_sinf(th) of th = fl32(float(pos)
    nv_powf(0x3F1AB32B, pair)) (basalt_emu.nv_trig and basalt_emu.nv_powf), every pair, at positions 0..16383 (64 runs
    of 256 tokens at p0 = 256 k) and KE 8.2's three Payne-Hanek chunks (p0 = 105600, 174720 and 261888, the last 256
    positions of the trained context of 262,144, rope_scaling.hpp:72), whose 1,211 angles of at least 105615 take
    nv_trig_slow. So it holds what only the kernel decides (p0 + t read from StepState, float() of the position,
    nv_powf's arguments, nv_cosf and nv_sinf rather than another cosine), which RO-1's 3e-3 would not see;
    test_rope_angle_bit_for_bit (RO-3h) holds the helpers alone. Every word of both heads equals basalt_emu.rope_emu
    with the twins' (c, s): the random head 1 is the rotation's check at the Payne-Hanek positions."""
    heads, dim = ROPE_SHAPES["A2"][:2]
    x = rope_fixture(heads, dim).reshape(B, heads, dim)
    x[:, 0, :32], x[:, 0, 32:64] = 1, 0                                     # the probe head
    p0s = [256 * k for k in range(64)] + list(SLOW_CHUNKS)
    pos = np.array(p0s)[:, None] + np.arange(B)                             # (run, token)
    th = pos.astype(F32)[..., None] * emu.nv_powf(U32(THETA_SCALE).view(F32), np.arange(32, dtype=F32))
    assert int((th.view(U32) >= SLOW_MIN).sum()) == 1211
    c, s = emu.nv_trig(th, 1), emu.nv_trig(th, 0)
    for k, p0 in enumerate(p0s):
        got = rope(x.reshape(B, -1), heads, dim, B, p0)
        bad = np.argwhere(got[:, 0, :64] != np.concatenate([c[k], s[k]], axis=-1).view(U32))
        assert not len(bad), (f"p0 {p0}: {len(bad)} probe words differ from the twins' (c, s); the first at token "
                              f"{bad[0][0]}, {'s' if bad[0][1] >= 32 else 'c'} of pair {bad[0][1] % 32}")
        bad = np.argwhere(got != emu.rope_emu(x, c[k][:, None], s[k][:, None]))
        assert not len(bad), f"p0 {p0}: {len(bad)} words differ from the twin, the first at {bad[0]}"


@pytest.mark.parametrize("name", ["A2", "A6", "A8"])
def test_rope_rotation_bit_for_bit(name):
    """RO-2 (TA 3.2, D25): the rotation's two FFMA sites, y[pair] = fma(c, a, -fl(b s)) and y[pair + 32] = fma(c, b,
    fl(a s)) (the SASS's FMUL, then FFMA), bitwise. Head 0 of every token is the probe row (a = 1, b = 0), so the
    kernel's stores there are its own (c, s) (RO-3 holds them to the twins); every word of every head equals
    basalt_emu.rope_emu with those (c, s), over 256 tokens from position 0: random rows and, in head 1 at positions
    1..32 (where s is not +0), TA 3.2's IEEE rows (i) a of 1e-40 N(0, 1) (subnormal) beside b = 0, so that c a and a s
    are the results, (ii) b of 1e-40 N(0, 1) beside a = 0 and (iii) a and b of 1e-36 N(0, 1), whose b s falls subnormal
    at a small s, D25's (iv) a and b of 1e-38 N(0, 1), whose results round into the subnormals, and a subnormal and a
    -0 among the copied dims. The flushing twin (every
    operation's inputs and result flushed) must differ from the IEEE twin in each of (i)-(iv). A second run plants
    D25's NaN and infinite entries: a NaN a or b stores cuda_nan's 0x7fffffff at both words of its pair, an infinite b
    times s = +0 (position 0) a NaN 0x7fffffff, the others the twin's words; NaN payloads among the copied dims keep
    their bits."""
    heads, dim, _ = ROPE_SHAPES[name]
    x = rope_fixture(heads, dim).reshape(B, heads, dim)
    x[:, 0, :32], x[:, 0, 32:64] = 1, 0                                     # the probe head
    rng = np.random.default_rng(756)
    groups = {"(i)": (1e-40, 0), "(ii)": (0, 1e-40), "(iii)": (1e-36, 1e-36), "(iv)": (1e-38, 1e-38)}
    for g, (sa, sb) in enumerate(groups.values()):                          # tokens 1-8, 9-16, 17-24, 25-32
        x[1 + 8 * g: 9 + 8 * g, 1, :32] = rng.standard_normal((8, 32)) * sa
        x[1 + 8 * g: 9 + 8 * g, 1, 32:64] = rng.standard_normal((8, 32)) * sb
    x[33, 1, 64:66] = (1e-40, -0.0)
    got = rope(x.reshape(B, -1), heads, dim, B, 0)
    c, s = got[:, :1, :32].view(F32), got[:, :1, 32:64].view(F32)          # per token, for every head
    want = emu.rope_emu(x, c, s)
    bad = np.argwhere(got != want)
    assert not len(bad), f"{name}: {len(bad)} words differ from the twin, the first at {bad[0]}"
    flushed = emu.rope_emu(x, c, s, flush=True)
    for g, label in enumerate(groups):
        t = slice(1 + 8 * g, 9 + 8 * g)
        assert (flushed[t, 1] != want[t, 1]).any(), f"{name} {label}: the flushing twin's words equal the IEEE twin's"
    v = x.view(U32)
    v[0, 1, 0], v[3, 1, 34] = 0x7FA00001, 0xFFC00005                        # NaN a at position 0, NaN b at 3
    x[0, 1, 37], x[0, 1, 9] = np.inf, -np.inf                               # inf b where s = +0, -inf a
    x[3, 1, 4], x[3, 1, 38] = np.inf, -np.inf
    v[5, 1, 100], v[5, 1, 70] = 0x7FA00002, 0xFF800001                      # copied NaN payloads
    got = rope(x.reshape(B, -1), heads, dim, B, 0, finite=False)
    assert np.array_equal(got[:, :1, :64].view(F32), np.concatenate([c, s], axis=-1)), f"{name}: (c, s) moved"
    bad = np.argwhere(got != emu.rope_emu(x, c, s))
    assert not len(bad), f"{name}: {len(bad)} words differ from the twin with NaN and inf entries, the first at {bad[0]}"
    assert (got[0, 1, [0, 32, 5]] == 0x7FFFFFFF).all() and (got[3, 1, [2, 34]] == 0x7FFFFFFF).all(), "a NaN's words"
    assert got[5, 1, 100] == 0x7FA00002 and got[5, 1, 70] == 0xFF800001, f"{name}: a copied NaN lost its payload"


@pytest.mark.parametrize("name", ["A2", "A6", "A8"])
def test_rope_row_invariance(name):
    """RO-4 (TA 3.2): one threadgroup per (token, head) and nothing shared between them (the grid of host :1021-1023),
    so token t alone (its rows copied to token 0, R = 1 from position p0 + t) equals token t's rows of the 256-token
    run from p0, bitwise, t in {0, 1, 255}, at p0 = 105600 (pair 0 takes the Payne-Hanek path from token 15 on, so
    both paths run); dims 64.. of every head keep x's bits; a second run stores the same bits; a run with
    StepState.done = 1 stores nothing."""
    heads, dim, _ = ROPE_SHAPES[name]
    x = rope_fixture(heads, dim)
    p0 = SLOW_CHUNKS[0]
    full = rope(x, heads, dim, B, p0)
    for t in (0, 1, B - 1):
        xt = x.copy()
        xt[0] = x[t]
        assert np.array_equal(rope(xt, heads, dim, 1, p0 + t)[0], full[t]), f"{name}: token {t} alone differs"
    assert np.array_equal(full[..., 64:], x.view(U32).reshape(B, heads, dim)[..., 64:]), f"{name}: a copied dim moved"
    assert np.array_equal(rope(x, heads, dim, B, p0), full), f"{name}: two runs differ"
    rope(x, heads, dim, B, p0, done=1)


def rope_worst(got, x, pos):
    """TA 3.2's measure of the words ``got`` [T, heads, dim] for the inputs ``x`` [T, heads, dim] at the positions
    ``pos`` [T], against float64: theta = pos 1e7^(-2 pair / 64) exactly (rope_parity's ang = p * pow(theta, -2.0 * i /
    n_rot), :67-84), ref[pair] = a cos - b sin and ref[pair + 32] = a sin + b cos, the other dims x; rel = |got - ref| /
    max(max_d |x|, 1e-30) for each (token, head) row of dim values (rope_parity.cpp:392-399). Returns the worst rel."""
    xr = np.asarray(x, np.float64)
    th = np.asarray(pos, np.float64)[:, None, None] * 1e7 ** (-2 * np.arange(32) / 64)
    a, b = xr[..., :32], xr[..., 32:64]
    ref = xr.copy()
    ref[..., :32], ref[..., 32:64] = a * np.cos(th) - b * np.sin(th), a * np.sin(th) + b * np.cos(th)
    scale = np.maximum(np.abs(xr).max(axis=-1, keepdims=True), 1e-30)
    return float((np.abs(got.view(F32).astype(np.float64) - ref) / scale).max())


ROPE_GATED = 16384     # positions below it are gated: one maximal chunk (m.T <= 16384, prefill.cpp:1251-1253)
ROPE_RUNS = ([(n, R, 0) for n, (_, _, counts) in ROPE_SHAPES.items() for R in counts]
             + [(n, B, p0) for p0 in (16128, 32512) for n in ("A2", "A6", "A8")])


@pytest.mark.parametrize("name,R,p0", ROPE_RUNS)
def test_rope_vs_double(name, R, p0):
    """RO-1 (TA 3.2). Nearest bound; no basalt test launches this kernel: rope_parity.cpp:400, ``if (!(rel <= 3e-3))
    ++agree_bad;`` (and :429, the same rule at scaling None), gates basalt's fast-math native rotation against its
    float64 table path, rel = |got - ref| / the row's largest |x|. Here against float64 at the exact angle, at every
    call shape (A2, A6, A8 at T in {1, 7, 256} from position 0 and at 256 tokens from 16128; draft_kv's D16 at nb in
    {1, 192}), gated at positions up to 16383: the float32 angle's error grows with the position, so the bound can
    fail past about 19,000 (TA 3.2), and the worst over 32512..32767 is printed, not gated. Dims 64.. keep x's bits."""
    heads, dim, _ = ROPE_SHAPES[name]
    xb = rope_fixture(heads, dim)
    got, x = rope(xb, heads, dim, R, p0), xb[:R].reshape(R, heads, dim)
    assert np.array_equal(got[..., 64:], x.view(U32)[..., 64:]), f"{name}: a copied dim moved"
    worst = rope_worst(got, x, p0 + np.arange(R))
    gated = p0 + R <= ROPE_GATED
    print(f"RO-1 {name} ({R} tokens from {p0}): worst {worst:.3e}{'' if gated else ', recorded, not gated'}")
    assert worst <= 3e-3 or not gated, f"{ROPE_GATE} ({name}, {R} tokens from {p0}): worst {worst:.3e}"


@pytest.mark.parametrize("name", ["A2", "A6", "A8"])
def test_rope_vs_double_r5_positions(name):
    """RO-1 (TA 3.2) at rope_parity R5's positions (r * 127) % 2048, r = 0..383 (rope_parity.cpp:368-369), one token
    per run (R = 1 at that position; token 0 the r-th of 384 rows from default_rng(400)), held to
    test_rope_vs_double's bound. (D16 has A2's shape, so A2's pipeline.)"""
    heads, dim, _ = ROPE_SHAPES[name]
    rows = rope_fixture(heads, dim, 384).reshape(384, 1, heads, dim)
    x = rope_fixture(heads, dim)
    worst = 0.0
    for r in range(384):
        x[0] = rows[r].reshape(-1)
        worst = max(worst, rope_worst(rope(x, heads, dim, 1, r * 127 % 2048), rows[r], [r * 127 % 2048]))
    print(f"RO-1 {name} (R5's 384 positions): worst {worst:.3e}")
    assert worst <= 3e-3, f"{ROPE_GATE} ({name}, R5's positions): worst {worst:.3e}"
