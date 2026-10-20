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
"""

from fractions import Fraction

import numpy as np

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32, U32 = np.float32, np.uint32


def probe(src, name, *inputs):
    """The probe kernel ``name`` (``src``, appended to basalt_sampler_source(), which composes the helpers
    basalt_libdevice.metal needs ahead of it), one thread per element of the equal-length ``inputs`` (buffers 0, 1,
    ...): one 32-bit word each, starting as 0xFF bytes, then the element count at the next slot."""
    n = len(inputs[0])
    pso = gpu.pipeline(kernels.basalt_sampler_source() + src, name)
    bufs = [np.ascontiguousarray(v).tobytes() for v in inputs] + [4 * n]
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
