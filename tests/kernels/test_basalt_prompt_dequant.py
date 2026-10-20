# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dequant_f16_parity.cpp, src/kernels/dequant_bf16_test.cpp and
# src/kernels/elementwise_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their
# fixtures (the std::mt19937(7) blocks with pinned scales, the pack's tensors, the NaN-scale Q8_0 pair) and criteria.
"""basalt's group dequantizers on Metal (kernels/common/basalt_dequant16.metal: dequant_bf16.cu's group32 kernel with
its fp32, fp16 and BF16 stores, as dequant_f32, dequant_f16 and dequant_bf16 launch it) and the two BF16 roundings of
kernels/common/basalt_fp.metal.

- dequant_f16_parity, ported whole (basalt src/kernels/dequant_f16_parity.cpp), for the ten types of this pack: its
  blocks (one std::mt19937(7) across the run, 10240 x 2560 values per type, replayed draw for draw), the fp16 output
  bitwise the fp32 output rounded to half (:73) and the same at a 2-byte output offset (:64, :74), and the first 64
  rows within 1e-3 relative of the CPU dequantizer (:76-84). The timing (:87-97, :100-102) is ungated, not ported.
- dequant_bf16_test, ported whole (basalt src/kernels/dequant_bf16_test.cpp): the fp32 output within 1e-6 relative of
  the CPU dequantizer (:90-92) and the BF16 output within half a BF16 ulp of it (:94-95), on one tensor of every type
  of the committed pack fixture (tests/contract/fixtures/basalt_pack.safetensors) and on dequant_f16_parity's blocks.
- elementwise_parity's dequant_bf16 check (basalt src/kernels/elementwise_parity.cpp:438-467): a NaN-scale Q8_0 block
  stores NaN, the finite block beside it ggml's BF16 of its values.
- Beyond basalt, bitwise: every output against the host dequantizer (tests/contract/gguf_reference.py) and its stores;
  NaN, infinite, overflowing, signed-zero and fp16-subnormal scales; row slices; the store bounds; and the two BF16
  roundings, f2bf (dequant_bf16.cu:20-27) and bf16_rne_nonan (basalt's bf, src/prefill/kernels.cu:31-35), against
  their numpy twins."""

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import f32_to_bf16
from monolith.formats.gguf import GGML_TYPES
from monolith.runtime import _native as nt
from tests.contract import gguf_reference
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

ROWS, COLS = 10240, 2560                           # dequant_f16_parity.cpp:35's default shape
F_TYPES = (23, 8, 13, 14, 42, 11, 12, 2, 6, 20)    # its four types in its order (:37-38), then this pack's six
D_AT = {11: 108, 14: 208}                          # the block's fp16 d (Q3_K's d_all) at this byte (:48), else at 0
DMIN = 0x211F                                      # __float2half(0.01f): Q4_K's and Q5_K's dmin, at byte 2 (:51)


def run_dq(type_id, out, blocks, rows, cols, *, row0=0, out_offset=0, guard=0):
    """``basalt_dequant_g32`` (basalt's dequant_f32, dequant_f16 or dequant_bf16 for ``out`` "f32", "f16", "bf16") over
    rows [row0, row0 + rows) of ``cols`` values of type ``type_id``, from a buffer of exactly the bytes ``blocks``. The
    output buffer holds guard + out_offset + rows * cols + guard values of the output's width, every byte 0xFF, and is
    bound at value guard + out_offset. Returns the whole buffer, guards included, as uint32 or uint16 words."""
    dt = np.uint32 if out == "f32" else np.uint16
    width = np.dtype(dt).itemsize
    pso = gpu.pipeline(kernels.basalt_dequant16_source(), "basalt_dequant_g32",
                       tuple(kernels.basalt_dequant16_macros(type_id, out).items()))
    grid, threads = kernels.basalt_dequant16_grid(rows, cols)
    dev = gpu.device()
    b_in = nt.Buffer(dev, np.ascontiguousarray(blocks, np.uint8).tobytes())
    b_out = nt.Buffer(dev, (2 * guard + out_offset + rows * cols) * width)
    b_out.fill(0xFF)
    d = (nt.Dispatch().pipeline(pso).buffer(0, b_in).buffer(1, b_out, (guard + out_offset) * width)
         .bytes(2, kernels.basalt_dequant16_params(type_id, row0, rows, cols)).grid(grid).threadgroup(threads))
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    return np.frombuffer(b_out.read(0, b_out.nbytes), dt)


def reference(type_id, raw):
    """The host dequantizer (tests/contract/gguf_reference.py), float32 [rows, cols] from GGUF rows [rows, row bytes]:
    bit for bit basalt's CPU and device fp32 value on every block with finite scales (TG 2.1-2.2)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return gguf_reference.dequantize(type_id, raw)


def stored(ref, out):
    """The kernel's stores (basalt's put, dequant_bf16.cu:28-31) of the float32 values ``ref``: fp32 with a NaN as
    sm_120's 0x7fffffff (cuda_nan); fp16 by __float2half_rn, a NaN as 0x7fff (basalt_emu.cuda_half_bits); BF16 by f2bf
    of the cuda_nan'd value (monolith.formats.fp.f32_to_bf16, f2bf bit for bit). uint32 or uint16 words."""
    if out == "f16":
        return emu.cuda_half_bits(ref)
    bits = np.where(np.isnan(ref), np.uint32(0x7FFFFFFF), np.asarray(ref, np.float32).view(np.uint32))
    return bits if out == "f32" else f32_to_bf16(bits.view(np.float32))


@pytest.fixture(scope="module")
def f_blocks():
    """Fixture F's blocks, dequant_f16_parity.cpp:35-52 byte for byte: one std::mt19937(7) across the run (numpy's
    MT19937 with the legacy seeding gives the same raw words); per type every byte the low 8 bits of one draw, then per
    block one draw k for its fp16 d = __float2half(0.001f + (float) (k % 1000) * 1e-4f), and Q4_K's and Q5_K's dmin
    0x211f. All ten types in F_TYPES' order, so each type's draws are basalt's. {type id: uint8 [ROWS, row bytes]}."""
    def mt19937_7():
        g = np.random.MT19937(0)
        g._legacy_seeding(7)                                           # std::mt19937 rng(7) (:40)
        return g

    assert mt19937_7().random_raw(2).tolist() == [327741615, 976413892]   # std::mt19937(7)'s first two words
    bg = mt19937_7()
    half = (np.float32(0.001) + np.arange(1000, dtype=np.float32) * np.float32(1e-4)).astype(np.float16)
    blocks = {}
    for t in F_TYPES:
        _, be, bb = GGML_TYPES[t]
        blk = bg.random_raw(ROWS * COLS // be * bb).astype(np.uint8).reshape(-1, bb)                 # :43-44
        at = D_AT.get(t, 0)
        blk[:, at: at + 2] = half[bg.random_raw(len(blk)) % 1000].view(np.uint8).reshape(-1, 2)      # :46-50
        if t in (12, 13):
            blk[:, 2:4] = np.array([DMIN], "<u2").view(np.uint8)                                    # :51
        blocks[t] = blk.reshape(ROWS, -1)
    return blocks


@pytest.fixture(scope="module", params=kernels.DEQUANT16_TYPES, ids=lambda t: GGML_TYPES[t][0])
def f(request, f_blocks):
    """Fixture F for one type, (type id, blocks, host values); one type's instance at a time (pytest frees it before
    the next). C9's coverage assertion: every fp16 scale finite and nonzero; negative, zero and positive values
    (IQ4_NL negative and positive only: its codebook has no zero entry and d >= 0.001, so fixture E brings its zero);
    at least 1,000 distinct values, counted on the first 64 rows."""
    t = request.param
    raw = f_blocks[t]
    blk = raw.reshape(-1, GGML_TYPES[t][2])
    for at in [D_AT.get(t, 0)] + [2] * (t in (12, 13)):
        s = blk[:, at: at + 2].copy().view(np.float16)
        assert np.isfinite(s).all() and (s != 0).all(), at
    ref = reference(t, raw)
    assert ref.min() < 0 < ref.max() and (t == 20 or (ref == 0).any())
    assert np.unique(ref[:64]).size >= 1000
    return t, raw, ref


def test_outputs_equal_the_host_dequantizer(f):
    """F16-1, added (TG 3.4): over all 10240 x 2560 values of fixture F, the fp32 output bit for bit the host
    dequantizer's value and the fp16 output its round to nearest even (basalt_emu.cuda_half_bits). F16-1 itself (:73)
    passes a wrong fp32 value that both outputs share; this pins the fp32 path too."""
    t, raw, ref = f
    for out in ("f32", "f16"):
        bad = np.count_nonzero(run_dq(t, out, raw, ROWS, COLS) != stored(ref, out).ravel())
        assert bad == 0, f"{out}: {bad} of {ref.size} values differ"


BF16_PROBE = """
kernel void bf16_helpers(device const uint* bits [[buffer(0)]], device ushort* rne [[buffer(1)]],
                         device ushort* nonan [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float x = as_type<float>(bits[i]);
  rne[i] = f2bf(x);
  nonan[i] = bf16_rne_nonan(x);
}
"""


def test_bf16_helpers_bitwise():
    """DQ-4: basalt_fp.metal's f2bf (basalt src/kernels/cuda/dequant_bf16.cu:20-27) and bf16_rne_nonan (basalt's bf,
    src/prefill/kernels.cu:31-35), bit for bit, against monolith.formats.fp.f32_to_bf16 (f2bf's rounding with its NaN
    case) and basalt_emu.bf16_bits (without it), on the edge patterns (signed zeros, the smallest and largest
    subnormals, ties with bit 16 clear and set, the carry into inf, infinities, NaNs) and 2^20 random ones. Both take
    a float: the probe passes as_type<float> of each pattern, the twins the patterns viewed as float32."""
    edges = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x3f808000, 0x3f818000,
                      0x7f7f7fff, 0x7f7f8000, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000, 0x7fffffff, 0xffffffff,
                      0xff810000], np.uint32)
    bits = np.concatenate([edges, np.random.default_rng(7).integers(0, 1 << 32, 1 << 20, dtype=np.uint32)])
    n = bits.size
    pso = gpu.pipeline(kernels.PRELUDE + kernels.template("basalt_fp.metal") + BF16_PROBE, "bf16_helpers")
    out = gpu.run(pso, [bits.tobytes(), 2 * n, 2 * n], np.uint32(n).tobytes(), 3, -(-n // 256), 256, fill=0xFF)
    rne, nonan = (np.frombuffer(b, np.uint16) for b in out[1:])
    x = bits.view(np.float32)
    assert np.array_equal(rne, f32_to_bf16(x))
    assert np.array_equal(nonan, emu.bf16_bits(x))
    # basalt's rules where the two differ or carry: ties to even, the carry into inf, the NaN case f2bf alone has
    for b, want_rne, want_nonan in [(0x3f808000, 0x3f80, 0x3f80), (0x3f818000, 0x3f82, 0x3f82),
                                    (0x7f7f8000, 0x7f80, 0x7f80), (0x7f800001, 0x7fc0, 0x7f80),
                                    (0x7fffffff, 0x7fff, 0x8000), (0xffffffff, 0xffff, 0x0000)]:
        k = int(np.flatnonzero(bits == b)[0])
        assert (rne[k], nonan[k]) == (want_rne, want_nonan), hex(b)
