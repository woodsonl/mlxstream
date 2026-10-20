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

from monolith import kernels
from monolith.formats.fp import f32_to_bf16
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

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
