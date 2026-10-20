"""basalt's prompt-path products on Metal (kernels/common/basalt_prompt_gemm.metal). Gemm::bf16's product and
Gemm::native's (Gemm::f16 on the dequantized FP16 image) are cuBLAS calls (src/prefill/gemm.cu:122-132, :158-185) that
fix no summation order; the port computes each output as one fp32 fma chain from +0 in increasing k, and
``emu.gemm_chain`` is its numpy twin on this GPU's fma (``emu.fma_metal``).

- GM-TWIN (host only): the twin flushes where test_subnormals_in_safe_mode measures this GPU's fma flushing
  (tests/kernels/test_basalt_platform.py:114-126), on the planted BF16 cases ``PLANTED``, against the same fold on
  ``emu.fma32``, an IEEE chain."""

import numpy as np

from tests.kernels import basalt_emu as emu

F32 = np.float32


def bf16_rows(*rows):
    """BF16 bits (len(rows), 16): each row's values, then zeros; every value must be a BF16 value (asserted)."""
    v = np.zeros((len(rows), 16), F32)
    for i, r in enumerate(rows):
        v[i, :len(r)] = r
    b = emu.bf16_bits(v)
    assert np.array_equal(b.astype(np.uint32) << 16, v.view(np.uint32)), "a planted operand is not a BF16 value"
    return b


# GM-EMU's planted BF16 cases, where the chain flushes a subnormal partial sum ((a), (b)) or reads a subnormal operand
# as zero ((c)), as (x [R, 16], w [N, 16], the twin's bits [R, N], the bits of the same fold on fma32). (a) the partial
# sum 2^-127 is subnormal. (b) the exact sum 2^-126 - 2^-150, the tie IEEE rounds up to 2^-126. (c) a subnormal
# operand, x[0][0] = 2^-127 (0x0040) and w[1][0] = 2^-133 (0x0001), times a normal one gives a normal exact product at
# (0, 0) and (1, 1), 2^-117 and 2^-123.
PLANTED = {
    "a": (bf16_rows([1, -1, 1]), bf16_rows([2.0 ** -120, 2.0 ** -120 - 2.0 ** -127, 2.0 ** -126]),
          [[0x00800000]], [[0x00C00000]]),
    "b": (bf16_rows([1, 2.0 ** -75, 3 * 2.0 ** -75, 1]), bf16_rows([2.0 ** -126, 2.0 ** -74, -2.0 ** -75, 2.0 ** -126]),
          [[0x00800000]], [[0x01000000]]),
    "c": (bf16_rows([2.0 ** -127, 1], [2.0 ** 10, 1]), bf16_rows([2.0 ** 10, 2.0 ** -126], [2.0 ** -133, 2.0 ** -126]),
          [[0x00800000, 0x00800000], [0x49800000, 0x00800000]],      # (1, 0): 2^20 + 2^-126 rounds to 2^20
          [[0x05004000, 0x00800000], [0x49800000, 0x02100000]]),     # 2^-117 + 2^-126 and 2^-123 + 2^-126 kept
}


def test_twin_flushes_as_measured():
    """GM-TWIN: each planted case through emu.gemm_chain and through the same fold on emu.fma32 (flush=False)."""
    for case, (xb, wb, twin, ieee) in PLANTED.items():
        got, chain = emu.gemm_chain(xb, wb, True), emu.gemm_chain(xb, wb, True, flush=False)
        assert np.array_equal(got, twin) and np.array_equal(chain, ieee), (
            "port: the twin gives 0x00800000 for (a) and (b), where the fma32 chain gives 0x00c00000 and 0x01000000, "
            "and 0x00800000 at (c)'s (0, 0) and (1, 1), where it gives 0x05004000 and 0x02100000 (the flush "
            "test_subnormals_in_safe_mode measures, tests/kernels/test_basalt_platform.py:114-126); "
            f"({case}): twin {[hex(v) for v in got.ravel()]}, fma32 chain {[hex(v) for v in chain.ravel()]}")
