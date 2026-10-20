"""basalt's prompt-path products on Metal (kernels/common/basalt_prompt_gemm.metal). Gemm::bf16's product and
Gemm::native's (Gemm::f16 on the dequantized FP16 image) are cuBLAS calls (src/prefill/gemm.cu:122-132, :158-185) that
fix no summation order; the port computes each output as one fp32 fma chain from +0 in increasing k, and
``emu.gemm_chain`` is its numpy twin on this GPU's fma (``emu.fma_metal``).

- GM-TWIN (host only): the twin flushes where test_subnormals_in_safe_mode measures this GPU's fma flushing
  (tests/kernels/test_basalt_platform.py:114-126), on the planted BF16 cases ``PLANTED``, against the same fold on
  ``emu.fma32``, an IEEE chain.
- GM-EMU: the kernel bit for bit against the twin at 1, 17 and 64 rows of a 64-row bound (a partial row tile), N 1, 4,
  48 and 72 (partial column tiles), K 320 and 640, BF16 operands (+-0 or magnitudes in [2^-20, 2^10]) and FP16 ones
  (activations N(0, 1), weights 0.02 N(0, 1)); the rows at and past the chunk's keep the 0xFF fill. A NaN at x[3][7]
  or w[5][9] gives 0x7fffffff in its row or its column only. ``PLANTED`` on the GPU: the flushed partial sums ((a),
  (b)) and the subnormal operands read as zero ((c)) give the twin's bits, where the IEEE fold gives others."""

import itertools

import numpy as np
import pytest

from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_prompt_gpu as gpu

F32 = np.float32
FILL = 0xFFFFFFFF              # a y word no dispatch wrote (prompt_gemm's 0xFF bytes)
CUDA_NAN = 0x7FFFFFFF


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


def operands(rng, bf16, rows, n, k):
    """GM-EMU's operand bits, x [rows, k] and w [n, k]. BF16: about one in eight +-0, the rest a random sign times
    2^u, u uniform in [-20, 10], rounded to BF16 (every nonzero magnitude in [2^-20, 2^10], and zeros drawn: asserted).
    FP16: activations N(0, 1) and weights 0.02 N(0, 1), rounded to fp16."""
    if not bf16:
        return (emu.half_bits(rng.standard_normal((rows, k)).astype(F32)),
                emu.half_bits(F32(0.02) * rng.standard_normal((n, k)).astype(F32)))

    def draw(shape):
        v = np.exp2(rng.uniform(-20.0, 10.0, shape)).astype(F32) * rng.choice(F32([-1.0, 1.0]), shape)
        b = emu.bf16_bits(np.where(rng.random(shape) < 0.125, v * F32(0.0), v))
        mag = np.abs((b.astype(np.uint32) << 16).view(F32))
        assert (mag == 0).any() and ((mag == 0) | ((mag >= 2.0 ** -20) & (mag <= 2.0 ** 10))).all(), \
            "a BF16 operand is neither 0 nor of magnitude in [2^-20, 2^10], or no zero was drawn"
        return b

    return draw((rows, k)), draw((n, k))


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
def test_product_emulates_the_order(bf16):
    """GM-EMU: every (N, K) at 1, 17 and 64 rows of a 64-row bound; then the NaN cases at 17 rows, N 72, K 320; then,
    for BF16, PLANTED at a bound of its own rows."""
    order = ("bitwise equal to emu.gemm_chain, one fp32 fma chain per output from +0 in increasing k (fully specified, "
             "CPU-reproducible)")
    rng = np.random.default_rng(29 if bf16 else 31)
    for n, k in itertools.product((1, 4, 48, 72), (320, 640)):
        xb, wb = operands(rng, bf16, 64, n, k)
        want = emu.gemm_chain(xb, wb, bf16)
        assert np.isfinite(want.view(F32)).all() and np.count_nonzero(want), f"a vacuous fixture (N {n}, K {k})"
        for r in (1, 17, 64):
            y = gpu.prompt_gemm(xb, wb, bf16=bf16, B=64, rows=r)
            assert (np.array_equal(y[:r], want[:r]) and np.isfinite(y[:r].view(F32)).all()
                    and (y[r:] == FILL).all()), (
                f"port: {order}; the rows at and past R keep 0xFF (rows {r} of 64, N {n}, K {k}): "
                f"{np.count_nonzero(y[:r] != want[:r])} outputs differ, {np.count_nonzero(~np.isfinite(y[:r].view(F32)))} "
                f"not finite, {np.count_nonzero(y[r:] != FILL)} past R written")

    xb, wb = operands(rng, bf16, 64, 72, 320)
    nans = {"x": (3, 7, 0x7FC1 if bf16 else 0x7E01), "w": (5, 9, 0xFF81 if bf16 else 0xFC01)}  # quiet +, signaling -
    for where, (i, j, bits) in nans.items():
        a, b = xb.copy(), wb.copy()
        (a if where == "x" else b)[i, j] = bits
        with np.errstate(invalid="ignore"):                  # numpy's cast of a signaling NaN (emu.h2f's case)
            want = emu.gemm_chain(a, b, bf16)
        y = gpu.prompt_gemm(a, b, bf16=bf16, B=64, rows=17)
        hit = np.zeros((17, 72), bool)
        if where == "x":
            hit[i, :] = True
        else:
            hit[:, i] = True
        assert (np.array_equal(y[:17], want[:17]) and np.array_equal(y[:17] == CUDA_NAN, hit)
                and np.isfinite(y[:17][~hit].view(F32)).all() and (y[17:] == FILL).all()), (
            f"port: a NaN operand gives 0x7fffffff in its row (column) only, every output {order} (a NaN at "
            f"{where}[{i}][{j}], {bits:#06x}): {np.count_nonzero(y[:17] != want[:17])} outputs differ from the twin, "
            f"0x7fffffff at {np.argwhere(y[:17] == CUDA_NAN).tolist()[:4]}...")

    if bf16:
        for case, (xb, wb, twin, ieee) in PLANTED.items():
            y = gpu.prompt_gemm(xb, wb, bf16=True, B=len(xb))
            assert (np.array_equal(y, twin) and np.array_equal(emu.gemm_chain(xb, wb, True), twin)
                    and np.array_equal(emu.gemm_chain(xb, wb, True, flush=False), ieee)
                    and not np.array_equal(twin, ieee)), (
                "port: (a) and (b) give 0x00800000 on the GPU and in the twin, where an IEEE chain gives 0x00c00000 and "
                "0x01000000; (c) gives 0x00800000 at (0, 0) and (1, 1) on the GPU and in the twin, where a chain that "
                "keeps subnormal operands (emu.fma32's) gives 2^-117 + 2^-126 (0x05004000) and 2^-123 + 2^-126 "
                f"(0x02100000); ({case}): GPU {[hex(v) for v in y.ravel()]}, twin {[hex(v) for v in np.ravel(twin)]}")
