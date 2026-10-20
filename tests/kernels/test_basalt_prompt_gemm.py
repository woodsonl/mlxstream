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
  (b)) and the subnormal operands read as zero ((c)) give the twin's bits, where the IEEE fold gives others.
- GM-REF: the product (the handler's slices) against float64 on the same operand bits, ``emu.rel_norm``
  (hc8_parity.cpp:57-61's Err) over every output under hc8_parity.cpp:269's 5e-3, and no output non-finite, at the
  pack's BF16 shapes (Gemm::bf16's own gate, which hc8_parity applies at hc8's shapes) and FP16 ones (Gemm::native's
  second stage: the nearest bound, as no basalt test launches it), 1, 17 and 512 rows (attn_q's 12288 outputs at 1
  and 17), activations N(0, 1) and weights 0.02 N(0, 1).

basalt promises no invariance (cuBLAS picks the kernel per product, hc8_parity.cpp:263); the chain gives each by
construction, and these gates keep it bit for bit against one whole-product dispatch (``whole``):
- GM-ROWS (row invariance), BF16 (48, 2560) and FP16 (2560, 640): rows 0, 63, 64, 255 and 511 each alone, the first
  63, 65, 17 and 37 rows at those bounds, the 512 rows permuted, row 5 at row 300 of fresh rows, explicit row slices of
  64 and of 192 rows give the same rows as B 512; FP16 (2560, 6144) at B 512 in the handler's slices (more than one).
- GM-STEP (the rows are StepState data, invariant 4): compiled at B 512 and run with t_this_step 1, 17, 65, 511, 512
  and 600 (prompt_rows clamps to B), the rows below min(r, B) are the B 512 rows and the rest keep 0xFF, every row at
  done 1; at B 37 and t_this_step 600 only the clamp keeps the row tile's loads and stores inside x and y.
- GM-COLS (column invariance), BF16 (512, 2560) (48 columns hold neither n 65 at 37 nor a 192-column slice) and FP16
  (2560, 640), 37 rows: the w rows [r0, r0 + n) alone (n 1, 63, 65 at r0 0 and 37), the w rows permuted, explicit
  column slices of 64 and of 192 give the whole product's columns.
- GM-REPEAT (determinism), FP16 (6144, 2560) at B 512 in the handler's slices: y prefilled 0x00, then 0xFF, on one
  pipeline, then on a freshly compiled one, every run the same bits."""

import itertools

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import bf16_to_f32
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu
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


# GM-REF's rows (TG 7.4: the pack's shapes, KG 3) as (BF16 operands, N, K, B). BF16, Gemm::bf16's product: the shared
# expert's gate (prefill.cpp:1707), ssm_alpha and ssm_beta (:1587-1588), indexer k (:1608), indexer q and the router
# (:1609, :1698), the PLE value (:1538), hc inject (:1572). FP16, Gemm::native's second stage (Gemm::f16's product):
# the shared expert's gate and up (:1702-1703), its down (:1705), ssm_out and attn_output (:1592, :1688), attn_gate
# (:1586), attn_q (:1607; at B 1 and 17 only).
REF_BF16 = ((1, 2560), (48, 2560), (128, 2560), (512, 2560), (2560, 2560), (4, 10240))
REF_FP16 = ((640, 2560), (2560, 640), (2560, 6144), (6144, 2560), (12288, 2560))
REF_ROWS = [(bf16, n, k, b) for bf16, shapes in ((True, REF_BF16), (False, REF_FP16)) for n, k in shapes
            for b in ((1, 17) if (n, k) == (12288, 2560) else (1, 17, 512))]
REF_GATE = {   # the bound's standing per entry (ER-D5's label) with hc8_parity.cpp:269's criterion
    True: "hc8_parity.cpp:269, Gemm::bf16's gate at this pack's plain-BF16 shapes: "
          'expect(ex.rel() < 5e-3, "the product against the copy in double");',
    False: "nearest bound; no basalt test launches Gemm::native or Gemm::f16: hc8_parity.cpp:269 "
           'expect(ex.rel() < 5e-3, "the product against the copy in double");',
}


@pytest.mark.parametrize("bf16, n, k, B", REF_ROWS,
                         ids=[f"{'bf16' if b else 'fp16'}-{n}x{k}-B{B}" for b, n, k, B in REF_ROWS])
def test_product_against_double(bf16, n, k, B):
    """GM-REF: y against x64 @ w64.T on the operands' exact values, emu.rel_norm over all B x N outputs (basalt
    samples 256, hc8_parity.cpp:253-262). The fixture keeps every operand at most 2^10 and every nonzero BF16 one at
    least 2^-40 (TG 7.1: nothing overflows, no partial sum is subnormal). Prints the measured value."""
    rng = np.random.default_rng((n, k, B))
    to16, widen = (emu.bf16_bits, bf16_to_f32) if bf16 else (emu.half_bits, emu.h2f)
    xb = to16(rng.standard_normal((B, k)).astype(F32))
    wb = to16(F32(0.02) * rng.standard_normal((n, k)).astype(F32))
    x64, w64 = widen(xb).astype(np.float64), widen(wb).astype(np.float64)
    for v in (x64, w64):
        mag = np.abs(v)
        assert mag.max() <= 2.0 ** 10 and (not bf16 or mag[mag > 0].min(initial=np.inf) >= 2.0 ** -40), \
            "an operand above 2^10 in magnitude (or not finite), or a nonzero BF16 operand below 2^-40"
    ref = x64 @ w64.T
    assert np.abs(ref).max() > 0, "a vacuous reference: every output 0"
    y = gpu.prompt_gemm(xb, wb, bf16=bf16, B=B).view(F32)
    bad, rel = np.count_nonzero(~np.isfinite(y)), emu.rel_norm(y, ref)
    print(f"GM-REF {'bf16' if bf16 else 'fp16'} N {n} K {k} B {B}: rel_norm {rel:.3e}")
    assert bad == 0 and rel < 5e-3, f"{REF_GATE[bf16]} (N {n}, K {k}, B {B}): rel_norm {rel:.3e}, {bad} not finite"


def whole(xb, wb, bf16):
    """The reference of the invariance gates: the product as one dispatch over every row and column (never the
    handler's slices), asserted live: every output finite (so no 0xFF word) and one nonzero."""
    B, n = len(xb), len(wb)
    y = gpu.prompt_gemm(xb, wb, bf16=bf16, B=B, slices=[(0, B, 0, n)])
    assert np.isfinite(y.view(F32)).all() and np.count_nonzero(y), f"a vacuous reference (B {B}, N {n})"
    return y


def differ(y, want):
    """How the outputs ``y`` miss ``want``, for an assert message: those that differ, those not finite."""
    return f"{np.count_nonzero(y != want)} outputs differ, {np.count_nonzero(~np.isfinite(y.view(F32)))} not finite"


# GM-ROWS' and GM-STEP's shapes (N, K): BF16 ssm_alpha and ssm_beta (prefill.cpp:1587-1588), FP16 the shared expert's
# down (:1705).
ROW_SHAPES = {True: (48, 2560), False: (2560, 640)}
ROWS = ("port: every row bitwise equal to the same row of one whole-product dispatch at B 512 (row invariance: an "
        "output depends only on its x row, its w row and K)")


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
def test_rows_do_not_depend_on_the_batch(bf16):
    """GM-ROWS: each run's rows against the same rows of the whole product at B 512 (equal to a live reference, so
    finite); for FP16 also (2560, 6144) at B 512 in the handler's own slices, more than one, against its whole
    product."""
    (n, k), tag = ROW_SHAPES[bf16], "bf16" if bf16 else "fp16"
    rng = np.random.default_rng(43 if bf16 else 47)
    xb, wb = operands(rng, bf16, 512, n, k)
    ref, perm = whole(xb, wb, bf16), rng.permutation(512)
    runs = [(f"row {i} alone", xb[i:i + 1], None, ref[i:i + 1]) for i in (0, 63, 64, 255, 511)]
    runs += [(f"the first {b} rows at B {b}", xb[:b], None, ref[:b]) for b in (63, 65, 17, 37)]
    runs += [("the 512 rows permuted", xb[perm], None, ref[perm])]
    runs += [(f"row slices of {rb}", xb, [(r0, min(rb, 512 - r0), 0, n) for r0 in range(0, 512, rb)], ref)
             for rb in (64, 192)]
    for what, x, slices, want in runs:
        y = gpu.prompt_gemm(x, wb, bf16=bf16, B=len(x), slices=slices)
        assert np.array_equal(y, want), f"{ROWS} ({what}; {tag} N {n} K {k}): {differ(y, want)}"

    fresh = operands(rng, bf16, 512, 1, k)[0]
    fresh[300] = xb[5]
    assert np.delete((fresh != xb).any(axis=1), 300).all(), "a fresh row equals the batch's row"
    y = gpu.prompt_gemm(fresh, wb, bf16=bf16, B=512)
    assert np.array_equal(y[300], ref[5]) and np.isfinite(y.view(F32)).all(), (
        f"{ROWS} (row 5 at row 300 of fresh rows; {tag} N {n} K {k}): {differ(y[300], ref[5])} in row 300, "
        f"{np.count_nonzero(~np.isfinite(y.view(F32)))} not finite in the run")

    if not bf16:                                 # F7: a product the handler's own rule slices
        x7, w7 = operands(rng, False, 512, 2560, 6144)
        assert len(kernels.basalt_prompt_gemm_slices(512, 2560, 6144)) > 1, "FP16 (2560, 6144) at B 512 is one slice"
        want, y = whole(x7, w7, False), gpu.prompt_gemm(x7, w7, bf16=False, B=512)
        assert np.array_equal(y, want), f"{ROWS} (FP16 N 2560 K 6144 in the handler's slices): {differ(y, want)}"


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
def test_rows_from_step_state(bf16):
    """GM-STEP: compiled at B 512 and run with t_this_step 1, 17, 65, 511, 512 and 600, then done 1 at 511; and the
    first 37 rows compiled at B 37 with t_this_step 600, where the row tile reaches row 63 (at B 512 the grid ends at
    row 511) and only prompt_rows' clamp keeps its loads and stores inside x and y."""
    (n, k), tag = ROW_SHAPES[bf16], "bf16" if bf16 else "fp16"
    rng = np.random.default_rng(53 if bf16 else 59)
    xb, wb = operands(rng, bf16, 512, n, k)
    ref = whole(xb, wb, bf16)
    for B, r, done in [(512, r, 0) for r in (1, 17, 65, 511, 512, 600)] + [(512, 511, 1), (37, 600, 0)]:
        y = gpu.prompt_gemm(xb[:B], wb, bf16=bf16, B=B, rows=r, done=done)
        m = 0 if done else min(r, B)
        assert np.array_equal(y[:m], ref[:m]) and (y[m:] == FILL).all(), (
            "port: rows < min(r, B) equal the B 512 rows bitwise; rows >= r, and every row at done 1, keep 0xFF (the "
            f"rows are StepState data, the program the same) (B {B}, t_this_step {r}, done {done}; {tag} N {n} K {k}): "
            f"{differ(y[:m], ref[:m])} below row {m}, {np.count_nonzero(y[m:] != FILL)} at or past it written")


# GM-COLS' shapes (N, K): BF16 indexer q and the router (prefill.cpp:1609, :1698), as GM-ROWS' 48 columns hold neither
# n 65 at r0 37 nor a slice of 192; FP16 the shared expert's down (:1705).
COL_SHAPES = {True: (512, 2560), False: (2560, 640)}


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp16"])
def test_columns_do_not_depend_on_the_matrix_rows(bf16):
    """GM-COLS at 37 rows (a partial row tile): the w rows [r0, r0 + n) as their own matrix, the w rows permuted, and
    explicit column slices of 64 and of 192, against the matching columns of the whole product."""
    (n, k), tag, B = COL_SHAPES[bf16], "bf16" if bf16 else "fp16", 37
    rng = np.random.default_rng(61 if bf16 else 67)
    xb, wb = operands(rng, bf16, B, n, k)
    full, perm = whole(xb, wb, bf16), rng.permutation(n)
    runs = [(f"w rows [{r0}, {r0 + c}) alone", wb[r0:r0 + c], None, full[:, r0:r0 + c])
            for r0 in (0, 37) for c in (1, 63, 65)]
    runs += [("the w rows permuted", wb[perm], None, full[:, perm])]
    runs += [(f"column slices of {cb}", wb, [(0, B, c0, min(cb, n - c0)) for c0 in range(0, n, cb)], full)
             for cb in (64, 192)]
    for what, w, slices, want in runs:
        y = gpu.prompt_gemm(xb, w, bf16=bf16, B=B, slices=slices)
        assert np.array_equal(y, want), (
            "port: bitwise equal to the matching columns of the full product, computed as one dispatch (column "
            f"invariance: an output depends only on its x row, its w row and K) ({what}; {tag} N {n} K {k}, B {B}): "
            f"{differ(y, want)}")


def test_repeats_are_bit_identical():
    """GM-REPEAT (FP16): FP16 (6144, 2560) at B 512 in the handler's slices, into a y prefilled 0x00, then 0xFF, on one
    pipeline, then into 0xFF on a freshly compiled one (basalt_gpu.pipeline's cache cleared)."""
    rng = np.random.default_rng(71)
    xb, wb = operands(rng, False, 512, 6144, 2560)
    key = (kernels.basalt_prompt_gemm_source(), "basalt_prompt_gemm", 512, {"GEMM_BF16": "0"})
    pso = gpu.prompt_pipeline(*key)
    runs = [gpu.prompt_gemm(xb, wb, bf16=False, B=512, fill=f) for f in (0x00, 0xFF)]
    assert gpu.prompt_pipeline(*key) is pso, "the second run compiled a pipeline of its own"
    basalt_gpu.pipeline.cache_clear()
    runs.append(gpu.prompt_gemm(xb, wb, bf16=False, B=512))
    assert gpu.prompt_pipeline(*key) is not pso, "the third run reused the first pipeline"
    assert np.isfinite(runs[0].view(F32)).all() and np.count_nonzero(runs[0]), "a vacuous first run"
    for i, y in enumerate(runs[1:], 2):
        assert np.array_equal(y, runs[0]), (
            "port: every run bitwise identical (deterministic: no atomics, no split of K, no dependence on scheduling "
            f"or on y's prior bytes) (run {i} against run 1; fp16 N 6144 K 2560, B 512): {differ(y, runs[0])}")
