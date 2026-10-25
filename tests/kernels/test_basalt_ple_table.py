# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ngram.cpp:90-108, include/basalt/kernels/f16_bits.hpp:69-94
# and src/kernels/ple_parity.cpp:536-559 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE):
# the codebook, the split-half decode, the exact fp16 widening, and the head-slowest versus head-fastest check.
"""The PLE table decode of ``runtime/src/basalt_ple_table.h``: 90 IQ4_NL bytes to 160 floats, and sixteen such
rows to one 2560-wide embedding.

**What this suite can hold completely, and what it cannot.** The fp16 widening has 65,536 possible inputs, so it
is checked over its whole domain rather than sampled. The row decode is checked against an independent model
written here in numpy, over random rows and over the edge patterns, which is a complete check of the arithmetic
for any row the model and the port agree on.

What it cannot check is the artifact. basalt's oracle also records sampled values of row 0, row 12345 and the
last row of the real ``per_layer_token_embd.weight``, with whole-row sums so a wrong codebook or a wrong scale
cannot hide between the samples (``src/kernels/ple_oracle_vectors.inc:76-92``). That tensor is 28.8 GB and is
not in this tree, so those probes are not reproducible here and are not stubbed out: the decode is pinned by the
numpy model and by the three traps below instead, and the artifact probes belong with the model-file tier when
the file is present.

**The three traps.** Each produces a plausible result rather than an obviously broken one, which is why each
gets its own case rather than being left to a value comparison:

* **A2, split-half nibbles.** Byte ``j`` of a block carries elements ``j`` and ``j + 16``, never ``2j`` and
  ``2j + 1``. The interleaved reading is the natural one and decodes to 160 perfectly plausible wrong values.
* **A3, no minus eight.** Some releases of ``dequantize_row_iq4_nl`` subtract 8 from the codebook value. At the
  commit this ports they do not, and the difference shifts every element of the table.
* **C1, head-slowest flatten.** Row ``h`` occupies ``out[h*160, (h+1)*160)``. Placing element ``(d, h)`` at
  ``d*16 + h`` is a transpose, not a reshape, and the two agree on no input but a constant one.
"""

import numpy as np
import pytest

from monolith.runtime import _native as _nt

HEAD_DIM = 160
ROW_BYTES = 90
N_HEADS = 16
N_EMBD = 2560
BLK_BYTES = 18
CODEBOOK = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int32)


def model_row(row):
    """An independent model of the decode, written from the rule rather than from the port: for each of the five
    blocks, widen the fp16 scale and multiply the codebook entry of each nibble, low nibble to element j and high
    to element j + 16."""
    out = np.zeros(HEAD_DIM, np.float32)
    for b in range(HEAD_DIM // 32):
        blk = row[b * BLK_BYTES : (b + 1) * BLK_BYTES]
        d = np.frombuffer(blk[:2].tobytes(), np.float16)[0].astype(np.float32)
        qs = blk[2:].astype(np.int32)
        out[b * 32 : b * 32 + 16] = d * CODEBOOK[qs & 0x0F]
        out[b * 32 + 16 : b * 32 + 32] = d * CODEBOOK[qs >> 4]
    return out


def dequant(row):
    out = np.zeros(HEAD_DIM, np.float32)
    _nt.basalt_iq4nl_dequant_row(row, out)
    return out


def finite_rows(rng, n):
    """Random rows whose scales are finite: the decode is exact for every input, but a NaN or inf scale makes an
    equality comparison against the model meaningless rather than wrong."""
    rows = rng.integers(0, 256, size=(n, ROW_BYTES), dtype=np.uint8)
    for b in range(HEAD_DIM // 32):
        rows[:, b * BLK_BYTES + 1] &= 0x3B          # clear the exponent's top bits: no inf, no NaN
    return rows


# ---- A: one row -------------------------------------------------------------------------------------------------

def test_a1_random_rows_match_an_independent_model():
    rng = np.random.default_rng(20261025)
    rows = finite_rows(rng, 256)
    for i, row in enumerate(rows):
        got, want = dequant(row), model_row(row)
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), f"row {i} differs"


def test_a2_the_nibbles_are_split_half_not_interleaved():
    """The trap. Build the interleaved reading explicitly and assert the port does not produce it, on a row where
    the two genuinely differ."""
    rng = np.random.default_rng(7)
    row = finite_rows(rng, 1)[0]
    got = dequant(row)

    interleaved = np.zeros(HEAD_DIM, np.float32)
    for b in range(HEAD_DIM // 32):
        blk = row[b * BLK_BYTES : (b + 1) * BLK_BYTES]
        d = np.frombuffer(blk[:2].tobytes(), np.float16)[0].astype(np.float32)
        qs = blk[2:].astype(np.int32)
        interleaved[b * 32 + 0 : b * 32 + 32 : 2] = d * CODEBOOK[qs & 0x0F]
        interleaved[b * 32 + 1 : b * 32 + 32 : 2] = d * CODEBOOK[qs >> 4]

    assert not np.array_equal(got, interleaved), "the port decoded the nibbles interleaved"
    assert np.array_equal(np.sort(got), np.sort(interleaved)), "the two readings should be a permutation apart"


def test_a3_the_codebook_has_no_minus_eight():
    """Some ggml releases subtract 8 from the codebook value. The ported commit does not, and the two differ on
    every element whose scale is nonzero."""
    assert np.array_equal(np.array(_nt.basalt_iq4nl_codebook(), np.int32), CODEBOOK)
    assert CODEBOOK.min() == -127 and CODEBOOK.max() == 113
    assert 0 not in CODEBOOK, "a zero entry would make the shifted and unshifted readings agree somewhere"


def test_a4_a_zero_scale_gives_exact_zeros():
    """`value = codebook[code] * scale` has no offset term, because IQ4_NL has no min. A zero scale therefore
    gives exactly zero for every code, with no residual from an offset."""
    row = np.full(ROW_BYTES, 0xFF, np.uint8)
    for b in range(HEAD_DIM // 32):
        row[b * BLK_BYTES : b * BLK_BYTES + 2] = 0        # fp16 +0
    assert np.array_equal(dequant(row), np.zeros(HEAD_DIM, np.float32))


def test_a5_every_code_is_reachable_and_distinct():
    """All sixteen codes appear in one row with a scale of one, so a codebook typo shows as a wrong value rather
    than hiding behind codes the fixture never uses."""
    row = np.zeros(ROW_BYTES, np.uint8)
    one = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)
    for b in range(HEAD_DIM // 32):
        row[b * BLK_BYTES : b * BLK_BYTES + 2] = one
        for j in range(16):
            row[b * BLK_BYTES + 2 + j] = j | (((j + 1) % 16) << 4)
    got = dequant(row)
    assert np.array_equal(got[:16], CODEBOOK.astype(np.float32))
    assert np.array_equal(got[16:32], CODEBOOK[(np.arange(16) + 1) % 16].astype(np.float32))


# ---- B: the fp16 widening, over its whole domain ------------------------------------------------------------------

def test_b1_every_fp16_bit_pattern_widens_exactly():
    """All 65,536 inputs, compared as bits because a NaN never equals itself.

    numpy is the oracle for the 63,490 patterns that are not NaN, where its conversion is exact. It is NOT the
    oracle over the 2,046 NaN patterns, which are checked against the exponent-31 rule itself. Measured, the two
    disagree on exactly 1,022 of those: the signalling ones, whose mantissa top bit is clear. numpy's cast quiets
    them, turning 0x7c01 into 0x7fc02000 where the ported bit manipulation carries the payload through unchanged
    as 0x7f802000. The other 1,024 NaNs are already quiet and numpy passes them through, so they would have
    agreed either way. basalt rearranges bits and never quiets, which is what the rule arm asserts."""
    got = np.zeros(65536, np.float32)
    _nt.basalt_f16_widen_all(got)
    bits = np.arange(65536, dtype=np.uint32)
    ex, man = (bits >> 10) & 0x1F, bits & 0x3FF
    is_nan = (ex == 31) & (man != 0)

    want = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    assert np.array_equal(got.view(np.uint32)[~is_nan], want.view(np.uint32)[~is_nan])

    # sign | 0x7F800000 | (mantissa << 13), the exponent-31 branch written out
    rule = ((bits & 0x8000) << 16) | 0x7F800000 | (man << 13)
    assert np.array_equal(got.view(np.uint32)[is_nan], rule[is_nan])
    assert int(is_nan.sum()) == 2046, "the NaN domain moved"
    signalling = is_nan & ((man & 0x200) == 0)
    assert int(signalling.sum()) == 1022
    # and those signalling ones are exactly where numpy and the port part company
    differs = got.view(np.uint32) != want.view(np.uint32)
    assert np.array_equal(differs, signalling)


def test_b4_a_signalling_nan_keeps_its_payload():
    """The quieting numpy does is a real difference, not a representational one: it sets the mantissa's top bit.
    The port must not, because the widening is specified as a bit rearrangement."""
    got = np.zeros(65536, np.float32)
    _nt.basalt_f16_widen_all(got)
    quiet_bit = got.view(np.uint32)[0x7C01] & 0x00400000
    assert quiet_bit == 0, "the port quieted a signalling NaN"
    assert np.isnan(got[0x7C01])


def test_b2_subnormals_are_normalised_not_flushed():
    """The half that matters. A hardware convert with flush-to-zero on the input maps the smallest subnormal to
    0.0 instead of 2^-24, which is a 100 percent relative error on any scale that lands there."""
    got = np.zeros(65536, np.float32)
    _nt.basalt_f16_widen_all(got)
    subnormals = got[1:1024]                                  # exponent 0, mantissa 1..1023
    assert (subnormals > 0).all(), "an fp16 subnormal widened to zero"
    assert got[1] == pytest.approx(2.0**-24, rel=0, abs=0), "the smallest subnormal is not 2^-24"
    assert got[0] == 0.0 and np.signbit(got[0x8000]), "the signed zeros moved"


def test_b3_inf_and_nan_carry_through():
    got = np.zeros(65536, np.float32)
    _nt.basalt_f16_widen_all(got)
    assert np.isinf(got[0x7C00]) and got[0x7C00] > 0
    assert np.isinf(got[0xFC00]) and got[0xFC00] < 0
    assert np.isnan(got[0x7E00]) and np.isnan(got[0xFE00])


# ---- C: sixteen rows into one embedding ---------------------------------------------------------------------------

def test_c1_the_flatten_is_head_slowest_not_head_fastest():
    """Row h's 160 values occupy [h*160, (h+1)*160). The rival placement is a real transpose; this asserts the
    port produces the first and not the second, and that on this input the two actually differ."""
    rng = np.random.default_rng(99)
    rows = finite_rows(rng, N_HEADS)
    flat = rows.reshape(-1)
    got = np.zeros(N_EMBD, np.float32)
    _nt.basalt_ple_rows_to_embd(flat, got)

    per_head = np.stack([model_row(r) for r in rows])          # (16, 160)
    assert np.array_equal(got.view(np.uint32), per_head.reshape(-1).view(np.uint32))

    head_fastest = per_head.T.reshape(-1)                      # element (d, h) at d*16 + h
    assert not np.array_equal(got, head_fastest), "the flatten is a transpose"


def test_c2_sixteen_heads_of_160_are_exactly_the_embedding():
    assert N_HEADS * HEAD_DIM == N_EMBD


def test_c3_the_binding_refuses_wrong_lengths():
    rng = np.random.default_rng(1)
    rows = finite_rows(rng, N_HEADS).reshape(-1)
    with pytest.raises(ValueError, match="16 rows of 90 bytes"):
        _nt.basalt_ple_rows_to_embd(rows[:-1], np.zeros(N_EMBD, np.float32))
    with pytest.raises(ValueError, match="2560 floats"):
        _nt.basalt_ple_rows_to_embd(rows, np.zeros(N_EMBD - 1, np.float32))
    with pytest.raises(ValueError, match="90 bytes"):
        _nt.basalt_iq4nl_dequant_row(rows[:89], np.zeros(HEAD_DIM, np.float32))
