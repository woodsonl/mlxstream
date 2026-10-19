# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1171-1197 (hc8_quant_kernel),
# src/core/model_file.cpp:487-513 (hc8_read) and src/kernels/hc8_parity.cpp:69-99 (hc8_parity's matrices) @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""The int8 hyper-connection copies (``monolith/packs/basalt_hc8.py``), GPU-free and bit for bit. No basalt test gates
the copies' bits (hc8_parity prints the down copy's relative rms error and gates only what reads the copies), so every
test here is the port's own, under the contract's rule for quantizer codes: byte for byte (quantize_act_parity's ``if
(g_blocks[i] != r_blocks[i]) ++byte_bad;``). ``quantize`` is held against ``tests/basalt_hc_ref.py``'s scalar
transcription of ``hc8_quant_kernel``, written apart from it, and against groups whose scale and codes are worked out
by hand from basalt's arithmetic and written below."""

import numpy as np

from monolith.formats.fp import f32_to_bf16
from monolith.packs.basalt_hc8 import quantize
from tests.basalt_hc_ref import hc8_quant_scalar


def _differing_groups(w, q, s, q_ref, sc_ref):
    """The groups whose codes or scale differ between ``quantize``'s ``(q, s)`` of the BF16 bits ``w [rows, cols]`` and
    the scalar transcription's ``(q_ref, sc_ref)``: a message counting them and printing the first, ``""`` if none."""
    rows, cols = w.shape
    gpr = cols // 32
    q, q_ref = q.view(np.int8).reshape(rows, gpr, 32), q_ref.reshape(rows, gpr, 32)
    s, sc_ref = s[:, :gpr], sc_ref.reshape(rows, -1)[:, :gpr]
    bad = np.argwhere((q != q_ref).any(axis=2) | (s != sc_ref))
    if not len(bad):
        return ""
    r, j = bad[0]
    return (f"{len(bad)} of {rows * gpr} groups differ; the first, row {r} group {j} (BF16 "
            f"{[hex(b) for b in w[r, 32 * j:32 * j + 32].tolist()]}): scale {int(s[r, j]):#06x} codes {q[r, j].tolist()}, "
            f"the transcription's scale {int(sc_ref[r, j]):#06x} codes {q_ref[r, j].tolist()}")


def test_host_keeps_subnormals():
    """HC8-HOST: the host CPU computes binary32 with subnormals kept (macOS arm64 runs with FPCR.FZ clear), which is
    hc8_quant_kernel's own IEEE arithmetic and what makes ``quantize`` exact without a flush argument."""
    assert (np.float32(2.0 ** -149) * np.float32(2) == np.float32(2.0 ** -148)
            and (np.float32(2.0 ** -126) / np.float32(4)).view(np.uint32) == 0x00200000), \
        "HC8-HOST: np.float32(2.0 ** -149) * np.float32(2) == np.float32(2.0 ** -148) and " \
        "(np.float32(2.0 ** -126) / np.float32(4)).view(np.uint32) == 0x00200000: this host flushes binary32 subnormals"


def test_quantizer_equals_the_scalar_transcription():
    """HC8-VEC: a ``[64][4096]`` BF16 matrix (``gpr = srow = 128``) whose groups are each ``bf16(N(0, 1) * 2^E)`` with
    their own exponent ``E`` drawn uniformly from -140..30, except that the last group of each row, 64 in all, is random
    finite BF16 bit patterns: all-subnormal groups, f16-subnormal scales where the clamp binds, normal scales and scales
    that overflow f16 to inf. ``quantize`` and ``hc8_quant_scalar`` give the same codes, scales and bytes."""
    rng = np.random.default_rng(9)
    e = rng.integers(-140, 31, 64 * 128)
    x = (rng.standard_normal((64 * 128, 32)) * np.exp2(e)[:, None]).astype(np.float32)
    w = f32_to_bf16(x).reshape(64, 4096)
    w[:, -32:] = (rng.integers(0, 2, (64, 32)) << 15) | (rng.integers(0, 0xFF, (64, 32)) << 7) | rng.integers(0, 0x80, (64, 32))
    q, s = quantize(w)
    q_ref, sc_ref = hc8_quant_scalar(w, 128, 128)
    assert not (bad := _differing_groups(w, q, s, q_ref, sc_ref)) and np.array_equal(q.view(np.int8).ravel(), q_ref) \
        and np.array_equal(s.ravel(), sc_ref) and q.tobytes() + s.tobytes() == q_ref.tobytes() + sc_ref.tobytes(), \
        f"HC8-VEC: np.array_equal(q, q_ref), np.array_equal(sc, sc_ref) and the bytes equal: {bad}"

    h, g = s.ravel(), w.reshape(-1, 32)                                # the input reaches every regime it is for
    sub = ((h & 0x7C00) == 0) & (h != 0)
    amax = np.abs((g.astype(np.uint32) << 16).view(np.float32)).max(axis=1).astype(np.float64)
    regimes = {"all-subnormal groups": int((((g & 0x7F80) == 0).all(axis=1) & (g & 0x7FFF).any(axis=1)).sum()),
               "the clamp binding at an f16-subnormal scale":
                   int((sub & (amax > 127.5 * h.view(np.float16).astype(np.float64))).sum()),
               "normal scales": int((((h & 0x7C00) != 0) & ((h & 0x7C00) != 0x7C00)).sum()),
               "inf scales": int((h == 0x7C00).sum())}
    assert min(regimes.values()) > 0, f"HC8-VEC's input misses a regime: {regimes}"


# HC8-CASES: (case, the group's BF16 bits, zero after those listed; its scale's f16 bits; its codes, zero after those
# listed), worked out from hc8_quant_kernel's arithmetic: s = f16_rne(amax / 127), q = clamp(rint(w / s), -127, 127).
CASES = (
    ("all +0", [0x0000] * 32, 0x0000, []),
    ("all -0", [0x8000] * 32, 0x0000, []),
    ("all subnormal (amax / 127 far below 2^-25)", list(range(0x0001, 0x0021)), 0x0000, []),
    ("all subnormal, negative", [0x8000 | b for b in range(0x0001, 0x0021)], 0x0000, []),
    ("one normal among subnormals (scale 0.00787353515625)", [0x3F80] + list(range(0x0001, 0x0020)), 0x2008, [127]),
    ("the clamp at an f16-subnormal scale (amax 178 x 2^-24: amax / 127 = 1.4016 x 2^-24, scale 2^-24, quotient 178)",
     [0x3732], 0x0001, [127]),
    ("quotient ties, to even (amax 127 x 2^-7, then k x 2^-7 for k = 0.5, 1.5, 2.5, 3.5, -2.5, -3.5, 126.5, and -0)",
     [0x3F7E, 0x3B80, 0x3C40, 0x3CA0, 0x3CE0, 0xBCA0, 0xBCE0, 0x3F7D, 0x8000], 0x2000, [127, 0, 2, 2, 4, -2, -4, 126, 0]),
    ("the f16 rounding tie (amax 3.7848949e-06: amax / 127 = 2^-25 exactly, between +0 and 2^-24, to even +0)",
     [0x367E], 0x0000, []),
    ("beside the tie (amax 3.7997961e-06: scale 2^-24, quotient 63.75)", [0x367F], 0x0001, [64]),
    ("the division by the decoded scale (0.0119018555 / 0.0079345703125 = 1.5 exactly, to even 2; divided by amax / 127 "
     "it would be 1.49982, code 1)", [0x3F81, 0x3C43], 0x2010, [127, 2]),
    ("the largest finite scale (amax 8290304, scale 65280)", [0x4AFD, 0xCA7D] + [0x3F80] * 30, 0x7BF8, [127, -63]),
    ("the scale overflows to inf (amax 8323072, the first BF16 amax with amax / 127 above f16's range: x / inf = +-0)",
     [0x4AFE, 0xCA7E] + [0x3F80] * 30, 0x7C00, []),
)


def test_hand_computed_groups():
    """HC8-CASES: the groups of ``CASES`` (each one row of 32: ``gpr = 1``, ``srow = 8``); every positive BF16 ``amax``
    whose scale is a normal f16, alone in its group, has the code 127; and an up-shaped copy's layout. Both ``quantize``
    and ``hc8_quant_scalar``."""
    w = np.array([b + [0] * (32 - len(b)) for _, b, _, _ in CASES], np.uint16)
    for who, (q, s) in (("quantize", quantize(w)), ("hc8_quant_scalar", hc8_quant_scalar(w, 1, 8))):
        q, s = q.view(np.int8).reshape(len(CASES), 32), s.reshape(len(CASES), 8)
        for (case, _, scale, codes), qi, si in zip(CASES, q, s):
            assert si.tolist() == [scale] + [0] * 7 and qi.tolist() == codes + [0] * (32 - len(codes)), \
                f"HC8-CASES, {who}, {case}: scale bits {[hex(v) for v in si.tolist()]} codes {qi.tolist()}; " \
                f"the table's scale {scale:#06x} (padding slots 0) and codes {codes}, zero after"

    b = np.arange(0x0001, 0x7F80, dtype=np.uint16)                     # every positive finite BF16, the amax of a group
    w = np.zeros((b.size, 32), np.uint16)
    w[:, 0] = b
    q, s = quantize(w)
    normal = ((s[:, 0] & 0x7C00) != 0) & ((s[:, 0] & 0x7C00) != 0x7C00)
    finite_from = (b.astype(np.uint32) << 16).view(np.float32) >= np.float32(127 * 2.0 ** -14)
    assert normal.sum() == 3840 and np.array_equal(normal, finite_from & (s[:, 0] != 0x7C00)) \
        and (q[normal, 0].view(np.int8) == 127).all(), \
        f"HC8-CASES: the 3840 positive BF16 amax with a normal f16 scale (127 x 2^-14 <= amax, scale finite) each have " \
        f"the code 127: {int(normal.sum())} such amax, codes {sorted(set(q[normal, 0].view(np.int8).tolist()))}"
    q_ref, sc_ref = hc8_quant_scalar(w[normal], 1, 8)
    assert not (bad := _differing_groups(w[normal], q[normal], s[normal], q_ref, sc_ref)) \
        and np.array_equal(sc_ref.reshape(-1, 8), s[normal]), f"HC8-CASES, the 3840 amax through hc8_quant_scalar: {bad}"

    w = f32_to_bf16((np.random.default_rng(10).standard_normal((4, 320)) * 0.05).astype(np.float32))
    q, s = quantize(w)
    q_ref, sc_ref = hc8_quant_scalar(w, 10, 16)
    copy = q.tobytes() + s.tobytes()
    scales = np.frombuffer(copy, "<u2", offset=1280).reshape(4, 16)
    assert q.shape == (4, 320) and s.shape == (4, 16) and len(copy) == 1408 \
        and copy == q_ref.tobytes() + sc_ref.tobytes() and (scales[:, :10] != 0).all() and not scales[:, 10:].any(), \
        f"HC8-CASES: an up-shaped copy (rows = 4, cols = 320: gpr = 10, srow = 16) has its scales at byte 1280, slots " \
        f"10..15 of every row 0x0000, 1408 bytes; got q {q.shape}, s {s.shape}, {len(copy)} bytes, scales {scales.tolist()}"
