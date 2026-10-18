# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/expert.cpp:143-148, :336-370, :503-556 and
# include/basalt/kernels/cpu/expert.hpp:25-61 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
# which adapt ggml-org/llama.cpp ggml/src/ggml-common.h (block_q2_0) and ggml/src/ggml-quants.c
# (dequantize_row_q2_0's code rule) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
"""numpy emulation of basalt's two CPU Q2_0 kernels as AVX-512 defines them, for the bit-for-bit tests of the ARM
NEON port (runtime/src/basalt_cpu.cpp).

A sibling of ``basalt_emu`` rather than part of it: that module is basalt's float32 arithmetic *as nvcc compiles it*
and its ``*_ftz`` forms are CUDA's, while these are AVX-512 semantics on a machine whose MXCSR (x86) and FPCR
(arm64) are at their defaults - round to nearest even, subnormals kept, every exception masked. Same primitives
(``F32``, ``fma32``, ``h2f``), different source machine.

Every step has an exact numpy counterpart, so the NaN, infinity and subnormal-scale cases fall out rather than
being special-cased:

- float32 ``+``, ``*`` and ``/`` are IEEE single, correctly rounded - VMULPS, VADDPS and the two scalar divisions.
- ``fma32`` rounds once - VFMADD231PS, for ``acc`` and ``corr``.
- MAXPS is an ordered ``a > b`` select at every node of the explicit reduction trees, which gives its NaN rule
  (unordered takes SRC2) and its (+-0, -+0) rule for free.
- VCVTTPS2DQ is the range test evaluated *before* any integer cast, so NaN, +-inf and out-of-range values all give
  the integer indefinite 0x80000000, which the clamp then turns into -127.
- the fp16 block scale is uint16 -> float16 -> float32, exact for subnormals, as VCVTPH2PS is.

``order`` selects the one compiler-dependent value in either kernel: ``_mm512_reduce_max_ps``'s first two levels
pass the high half as SRC1 under GCC and the low half under Clang, which changes ``amax`` only when a chunk holds a
NaN. See runtime/src/basalt_cpu.cpp's transcription contract.
"""

from __future__ import annotations

import numpy as np

from tests.kernels.basalt_emu import F32, fma32, h2f

QK, QKA, BB, H, MAXC, MAXT = 64, 32, 18, 2560, 80, 8
INDEF = -(2 ** 31)                     # the integer indefinite VCVTTPS2DQ returns (Intel SDM Vol. 1 D.4.2.2)


def maxps(src1, src2):
    """MAXPS(SRC1, SRC2) per lane: an ordered ``SRC1 > SRC2`` select, so either operand being NaN, and (+-0, -+0),
    give SRC2."""
    with np.errstate(invalid="ignore"):
        return np.where(src1 > src2, src1, src2).astype(F32)


def reduce_max_ps(v, order="gcc"):
    """``_mm512_reduce_max_ps`` over the last axis (16 lanes). The four-level tree of both compilers' headers; only
    the first two levels' operand order differs (GCC passes the high half as SRC1, Clang the low half). Clang's
    lowering carries ``nnan``, so ``order="clang"`` is the measured realisation of a value its compiler leaves
    undefined for a chunk holding a NaN, not a prediction of what every Clang build gives."""

    def node(hi, lo):
        return maxps(hi, lo) if order == "gcc" else maxps(lo, hi)

    u = node(v[..., 8:], v[..., :8])
    t = node(u[..., 4:], u[..., :4])
    p = maxps(t[..., :2], t[..., 2:])                       # p0 = maxps(t0, t2), p1 = maxps(t1, t3); both agree
    return maxps(p[..., 0], p[..., 1])


def reduce_add_ps(v):
    """``_mm512_reduce_add_ps`` over the last axis: the same four-level tree, compiler-independent because float
    addition is bitwise commutative on non-NaN operands."""
    with np.errstate(invalid="ignore", over="ignore"):
        u = (v[..., 8:] + v[..., :8]).astype(F32)
        t = (u[..., 4:] + u[..., :4]).astype(F32)
        p = (t[..., :2] + t[..., 2:]).astype(F32)
        return (p[..., 0] + p[..., 1]).astype(F32)


def cvttps_epi32(r):
    """``_mm512_cvttps_epi32``: truncation toward zero, and the integer indefinite for a NaN, an infinity or a value
    outside int32. The comparison is false for a NaN, so the cast never sees one."""
    rr = np.asarray(r, F32).astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(rr) < 2.0 ** 31, np.trunc(rr), float(INDEF)).astype(np.int64)


def act_quant_q8_1(x, order="gcc"):
    """``act_quant_q8_1`` (expert.cpp:336-370, the AVX-512 block). Returns (q, scale, sum, hx, nchunks)."""
    x = np.ascontiguousarray(x, F32)
    nchunks = x.size // QKA
    xb = x.reshape(nchunks, QKA)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        a = np.abs(xb)                                      # _mm512_and_ps(x, 0x7fffffff): clears the sign bit
        amax = reduce_max_ps(maxps(a[:, :16], a[:, 16:]), order)      # SRC1 = |x0|
        s = np.where(amax > 0, (amax / F32(127.0)).astype(F32), F32(0)).astype(F32)
        inv = np.where(s > 0, (F32(1.0) / s).astype(F32), F32(0)).astype(F32)
        t = (xb * inv[:, None]).astype(F32)                 # VMULPS, one rounding
        c = np.where(t >= 0, F32(0.5), F32(-0.5)).astype(F32)         # ordered: true for +0 and -0, false for NaN
        r = (t + c).astype(F32)                             # VADDPS, a second rounding
        v = np.clip(cvttps_epi32(r), -127, 127).astype(np.int32)
        total = v.sum(axis=1, dtype=np.int64).astype(np.int32)        # |sum| <= 4064, exact in any order
        hx = (s * total.astype(F32)).astype(F32)
    return v.astype(np.int8).reshape(-1), s, total, hx, nchunks


def unpack64_q2_0(block_codes):
    """The 64 codes of a block in weight order from its 16 code bytes (last axis): code m is
    ``(qs[m // 4] >> (2 * (m % 4))) & 3`` (ggml_quants.c:220-222)."""
    m = np.arange(64)
    return (np.asarray(block_codes, np.uint8)[..., m // 4] >> (2 * (m % 4)).astype(np.uint8)) & np.uint8(3)


def q2_0_rows(w, row_bytes, nblocks, q, scale, hx, rows):
    """``q2g_row_multi`` (expert.cpp:503-533) for one token over the given absolute row indices, vectorized over
    rows. Reads q[:64 * nblocks], scale[:2 * nblocks] and hx[:2 * nblocks] and nothing else."""
    raw = np.ascontiguousarray(w, np.uint8)
    q = np.ascontiguousarray(q, np.int8).astype(np.int32)
    scale = np.ascontiguousarray(scale, F32)
    hx = np.ascontiguousarray(hx, F32)
    base = np.asarray(rows, np.int64) * np.int64(row_bytes)
    nr = base.size
    acc = np.zeros((nr, 16), F32)
    corr = np.zeros((nr, 16), F32)
    lanes = np.arange(16)
    with np.errstate(invalid="ignore", over="ignore"):
        for b0 in range(0, nblocks, 8):
            nb = min(8, nblocks - b0)
            d = np.zeros((nr, 8), F32)                      # basalt's sc[8] keeps its zero tail -> h2f(0) = +0.0
            for i in range(nb):
                off = base + (b0 + i) * BB
                d[:, i] = h2f(raw[off].astype(np.uint16) | (raw[off + 1].astype(np.uint16) << 8))
            dd = d[:, lanes // 2]                           # _mm512_permutexvar_ps(dup, d8), masked beyond 2 * nb
            hxv = np.zeros(16, F32)
            hxv[: 2 * nb] = hx[2 * b0: 2 * b0 + 2 * nb]     # the masked lanes are never read
            corr = fma32(dd, hxv[None, :], corr)
            for i in range(nb):
                b = b0 + i
                off = base + b * BB + 2
                codes = unpack64_q2_0(raw[off[:, None] + np.arange(16)[None, :]]).astype(np.int32)
                dot = (codes.reshape(nr, 16, 4) * q[b * QK: (b + 1) * QK].reshape(1, 16, 4)).sum(-1)
                p0 = (d[:, i] * scale[2 * b]).astype(F32)   # RN(d_b * s_{2b}), rounded before use
                p1 = (d[:, i] * scale[2 * b + 1]).astype(F32)
                p = np.where(lanes[None, :] < 8, p0[:, None], p1[:, None]).astype(F32)
                acc = fma32(p, dot.astype(F32), acc)
        return (reduce_add_ps(acc) - reduce_add_ps(corr)).astype(F32)


def q2_0_gguf_rows_multi(w, row_bytes, nblocks, acts, out, r0, r1):
    """``q2_0_gguf_rows_multi`` (expert.cpp:544-556): writes out[t][r] for r0 <= r < r1, at the ABSOLUTE row r, and
    leaves every other element of ``out`` alone. ``acts`` is a sequence of (q, scale, hx)."""
    if r1 <= r0:
        return out
    rows = np.arange(r0, r1)
    for t, (q, scale, hx) in enumerate(acts):
        out[t, rows] = q2_0_rows(w, row_bytes, nblocks, q, scale, hx, rows)
    return out
