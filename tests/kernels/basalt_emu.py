# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu, src/kernels/cuda/iq_kernels.cu,
# src/kernels/cuda/native_bf16.cu, src/kernels/cuda/dense_tc.cu, src/kernels/cuda/q5k_head.cu,
# src/kernels/cuda/head_proxy.cuh, src/kernels/cuda/quantize_act.cu, src/kernels/cuda/s2_expert_grouped.cu,
# include/basalt/kernels/q8_1_finite.hpp and include/basalt/kernels/f16_bits.hpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36, which adapt ggml-org/llama.cpp ggml/src/ggml-common.h (kvalues_iq4nl),
# ggml/src/ggml-quants.c (get_scale_min_k4), ggml/src/ggml-cuda/quantize.cu and ggml/src/ggml-cuda/mmvf.cu @
# 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
"""numpy emulation of basalt's float32 arithmetic as nvcc compiles it (docs/research/basalt-nvcc/matvec.md), for the
bit-for-bit tests of the Metal port. Every float operation rounds once to float32, to nearest even on the float32
grid, subnormals included (gradual underflow): numpy's float32 add, multiply and divide do; ``fma32`` does by rounding
the exact float64 sum to odd before the float32 cast. The ``*_ftz`` forms flush subnormal inputs, then flush the
rounded result if it is subnormal, as PTX defines ``.ftz`` (``fma.f32``: the exact value "rounded to single
precision"; ``.ftz`` flushes "subnormal inputs and results"): an exact result in [2^-126 - 2^-150, 2^-126) gives
2^-126, the tie 2^-126 - 2^-150 included (2^-126 is the even neighbour)."""

from types import SimpleNamespace

import numpy as np

F32 = np.float32


def ftz(x):
    """A subnormal becomes the zero of its sign; every other value keeps its bits."""
    x = np.asarray(x, F32)
    b = x.view(np.uint32)
    return np.where((b & 0x7F800000) == 0, (b & 0x80000000).view(F32), x).astype(F32)


def _fma64_odd(a, b, c):
    """a * b + c of float32 inputs as a float64 rounded to odd, fma32's and fma_metal's common step."""
    a, b, c = (np.asarray(v, F32).astype(np.float64) for v in (a, b, c))
    s = a * b
    t = s + c
    bp = t - s
    err = (s - (t - bp)) + (c - bp)
    even = (t.view(np.int64) & 1) == 0
    bump = (err != 0) & even & np.isfinite(t)
    return np.where(bump, np.nextafter(t, np.where(err > 0, np.inf, -np.inf)), t)


def fma32(a, b, c):
    """a * b + c rounded once to float32 (to nearest even, on the grid with subnormals). a * b is exact in float64;
    the float64 sum is rounded to odd (TwoSum gives its exact error), which a float32 cast then rounds correctly
    (53 >= 24 + 2 bits)."""
    with np.errstate(over="ignore"):
        return _fma64_odd(a, b, c).astype(F32)


def fma_metal(a, b, c):
    """This GPU's fma in MTLMathModeSafe as test_subnormals_in_safe_mode measures it (basalt_fp.metal's header): a
    subnormal input reads as the zero of its sign, an exact result below 2^-126 in magnitude flushes to the zero of
    its sign before rounding, and every other result is fma32's on the flushed inputs. So an exact 2^-126 - 2^-150
    gives 0 here and 2^-126 in fma_ftz (.ftz flushes the rounded result). The test is on _fma64_odd's value, which is
    below 2^-126 exactly when the exact result is: rounding to odd never reaches 2^-126, an even significand, from
    below."""
    t = _fma64_odd(ftz(a), ftz(b), ftz(c))
    with np.errstate(over="ignore"):
        return np.where(np.abs(t) < 2.0 ** -126, np.copysign(0.0, t), t).astype(F32)


def add_ftz(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) + ftz(b))


def mul_ftz(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) * ftz(b))


def fma_ftz(a, b, c):
    """fma.rn.ftz.f32: rounded first, then flushed, so 2^-126 - 2^-151 exactly gives 2^-126 (module docstring)."""
    return ftz(fma32(ftz(a), ftz(b), ftz(c)))


def max_ftz(a, b):
    return np.fmax(ftz(a), ftz(b))


def rcp_ftz(x):
    """The correctly rounded reciprocal the port writes where SASS has MUFU.RCP."""
    with np.errstate(divide="ignore", over="ignore"):
        return ftz(F32(1.0) / ftz(x))


def roundf(x):
    """CUDA's roundf: half away from zero (trunc of |x| + 0.5 rounded toward zero, which is exact here)."""
    x = np.asarray(x, F32).astype(np.float64)
    return np.trunc(x + np.copysign(0.5, x)).astype(F32)


def half_bits(x):
    """float32 to fp16 bits, round to nearest even (CUDA's __float2half, Metal's half()); a NaN's bits are numpy's."""
    with np.errstate(over="ignore"):
        return np.asarray(x, F32).astype(np.float16).view(np.uint16)


def cuda_half_bits(x):
    """half_bits with a NaN as sm_120's conversion stores it, 0x7fff (docs/research/basalt-nvcc/nan.md)."""
    return np.where(np.isnan(np.asarray(x, F32)), np.uint16(0x7FFF), half_bits(x)).astype(np.uint16)


def h2f(bits):
    with np.errstate(invalid="ignore"):                      # NaN patterns stay NaN
        return np.asarray(bits, np.uint16).view(np.float16).astype(F32)


# ---- the q8_1 quantizers (matvec.md sections 1 and 3) -------------------------------------------------------------

RCP127 = np.uint32(0x3C010204).view(F32)      # fl(1/127): ptxas's multiply for the native amax / 127


def butterfly(v, op):
    """basalt's XOR butterfly over the 32 lanes of each row of ``v`` (offsets 16, 8, 4, 2, 1); every lane ends equal."""
    for o in (16, 8, 4, 2, 1):
        v = op(v, v[:, np.arange(32) ^ o])
    return v[:, 0]


def q8_1_finite(v):
    v = np.asarray(v, F32)
    return np.where(np.abs(v) > F32(65504.0), np.copysign(F32(65504.0), v), v).astype(F32)


def q8_1_codes(q, amax):
    with np.errstate(invalid="ignore"):
        c = np.where(np.isnan(q), 0, np.clip(q, -127, 127)).astype(np.int8)
    return np.where((amax == 0)[:, None], np.int8(0), c)


def q8_1_pack(d, s, codes):
    out = np.zeros((len(d), 36), np.uint8)
    out[:, 0:2] = cuda_half_bits(q8_1_finite(d)).view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = cuda_half_bits(q8_1_finite(s)).view(np.uint8).reshape(-1, 2)
    out[:, 4:] = codes.view(np.uint8)
    return out.reshape(-1)


def q8_1_native(x):
    """native_quantize_q8_1 as nvcc compiles it, with the correctly rounded reciprocal in place of MUFU.RCP."""
    v = np.asarray(x, F32).reshape(-1, 32)
    amax = butterfly(ftz(np.abs(v)), max_ftz)
    s = butterfly(v, add_ftz)
    d = q8_1_finite(mul_ftz(amax, RCP127))
    with np.errstate(invalid="ignore"):
        q = roundf(mul_ftz(v, rcp_ftz(d)[:, None]))
    return q8_1_pack(d, s, q8_1_codes(q, amax))


def q8_1_ieee(x):
    """quantize_q8_1_kernel (IEEE: correctly rounded divisions, subnormals kept), the host formula numpy computes."""
    v = np.asarray(x, F32).reshape(-1, 32)
    amax = np.fmax.reduce(np.abs(v), axis=1)                  # fmaxf drops a NaN operand
    s = butterfly(v, np.add)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        d = q8_1_finite(amax / F32(127.0))
        q = roundf(v / d[:, None])
    return q8_1_pack(d, s, q8_1_codes(q, amax))


# ---- native_mmvq (matvec.md sections 4-5) -------------------------------------------------------------------------
# The integer parts are computed from the decoded codes (each value's code, sub-block scale and min as ggml defines
# them), independently of basalt's byte tricks; the float parts follow the sequence nvcc compiles, per thread.

TRAITS = {2: (32, 2, 64, 18), 6: (32, 2, 64, 22), 8: (32, 4, 32, 34), 20: (32, 2, 64, 18), 11: (256, 16, 8, 110),
          12: (256, 16, 8, 144), 13: (256, 16, 8, 176), 14: (256, 32, 4, 210), 23: (256, 8, 16, 136), 42: (64, 2, 64, 18)}
KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int64)


def kqs_of(type_id, tid):
    return {2: 2 * (tid % 2), 6: 2 * (tid % 2), 20: 2 * (tid % 2), 8: 2 * (tid % 4), 11: tid % 16, 12: 2 * (tid % 16),
            13: 2 * (tid % 16), 14: tid % 32, 23: 4 * (tid % 8), 42: tid % 2}[type_id]


def _h(b, off):
    return h2f(b[..., off].astype(np.uint16) | (b[..., off + 1].astype(np.uint16) << 8))


def _scale_min_k4(sc, sb):
    """get_scale_min_k4 on the 12 scale bytes ``sc`` (..., 12) for sub-block indices ``sb`` (broadcast)."""
    g = lambda k: np.take_along_axis(sc, np.broadcast_to(k, sc.shape[:-1] + k.shape[-1:]), axis=-1).astype(np.int64)
    lo, hi = sb < 4, np.clip(sb, 4, 7)
    s = np.where(lo, g(np.clip(sb, 0, 3)) & 63, (g(hi + 4) & 0xF) | ((g(hi - 4) >> 6) << 4))
    m = np.where(lo, g(np.clip(sb, 0, 3) + 4) & 63, (g(hi + 4) >> 4) | ((g(hi) >> 6) << 4))
    return s, m


def _dot(codes, a):
    return (codes[None] * a).sum(axis=-1)


def _contrib(type_id, tmp, blk, kby, kqs, d8, s8, a):
    """One block step for every (column, row, thread): ``blk`` (rows, nt, bytes), ``kby`` and ``kqs`` (nt,), the q8_1
    scales d8 / sums s8 (cols, blocks) and codes a (cols, blocks, 32) of the activations."""
    nt = kqs.shape[0]
    t = np.arange(nt)
    xa = lambda b, pos: a[:, b[:, None], pos][:, None]                     # (cols, 1, nt, npos)
    if type_id in (2, 6, 20, 8):
        if type_id == 8:
            pos = 4 * kqs[:, None] + np.arange(8)
            codes = blk[..., 2:34].view(np.int8).astype(np.int64)
        else:
            pos = np.concatenate([4 * kqs[:, None] + np.arange(8), 16 + 4 * kqs[:, None] + np.arange(8)], axis=1)
            qs = blk[..., 6:22] if type_id == 6 else blk[..., 2:18]
            nib = np.concatenate([qs & 15, qs >> 4], axis=-1).astype(np.int64)
            if type_id == 6:
                qh = blk[..., 2:6].astype(np.int64)
                qh = qh[..., 0] | (qh[..., 1] << 8) | (qh[..., 2] << 16) | (qh[..., 3] << 24)
                nib = nib | (((qh[..., None] >> np.arange(32)) & 1) << 4)
            codes = KVALUES[nib] if type_id == 20 else nib
        c = np.take_along_axis(codes, np.broadcast_to(pos, codes.shape[:-1] + pos.shape[-1:]), axis=-1)
        sumi = _dot(c, xa(kby, pos)).astype(F32)
        d, dx, sx = _h(blk, 0)[None], d8[:, kby][:, None], s8[:, kby][:, None]
        if type_id in (8, 20):
            return fma_ftz(mul_ftz(d, dx), sumi, tmp)
        k = F32(-4.0) if type_id == 2 else F32(-8.0)
        return fma_ftz(d, fma_ftz(sumi, dx, mul_ftz(sx, k)), tmp)
    if type_id == 42:
        qs = blk[..., 2:18].astype(np.int64)
        codes = ((qs[..., np.arange(64) // 4] >> (2 * (np.arange(64) % 4))) & 3) - 1
        pos = 32 * kqs[:, None] + np.arange(32)
        c = np.take_along_axis(codes, np.broadcast_to(pos, codes.shape[:-1] + pos.shape[-1:]), axis=-1)
        b = kby + kqs
        sumi = _dot(c, xa(b, np.broadcast_to(np.arange(32), (nt, 32)))).astype(F32)
        return fma_ftz(mul_ftz(_h(blk, 0)[None], d8[:, b][:, None]), sumi, tmp)
    if type_id == 23:
        ib = kqs // 4
        qs = blk[..., 8:136].astype(np.int64)
        idx = 16 * ib[:, None] + np.arange(32) % 16
        nib = (np.take_along_axis(qs, np.broadcast_to(idx, qs.shape[:-1] + idx.shape[-1:]), axis=-1)
               >> (4 * (np.arange(32) // 16))) & 15
        b = kby + ib
        sumi = _dot(KVALUES[nib], xa(b, np.broadcast_to(np.arange(32), (nt, 32))))
        sl = np.take_along_axis(blk[..., 4:8].astype(np.int64), np.broadcast_to((ib // 2)[None, :, None], blk.shape[:-1] + (1,)), axis=-1)[..., 0]
        sh = blk[..., 2].astype(np.int64) | (blk[..., 3].astype(np.int64) << 8)
        ls = ((sl >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
        sumi = (sumi * (ls - 32)[None]).astype(F32)
        return fma_ftz(mul_ftz(_h(blk, 0)[None], d8[:, b][:, None]), sumi, tmp)
    if type_id in (12, 13):
        t16 = kqs // 2
        ch, mm = t16 // 4, t16 % 4
        pos = np.concatenate([4 * mm[:, None] + np.arange(4), 16 + 4 * mm[:, None] + np.arange(4)], axis=1)  # (nt, 8)
        qs0 = 48 if type_id == 13 else 16
        sc = blk[..., 4:16]
        sumf_d = sumf_m = np.zeros(tmp.shape, F32)
        for i in range(2):
            sb = 2 * ch + i
            qidx = qs0 + 32 * ch[:, None] + pos
            q = np.take_along_axis(blk, np.broadcast_to(qidx, blk.shape[:-1] + qidx.shape[-1:]), axis=-1).astype(np.int64)
            code = (q >> (4 * i)) & 15
            if type_id == 13:
                h = np.take_along_axis(blk, np.broadcast_to(16 + pos, blk.shape[:-1] + pos.shape[-1:]), axis=-1).astype(np.int64)
                code = code | (((h >> sb[:, None]) & 1) << 4)
            b = kby + sb
            av = xa(b, pos)
            dot1, dot2 = _dot(code, av), av.sum(axis=-1)
            s, m = _scale_min_k4(sc, np.broadcast_to(sb[None, :, None], sc.shape[:-1] + (1,)))
            dx = d8[:, b][:, None]
            sumf_d = fma_ftz(dx, (dot1 * s[None, ..., 0]).astype(F32), sumf_d)
            sumf_m = fma_ftz(dx, (dot2 * m[None, ..., 0]).astype(F32), sumf_m)
        return add_ftz(tmp, fma_ftz(_h(blk, 0)[None], sumf_d, -mul_ftz(_h(blk, 2)[None], sumf_m)))
    if type_id == 11:
        iqs = kqs
        h, pos = iqs // 8, 4 * (iqs % 8)[:, None] + np.arange(4)
        sumf = np.zeros(tmp.shape, F32)
        sc = blk[..., 96:108].astype(np.int64)
        for i in range(4):
            q = np.take_along_axis(blk, np.broadcast_to(32 + 32 * h[:, None] + pos, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            hm = np.take_along_axis(blk, np.broadcast_to(pos, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            code = ((q >> (2 * i)) & 3) - np.where((hm >> (4 * h[:, None] + i)) & 1, 0, 4)
            isc = 8 * h + 2 * i + (iqs % 8) // 4
            g = lambda k: np.take_along_axis(sc, np.broadcast_to(k[None, :, None], sc.shape[:-1] + (1,)), axis=-1)[..., 0]
            s = (((g(isc % 8) >> (4 * (isc // 8))) & 15) | (((g(8 + isc % 4) >> (2 * (isc // 4))) & 3) << 4)) - 32
            b = kby + 4 * h + i
            sumf = fma_ftz(d8[:, b][:, None], (_dot(code, xa(b, pos)) * s[None]).astype(F32), sumf)
        return fma_ftz(_h(blk, 108)[None], sumf, tmp)
    if type_id == 14:
        iqs = kqs
        h, l = iqs // 16, 4 * (iqs % 16)[:, None] + np.arange(4)
        sumf = np.zeros(tmp.shape, F32)
        for i in range(2):
            ql = np.take_along_axis(blk, np.broadcast_to(64 * h[:, None] + l, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            qh = np.take_along_axis(blk, np.broadcast_to(128 + 32 * h[:, None] + l % 32, blk.shape[:-1] + (4,)), axis=-1).astype(np.int64)
            code = (((ql >> (4 * i)) & 15) | (((qh >> (2 * (l // 32) + 4 * i)) & 3) << 4)) - 32
            sidx = 192 + 8 * h + 4 * i + (iqs % 16) // 4
            s = np.take_along_axis(blk, np.broadcast_to(sidx[None, :, None], blk.shape[:-1] + (1,)), axis=-1)[..., 0].view(np.int8).astype(np.int64)
            b = kby + 4 * h + 2 * i + (iqs % 16) // 8
            sumf = fma_ftz(d8[:, b][:, None], (_dot(code, xa(b, l % 32)) * s[None]).astype(F32), sumf)
        return fma_ftz(_h(blk, 208)[None], sumf, tmp)
    raise ValueError(type_id)


def mmvq(type_id, w, xq, n_in, n_out, ncols, nw=4, row_chunk=64):
    """native_mmvq's output (ncols, n_out) for GGUF rows ``w`` (n_out, row bytes) and q8_1 bytes ``xq``, in basalt's
    thread mapping with ``nw`` SIMD-groups (4: the one-column and exact layouts; 2: the upstream layout above 4
    columns): per thread the blocks kbx = tid / T + k * BPI', then SIMD-group order and the XOR butterfly."""
    div, T, bpi, bb = TRAITS[type_id]
    nb = n_in // div
    blocks = np.asarray(w, np.uint8).reshape(n_out, nb, bb)
    xb = np.asarray(xq, np.uint8).reshape(ncols, n_in // 32, 36)
    d8, s8, a = _h(xb, 0), _h(xb, 2), xb[..., 4:].view(np.int8).astype(np.int64)
    nt, bpi_l = 32 * nw, bpi * nw // 4
    tid = np.arange(nt)
    kqs, kby_per_block = kqs_of(type_id, tid), div // 32
    out = np.zeros((ncols, n_out), F32)
    for r0 in range(0, n_out, row_chunk):
        rows = blocks[r0: r0 + row_chunk]
        tmp = np.zeros((ncols, len(rows), nt), F32)
        for k in range(-(-nb // bpi_l) + 1):
            kbx = tid // T + k * bpi_l
            valid = kbx < nb
            if not valid.any():
                break
            kb = np.minimum(kbx, nb - 1)
            new = _contrib(type_id, tmp, rows[:, kb], kb * kby_per_block, kqs, d8, s8, a)
            tmp = np.where(valid[None, None], new, tmp)
        t = tmp[..., :32]
        for sgi in range(1, nw):
            t = add_ftz(t, tmp[..., 32 * sgi: 32 * sgi + 32])
        out[:, r0: r0 + row_chunk] = butterfly(t.reshape(-1, 32), add_ftz).reshape(ncols, len(rows))
    return out


# ---- the routed experts (docs/research/basalt-nvcc/experts.md) ----------------------------------------------------
# iq_kernels.cu builds with nvcc's IEEE defaults: no flush; numpy's float32 operations keep subnormals as CUDA does.

def q2_0_dots(rows, n, xq):
    """Q2_0 rows (R rows of ``n`` values, GGUF blocks) against q8_1 rows ``xq`` (E rows of n / 32 blocks): (E, R), in
    the order basalt's row_dot_multi sums (experts.md sections 1 and 3): call k (Q2_0 block k / 2, half k % 2, q8_1
    block k) on lane k % 32, s = fma(fl(d2 * d8), f(sumi), s) over the lane's calls in ascending k from +0, then the
    32-lane butterfly of plain adds. gu2's rows2 and the down kernel's 16-lane tree give the same bits."""
    rows = np.asarray(rows, np.uint8).reshape(-1, n // 64, 18)
    nr, nk = rows.shape[0], n // 32
    qs = rows[..., 2:18].astype(np.int64)
    codes = ((qs[..., np.arange(64) // 4] >> (2 * (np.arange(64) % 4))) & 3) - 1     # (R, n / 64, 64): code - 1
    xb = np.asarray(xq, np.uint8).reshape(-1, nk, 36)
    a = xb[..., 4:].view(np.int8).astype(np.float64)                                   # (E, K, 32)
    # exact integer dots (|sumi| <= 32 * 2 * 127), per call: (K, R, 32) @ (K, 32, E)
    sumi = np.matmul(codes.reshape(nr, nk, 32).transpose(1, 0, 2).astype(np.float64), a.transpose(1, 2, 0))
    d = _h(rows, 0)[:, np.arange(nk) // 2][None] * _h(xb, 0)[:, None, :]                # (E, R, K): exact products
    f = sumi.transpose(2, 1, 0).astype(F32)
    s = np.zeros(d.shape[:2] + (32,), F32)
    for k0 in range(0, nk, 32):
        ks = np.arange(k0, min(k0 + 32, nk))
        s[..., ks - k0] = fma32(d[..., ks], f[..., ks], s[..., ks - k0])
    for o in (16, 8, 4, 2, 1):
        s = s + s[..., np.arange(32) ^ o]
    return s[..., 0]


def swiglu_q8_1(gate, up, e):
    """The experts' SwiGLU + q8_1 (experts.md section 2): t = fl(1 + e), s = fl(g / t) correctly rounded,
    h = fl(s * up) with IEEE subnormals, then the IEEE quantizer. ``e`` is __expf(-g), taken from the GPU's expf_neg
    (MUFU.EX2 and fast::exp2 have no bit-exact host model)."""
    g, up, e = (np.asarray(v, F32) for v in (gate, up, e))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        h = (g / (F32(1.0) + e)) * up
    return q8_1_ieee(h)


# ---- the drafter's experts (docs/research/basalt-nvcc/s2_experts.md) ----------------------------------------------
# quantize_act.cu and s2_expert_grouped.cu build with nvcc's IEEE defaults: no flush, as numpy's float32 operations.

def f16_from_f32(x):
    """f16_bits.hpp:36-65's f16_from_f32, transcribed in integer operations: round to nearest even, a finite overflow
    to infinity, fp16 subnormals kept, a NaN to sign | 0x7E00."""
    xb = np.asarray(x, F32).view(np.uint32).astype(np.int64)
    sign, rawexp, man = (xb >> 16) & 0x8000, (xb >> 23) & 0xFF, xb & 0x7FFFFF
    exp = rawexp - 127 + 15
    sh = np.clip(14 - exp, 1, 24)
    mm = man | 0x800000
    h_sub = (mm >> sh) & 0x3FF
    rem, half = mm & ((1 << sh) - 1), 1 << (sh - 1)
    h_sub = h_sub + ((rem > half) | ((rem == half) & ((h_sub & 1) == 1)))
    h_norm = sign | (np.clip(exp, 0, 31) << 10) | (man >> 13)
    rem_n = man & 0x1FFF
    h_norm = h_norm + ((rem_n > 0x1000) | ((rem_n == 0x1000) & ((h_norm & 1) == 1)))
    out = np.where(exp <= 0, np.where(exp < -10, sign, sign | h_sub), h_norm)
    out = np.where(exp >= 31, sign | 0x7C00, out)
    out = np.where(rawexp == 0xFF, sign | 0x7C00 | np.where(man != 0, 0x200, 0), out)
    return out.astype(np.uint16)


def q8_0_scaled(x):
    """quantize_q8_0_scaled_warp_kernel (s2_experts.md section 1), the host transcription: per 32-value block amax =
    fmaxf over |x| (a NaN dropped), s = amax > 0 ? amax / 127 : 0, inv = s > 0 ? 1 / s : 0, t = x * inv,
    r = t + (t >= 0 ? 0.5 : -0.5), code (int) r clamped to +-127 (NaN to 0); d = f16_from_f32(s). Returns (34-byte
    blocks as bytes, fp32 scales)."""
    v = np.asarray(x, F32).reshape(-1, 32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        amax = np.fmax.reduce(np.abs(v), axis=1)
        s = np.where(amax > 0, amax / F32(127.0), F32(0.0)).astype(F32)
        inv = np.where(s > 0, F32(1.0) / s, F32(0.0)).astype(F32)
        t = v * inv[:, None]
        r = t + np.where(t >= 0, F32(0.5), F32(-0.5))
        code = np.where(np.isnan(r), 0, np.trunc(np.clip(r, -127, 127))).astype(np.int8)
    out = np.zeros((len(v), 34), np.uint8)
    out[:, :2] = f16_from_f32(s).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = code.view(np.uint8)
    return out.reshape(-1), s


def s2_dots(codes, scales, xq, xs):
    """Q2_0 rows of the drafter's blob against Q8_0 activation rows with fp32 scales, as gu_grouped_t_kernel and
    down_grouped_t_kernel sum them (s2_experts.md sections 2 and 5): (E, R) for code rows ``codes`` (R, n / 4 bytes),
    their fp16 scale bits ``scales`` (R, n / 64), the activation rows ``xq`` (E, n / 32 blocks of 34 bytes) and their
    scales ``xs`` (E, n / 32). Per (row, entry) lane L takes the chunks L, L + 32, ... in order, acc = fma(fl(dw * dx),
    f(s - hx), acc) from +0, s - hx the exact sum of (code - 1) * x; then the 32-lane butterfly of IEEE adds."""
    codes = np.asarray(codes, np.uint8)
    nr, nk = codes.shape[0], codes.shape[1] // 8
    w = ((codes[:, np.arange(32 * nk) // 4].astype(np.int64) >> (2 * (np.arange(32 * nk) % 4))) & 3) - 1
    xb = np.asarray(xq, np.uint8).reshape(-1, nk, 34)
    a = xb[..., 2:].view(np.int8).astype(np.float64)                                   # (E, K, 32)
    n = np.einsum("rkv,ekv->erk", w.reshape(nr, nk, 32).astype(np.float64), a).astype(F32)   # exact integers
    dw = h2f(np.asarray(scales, np.uint16))[:, np.arange(nk) // 2]                      # (R, K)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = dw[None] * np.asarray(xs, F32).reshape(-1, nk)[:, None, :]                  # (E, R, K): fl(dw * dx)
    acc = np.zeros(n.shape[:2] + (32,), F32)
    for k0 in range(0, nk, 32):
        ks = np.arange(k0, min(k0 + 32, nk))
        acc[..., ks - k0] = fma32(p[..., ks], n[..., ks], acc[..., ks - k0])
    with np.errstate(over="ignore", invalid="ignore"):
        for o in (16, 8, 4, 2, 1):
            acc = acc + acc[..., np.arange(32) ^ o]
    return acc[..., 0]


def s2_swiglu(g, u, e):
    """swiglu_kernel (s2_experts.md section 3): fl(fl(g / fl(1 + e)) * u), IEEE; ``e`` is __expf(-g), taken from the
    GPU's expf_neg (MUFU.EX2 and fast::exp2 have no bit-exact host model)."""
    g, u, e = (np.asarray(v, F32) for v in (g, u, e))
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        return (g / (e + F32(1.0))) * u


# ---- the BF16 GEMVs on fp32 activations (docs/research/basalt-nvcc/bf16.md) -----------------------------------------
# native_bf16.cu builds with --use_fast_math: every FMA and add is .FTZ.

def bf16_bits(x):
    """float32 to bf16 bits, round to nearest even (bf16_bits.hpp's bf16_from_f32 for non-NaN values)."""
    b = np.asarray(x, F32).view(np.uint32).astype(np.uint64)
    return ((b + ((b >> 16) & 1) + 0x7FFF) >> 16).astype(np.uint16)


def mmvf(w, x, block, flush=True):
    """bf16_gemv_fp32_mmvf's outputs (n_tok, n_out) for BF16 bits ``w`` (n_out, n_in) and float32 rows ``x``
    (n_tok, n_in) with ``block`` threads (basalt's mmvf_block_size), as bf16.md records it: thread t takes the pairs
    t, t + block, ... in order, acc = fma(bf(w[2p]), x[2p], acc) then fma(bf(w[2p + 1]), x[2p + 1], acc) from +0,
    all .FTZ; the XOR butterfly per warp; for block > 32 the butterfly again over 32 slots, the warp sums and +0s.
    ``flush=False``: the same sequence in IEEE float32 (``fma32`` and plain adds), what the kernel would give without
    .FTZ."""
    fma, add = (fma_ftz, add_ftz) if flush else (fma32, np.add)
    w = np.asarray(w, np.uint16)
    n_out, n_in = w.shape
    x = np.asarray(x, F32).reshape(-1, n_in)
    wf = (w.astype(np.uint32) << 16).view(F32)
    acc = np.zeros((x.shape[0], n_out, block), F32)
    pairs = n_in // 2
    for p0 in range(0, pairs, block):
        p = np.arange(p0, min(p0 + block, pairs))
        a = fma(wf[None, :, 2 * p], x[:, None, 2 * p], acc[..., p - p0])
        acc[..., p - p0] = fma(wf[None, :, 2 * p + 1], x[:, None, 2 * p + 1], a)
    s = butterfly(acc.reshape(-1, 32), add).reshape(x.shape[0], n_out, block // 32)
    if block > 32:
        slots = np.zeros(s.shape[:2] + (32,), F32)
        slots[..., : block // 32] = s
        s = butterfly(slots.reshape(-1, 32), add).reshape(x.shape[0], n_out, 1)
    return s[..., 0]


# ---- dense_tc and dense_tc_pair (docs/research/basalt-nvcc/dense_tc.md) ----------------------------------------------
# dense_tc.cu builds with nvcc's IEEE defaults: no flush. The integer dots come from the decoded codes (each value's
# code and its block's scales as ggml defines the format), independently of basalt's byte tricks; the float steps
# follow dense_tc.md section 1 per (row, column) and section 2's warp chunks and trees.

DTC_SB = {11: 110, 12: 144, 13: 176, 14: 210, 23: 136, 42: 72}
DTC_WARPS = {"one": 24, "tile": 8, "pair": 8, "s1": 4, "s2": 8}


def _u16(b, off):
    return b[..., off].astype(np.int64) | (b[..., off + 1].astype(np.int64) << 8)


def dtc_decode(type_id, w, n_in, n_out):
    """The codes (n_out, n_in / 32, 32) of every 32-value block and its scale terms, as ggml's formats define them:
    Q4_K / Q5_K {"sc", "m"} per block and {"d", "dmin"} per super-block; Q6_K / Q3_K {"s16"} per 16 values and {"d"}
    per super-block; IQ4_XS {"ls"} (the scale minus 32) per block and {"d"} per super-block; Q2_0 {"d"} per 64-value
    block."""
    nk = n_in // 32
    if type_id == 42:
        blk = np.asarray(w, np.uint8).reshape(n_out, n_in // 64, 18)
        v = np.arange(64)
        codes = ((blk[..., 2 + v // 4].astype(np.int64) >> (2 * (v % 4))) & 3) - 1
        return codes.reshape(n_out, nk, 32), {"d": _h(blk, 0)}
    blk = np.asarray(w, np.uint8).reshape(n_out, n_in // 256, DTC_SB[type_id])
    v = np.arange(256)
    if type_id in (12, 13):
        qs = blk[..., 48:176] if type_id == 13 else blk[..., 16:144]
        q = (qs[..., 32 * (v // 64) + v % 32].astype(np.int64) >> (4 * ((v // 32) % 2))) & 15
        if type_id == 13:
            q = q | (((blk[..., 16 + v % 32].astype(np.int64) >> (v // 32)) & 1) << 4)
        sc = np.broadcast_to(blk[..., None, 4:16], blk.shape[:-1] + (8, 12))
        s, m = _scale_min_k4(sc, np.arange(8)[:, None])
        terms = {"sc": s[..., 0].reshape(n_out, nk), "m": m[..., 0].reshape(n_out, nk), "d": _h(blk, 0),
                 "dmin": _h(blk, 2)}
    elif type_id == 14:
        hh, l = v // 128, v % 128
        ql = (blk[..., 64 * hh + l % 64].astype(np.int64) >> (4 * (l // 64))) & 15
        qh = (blk[..., 128 + 32 * hh + l % 32].astype(np.int64) >> (2 * (l // 32))) & 3
        q = (ql | (qh << 4)) - 32
        terms = {"s16": blk[..., 192:208].view(np.int8).astype(np.int64).reshape(n_out, 2 * nk), "d": _h(blk, 208)}
    elif type_id == 11:
        hh, j, pos = v // 128, (v % 128) // 32, v % 32
        q = (blk[..., 32 + 32 * hh + pos].astype(np.int64) >> (2 * j)) & 3
        hm = (blk[..., pos].astype(np.int64) >> (4 * hh + j)) & 1
        q = q - np.where(hm == 1, 0, 4)
        sc = blk[..., 96:108].astype(np.int64)
        isc = np.arange(16)
        s16 = (((sc[..., isc % 8] >> (4 * (isc // 8))) & 15) | (((sc[..., 8 + isc % 4] >> (2 * (isc // 4))) & 3) << 4)) - 32
        terms = {"s16": s16.reshape(n_out, 2 * nk), "d": _h(blk, 108)}
    elif type_id == 23:
        b, pos = v // 32, v % 32
        q = KVALUES[(blk[..., 8 + 16 * b + pos % 16].astype(np.int64) >> (4 * (pos // 16))) & 15]
        ib = np.arange(8)
        sl = (blk[..., 4 + ib // 2].astype(np.int64) >> (4 * (ib % 2))) & 15
        sh = (_u16(blk, 2)[..., None] >> (2 * ib)) & 3
        terms = {"ls": ((sl | (sh << 4)) - 32).reshape(n_out, nk), "d": _h(blk, 0)}
    else:
        raise ValueError(type_id)
    return q.reshape(n_out, nk, 32), terms


def dense_tc(type_id, w, xq, n_in, n_out, ncols, order):
    """dense_tc's output (ncols, n_out) in summation order ``order`` ("one", "tile", "pair", "s1", "s2") for GGUF rows
    ``w`` and q8_1 bytes ``xq`` (at least ncols columns)."""
    codes, terms = dtc_decode(type_id, w, n_in, n_out)
    nk, nchunk = n_in // 32, n_in // 512
    xb = np.asarray(xq, np.uint8).reshape(-1, nk, 36)[:ncols]
    d8, a = _h(xb, 0), xb[..., 4:].view(np.int8).astype(np.int64)                     # (C, K), (C, K, 32)
    if type_id in (14, 11):                                                           # the 16-value halves' dots
        dots = [np.einsum("rkv,ckv->rck", codes[..., 16 * e: 16 * e + 16], a[..., 16 * e: 16 * e + 16]) for e in (0, 1)]
    else:
        dot = np.einsum("rkv,ckv->rck", codes, a)
    isum = a.sum(axis=-1)                                                             # (C, K): the code sums
    col = lambda v: np.asarray(v, F32)[:, None]                                      # a row term over the columns
    f32 = lambda n: np.asarray(n).astype(F32)
    zero = np.zeros((n_out, ncols), F32)

    def half_sb(acc, c, hb):
        sb, ks = 2 * c + (hb >> 1), [16 * c + 4 * hb + i for i in range(4)]
        if type_id in (12, 13):
            for k in ks:
                f1 = terms["d"][:, sb] * f32(terms["sc"][:, k])
                f2 = terms["dmin"][:, sb] * f32(terms["m"][:, k])
                inner = fma32(col(f1), f32(dot[:, :, k]), -(col(f2) * f32(isum[None, :, k])))
                acc = fma32(inner, d8[None, :, k], acc)
            return acc
        if type_id == 42:
            for b in (0, 1):
                part = zero
                for k in ks[2 * b: 2 * b + 2]:
                    part = fma32(f32(dot[:, :, k]), d8[None, :, k], part)
                acc = fma32(col(terms["d"][:, ks[2 * b] // 2]), part, acc)
            return acc
        part = zero
        for k in ks:
            if type_id == 23:
                ival = dot[:, :, k] * terms["ls"][:, k, None]
            else:
                ival = terms["s16"][:, 2 * k, None] * dots[0][:, :, k] + terms["s16"][:, 2 * k + 1, None] * dots[1][:, :, k]
            part = fma32(f32(ival), d8[None, :, k], part)
        return fma32(col(terms["d"][:, sb]), part, acc)

    nw = DTC_WARPS[order]
    ws = []
    for wp in range(nw):
        acc = zero
        for c in range(wp >> 2, nchunk, nw // 4):
            acc = half_sb(acc, c, wp & 3)
        ws.append(acc)
    if order == "one":
        h = []
        for q in range(4):
            v = ws[q]
            for j in range(1, 6):
                v = v + ws[4 * j + q]
            h.append(v)
        out = (h[0] + h[1]) + (h[2] + h[3])
    elif order in ("tile", "pair"):
        out = ((ws[0] + ws[4]) + (ws[1] + ws[5])) + ((ws[2] + ws[6]) + (ws[3] + ws[7]))
    elif order == "s1":
        out = ((ws[0] + ws[1]) + ws[2]) + ws[3]
    else:
        out = (((ws[0] + ws[1]) + ws[2]) + ws[3]) + (((ws[4] + ws[5]) + ws[6]) + ws[7])
    return out.T.copy()


# ---- the Q5_K head's top-1 (docs/research/basalt-nvcc/q5k_head.md section 2) -------------------------------------
# A Top is (m, idx, s) arrays. m and idx change only through comparisons of logits, so every __expf argument follows
# from the logits; ``expf`` evaluates the kernel's own expf_ftz on an array (MUFU.EX2 has no bit-exact model).

TOP_NONE = 0x7FFFFFFF


def top_take(t, l, row, on, expf):
    """Top::take of logit l at row ``row`` where ``on`` (q5k_head.cu:155-164)."""
    m, idx, s = t
    up = l > m
    with np.errstate(invalid="ignore"):
        e = expf(np.where(up, add_ftz(m, -l), add_ftz(l, -m)))
    s2 = np.where(up, fma_ftz(s, e, F32(1.0)), add_ftz(s, e))
    i2 = np.where(up | ((l == m) & (row < idx)), row, idx)
    return np.where(on, np.where(up, l, m), m).astype(F32), np.where(on, i2, idx), np.where(on, s2, s).astype(F32)


def top_merge(t, o, expf):
    """Top::merge of o into t (q5k_head.cu:165-175), elementwise."""
    (m, idx, s), (om, oidx, os_) = t, o
    on = oidx != TOP_NONE
    take = (idx == TOP_NONE) | (om > m)
    with np.errstate(invalid="ignore"):
        e = expf(np.where(on, np.where(take, add_ftz(m, -om), add_ftz(om, -m)), F32(0.0)))
    s2 = np.where(take, fma_ftz(s, e, os_), fma_ftz(os_, e, s))
    i2 = np.where(take | ((om == m) & (oidx < idx)), oidx, idx)
    return np.where(on & take, om, m).astype(F32), np.where(on, i2, idx), np.where(on, s2, s).astype(F32)


def q5k_top1(logits, mapped, blocks, expf):
    """q5k_head_top1 from the kernel's logits: SIMD-group k of threadgroup g takes rows 8g + k + 8 blocks i in order,
    unmapped rows skipped (:191-219); thread 0 merges SIMD-groups 0..7 (:297-299) into the partial; the merge: thread t
    merges partials t, t + 256, ... (:309-317), the XOR butterfly 16..1 over each SIMD-group (:318-324), thread 0 the
    SIMD-groups in order (:327-329). Returns (the partials (m, idx, s) per threadgroup, the row, the probability)."""
    n = len(logits)
    nw = 8 * blocks
    t = (np.full(nw, -np.inf, F32), np.full(nw, TOP_NONE, np.int64), np.zeros(nw, F32))
    for r0 in range(0, n, nw):
        row = r0 + np.arange(nw)
        ok = row < n
        rr = np.minimum(row, n - 1)
        t = top_take(t, np.asarray(logits, F32)[rr], row, ok & np.asarray(mapped)[rr], expf)
    w = [a.reshape(blocks, 8) for a in t]
    b = tuple(a[:, 0] for a in w)
    for k in range(1, 8):
        b = top_merge(b, tuple(a[:, k] for a in w), expf)
    part = b
    g = (np.full(256, -np.inf, F32), np.full(256, TOP_NONE, np.int64), np.zeros(256, F32))
    for i0 in range(0, blocks, 256):
        i = i0 + np.arange(256)
        ok = i < blocks
        ii = np.minimum(i, blocks - 1)
        o = (part[0][ii], np.where(ok, part[1][ii], TOP_NONE), part[2][ii])
        g = top_merge(g, o, expf)
    for off in (16, 8, 4, 2, 1):
        src = np.arange(256) ^ off
        g = top_merge(g, tuple(a[src] for a in g), expf)
    f = tuple(a[0:1] for a in g)
    for k in range(1, 8):
        f = top_merge(f, tuple(a[32 * k: 32 * k + 1] for a in g), expf)
    return part, int(f[1][0]), rcp_ftz(f[2])[0]


# ---- the 2-bit proxy (q5k_head.md section 4) ------------------------------------------------------------------------

PX_RCP15 = np.uint32(0x3F2AAAAB).view(F32)   # fl(1 / 1.5): ptxas's multiply for amax / 1.5f


def px_tiles(n):
    return (n + 31) // 32


def px_level(w, rs):
    """level(w, sh) as compiled: floor(FFMA.FTZ(w, rcp(sh), 2)) converted by F2I (NaN to 0, saturating), clamped."""
    with np.errstate(invalid="ignore"):
        f = np.floor(fma_ftz(w, rs, F32(2.0)))
        return np.where(np.isnan(f), 0, np.clip(f, 0, 3)).astype(np.int64)


def px_fit(w):
    """fit_group<false> on groups ``w`` (..., 32): (the fp16 scale bits, its value)."""
    amax = np.full(w.shape[:-1], F32(1e-9), F32)
    for l in range(32):
        amax = max_ftz(amax, ftz(np.abs(w[..., l])))
    a15 = mul_ftz(amax, PX_RCP15)
    best_s, best_e = a15, np.full(a15.shape, np.inf, F32)
    for t in range(6):
        s = mul_ftz(a15, fma_ftz(F32(t), F32(-0.1), F32(1.0)))
        sh = h2f(half_bits(s))
        lv = px_level(w, rcp_ftz(sh)[..., None])
        with np.errstate(invalid="ignore", over="ignore"):
            r = fma_ftz(add_ftz(lv.astype(F32), F32(-1.5)), sh[..., None], -w)
            e = np.zeros(a15.shape, F32)
            for l in range(32):
                e = fma_ftz(r[..., l], r[..., l], e)
        better = e < best_e
        best_e, best_s = np.where(better, e, best_e), np.where(better, s, best_s)
    hs = half_bits(best_s)
    return hs, h2f(hs)


def q5k_proxy_pack(q5, row0, n, n_cap, proxy=None):
    """q5k_proxy_pack_kernel with pack_group<false>: rows row0 .. row0 + n - 1 of the Q5_K rows ``q5`` (1760 bytes
    each) into the copy ``proxy`` (uint8 bytes; zeros when None). Returns the copy."""
    out = np.zeros(px_tiles(n_cap) * 40 * 32 * 20, np.uint8) if proxy is None else np.array(proxy, np.uint8)
    rows = row0 + np.arange(n)
    b = np.asarray(q5, np.uint8).reshape(-1, 1760)[rows].reshape(n, 10, 176)[:, np.arange(80) // 8]   # (n, 80, 176)
    j = np.arange(80) % 8
    scl = b[..., 4:16].astype(np.int64)
    at = lambda k: np.take_along_axis(scl, np.broadcast_to(k[None, :, None], (n, 80, 1)), axis=-1)[..., 0]
    lo = j < 4
    sci = np.where(lo, at(np.clip(j, 0, 3)) & 63, (at(np.clip(j, 4, 7) + 4) & 0xF) | ((at(np.clip(j, 4, 7) - 4) >> 6) << 4))
    mi = np.where(lo, at(np.clip(j, 0, 3) + 4) & 63, (at(np.clip(j, 4, 7) + 4) >> 4) | ((at(np.clip(j, 4, 7)) >> 6) << 4))
    d = mul_ftz(_h(b, 0), sci.astype(F32))
    mn = mul_ftz(_h(b, 2), mi.astype(F32))
    l = np.arange(32)
    qs = np.take_along_axis(b, np.broadcast_to((48 + 32 * (j // 2))[None, :, None] + l, (n, 80, 32)), axis=-1)
    qs = qs.astype(np.int64)
    nib = np.where((j & 1)[None, :, None] == 1, qs >> 4, qs & 15)
    q = nib | (((b[..., 16:48].astype(np.int64) >> j[None, :, None]) & 1) << 4)
    with np.errstate(invalid="ignore", over="ignore"):
        w = fma_ftz(d[..., None], q.astype(F32), -mn[..., None])
    hs, sh = px_fit(w)
    lv = px_level(w, rcp_ftz(sh)[..., None])                              # (n, 80, 32)
    sh_bits = (8 * (np.arange(16) % 4) + 2 * (np.arange(16) // 4)).astype(np.uint64)   # value 4 ii + bb: bit 8 bb + 2 ii
    words = [(lv[..., 16 * c: 16 * c + 16].astype(np.uint64) << sh_bits).sum(axis=-1).astype(np.uint32) for c in (0, 1)]
    g = np.arange(80)
    slot = ((rows[:, None] >> 5) * 40 + g[None] // 2) * 32 + (rows[:, None] & 31)
    h = (g & 1)[None]
    u32 = out[: px_tiles(n_cap) * 40 * 32 * 16].view(np.uint32)
    u32[slot * 4 + 2 * h] = words[0]
    u32[slot * 4 + 2 * h + 1] = words[1]
    out[px_tiles(n_cap) * 40 * 32 * 16:].view(np.uint16)[slot * 2 + h] = hs
    return out



def q5k_proxy_score(proxy, xq, n_rows, n_cap, map_, tiles, expf):
    """px::score_kernel on the tiles ``tiles`` of the copy ``proxy`` (``n_cap`` rows) against one q8_1 column ``xq``:
    (cand (tiles, 4, 2) float32 with the rows' bits in [..., 1], part (tiles, 2)). The exponent arguments follow from
    the scores; ``expf`` evaluates the kernel's own expf_ftz on them."""
    ntc, tiles = px_tiles(n_cap), np.asarray(tiles)
    nt_ = len(tiles)
    proxy = np.asarray(proxy, np.uint8)
    words = proxy[: ntc * 40 * 32 * 16].view(np.uint32).reshape(ntc, 40, 32, 4)[tiles]     # (tile, w4, lane, word)
    scw = proxy[ntc * 40 * 32 * 16:].view(np.uint32).reshape(ntc, 40, 32)[tiles]
    xb = np.asarray(xq, np.uint8).reshape(80, 36)
    d8 = h2f(xb[:, 0].astype(np.uint16) | (xb[:, 1].astype(np.uint16) << 8))
    codes = xb[:, 4:].view(np.int8).astype(np.int64)                                         # (80, 32)
    xs = mul_ftz(d8, codes.sum(axis=1).astype(F32))
    shifts = (8 * (np.arange(16) % 4) + 2 * (np.arange(16) // 4)).astype(np.uint32)        # value 4 ii + bb of a word
    lv = ((words[..., None] >> shifts) & 3).astype(np.int64).reshape(nt_, 40, 32, 2, 32)   # (.., h, 16 c + value)
    g = 2 * np.arange(40)[:, None] + np.arange(2)[None, :]                                  # group of (w4, h)
    P = (lv * codes[g][None, :, None]).sum(axis=-1)                                         # exact, as DP4A
    inner = fma_ftz(d8[g][None, :, None], P.astype(F32), -mul_ftz(xs[g], F32(1.5))[None, :, None])
    scale = h2f(np.stack([scw & 0xFFFF, scw >> 16], axis=-1).astype(np.uint16))
    red = []
    for k in range(4):                                     # SIMD-group k: w4 = 10 k + j for j = 0..9, then h = 0, 1
        acc = np.zeros((nt_, 32), F32)
        for j in range(10):
            for h in range(2):
                acc = fma_ftz(scale[:, 10 * k + j, :, h], inner[:, 10 * k + j, :, h], acc)
        red.append(acc)
    v = add_ftz(add_ftz(add_ftz(red[0], red[1]), red[2]), red[3])
    row = tiles[:, None] * 32 + np.arange(32)[None]
    masked = (row >= n_rows) | (np.asarray(map_, np.int64)[np.minimum(row, n_rows - 1)] < 0)
    v = np.where(masked, F32(-np.inf), v).astype(F32)
    m = butterfly(v, max_ftz)
    live = v > -np.inf
    with np.errstate(invalid="ignore"):
        arg = np.where(live, add_ftz(v, -m[:, None]), F32(0.0))
    e = np.where(live, expf(arg), F32(0.0)).astype(F32)
    part = np.stack([m, butterfly(e, add_ftz)], axis=-1)
    cand = np.zeros((nt_, 4, 2), F32)
    vv, at = v.copy(), np.arange(nt_)
    for k in range(4):
        bi = np.argmax(vv, axis=1)                         # the first maximum: ties to the lower row
        cand[:, k, 0] = vv[at, bi]
        cand[:, k, 1] = (tiles * 32 + bi).astype(np.int32).view(F32)
        vv[at, bi] = -np.inf
    return cand, part


def q5k_proxy_final(cand, part, exact, map_, expf):
    """q5k_proxy_final_kernel (collect, the exact rescoring, pick) from the score's ``cand`` (tiles, 4, 2) and ``part``
    (tiles, 2); ``exact``: the head's logits of every row (the kernel rescores the listed rows with the head's own row,
    native_mmvq's bits). The list in ascending candidate order, the port's fixed order (q5k_head.md section 4); ``expf``
    evaluates the kernel's expf_ftz. Returns (id, probability, the listed rows)."""
    c, pt = np.asarray(cand, F32).reshape(-1, 2), np.asarray(part, F32)
    m = np.max(c[:, 0])
    if m == -np.inf:                                        # every row is masked
        return (0 if map_[0] < 0 else int(map_[0])), F32(1.0), np.zeros(0, np.int64)
    sel = np.flatnonzero((c[:, 0] > -np.inf) & (c[:, 0] >= add_ftz(m, F32(-5.0))))[:64]
    rows, lp = c[sel, 1].view(np.int32).astype(np.int64), c[sel, 0]
    ex = np.asarray(exact, F32)[rows]
    bv = np.max(ex)
    bi = int(np.min(rows[ex == bv]))                        # the largest exact score, the lowest row on ties
    live = pt[:, 1] > 0
    et = expf(np.where(live, add_ftz(pt[:, 0], -bv), F32(0.0)))
    ed = add_ftz(expf(add_ftz(ex, -bv)), -expf(add_ftz(lp, -bv)))
    s = np.zeros(32, F32)                                   # lane L: tiles L, L + 32, ..., then rows L, L + 32
    for t0 in range(0, len(pt), 32):
        k = min(32, len(pt) - t0)
        s[:k] = np.where(live[t0:t0 + k], fma_ftz(pt[t0:t0 + k, 1], et[t0:t0 + k], s[:k]), s[:k])
    for c0 in range(0, len(rows), 32):
        k = min(32, len(rows) - c0)
        s[:k] = add_ftz(s[:k], ed[c0:c0 + k])
    total = max_ftz(butterfly(s[None], add_ftz)[0], F32(1.0 if bi in rows else 0.0))
    return int(np.asarray(map_)[bi]), rcp_ftz(total), rows


# ---- the prompt path's products (kernels/common/basalt_prompt_gemm.metal) -------------------------------------------
# Gemm::bf16's product and Gemm::native's (Gemm::f16 on the FP16 image) are cuBLAS calls (src/prefill/gemm.cu:122-132,
# :158-185), which fix no summation order; the port's is one fp32 fma chain per output, from +0 in increasing k.

def gemm_chain(xb, wb, bf16, flush=True):
    """basalt_prompt_gemm's outputs as bits, uint32 (R, N), for 16-bit operand bits ``xb`` (R, K) and ``wb`` (N, K),
    BF16 if ``bf16`` else FP16: acc = fma_metal(x[t][k], w[n][k], acc) from +0 in increasing k, so subnormal BF16
    operands read as zero and every partial sum whose exact value is below 2^-126 flushes; a NaN stored as sm_120's
    0x7fffffff (cuda_nan). ``flush=False``: the same fold on fma32, an IEEE chain that keeps subnormals."""
    widen = (lambda b: (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(F32)) if bf16 else h2f
    xf, wf = widen(xb), widen(wb)
    fma = fma_metal if flush else fma32
    acc = np.zeros((xf.shape[0], wf.shape[0]), F32)
    for k in range(xf.shape[1]):
        acc = fma(xf[:, k, None], wf[None, :, k], acc)
    return np.where(np.isnan(acc), np.uint32(0x7FFFFFFF), acc.view(np.uint32))


def rel_norm(y, ref):
    """The GEMM gate's metric: src/kernels/hc8_parity.cpp:57-61's Err folded over every element of the float32 values
    ``y`` (Err::add's a) against ``ref`` (its b), as part 2 feeds it (``ex.add(y1[t * ldy + n], (float) ref);``,
    :261): ``ref`` cast to float32, |y - ref| in float32 widened to float64, each fold std::max's (a NaN never replaces
    the running value, which starts at 0, so callers count non-finite outputs themselves), then rel():

        struct Err {
            double abs = 0, ref = 0;
            void add(float a, float b) { abs = std::max(abs, (double) std::fabs(a - b)); ref = std::max(ref, (double) std::fabs(b)); }
            double rel() const { return ref > 0 ? abs / ref : abs; }
        };
    """
    with np.errstate(over="ignore", invalid="ignore"):
        b = np.asarray(ref).astype(F32)
        d = np.abs(np.asarray(y, F32) - b).astype(np.float64)
    e = np.fmax.reduce(d.ravel(), initial=0.0)
    r = np.fmax.reduce(np.abs(b).astype(np.float64).ravel(), initial=0.0)
    return float(e / r) if r > 0 else float(e)


# ---- the prompt path's routed experts: the host expert walk and MMQ's launches (kernels/common/basalt_mmq.metal) -----
# moe_plan transcribes basalt's host walk (src/prefill/prefill.cpp:1840-1932, :1993-1996 @ b4c63b5); pick_j,
# mmq_launches and mmq_walk transcribe the launch rules and the stream-k walk of basalt's src/prefill/mmq/mmq.cuh
# (:487-629, :654-698, :765, :801-833, :867-889) and the per-group launch of prefill.cpp:2242-2248 @ b4c63b5; mmq.cuh is
# ggml-org/llama.cpp ggml/src/ggml-cuda/mmq.cuh @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
# mmq_quantize, mmq_product and mmq_moe emulate the chain the port computes (spec-tests-moe.md 5.4): basalt's
# src/prefill/mmq/quantize.cu:10-84, mmq_load_tiles.cuh:11-75, mmq_vec_dot.cuh:14-92 and mmq.cuh's fixup :632-767 (the
# same llama.cpp files at that commit) and src/prefill/moe_mmq.cu:85-93's SwiGLU.

MMQ_GROUP = 16                                       # experts per MMQ launch (prefill.cpp:607)
MMQ_J = (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128)   # Q2_0's tile widths with fallback false (mmq_config.cuh:15-25)
# Each product's weights in a native Q2_0/Q2_0 blob at 2560 / 640 (native_expert.cpp:49-55; spec-kernels-mmq.md 4.2):
# (byte offset in the blob, rows, K): gate then up, 1280 rows of 2560 at 0; down, 2560 rows of 640 at 921,600.
MMQ_PRODUCTS = {"gate_up": (0, 1280, 2560), "down": (921600, 2560, 640)}


def moe_plan(ids, n_expert, cpu=()):
    """basalt's host walk over one chunk's routed ids ``ids`` [T, K] on one GPU (no peer): the counts with the range
    check (prefill.cpp:1840-1845), ``off`` (:1896-1909), ``slot`` and ``src`` (:1911-1917), ``order`` (:1926-1932) and
    the absolute bounds (:1995-1996). ``cpu`` holds the experts basalt's ``on_cpu`` marks: their rows a last block in
    id order (:1907-1908), out of ``rows_local`` (:1893) and of ``order``. An id outside [0, n_expert) raises basalt's
    error (:1843). Returns ``slot`` [T, K] (pair (t, k) to its sorted row), ``src`` [T * K] (sorted row to token),
    ``n``, ``order`` [n], ``rows_local``, ``bounds`` [n + 1] and ``wpre`` [n + 1], the port's work map: the exclusive
    prefix of ceil(cnt / 8) in launch order, one work item per 8 rows of an expert."""
    ids = np.asarray(ids)
    T, K = ids.shape
    cnt = np.zeros(n_expert, np.int64)
    for e in ids.ravel():                                  # :1840-1845
        if e < 0 or e >= n_expert:
            raise ValueError("prefill: routed id out of range")
        cnt[e] += 1
    on_cpu = np.zeros(n_expert, bool)
    on_cpu[list(cpu)] = True
    off, r = np.zeros(n_expert + 1, np.int64), 0
    for e in range(n_expert):                              # :1898-1902, the GPU's experts in id order
        if not on_cpu[e]:
            off[e], r = r, r + cnt[e]
    for e in range(n_expert):                              # :1907-1908, the CPU's block, last
        if on_cpu[e]:
            off[e], r = r, r + cnt[e]
    off[n_expert] = r
    rows_local = T * K - int(cnt[on_cpu].sum())            # :1849, :1893
    fill = off[:-1].copy()
    slot, src = np.zeros(T * K, np.int32), np.zeros(T * K, np.int32)
    for i, e in enumerate(ids.ravel()):                    # :1912-1917
        p = fill[e]
        fill[e] += 1
        slot[i], src[p] = p, i // K
    order = np.array([e for e in range(n_expert) if cnt[e] > 0 and not on_cpu[e]], np.int32)   # :1926-1932
    bounds = np.append(off[order], rows_local).astype(np.int32)                                 # :1995-1996
    wpre = np.append(0, np.cumsum(-(-cnt[order] // 8))).astype(np.int32)
    return SimpleNamespace(slot=slot.reshape(T, K), src=src, n=len(order), order=order, rows_local=rows_local,
                           bounds=bounds, wpre=wpre)


def pick_j(maxr):
    """mul_mat_q_switch_J (mmq.cuh:867-889) with ncols_opt = max_rows (prefill.cpp never sets opt_rows,
    moe_mmq.cu:173): the first configured tile width reaching the fewest column tiles. Its shared-memory test never
    refuses a width on basalt's GPU (57,856 B at most against sm_120's 99 KiB opt-in), so it has no term here."""
    best, nbest = 0, 2 ** 31 - 1
    for j in range(8, 129, 8):                             # for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8)
        if nbest <= 1:
            break
        if j in MMQ_J and -(-maxr // j) < nbest:           # a configured J with fewer column tiles
            best, nbest = j, -(-maxr // j)
    return best


def mmq_launches(bounds, nty, nsm):
    """The launches of one MoE product over a plan's absolute ``bounds`` [n + 1]: groups of MMQ_GROUP consecutive
    experts of ``order`` (prefill.cpp:2242-2248: ``ngx``, ``maxr`` the group's largest row count, here a bounds
    difference), each with its tile width ``J`` (pick_j), column tiles ``ntx`` (mmq.cuh:802), ``ntiles = ntx * nty *
    ngx`` and the stream-k grid at ``nsm`` multiprocessors (:830-833); ``nty`` is the weight's 128-row tiles (:801)."""
    cnt = np.diff(np.asarray(bounds, np.int64))
    out = []
    for j0 in range(0, len(cnt), MMQ_GROUP):
        c = cnt[j0:j0 + MMQ_GROUP]
        ngx, maxr = len(c), int(c.max())
        J = pick_j(maxr)
        ntx = -(-maxr // J)
        ntiles = ntx * nty * ngx
        nwaves = -(-ntiles // nsm)
        grid = ntiles if 100 * ntiles // (nsm * nwaves) >= 90 else nsm
        out.append(SimpleNamespace(j0=j0, ngx=ngx, maxr=maxr, J=J, ntx=ntx, ntiles=ntiles, grid=grid))
    return out


def mmq_pieces(expr):
    """The chains (lo, hi) of a tile's mmq_walk expression in the order they are added."""
    return [expr[1:]] if expr[0] == "P" else [] if expr[0] == "zero" else mmq_pieces(expr[1]) + mmq_pieces(expr[2])


def mmq_walk(ntiles, bpne, grid, bpi=4):
    """Every tile's output in one stream-k launch of ``grid`` CTAs over ``ntiles`` tiles of ``bpne`` weight blocks
    each (``bpi`` blocks per 256-value k iteration), walked over the CTAs as mul_mat_q (mmq.cuh:487-629) and
    mul_mat_q_stream_k_fixup (:654-698, :765) are written. Tile ``t = (it * ngx + zt) * ntx + jt`` maps to an
    expression: ``("P", lo, hi)`` one chain from +0 over the tile's k iterations covering blocks [lo, hi) in k order
    (whole iterations: the down product's last one runs blocks 8 to 11), ``("add", a, b)`` FADD.FTZ, ``("zero",)``
    +0. A tile a CTA finishes is stored (``=``); the piece a CTA ends inside goes to the fixup's scratch, and the CTA
    that finishes that tile adds the earlier pieces, nearest first from +0, onto its own. An empty column tile
    (``jt * J`` at or past its expert's rows) is skipped without a store but keeps its expression here."""
    W = ntiles * bpne
    kb = [k - (k % bpne) % bpi for k in (b * W // grid for b in range(grid + 1))]    # :487-491
    direct, tmp = {}, {}
    for b in range(grid):
        kbc, stop = kb[b], kb[b + 1]
        k0 = kbc % bpne
        k1 = min(bpne, k0 + stop - kbc)
        while kbc < stop and k1 == bpne:                   # :496-566, the tiles this CTA finishes
            direct[kbc // bpne] = ("P", k0, k1)
            kbc += bpne
            kbc -= kbc % bpne
            k0, k1 = 0, min(bpne, stop - kbc)
        if kbc < stop:                                     # :568-629, the piece it ends inside
            tmp[b] = (kbc // bpne, ("P", k0, k1))
    tiles = dict(direct)
    for b in range(grid):                                  # the fixup, one CTA each (:654-698)
        k0, k1 = kb[b], kb[b + 1]
        if k0 == k1 or k0 % bpne == 0 or (k0 // bpne == k1 // bpne and k1 % bpne != 0):
            continue
        s, bidx, kstop = ("zero",), b - 1, k0
        while True:
            kbc = kb[bidx]
            if kbc == kstop:                               # no data
                bidx, kstop = bidx - 1, kbc
                continue
            assert tmp[bidx][0] == k0 // bpne
            s = ("add", s, tmp[bidx][1])
            if kbc % bpne == 0 or kbc // bpne < k0 // bpne:
                break
            bidx, kstop = bidx - 1, kbc
        t = k0 // bpne
        tiles[t] = ("add", s, tiles[t])                    # dst += sum (:765): FADD.FTZ commutes
    for t in range(ntiles):                                # every tile's k range covered once
        p = sorted(mmq_pieces(tiles[t]))
        assert p[0][0] == 0 and p[-1][1] == bpne and all(a[1] == c[0] for a, c in zip(p, p[1:])), (t, tiles[t])
    return tiles


def mmq_fixup(*pieces):
    """A tile's expression in mmq_walk's terms when its k range splits into ``pieces`` (k order): P_0 alone, else
    ((+0 + P_{m-1}) + ... + P_0) + P_m, the fixup's order (mmq.cuh:646, :689, :765)."""
    *rest, last = [("P", lo, hi) for lo, hi in pieces]
    s = ("zero",)
    for p in reversed(rest):
        s = ("add", s, p)
    return ("add", s, last) if rest else last


def cuda_nan(x):
    """basalt_fp.metal's cuda_nan: every NaN as sm_120 stores it, 0x7fffffff (docs/research/basalt-nvcc/nan.md)."""
    x = np.asarray(x, F32)
    return np.where(np.isnan(x), np.uint32(0x7FFFFFFF).view(F32), x)


def mmq_code(v):
    """basalt_mmq.metal's mmq_code (spec-kernels-mmq.md 3.2): quantize.cu:53-56's ``roundf(v)`` stored to a char as
    sm_120 runs it, FADD.FTZ.RZ by copysign(0.5, v) then F2I.TRUNC.NTZ (s32, saturating, NaN to 0) and the low byte: a
    NaN gives 0, r >= 2^31 gives -1 (INT_MAX's low byte), r <= -2^31 gives 0 (INT_MIN's), else (char) (int) r."""
    r = roundf(v).astype(np.float64)
    hi, lo = r >= 2.0 ** 31, r <= -2.0 ** 31
    return np.where(hi, -1, np.where(np.isnan(r) | hi | lo, 0.0, r).astype(np.int64)).astype(np.int8)


def mmq_quantize(x):
    """quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4> (quantize.cu:10-84; spec-kernels-mmq.md 3.2, 9.7) of float32 rows
    [R, n], n a multiple of 4, as mmq::quantize launches it (moe_mmq.cu:154-159: pad512(n) values a row, those past n
    quantize.cu:39's float4 zero) and basalt_mmq.metal stores them: per row pad512(n) / 128 block_q8_1_mmq of 144
    bytes, its d4[4] (each 32-value group's d, through cuda_nan) then its 128 int8 codes. Per group, in fast math: amax
    the max of abs_ftz over the values (FMNMX.FTZ: max is exact and order-free, fmax drops a NaN), d_inv = mul_ftz(127,
    rcp_ftz(amax)) (MUFU.RCP, FMUL.FTZ), the codes mmq_code(mul_ftz(x, d_inv)), d = rcp_ftz(d_inv) (MUFU.RCP); rcp_ftz
    is the contract's correctly rounded reciprocal for MUFU.RCP."""
    x = np.asarray(x, F32)
    x = np.pad(x, ((0, 0), (0, -x.shape[1] % 512)))           # pad512 (moe_mmq.cu:21)
    v = x.reshape(-1, 32)
    amax = np.fmax.reduce(ftz(np.abs(v)), axis=1)
    d_inv = mul_ftz(F32(127.0), rcp_ftz(amax))
    out = np.empty((x.shape[0], x.shape[1] // 128, 144), np.uint8)
    out[..., :16] = cuda_nan(rcp_ftz(d_inv)).reshape(out.shape[:2] + (4,)).view(np.uint8)
    out[..., 16:] = mmq_code(mul_ftz(v, d_inv[:, None])).reshape(out.shape[:2] + (128,)).view(np.uint8)
    return out


def mmq_product(xq, plan, buf, offsets, product, nsm):
    """One MoE product, ``product`` "gate_up" or "down" (MMQ_PRODUCTS), over every launch of a plan (moe_plan's), as
    basalt_mmq.metal computes each output element (spec-kernels-mmq.md 6, 9.2-9.6): mul_mat_q<Q2_0> and its stream-k
    fixup (mmq.cuh:390-864) at ``nsm`` multiprocessors, on the activation rows ``xq`` (rows_local rows of
    mmq_quantize's blocks), with the weights read in place from the blob of expert e at ``buf[offsets[e]:]`` (the
    address table). Returns float32 [rows_local, rows] through cuda_nan.

    Per element, each piece of its tile (mmq_walk's expression, asserted to be mmq_fixup's form) is a chain from +0 over
    whole k iterations of 4 blocks, per 32-value sub-block s: t = mul_ftz(dA, float(c)) (I2FP, FMUL.FTZ) and sum =
    fma_ftz(dB, t, sum) (FFMA.FTZ; mmq_vec_dot.cuh:87), c the exact integer dot of the weight codes (crumb - 1) with the
    activation codes, dA the weight block's fp16 d, dB the activation group's d4; the pieces then fold in the fixup's
    order. The down product's last iteration also runs sub-blocks 20-23, past each 640-value row: c = 0 (the activation
    codes there are 0), dB xq's (Hq's padding, +0), dA what basalt's group buffer holds there (prefill.cpp:608-611,
    :2249-2251; spec-kernels-mmq.md 9.6): blocks 0 and 1 of the next row; past an expert's last row the next launch
    slot's row 0; past the group's last slot the zeroed MMQ_TAIL, +0. Nothing past a blob is read."""
    from tests.kernels.basalt_cpu_emu import unpack64_q2_0     # basalt_cpu_emu imports this module
    off, nout, k = MMQ_PRODUCTS[product]
    bpne, nty = k // 64, nout // 128
    bounds, order, offsets = np.asarray(plan.bounds, np.int64), np.asarray(plan.order), np.asarray(offsets)
    xq = np.asarray(xq, np.uint8).reshape(len(xq), -1)
    out = np.full((bounds[-1], nout), np.uint32(0xFFFFFFFF).view(F32), F32)      # a row never computed stays NaN
    for la in mmq_launches(bounds, nty, nsm):
        tiles = mmq_walk(la.ntiles, bpne, la.grid)
        r0, r1 = bounds[la.j0], bounds[la.j0 + la.ngx]
        rel = bounds[la.j0: la.j0 + la.ngx + 1] - r0
        starts = np.zeros((r1 - r0, nty), np.int64)            # bit q: a piece of its tile starts at iteration q
        for zt in range(la.ngx):
            for jt in range(-(-(rel[zt + 1] - rel[zt]) // la.J)):     # a column tile without rows stores nothing
                rows = slice(rel[zt] + jt * la.J, min(rel[zt] + (jt + 1) * la.J, rel[zt + 1]))
                for it in range(nty):
                    expr = tiles[(it * la.ngx + zt) * la.ntx + jt]
                    p = sorted(mmq_pieces(expr))
                    assert expr == mmq_fixup(*p) and all(lo % 4 == 0 for lo, _ in p), expr
                    starts[rows, it] = sum(1 << (lo // 4) for lo, _ in p[1:])
        mask = np.repeat(starts, 128, axis=1)
        blk = [buf[o + off: o + off + nout * bpne * 18].reshape(nout, bpne, 18)
               for o in offsets[order[la.j0: la.j0 + la.ngx]]]
        codes = [unpack64_q2_0(b[..., 2:]).reshape(nout, 2 * bpne, 32).astype(np.int8) - 1 for b in blk]
        dA = [h2f(b[..., :2].copy().view("<u2")[..., 0]) for b in blk]              # (nout, bpne): exact
        a = xq[r0:r1]
        sm, ends = np.zeros((r1 - r0, nout), F32), {}
        for q in range(-(-bpne // 4)):
            on = ((mask >> q) & 1).astype(bool)
            if q and on.any():                                 # a piece starts: keep the one ending here, restart at +0
                ends[q], sm = sm, np.where(on, F32(0.0), sm)
            for s in range(8 * q, 8 * q + 8):
                kb, g = divmod(s, 4)                           # block_q8_1_mmq kb, its 32-value group g
                dB = a[:, 144 * kb + 4 * g: 144 * kb + 4 * g + 4].copy().view(F32)
                qa = a[:, 144 * kb + 16 + 32 * g: 144 * kb + 48 + 32 * g].view(np.int8).astype(np.float64)
                if s < 2 * bpne:
                    c = np.empty((r1 - r0, nout), F32)
                    for zt in range(la.ngx):                   # exact: |c| <= 32 * 2 * 127
                        c[rel[zt]: rel[zt + 1]] = qa[rel[zt]: rel[zt + 1]] @ codes[zt][:, s].T.astype(np.float64)
                    da = np.stack([d[:, s // 2] for d in dA])
                else:                                          # the down product's padded sub-blocks
                    assert not qa.any(), "the padded sub-blocks' activation codes are not 0"
                    c, b = np.zeros((r1 - r0, nout), F32), s // 2 - bpne
                    da = np.stack([np.append(dA[zt][1:, b], dA[zt + 1][0, b] if zt + 1 < la.ngx else F32(0.0))
                                   for zt in range(la.ngx)])
                sm = fma_ftz(dB, mul_ftz(np.repeat(da, np.diff(rel), axis=0), c), sm)
        acc, split = np.zeros_like(sm), np.zeros(sm.shape, bool)
        for q in sorted(ends, reverse=True):                   # the fixup: +0, then the earlier pieces nearest first
            on = ((mask >> q) & 1).astype(bool)
            acc, split = np.where(on, add_ftz(acc, ends[q]), acc), split | on
        out[r0:r1] = cuda_nan(np.where(split, add_ftz(acc, sm), sm))    # dst, the last piece, += that sum (:765)
    return out


def mmq_swiglu(gu, expf):
    """swiglu_kernel (moe_mmq.cu:85-93, interleaved false: g and u the split halves of each GU row [R, 1280]) as basalt
    stores H [R, 640]: ``h[i] = g / (1.0f + __expf(-g)) * u;`` in fast math (FMUL.FTZ by -log2 e and MUFU.EX2, FADD.FTZ,
    MUFU.RCP then FMUL.FTZ by g, FMUL.FTZ by u; docs/research/basalt-nvcc/mmq.md section 7), cuda_nan(mul_ftz(mul_ftz(
    rcp_ftz(add_ftz(e, 1)), g), u)) with e = expf(-g) (spec-tests-moe.md 5.4 step 3). ``expf`` evaluates the kernel's
    expf_ftz on an array."""
    ff = MMQ_PRODUCTS["down"][2]
    gu = np.asarray(gu, F32)
    g, u = gu[:, :ff], gu[:, ff:]
    return cuda_nan(mul_ftz(mul_ftz(rcp_ftz(add_ftz(expf(-g), F32(1.0))), g), u))


def mmq_moe(x, plan, buf, offsets, *, expf, nsm):
    """The routed experts' MMQ chain of one MoE half on Q2_0/Q2_0 blobs at 2560 / 640, as basalt_mmq.metal's four
    kernels compute it (spec-tests-moe.md 5.4 steps 1-7): the tier-1 gates' emulator. ``x`` [T, 2560] float32,
    ``plan`` moe_plan's for the chunk's ids (CPU-held experts excluded), expert e's blob at ``buf[offsets[e]:]``, at
    ``nsm`` multiprocessors (basalt's 170, never this GPU's). ``expf`` evaluates the kernel's expf_ftz on an array
    (MUFU.EX2 and fast::exp2 have no bit-exact host model; GPU tests pass basalt_gpu.expf_ftz). Returns, rows the
    plan's rows_local in sorted order: ``xq`` [rows, 20, 144], kernel 1 (each slot's token through mmq_quantize; the
    K copies of a token are byte-identical, so each token is quantized once); ``gu`` [rows, 1280], kernel 2; ``hq``
    [rows, 8, 144], kernel 3: mmq_quantize of mmq_swiglu's H, its 640 values and 384 zeros (pad512(640)); ``d``
    [rows, 2560], kernel 4."""
    xq = mmq_quantize(x)[plan.src[:plan.rows_local]]
    gu = mmq_product(xq, plan, buf, offsets, "gate_up", nsm)
    hq = mmq_quantize(mmq_swiglu(gu, expf))
    return SimpleNamespace(xq=xq, gu=gu, hq=hq, d=mmq_product(hq, plan, buf, offsets, "down", nsm))
