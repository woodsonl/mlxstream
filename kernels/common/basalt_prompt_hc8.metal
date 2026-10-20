// Copyright (c) 2026 Jesús Gulfo (MIT)
// SPDX-License-Identifier: MIT
// adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1229-1246 and :1328-1333
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE)
//
// basalt's hc8_dequant_bf16_kernel and its launch hc8_dequant_bf16, as basalt_hc8_dequant_bf16: the BF16 image of rows
// [row0, row0 + n) of an int8 hyper-connection copy, which Gemm::bf16_w (src/prefill/gemm.cu:134-156) writes into the
// GEMM scratch and then multiplies by (Gemm::bf16, basalt_prompt_gemm.metal). A copy is the codes q [rows][cols]
// (int8, row-major) and the scales s [rows][srow] (fp16 bits), one scale per 32 consecutive codes of a row, srow =
// (cols / 32 + 7) & ~7 (hc_fused.cu:1212); row row0 + a of the copy goes to out[a * cols .. (a + 1) * cols). Needs
// basalt_fp.metal ahead of it.
//
// One thread per 8 consecutive codes of a row (cols % 32 == 0, so the 8 never straddle a row or a scale group): the
// thread with global index g takes elements i = 8 g .. 8 g + 7 if i < n * cols, and every store is below n * cols.
// Grid: ceil(n * cols / 8 / 256) threadgroups of 256 (monolith/kernels.py's basalt_hc8_image_grid). Bindings: 0 the
// codes, 1 the scales, 2 the image (n * cols BF16 values), 3 the Hc8ImageParams record (basalt_hc8_image_params, which
// refuses rows * cols >= 2^31, so every index fits 32 bits). The kernel reads no per-step value.
//
// The arithmetic is the SASS nvcc compiles for this IEEE-built file (hc_fused.cu:1-5; docs/research/basalt-nvcc/
// hc_fused.md section 1): I2F.S8 (the code, exact: float of a signed char), HADD2.F32 (the scale, exact: h2f), one
// FMUL without .FTZ, F2FP.BF16.F32.PACK_AB (round to nearest even). A code times a finite scale is exact in fp32 (7
// bits times 11) and never subnormal (|q| >= 1 and s >= 2^-24 when neither is 0), so this GPU's flushing multiply is
// CUDA's FMUL and no ieee_* helper is needed; one multiply and no add, so nothing to contract and no fma(). The store
// is f2bf(cuda_nan(p)) (basalt_fp.metal): basalt's round to nearest even for every finite p, and a NaN as sm_120
// stores it. A NaN does reach it: a group whose amax / 127 overflows fp16 has the scale 0x7c00 and every code 0
// (hc_fused.cu:1181, :1185), so its products are 0 * inf, sm_120's 0x7fffffff, which F2FP.BF16 stores as 0x7fff [H];
// cuda_nan gives f2bf that 0x7fffffff, so it stores 0x7fff too, where hc_fused.md's bfloat(s * float(v)), or f2bf of
// this GPU's NaN 0x7fc00000, would store 0x7fc0.
#pragma STDC FP_CONTRACT OFF

struct Hc8ImageParams { uint cols, srow, row0, n; };

kernel void basalt_hc8_dequant_bf16(device const char* q [[buffer(0)]], device const ushort* s [[buffer(1)]],
                                    device ushort* out [[buffer(2)]], constant Hc8ImageParams& p [[buffer(3)]],
                                    uint g [[thread_position_in_grid]]) {
    const uint i = g * 8u;                                   // 8 values, one scale group (:1231)
    if (i >= p.n * p.cols) return;
    const uint r = p.row0 + i / p.cols;
    const uint k = i % p.cols;
    const float sc = h2f(s[r * p.srow + k / 32u]);
    // ponytail: eight char loads and eight ushort stores where basalt loads one uint2 and stores one uint4 (:1236,
    // :1243), so no alignment of a pack window or a scratch offset is assumed; the values are the same. Vector loads
    // if the image's time (~25 us a copy [H]) ever matters.
    for (uint j = 0; j < 8u; ++j) {
        const int v = int(q[r * p.cols + k + j]);           // MSL's char is signed: (int) (int8_t), I2F.S8
        out[i + j] = f2bf(cuda_nan(float(v) * sc));
    }
}
