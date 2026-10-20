// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/gemm.cu:122-185 and src/prefill/kernels.cu:31-40, :818-860
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
//
// basalt's prompt-path products: Gemm::bf16 (gemm.cu:122-132), BF16 x BF16, and Gemm::native's second stage, Gemm::f16
// (:158-167) on the FP16 image of a GGUF matrix (:169-185), FP16 x FP16. Each is one cublasGemmEx with an fp32 output
// and compute type, alpha 1 and beta 0; gemm.cu has no device code (docs/research/basalt-nvcc/prefill_kernels.md,
// Method), so cuBLAS picks the summation order and this port fixes its own: every output is one fp32 fma chain from
// +0 in increasing k,
//
//     acc = 0; for k in 0 .. K - 1: acc = fma(x[t][k], w[n][k], acc);  y[t][n] = cuda_nan(acc)
//
// on the operands widened exactly (h2f; a BF16 field as the high half of a float). Metal's fma reads a subnormal BF16
// operand as the zero of its sign and flushes a partial sum whose exact value is below 2^-126 (a lone product
// included): the order is the port's, not basalt code. FP16 operands, products and partial sums are never subnormal
// (multiples of 2^-24 and 2^-48). A NaN is stored as sm_120's 0x7fffffff (cuda_nan). Each output's bits depend only on
// its x row, its w row and K: not on the chunk's rows, the tile or the slice that computes it.
//
// basalt_prompt_gemm: x [B][K] (bf16 or fp16 bits), w [N][K] (the same type; a BF16 tensor in place or the FP16
// image), y [B][N] floats. Rows come from StepState through prompt_rows (monolith/kernels.py's PROMPT_ROWS, ahead of
// this file with basalt_fp.metal and the StepState struct): R = prompt_rows(st). A threadgroup computes a 64 x 64 tile
// of y, rows row0 + 64 * gy and columns col0 + 64 * gx onward (row0 and col0 are the slice's, from its params record):
// 256 threads, each 4 x 4 outputs (rows tr + 16 i, columns tc + 16 j), K in steps of 16 staged as floats. A tile at or
// past R returns before its first barrier; loads past R or N stage zeros (never read past x's B rows or w's N rows);
// stores need t < R and n < N. Grid: (ceil(ncols / 64), nrows / 64) threadgroups of 256 per slice. K % 16 == 0.
//
// Macros: GEMM_BF16 (1: BF16 operands, 0: FP16), and PROMPT_ROWS' T_SRC, ROWS_MAX and T_STATIC_ROWS.
#pragma STDC FP_CONTRACT OFF

#ifndef GEMM_BF16
#define GEMM_BF16 0
#endif

// ponytail: one geometry for every shape (a 64 x 64 tile, 4 x 4 outputs per thread, K steps of 16, single-buffered);
// a per-shape geometry macro, gated unchanged by GM-EMU and GM-ROWS (tests/kernels/test_basalt_prompt_gemm.py), when
// prompt speed is measured to matter.
#define PG_TILE 64u
#define PG_BK 16u

struct PromptGemmParams { uint n, k, row0, col0; };

static inline float pg_widen(ushort b) {
#if GEMM_BF16
    return as_type<float>(uint(b) << 16);
#else
    return h2f(b);
#endif
}

kernel void basalt_prompt_gemm(device const ushort* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                               device float* y [[buffer(2)]], constant PromptGemmParams& p [[buffer(3)]],
                               device const StepState* st [[buffer(15)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float s[2][PG_BK][PG_TILE];               // [0][kk][r]: x[t0 + r][k0 + kk]; [1][kk][c]: w[n0 + c][k0 + kk]
    const uint t0 = p.row0 + PG_TILE * tg.y, n0 = p.col0 + PG_TILE * tg.x;
    const uint R = prompt_rows(st);
    if (st->done || t0 >= R) return;                      // the whole threadgroup, before any barrier
    const uint tr = tid / 16u, tc = tid % 16u;
    float acc[4][4];
    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < p.k; k0 += PG_BK) {
        for (uint e = tid; e < PG_TILE * PG_BK; e += 256u) {
            const uint r = e / PG_BK, kk = e % PG_BK, t = t0 + r, n = n0 + r;
            s[0][kk][r] = t < R ? pg_widen(x[ulong(t) * p.k + k0 + kk]) : 0.0f;
            s[1][kk][r] = n < p.n ? pg_widen(w[ulong(n) * p.k + k0 + kk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < PG_BK; ++kk)
            for (uint i = 0; i < 4u; ++i)
                for (uint j = 0; j < 4u; ++j) acc[i][j] = fma(s[0][kk][tr + 16u * i], s[1][kk][tc + 16u * j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);  // every thread has read the step before the next load
    }
    for (uint i = 0; i < 4u; ++i) {
        const uint t = t0 + tr + 16u * i;
        for (uint j = 0; j < 4u; ++j) {
            const uint n = n0 + tc + 16u * j;
            if (t < R && n < p.n) y[ulong(t) * p.n + n] = cuda_nan(acc[i][j]);
        }
    }
}
