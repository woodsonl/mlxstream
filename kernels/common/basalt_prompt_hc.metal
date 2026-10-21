// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/kernels.cu:39, :45-58, :85-170 and :192-204
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
//
// basalt's prompt-path hyper-connection kernels (kernels.cu:85-204, launched by :862-900) in the rounding sequence
// nvcc compiles for them (docs/research/basalt-nvcc/prefill_kernels.md section 1). kernels.cu is IEEE-built: every
// product, sum and fused product that can see or make a subnormal goes through ieee_mul, ieee_add, ieee_div or
// fma_ieee (this GPU flushes subnormals), with fma_ieee exactly where the SASS has an FFMA; __expf is
// expf_ieee(sign_flip(x)) and rsqrtf fast::rsqrt. The residual R holds B tokens of four streams of HC_N floats,
// stream-major: R[t][c * HC_N + d], row (t, c) = 4 t + c. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal,
// basalt_ieee_f64.metal, basalt_q8_1.metal (block_sum), PROMPT_ROWS and the StepState struct ahead of it
// (monolith/kernels.py's basalt_prompt_hc_source).
//
// Every entry reads the chunk's rows nr = prompt_rows(st) from StepState (slot 15) and returns, before any barrier,
// when st->done or its row or element is at or past nr's. A dispatch covers one static row slice [row0, row0 + S),
// row0 from its params record (slot 7): a row kernel runs 4 S threadgroups of 256, one per row, row = 4 row0 +
// threadgroup; an element kernel runs ceil(S * W / 256) threadgroups of 256 over i = row0 * W + thread, W its row
// width. Each output depends only on its own token's inputs and the weights: not on nr, the slice or the token's index.
//
// - prompt_gr_norm_rs (:85-102): 0 R, 1 w [4 HC_N], 2 rs [B][4], 3 xn16 [B][4 HC_N] (BF16). Thread k sums r[d]^2,
//   d = k + 256 j, j ascending, by fma_ieee from +0 (nvcc contracts ss += r * r); block_sum; then the quotient by
//   HC_N and + eps plain (a quotient below 2^-126, which Metal's division flushes, cannot move fl(q + eps) for eps at
//   least 2^-102, the handler's floor) and fast::rsqrt (its argument is at least eps, never subnormal); rs[row] by
//   thread 0; xn16 = bf(fl(fl(r * rs) * w[c])), two ieee_mul.
// - prompt_gr_mix_r (:103-124): 0 R, 1 rs [B][4] (gr_norm_rs's), 2 w, 3 gated [B][4 HC_N], 4 mixed [B][HC_N], 5
//   mixed_bf [B][HC_N] (BF16; declared and stored only when MIX_BF is 1: the drafter's KV pass binds no BF16 image),
//   6 mixed_h [B][HC_N] (FP16). Thread i = t HC_N + d: for c = 0..3 in order, x = fl(fl(R * rs) * w[c]) by two
//   ieee_mul (gr_norm_rs's product, bit for bit) and s = fma_ieee(x, sigm(gated), s) from +0 (the SASS's first FFMA
//   adds RZ, which turns a -0 product into +0); then s = ieee_mul(s, 0.25f), the / HC.
// - prompt_gr_silu (:161-170): 0 lo [B][HC_LR], 1 lo16 [B][HC_LR] (BF16). x = ieee_mul(lo, 0.25f), the / HC; y = 1 +
//   expf_ieee(sign_flip(x)) plain (1 + a subnormal is 1 either way); lo16 = bf(ieee_div(x, y)), the source's true
//   division (div.rn), not x * sigm(x).
// - prompt_gr_write (:192-198): 0 R, 1 bo [B][HC_N], 2 inj [B][4], 3 R' [B][4 HC_N]. Thread i = t HC_D + c HC_N + d:
//   sg = sigm(inj[t][c] * 0.25f), the quotient plain (it feeds only __expf(-q), which is 1 for any |q| below 2^-100,
//   flushed or not; for a normal q the SASS's two FMULs, by -0.25 and log2 e, round as -q times log2 e does); sc =
//   ieee_add(sg, sg) (2.0f * sg is an FADD; sg is subnormal for inj below about -349, sc need not be); R' =
//   fma_ieee(bo[t][d], sc, R).
// - prompt_gr_write_norm_rs (:125-160): 0 R, 1 bo, 2 inj, 3 w_next [4 HC_N], 4 R', 5 rs [B][4], 6 xn16 [B][4 HC_N]
//   (BF16). gr_write on row (t, c), its HC_N / 256 values kept in registers, then gr_norm_rs's chain on them with
//   w_next: the same mapping, +0 seed, block_sum and two products, so R' is gr_write's and rs and xn16 are gr_norm_rs's
//   on that R' (:125-127), bit for bit.
//   In both writes R and R' are two SSA values in two buffers (the arena never gives an output its own op's input
//   slot), but one buffer would do: each element is read and written by the same thread, as in basalt's in-place
//   kernels.
// - prompt_gr_broadcast (:199-204): 0 e [B][HC_N], 1 R' [B][4 HC_N]: each of a token's four streams is its embedding
//   row, copied as 32-bit words (NaN payloads and subnormals unchanged).
//
// Stores: fp32 through cuda_nan (sm_120's NaN 0x7fffffff), BF16 as bf16_rne_nonan(cuda_nan(x)) (kernels.cu:31-35's bf,
// which makes that NaN 0x8000), FP16 as cuda_nan_f16(f16_bits(x)) (:40's hf: to nearest even, a NaN 0x7fff). Macros:
// HC_N (a stream's width), HC_LR (the low-rank width), HC_EPS (eps as a float expression), MIX_BF (0 or 1: mix's BF16
// image), and PROMPT_ROWS' T_SRC, ROWS_MAX and T_STATIC_ROWS. basalt's HC = 4 is a literal here.
#pragma STDC FP_CONTRACT OFF

#define HC_D (4u * uint(HC_N))   // a token's four streams (basalt's D)

struct PromptHcParams { uint row0; };   // the slice's first row: compile-time, one record per slice

// sigm (:39), 1 / (1 + __expf(-x)): __expf(-x) as expf_ieee(sign_flip(x)); 1 + e plain (1 + a subnormal e is 1 either
// way); the reciprocal is rcp.rn, IEEE's quotient, which ieee_div gives for a divisor of at least 1.
static inline float sigm(float x) { return ieee_div(1.0f, 1.0f + expf_ieee(sign_flip(x))); }

kernel void prompt_gr_norm_rs(device const float* R [[buffer(0)]], device const float* w [[buffer(1)]],
                              device float* rs_out [[buffer(2)]], device ushort* xn16 [[buffer(3)]],
                              constant PromptHcParams& p [[buffer(7)]], device const StepState* st [[buffer(15)]],
                              uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const uint row = 4u * p.row0 + tg;                                  // t * 4 + c
    if (st->done || row >= 4u * prompt_rows(st)) return;                // the whole threadgroup, before any barrier
    const uint c = row % 4u;
    device const float* r = R + row * uint(HC_N);
    float ss = 0.0f;
    for (uint d = tid; d < uint(HC_N); d += 256u) ss = fma_ieee(r[d], r[d], ss);
    const float rs = fast::rsqrt(block_sum(ss, sh, tid, 256u) / float(HC_N) + HC_EPS);
    if (tid == 0u) rs_out[row] = cuda_nan(rs);
    for (uint d = tid; d < uint(HC_N); d += 256u)
        xn16[row * uint(HC_N) + d] = bf16_rne_nonan(cuda_nan(ieee_mul(ieee_mul(r[d], rs), w[c * uint(HC_N) + d])));
}

kernel void prompt_gr_mix_r(device const float* R [[buffer(0)]], device const float* rs [[buffer(1)]],
                            device const float* w [[buffer(2)]], device const float* gated [[buffer(3)]],
                            device float* mixed [[buffer(4)]],
#if MIX_BF
                            device ushort* mixed_bf [[buffer(5)]],
#endif
                            device ushort* mixed_h [[buffer(6)]], constant PromptHcParams& p [[buffer(7)]],
                            device const StepState* st [[buffer(15)]], uint gid [[thread_position_in_grid]]) {
    const uint i = p.row0 * uint(HC_N) + gid;                           // t * HC_N + d
    if (st->done || i >= prompt_rows(st) * uint(HC_N)) return;
    const uint t = i / uint(HC_N), d = i % uint(HC_N);
    float s = 0.0f;                                                     // a -0 first product gives +0, as in the SASS
    for (uint c = 0u; c < 4u; ++c) {
        const uint j = t * HC_D + c * uint(HC_N) + d;
        const float x = ieee_mul(ieee_mul(R[j], rs[t * 4u + c]), w[c * uint(HC_N) + d]);   // gr_norm_rs's product
        s = fma_ieee(x, sigm(gated[j]), s);
    }
    s = ieee_mul(s, 0.25f);                                             // s /= HC: FMUL by 0.25
    mixed[i] = cuda_nan(s);
#if MIX_BF
    mixed_bf[i] = bf16_rne_nonan(cuda_nan(s));
#endif
    mixed_h[i] = cuda_nan_f16(f16_bits(s));
}

kernel void prompt_gr_silu(device const float* lo [[buffer(0)]], device ushort* lo16 [[buffer(1)]],
                           constant PromptHcParams& p [[buffer(7)]], device const StepState* st [[buffer(15)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint i = p.row0 * uint(HC_LR) + gid;
    if (st->done || i >= prompt_rows(st) * uint(HC_LR)) return;
    const float x = ieee_mul(lo[i], 0.25f);                             // lo / HC: FMUL by 0.25
    const float y = 1.0f + expf_ieee(sign_flip(x));                     // 1 + __expf(-x)
    lo16[i] = bf16_rne_nonan(cuda_nan(ieee_div(x, y)));                 // div.rn: x / y, not x * sigm(x)
}

kernel void prompt_gr_write(device const float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                            device const float* inj [[buffer(2)]], device float* R_out [[buffer(3)]],
                            constant PromptHcParams& p [[buffer(7)]], device const StepState* st [[buffer(15)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint i = p.row0 * HC_D + gid;                                 // t * HC_D + c * HC_N + d
    if (st->done || i >= prompt_rows(st) * HC_D) return;
    const uint t = i / HC_D, c = i % HC_D / uint(HC_N), d = i % uint(HC_N);
    const float sg = sigm(inj[t * 4u + c] * 0.25f);                    // inj / HC plain: feeds only __expf(-q)
    R_out[i] = cuda_nan(fma_ieee(bo[t * uint(HC_N) + d], ieee_add(sg, sg), R[i]));   // 2.0f * sg is FADD sg, sg
}

kernel void prompt_gr_write_norm_rs(device const float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                                    device const float* inj [[buffer(2)]], device const float* w [[buffer(3)]],
                                    device float* R_out [[buffer(4)]], device float* rs_out [[buffer(5)]],
                                    device ushort* xn16 [[buffer(6)]], constant PromptHcParams& p [[buffer(7)]],
                                    device const StepState* st [[buffer(15)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    const uint row = 4u * p.row0 + tg;                                  // t * 4 + c
    if (st->done || row >= 4u * prompt_rows(st)) return;                // the whole threadgroup, before any barrier
    const uint t = row / 4u, c = row % 4u;
    const float sg = sigm(inj[t * 4u + c] * 0.25f);                    // gr_write's sc, the same sites
    const float sc = ieee_add(sg, sg);
    float v[HC_N / 256];                                                // basalt's GRW_PER: the row stays in registers
    float ss = 0.0f;
    for (uint j = 0u; j < uint(HC_N) / 256u; ++j) {                     // d = tid + 256 j, gr_norm_rs's order
        const uint d = tid + 256u * j;
        const float x = fma_ieee(bo[t * uint(HC_N) + d], sc, R[row * uint(HC_N) + d]);
        R_out[row * uint(HC_N) + d] = cuda_nan(x);
        v[j] = x;
        ss = fma_ieee(x, x, ss);
    }
    const float rs = fast::rsqrt(block_sum(ss, sh, tid, 256u) / float(HC_N) + HC_EPS);
    if (tid == 0u) rs_out[row] = cuda_nan(rs);
    for (uint j = 0u; j < uint(HC_N) / 256u; ++j) {
        const uint d = tid + 256u * j;
        xn16[row * uint(HC_N) + d] = bf16_rne_nonan(cuda_nan(ieee_mul(ieee_mul(v[j], rs), w[c * uint(HC_N) + d])));
    }
}

kernel void prompt_gr_broadcast(device const uint* e [[buffer(0)]], device uint* R_out [[buffer(1)]],
                                constant PromptHcParams& p [[buffer(7)]], device const StepState* st [[buffer(15)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint i = p.row0 * HC_D + gid;
    if (st->done || i >= prompt_rows(st) * HC_D) return;
    R_out[i] = e[i / HC_D * uint(HC_N) + i % uint(HC_N)];
}
