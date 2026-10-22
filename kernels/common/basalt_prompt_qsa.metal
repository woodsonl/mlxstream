// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/kernels.cu:726-734
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), written in the rounding sequence nvcc
// compiles for them (docs/research/basalt-nvcc/prefill_kernels.md).
//
// basalt's prompt-path attention row kernels (prefill_kernels.md section 4). kernels.cu is IEEE-built: every product,
// quotient and fused product that can see or make a subnormal goes through ieee_mul, ieee_div or fma_ieee (this GPU
// flushes subnormals), with fma_ieee exactly where the SASS has an FFMA; rsqrtf is fast::rsqrt (invariant 5). Needs
// basalt_fp.metal, basalt_ieee.metal, basalt_q8_1.metal (ieee_div, block_sum), basalt_f64.metal,
// basalt_ieee_f64.metal (fma_ieee), PROMPT_ROWS and the StepState struct ahead of it (monolith/kernels.py's
// basalt_prompt_qsa_source).
//
// Every entry reads the chunk's token count R = prompt_rows(st) from StepState (slot 15) and returns, before any
// barrier, when st->done or its row belongs to a token at or past R; the grid is sized for the bound ROWS_MAX. No
// params record: widths and constants are macros, a float one as its bits (invariant 4). Each entry sits under #if of
// every macro it reads, so a compile with one entry's macros builds that entry alone and a dropped macro fails at
// pipeline creation. A row's output depends only on its own input row and the weights: not on R or the row's place in
// the chunk.
//
// - prompt_rms_rows (:726-734; COLS, HEADS, EPS_BITS): 0 x [B][HEADS * COLS], 1 w [COLS], 2 y [B][HEADS * COLS]; one
//   threadgroup of 256 per row, row = t HEADS + h (basalt's in-place row, here x's into y's). Thread k sums x[c]^2,
//   c = k + 256 j ascending, by fma_ieee from +0 (nvcc contracts ss += r * r); block_sum; q = ieee_div(S, COLS)
//   (div.rn: the quotient can be subnormal); q + eps plain (a q below 2^-126, which Metal's add flushes, cannot move
//   fl(q + eps) for eps at least 2^-102, the handler's floor) and fast::rsqrt (its argument is at least eps, never
//   subnormal); y = fl(fl(s x) w), two ieee_mul (s is at most eps^-1/2, so a small x gives a subnormal product).
//
// Stores: fp32 through cuda_nan (sm_120's NaN 0x7fffffff).
#pragma STDC FP_CONTRACT OFF

#if defined(COLS) && defined(HEADS) && defined(EPS_BITS)
kernel void prompt_rms_rows(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                            device float* y [[buffer(2)]], device const StepState* st [[buffer(15)]],
                            uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[32];
    if (st->done || row >= prompt_rows(st) * uint(HEADS)) return;        // the whole threadgroup, before any barrier
    device const float* r = x + row * uint(COLS);
    float ss = 0.0f;
    for (uint c = tid; c < uint(COLS); c += 256u) ss = fma_ieee(r[c], r[c], ss);
    const float q = ieee_div(block_sum(ss, sh, tid, 256u), float(COLS));
    const float s = fast::rsqrt(q + as_type<float>(EPS_BITS));
    threadgroup_barrier(mem_flags::mem_threadgroup);                     // :732's __syncthreads (no effect on bits)
    for (uint c = tid; c < uint(COLS); c += 256u) y[row * uint(COLS) + c] = cuda_nan(ieee_mul(ieee_mul(s, r[c]), w[c]));
}
#endif
