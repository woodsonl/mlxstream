// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/kernels.cu:726-758,
// include/basalt/kernels/mrope.hpp:48 and include/basalt/kernels/rope_scaling.hpp:149-162
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), written in the rounding sequence nvcc
// compiles for them (docs/research/basalt-nvcc/prefill_kernels.md).
//
// basalt's prompt-path attention row kernels (prefill_kernels.md section 4). kernels.cu is IEEE-built: every product,
// quotient and fused product that can see or make a subnormal goes through ieee_mul, ieee_div or fma_ieee (this GPU
// flushes subnormals), with fma_ieee exactly where the SASS has an FFMA; rsqrtf is fast::rsqrt (invariant 5), and the
// library powf, cosf and sinf are libdevice's, transcribed (nv_powf, nv_cosf, nv_sinf). Needs basalt_fp.metal,
// basalt_ieee.metal (ieee_mul, sign_flip), basalt_q8_1.metal (ieee_div, block_sum), basalt_f64.metal,
// basalt_ieee_f64.metal (fma_ieee), basalt_libdevice.metal, PROMPT_ROWS and the StepState struct ahead of it
// (monolith/kernels.py's basalt_prompt_qsa_source).
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
// - prompt_rope (:738-758, rope_kernel<false> at scaling None; HEADS, DIM, THETA_SCALE_BITS): 0 x [B][HEADS * DIM],
//   1 y [B][HEADS * DIM]; one threadgroup of 32 per row, thread = pair (basalt's in-place row, here x's into y's).
//   th = float(pos) nv_powf(theta_scale, pair) at the absolute position pos = StepState.position + t (mrope_pos
//   without a table), a plain product: both factors are normal or zero on the handler's theta_scale range; scaling
//   None's freq_scale and mscale multiplies by 1 are the identity and left out; c = nv_cosf(th), s = nv_sinf(th),
//   normal or zero. Then the SASS's rotation of a = x[pair], b = x[pair + 32]: y[pair] = fma(c, a, -fl(b s)) and
//   y[pair + 32] = fma(c, b, fl(a s)), by ieee_mul and fma_ieee (a and b are any floats); the negation is the FFMA's
//   operand sign, sign_flip on the bits (unary minus on a subnormal is not measured here). Dims 64 on are copied by
//   their bits (the port addition: basalt leaves them in place).
//
// Stores: fp32 through cuda_nan (sm_120's NaN 0x7fffffff), but for rope's copied words.
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

#if defined(HEADS) && defined(DIM) && defined(THETA_SCALE_BITS)
kernel void prompt_rope(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        device const StepState* st [[buffer(15)]],
                        uint row [[threadgroup_position_in_grid]], uint pair [[thread_index_in_threadgroup]]) {
    if (st->done || row >= prompt_rows(st) * uint(HEADS)) return;        // the whole threadgroup (no barrier here)
    const int pos = int(st->position + row / uint(HEADS));               // (int) (pos0 + t): mrope_pos, no table
    const float th = float(pos) * nv_powf(as_type<float>(THETA_SCALE_BITS), float(pair));
    const float c = nv_cosf(th), s = nv_sinf(th);
    device const float* p = x + row * uint(DIM);
    device float* o = y + row * uint(DIM);
    const float a = p[pair], b = p[pair + 32];
    o[pair] = cuda_nan(fma_ieee(c, a, sign_flip(ieee_mul(b, s))));      // a * c - b * s: FMUL, then FFMA
    o[pair + 32] = cuda_nan(fma_ieee(c, b, ieee_mul(a, s)));            // a * s + b * c
    for (uint j = 64u + pair; j < uint(DIM); j += 32u)                  // the rest of the head, by its bits
        ((device uint*) o)[j] = ((device const uint*) p)[j];
}
#endif
