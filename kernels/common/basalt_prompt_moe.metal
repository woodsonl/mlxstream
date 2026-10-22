// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/kernels.cu:21-44, :626-664, :698-704, :713-723,
// :929-947, :961-964 and :995-999 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), written in
// the rounding sequence nvcc compiles for them (docs/research/basalt-nvcc/prefill_kernels.md).
//
// basalt's prompt-path MoE element-wise kernels (kernels.cu:625-723, launched by :929-999; prefill_kernels.md
// section 3). kernels.cu is IEEE-built: every product and quotient that can see or make a subnormal goes through
// ieee_mul or ieee_div (this GPU flushes subnormals), and __expf(-x) is expf_ieee(sign_flip(x)). monolith/kernels.py's
// basalt_prompt_moe_source() places this file behind basalt_fp.metal, basalt_ieee.metal, basalt_q8_1.metal,
// basalt_f64.metal, basalt_ieee_f64.metal, basalt_libdevice.metal, PROMPT_ROWS and the StepState struct.
//
// Every entry reads the chunk's rows R = prompt_rows(st) from StepState (slot 15) and returns when st->done or its
// element is at or past R's; grids are encoded at the compiled bound B (ROWS_MAX) in threadgroups of 256
// (monolith/kernels.py's basalt_prompt_moe_launch). Each output depends only on its own token's inputs: not on R or
// on the token's index in the chunk.
//
// - basalt_prompt_swiglu_pair (:698-704, hf_sat :41-44; launched by :961-964): 0 g [B][MOE_F], 1 u [B][MOE_F], 2 h16
//   [B][MOE_F] (FP16): the shared expert's h16 = hf_sat(g / (1 + __expf(-g)) * u), one thread per element, i < R
//   MOE_F. For a = g[i]: e = expf_ieee(sign_flip(a)); y = 1 + e plain (at least 1, +inf or NaN: 1 + a subnormal e is
//   1 either way); q = ieee_div(a, y) (div.rn: q is subnormal for |a| below 2^-125); p = ieee_mul(q, u[i]) (a separate
//   FMUL: a subnormal q times an infinite u is +-inf, and times a u above 2^100 can reach FP16's range); then hf_sat:
//   a NaN p stays NaN, any other is clamped to [-65504, 65504] by max and min, which choose against the normal bounds
//   as fmaxf and fminf do and return a subnormal p's own bits.
//
// Stores: FP16 as cuda_nan_f16(f16_bits(x)) (:40's hf: to nearest even, an fp32 subnormal to the zero of its sign, a
// NaN 0x7fff). Macros: basalt_prompt_moe_launch's MOE_E (experts, 512), MOE_K (picks per token, 10), MOE_F (the
// shared expert's width, 640) and MOE_N (the residual's width, basalt's N = 2560), and the row macros PROMPT_ROWS reads
// (T_SRC, ROWS_MAX, T_STATIC_ROWS), which the emitter's prompt_rows gives.
#pragma STDC FP_CONTRACT OFF

kernel void basalt_prompt_swiglu_pair(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                                      device ushort* h16 [[buffer(2)]], device const StepState* st [[buffer(15)]],
                                      uint i [[thread_position_in_grid]]) {
    if (st->done || i >= prompt_rows(st) * uint(MOE_F)) return;                // :701
    const float a = g[i];
    const float e = expf_ieee(sign_flip(a));                                    // __expf(-a)
    const float y = 1.0f + e;
    const float q = ieee_div(a, y);                                             // div.rn
    const float p = ieee_mul(q, u[i]);
    h16[i] = cuda_nan_f16(f16_bits(isnan(p) ? p : min(max(p, -65504.0f), 65504.0f)));   // hf_sat (:44), hf (:40)
}
