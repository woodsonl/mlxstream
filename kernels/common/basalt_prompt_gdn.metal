// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/kernels.cu:206-216 (and :39, :235, :485)
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
//
// basalt's prompt-path GDN kernels on Metal: gdn_gates_kernel. kernels.cu is built without fast math
// (CMakeLists.txt:285), so every kernel keeps IEEE subnormals and is written in the rounding sequence nvcc compiles for
// it (docs/research/basalt-nvcc/prefill_kernels.md section 2), through the gdn lane's shared GDN steps
// (basalt_gdn_steps.metal), which this file calls and never copies. Needs basalt_fp.metal, basalt_ieee.metal,
// basalt_f64.metal, basalt_ieee_f64.metal, basalt_q8_1.metal, basalt_libdevice.metal, basalt_gdn_steps.metal, the
// StepState struct and PROMPT_ROWS ahead of it (monolith/kernels.py's basalt_prompt_gdn_source()).
//
// Every entry reads the chunk's rows R = prompt_rows(st) from StepState at slot 15 (invariant 4; grids are encoded at
// the program's bound B) and opens with its guard, st->done or its first row or element at or past its bound, before
// it reads an input: a replay after the request is done writes nothing.
//
// Macros: the model's constants from the handler, GDN_C (the conv channels), GDN_HK (the q and k heads), GDN_HV (the
// value heads) and GDN_EPS_BITS (the norms' eps as float bits); PROMPT_ROWS' T_SRC, ROWS_MAX and T_STATIC_ROWS.
#pragma STDC FP_CONTRACT OFF

#define PGDN_TILE 64u        // CONV_TILE (kernels.cu:235): the tokens of one conv tile
#define PGDN_WC 16u          // GDN_WC (:485): the state columns of one wl threadgroup
#define PGDN_WNCB 8u         // GDN_WNCB = S / GDN_WC (:485): the wl threadgroups of one key head

// gdn_gates_kernel (kernels.cu:207-216; gdn_gates, :901-903): one thread per element i = t * GDN_HV + h of the chunk's
// rows, gate = softplus(alpha + dt[h]) * ssm_a[h] and beta = sigm(the beta logit) (:39), as the gdn lane's gdn_gate
// and gdn_beta compute them (prefill_kernels.md section 2's rows: FADD, u itself above 20, else log1pf_nv of
// expf_ieee(u), FMUL; expf_ieee(sign_flip(x)), FADD 1, the correctly rounded reciprocal), each stored through
// cuda_nan. basalt reads ab[t * 96 + h] and ab[t * 96 + 48 + h] (prefill.cpp:1587-1588); here the two products are
// two [B][GDN_HV] buffers with the same values. Grid: ceil(B * GDN_HV / 256) threadgroups of 256; no barrier.
kernel void gdn_gates_kernel(device const float* alpha [[buffer(0)]], device const float* beta_logit [[buffer(1)]],
                             device const float* dt [[buffer(2)]], device const float* ssm_a [[buffer(3)]],
                             device float* gate [[buffer(4)]], device float* beta [[buffer(5)]],
                             device const StepState* st [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if (st->done || i >= prompt_rows(st) * GDN_HV) return;
    const uint h = i % GDN_HV;
    gate[i] = cuda_nan(gdn_gate(alpha[i], dt[h], ssm_a[h]));
    beta[i] = cuda_nan(gdn_beta(beta_logit[i]));
}
