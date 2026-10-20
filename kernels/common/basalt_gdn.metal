// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/dec_gdn_rec.cu:30-307,
// src/kernels/cuda/verify_kernels.cu:29-57, :113-213 and :377-461 and include/basalt/kernels/dec_gdn.hpp:16-39 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), written in the rounding sequence nvcc compiles
// for them (docs/research/basalt-nvcc/dec_gdn.md, docs/research/basalt-nvcc/verify_gdn.md).
//
// basalt's GDN decode window on Metal: one function per GDN step, called alike by dec_gdn_rec and by the per-stage
// kernels dec_gdn_parity grades it against (gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi: a test-only
// reference side), so that its bitwise checks hold here. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal,
// basalt_ieee_f64.metal, basalt_q8_1.metal and basalt_libdevice.metal ahead of it (monolith/kernels.py's
// basalt_gdn_source()).
#pragma STDC FP_CONTRACT OFF

// Every entry point's record (monolith/kernels.py's basalt_gdn_params): the window's width T (1..8, a program
// constant), the channels C = 2 * 128 * h_k + 128 * h_v (q, k, then v), n_embd (2560), the key and value heads, and
// the norms' eps.
struct BsGdnParams { uint T; uint C; uint n_embd; uint h_k; uint h_v; float eps; uint pad0; uint pad1; };

// ---- the shared steps --------------------------------------------------------------------------------------------
//
// The sequence nvcc compiles for both files (docs/research/basalt-nvcc/dec_gdn.md section 1, verify_gdn.md sections
// 2-5), with IEEE subnormals: every FFMA an fma inside fma_ieee, every FMUL and FADD an ieee_mul and ieee_add, every
// __expf an expf_ieee (__expf(-x) on sign_flip(x)), each its own call.

// The conv sum of one channel and token (dec_gdn_rec.cu:64, verify_kernels.cu:45): the taps v0, v1, v2 (oldest first)
// and the token's own x against the weights w0..w3. nvcc rounds tap 1's product alone and fuses the other three into
// it in tap order: FMUL p, w1, v1, then three FFMA (dec_gdn.md section 6d).
static inline float gdn_conv(float v0, float v1, float v2, float x, float w0, float w1, float w2, float w3) {
    return fma_ieee(x, w3, fma_ieee(v2, w2, fma_ieee(v0, w0, ieee_mul(v1, w1))));
}

// SiLU of the conv sum (dec_gdn_rec.cu:65, verify_kernels.cu:46): s / (1 + __expf(-s)) by div.rn, the correctly
// rounded quotient, subnormal ones kept; the divisor is at least 1, infinite or NaN (ieee_div's domain).
static inline float gdn_silu(float s) { return ieee_div(s, ieee_add(1.0f, expf_ieee(sign_flip(s)))); }

// ((p0 + p1) + p2) + p3, three FADD: four partial sums in their row-group, SIMD-group or rank order.
static inline float gdn_sum4(float p0, float p1, float p2, float p3) {
    return ieee_add(ieee_add(ieee_add(p0, p1), p2), p3);
}

// The alpha or beta logit of one row and token (dec_gdn_rec.cu:89-108; verify_kernels.cu:183-203, the unrolled
// kernel the launcher takes at n_embd = 2560, where the loop form's order is the same): lane `lane`'s chain over the
// row's words j = lane, lane + 32, ... below n8, each 8 bf16 weights against the token's activations 8j .. 8j + 7 in
// element order (a word's low half is the even element), from +0; then the butterfly, whose sum every lane returns.
static inline float gdn_ab_dot(device const uint4* w, device const float4* xt, uint lane, uint n8) {
    float v = 0.0f;
    for (uint j = lane; j < n8; j += 32u) {
        const uint4 wj = w[j];
        const float4 xa = xt[2u * j], xb = xt[2u * j + 1u];
        v = fma_ieee(as_type<float>(wj.x << 16), xa.x, v);
        v = fma_ieee(as_type<float>(wj.x & 0xffff0000u), xa.y, v);
        v = fma_ieee(as_type<float>(wj.y << 16), xa.z, v);
        v = fma_ieee(as_type<float>(wj.y & 0xffff0000u), xa.w, v);
        v = fma_ieee(as_type<float>(wj.z << 16), xb.x, v);
        v = fma_ieee(as_type<float>(wj.z & 0xffff0000u), xb.y, v);
        v = fma_ieee(as_type<float>(wj.w << 16), xb.z, v);
        v = fma_ieee(as_type<float>(wj.w & 0xffff0000u), xb.w, v);
    }
    return warp_sum_ieee(v);
}

// beta (dec_gdn_rec.cu:109, verify_kernels.cu:151, :206): 1 / (1 + __expf(-v)) by rcp.rn, the correctly rounded
// reciprocal, subnormal for v below about -87.34, where the divisor passes 2^126 (ieee_div's domain, not ieee_rcp's).
static inline float gdn_beta(float v) { return ieee_div(1.0f, ieee_add(1.0f, expf_ieee(sign_flip(v)))); }

// The gate, the decay's logarithm (dec_gdn_rec.cu:110-112, verify_kernels.cu:153-155, :208-210): u = v + dt; the
// softplus, u itself above 20, else __nv_log1pf of __expf(u) (a subnormal exponential comes back as it is); times sa.
static inline float gdn_gate(float v, float dt, float sa) {
    const float u = ieee_add(v, dt);
    const float sp = u > 20.0f ? u : log1pf_nv(expf_ieee(u));
    return ieee_mul(sp, sa);
}
