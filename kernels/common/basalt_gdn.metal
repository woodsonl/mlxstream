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

// ---- the recurrence and the output -------------------------------------------------------------------------------
//
// dec_gdn_rec.cu:245-297 with norm_q8 and sq32 (:115-139), verify_kernels.cu:405-441 (dec_gdn.md section 2,
// verify_gdn.md section 6). A thread holds rows 32 rg .. 32 rg + 31 of one column of the head's state, s[r] in
// registers; k and q point at the same rows of the token's L2-normed key and query in threadgroup memory. The decay is
// g = expf_ieee(gate) (dec_gdn_rec.cu:248, verify_kernels.cu:407); kv_col and the head's sum of squares are gdn_sum4 of
// the four row groups' and column groups' partials, in that order.

// kv, the thread's part of the column's k-weighted state sum (dec_gdn_rec.cu:251, verify_kernels.cu:410): one FFMA
// chain from +0, r ascending.
static inline float gdn_kv(thread const float* s, threadgroup const float* k) {
    float kv = 0.0f;
    for (uint r = 0; r < 32u; ++r) kv = fma_ieee(s[r], k[r], kv);
    return kv;
}

// delta = (v - g kv_col) b (dec_gdn_rec.cu:255, verify_kernels.cu:414): ptxas contracts the product into the
// difference, FFMA d, -g, kv_col, v (-g by its bits), then FMUL by b.
static inline float gdn_delta(float g, float kv_col, float v, float b) {
    return ieee_mul(fma_ieee(sign_flip(g), kv_col, v), b);
}

// The state update and o (dec_gdn_rec.cu:259-260, verify_kernels.cu:423-424), per row r ascending: kd = k_r delta
// rounded alone (FMUL), s[r] = g s[r] + kd (FFMA), then o's FFMA chain, from +0, on the updated s[r] and q_r. s holds
// the updated rows; returns the thread's o.
static inline float gdn_update(thread float* s, threadgroup const float* k, threadgroup const float* q, float g,
                               float delta) {
    float o = 0.0f;
    for (uint r = 0; r < 32u; ++r) {
        const float kd = ieee_mul(k[r], delta);
        s[r] = fma_ieee(g, s[r], kd);
        o = fma_ieee(s[r], q[r], o);
    }
    return o;
}

// oc, the column's output (dec_gdn_rec.cu:265-266, verify_kernels.cu:431): the four row groups' o in order, times
// rsqrtf((float) S), which nvcc folds to RN(2^-3.5) = 0x3DB504F3 (never fast::rsqrt(128.0f)).
static inline float gdn_oc(float o0, float o1, float o2, float o3) {
    return ieee_mul(gdn_sum4(o0, o1, o2, o3), as_type<float>(0x3DB504F3u));
}

// The sum of squares of a SIMD-group's 32 columns (sq32, dec_gdn_rec.cu:136-137; verify_kernels.cu:432-434): FMUL,
// then the butterfly's FADDs; every lane returns the sum.
static inline float gdn_sq32(float oc) { return warp_sum_ieee(ieee_mul(oc, oc)); }

// The gated RMS norm of one column (norm_q8, dec_gdn_rec.cu:119-120; verify_kernels.cu:439-441): rsqrtf(ss / 128 +
// eps), the division made the exact multiply by 2^-7 and fused into the add (FFMA); then ((oc scale) gam) times the
// sigmoid of z, gdn_beta's sequence, left to right.
static inline float gdn_out(float oc, float ss, float eps, float gam, float z) {
    return ieee_mul(ieee_mul(ieee_mul(oc, rsqrt_ieee(fma_ieee(ss, 0.0078125f, eps))), gam), gdn_beta(z));
}

// ---- the per-stage kernels: dec_gdn_parity's reference side (test-only) --------------------------------------------

// gdn_ab_multi (verify_kernels.cu:113-213, the launcher :873-898): the window's gate and beta for every value head
// r and token t < T, gate[t h_v + r] from the alpha row r of x row t (softplus(v + dt[r]) ssm_a[r], the decay's
// logarithm) and beta[t h_v + r] = sigmoid(v) from the beta row r. Grid h_v threadgroups of 64 threads: SIMD-group sg
// of threadgroup tg takes row 2 tg + sg, rows below h_v alpha, the next h_v beta. One kernel for basalt's five
// instantiations (gdn_ab_multi_kernel<4/8>, gdn_ab_multi_unrolled_kernel<4/6/8, 10>), whose token sums are one chain
// in one order whatever MAX_T (verify_gdn.md section 5): here each token's dot runs on its own, then lane 0 stores.
kernel void gdn_ab_multi(device const float4* x [[buffer(0)]], device const uint4* wa [[buffer(1)]],
                         device const uint4* wb [[buffer(2)]], device const float* dt [[buffer(3)]],
                         device const float* ssm_a [[buffer(4)]], device float* gate [[buffer(5)]],
                         device float* beta [[buffer(6)]], constant BsGdnParams& p [[buffer(16)]],
                         uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = 2u * tg + sg;
    if (row >= 2u * p.h_v) return;                                // a whole SIMD-group
    const bool is_beta = row >= p.h_v;
    const uint r = is_beta ? row - p.h_v : row, n8 = p.n_embd / 8u, T = min(p.T, 8u);
    device const uint4* w = (is_beta ? wb : wa) + r * n8;
    for (uint t = 0; t < T; ++t) {
        const float a = gdn_ab_dot(w, x + 2u * n8 * t, lane, n8);    // x row t: n_embd floats, 2 n8 float4
        if (lane == 0u) {
            if (is_beta) beta[t * p.h_v + r] = cuda_nan(gdn_beta(a));
            else gate[t * p.h_v + r] = cuda_nan(gdn_gate(a, dt[r], ssm_a[r]));
        }
    }
}
