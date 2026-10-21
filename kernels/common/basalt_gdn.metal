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

// gdn_conv_l2_multi (verify_kernels.cu:29-57, the launcher :817-826, t_begin 0): the window's h for every channel c
// and token t < T, h[t C + c], the conv of the taps [history | qkv rows < t] and qkv row t, SiLU'd; on the q and k
// heads (threadgroups below 2 h_k) L2-normed over the head's 128 channels. The history is the state's conv tail,
// [C][3] oldest first after the S x h_v x S recurrent state, read only. Grid (C / 128, T) threadgroups of 128 threads,
// one head of 128 channels each, SIMD-group sg holding channels 32 sg .. 32 sg + 31 (CUDA's warps).
kernel void gdn_conv_l2_multi(device const float* state [[buffer(0)]], device const float* qkv [[buffer(1)]],
                              device const float* w [[buffer(2)]], device float* h [[buffer(3)]],
                              constant BsGdnParams& p [[buffer(16)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    const uint t = tg.y, c = 128u * tg.x + 32u * sg + lane;
    if (t >= min(p.T, 8u)) return;                                 // the whole threadgroup, before the barrier
    device const float* hist = state + 128u * p.h_v * 128u;        // the conv tail
    float win[3];
    for (uint j = 0; j < 3u; ++j) {
        const uint src = t + j;                                    // into [history(3) | qkv rows]
        win[j] = src < 3u ? hist[3u * c + src] : qkv[(src - 3u) * p.C + c];
    }
    const float x = qkv[t * p.C + c];
    device const float* wc = w + 4u * c;
    float y = gdn_silu(gdn_conv(win[0], win[1], win[2], x, wc[0], wc[1], wc[2], wc[3]));
    if (tg.x < 2u * p.h_k) {                                       // threadgroup-uniform
        const float sq = warp_sum_ieee(ieee_mul(y, y));
        if (lane == 0u) part[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        y = ieee_mul(y, rsqrt_ieee(ieee_add(gdn_sum4(part[0], part[1], part[2], part[3]), p.eps)));
    }
    h[t * p.C + c] = cuda_nan(y);
}

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

// gdn_step_norm_multi (verify_kernels.cu:377-461, the launcher :900-916; its verify form: n = T, t_out_begin 0, no
// write-back): the window's y for every value head and token t < T, y[t 128 h_v + 128 head + col], the recurrence
// from the head's state over tokens 0..t (gate, beta and h from gdn_ab_multi and gdn_conv_l2_multi) and the gated RMS
// norm of its read-out. The state is read only. Grid h_v threadgroups of 512 threads, one value head each: thread tid
// = 32 sg + lane (CUDA's linear warp order) holds column col = tid % 128 of the head's state, rows 32 rg .. 32 rg + 31
// (rg = tid / 128), in registers; q/k head head % h_k. basalt's six barriers per token; the SIMD-groups of rg > 0
// butterfly zeros into wsum[4..15], which nothing reads.
kernel void gdn_step_norm_multi(device const float* state [[buffer(0)]], device const float* h [[buffer(1)]],
                                device const float* gate [[buffer(2)]], device const float* beta [[buffer(3)]],
                                device const float* z [[buffer(4)]], device const float* gamma [[buffer(5)]],
                                device float* y [[buffer(6)]], constant BsGdnParams& p [[buffer(16)]],
                                uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sk[128], sq[128], red[4][128], wsum[16];
    const uint tid = 32u * sg + lane, col = tid % 128u, rg = tid / 128u;
    const uint qh = head % p.h_k, qk = 128u * p.h_k, vd = 128u * p.h_v, T = min(p.T, 8u);
    float s[32];
    for (uint r = 0; r < 32u; ++r) s[r] = state[((32u * rg + r) * p.h_v + head) * 128u + col];
    for (uint t = 0; t < T; ++t) {
        device const float* ht = h + t * p.C;
        threadgroup_barrier(mem_flags::mem_threadgroup);           // the previous token is done with sk, sq, red, wsum
        if (tid < 128u) {
            sk[tid] = ht[qk + qh * 128u + tid];
            sq[tid] = ht[qh * 128u + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float g = expf_ieee(gate[t * p.h_v + head]);
        red[rg][col] = gdn_kv(s, sk + 32u * rg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = gdn_sum4(red[0][col], red[1][col], red[2][col], red[3][col]);
        const float delta = gdn_delta(g, kv_col, ht[2u * qk + head * 128u + col], beta[t * p.h_v + head]);
        const float o = gdn_update(s, sk + 32u * rg, sq + 32u * rg, g, delta);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[rg][col] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float oc = rg == 0u ? gdn_oc(red[0][col], red[1][col], red[2][col], red[3][col]) : 0.0f;
        const float sq_part = gdn_sq32(oc);                        // every SIMD-group: uniform control flow
        if (lane == 0u) wsum[sg] = sq_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0u) {
            const uint i = t * vd + head * 128u + col;
            y[i] = cuda_nan(gdn_out(oc, gdn_sum4(wsum[0], wsum[1], wsum[2], wsum[3]), p.eps, gamma[col], z[i]));
        }
    }
}

// ---- dec_gdn_rec: the window ----------------------------------------------------------------------------------------

// The window's most tokens a dec_gdn_rec pipeline keeps in registers (basalt's MAXT; 8 = kDecMaxT unless a program
// of at most that many tokens builds it lower). Every bit is the same whatever it is; the threadgroup arrays keep
// kDecMaxT rows, as RecSmem does.
#ifndef GDN_MAXT
#define GDN_MAXT 8
#endif

// 1 where the caller binds dec_gdn_rec's optional fp32 y at buffer 17 (basalt's `if (a.y)`, dec_gdn_rec.cu:121): the
// tests' build. The engine binds none, as basalt's call site passes none, and builds 0.
#ifndef GDN_Y_BOUND
#define GDN_Y_BOUND 0
#endif

// conv_silu (dec_gdn_rec.cu:53-66): token t of one channel, the conv of [hist(3) | xs rows 0..t-1] with xs row t (the
// channel's raw projections, oldest first) and its SiLU: gdn_conv_l2_multi's sequence on the channel's registers.
static inline float gdn_conv_silu(thread const float* hist, thread const float* w, thread const float* xs, uint t) {
    float win[3];
    for (uint j = 0; j < 3u; ++j) {
        const uint src = t + j;
        win[j] = src < 3u ? hist[src] : xs[src - 3u];
    }
    return gdn_silu(gdn_conv(win[0], win[1], win[2], xs[t], w[0], w[1], w[2], w[3]));
}

// dec_gdn_rec (dec_gdn_rec.cu:141-307, rec_block<MAXT, false> of rec_kernel; host dec_gdn_rec's checks, :355-367, are
// basalt_gdn_params'): the window of T tokens of one GDN layer, from the layer's state (the S x h_v x S recurrent
// state, then the conv history [C][3]; read only) and the token rows of qkv, x and z. It writes rows t < T of the
// window stores the commit replays: win_qkv, the qkv rows as they are (the commit's conv history; basalt's projection
// writes them in place, spec-engine.md section 9.3), win_h, gdn_conv_l2_multi's h, and win_gate and win_beta,
// gdn_ab_multi's gate and beta; and yq, the q8_1 rows of gdn_step_norm_multi's y, y[t 128 h_v + 128 head + col] the
// gated RMS norm of the recurrence's read-out (and, built with GDN_Y_BOUND 1, y itself at buffer 17). Grid h_v
// threadgroups of 512 threads, one value head each: basalt's cluster of four 128-thread blocks as one threadgroup,
// thread tid = 128 rank + 32 rg + lane of SIMD-group sg = 4 rank + rg playing lane `lane` of warp rg of block `rank`,
// the head's state column col = 32 rank + lane, rows 32 rg .. 32 rg + 31. What the four blocks compute alike runs
// once: the 2 T gate jobs over the 16 SIMD-groups, the key head's q and k (head % h_k) on rank 0's 128 threads in
// basalt's partition; SIMD-group 4 rank + 1 takes the v channel of column col, SIMD-group 4 rank the output.
kernel void dec_gdn_rec(device const float* state [[buffer(0)]], device const float* qkv [[buffer(1)]],
                        device const float* w [[buffer(2)]], device const float4* x [[buffer(3)]],
                        device const uint4* wa [[buffer(4)]], device const uint4* wb [[buffer(5)]],
                        device const float* dt [[buffer(6)]], device const float* ssm_a [[buffer(7)]],
                        device const float* z [[buffer(8)]], device const float* gamma [[buffer(9)]],
                        device float* win_qkv [[buffer(10)]], device float* win_h [[buffer(11)]],
                        device float* win_gate [[buffer(12)]], device float* win_beta [[buffer(13)]],
                        device uchar* yq [[buffer(14)]],
#if GDN_Y_BOUND
                        device float* y [[buffer(17)]],
#endif
                        constant BsGdnParams& p [[buffer(16)]], uint head [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qn[8][128], kn[8][128], v[8][128], part[2][8][4], gm[8], bm[8];
    threadgroup float oc[8][128], red_kv[4][4][32], red_o[4][4][32], wsum[8][4];
    const uint tid = 32u * sg + lane, rank = sg / 4u, rg = sg % 4u, col = 32u * rank + lane;
    const uint qh = head % p.h_k, qk = 128u * p.h_k, n8 = p.n_embd / 8u, T = min(p.T, uint(GDN_MAXT));
    const uint cq = qh * 128u + tid, ck = qk + cq, cv = 2u * qk + head * 128u + col;   // cq, ck: rank 0's
    device const float* hist = state + 128u * p.h_v * 128u;        // the conv tail

    // A: the thread's 32 state rows of column col, the head's constants
    float s[32];
    for (uint r = 0; r < 32u; ++r) s[r] = state[((32u * rg + r) * p.h_v + head) * 128u + col];
    const float dth = dt[head], sa = ssm_a[head], gam = gamma[col];

    // B: the (gate, token) jobs, job 2 t the alpha gate of token t, 2 t + 1 its beta; a job's bits are its SIMD-group's
    // alone (gdn_ab_multi's dot)
    for (uint job = sg; job < 2u * T; job += 16u) {
        const uint t = job >> 1;
        const bool is_beta = (job & 1u) != 0u;
        const float a = gdn_ab_dot((is_beta ? wb : wa) + head * n8, x + 2u * n8 * t, lane, n8);
        if (lane == 0u) {
            if (is_beta) bm[t] = gdn_beta(a);
            else gm[t] = gdn_gate(a, dth, sa);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // C, D: the conv and SiLU of the tokens t < T; rank 0's SIMD-group rg holds channels 32 rg .. 32 rg + 31 of the
    // key head's q and k (q, k: basalt's yq, yk), lane 0 keeping the butterflies of their squares
    float xq[GDN_MAXT], xk[GDN_MAXT], q[GDN_MAXT], k[GDN_MAXT];
    if (rank == 0u) {                                              // SIMD-groups 0..3
        float hq[3], hk[3], wq[4], wk[4];
        for (uint j = 0; j < 3u; ++j) { hq[j] = hist[3u * cq + j]; hk[j] = hist[3u * ck + j]; }
        for (uint j = 0; j < 4u; ++j) { wq[j] = w[4u * cq + j]; wk[j] = w[4u * ck + j]; }
        for (uint t = 0; t < GDN_MAXT; ++t) {
            if (t >= T) break;
            xq[t] = qkv[t * p.C + cq];
            xk[t] = qkv[t * p.C + ck];
        }
        for (uint t = 0; t < GDN_MAXT; ++t) {
            if (t >= T) break;
            q[t] = gdn_conv_silu(hq, wq, xq, t);
            k[t] = gdn_conv_silu(hk, wk, xk, t);
            const float sq = warp_sum_ieee(ieee_mul(q[t], q[t])), sk = warp_sum_ieee(ieee_mul(k[t], k[t]));
            if (lane == 0u) {
                part[0][t][rg] = sq;
                part[1][t][rg] = sk;
            }
        }
    }
    if (rg == 1u) {                                                // SIMD-group 4 rank + 1: the v channel cv
        float hv[3], wv[4], xv[GDN_MAXT];
        for (uint j = 0; j < 3u; ++j) hv[j] = hist[3u * cv + j];
        for (uint j = 0; j < 4u; ++j) wv[j] = w[4u * cv + j];
        for (uint t = 0; t < GDN_MAXT; ++t) {
            if (t >= T) break;
            xv[t] = qkv[t * p.C + cv];
        }
        for (uint t = 0; t < GDN_MAXT; ++t) {
            if (t >= T) break;
            const float yv = gdn_conv_silu(hv, wv, xv, t);
            v[t][col] = yv;
            win_h[t * p.C + cv] = cuda_nan(yv);
            win_qkv[t * p.C + cv] = xv[t];                         // a copy: its bits
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // E: the L2 norms of q and k; rank 0 of value head qh (head < h_k) writes the key head's channels, once each
    if (rank == 0u) {
        const bool write_qk = head < p.h_k;
        for (uint t = 0; t < GDN_MAXT; ++t) {
            if (t >= T) break;
            q[t] = ieee_mul(q[t], rsqrt_ieee(ieee_add(gdn_sum4(part[0][t][0], part[0][t][1], part[0][t][2],
                                                               part[0][t][3]), p.eps)));
            k[t] = ieee_mul(k[t], rsqrt_ieee(ieee_add(gdn_sum4(part[1][t][0], part[1][t][1], part[1][t][2],
                                                               part[1][t][3]), p.eps)));
            qn[t][tid] = q[t];
            kn[t][tid] = k[t];
            if (write_qk) {
                win_h[t * p.C + cq] = cuda_nan(q[t]);
                win_h[t * p.C + ck] = cuda_nan(k[t]);
                win_qkv[t * p.C + cq] = xq[t];
                win_qkv[t * p.C + ck] = xk[t];
            }
        }
    }

    // F: the gates, for the commit: thread t (rank 0) and thread 128 + t (rank 1) of token t
    if (tid < T) win_gate[tid * p.h_v + head] = cuda_nan(gm[tid]);
    if (tid >= 128u && tid - 128u < T) win_beta[(tid - 128u) * p.h_v + head] = cuda_nan(bm[tid - 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // G: the recurrence over the tokens t < T on the thread's 32 state rows, dropped at the end (the window never
    // writes the state); kv and o summed over the rank's four row groups, two barriers per token (dec_gdn_rec.cu:245-
    // 275). SIMD-group 4 rank keeps the column's oc and, on lane 0, the rank's sum of squares of the token, which every
    // lane's butterfly holds: basalt's lane-t keep and cluster push (:270-285) as one threadgroup store
    for (uint t = 0; t < T; ++t) {
        const float g = expf_ieee(gm[t]);
        red_kv[rank][rg][lane] = gdn_kv(s, &kn[t][32u * rg]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = gdn_sum4(red_kv[rank][0][lane], red_kv[rank][1][lane], red_kv[rank][2][lane],
                                      red_kv[rank][3][lane]);
        const float delta = gdn_delta(g, kv_col, v[t][col], bm[t]);
        red_o[rank][rg][lane] = gdn_update(s, &kn[t][32u * rg], &qn[t][32u * rg], g, delta);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rg == 0u) {                                            // SIMD-group 4 rank: basalt's warp 0
            const float o = gdn_oc(red_o[rank][0][lane], red_o[rank][1][lane], red_o[rank][2][lane],
                                   red_o[rank][3][lane]);
            const float sq = gdn_sq32(o);
            oc[t][col] = o;
            if (lane == 0u) wsum[t][rank] = sq;
        }
    }

    // H: every rank's sums of squares in wsum; one barrier for the cluster arrive, wait and sync (:177, :280, :286)
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // I: the gated RMS norm and the q8_1 rows (norm_q8, :290-297 and :115-132), SIMD-group 4 rank's 32-value block of
    // each token, block t 4 h_v + 4 head + rank, lane i % 32; the head's sum of squares in rank order. z is read here
    // (basalt's phase C preloads it, :198)
    if (rg == 0u) {
        for (uint t = 0; t < T; ++t) {
            const uint i = t * 128u * p.h_v + head * 128u + col;
            const float yv = gdn_out(oc[t][col], gdn_sum4(wsum[t][0], wsum[t][1], wsum[t][2], wsum[t][3]), p.eps, gam,
                                     z[i]);
#if GDN_Y_BOUND
            y[i] = cuda_nan(yv);
#endif
            q8_1_store_rec(yv, yq, i);
        }
    }
}
