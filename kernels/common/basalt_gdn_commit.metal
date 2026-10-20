// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/verify_kernels.cu:20-22, :88-101 and :265-361 and
// include/basalt/kernels/verify_kernels.hpp:21-38 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// written in the rounding sequence nvcc compiles for them (docs/research/basalt-nvcc/verify_gdn.md sections 2, 4 and 7).
//
// basalt's GDN commit on Metal (gdn_commit_layers_pipe_kernel; contract invariant 3): once acceptance is known, the
// first n rows of a window are replayed from the state the window left untouched and the h, gate, beta and qkv rows it
// stored, and each layer's recurrent state and conv history are written. Needs basalt_fp.metal, basalt_ieee.metal,
// basalt_f64.metal and basalt_ieee_f64.metal ahead of it (monolith/kernels.py's basalt_gdn_commit_source()).
#pragma STDC FP_CONTRACT OFF

// basalt's S, RG and RPG (verify_kernels.cu:20-22) and kVerifyMaxT (verify_kernels.hpp:21)
constant constexpr uint GDC_S = 128u;      // a head's k rows and v columns: the state's side
constant constexpr uint GDC_RG = 4u;       // row groups: a thread holds GDC_RPG rows of one column
constant constexpr uint GDC_RPG = 32u;     // GDC_S / GDC_RG
constant constexpr uint GDC_MAXT = 8u;     // the widest window: the token arrays' size

// The record (monolith/kernels.py's basalt_gdn_commit_params): basalt's GdnCommitLayers (verify_kernels.hpp:32-38)
// without its pointers and n_keep, the strides in floats: layer l's state at l * state_stride, its C x 3 conv history
// right after the S x h_v x S state, its h and qkv rows at l * h_stride and its gate and beta rows at l * gb_stride;
// h_k = (C - 128 h_v) / 256 (q, k, then v). Then the launcher's conv_blocks = ceil(C / 128) (verify_kernels.cu:861) and
// mt, the window buffers' rows, to which n_keep is clamped where basalt clamps it to kVerifyMaxT (:358).
struct GdnCommitParams {
    uint state_stride; uint h_stride; uint gb_stride; uint c; uint h_k; uint h_v; uint n_layers; uint conv_blocks;
    uint mt; uint pad0; uint pad1; uint pad2;
};

// A conv item's channel c (gdn_conv_commit_n, verify_kernels.cu:88-101): its history becomes the last three of
// [hist(3) | qkv_0 .. qkv_{n-1}], word src = n + j for j < 3, all three loaded before any store. 32-bit word moves: no
// float operation touches them, so a subnormal or a NaN keeps its bits (verify_gdn.md section 4). A channel past C
// returns from here, never from the kernel: the threadgroup may reach a state item's barriers on a later item.
static inline void gdc_conv(device uint* hist, device const uint* qkv, uint C, uint n, uint c) {
    if (c >= C) return;
    uint seq[3];
#pragma unroll
    for (uint j = 0; j < 3u; ++j) {
        const uint src = n + j;
        seq[j] = src < 3u ? hist[c * 3u + src] : qkv[(src - 3u) * C + c];
    }
    hist[c * 3u] = seq[0];
    hist[c * 3u + 1u] = seq[1];
    hist[c * 3u + 2u] = seq[2];
}

// A state item (commit_pre_load and commit_pre_run, verify_kernels.cu:271-331): value head `head`, the 32 columns from
// 32 cblk, in layer l. Thread tid = 32 rg + lane holds rows 32 rg .. 32 rg + 31 of column col = 32 cblk + lane in s[r]
// and stages row tid of each kept token's key in sk[t][tid]. Per token, the sequence nvcc compiles (verify_gdn.md
// section 7): the decay gt = __expf(gate); kv, an FFMA chain from +0 over the thread's rows, r ascending; kv_col, the
// four row groups' kv added in order; delta = (v - gt kv_col) b with the product contracted into the difference (FFMA
// d, -gt, kv_col, v, then FMUL by b); then per row kd = k_r delta (FMUL) and s = gt s + kd (FFMA). A subnormal can
// occur at each (a strongly forgetting head, a decayed state), so every FFMA is fma_ieee, every FMUL and FADD ieee_mul
// and ieee_add, the decay expf_ieee; the stores go through cuda_nan (sm_120's FFMA writes a NaN as 0x7fffffff).
// red[t & 1] is double-buffered as basalt's: a thread writes it again only at token t + 2, after token t + 1's barrier,
// which every thread reaches after its reads of token t, so one barrier per token suffices.
static inline void gdc_state(device float* state, device const float* h, device const float* gate,
                             device const float* beta, constant GdnCommitParams& p, uint l, uint j, uint n, uint tid,
                             threadgroup float* sk, threadgroup float* red) {
    const uint lane = tid & 31u, rg = tid >> 5;
    const uint head = j >> 2, col = (j & 3u) * 32u + lane;
    const uint qk = GDC_S * p.h_k, qh = head % p.h_k;
    device const float* hl = h + l * p.h_stride;
    device const float* gl = gate + l * p.gb_stride;
    device const float* bl = beta + l * p.gb_stride;
    float k[GDC_MAXT], v[GDC_MAXT], g[GDC_MAXT], b[GDC_MAXT], s[GDC_RPG];
    for (uint t = 0; t < n; ++t) {
        device const float* ht = hl + t * p.c;
        k[t] = ht[qk + qh * GDC_S + tid];
        v[t] = ht[2u * qk + head * GDC_S + col];
        g[t] = gl[t * p.h_v + head];
        b[t] = bl[t * p.h_v + head];
    }
    device float* base = state + l * p.state_stride + (rg * GDC_RPG * p.h_v + head) * GDC_S + col;
    const uint row = p.h_v * GDC_S;
#pragma unroll
    for (uint r = 0; r < GDC_RPG; ++r) s[r] = base[r * row];
    threadgroup_barrier(mem_flags::mem_threadgroup);     // the previous item is done with sk and red
    for (uint t = 0; t < n; ++t) sk[t * GDC_S + tid] = k[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < n; ++t) {
        const float gt = expf_ieee(g[t]);
        threadgroup const float* kt = sk + t * GDC_S + rg * GDC_RPG;     // sk[t][32 rg + r]
        threadgroup float* rt = red + (t & 1u) * (GDC_RG * 32u);         // red[t & 1][rg][lane]
        float kv = 0.0f;
#pragma unroll
        for (uint r = 0; r < GDC_RPG; ++r) kv = fma_ieee(s[r], kt[r], kv);
        rt[rg * 32u + lane] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = ieee_add(ieee_add(ieee_add(rt[lane], rt[32u + lane]), rt[64u + lane]), rt[96u + lane]);
        const float delta = ieee_mul(fma_ieee(sign_flip(gt), kv_col, v[t]), b[t]);
#pragma unroll
        for (uint r = 0; r < GDC_RPG; ++r) {
            const float kd = ieee_mul(kt[r], delta);
            s[r] = fma_ieee(gt, s[r], kd);
        }
    }
#pragma unroll
    for (uint r = 0; r < GDC_RPG; ++r) base[r * row] = cuda_nan(s[r]);
}

// The commit (gdn_commit_layers_pipe_kernel and commit_layers_body, verify_kernels.cu:333-361), 128 threads per
// threadgroup: basalt's (32, RG) block, lane = threadIdx.x and rg = threadIdx.y. n = min(commit[0], mt) kept rows;
// threadgroup 0's thread 0 stores the n it used, or 0, in used[0], and for n <= 0 nothing else is stored. Otherwise each
// layer's items, 4 h_v state items (a head's 32-column block each) then conv_blocks conv items (128 channels each), are
// visited by every threadgroup from its own index, striding by the threadgroup count. The items write disjoint words,
// so the count is a schedule, never a value; basalt's prefetch of the next item's registers changes no bit either and
// is not done here. The item's kind is uniform per threadgroup, so all 128 threads reach every barrier. Bounded: at
// most ceil(total / threadgroups) items per threadgroup, n <= mt <= 8 tokens, 32 rows per thread. Loads stay inside
// their bindings because the record's strides are at least the dense layout's (basalt_gdn_commit_params) and every row
// read is t < n <= mt.
[[max_total_threads_per_threadgroup(128)]]
kernel void gdn_commit_layers(device float* state [[buffer(0)]], device const float* h [[buffer(1)]],
                              device const float* qkv [[buffer(2)]], device const float* gate [[buffer(3)]],
                              device const float* beta [[buffer(4)]], device const int* commit [[buffer(5)]],
                              device uint* used [[buffer(6)]], constant GdnCommitParams& p [[buffer(7)]],
                              uint tgid [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sk[GDC_MAXT * GDC_S];          // sk[8][128]: t < n <= 8, tid < 128
    threadgroup float red[2u * GDC_RG * 32u];         // red[2][4][32]: t & 1, rg < 4, lane < 32
    const int n = min(commit[0], int(p.mt));
    if (tgid == 0u && tid == 0u) used[0] = uint(max(n, 0));
    if (n <= 0) return;                               // uniform: every thread read the same word
    const uint per_state = p.h_v * 4u, per_layer = per_state + p.conv_blocks, total = per_layer * p.n_layers;
    for (uint i = tgid; i < total; i += ntg) {
        const uint l = i / per_layer, j = i - l * per_layer;
        if (j < per_state) {
            gdc_state(state, h, gate, beta, p, l, j, uint(n), tid, sk, red);
        } else {
            gdc_conv(reinterpret_cast<device uint*>(state + l * p.state_stride + GDC_S * p.h_v * GDC_S),
                     reinterpret_cast<device const uint*>(qkv + l * p.h_stride), p.c, uint(n),
                     (j - per_state) * GDC_S + tid);
        }
    }
}
