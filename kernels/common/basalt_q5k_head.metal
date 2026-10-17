// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/q5k_head.cu:19-496 and src/kernels/cuda/head_proxy.cuh:17-240
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), whose Q5_K dot repeats native_mmvq.cu's, an
// adaptation of ggml-org/llama.cpp
// ggml/src/ggml-cuda/vecdotq.cuh @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's Q5_K head (n_in a multiple of 256, at most 64 blocks a row): y[j][r] = Q5_K row r times q8_1 column j
// (basalt_q8_1.metal's layout, column j's blocks from block j * n_in / 32), each value bit for bit native_mmvq's
// (basalt_mmvq.metal, MMVQ_TYPE 13): the same q5k_load / q5k_dot (basalt_fp.metal) per (row, block, column), the
// same order. Needs basalt_fp.metal ahead of it. Arithmetic: docs/research/basalt-nvcc/q5k_head.md.
//
// Mapping (q5k_head.cu:108-141, 225-252): threadgroups of 8 SIMD-groups (basalt's WPB warps), one row per SIMD-group,
// rows row0 = 8 tg + sg strided by 8 * threadgroups; lane L stands for native_mmvq's threads L + 32v, v = 0..3, whose
// blocks kbx = (L + 32v) / 16, + 8, ... it takes in ascending order at kqs = 2 (L % 16), each thread's sum from +0;
// then ((p0 + p1) + p2) + p3 (native_mmvq's warp order) and the XOR butterfly. At 10 blocks a row this is
// q5k_rows_10's assignment (thread L: a = L / 16 and a + 8; threads L + 32, L + 64, L + 96: a + 2, a + 4, a + 6).
// q5k_rows_10's next-row register prefetch and the top-1's L2 prefetch schedule loads only and are not ported. The
// columns are read from device memory: basalt stages them in shared memory (61,440 bytes at 8 columns and n_in 2560,
// more than Apple9's 32 KiB), which is not arithmetic.
//
// Kernels:
// - basalt_q5k_head (q5k_head_mmvq; NC columns, 1..8): grid basalt_q5k_head_grid threadgroups of 256 threads.
// - basalt_q5k_head_mask_unmapped (q5k_head_mask_unmapped): y[i] = -inf where map[i] < 0; ceil(n / 256) x 256.
// - basalt_q5k_head_top1 then basalt_q5k_head_top1_merge (q5k_head_top1, one column, n_in 2560): the first writes the
//   logits and each threadgroup's (max, its first row, sum of exp(l - max)) to part[3 tg .. 3 tg + 2]; the second, one
//   threadgroup of 256, merges the partials in basalt's order and writes the token id and its probability. basalt
//   elects the last block with an atomic counter and merges there (q5k_head.cu:303-334); Metal documents no ordering
//   that makes the other threadgroups' partials visible to the elected one, so the merge is a second dispatch, ordered
//   after the first by the dispatch boundary (q5k_head.md section 3). Same arithmetic, same order.
#pragma STDC FP_CONTRACT OFF

#ifndef NC
#define NC 1
#endif

constant constexpr int Q5K_WPB = 8;                 // SIMD-groups (rows in flight) per threadgroup
constant constexpr int Q5K_USHORTS = 88;            // a Q5_K block, as 16-bit words
constant constexpr int TOP_NONE = 0x7fffffff;       // Top::idx before any row

// n_in, n_out, has_map (0: no map; basalt's nullptr) and, for the merge, the first dispatch's threadgroup count
struct Q5kHeadParams { uint n_in; uint n_out; uint has_map; uint n_blocks; };

static inline float neg_inf() { return as_type<float>(0xff800000u); }

// q5k_row_dev (q5k_head.cu:110-141): row `wr` (bpr blocks) times NCOL columns of x, every lane's t[j] the same bits.
template <int NCOL>
static inline void q5k_row(device const ushort* wr, device const uint* x, int x_stride, int bpr, int lane,
                           thread float* t) {
    const int kqs = 2 * (lane % 16);
    float p[4][NCOL];
    for (int v = 0; v < 4; ++v)
        for (int j = 0; j < NCOL; ++j) p[v][j] = 0.0f;
    for (int v = 0; v < 4; ++v) {
        for (int kbx = (lane + 32 * v) / 16; kbx < bpr; kbx += 8) {
            const Q5KW wv = q5k_load(wr + kbx * Q5K_USHORTS, kqs);
            for (int j = 0; j < NCOL; ++j) p[v][j] = add_ftz(p[v][j], q5k_dot(wv, x + 9 * (j * x_stride + 8 * kbx), kqs));
        }
    }
    for (int j = 0; j < NCOL; ++j) {
        float s = p[0][j];
        for (int v = 1; v < 4; ++v) s = add_ftz(s, p[v][j]);
        t[j] = warp_sum_ftz(s);
    }
}

kernel void basalt_q5k_head(device const ushort* w [[buffer(0)]], device const uint* x [[buffer(1)]],
                            device float* y [[buffer(2)]], constant Q5kHeadParams& p [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                            ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int n_in = int(p.n_in), n_out = int(p.n_out), bpr = n_in / 256;
    for (int row = Q5K_WPB * int(tg) + int(sg); row < n_out; row += Q5K_WPB * int(ntg)) {
        float t[NC];
        q5k_row<NC>(w + uint(row) * uint(bpr * Q5K_USHORTS), x, n_in / 32, bpr, int(lane), t);
        if (lane == 0)
            for (int j = 0; j < NC; ++j) y[j * n_out + row] = t[j];
    }
}

kernel void basalt_q5k_head_mask_unmapped(device float* y [[buffer(0)]], device const int* map [[buffer(1)]],
                                          constant Q5kHeadParams& p [[buffer(2)]],
                                          uint i [[thread_position_in_grid]]) {
    if (i < p.n_out && map[i] < 0) y[i] = neg_inf();
}

// ---- the top-1 (q5k_head.cu:146-176, 256-335) ----------------------------------------------------------------------

// the running (max, its first row, sum of exp(l - max)) of the logits taken so far
struct Top { float m; int idx; float s; };
static inline Top top_init() { return Top{neg_inf(), TOP_NONE, 0.0f}; }
static inline void top_take(thread Top& t, float l, int row) {
    if (l > t.m) {
        t.s = fma_ftz(t.s, expf_ftz(add_ftz(t.m, -l)), 1.0f);
        t.m = l;
        t.idx = row;
    } else {
        t.s = add_ftz(t.s, expf_ftz(add_ftz(l, -t.m)));
        if (l == t.m && row < t.idx) t.idx = row;
    }
}
static inline void top_merge(thread Top& t, Top o) {
    if (o.idx == TOP_NONE) return;
    if (t.idx == TOP_NONE || o.m > t.m) {
        t.s = fma_ftz(t.s, expf_ftz(add_ftz(t.m, -o.m)), o.s);
        t.m = o.m;
        t.idx = o.idx;
    } else {
        t.s = fma_ftz(o.s, expf_ftz(add_ftz(o.m, -t.m)), t.s);
        if (o.m == t.m && o.idx < t.idx) t.idx = o.idx;
    }
}

kernel void basalt_q5k_head_top1(device const ushort* w [[buffer(0)]], device const uint* x [[buffer(1)]],
                                 device float* y [[buffer(2)]], device const int* map [[buffer(3)]],
                                 device float* part [[buffer(4)]], constant Q5kHeadParams& p [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                                 ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Top wt[Q5K_WPB];
    const int n_out = int(p.n_out);
    Top top = top_init();
    for (int row = Q5K_WPB * int(tg) + int(sg); row < n_out; row += Q5K_WPB * int(ntg)) {
        float t[1];
        q5k_row<1>(w + uint(row) * uint(10 * Q5K_USHORTS), x, 80, 10, int(lane), t);
        // a row whose id is negative holds no token: it never wins and adds nothing to the sum
        if (p.has_map != 0u && map[row] < 0) t[0] = neg_inf();
        else top_take(top, t[0], row);
        if (lane == 0) y[row] = t[0];
    }
    if (lane == 0) wt[sg] = top;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        Top b = wt[0];
        for (int i = 1; i < Q5K_WPB; ++i) top_merge(b, wt[i]);
        part[3 * tg] = b.m;
        part[3 * tg + 1] = as_type<float>(b.idx);
        part[3 * tg + 2] = b.s;
    }
}

kernel void basalt_q5k_head_top1_merge(device const float* part [[buffer(0)]], device const int* map [[buffer(1)]],
                                       device int* out_id [[buffer(2)]], device float* out_prob [[buffer(3)]],
                                       constant Q5kHeadParams& p [[buffer(4)]],
                                       ushort tid [[thread_index_in_threadgroup]],
                                       ushort lane [[thread_index_in_simdgroup]],
                                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup Top wt[Q5K_WPB];
    Top g = top_init();   // each thread a strided share of the block partials, then the block
    for (int i = int(tid); i < int(p.n_blocks); i += 32 * Q5K_WPB)
        top_merge(g, Top{part[3 * i], as_type<int>(part[3 * i + 1]), part[3 * i + 2]});
    for (ushort off = 16; off > 0; off >>= 1)
        top_merge(g, Top{shfl_xor(g.m, off), simd_shuffle_xor(g.idx, off), shfl_xor(g.s, off)});
    if (lane == 0) wt[sg] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        Top b = wt[0];
        for (int i = 1; i < Q5K_WPB; ++i) top_merge(b, wt[i]);
        const int id = b.idx < int(p.n_out) ? b.idx : 0;
        *out_id = p.has_map != 0u ? map[id] : id;
        *out_prob = rcp_ftz(b.s);
    }
}

// ---- the 2-bit proxy (head_proxy.cuh; q5k_head.cu:434-535) --------------------------------------------------------
// A copy of the head's leading rows (n_in 2560) at four levels and an fp16 scale per 32 weights; the proxy buffer is
// basalt's px::bytes(n_cap): per tile of 32 rows and per 64 weights (w4 = 0..39) a 16-byte word per row (lane = row),
// then from px::scales(proxy, n_cap) the fp16 scales, two per (tile, w4, lane) (head_proxy.cuh:17-28, 72-87).

constant constexpr uint PX_NW4 = 40;        // kNw4: 16-byte weight words per row
constant constexpr int PX_PER = 4;          // kPer: candidates kept per tile
constant constexpr int PX_MAX_CAND = 64;    // kMaxCand: rows rescored exactly

// pack: the rows row0 .. row0 + n - 1; score and final: the first n_rows rows; n_cap: the copy's row capacity
struct PxParams { uint row0; uint n; uint n_cap; uint n_rows; };

static inline uint px_tiles(uint n) { return (n + 31u) / 32u; }

// level(w, sh) (head_proxy.cuh:36) as ptxas compiles it: F2I.FTZ.FLOOR.NTZ(FFMA.FTZ(w, MUFU.RCP(sh), 2)), then the
// clamp to 0..3; rs = rcp_ftz(sh). F2I takes NaN to 0 and saturates the infinities, which the clamp makes 0 and 3: the
// NaN case is real (a group of zeros has sh = 0, rs = +inf, 0 * inf + 2 = NaN). MSL takes NaN to 0 (section 8.6) but
// converts without saturation (section 2.23), so the clamp is done in float.
static inline int px_level(float w, float rs) {
    const float f = floor(fma_ftz(w, rs, 2.0f));
    return isnan(f) ? 0 : int(clamp(f, 0.0f, 3.0f));
}

// fit_group<false> (head_proxy.cuh:42-70; q5k_head.md section 4): the fp16 scale's bits, sh_out its value.
// amax / 1.5f is FMUL.FTZ by fl(1 / 1.5) (ptxas's constant division), 1.0f - 0.1f * t is FFMA.FTZ(t, -0.1f, 1),
// each squared error FFMA.FTZ(r, r, e) from 0 in weight order, r = FFMA.FTZ(level - 1.5, sh, -w).
static inline ushort px_fit(thread const float* w, thread float& sh_out) {
    float amax = as_type<float>(0x3089705Fu);                           // 1e-9f
    for (int l = 0; l < 32; ++l) amax = max_ftz(amax, abs_ftz(w[l]));
    const float a15 = mul_ftz(amax, as_type<float>(0x3F2AAAABu));
    float best_s = a15, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = mul_ftz(a15, fma_ftz(float(t), -0.1f, 1.0f));
        const float sh = float(half(s));                                  // __half2float(__float2half(s)): RNE
        const float rs = rcp_ftz(sh);
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = fma_ftz(add_ftz(float(px_level(w[l], rs)), -1.5f), sh, -w[l]);
            e = fma_ftz(r, r, e);
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    const half hs = half(best_s);
    sh_out = float(hs);
    return as_type<ushort>(hs);
}

// q5k_proxy_pack_kernel (q5k_head.cu:438-462) with pack_group<false> (head_proxy.cuh:72-87): one thread per 32-weight
// group g of row row0 + i / 80, w = FFMA.FTZ(FMUL.FTZ(d, sc), q, -FMUL.FTZ(dmin, m)). ceil(n * 80 / 128) x 128.
kernel void basalt_q5k_proxy_pack(device const uchar* q5 [[buffer(0)]], device uint* proxy [[buffer(1)]],
                                  constant PxParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n * 80u) return;
    const uint row = p.row0 + i / 80u;
    const int g = int(i % 80u), j = g % 8;
    device const uchar* b = q5 + row * 1760u + uint(g / 8) * 176u;     // Q5KBlock: dm, scales[12], qh[32], qs[128]
    device const uchar* scl = b + 4;
    int sci, mi;
    if (j < 4) { sci = scl[j] & 63; mi = scl[j + 4] & 63; }
    else {
        sci = (scl[j + 4] & 0xF) | ((scl[j - 4] >> 6) << 4);
        mi = (scl[j + 4] >> 4) | ((scl[j] >> 6) << 4);
    }
    const float d = mul_ftz(h2f(ushort(b[0] | (b[1] << 8))), float(sci));
    const float mn = mul_ftz(h2f(ushort(b[2] | (b[3] << 8))), float(mi));
    float w[32];
    for (int l = 0; l < 32; ++l) {
        const int qs = b[48 + 32 * (j / 2) + l];
        const int q = ((j & 1) ? (qs >> 4) : (qs & 15)) | (((b[16 + l] >> j) & 1) << 4);
        w[l] = fma_ftz(d, float(q), -mn);
    }
    float sh;
    const ushort hs = px_fit(w, sh);
    const float rs = rcp_ftz(sh);
    uint words[2] = {0u, 0u};
    for (int c = 0; c < 2; ++c)
        for (int ii = 0; ii < 4; ++ii)
            for (int bb = 0; bb < 4; ++bb) words[c] |= uint(px_level(w[c * 16 + 4 * ii + bb], rs)) << (8 * bb + 2 * ii);
    const uint slot = ((row >> 5) * PX_NW4 + uint(g / 2)) * 32u + (row & 31u), h = uint(g & 1);
    proxy[slot * 4u + 2u * h] = words[0];
    proxy[slot * 4u + 2u * h + 1u] = words[1];
    device ushort* sc = (device ushort*)(proxy + px_tiles(p.n_cap) * PX_NW4 * 32u * 4u);   // px::scales(proxy, n_cap)
    sc[slot * 2u + h] = hs;
}

// px::score_kernel (head_proxy.cuh:89-167): a threadgroup of 4 SIMD-groups per tile of 32 rows (lane = row), SIMD-group
// k scoring the weights 640 k .. 640 k + 639 of its rows; the tile's 4 best approximate scores with their rows go to
// cand[4 tile + k] and its softmax partial (max, sum of exp(score - max)) to part[tile], cand and part in one scratch
// (basalt's layout, q5k_head.cu:523-524). Per group g: P the exact integer dot of the levels and the q8_1 codes, then
// acc = FFMA.FTZ(scale, FFMA.FTZ(d8, P, -FMUL.FTZ(d8 * sum, 1.5)), acc) (q5k_head.md section 4). Grid: tiles(n_rows).
kernel void basalt_q5k_proxy_score(device const uint4* proxy [[buffer(0)]], device const uint* x [[buffer(1)]],
                                   device const int* map [[buffer(2)]], device float2* scratch [[buffer(3)]],
                                   constant PxParams& p [[buffer(4)]], uint tile [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup int xi[640];
    threadgroup float xs[80], xdd[80];
    threadgroup float red[4][32];
    const uint ntc = px_tiles(p.n_cap);
    device const uint* sc = (device const uint*)(proxy + ntc * PX_NW4 * 32u);    // px::scales, two per word
    const uint base = (tile * PX_NW4 + uint(sg) * 10u) * 32u + lane;
    const int row = int(tile) * 32 + int(lane), n_rows = int(p.n_rows);
    const int id = (sg == 0 && row < n_rows) ? map[row] : 0;
    for (int i = tid; i < 640; i += 128) xi[i] = q8_w(x, i / 8, i % 8);
    if (tid < 80) {
        int sum = 0;
        for (int i = 0; i < 8; ++i) sum = dp4a(0x01010101, q8_w(x, tid, i), sum);
        const float d8 = q8_d(x, tid);
        xdd[tid] = d8;
        xs[tid] = mul_ftz(d8, float(sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (int j = 0; j < 10; ++j) {
        const uint4 wv = proxy[base + uint(j) * 32u];
        const uint s = sc[base + uint(j) * 32u];
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * (int(sg) * 10 + j) + h;
            int P = 0;
            for (int c = 0; c < 2; ++c) {
                const uint v = wv[2 * h + c];
                const int o = g * 8 + c * 4;
                P = dp4a(int(v & 0x03030303u), xi[o], P);
                P = dp4a(int((v >> 2) & 0x03030303u), xi[o + 1], P);
                P = dp4a(int((v >> 4) & 0x03030303u), xi[o + 2], P);
                P = dp4a(int((v >> 6) & 0x03030303u), xi[o + 3], P);
            }
            const float scale = h2f(ushort(h ? (s >> 16) : (s & 0xffffu)));
            acc = fma_ftz(scale, fma_ftz(xdd[g], float(P), -mul_ftz(xs[g], 1.5f)), acc);
        }
    }
    red[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float v = add_ftz(add_ftz(add_ftz(red[0][lane], red[1][lane]), red[2][lane]), red[3][lane]);
    if (row >= n_rows || id < 0) v = neg_inf();
    float m = v;
    for (ushort o = 16; o > 0; o >>= 1) m = max_ftz(m, shfl_xor(m, o));
    const float e = warp_sum_ftz(v > neg_inf() ? expf_ftz(add_ftz(v, -m)) : 0.0f);
    device float2* cand = scratch;
    device float2* part = scratch + ntc * uint(PX_PER);
    if (lane == 0) part[tile] = float2(m, e);
    for (int k = 0; k < PX_PER; ++k) {          // the tile's best, then the next: a butterfly argmax, ties to the lower row
        float bv = v;
        int bi = lane;
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = shfl_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * uint(PX_PER) + uint(k)] = float2(bv, as_type<float>(int(tile) * 32 + bi));
        if (int(lane) == bi) v = neg_inf();
    }
}
