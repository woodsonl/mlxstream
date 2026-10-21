// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/prefill/mmq/{mmq.cuh:390-864, quantize.cu:10-84, mmq_load_tiles.cuh:11-124,
// mmq_vec_dot.cuh:14-92, mmq_config.cuh:15-42}, src/prefill/moe_mmq.cu:85-93, :168-199, src/prefill/prefill.cpp:1840-2008,
// :2241-2264 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE); the mmq/ files are ggml-org/llama.cpp
// ggml/src/ggml-cuda/{mmq.cuh, quantize.cu, mmq-load-tiles.cuh, mmq-vec-dot.cuh, mmq-config-ampere.cuh}
// @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's prompt-path MMQ: the routed experts of one MoE half on the Q2_0 pack's native blobs (moe_mmq.cu's
// Context::run and llama.cpp's mul_mat_q<Q2_0> with its stream-k fixup, launched per group of 16 experts by
// prefill.cpp:2241-2264), and its Q8_0 instance for draft_kv's dense projections (prefill.cpp:971-992). basalt builds
// moe_mmq.cu and mmq/ with -use_fast_math (CMakeLists.txt:721-728), so every fp32 step here is one of basalt_fp.metal's
// flushing helpers, which this file needs ahead of it (add_ftz, mul_ftz, fma_ftz, max_ftz, abs_ftz; rcp_ftz where SASS
// has MUFU.RCP and expf_ftz for __expf, contract invariant 5), never an IEEE one, and no float is compared or ordered.
// The int8 tensor-core MMA is an exact int32 sum of dp4a per 32-value sub-block; the fp32 steps after it follow nvcc's
// order per output element (docs/research/basalt-nvcc/mmq.md). That order includes how stream-k cuts each output's K
// range into pieces, a function of the launch's tile count and of the GPU's multiprocessor count (mmq.cuh:830-833):
// the port evaluates it at basalt's reference GPU's count, MMQ_SM_COUNT (170), never at this GPU's cores.
//
// The launch helpers (spec-kernels-mmq.md 9.3). Nothing per launch is stored (invariant 4): a threadgroup derives its
// launch from the plan's bounds.
// - mmq_launch(bounds, n, g, nty): group g of a plan of n launch slots, MMQ_GROUP = 16 experts a launch: its slots
//   ngx, its largest row count maxr, the tile width J (mul_mat_q_switch_J over MMQ_J, ncols_opt = maxr), the column
//   tiles ntx, the tiles ntiles of nty 128-row weight tiles each and the stream-k grid.
// - mmq_piece_mask(t, ntiles, bpne, bpi, grid): bit q is set where a piece of tile t's K range starts at k iteration q
//   (bpi weight blocks of bpne a row), the closed form of mul_mat_q's walk over its CTAs (mmq.cuh:487-629) and its
//   fixup (:632-767). The tile's outputs are its pieces' chains, each from +0, folded in the fixup's order.
// - mmq_code(v): the quantizer's roundf stored to a char as sm_120 converts it (spec-kernels-mmq.md 3.2).
// - mmq_quantize_block(xi, blk, iqs): quantize_mmq_q8_1<D4>'s body (quantize.cu:39-81), which kernels 1 and 3 share.
// - mmq_dA(tab, order, j, jl, i, s): the down product's weight scale past row i, where basalt's tile load reads its
//   group buffer (spec-kernels-mmq.md 9.6), read in place.
//
// The kernels (spec-kernels-mmq.md 9.1). Each returns at StepState.done before its first load, so once the plan
// refuses a routed id nothing more of the step is stored.
// - basalt_mmq_plan_count, basalt_mmq_plan_scan, basalt_mmq_plan_fill: basalt's host walk over the chunk's routed ids
//   (prefill.cpp:1840-1932, :1993-1996) as integer passes: per-range counts, one threadgroup's scans in expert-id
//   order, a stable fill; the plan, slot and src every later kernel reads.
// - basalt_mmq_quantize (kernel 1): the layer's activations to Xq through the plan's src (quantize.cu:10-84, launched
//   :94-96 from prefill.cpp:1990).
// - basalt_mmq (kernel 2, compiled with the gate/up product's macros): every group's mul_mat_q<Q2_0> on Xq and the
//   experts' gate and up rows with its stream-k fixup folded per output element (mmq.cuh:390-864; prefill.cpp:2243-2256
//   launches it per group of 16 experts), one dispatch for all of them, into GU.
// - basalt_mmq_swiglu_quantize (kernel 3): GU's split halves to Hq, mmq::swiglu (moe_mmq.cu:85-93) fused with the
//   quantizer on H (prefill.cpp:2257-2258's group calls), compiled with the down product's macros (K = n_ff).
// - basalt_mmq (kernel 4, compiled with the down product's macros, MMQ_DOWN 1): every group's down mul_mat_q<Q2_0> on
//   Hq and the experts' down rows (prefill.cpp:2259-2264) into D, the padded last k iteration through mmq_dA.
//
// Macros (monolith/kernels.py basalt_mmq_macros): MMQ_TYPE (42 Q2_0, 8 Q8_0), K, NTY, LD, BPNE, BPI, NITER, MMQ_DOWN,
// MMQ_DENSE, ROWS_PER_T, MMQ_SM_COUNT, MAT_OFF, MMQ_ERR, N_EXPERT, MMQ_RANGE; and PROMPT_ROWS' T_SRC, ROWS_MAX and
// T_STATIC_ROWS.
#pragma STDC FP_CONTRACT OFF

// quantize.cu:53-56 (q.x = roundf(xi.x*d_inv)): FADD.FTZ.RZ of copysign(0.5, v), F2I.TRUNC.NTZ (s32, saturating, NaN
// to 0), then the low byte. round() equals roundf but for a zero's sign, which the conversion erases (pinned by
// test_round_matches_cuda_roundf_but_for_the_zero_sign); v = +-inf (d_inv = +inf) saturates as F2I does, where MSL
// leaves int(+-inf) undefined.
static inline char mmq_code(float v) {
    if (isnan(v)) return 0;
    const float r = round(v);
    if (r >= 2147483648.0f) return (char) -1;   // INT_MAX: low byte 0xFF
    if (r <= -2147483648.0f) return 0;          // INT_MIN: low byte 0x00
    return (char) (int) r;
}

// mmq_config.cuh:15-25 (Q2_0) and :32-42 (Q8_0): the tile widths J with fallback false (w_rows % 128 == 0)
constant int MMQ_J[11] = {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128};

struct MmqLaunch { int ngx, maxr, J, ntx, ntiles, grid; };

// group g of a product with nty = rows / 128 (prefill.cpp:2243-2263; switch_J mmq.cuh:866-946; launch :786-864)
static inline MmqLaunch mmq_launch(device const int* bounds, int n, int g, int nty) {
    MmqLaunch L;
    L.ngx = min(16, n - 16 * g);                                   // MMQ_GROUP = 16 (prefill.cpp:607)
    L.maxr = 0;
    for (int z = 0; z < L.ngx; ++z) L.maxr = max(L.maxr, bounds[16 * g + z + 1] - bounds[16 * g + z]);
    const int best = (L.maxr + 127) / 128;                         // the fewest column tiles any J gives
    L.J = 128;
    for (int c = 0; c < 11; ++c)                                   // the first J reaching it (ncols_opt = maxr)
        if ((L.maxr + MMQ_J[c] - 1) / MMQ_J[c] == best) { L.J = MMQ_J[c]; break; }
    L.ntx = (L.maxr + L.J - 1) / L.J;
    L.ntiles = L.ntx * nty * L.ngx;
    const int nw = (L.ntiles + MMQ_SM_COUNT - 1) / MMQ_SM_COUNT;
    const int eff = 100 * L.ntiles / (MMQ_SM_COUNT * nw);           // tiles_efficiency_percent (mmq.cuh:832)
    L.grid = eff >= 90 ? L.ntiles : MMQ_SM_COUNT;                  // block_nums_stream_k (mmq.cuh:833)
    return L;
}

// mmq.cuh:487-491 with the fixup's walk :654-698: the piece starts of tile t in iterations q in [1, niter)
// (spec-kernels-mmq.md 7.5): k is the first CTA whose unrounded start floor(k W / grid) reaches x = t*bpne + bpi*q; a
// piece starts at q iff that start lies inside iteration q
static inline uint mmq_piece_mask(int t, int ntiles, int bpne, int bpi, int grid) {
    const ulong W = ulong(ntiles) * ulong(bpne), G = ulong(grid), t0 = ulong(t) * ulong(bpne);
    uint mask = 0u;
    for (int q = 1; q * bpi < bpne; ++q) {                         // at most 9 tests
        const ulong x = t0 + ulong(bpi * q);
        const ulong k = (x * G + W - 1) / W;
        if (k * W / G < t0 + ulong(min(bpi * q + bpi, bpne))) mask |= 1u << q;
    }
    return mask;
}

// The plan's int32 words (monolith/kernels.py MMQ_PLAN_WORDS): n, rows_local, order[512], then bounds[513] from word
// MMQ_BOUNDS, then wpre[513] from word MMQ_WPRE.
constant int MMQ_BOUNDS = 514;
constant int MMQ_WPRE = 1027;
constant uint MMQ_EXPERTS = 512;   // the plan's expert slots; the model's N_EXPERT <= 512
constant uint MMQ_TOPK = 10;       // routed experts per token: ids rows of 10 pairs (t, k), pair i = 10 t + k

// The plan kernels (spec-kernels-mmq.md 9.4): basalt's host walk over one chunk's routed ids, integer only, so every
// output equals the sequential walk's whatever the ranges, the threadgroups or the order in which atomic adds land.
// ids [B][MMQ_TOPK] (rows t >= R never read); hist, the mmq_hist scratch [ceil(B / MMQ_RANGE)][MMQ_EXPERTS + 1],
// shared by every layer and chunk and never cleared: only the NR = ceil(R / MMQ_RANGE) ranges of this chunk's R rows
// are touched, range r holding tokens MMQ_RANGE r .. + MMQ_RANGE - 1. No peer block: the GPU's experts in id order,
// then cpu_mask's (basalt's on_cpu) as the last block.

// Count (prefill.cpp:1840-1845): range r's pairs per expert id, an id outside [0, N_EXPERT) at column MMQ_EXPERTS,
// basalt's ``if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }`` (:1843),
// which basalt_mmq_plan_scan raises. Integer adds, so their order changes no count. Grid ceil(B / MMQ_RANGE) x
// MMQ_THREADS; ranges at or past R return.
kernel void basalt_mmq_plan_count(device const int* ids [[buffer(0)]], device int* hist [[buffer(1)]],
                                  device const StepState* st [[buffer(15)]],
                                  uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                  uint nt [[threads_per_threadgroup]]) {
    if (st->done) return;                                                    // uniform: before any load
    const uint R = prompt_rows(st);
    if (r * MMQ_RANGE >= R) return;                                          // uniform per threadgroup
    threadgroup atomic_int h[MMQ_EXPERTS + 1];
    for (uint e = tid; e <= MMQ_EXPERTS; e += nt) atomic_store_explicit(&h[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const int* rid = ids + ulong(r) * (MMQ_RANGE * MMQ_TOPK);
    const uint np = min(R - r * MMQ_RANGE, uint(MMQ_RANGE)) * MMQ_TOPK;     // the range's pairs with t < R
    for (uint i = tid; i < np; i += nt) {                                    // :1841-1844
        const int e = rid[i];
        atomic_fetch_add_explicit(&h[e < 0 || e >= N_EXPERT ? MMQ_EXPERTS : uint(e)], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e <= MMQ_EXPERTS; e += nt)
        hist[ulong(r) * (MMQ_EXPERTS + 1) + e] = atomic_load_explicit(&h[e], memory_order_relaxed);
}

// Scan (prefill.cpp:1843, :1896-1909, :1926-1932, :1995-1996), one threadgroup of MMQ_EXPERTS threads, thread e
// expert id e. First the refusal: if any of the NR ranges counted an absent pair, StepState takes error MMQ_ERR and
// done and nothing else is stored, as basalt returns inside its counting loop before its first off, slot or src write
// (:1897-1916). Else cnt[e] = column e over the NR ranges; off[e] the exclusive scan in id order of the GPU's experts'
// counts from 0 (:1897-1902), then of cpu_mask's experts' from rows_local, the first scan's total (:1907-1908, :1893);
// n and order the GPU's experts with rows in id order (:1927-1930); bounds[j] = off[order[j]], bounds[n] = rows_local
// (:1995-1996); wpre the exclusive prefix of ceil(cnt / 8) in that order (kernels 2 and 4's work items of 8 rows).
// Column e of each range becomes fill[e] at the range's first pair (:1911): off[e] plus the earlier ranges' counts.
kernel void basalt_mmq_plan_scan(device int* hist [[buffer(1)]], device int* plan [[buffer(2)]],
                                 device const uint* cpu_mask [[buffer(5)]], device StepState* st [[buffer(15)]],
                                 uint e [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    if (st->done) return;                                                    // uniform: before any load
    const uint nr = (prompt_rows(st) + MMQ_RANGE - 1) / MMQ_RANGE;
    threadgroup bool absent[MMQ_EXPERTS / 32];
    threadgroup int4 part[MMQ_EXPERTS / 32];
    bool bad = false;
    for (uint r = e; r < nr; r += MMQ_EXPERTS) bad = bad || hist[ulong(r) * (MMQ_EXPERTS + 1) + MMQ_EXPERTS] != 0;
    bad = simd_any(bad);
    if (lane == 0) absent[sg] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < MMQ_EXPERTS / 32; ++s) bad = bad || absent[s];
    if (bad) {                                                               // uniform: the threadgroup's OR
        if (e == 0) { st->error = uint(MMQ_ERR); st->done = 1u; }            // advance.metal:22's form
        return;
    }
    device int* col = hist + e;
    int cnt = 0;
    for (uint r = 0; r < nr; ++r) cnt += col[ulong(r) * (MMQ_EXPERTS + 1)];
    const bool cpu = int(e) < N_EXPERT && cpu_mask[e] != 0u, run = !cpu && cnt > 0;   // cpu_mask is U32 [N_EXPERT]
    const int4 v = int4(cpu ? 0 : cnt, cpu ? cnt : 0, run ? 1 : 0, run ? (cnt + 7) / 8 : 0);
    int4 x = simd_prefix_exclusive_sum(v);                                   // integer: exact in any order
    if (lane == 31) part[sg] = x + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int4 all = int4(0);
    for (uint s = 0; s < MMQ_EXPERTS / 32; ++s) {
        if (s < sg) x += part[s];
        all += part[s];
    }
    const int off = cpu ? all.x + x.y : x.x;                                 // all.x = rows_local
    int f = off;
    for (uint r = 0; r < nr; ++r) {
        device int* w = col + ulong(r) * (MMQ_EXPERTS + 1);
        const int c = *w;
        *w = f;
        f += c;
    }
    if (run) {
        plan[2 + x.z] = int(e);
        plan[MMQ_BOUNDS + x.z] = off;
        plan[MMQ_WPRE + x.z] = x.w;
    }
    if (e == 0) {
        plan[0] = all.z;                                                     // n
        plan[1] = all.x;                                                     // rows_local
        plan[MMQ_BOUNDS + all.z] = all.x;
        plan[MMQ_WPRE + all.z] = all.w;
    }
}

// Fill (prefill.cpp:1911-1917): pair i = MMQ_TOPK t + k of range r, one thread each, takes p = fill[e]++ as basalt's
// walk in pair order gives it, fill[e] at the range's first pair (basalt_mmq_plan_scan's column) plus the range's
// earlier pairs routed to e. An expert's rows are its pairs in pair order, a token's repeated id included (consecutive
// rows), so no two pairs share a row and every row below 10 R is written once. Grid ceil(B / MMQ_RANGE) x
// MMQ_RANGE * MMQ_TOPK; pairs with t >= R return. It never meets an absent id: basalt_mmq_plan_scan set done for one.
kernel void basalt_mmq_plan_fill(device const int* ids [[buffer(0)]], device const int* hist [[buffer(1)]],
                                 device int* slot [[buffer(3)]], device int* src [[buffer(4)]],
                                 device const StepState* st [[buffer(15)]],
                                 uint r [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    if (st->done) return;                                                    // uniform: before any load
    const uint t = r * MMQ_RANGE + i / MMQ_TOPK;
    if (t >= prompt_rows(st)) return;                                        // ranges at or past R too
    device const int* rid = ids + ulong(r) * (MMQ_RANGE * MMQ_TOPK);
    const int e = rid[i];
    int p = hist[ulong(r) * (MMQ_EXPERTS + 1) + uint(e)];
    for (uint j = 0; j < i; ++j) p += int(rid[j] == e);                      // at most 159 earlier pairs
    slot[ulong(r) * (MMQ_RANGE * MMQ_TOPK) + i] = p;                         // :1914-1915
    src[p] = int(t);                                                         // :1916
}

// quantize.cu:39-81 in the D4 layout (spec-kernels-mmq.md 3.2 steps 1-6, 9.7): the thread's four values xi are values
// iqs .. iqs + 3 of the block_q8_1_mmq at blk (36 words: d4[4], then the 128 int8 codes), and the 8 lanes of each
// 32-value group (lanes 8g .. 8g + 7 of the SIMD-group) share its amax through the butterfly. The max of flushed
// magnitudes is order-free (fmax drops a NaN, so amax is NaN only for an all-NaN group); d_inv = 127 / amax is
// MUFU.RCP then FMUL.FTZ, d = 1 / d_inv MUFU.RCP: rcp_ftz, contract invariant 5.
static inline void mmq_quantize_block(float4 xi, device uint* blk, uint iqs) {
    float amax = max_ftz(max_ftz(max_ftz(abs_ftz(xi.x), abs_ftz(xi.y)), abs_ftz(xi.z)), abs_ftz(xi.w));   // :40-43
    for (ushort o = 4; o > 0; o >>= 1) amax = max_ftz(amax, shfl_xor(amax, o));                              // :46-49
    const float d_inv = mul_ftz(127.0f, rcp_ftz(amax));                                                      // :51
    blk[4u + iqs / 4u] = as_type<uint>(char4(mmq_code(mul_ftz(xi.x, d_inv)), mmq_code(mul_ftz(xi.y, d_inv)),  // :53-56,
                                             mmq_code(mul_ftz(xi.z, d_inv)), mmq_code(mul_ftz(xi.w, d_inv))));  // :74-75
    if (iqs % 32u == 0u) blk[iqs / 32u] = as_type<uint>(cuda_nan(rcp_ftz(d_inv)));                            // :57, :77-81
}

// Kernel 1 (quantize.cu:10-84 as quantize.cu:94-96 launches it from prefill.cpp:1990: the layer's activations through
// src_dev; spec-kernels-mmq.md 9.7). x [B][K] floats, its binding 16-byte aligned (basalt's float4 loads,
// quantize.cu:32, :39); src [10 B] the plan's slot-to-token rows; Xq [10 B][K / 128] blocks of 144 bytes, a slot's
// row contiguous (basalt's are k-block-major with the row count as stride: an address change, 9.1); plan its words.
// Threadgroup (p, by) of MMQ_THREADS quantizes values 512 by .. 512 by + 511 of x's row src[p] into blocks 4 by ..
// 4 by + 3 of Xq's row p, four values a thread (quantize.cu:18, :35-36). Slots p >= bounds[n] return (bounds[n] is
// the plan's rows_local, prefill.cpp:1996: below 10 R only when the CPU holds experts, whose last block of rows basalt
// quantizes too and no GPU kernel reads). Grid (10 B, K / 512).
kernel void basalt_mmq_quantize(device const float* x [[buffer(0)]], device const int* src [[buffer(1)]],
                                device uint* xq [[buffer(2)]], device const int* plan [[buffer(3)]],
                                device const StepState* st [[buffer(15)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (st->done) return;                                                    // uniform: before any load
    if (int(tg.x) >= plan[MMQ_BOUNDS + plan[0]]) return;                     // uniform per threadgroup
    const uint i0 = 512u * tg.y + 4u * tid;
    const float4 xi = *(device const float4*) (x + ulong(src[tg.x]) * K + i0);                     // :29-39
    mmq_quantize_block(xi, xq + (ulong(tg.x) * (K / 128) + i0 / 128u) * 36u, i0 % 128u);          // :70-71
}

// The down product's padded sub-blocks (spec-kernels-mmq.md 4.3, 9.6). Its rows of BPNE = 10 blocks are not a multiple
// of BPI = 4, so its last k iteration (kb0 = 8) runs sub-blocks 2 BPNE .. 2 BPNE + 3, blocks BPNE and BPNE + 1 past
// weight row i, which basalt's tile load reads from its group buffer: the group's down matrices packed in launch-slot
// order, then MMQ_TAIL zeroed bytes (prefill.cpp:608-611, :848-849, :2249-2251). Their activation codes are Hq's
// padding, 0, so c = 0 and only their dA reaches the sum: the sign of a -0 sum. mmq_dA reads that dA in place through
// the address table, never past an expert's matrix: block b - BPNE of row i + 1; past the matrix's last row (NTY 128
// rows) block b - BPNE of the next launch slot's row 0; past the group's last slot jl the tail's d = 0x0000.
static inline float mmq_dA(device const GrpPtr* tab, device const int* order, int j, int jl, int i, int s) {
    int b = s / 2;
    if (b >= BPNE) { b -= BPNE; ++i; }            // block b - 10 of row i + 1
    if (i == NTY * 128) {                         // past the expert's matrix
        if (j == jl) return 0.0f;                 // MMQ_TAIL: d = 0x0000
        i = 0; ++j;                               // the next slot's row 0
    }
    return h2f(tab[order[j]].p[MAT_OFF + (ulong(i) * BPNE + ulong(b)) * 9]);
}

// Kernels 2 and 4 (spec-kernels-mmq.md 9.2-9.6): mul_mat_q<Q2_0> (mmq.cuh:390-864) of every launch of the plan, the
// gate/up product's groups of prefill.cpp:2243-2256 (kernel 2) or the down product's of :2259-2264 (kernel 4,
// MMQ_DOWN), in one dispatch, with the stream-k fixup (:632-767) folded per output element. xq [10 B][pad512(K) / 128]
// blocks of 144 bytes, kernel 1's slots (kernel 3's for the down product); tab [512] one GrpPtr per expert id, its
// blob (what basalt gathers into its group buffers, prefill.cpp:2222-2240, read in place: weight row i at MAT_OFF + 9
// BPNE i ushorts); plan its words; y [10 B][LD] floats, slot p's outputs at row p (basalt's ids_identity). Grid (NTY,
// W_cap) x MMQ_THREADS, W_cap >= wpre[n] (monolith/kernels.py basalt_mmq_grids): threadgroup (it, w) computes weight
// rows 128 it .. 128 it + 127, one a thread, against the 8-row window u = w - wpre[j] of launch slot j, the j with
// wpre[j] <= w < wpre[j + 1]: slots bounds[j] + 8 u + c, c < ncol = min(8, bounds[j + 1] - bounds[j] - 8 u). Every J
// is a multiple of 8 and basalt's column tiles start at multiples of J from the expert's first row, so the window lies
// in one tile t = (it ngx + zt) ntx + jt (:497-506), jt = 8 u / J, whose elements share its pieces; an empty column
// tile (:523-531) has no window, and columns past the expert's rows (:206-208) are not computed. Work items w >=
// wpre[n] return. Each element is the tile's pieces P_0 .. P_m (mmq_piece_mask), each a chain from +0 (:340) over its
// k iterations in K order, kb0 then the two 128-value halves then k01 (:344-377, mmq_vec_dot.cuh:63): per 32-value
// sub-block s the exact int32 dot c of the weight codes (mmq_load_tiles.cuh:42-58: half s % 2 of block s / 2) with the
// activation codes, then sum = fma_ftz(dB, mul_ftz(dA, float(c)), sum) (mmq_vec_dot.cuh:87: I2FP, FMUL.FTZ dA * c,
// FFMA.FTZ), dA the block's fp16 d (mmq_load_tiles.cuh:73), dB the activation group's d4 (mmq_vec_dot.cuh:74). One
// piece is stored as it is (:216); else P_{m-1} .. P_0 are added from +0, nearest first (the fixup, :646, :689), and
// that sum onto P_m (:765), add_ftz each. Stores through cuda_nan (sm_120's NaN). The down product's last iteration's
// sub-blocks past row i run the same step with c = 0 and mmq_dA's dA, their weight codes never loaded.
kernel void basalt_mmq(device const uint* xq [[buffer(0)]], device const GrpPtr* tab [[buffer(1)]],
                       device const int* plan [[buffer(2)]], device float* y [[buffer(3)]],
                       device const StepState* st [[buffer(15)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (st->done) return;                                                    // uniform: before any load
    const int n = plan[0], w = int(tg.y);
    device const int* bounds = plan + MMQ_BOUNDS;
    device const int* wpre = plan + MMQ_WPRE;
    if (w >= wpre[n]) return;                                                // uniform per threadgroup
    int j = 0;                                                               // the last slot with wpre[j] <= w: wpre
    for (int h = 256; h > 0; h >>= 1)                                        // rises (cnt >= 1), n <= 512
        if (j + h < n && wpre[j + h] <= w) j += h;
    const int u = w - wpre[j];
    const MmqLaunch L = mmq_launch(bounds, n, j / 16, NTY);
    const int jl = 16 * (j / 16) + L.ngx - 1;                                // the group's last launch slot
    const uint mask = mmq_piece_mask((int(tg.x) * L.ngx + j % 16) * L.ntx + 8 * u / L.J, L.ntiles, BPNE, BPI, L.grid);
    int b[NITER + 1], m = 0;                                                 // the pieces' first k iterations
    b[0] = 0;
    for (int q = 1; q < NITER; ++q)
        if ((mask >> q) & 1u) b[++m] = q;
    b[m + 1] = NITER;
    // ponytail: a window re-reads its 128-row weight slab, so an expert's weights stream ceil(cnt / 8) times a product
    // (26 ms a gate/up dispatch at B = 256 on the M3 Max); windows of 16 or 32 rows, up to J, inside one basalt tile
    // change only wpre and no bit (spec-kernels-mmq.md 9.5).
    const int p0 = bounds[j] + 8 * u, ncol = min(8, bounds[j + 1] - p0);
    const int i = 128 * int(tg.x) + int(tid);
    device const ushort* wr = tab[plan[2 + j]].p + MAT_OFF + ulong(i) * (9 * BPNE);
    float S[8];
    for (int c = 0; c < 8; ++c) S[c] = 0.0f;
    for (int pp = m - 1; pp >= -1; --pp) {                                   // P_{m-1} .. P_0 into S, then P_m
        const int pc = pp < 0 ? m : pp;
        float sum[8];
        for (int c = 0; c < 8; ++c) sum[c] = 0.0f;
        for (int q = b[pc]; q < b[pc + 1]; ++q)
            for (int s = 8 * q; s < 8 * q + 8; ++s) {
                const bool pad = MMQ_DOWN && s >= 2 * BPNE;                     // past row i: c = 0, mmq_dA's dA
                const Q20W wq = pad ? Q20W{} : q2_0_load(wr + 9 * (s / 2), s % 2);  // mmq_load_tiles.cuh:42-58, :73
                const float dA = pad ? mmq_dA(tab, plan + 2, j, jl, i, s) : wq.d2;
                for (int c = 0; c < 8 && c < ncol; ++c) {
                    device const uint* yb = xq + (ulong(p0 + c) * ((K + 511) / 512 * 4) + ulong(s / 4)) * 36u;
                    const float dB = as_type<float>(yb[s % 4]);                         // mmq_vec_dot.cuh:74
                    int ci = 0;
                    for (int k = 0; k < 4 && !pad; ++k) {
                        ci = dp4a(int(yb[4 + 8 * (s % 4) + 2 * k]), wq.qx[k], ci);
                        ci = dp4a(int(yb[4 + 8 * (s % 4) + 2 * k + 1]), wq.qy[k], ci);
                    }
                    sum[c] = fma_ftz(dB, mul_ftz(dA, float(ci)), sum[c]);               // mmq_vec_dot.cuh:87
                }
            }
        for (int c = 0; c < 8 && c < ncol; ++c) {
            if (pp >= 0) S[c] = add_ftz(S[c], sum[c]);                                          // mmq.cuh:646, :689
            else y[ulong(p0 + c) * LD + ulong(i)] = cuda_nan(m == 0 ? sum[c] : add_ftz(sum[c], S[c]));   // :216, :765
        }
    }
}

// Kernel 3 (spec-kernels-mmq.md 2, 9.7): swiglu_kernel (moe_mmq.cu:85-93, interleaved false) then quantize.cu:10-84
// on its H, as prefill.cpp:2257-2258 run mmq::swiglu and mmq::quantize on a group's rows (pad512(n_ff) values a row,
// moe_mmq.cu:154-159), fused: H never exists, which is exact because the quantizer reads each stored h once and an
// h's NaN payload never reaches a store (an all-NaN group stores cuda_nan(1 / NaN)). Compiled with the down
// product's macros: K = n_ff = 640. gu [10 B][2 K] floats, slot p's gate outputs then its up outputs, its binding
// 16-byte aligned; Hq [10 B][pad512(K) / 128] blocks of 144 bytes, a slot's row contiguous; plan its words.
// Threadgroup (p, by) of MMQ_THREADS forms values 512 by .. 512 by + 511 of H's row p, four a thread (h below K, the
// float4 zero of quantize.cu:39 above), and quantizes them into blocks 4 by .. 4 by + 3 of Hq's row p. Slots p >=
// bounds[n] return. Grid (10 B, pad512(K) / 512).
kernel void basalt_mmq_swiglu_quantize(device const float* gu [[buffer(0)]], device const int* plan [[buffer(2)]],
                                       device uint* hq [[buffer(3)]], device const StepState* st [[buffer(15)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
    if (st->done) return;                                                    // uniform: before any load
    if (int(tg.x) >= plan[MMQ_BOUNDS + plan[0]]) return;                     // uniform per threadgroup
    const uint i0 = 512u * tg.y + 4u * tid;
    float4 h = float4(0.0f);                                                 // quantize.cu:39 past ne00 = K
    if (i0 < uint(K)) {
        device const float* row = gu + ulong(tg.x) * (2 * K);                                         // moe_mmq.cu:90
        const float4 g = *(device const float4*) (row + i0), u = *(device const float4*) (row + K + i0);     // :91
        // :92 h[i] = g / (1.0f + __expf(-g)) * u: FMUL.FTZ by -log2 e and MUFU.EX2 (expf_ftz), FADD.FTZ, MUFU.RCP
        // then FMUL.FTZ by g (div.approx.ftz), FMUL.FTZ by u (docs/research/basalt-nvcc/mmq.md section 7)
        for (int c = 0; c < 4; ++c) h[c] = mul_ftz(mul_ftz(rcp_ftz(add_ftz(expf_ftz(-g[c]), 1.0f)), g[c]), u[c]);
    }
    mmq_quantize_block(h, hq + (ulong(tg.x) * ((K + 511) / 512 * 4) + i0 / 128u) * 36u, i0 % 128u);     // :70-71
}
