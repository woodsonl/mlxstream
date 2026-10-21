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
//
// The kernels (spec-kernels-mmq.md 9.1). Each returns at StepState.done before its first load, so once the plan
// refuses a routed id nothing more of the step is stored.
// - basalt_mmq_quantize (kernel 1): the layer's activations to Xq through the plan's src (quantize.cu:10-84, launched
//   :94-96 from prefill.cpp:1990).
// - basalt_mmq_swiglu_quantize (kernel 3): GU's split halves to Hq, mmq::swiglu (moe_mmq.cu:85-93) fused with the
//   quantizer on H (prefill.cpp:2257-2258's group calls), compiled with the down product's macros (K = n_ff).
//
// Macros (monolith/kernels.py basalt_mmq_macros): MMQ_TYPE (42 Q2_0, 8 Q8_0), K, NTY, LD, BPNE, BPI, NITER, MMQ_DOWN,
// MMQ_DENSE, ROWS_PER_T, MMQ_SM_COUNT, MAT_OFF, MMQ_ERR; and PROMPT_ROWS' T_SRC, ROWS_MAX and T_STATIC_ROWS.
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
// MMQ_BOUNDS, then wpre[513].
constant int MMQ_BOUNDS = 514;

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
