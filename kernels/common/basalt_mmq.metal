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
