// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/sampler.cu, include/basalt/kernels/sampler.hpp and
// include/basalt/core/coupled_draft.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// written in the rounding sequence nvcc compiles for them (docs/research/basalt-nvcc/sampler.md).
//
// basalt's sampler family on Metal. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal and
// basalt_libdevice.metal ahead of it (monolith/kernels.py's basalt_sampler_source()).
//
// The entry points need STEP_STATE=1 and the StepState struct (the emitter splices it in): every one returns at its
// first instruction when st->done or its row is at or past st->t_this_step. Without STEP_STATE only the helpers
// compile. Their bindings, one slot per buffer across the family:
//   0 logits (T, V) f32 | 1 ctr_base (2,) u32 | 2 nq (1,) u32 | 3 bias (V,) f32 | 4 history (T, H) i32
//   5 token (T,) i32 | 6 qprob (T,) f32 | 7 BsSampleParams | 8 part: the greedy partials, (key, id) per row and
//   threadgroup | 9 cand | 10 bits | 15 StepState
#pragma STDC FP_CONTRACT OFF
#ifndef STEP_STATE
#define STEP_STATE 0
#endif

// ---- Keys and compares on the bits ----------------------------------------------------------------------------------
//
// CUDA's setp and max.f32 without .ftz order subnormals exactly and compare -0 equal to +0; this GPU's <, > and ==
// read a subnormal as zero (tests/kernels/test_basalt_platform.py), so every compare that orders or selects a float
// here runs on its bits (tests/kernels/test_basalt_sampler_helpers.py, H.6).

// The order-preserving key of a non-NaN float's bits: a < b exactly when okey(a) < okey(b), and a == b exactly when
// the keys are equal (-0 and +0 share one key).
static inline uint okey(uint b) {
    b = (b == 0x80000000u) ? 0u : b;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
static inline bool fnan(uint b) { return (b & 0x7fffffffu) > 0x7f800000u; }
// CUDA's ordered > and < (false when either operand is NaN)
static inline bool f_gt(uint a, uint b) { return !fnan(a) && !fnan(b) && okey(a) > okey(b); }
static inline bool f_lt(uint a, uint b) { return !fnan(a) && !fnan(b) && okey(a) < okey(b); }

// CUDA's fmaxf (max.f32 without .ftz): the other operand when one is NaN, 0x7fffffff (NVIDIA's canonical NaN) when
// both are, else the larger by okey; on -0 against +0 the first operand (no output depends on that zero's sign).
static inline uint fmaxf_bits(uint a, uint b) {
    if (fnan(a)) return fnan(b) ? 0x7fffffffu : b;
    if (fnan(b)) return a;
    return okey(b) > okey(a) ? b : a;
}

// ---- IEEE division and fma where a subnormal can occur -------------------------------------------------------------
//
// CUDA's div.rn.f32 and fma.rn.f32 in an IEEE-built kernel keep subnormal operands and results; this GPU flushes them,
// an fma on its exact value, before rounding (tests/kernels/test_basalt_platform.py). As ieee_mul does,
// each helper takes the hardware's correctly rounded result when every operand has a nonzero exponent field (normal,
// infinite or NaN) and the result is at least 2^-125, infinite or NaN: the hardware rounds such an exact value once,
// a product below the normal range or beyond FLT_MAX in the fma included (tests/kernels/test_basalt_sampler_helpers.py,
// H.3). Every other case (a zero or subnormal operand, a result the hardware may have flushed) is rounded from the
// software binary64. A NaN from the software is 0x7fffffff, NVIDIA's canonical NaN; the hardware's NaN keeps its own
// bits.

// a / b, and ieee_div(1, x) for rcp.rn.f32: the binary64 quotient, rounded to nearest even and then to binary32. That
// double rounding is the correctly rounded quotient: on the normal grid since 53 >= 2 * 24 + 2 (Figueroa), and on the
// subnormal grid because a quotient of floats off a midpoint lies more than 2^-48 of itself from it, beyond binary64's
// rounding, while one on a midpoint stays on it (H.3).
static inline float ieee_div(float a, float b) {
    const float q = a / b;
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b);
    if ((ab & 0x7f800000u) != 0u && (bb & 0x7f800000u) != 0u && !(fabs(q) < 0x1p-125f)) return q;
    return as_type<float>(f32_from_f64(f64_div(f64_from_f32(ab), f64_from_f32(bb))));
}

// fma(a, b, c), the penalty's fma(float(count), freq, present) (sampler.cu:75): the exact a * b + c rounded to odd in
// binary64 and then to nearest even in binary32, which is the correctly rounded fma (53 >= 24 + 2: Boldo and
// Melquiond). f32_from_f64(f64_fma(...)) would round twice and can land on the wrong neighbour of a binary32 midpoint.
static inline float ieee_fma_count(float a, float b, float c) {
    const float r = fma(a, b, c);
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b), cb = as_type<uint>(c);
    if ((ab & 0x7f800000u) != 0u && (bb & 0x7f800000u) != 0u && (cb & 0x7f800000u) != 0u && !(fabs(r) < 0x1p-125f))
        return r;
    return as_type<float>(f32_from_f64(f64_fma_odd(f64_from_f32(ab), f64_from_f32(bb), f64_from_f32(cb))));
}

// ---- The approximate exponential ------------------------------------------------------------------------------------

// CUDA's __expf(x) in an IEEE-built kernel (sampler.cu:311): ex2.approx.f32 of the IEEE product x * log2(e)
// (mul.f32 by 0x3fb8aa3b). Invariant 5 maps it to fast::exp2 of that product, without ex2.approx's branch for
// subnormal results; the product is ieee_mul's, since x = l[i] - m can be subnormal.
static inline float expf_approx(float x) { return fast::exp2(ieee_mul(x, as_type<float>(0x3fb8aa3bu))); }

// ---- Philox (sampler.cu:37-59) --------------------------------------------------------------------------------------
//
// basalt's own variant, not Random123's Philox4x32-10: the multipliers are the Weyl constants, the round key is
// (i, 0), and the seed fills counter words c2 and c3. Integers only, so it ports bit for bit. The draw is
// (c0 >> 8) * 2^-24, exact: a multiple of 2^-24 in [0, 1 - 2^-24].
static inline float philox_uniform(ulong seed, ulong counter) {
    uint c0 = uint(counter), c1 = uint(counter >> 32), c2 = uint(seed), c3 = uint(seed >> 32);
    for (uint i = 0u; i < 10u; ++i) {                                   // bounded: ten rounds
        const uint hi0 = mulhi(0x9E3779B9u, c0), hi1 = mulhi(0xBB67AE85u, c2);
        const uint lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint n0 = hi1 ^ c1 ^ i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return float(c0 >> 8) * 0x1p-24f;
}

// ---- Penalties (sampler.cu:65-77) ------------------------------------------------------------------------------------
//
// llama.cpp's penalties stage as nvcc compiles it (docs/research/basalt-nvcc/sampler.md 2.1). A logit the window
// does not count keeps its bits. A counted one is divided by repeat when it is positive or NaN (setp.gtu.f32 on its
// bits, so a positive subnormal divides), else multiplied, and then less fma(count, freq, present): the presence term
// (count > 0 ? 1 : 0) * present folds to present, since count > 0 there. Every operation is CUDA's IEEE one, with
// subnormals.

// -x by its sign bit (unary minus on a subnormal is not measured on this GPU)
static inline float neg(float x) { return as_type<float>(as_type<uint>(x) ^ 0x80000000u); }

static inline uint history_count(device const int* h, uint n, uint v) {
    uint c = 0u;
    for (uint i = 0u; i < n; ++i)                                       // bounded: the window's n slots
        if (h[i] == int(v)) ++c;
    return c;
}

// apply_penalties on a logit's bits, with count its exact count over the window
static inline uint apply_penalties(uint b, uint count, float repeat, float freq, float present) {
    if (count == 0u) return b;
    const float x = as_type<float>(b);
    const float y = (fnan(b) || f_gt(b, 0u)) ? ieee_div(x, repeat) : ieee_mul(x, repeat);
    return as_type<uint>(ieee_add(y, neg(ieee_fma_count(float(count), freq, present))));
}

// ---- The request's record (monolith/kernels.py's basalt_sampler_params) ---------------------------------------------
//
// basalt's SamplerParams without its per-step words: the seed is StepState's rng_lo / rng_hi, row t's counter
// ctr_base + st->position + t. vocab, t_rows, n_blocks and hist_len are the program's; the rest the request's. mode:
// 0 greedy (the cluster argmax alone), 1 greedy with a penalty window or a bias (the one-block greedy kernel after
// it), 2 sampled. flags: 1 BIAS, 2 PROBE.
struct BsSampleParams {
    uint vocab, t_rows, n_blocks, hist_len;
    uint mode, flags;
    int top_k;
    float top_p, min_p, temperature;
    int min_keep, penalty_last_n;
    float penalty_repeat, penalty_freq, penalty_present;
    uint pad;
};
constant uint BS_BIAS = 1u;                                            // flags: add bias[v] to every logit first

// ---- The greedy argmax on (key, id) pairs: am_take, am_merge (sampler.cu:180-185) -----------------------------------
//
// CUDA's s > bv and ov > bv || (ov == bv && oi < best) on the keys of the bits. A thread starts at (-inf, vocab), so
// a NaN or -inf logit is never taken, and every pair a tree meets is a taken value or that sentinel: over them the
// merge is a strict total order (the larger value, the lower id on equal values, -0 equal to +0), which makes the
// pick the row's lowest id of its largest non-NaN value above -inf whatever the partition, else vocab
// (sampler.cu:163-167).
constant uint BS_KEY_NINF = 0x007fffffu;                               // okey(-inf)

static inline void am_take(uint s, uint v, thread uint& bk, thread uint& bi) {
    if (!fnan(s) && okey(s) > bk) { bk = okey(s); bi = v; }
}
static inline void am_merge(uint ok, uint oi, thread uint& bk, thread uint& bi) {
    if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
}
// __shfl_down_sync at 16, 8, 4, 2, 1: lane 0 ends with the SIMD-group's first pair
static inline void am_tree(thread uint& bk, thread uint& bi) {
    for (ushort off = 16; off > 0; off >>= 1) am_merge(simd_shuffle_down(bk, off), simd_shuffle_down(bi, off), bk, bi);
}
// The SIMD-group tree, then SIMD-group 0's tree over the nsg SIMD-groups' pairs (sampler.cu:135-153, :211-221): true
// on SIMD-group 0's lane 0, which ends with the threadgroup's first pair. Every thread of the threadgroup calls it.
static inline bool am_block(thread uint& bk, thread uint& bi, threadgroup uint* sk, threadgroup uint* si, uint lane,
                            uint sg, uint nsg, uint nv) {
    am_tree(bk, bi);
    if (lane == 0u) { sk[sg] = bk; si[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u) return false;
    bk = lane < nsg ? sk[lane] : BS_KEY_NINF;
    bi = lane < nsg ? si[lane] : nv;
    am_tree(bk, bi);
    return lane == 0u;
}

#if STEP_STATE
constant uint BS_AM_CTAS = 8u;                                         // kAmCtas: threadgroups per row

// sampler_greedy_cluster_kernel (sampler.cu:188-244) up to its cluster exchange: threadgroup r of row t (grid
// (8, rows), basalt's 1,024 threads) scans v = r * tpg + tid + m * 8 * tpg in ascending order, runs the SIMD-group and
// threadgroup trees, and stores its pair to part[t * 8 + r].
kernel void bs_greedy_partial(device const uint* logits [[buffer(0)]], constant BsSampleParams& p [[buffer(7)]],
                              device uint2* part [[buffer(8)]], device const StepState* st [[buffer(15)]],
                              uint2 g [[threadgroup_position_in_grid]], uint2 tg [[threads_per_threadgroup]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup uint sk[32], si[32];
    const uint r = g.x, t = g.y, tpg = tg.x;
    if (st->done || t >= st->t_this_step || r >= BS_AM_CTAS) return;     // r: the store's slot stays in row t
    const uint nv = p.vocab;
    device const uint* l = logits + ulong(t) * nv;
    uint bk = BS_KEY_NINF, bi = nv;
    for (uint v = r * tpg + tid; v < nv; v += BS_AM_CTAS * tpg) am_take(l[v], v, bk, bi);   // bounded by nv
    if (am_block(bk, bi, sk, si, lane, sg, nsg, nv)) part[ulong(t) * BS_AM_CTAS + r] = uint2(bk, bi);
}

// The cluster kernel's CTA-0 merge (sampler.cu:233-238): one SIMD-group per row over its 8 partials.
kernel void bs_greedy_merge(device int* out [[buffer(5)]], constant BsSampleParams& p [[buffer(7)]],
                            device const uint2* part [[buffer(8)]], device const StepState* st [[buffer(15)]],
                            uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (st->done || t >= st->t_this_step) return;
    const uint nv = p.vocab;
    const uint2 c = lane < BS_AM_CTAS ? part[ulong(t) * BS_AM_CTAS + lane] : uint2(BS_KEY_NINF, nv);
    uint bk = c.x, bi = c.y;
    am_tree(bk, bi);
    if (lane == 0u) out[t] = bi < nv ? int(bi) : 0;
}

// sampler_greedy_kernel (sampler.cu:91-157), mode 1: each row's argmax after the logit bias (flags BIAS) and the
// penalties over the tail of its history row, its last min(penalty_last_n, hist_len) slots. One threadgroup per row,
// of any multiple of 32 threads up to basalt's 1,024 (the pick is order-free). The window's membership bitmap filters
// the exact count, as basalt's shared one does: row t's ceil(vocab / 32) words of bits (device scratch, one region
// per row), zeroed and ORed in every step that has a window, never touched without one; ids outside the vocabulary
// set nothing.
kernel void bs_greedy_one_block(device const uint* logits [[buffer(0)]], device const float* bias [[buffer(3)]],
                                device const int* history [[buffer(4)]], device int* out [[buffer(5)]],
                                constant BsSampleParams& p [[buffer(7)]], device atomic_uint* pbits [[buffer(10)]],
                                device const StepState* st [[buffer(15)]],
                                uint t [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup uint sk[32], si[32];
    if (st->done || t >= st->t_this_step || p.mode != 1u) return;
    const uint nv = p.vocab, words = (nv + 31u) / 32u;
    const uint hlen = uint(clamp(p.penalty_last_n, 0, int(p.hist_len)));                 // :97-103
    device const int* hrow = history + ulong(t) * p.hist_len + (p.hist_len - hlen);      // the window is the tail
    device atomic_uint* bits = pbits + ulong(t) * words;
    if (hlen > 0u) {                                                                      // :110-121
        for (uint w = tid; w < words; w += tpg) atomic_store_explicit(bits + w, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < hlen; i += tpg) {
            const int id = hrow[i];
            if (id >= 0 && uint(id) < nv)
                atomic_fetch_or_explicit(bits + (uint(id) >> 5), 1u << (uint(id) & 31u), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    device const uint* l = logits + ulong(t) * nv;
    const bool biased = (p.flags & BS_BIAS) != 0u;
    uint bk = BS_KEY_NINF, bi = nv;
    for (uint v = tid; v < nv; v += tpg) {                                                // :131-134, bounded by nv
        const uint x = biased ? as_type<uint>(ieee_add(as_type<float>(l[v]), bias[v])) : l[v];
        const bool hit = hlen > 0u
                         && (atomic_load_explicit(bits + (v >> 5), memory_order_relaxed) & (1u << (v & 31u))) != 0u;
        am_take(apply_penalties(x, hit ? history_count(hrow, hlen, v) : 0u, p.penalty_repeat, p.penalty_freq,
                                p.penalty_present), v, bk, bi);
    }
    if (am_block(bk, bi, sk, si, lane, sg, nsg, nv)) out[t] = bi < nv ? int(bi) : 0;     // :135-156
}
#endif
