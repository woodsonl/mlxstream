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
//   5 token (T,) i32 | 6 qprob (T,) f32: the probe's probability, or the top-prob kernel's | 7 BsSampleParams
//   8 part: the greedy partials, (key, id) per row and threadgroup | 9 cand | 10 bits
//   11 map (V,) i32: the top-prob kernel's subset map | 15 StepState
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
// it), 2 sampled. flags: 1 BIAS, 2 PROBE, 4 MAP (the top-prob kernel's subset map, the program's).
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
constant uint BS_PROBE = 2u;                                           // flags: the sampled tail writes qprob
constant uint BS_MAP = 4u;                                             // flags: the top-prob pick goes through map

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

// ---- The sampled list: sampled_k, take_first, warp_first and the list merge (sampler.cu:327-360, :532-575) ---------
//
// The selection order (sampler.cu:327-333): (v, i) comes before (v', i') when v > v', or v == v' and i < i', on the
// keys of the bits: a strict total order over non-NaN values, -0 and +0 tying and falling to the id. Every round
// starts from -inf and takes on a strict >, so NaN and -inf are never picked; a round that finds nothing yields the
// sentinel (-inf, vocab). A list entry is (id, value bits), x the id: the value's own bits, which the tail computes
// with (okey merges -0 with +0, so it could not give them back).
constant uint BS_SEL_MAX = 64u;                                        // kSelMax: the longest list
constant uint BS_NINF_BITS = 0xff800000u;                              // -inf, the sentinel's value

// top_k 1..63 as given, anything else 64, never more than the vocabulary (sampler.cu:339-342)
static inline uint sampled_k(int top_k, uint nv) {
    return min((top_k > 0 && top_k < int(BS_SEL_MAX)) ? uint(top_k) : BS_SEL_MAX, nv);
}

// (bb, bi) <- the first of (bb, bi) and (ob, oi) in the selection order (sampler.cu:344-347); neither value is NaN
static inline void take_first(thread uint& bb, thread uint& bi, uint ob, uint oi) {
    const uint kb = okey(bb), ko = okey(ob);
    if (ko > kb || (ko == kb && oi < bi)) { bb = ob; bi = oi; }
}

// The first of the SIMD-group's 32 pairs, left in every lane (sampler.cu:349-359): the XOR butterfly at 16, 8, 4, 2,
// 1. The first of two is associative and commutative under the strict order, and two lanes never hold different
// pairs that compare equal (an id is in one lane at most, and every sentinel is the same pair), so every lane ends
// with the winner's own bits.
static inline void warp_first(thread uint& bb, thread uint& bi) {
    for (ushort off = 16; off > 0; off >>= 1) {
        const uint ob = simd_shuffle_xor(bb, off), oi = simd_shuffle_xor(bi, off);
        take_first(bb, bi, ob, oi);
    }
}

// warp_merge_lists (sampler.cu:532-575): the first k pairs of nl <= 64 lists of k pairs, list L at lists[L * stride],
// each in the selection order and padded with sentinels. One template over the lists' pointer: threadgroup for the
// part kernels' SIMD-group lists, device for the merges' block lists in cand. Lane owns lists lane and lane + 32:
// merge_heads loads their heads, and each merge_lists call is one round, the first of every lane's heads, returned in
// every lane as (id, value bits), whose list then advances to its next entry, or to the sentinel after its k-th
// (exactly one head holds a real id: the lists cover disjoint ids; a sentinel round advances nothing). The caller runs
// the k rounds and stores from lane 0. Loads stay inside lists[nl * stride]: only a list below nl advances, and its
// position stays below k <= stride.
struct BsHeads {
    uint b[2], i[2], pos[2];
};

template <typename L>
static inline BsHeads merge_heads(L lists, uint nl, uint stride, uint nv, uint lane) {
    BsHeads h;
    for (uint m = 0u; m < 2u; ++m) {
        const uint li = lane + 32u * m;
        h.pos[m] = 0u;
        h.b[m] = BS_NINF_BITS;
        h.i[m] = nv;
        if (li < nl) {
            const uint2 c = lists[li * stride];
            h.i[m] = c.x;
            h.b[m] = c.y;
        }
    }
    return h;
}

template <typename L>
static inline uint2 merge_lists(thread BsHeads& h, L lists, uint stride, uint k, uint nv, uint lane) {
    uint bb = h.b[0], bi = h.i[0];
    take_first(bb, bi, h.b[1], h.i[1]);
    warp_first(bb, bi);
    if (bi < nv) {
        for (uint m = 0u; m < 2u; ++m) {
            if (h.i[m] != bi) continue;
            if (++h.pos[m] < k) {
                const uint2 c = lists[(lane + 32u * m) * stride + h.pos[m]];
                h.i[m] = c.x;
                h.b[m] = c.y;
            } else {
                h.i[m] = nv;
                h.b[m] = BS_NINF_BITS;
            }
        }
    }
    return uint2(bi, bb);
}

// ---- One block's top-k list: the split part and the coupled part (sampler.cu:577-661, :733-777) --------------------
//
// sampler_split_part_kernel's body and coupled_part_kernel's, one template so that the two lists cannot drift apart:
// block blk's list from KWARPS SIMD-groups of KPER logits per lane, KWARPS * 32 * KPER ids from blo (4,096 in both),
// into dst (the block's k entries). The caller's threadgroup is exactly KWARPS * 32 threads and wl holds KWARPS lists
// of BS_SEL_MAX entries. Lane lane of SIMD-group sg holds s[j], the bits of logit lo + 32 j + lane (one coalesced
// line per j), -inf past the vocabulary. PEN, the split part: the logit bias (biased) and the penalties of the window
// hrow[0, hlen), which a logit takes when its bit is set in the block's 128-word bitmap bits (the window's ids in the
// block): apply_penalties with the count over the whole window, a clear bit keeping the logit as a zero count does.
// Without PEN (the coupled part, whose logits arrive penalised) bits, bias and hrow are not read.
// The rounds (sampler.cu:630-652): round i takes the SIMD-group's (i + 1)-th pair, admitting only what comes after the
// last pick (kp, pi) in the selection order: sampler.cu:491's filter on keys, NaN's key 0 below every pick's and
// -inf's, so a NaN is never admitted and a -inf never beats the round's start. Each lane runs two chains, the even and
// the odd j, each in ascending id with a strict >, so each keeps its first admitted pair; take_first and warp_first
// then give every lane the SIMD-group's first. A round that finds nothing ends the rounds in every lane (the pick is
// the same in each), and sentinels fill the rest of the list. SIMD-group 0 then merges the KWARPS lists into dst.
template <uint KPER, uint KWARPS, bool PEN>
static inline void part_list(device const uint* l, uint nv, uint k, uint blk, device uint2* dst, threadgroup uint2* wl,
                             uint tid, uint lane, uint sg, threadgroup atomic_uint* bits = nullptr,
                             device const float* bias = nullptr, bool biased = false, device const int* hrow = nullptr,
                             uint hlen = 0u, float repeat = 1.0f, float freq = 0.0f, float present = 0.0f) {
    const uint span = 32u * KPER, block = KWARPS * span, blo = blk * block, lo = blo + sg * span;
    const bool use_bits = PEN && hlen > 0u;
    if (use_bits) {                                                                       // :597-610
        for (uint w = tid; w < block / 32u; w += KWARPS * 32u)
            atomic_store_explicit(bits + w, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < hlen; i += KWARPS * 32u) {                                 // bounded by hlen
            const int h = hrow[i];
            if (h >= 0 && uint(h) < nv && uint(h) >= blo && uint(h) - blo < block)
                atomic_fetch_or_explicit(bits + ((uint(h) - blo) >> 5), 1u << ((uint(h) - blo) & 31u),
                                         memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint s[KPER];                                                                         // :612-628
#pragma clang loop unroll(full)
    for (uint j = 0u; j < KPER; ++j) {
        const uint v = lo + 32u * j + lane;
        s[j] = v >= nv ? BS_NINF_BITS
                       : ((PEN && biased) ? as_type<uint>(ieee_add(as_type<float>(l[v]), bias[v])) : l[v]);
    }
    if (use_bits) {
#pragma clang loop unroll(full)
        for (uint j = 0u; j < KPER; ++j) {
            const uint v = lo + 32u * j + lane, b = v - blo;
            if (v < nv && (atomic_load_explicit(bits + (b >> 5), memory_order_relaxed) & (1u << (b & 31u))) != 0u)
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), repeat, freq, present);
        }
    }
    uint kp = okey(0x7f800000u);                                     // +inf and id -1: round 0 admits every logit
    int pi = -1;
    uint i = 0u;
    for (; i < k; ++i) {                                                                  // bounded: k <= 64 rounds
        uint b0 = BS_NINF_BITS, b1 = BS_NINF_BITS, i0 = nv, i1 = nv, k0 = BS_KEY_NINF, k1 = BS_KEY_NINF;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < KPER; j += 2u) {
            const uint v0 = lo + 32u * j + lane, v1 = v0 + 32u;
            const uint x0 = fnan(s[j]) ? 0u : okey(s[j]), x1 = fnan(s[j + 1u]) ? 0u : okey(s[j + 1u]);
            if ((x0 < kp || (x0 == kp && int(v0) > pi)) && x0 > k0) { b0 = s[j]; i0 = v0; k0 = x0; }
            if ((x1 < kp || (x1 == kp && int(v1) > pi)) && x1 > k1) { b1 = s[j + 1u]; i1 = v1; k1 = x1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(b0, i0);
        if (i0 >= nv) break;                                         // in every lane: nothing left in this slice
        if (lane == 0u) wl[sg * BS_SEL_MAX + i] = uint2(i0, b0);
        kp = okey(b0);
        pi = int(i0);
    }
    for (uint r = i + lane; r < k; r += 32u) wl[sg * BS_SEL_MAX + r] = uint2(nv, BS_NINF_BITS);   // the sentinels
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u) return;
    BsHeads h = merge_heads(wl, KWARPS, BS_SEL_MAX, nv, lane);                            // :654-660
    for (uint r = 0u; r < k; ++r) {                                                       // bounded: k rounds
        const uint2 c = merge_lists(h, wl, BS_SEL_MAX, k, nv, lane);
        if (lane == 0u) dst[r] = c;
    }
}

// ---- The sampled tail: sampled_tail_warp (sampler.cu:362-439) -------------------------------------------------------
//
// The chain after the top-k list, on one SIMD-group (docs/research/basalt-nvcc/sampler.md 2.4): top_p, then min_p on
// its survivors, the temperature, and one draw. sel_ids and sel_logit (the list's ids and value bits, k entries in
// the selection order) and ex (binary64 bits) are the caller's threadgroup arrays of BS_SEL_MAX; the chain's scalars
// come as their bits, the draw's seed and counter (the row's own: basalt's p.counter + t) as words. Every lane computes
// inv_t, the two maxima and the min_p cut from the same data; the lanes share the exps and the quotients (entries
// lane and lane + 32); lane 0 alone runs the ordered sums, the top_p scan and the draw, and broadcasts the sums and
// the cut. Each step is CUDA's IEEE operation with subnormals: the compares on bits (f_gt, f_lt, fmaxf_bits), the
// binary32 arithmetic through ieee_div, ieee_mul and ieee_add, the binary64 in software (basalt_f64.metal), exp(double)
// and logf as libdevice's (basalt_libdevice.metal). Returns, on lane 0: the pick; its probability's bits, (float)
// ex[pi], or 1.0 when nothing is kept (min_p above 1; the coupled merge's prob_out); and with probe, those of id q's
// probability, 0 outside the kept list (the probe's qprob[t]). The caller stores them.
struct BsDraw {
    int pick;
    uint prob, qprob;
};

// __shfl_sync of a double: its two words
static inline ulong broadcast_f64(ulong x) { return bits(simd_broadcast(lo(x), 0), simd_broadcast(hi(x), 0)); }

static inline BsDraw sampled_tail(threadgroup const int* sel_ids, threadgroup const uint* sel_logit, uint k,
                                  uint top_p, uint min_p, uint temperature, int min_keep, ulong seed, ulong counter,
                                  bool probe, int q, threadgroup ulong* ex, uint lane) {
    const float inv_t = f_gt(temperature, 0u) ? ieee_div(1.0f, as_type<float>(temperature)) : 0.0f;   // :374
    int n_keep = int(k);
    uint mx = sel_logit[0];
    for (uint i = 1u; i < k; ++i) mx = fmaxf_bits(mx, sel_logit[i]);                     // :376-377
    if (f_lt(top_p, 0x3f800000u)) {                                                       // :378-399
        const ulong dm = f64_from_f32(mx);
        for (uint i = lane; i < k; i += 32u) ex[i] = nv_exp(f64_sub(f64_from_f32(sel_logit[i]), dm));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        ulong sum = 0ul;
        if (lane == 0u)
            for (uint i = 0u; i < k; ++i) sum = f64_add(sum, ex[i]);
        sum = broadcast_f64(sum);
        simdgroup_barrier(mem_flags::mem_threadgroup);                // lane 0 has read every ex before it changes
        for (uint i = lane; i < k; i += 32u) ex[i] = f64_div(ex[i], sum);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        int cut = int(k);
        if (lane == 0u) {
            const ulong tp = f64_from_f32(top_p);
            ulong cum = 0ul;
            for (uint i = 0u; i < k; ++i) {
                cum = f64_add(cum, ex[i]);
                if (f64_ge(cum, tp)) {                                // ordered: a NaN never cuts
                    cut = int(i) + 1;
                    break;
                }
            }
        }
        cut = simd_broadcast(cut, 0);
        if (cut < min_keep) cut = min_keep < int(k) ? min_keep : int(k);
        n_keep = cut;
        simdgroup_barrier(mem_flags::mem_threadgroup);                // ex is written again below
    }
    if (f_gt(min_p, 0u)) {                                                                // :402-406
        const uint th = as_type<uint>(ieee_add(as_type<float>(sel_logit[0]), nv_logf(as_type<float>(min_p))));
        for (int i = 0; i < n_keep; ++i)
            if (f_lt(sel_logit[i], th)) {
                n_keep = i;
                break;
            }
    }
    uint smx = as_type<uint>(ieee_mul(as_type<float>(sel_logit[0]), inv_t));              // :408-409
    for (int i = 1; i < n_keep; ++i)
        smx = fmaxf_bits(smx, as_type<uint>(ieee_mul(as_type<float>(sel_logit[i]), inv_t)));
    const ulong ds = f64_from_f32(smx);
    for (int i = int(lane); i < n_keep; i += 32)                                          // :410
        ex[i] = nv_exp(f64_sub(f64_from_f32(as_type<uint>(ieee_mul(as_type<float>(sel_logit[i]), inv_t))), ds));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    ulong sum = 0ul;
    if (lane == 0u)
        for (int i = 0; i < n_keep; ++i) sum = f64_add(sum, ex[i]);
    sum = broadcast_f64(sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(lane); i < n_keep; i += 32) ex[i] = f64_div(ex[i], sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    BsDraw d = {0, 0u, 0u};
    if (lane == 0u) {                                                                     // :419-437
        const ulong u = f64_from_f32(as_type<uint>(philox_uniform(seed, counter)));
        ulong cum = 0ul;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        d.pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            cum = f64_add(cum, ex[i]);
            if (f64_lt(u, cum)) {                                         // ordered: a NaN never picks
                d.pick = sel_ids[i];
                pi = i;
                break;
            }
        }
        d.prob = n_keep > 0 ? f32_from_f64(ex[pi]) : 0x3f800000u;
        if (probe) {
            for (int i = 0; i < n_keep; ++i)
                if (sel_ids[i] == q) {
                    d.qprob = f32_from_f64(ex[i]);
                    break;
                }
            if (n_keep == 0 && sel_ids[0] == q) d.qprob = 0x3f800000u;
        }
    }
    return d;
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

constant uint BS_TOP_PROB_THREADS = 1024u;                             // kAmThreads: the probability's geometry

// sampler_greedy_top_prob_cluster_kernel (sampler.cu:249-325) after the cluster argmax bs_greedy_partial ran: CTA 0's
// merge of the row's 8 pairs (bs_greedy_merge's), then the pick's softmax probability with row_top_prob_kernel's
// arithmetic in its order (verify_kernels.cu:580-596), then the subset map. The partial sums are part of the bits, so
// the threadgroup is exactly 1,024 threads (a dispatch of any other width returns at once and writes nothing): thread
// i sums logits i, i + 1,024, ... in ascending order, then the XOR butterfly at 16, 8, 4, 2, 1, then thread 0 sums
// the 32 SIMD-groups' sums in order from +0. __expf is expf_approx (invariant 5); every add and subtraction around it
// is CUDA's IEEE one, subnormals kept, so the butterfly is ieee_add's, not basalt_fp.metal's plain-add warp_sum.
// probs = 1.0f / tot, correctly rounded: tot is at least 1 (the pick's own term, exp2(0)) or NaN. out is the pick, or
// map[pick] with flags MAP (an empty row's pick 0 maps to map[0]).
kernel void bs_greedy_top_prob(device const uint* logits [[buffer(0)]], device int* out [[buffer(5)]],
                               device float* probs [[buffer(6)]], constant BsSampleParams& p [[buffer(7)]],
                               device const uint2* part [[buffer(8)]], device const int* id_map [[buffer(11)]],
                               device const StepState* st [[buffer(15)]],
                               uint t [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                               uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wsum[BS_TOP_PROB_THREADS / 32u];
    threadgroup uint pick;
    if (st->done || t >= st->t_this_step || tpg != BS_TOP_PROB_THREADS) return;
    const uint nv = p.vocab;
    if (sg == 0u) {                                                                       // :299-305
        const uint2 c = lane < BS_AM_CTAS ? part[ulong(t) * BS_AM_CTAS + lane] : uint2(BS_KEY_NINF, nv);
        uint bk = c.x, bi = c.y;
        am_tree(bk, bi);
        if (lane == 0u) pick = bi < nv ? bi : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                                      // :306
    const uint id = pick;
    device const uint* l = logits + ulong(t) * nv;
    const float m = as_type<float>(l[id]);                                                // :309
    float s = 0.0f;
    for (uint i = tid; i < nv; i += BS_TOP_PROB_THREADS)                                  // :311, bounded by nv
        s = ieee_add(s, expf_approx(ieee_add(as_type<float>(l[i]), neg(m))));
    for (ushort o = 16; o > 0; o >>= 1) s = ieee_add(s, shfl_xor(s, o));                 // :312
    if (lane == 0u) wsum[sg] = s;                                                         // :313
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {                                                                      // :315-320
        float tot = 0.0f;
        for (uint w = 0u; w < BS_TOP_PROB_THREADS / 32u; ++w) tot = ieee_add(tot, wsum[w]);
        probs[t] = 1.0f / tot;
        out[t] = (p.flags & BS_MAP) != 0u ? id_map[id] : int(id);
    }
}

constant uint BS_BLOCK = 4096u;                                        // kSplitBlockSpan: logits per split block
constant uint BS_SPLIT_WARPS = 4u;                                     // kSplitWarps: bs_split_part's SIMD-groups

// sampler_split_part_kernel (sampler.cu:582-661), mode 2: block b's top-k list of row t (grid (n_blocks, rows) of
// basalt's 128 threads, four SIMD-groups of 32 logits per lane), k = sampled_k(top_k, vocab), after the logit bias
// (flags BIAS) and the penalties of the row's window (bs_greedy_one_block's: the last min(penalty_last_n, hist_len)
// slots of its history row), into cand at (t * n_blocks + b) * k as (id, value bits), padded with the sentinel
// (vocab, -inf). A dispatch of any other width returns at once (the SIMD-groups' slices and wl's four lists are the
// geometry's), and so does a block at or past n_blocks, whose entries would be the next row's.
kernel void bs_split_part(device const uint* logits [[buffer(0)]], device const float* bias [[buffer(3)]],
                          device const int* history [[buffer(4)]], constant BsSampleParams& p [[buffer(7)]],
                          device uint2* cand [[buffer(9)]], device const StepState* st [[buffer(15)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 tg [[threads_per_threadgroup]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint bits[BS_BLOCK / 32u];
    threadgroup uint2 wl[BS_SPLIT_WARPS * BS_SEL_MAX];
    const uint b = g.x, t = g.y;
    if (st->done || t >= st->t_this_step || p.mode != 2u || b >= p.n_blocks || tg.x * tg.y != BS_SPLIT_WARPS * 32u)
        return;
    const uint nv = p.vocab, k = sampled_k(p.top_k, nv);
    const uint hlen = uint(clamp(p.penalty_last_n, 0, int(p.hist_len)));                 // :590-596
    device const int* hrow = history + ulong(t) * p.hist_len + (p.hist_len - hlen);      // the window is the tail
    part_list<32u, BS_SPLIT_WARPS, true>(logits + ulong(t) * nv, nv, k, b, cand + (ulong(t) * p.n_blocks + b) * k, wl,
                                         tid, lane, sg, bits, bias, (p.flags & BS_BIAS) != 0u, hrow, hlen,
                                         p.penalty_repeat, p.penalty_freq, p.penalty_present);
}

constant uint BS_SPLIT_MAX_BLOCKS = 64u;                               // kSplitMaxBlocks: the lists one merge holds
constant uint BS_PENDING = sizeof(StepState::pending_tokens) / sizeof(int);   // the probe's queries, t + 1 below it

// sampler_split_merge_kernel (sampler.cu:663-682), mode 2: one SIMD-group per row (basalt's 32 threads) merges the
// row's n_blocks block lists of k from cand, in place of basalt's copy into 32 KiB of shared memory (the same entries
// read: the same list), into the row's top-k list, the sentinel's id vocab stored as 0, then runs the tail. The draw's
// counter is ctr_base (two words, the request's) + position + t, its seed StepState's rng_hi:rng_lo. With flags PROBE,
// row t < nq[0] also writes the probability of its query pending_tokens[t + 1] to qprob[t] (sampler.cu:430-436); no
// other row or request writes qprob, and nq is read only then. A dispatch of another width returns at once (two
// SIMD-groups would share the tail's arrays), as does a record of more blocks than the merge's 64 lists.
kernel void bs_split_merge(device const uint* ctr_base [[buffer(1)]], device const uint* nq [[buffer(2)]],
                           device int* out [[buffer(5)]], device uint* qprob [[buffer(6)]],
                           constant BsSampleParams& p [[buffer(7)]], device const uint2* cand [[buffer(9)]],
                           device const StepState* st [[buffer(15)]],
                           uint t [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int sel_ids[BS_SEL_MAX];
    threadgroup uint sel_logit[BS_SEL_MAX];
    threadgroup ulong ex[BS_SEL_MAX];
    if (st->done || t >= st->t_this_step || p.mode != 2u || tpg != 32u || p.n_blocks > BS_SPLIT_MAX_BLOCKS) return;
    const uint nv = p.vocab, k = sampled_k(p.top_k, nv);
    device const uint2* lists = cand + ulong(t) * p.n_blocks * k;                         // :674-679
    BsHeads h = merge_heads(lists, p.n_blocks, k, nv, lane);
    for (uint i = 0u; i < k; ++i) {                                                       // bounded: k rounds
        const uint2 c = merge_lists(h, lists, k, k, nv, lane);
        if (lane == 0u) {
            sel_ids[i] = c.x < nv ? int(c.x) : 0;
            sel_logit[i] = c.y;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const bool probe = (p.flags & BS_PROBE) != 0u && t < nq[0] && t + 1u < BS_PENDING;
    const ulong seed = (ulong(st->rng_hi) << 32) | st->rng_lo;
    const ulong counter = ((ulong(ctr_base[1]) << 32) | ctr_base[0]) + ulong(st->position) + ulong(t);
    const BsDraw d = sampled_tail(sel_ids, sel_logit, k, as_type<uint>(p.top_p), as_type<uint>(p.min_p),
                                  as_type<uint>(p.temperature), p.min_keep, seed, counter, probe,
                                  probe ? st->pending_tokens[t + 1u] : 0, ex, lane);
    if (lane == 0u) {
        out[t] = d.pick;
        if (probe) qprob[t] = d.qprob;
    }
}
#endif
