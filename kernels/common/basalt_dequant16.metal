// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/dequant_bf16.cu @ b4c63b55754637f4c72ad53a0f791d425f71eb36
// (MIT; third_party/NOTICE), whose arithmetic transcribes ggml-org/llama.cpp ggml/src/ggml-quants.c
// (dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs, dequant_bf16.cu:1-4) and
// ggml/src/ggml-common.h (kvalues_iq4nl) @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's group dequantizer, dequant_kernel<TYPE, T, false> with group32<TYPE> (:168-188, :40-166), as
// basalt_dequant_g32: one thread per 32-value group of a matrix's GGUF blocks, stored as fp32 (DQ_OUT 0, dequant_f32),
// fp16 bits (1, dequant_f16) or BF16 bits (2, dequant_bf16). DQ_TYPE: the ggml type id. Grid, bindings and params:
// monolith/kernels.py's basalt_dequant16_* builders. Needs basalt_fp.metal ahead of it.
//
// Numerics (docs/research/basalt-nvcc/dequant_bf16.md): for every finite fp16 scale, each product of a scale and the
// block's integers is exact and none is an fp32 subnormal [M] (the smallest nonzero value is 2^-24), so this GPU's
// flush never acts; the only fp32 rounding is Q4_K's and Q5_K's fma(d1, q, -m1); then one rounding into the output.
#pragma STDC FP_CONTRACT OFF

struct DequantG32Params { long row_bytes; long row0; long rows; long groups_per_row; };    // :171-172, in order

// h2f (:17-19): the little-endian fp16 field at p, any byte address, to fp32 (exact)
static inline float dq_h2f(device const uchar* p) { return h2f(ushort(p[0] | (p[1] << 8))); }

constant char kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};    // :33

// put (:28-31) by DQ_OUT. Every NaN here is a product on a NaN or infinite scale, sm_120's 0x7fffffff: the fp32 store
// takes cuda_nan, the BF16 store f2bf of it (0x7fff); __float2half_rn's NaN is cuda_nan_f16's 0x7fff.
#if DQ_OUT == 0
typedef float DQ_T;
static inline void dq_put(device DQ_T* o, int i, float v) { o[i] = cuda_nan(v); }
#elif DQ_OUT == 1
typedef ushort DQ_T;                                                // basalt's H16
static inline void dq_put(device DQ_T* o, int i, float v) { o[i] = cuda_nan_f16(f16_bits(v)); }
#elif DQ_OUT == 2
typedef ushort DQ_T;                                                // BF16 bits, basalt's uint16_t
static inline void dq_put(device DQ_T* o, int i, float v) { o[i] = f2bf(cuda_nan(v)); }
#else
#error "basalt_dequant16: unsupported DQ_OUT"
#endif

// group32<DQ_TYPE> (:40-166): the 32 values of group gi_in_row of the row whose blocks start at row_blocks, to out
static inline void dq_group32(device const uchar* row_blocks, int gi_in_row, device DQ_T* out) {
#if DQ_TYPE == 42                                                   // Q2_0: 64 per block of 18 B (:43-52)
    device const uchar* b = row_blocks + size_t(gi_in_row / 2) * 18;
    const float d = dq_h2f(b);
    const int e0 = (gi_in_row % 2) * 32;
    for (int j = 0; j < 32; ++j) {
        const int e = e0 + j;
        const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
        dq_put(out, j, float(q - 1) * d);
    }
#elif DQ_TYPE == 2                                                  // Q4_0 (:53-59)
    device const uchar* b = row_blocks + size_t(gi_in_row) * 18;
    const float d = dq_h2f(b);
    for (int j = 0; j < 16; ++j) {
        dq_put(out, j, float((b[2 + j] & 0x0F) - 8) * d);
        dq_put(out, j + 16, float((b[2 + j] >> 4) - 8) * d);
    }
#elif DQ_TYPE == 6                                                  // Q5_0 (:60-69)
    device const uchar* b = row_blocks + size_t(gi_in_row) * 22;
    const float d = dq_h2f(b);
    const uint qh = uint(b[2]) | (uint(b[3]) << 8) | (uint(b[4]) << 16) | (uint(b[5]) << 24);
    for (int j = 0; j < 16; ++j) {
        const int xh0 = int(((qh >> j) << 4) & 0x10u);
        const int xh1 = int((qh >> (j + 12)) & 0x10u);
        dq_put(out, j, float(((b[6 + j] & 0x0F) | xh0) - 16) * d);
        dq_put(out, j + 16, float(((b[6 + j] >> 4) | xh1) - 16) * d);
    }
#elif DQ_TYPE == 8                                                  // Q8_0 (:81-84)
    device const uchar* b = row_blocks + size_t(gi_in_row) * 34;
    const float d = dq_h2f(b);
    for (int j = 0; j < 32; ++j) dq_put(out, j, float(char(b[2 + j])) * d);
#elif DQ_TYPE == 20                                                 // IQ4_NL (:85-91)
    device const uchar* b = row_blocks + size_t(gi_in_row) * 18;
    const float d = dq_h2f(b);
    for (int j = 0; j < 16; ++j) {
        dq_put(out, j, d * float(kv_iq4nl[b[2 + j] & 0xf]));
        dq_put(out, j + 16, d * float(kv_iq4nl[b[2 + j] >> 4]));
    }
#else
// Q5_1 (7, :70-80) and the iq_only types (:251): no tensor of this pack has them; basalt_dequant16_macros refuses them
#error "basalt_dequant16: unsupported DQ_TYPE"
#endif
}

// dequant_kernel<DQ_TYPE, DQ_T, false> (:170-188): thread g of rows * groups_per_row (256-thread threadgroups,
// :221-222) dequantizes group g % groups_per_row of row row0 + g / groups_per_row into out[32 g .. 32 g + 31]. The VEC
// form (:177-184) stores the same 32 values as 16-byte copies; it is not ported.
kernel void basalt_dequant_g32(device const uchar* blocks [[buffer(0)]], device DQ_T* out [[buffer(1)]],
                               constant DequantG32Params& p [[buffer(2)]],
                               uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    const long g = long(tg) * 256 + long(tid);                         // :173
    if (g >= p.rows * p.groups_per_row) return;                        // :174
    const long r = g / p.groups_per_row, gi = g % p.groups_per_row;    // :175
    device const uchar* row = blocks + (p.row0 + r) * p.row_bytes;     // :176
    dq_group32(row, int(gi), out + g * 32);                            // :186
}
