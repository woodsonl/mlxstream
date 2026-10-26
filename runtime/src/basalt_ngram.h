// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) include/basalt/kernels/ngram.hpp and src/kernels/ngram.cpp @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the host n-gram hash, its constants, and
// the window build with its EOS cut.
//
// Sixteen row indices per token, from the last three token ids, by 64-bit multiply and xor. basalt computes this
// on the host rather than as a tensor op for the reason it states: "ggml has no int64 and no xor"
// (`ngram.hpp:5-7`). The same holds here, so it stays on the host and the GPU gathers the rows it names.
//
// WHAT A PORT MUST NOT SIMPLIFY. Each of these is a place where a reasonable-looking change is a different
// function, and the oracle vectors are the only thing that catches it:
//
//   * `%` by the vocabulary size, never `& (size - 1)`. The sizes are 20000003, 20000023, 20000033 and so on:
//     none is a power of two (`ngram.hpp:15-16`).
//   * XOR, never sum. Two equal tokens with equal multipliers cancel to zero under xor and double under a sum
//     (`ngram.hpp:14`).
//   * `prev` is OLDEST FIRST, so predecessor `s` back is entry `n_prev - s` and `s = 1` reads the NEWEST. Reading
//     `s - 1` walks the window backwards, still produces indices in range, and so no range check can catch it
//     (`ngram.cpp:68-70`).
//   * The cut is evaluated BEFORE the value is stored, so the position whose predecessor was EOS is itself EOS.
//     Storing first and cutting after leaves one token of history too much (`ngram.cpp:72-74`).
//   * A missing predecessor is tested `t < 0`, not `t == -1`: token id 0 is a real token (`ngram.hpp:18`).
//   * The token's OWN eos does not cut its own context; only the predecessors are examined (`ngram.hpp:19`).
//
// The vocabulary sizes and their offsets are transcribed from the artifact's GGUF metadata and written out rather
// than derived, which is basalt's own choice and its reason: `head_offsets` is in the metadata as its own array,
// and deriving one from the other would hide a mismatch between them instead of surfacing it (`ngram.cpp:32-34`).
// The test checks the two agree.
#pragma once

#include <cstdint>

namespace monolith::basalt_cpu {

/// The PLE geometry (`ngram.hpp:30-41`). Sixteen heads of 160 make the 2560-wide vector exactly, so the sixteen
/// gathered rows ARE the embedding (`ngram.hpp:43-44`).
inline constexpr int NG_N_EMBD = 2560;
inline constexpr int PLE_HEAD_DIM = 160;
inline constexpr int NGRAM_SIZE = 3;
inline constexpr int HEADS_PER_NGRAM = 8;
inline constexpr int PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM;   // 16
inline constexpr int32_t PLE_EOS_TOKEN_ID = 248044;
inline constexpr int32_t TOKEN_NULL = -1;                                // llama's LLAMA_TOKEN_NULL

/// The hash constants, all three arrays 64-bit: the modulus and the add below are 64-bit, not 32
/// (`ngram.hpp:62-66`).
struct PleConsts {
    uint64_t mult[NGRAM_SIZE] = {23703573157769ull, 20109073645365ull, 8052911324071ull};
    uint64_t vocab[PLE_N_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    uint64_t offset[PLE_N_HEADS] = {
        0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
};

/// Sixteen row indices per token.
///
///   tokens  (n_tokens,)      the batch's own tokens
///   prev    (n_tokens, 2)    predecessors OLDEST FIRST, TOKEN_NULL where there is none
///   out     (n_tokens, 16)   row indices into the PLE table, token-major and head-fastest
///
/// Heads 0 to 7 are the bigram heads and 8 to 15 the trigram heads; all eight of one order share one mixed
/// value, so this computes two 64-bit mixes and sixteen modulo-and-add pairs per token (`ngram.cpp:78-85`).
void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out);

}  // namespace monolith::basalt_cpu
