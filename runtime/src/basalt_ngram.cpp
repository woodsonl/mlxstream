// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/ngram.cpp:52-87 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): `ngram_mixed` and `ngram_rows`, with the
// window build, the EOS cut and the per-head modulo as written.

#include "basalt_ngram.h"

namespace monolith::basalt_cpu {

namespace {

/// The mix of one window (`ngram.cpp:52-59`). The first term is an ASSIGNMENT and the rest are xored in: it is
/// not a reduction over an identity element, which is how the source writes it (`ngram.cpp:54-55`).
///
/// `ctx` is int64, and `(uint64_t) ctx[j]` is a two's-complement reinterpretation, so a negative `ctx[0]` becomes
/// 2^64 + value. `ctx[0]` is the token itself and is never sanitized, so a negative token id is a legal input a
/// port has to wrap the same way. Every product is a 64 by 64 multiply truncated mod 2^64.
uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    uint64_t mixed = (uint64_t) ctx[0] * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];
    return mixed;
}

}  // namespace

void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out) {
    const int n_prev = NGRAM_SIZE - 1;                        // 2
    for (int i = 0; i < n_tokens; ++i) {
        int64_t ctx[NGRAM_SIZE];
        ctx[0] = tokens[i];                                   // sign-extended, and never compared against EOS
        bool cut = false;
        for (int s = 1; s < NGRAM_SIZE; ++s) {
            // `prev` is oldest first, so entry (n_prev - s): s = 1 reads the NEWEST predecessor.
            const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];
            // The cut is taken BEFORE the store, so the position whose predecessor was EOS is itself EOS, and
            // once set it stays set: every older position is forced to EOS too, never to TOKEN_NULL.
            cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
            ctx[s] = cut ? PLE_EOS_TOKEN_ID : t;
        }

        // Two mixes per token, one per order, each shared by its eight heads. The trigram value is computed over
        // the same ctx rather than derived from the bigram value.
        for (int n = 2; n <= NGRAM_SIZE; ++n) {
            const uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * HEADS_PER_NGRAM;
            for (int g = 0; g < HEADS_PER_NGRAM; ++g) {
                const int h = base + g;
                // Unsigned 64-bit modulo and a 64-bit add; the truncation to uint32 is last and is never lossy,
                // because the sum of the vocabulary sizes is 320001446 and so the largest index is 320001445.
                out[i * PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

}  // namespace monolith::basalt_cpu
