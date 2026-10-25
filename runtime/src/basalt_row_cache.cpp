// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/ngram/ple_reader.cpp:36-57, :544-545 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): `RowCache::init`, `mix`, `find` and
// `insert`, with the replacement as written.

#include "basalt_row_cache.h"

#include <cstring>

namespace monolith::basalt_cpu {

uint64_t RowCache::mix(uint32_t r) {
    // `r` promotes to 64 bits against the 64-bit constant, so this is a full multiply truncated mod 2^64.
    const uint64_t x = (uint64_t) r * 0x9E3779B97F4A7C15ull;
    return x ^ (x >> 29);
}

void RowCache::init(uint64_t rows, uint32_t row_bytes) {
    sets_ = rows / ROW_CACHE_WAYS;                 // truncating: the usable capacity rounds down
    rb_ = row_bytes;
    keys_.assign((size_t) (sets_ * ROW_CACHE_WAYS), ROW_CACHE_EMPTY);
    data_.assign((size_t) (sets_ * ROW_CACHE_WAYS * rb_), 0);
    next_.assign((size_t) sets_, 0);
    used_ = 0;
}

const uint8_t* RowCache::find(uint32_t row) const {
    if (sets_ == 0) return nullptr;
    const uint64_t s = set_of(row);                // modulo, never a mask: sets need not be a power of two
    for (uint32_t w = 0; w < ROW_CACHE_WAYS; ++w)
        if (keys_[(size_t) (s * ROW_CACHE_WAYS + w)] == row)
            return &data_[(size_t) ((s * ROW_CACHE_WAYS + w) * rb_)];
    return nullptr;
}

void RowCache::insert(uint32_t row, const uint8_t* bytes) {
    // A row already present is left exactly as it is, and the replacement pointer does not move: a hit updates
    // nothing, which is what makes this round robin rather than least-recently-used.
    if (sets_ == 0 || find(row) != nullptr) return;
    const uint64_t s = set_of(row);
    const uint32_t w = next_[(size_t) s];
    next_[(size_t) s] = (uint8_t) ((w + 1) % ROW_CACHE_WAYS);   // advances on every insert, occupied or not
    if (keys_[(size_t) (s * ROW_CACHE_WAYS + w)] == ROW_CACHE_EMPTY) ++used_;
    keys_[(size_t) (s * ROW_CACHE_WAYS + w)] = row;
    std::memcpy(&data_[(size_t) ((s * ROW_CACHE_WAYS + w) * rb_)], bytes, rb_);
}

}  // namespace monolith::basalt_cpu
