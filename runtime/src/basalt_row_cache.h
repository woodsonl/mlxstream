// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/ngram/ple_reader.cpp:30-71, :544-545 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the set-associative row cache, its mix,
// its round-robin replacement and its two size accessors.
//
// The rows this process has already fetched from the PLE table, kept so a row that recurs within one prompt is
// not read from the device twice. Eight ways per set, round robin inside a set, bounded by construction.
//
// THE CACHE IS BIT-NEUTRAL. It stores and returns the raw on-disk row bytes verbatim, and the dequantization
// happens afterwards and identically either way. Changing the mix, the way count or the replacement policy moves
// the hit rate, the device traffic and the latency, and can never move a produced value. That is why this file
// has a different standard of exactness from the hash and the decode: it is transcribed as written because the
// port is 1:1, and the tests assert the structure rather than an output oracle, since there is no output to
// oracle.
//
// WHAT A PORT CAN STILL GET WRONG, in ways that are invisible without a structural test:
//
//   * `mix` multiplies a 32-bit row index by a 64-bit constant. The usual arithmetic conversions promote the
//     row, so it is a full 64-bit multiply truncated mod 2^64, then `x ^ (x >> 29)`. Doing the multiply in 32
//     bits produces a different, still-valid set index and a different hit rate.
//   * The set index is `mix(row) % sets`, never `& (sets - 1)`. At the production capacity of 2^20 rows,
//     `sets` is 2^17 and the two agree, which is exactly what makes the substitution look safe; they diverge
//     for any capacity that is not a power of two times eight.
//   * Replacement is per-set ROUND ROBIN, not clock and not least-recently-used. basalt's own header says
//     clock; its code says round robin, and the code is what this ports (`ple_reader.cpp:34` states it
//     correctly). A hit updates nothing: `insert` early-returns on a row already present, so the replacement
//     pointer does not move and the entry is neither refreshed nor promoted.
//   * `used` counts first fills of an empty way and never decreases. It is a fill counter, and it coincides
//     with the live-entry count only because entries are never removed.
#pragma once

#include <cstdint>
#include <vector>

namespace monolith::basalt_cpu {

/// Ways per set, and the vacancy marker (`ple_reader.cpp:31-32`). `EMPTY` can never collide with a live key,
/// because the caller filters a row at or past the table's row count before any lookup and the table has
/// 320,001,536 rows.
inline constexpr uint32_t ROW_CACHE_WAYS = 8;
inline constexpr uint32_t ROW_CACHE_EMPTY = 0xFFFFFFFFu;

class RowCache {
  public:
    /// `rows` is the requested capacity in rows; the usable capacity rounds DOWN to a multiple of the way count,
    /// because `sets` is an integer division. Fewer than eight rows gives zero sets, which disables the cache:
    /// both lookup and insert then do nothing, and zero is basalt's documented "off" value.
    void init(uint64_t rows, uint32_t row_bytes);

    /// The row's bytes, or null when it is absent or the cache is off.
    const uint8_t* find(uint32_t row) const;

    /// Store a row. A row already present is left alone, pointer included.
    void insert(uint32_t row, const uint8_t* bytes);

    uint64_t capacity() const { return sets_ * ROW_CACHE_WAYS; }
    uint64_t size() const { return used_; }

    /// The set index. `find` and `insert` call this same function rather than repeating the arithmetic, so a
    /// test that holds this holds the path they actually take; two copies would let one drift from the other
    /// and leave the drift invisible at any power-of-two capacity.
    static uint64_t mix(uint32_t r);
    uint64_t set_of(uint32_t row) const { return sets_ == 0 ? 0 : mix(row) % sets_; }
    uint64_t sets() const { return sets_; }
    uint32_t row_bytes() const { return rb_; }

  private:
    uint64_t sets_ = 0;
    uint32_t rb_ = 0;
    std::vector<uint32_t> keys_;      // sets * WAYS
    std::vector<uint8_t> data_;       // sets * WAYS * rb
    std::vector<uint8_t> next_;       // one replacement pointer per set
    uint64_t used_ = 0;
};

}  // namespace monolith::basalt_cpu
