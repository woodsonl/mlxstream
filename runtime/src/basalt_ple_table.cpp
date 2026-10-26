// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/ngram.cpp:95-108, :456 and
// include/basalt/kernels/f16_bits.hpp:69-94 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT;
// third_party/NOTICE): `iq4nl_dequant_row`, `f32_from_f16`, and the head-slowest row flatten.

#include "basalt_ple_table.h"

#include <cstring>

namespace monolith::basalt_cpu {

float f32_from_f16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t out;
    if (ex == 0) {
        if (man == 0) {
            out = sign;                                   // signed zero, sign preserved
        } else {
            // A subnormal is normalised rather than flushed: shift the mantissa up until the implicit bit
            // appears, and pay for each shift in the exponent. Skipping this branch maps 0x0001 to 0.0 instead
            // of 2^-24.
            uint32_t m = man;
            int s = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++s;
            }
            out = sign | ((uint32_t) (113 - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7F800000u | (man << 13);           // inf and NaN, mantissa carried through
    } else {
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);                      // raw little-endian fp16 bits straight off the file
        const float d = f32_from_f16(dbits);
        const uint8_t* qs = blk + 2;
        // SPLIT HALVES: qs[j] holds elements j and j + 16, not 2j and 2j + 1. Low nibble first.
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + j] = d * (float) kIq4Nl[qs[j] & 0x0F];
            out160[b * 32 + j + 16] = d * (float) kIq4Nl[qs[j] >> 4];
        }
    }
}

void ple_rows_to_embd(const uint8_t* rows16, float* out2560) {
    // Head-slowest: row h's 160 values occupy [h*160, (h+1)*160), which is what makes the result a plain
    // n_embd-wide vector rather than a transpose of one.
    for (int h = 0; h < PLE_N_HEADS; ++h)
        iq4nl_dequant_row(rows16 + (size_t) h * PLE_ROW_BYTES, out2560 + (size_t) h * PLE_HEAD_DIM);
}

}  // namespace monolith::basalt_cpu
