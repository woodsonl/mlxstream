// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) include/basalt/kernels/ngram.hpp:43-49, :86-98, :170-173,
// include/basalt/kernels/f16_bits.hpp:69-94 and src/kernels/ngram.cpp:90-108, :456 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the IQ4_NL row decode, its codebook, the
// exact fp16 widening, and the head-slowest flatten of sixteen rows into one embedding.
//
// The PLE table is `per_layer_token_embd.weight`, [160, 320001536] of IQ4_NL, and basalt reads it straight from
// the source file's GGUF section rather than from its dense weight pool, because it is 28.8 GB. ne0 = 160 is the
// fast axis, so one row is 160 contiguous elements: 5 blocks of 32 at 18 bytes, which is 90 bytes. This header
// holds the part that turns those bytes into floats; reading them off the device is `basalt_ple_reader.h`.
//
// THE FOUR TRAPS, each of which produces a plausible embedding of the wrong values:
//
//   * THE NIBBLE ORDER IS SPLIT-HALF, NOT INTERLEAVED. Byte `j` of a block carries elements `j` and `j + 16`,
//     never `2j` and `2j + 1`, which is the natural reading (`ngram.hpp:94-98`).
//   * THERE IS NO -8 IN THIS CODEBOOK. Some releases of `dequantize_row_iq4_nl` subtract 8 from the codebook
//     value; at the commit this ports they do not, and the difference would shift every one of the table's
//     51.2e9 elements (`ngram.hpp:86-88`).
//   * THE FP16 WIDENING IS EXACT AND KEEPS SUBNORMALS. Every one of the 65,536 inputs maps to a value fp32
//     represents exactly, subnormals included, so the `ex == 0` branch cannot be skipped. Flushing them maps
//     0x0001 to 0.0 instead of 2^-24, a 100 percent relative error on any scale that lands there
//     (`f16_bits.hpp:67-68`, `:74-77`). This is why the conversion is written out rather than left to a
//     hardware `half` convert, which on this platform would flush the input subnormal when FZ16 is set.
//   * THE FLATTEN IS HEAD-SLOWEST. Row h's 160 values occupy `out[h*160, (h+1)*160)`. The rival placement, with
//     element (d, h) at `d*16 + h`, is a real transpose rather than a reshape (`ngram.hpp:170-173`).
//
// There is no accumulation anywhere in the decode: every output element is one fp32 multiply of two exactly
// representable operands, so no contraction or fast-math setting can move a bit.
#pragma once

#include <cstdint>

#include "basalt_ngram.h"

namespace monolith::basalt_cpu {

/// The table's shape (`ngram.hpp:45-49`). A row is 90 bytes for IQ4_NL; the two secondary formats are listed
/// because the reader's row stride is a run-time value and these are the strides it accepts.
inline constexpr uint64_t PLE_TABLE_ROWS = 320001536ull;
inline constexpr int PLE_ROW_BYTES = (PLE_HEAD_DIM / 32) * 18;        // 90
inline constexpr int PLE_ROW_BYTES_Q5_0 = (PLE_HEAD_DIM / 32) * 22;   // 110
inline constexpr int PLE_ROW_BYTES_Q8_0 = (PLE_HEAD_DIM / 32) * 34;   // 170
inline constexpr int PLE_ROW_BYTES_MAX = PLE_ROW_BYTES_Q8_0;          // 170

/// IQ4_NL's codebook at this commit, with no -8 (`ngram.cpp:90`).
inline constexpr int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1,    13,   25,  38,  53,  69,  89,  113};

/// Exact fp16 bits to fp32 (`f16_bits.hpp:69-94`). No rounding and no exceptions: subnormals are normalised,
/// never flushed, and inf and NaN carry through with the mantissa shifted left 13.
float f32_from_f16(uint16_t h);

/// One 90-byte IQ4_NL row to 160 floats. `value = codebook[code] * scale`, with no offset term, because IQ4_NL
/// has no min (`ngram.hpp:91-93`).
void iq4nl_dequant_row(const uint8_t* row, float* out160);

/// Sixteen decoded rows flattened head-slowest into one 2560-wide embedding. `rows16` are the row bytes already
/// read, each `PLE_ROW_BYTES` long and laid out back to back.
void ple_rows_to_embd(const uint8_t* rows16, float* out2560);

}  // namespace monolith::basalt_cpu
