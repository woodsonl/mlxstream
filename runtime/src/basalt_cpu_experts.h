// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) include/basalt/kernels/cpu/pool.hpp:54-63 and :156,
// include/basalt/kernels/cpu/native_expert.hpp:22-30 and src/kernels/cpu/pool.cpp:415-451, :543-572 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the multi-token expert job, the layer
// geometry that travels beside it, and the two phases the pool runs over them.
//
// What the CPU expert tier computes for one layer, given the experts the GPU cache came up short on. Two parallel
// phases with one serial step between them:
//
//   phase 1, gate and up: for each expert, rows [0, FF) on the Q2_0 row kernel twice, once at the blob's base for
//            the gate rows and once at up_off for the up rows, then the SwiGLU of the two into ff[t][r];
//   between:  the host thread alone re-quantizes each token's whole 640-wide ff vector into a2[t];
//   phase 2, down: for each expert, rows [0, H) of the down matrix on the same kernel, reading a2 and writing the
//            job's own out[t] rows.
//
// THE BITWISE PROMISE. Every token's output is bitwise the single-token job's, whatever the thread count, the
// placement or the order jobs are claimed in. Four independent facts make that true, and a port has to keep all
// four:
//   1. Per token inside the kernel: the accumulators are per token and the block loop order does not depend on how
//      many tokens a job carries, so a row's value for a token is the same for every grouping of tokens.
//   2. Per row across tasks: task i owns the global row range [rows * i / tasks, rows * (i + 1) / tasks) in int64
//      arithmetic, cut at expert boundaries, so every (expert, row) pair is computed whole by exactly one task.
//   3. The intermediate is quantized serially from the whole vector: the re-quantization between the phases runs
//      on the host over all 640 values at once, so the 20 chunk scales do not depend on how phase 1 was split.
//   4. The activation is shared, not recomputed: one ActQ per token per layer, built by the caller and pointed at
//      by every job; the pool never re-quantizes the FFN input.
//
// So the only arithmetic this layer performs itself is the SwiGLU, which is elementwise and therefore
// split-invariant by construction. Its `expf` is the transcribed glibc one (basalt_cpu_expf.h), which is where a
// transcription of this file can diverge from basalt, per element rather than per split.
#pragma once

#include <cstdint>
#include <string>

#include "basalt_cpu.h"
#include "basalt_pool.h"

namespace monolith::basalt_cpu {

/// Jobs per batch (`pool.hpp:157`). More distinct experts than buffers runs them in batches of this.
inline constexpr int kMaxSplitMulti = 96;

/// The layer's geometry, which travels beside the jobs rather than in them (`native_expert.hpp:22-30`).
struct NativeFmt {
    int n_embd = H;         // the gate and up rows' width, and the down matrix's row count
    int n_ff = FF;          // the down rows' width, and the gate and up matrices' row count
    size_t gu_row = 0;      // bytes per gate or up row
    size_t d_row = 0;       // bytes per down row
    size_t up_off = 0;      // where the up rows begin inside the blob
    size_t down_off = 0;    // where the down rows begin inside the blob
};

/// One expert, for the tokens of a verify window that were routed to it (`pool.hpp:54-63`).
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;      // [gate rows | up rows | down rows]; a pointer into the expert source
    int nt = 0;                         // how many of the window's tokens reached this expert, 1 to MAXT
    const ActQ* act[MAXT] = {};         // the FFN input per token, SHARED read-only across the whole batch
    float* out[MAXT] = {};              // the n_embd-wide destination row per token, written by exactly one worker
};

/// The per-expert intermediate of one batch. basalt sizes 96 of these once at construction and the token path
/// allocates nothing; this holds them for the caller the same way.
struct SplitBufMulti {
    alignas(64) float ff[MAXT][FF];     // the SwiGLU output per token, before re-quantization
    ActQ a2[MAXT];                      // that vector quantized, which phase 2 reads
};

/// Run one layer's CPU experts through the pool. `jobs[0, n)` may exceed kMaxSplitMulti; they are run in batches
/// of it. Returns false with `error` set when a pool wait stalls, which poisons the pool.
///
/// `scratch` must hold at least `kMaxSplitMulti` SplitBufMulti, and is the caller's so that a session allocates it
/// once. The router weight is applied by the caller, not here.
bool run_split_multi_native(ExpertPool& pool, const NativeFmt& fmt, ExpertJobMulti* jobs, int n,
                            SplitBufMulti* scratch, std::string& error, PhaseTimes* gate_up = nullptr,
                            PhaseTimes* down = nullptr);

}  // namespace monolith::basalt_cpu
