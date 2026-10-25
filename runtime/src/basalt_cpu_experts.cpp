// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cpu/pool.cpp:415-451 and :543-572 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the mode >= 5 branch of `drain` and
// `run_split_multi_native`, with the row split and the serial step between the phases as written.

#include "basalt_cpu_experts.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>

#include "basalt_cpu_expf.h"

namespace monolith::basalt_cpu {

namespace {

/// The gate and up row values of one task's slice, per token. basalt keeps these in function-local `thread_local`
/// arrays; here they come out of the pool's own per-worker scratch, which on Darwin saves the thread-local
/// descriptor's lazy allocation and two indirect calls per access. Indexed by the ABSOLUTE row, as the kernel
/// writes them, so a task touches only its own slice of each.
struct RowScratch {
    float gbuf[MAXT][FF];
    float ubuf[MAXT][FF];
};

}  // namespace

bool run_split_multi_native(ExpertPool& pool, const NativeFmt& fmt, ExpertJobMulti* jobs, int n,
                            SplitBufMulti* scratch, std::string& error, PhaseTimes* gate_up, PhaseTimes* down) {
    if (n <= 0) return true;
    if (scratch == nullptr) throw std::invalid_argument("run_split_multi_native: scratch must not be null");
    if (fmt.n_embd != H || fmt.n_ff != FF)
        throw std::invalid_argument("run_split_multi_native: this pack's geometry is n_embd = 2560, n_ff = 640");
    for (int j = 0; j < n; ++j)
        if (jobs[j].nt < 1 || jobs[j].nt > MAXT)
            throw std::invalid_argument("run_split_multi_native: a job carries 1 to 8 tokens");

    // Refused here, not left to `run_phase`: a poisoned pool makes every phase fail, and each failure takes the
    // leak path below, so a caller that retries would grow the process without bound. After this, a failure means
    // this call's own phase stalled, which is when the leak is the right answer.
    if (pool.poisoned()) {
        error = "the CPU expert pool stalled earlier and cannot be reused";
        return false;
    }

    const int threads = pool.workers() + (pool.host_works() ? 1 : 0);
    const int tasks = 3 * threads;              // about three tasks per thread, so the tail is short
    const int gu_blocks = fmt.n_embd / QK;      // the gate and up rows' reduction, in Q2_0 blocks
    const int d_blocks = fmt.n_ff / QK;         // the down rows' reduction

    // More distinct experts than buffers: run them in batches of kMaxSplitMulti.
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = std::min(kMaxSplitMulti, n - b0);
        ExpertJobMulti* mjobs = jobs + b0;

        // ---- phase 1: the gate and up rows, then the SwiGLU -----------------------------------------------
        const int64_t gu_rows = (int64_t) nb * FF;
        // On the heap and captured by value, for the reason `PoolTask::run` does the same: `run_phase` keeps only
        // a POINTER to the function, so a stalled phase leaves a worker that may still be inside a body whose
        // stack `std::function` has already gone, and by-reference captures would name dead locals of this frame.
        // Leaked below on the poisoned path. The caller's `jobs` and `scratch` are leaked by the caller for the
        // same reason, and a poisoned pool is itself never destroyed (`basalt_pool.h`).
        auto gate_up_body = std::make_unique<std::function<void(int, int)>>(
            [mjobs, scratch, fmt, gu_rows, tasks, gu_blocks, &pool](int task, int worker) {
            RowScratch& rs = *static_cast<RowScratch*>(pool.scratch(worker, sizeof(RowScratch)));
            const int64_t g0 = gu_rows * task / tasks, g1 = gu_rows * (task + 1) / tasks;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / FF), r0 = (int) (r % FF);
                const int r1 = (int) std::min<int64_t>(FF, r0 + (g1 - r));
                SplitBufMulti& sb = scratch[e];
                float* gp[MAXT];
                float* up[MAXT];
                for (int t = 0; t < mjobs[e].nt; ++t) { gp[t] = rs.gbuf[t]; up[t] = rs.ubuf[t]; }
                q2_0_gguf_rows_multi(mjobs[e].blob, fmt.gu_row, gu_blocks, mjobs[e].act, mjobs[e].nt, gp, r0, r1);
                q2_0_gguf_rows_multi(mjobs[e].blob + fmt.up_off, fmt.gu_row, gu_blocks, mjobs[e].act, mjobs[e].nt,
                                     up, r0, r1);
                for (int t = 0; t < mjobs[e].nt; ++t)
                    for (int row = r0; row < r1; ++row)
                        sb.ff[t][row] = swiglu(rs.gbuf[t][row], rs.ubuf[t][row]);
                r += r1 - r0;
            }
        });
        if (!pool.run_phase(tasks, *gate_up_body, error, gate_up)) {
            if (pool.poisoned()) (void) gate_up_body.release();   // a worker may still be calling it
            return false;
        }

        // ---- between the phases: the host thread alone --------------------------------------------------
        // Over all 640 values of each token's vector at once, so the 20 chunk scales do not depend on how the
        // phase above was split. This is what makes the result the single-token job's, bit for bit.
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs[e].nt; ++t)
                act_quant_q8_1(scratch[e].ff[t], FF, scratch[e].a2[t]);

        // ---- phase 2: the down rows ---------------------------------------------------------------------
        const int64_t d_rows = (int64_t) nb * H;
        auto down_body = std::make_unique<std::function<void(int, int)>>(
            [mjobs, scratch, fmt, d_rows, tasks, d_blocks](int task, int) {
            const int64_t g0 = d_rows * task / tasks, g1 = d_rows * (task + 1) / tasks;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / H), r0 = (int) (r % H);
                const int r1 = (int) std::min<int64_t>(H, r0 + (g1 - r));
                const ActQ* a2[MAXT];
                for (int t = 0; t < mjobs[e].nt; ++t) a2[t] = &scratch[e].a2[t];
                q2_0_gguf_rows_multi(mjobs[e].blob + fmt.down_off, fmt.d_row, d_blocks, a2, mjobs[e].nt,
                                     mjobs[e].out, r0, r1);
                r += r1 - r0;
            }
        });
        if (!pool.run_phase(tasks, *down_body, error, down)) {
            if (pool.poisoned()) (void) down_body.release();      // a worker may still be calling it
            return false;
        }
    }
    return true;
}

}  // namespace monolith::basalt_cpu
