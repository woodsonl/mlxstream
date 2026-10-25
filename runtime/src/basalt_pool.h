// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) include/basalt/kernels/cpu/pool.hpp and src/kernels/cpu/pool.cpp @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the CPU expert pool's structure, its
// epoch-tagged claim protocol, its park and its phase barriers.
//
// The worker pool that runs the CPU expert tier. basalt's CPU experts are computed by a pool of spinning threads
// the host drives one batch at a time, and nothing bitwise depends on the thread count, the placement or the order
// in which jobs are claimed: a job writes its own rows and the host sums them in a fixed order
// (docs/research/basalt-cpu-pool.md section 2). What the pool must reproduce exactly is its SYNCHRONISATION, since
// getting that wrong does not give different numbers, it gives a run that hangs or computes a job twice.
//
// THE QUEUE IS ONE 64-BIT WORD. There is no ring and no lock, because jobs cannot arrive while others are still
// being computed: the host must sum every output before the next layer starts, so a layer is a barrier by
// construction and the queue never holds more than one batch. The word is `epoch << 32 | njobs << 16 | index`, a
// claim is a compare-exchange that only succeeds for the epoch the worker woke for, and a late worker's claim
// simply fails.
//
// WHY THE EPOCH IS IN THE WORD. A worker that went to sleep is still counted as parked when it wakes: it reads
// the new epoch, and only then decrements `parked_`. In that gap the host may read `parked_ == n_`, pass its next
// pre-phase barrier and publish the next batch, so the worker enters `drain` holding the epoch it woke for while
// `head_` already carries the next. The park barriers narrow that window and do not close it. basalt's reason for
// the epoch is that such a worker could otherwise claim from the next batch while the host was still writing it,
// run a job twice, or add to `done` after the host had reset it (`pool.hpp:18-23`).
//
// Here, neither of the last two is reachable even with the epoch test removed, because three other things hold:
// the claim's acquire of `head_` synchronises with `begin_batch`'s release and the batch description is written
// before it, the index CAS hands out each job exactly once, and `wait_done` has seen every increment of the
// previous batch before the reset. A late claim therefore lands on the next batch as an early, correct
// participant. The epoch test stays because this is a 1:1 port of basalt's protocol, not because a measurement
// here showed its removal breaking (docs/research/basalt-cpu-pool.md section 4 records what was measured).
//
// WHAT THIS PORT CHANGES, and why each is not a numeric difference:
//   * A stall reports instead of aborting. basalt calls `release_gpu_waits` then `std::abort()`; this returns
//     false with a message, which the host task turns into StepState.error = 4 and Engine.run raises. A stalled
//     pool is POISONED: its workers may still hold pointers into buffers the caller is about to free, so it is
//     never destroyed and never reused, which is what abort() buys basalt for free.
//   * The stall clock is armed once per wait rather than every 2^32 spins. The transcribed spin body costs
//     0.422 ns here, so 2^32 iterations is 1.8 s against a 60 s limit and the abort could never fire [M].
//   * The worker count and the spin-before-sleep come from the chip profile. basalt derives the count from a Linux
//     CPU topology this platform does not expose, and no placement control here is more than a hint.
//   * Each worker's gate and up buffers are owned by the pool and indexed by worker id, where basalt uses
//     function-local `thread_local` arrays. On Darwin each such access goes through the thread-local-variable
//     descriptor and allocates lazily on first use; the buffers are scratch, so this is invisible to the output.
//   * The hot atomics are padded to 128 bytes, which is this chip's cache line, where basalt pads to 64 [M].
//   * `run_phase` refuses a second concurrent driver. basalt's pool has one host thread and needs no such flag;
//     here the Python bindings release the GIL and `PoolTask` runs on the pump, so two native drivers of one pool
//     are reachable, and two of them race on the batch description (ThreadSanitizer names it).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "metal_core.h"
#include <vector>

namespace monolith::basalt_cpu {

/// This chip's cache line, so each hot atomic gets a line of its own. basalt uses 64, which is x86's; here that
/// would leave two of them sharing. Not `std::hardware_destructive_interference_size`, which this toolchain
/// reports as 256 and which would over-pad by two [M].
inline constexpr size_t kCacheLine = 128;

/// A pool wait that sees no completion for this long is a bug (`pool.hpp:187`).
inline constexpr std::chrono::seconds kStall{60};

/// How often any spin reads the clock, in iterations (`pool.cpp:301`, `:359`, `:382`).
inline constexpr uint32_t kSpinSample = 1024;

/// Worker and host diagnostic states (`pool.hpp:224-225`).
enum : int32_t { kParked = -1, kSleeping = -2, kBetween = -3, kIdle = -10, kWaitParked = -11, kWaitDone = -12 };

/// What one phase cost, in milliseconds. basalt keeps three such timers and reports them; a slow pool poses one
/// question, slow at the work or slow at the synchronisation, and only these answer it (`pool.hpp:158-160`).
struct PhaseTimes {
    double wait_parked_before_ms = 0;
    double work_ms = 0;              // the host's own drain plus the wait for `done`
    double wait_parked_after_ms = 0;
};

/// basalt's CPU expert pool. One host thread drives it. Two exclusions, answering different questions: `run_phase`
/// refuses a second concurrent driver, and `claim_owner` refuses a second owner as a `HostTask` refuses a second
/// runner. An owner holds its claim across calls, so the owner claim cannot serve as the per-phase exclusion.
class ExpertPool {
  public:
    /// `n_workers` worker threads, plus the calling thread when `host_works`. A count at or below zero takes
    /// `std::thread::hardware_concurrency() - 1`, which is basalt's rule once this platform's topology collapses
    /// to "every core" (no SMT here, so the one-logical-per-physical dance has nothing to do).
    explicit ExpertPool(int n_workers = 0, bool host_works = true,
                        std::chrono::microseconds spin_before_sleep = std::chrono::milliseconds{20});

    /// Joins every worker. A poisoned pool must never reach this: see `poisoned()`.
    ~ExpertPool();

    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    bool host_works() const { return host_works_; }

    /// True once a wait has stalled. The workers may still hold pointers into the caller's buffers, so a poisoned
    /// pool is never destroyed, never reused and leaked for the process's life. `PoolTask` enforces that by
    /// abandoning its reference and its job body; a caller that drives the pool directly owns the same duty, and
    /// `run_phase` refuses a poisoned pool so the refusal does not depend on the caller remembering.
    bool poisoned() const { return poisoned_.load(std::memory_order_acquire); }

    /// Claim the pool for one owner, by compare-exchange. Returns false and names nothing it does not know when
    /// another owner holds it; `release` gives it back. Two distinct host tasks could otherwise drive one pool and
    /// overlap their phases.
    bool claim_owner(const void* owner);
    void release_owner(const void* owner);
    const void* owner() const { return owner_.load(std::memory_order_acquire); }

    /// Run one phase: `n_tasks` jobs, each `body(job_index, worker_id)`, where `worker_id` is 0..workers()-1 for a
    /// worker and -1 for the host's own drain. Returns false on a stall, leaving `error` set and the pool
    /// poisoned, and false without running anything when another thread is already inside a phase. The body must
    /// not throw.
    bool run_phase(int n_tasks, const std::function<void(int, int)>& body, std::string& error,
                   PhaseTimes* times = nullptr);

    /// Per-worker scratch the caller may index by the `worker_id` its body is given, sized `bytes` per worker on
    /// the first call and kept for the pool's life. Slot `workers()` is the host's. basalt uses function-local
    /// `thread_local` arrays for this, which on Darwin allocate lazily and cost two indirect calls per access.
    void* scratch(int worker_id, size_t bytes);

  private:
    void worker(int id);
    void drain(int id, uint32_t epoch);
    int claim(uint32_t epoch);
    uint32_t begin_batch(int n);
    void publish();
    void poison();                                 // marks the pool and exhausts the queue, in that order
    bool wait_parked(const char* what, std::string& error);
    bool wait_done(int n, std::string& error);

    const bool host_works_;
    int n_ = 1;
    std::chrono::microseconds spin_before_sleep_;

    // The four the workers and the host contend on, each on its own line: the line the host writes to publish work
    // and reads to confirm the workers are parked must not be hammered by the threads waiting for it.
    alignas(kCacheLine) std::atomic<uint64_t> head_{0};      // epoch << 32 | njobs << 16 | next index
    alignas(kCacheLine) std::atomic<uint32_t> done_{0};
    alignas(kCacheLine) std::atomic<uint32_t> parked_{0};
    alignas(kCacheLine) std::atomic<uint32_t> epoch_{0};
    alignas(kCacheLine) std::atomic<bool> stop_{false};
    alignas(kCacheLine) std::atomic<uint32_t> sleepers_{0};

    std::atomic<bool> poisoned_{false};
    std::atomic<const void*> owner_{nullptr};
    std::atomic<bool> driving_{false};             // a phase is in flight: `run_phase`'s own exclusion

    // The batch description. Plain, not atomic: it is made visible to a worker solely by the release/acquire pair
    // on `head_` plus the seq_cst epoch bump, so the host writes it BEFORE `begin_batch` and a worker reads it
    // only after its acquire. Publishing it after the epoch, or reading it before the acquire, is a data race.
    const std::function<void(int, int)>* body_ = nullptr;

    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::unique_ptr<std::atomic<int32_t>[]> wstate_;
    std::atomic<int32_t> hstate_{kIdle};
    std::vector<std::vector<unsigned char>> scratch_;
    std::vector<std::thread> threads_;
};

/// A host task that drives the pool, for a program whose points carry CPU expert work.
///
/// It is a C++ `HostTask` and not a `CallableTask` on purpose: a Python task takes the GIL on every call, and this
/// one runs on the pump thread once per MoE layer, which is 48 times a step. The pump commits nothing while it
/// runs, so the GPU's own experts for the layer must already be in flight; that is what the point's
/// [publish, wait) window is for.
///
/// The pool's release/acquire pair is what publishes the workers' rows to the GPU, not just to the waiting host:
/// a worker's stores land, then its `done_` increment releases, then the pump's `wait_done` acquires, then the
/// pump commits the command buffer that reads them. A port that made `done_` a plain counter would break GPU
/// visibility and not merely the wait.
class PoolTask : public HostTask {
  public:
    /// `work(step, point)` returns the jobs for this call: the count, and the body each job runs. A count of zero
    /// is a point with nothing to do, which still returns true.
    using Work = std::function<int(uint64_t, uint32_t, std::function<void(int, int)>&)>;

    PoolTask(std::shared_ptr<ExpertPool> pool, Work work);
    ~PoolTask() override;

    bool run(const HostCall& call, std::string& error) override;

    /// The phase times of the last call that ran jobs, for the caller to report.
    PhaseTimes last_times() const { return last_; }

  private:
    void abandon(std::unique_ptr<std::function<void(int, int)>> body);

    std::shared_ptr<ExpertPool> pool_;
    Work work_;
    PhaseTimes last_;
};

}  // namespace monolith::basalt_cpu
