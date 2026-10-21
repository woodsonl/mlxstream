// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cpu/pool.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36
// (MIT; third_party/NOTICE): the constructor, `publish`, `worker`, `claim`, `begin_batch`, `wait_parked`,
// `wait_done` and `run_phase`, transcribed with the memory orders as written.
//
// The memory orders are the specification, not a detail. docs/research/basalt-cpu-pool.md section 1.9 lists every
// one against its line in basalt, and two of them carry arguments that break if weakened:
//
//   * `publish` and the sleep path are BOTH seq_cst, and that is the whole lost-wakeup argument: a worker going to
//     sleep does `sleepers_++` then reads `epoch_`, the host does `epoch_++` then reads `sleepers_`. In one total
//     order at least one sees the other's write, so either the worker sees the new epoch and does not sleep, or
//     the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`. The
//     store-buffer interleaving in which neither sees the other is exactly what seq_cst forbids. On arm64 this is
//     not free, where basalt's comment says it costs x86 nothing: it lowers to `ldaddal`/`dmb ish` rather than an
//     already-fenced `lock xadd`. It is still required.
//   * The batch description is plain, and is made visible solely by the release store of `head_` and the matching
//     acquire load in `claim`.

#include "basalt_pool.h"

#include <chrono>
#include <cstdio>

namespace monolith::basalt_cpu {

namespace {

/// `_mm_pause()`'s place. There is no SMT sibling on this chip to yield to, so the hint's only job is to lower the
/// spin's power and memory traffic.
inline void spin_pause() {
#if defined(__aarch64__) || defined(__arm64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

/// epoch << 32 | njobs << 16 | next index (`pool.cpp:22-24`).
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

ExpertPool::ExpertPool(int n_workers, bool host_works, std::chrono::microseconds spin_before_sleep)
    : host_works_(host_works), spin_before_sleep_(spin_before_sleep) {
    if (n_workers > 0) {
        n_ = n_workers;
    } else {
        // basalt drops the first physical core and keeps one logical per physical. This chip has no SMT and no way
        // to pin, so that collapses to "every core but the host's".
        const unsigned hc = std::thread::hardware_concurrency();
        n_ = hc > 1 ? (int) hc - 1 : 1;
    }
    if (n_ < 1) n_ = 1;

    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    scratch_.resize((size_t) n_ + 1);           // one per worker, plus the host's at index n_
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) threads_.emplace_back([this, i] { worker(i); });
}

ExpertPool::~ExpertPool() {
    stop_.store(true, std::memory_order_release);
    publish();                                  // so a PARKED worker notices the stop flag rather than sleeping through it
    for (auto& t : threads_) t.join();
}

bool ExpertPool::claim_owner(const void* owner) {
    const void* expected = nullptr;
    return owner_.compare_exchange_strong(expected, owner, std::memory_order_acq_rel, std::memory_order_acquire) ||
           expected == owner;
}

void ExpertPool::release_owner(const void* owner) {
    const void* expected = owner;
    owner_.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel, std::memory_order_acquire);
}

void* ExpertPool::scratch(int worker_id, size_t bytes) {
    const size_t slot = worker_id < 0 ? (size_t) n_ : (size_t) worker_id;
    auto& buf = scratch_[slot];
    if (buf.size() < bytes) buf.assign(bytes, 0);
    return buf.data();
}

void ExpertPool::publish() {
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim, which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;    // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;                 // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                        // exhausted
        // h + 1 increments the index field only: i < n <= 0xffff, so it never carries into njobs.
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

void ExpertPool::drain(int id, uint32_t epoch) {
    for (;;) {
        const int job = claim(epoch);
        if (job < 0) return;
        // Read AFTER the claim, never before it. The claim's acquire load of `head_` is the only thing that
        // orders this plain member against the host's write of it, and a read hoisted above the claim is a data
        // race on the host's next phase even though every worker reaching here came through an acquire of
        // `epoch_` (ThreadSanitizer reports it; the first draft of this file had the read at the top).
        (*body_)(job, id);
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = epoch_.load(std::memory_order_acquire);
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction. Counting only on the
    // return from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first phase,
    // which waits for `parked_ == n_` before publishing, then deadlocks.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            spin_pause();
            if ((++spins & (kSpinSample - 1)) != 0) continue;         // the clock is read once per 1024 pauses,
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;   // so the spin is unchanged
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);              // leaving the park
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);
    }
}

bool ExpertPool::wait_parked(const char* what, std::string& error) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    uint32_t spins = 0;
    // Armed once, where basalt re-arms it whenever `spins` hits 1024 and so restarts the clock every 2^32
    // iterations. The transcribed body costs 0.422 ns here, which is 1.8 s per wrap against a 60 s limit [M].
    const auto t0 = std::chrono::steady_clock::now();
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        spin_pause();
        if ((++spins & (kSpinSample - 1)) != 0) continue;
        if (std::chrono::steady_clock::now() - t0 > kStall) {
            poisoned_.store(true, std::memory_order_release);
            error = std::string("the CPU expert pool stalled ") + what + " (" +
                    std::to_string(parked_.load(std::memory_order_acquire)) + " of " + std::to_string(n_) +
                    " workers parked)";
            return false;
        }
    }
    return true;
}

bool ExpertPool::wait_done(int n, std::string& error) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    uint32_t spins = 0;
    auto t0 = std::chrono::steady_clock::now();
    uint32_t seen = 0;
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return true;                           // >=: never a wait that an overshoot outlives
        spin_pause();
        if ((++spins & (kSpinSample - 1)) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (d != seen) {                                              // restarted on progress, which is deliberate
            t0 = now;
            seen = d;
        } else if (now - t0 > kStall) {
            poisoned_.store(true, std::memory_order_release);
            error = "the CPU expert pool stalled waiting for a batch to finish (" + std::to_string(d) + " of " +
                    std::to_string(n) + " jobs done)";
            return false;
        }
    }
}

bool ExpertPool::run_phase(int n_tasks, const std::function<void(int, int)>& body, std::string& error,
                           PhaseTimes* times) {
    if (poisoned_.load(std::memory_order_acquire)) {
        error = "the CPU expert pool stalled earlier and cannot be reused";
        return false;
    }
    if (n_tasks < 0 || n_tasks > 0xffff) {
        error = "a pool batch holds 0 to 65535 jobs, not " + std::to_string(n_tasks);
        return false;
    }

    const auto t_before = std::chrono::steady_clock::now();
    if (!wait_parked("before a phase", error)) return false;
    if (times) times->wait_parked_before_ms = ms_since(t_before);

    const auto t_work = std::chrono::steady_clock::now();
    body_ = &body;                                                    // plain; ordered by begin_batch's release
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, e);
    if (!wait_done(n_tasks, error)) return false;
    if (times) times->work_ms = ms_since(t_work);

    const auto t_after = std::chrono::steady_clock::now();
    if (!wait_parked("after a phase", error)) return false;
    if (times) times->wait_parked_after_ms = ms_since(t_after);

    hstate_.store(kIdle, std::memory_order_relaxed);
    return true;
}

PoolTask::PoolTask(std::shared_ptr<ExpertPool> pool, Work work) : pool_(std::move(pool)), work_(std::move(work)) {}

PoolTask::~PoolTask() {
    if (pool_) pool_->release_owner(this);
}

bool PoolTask::run(const HostCall& call, std::string& error) {
    if (!pool_) {
        error = "this host point's pool is gone";
        return false;
    }
    // One owner at a time, so two tasks cannot drive one pool and overlap their phases. Claimed on the first call
    // rather than at construction, because the owner is the task and the pool may outlive any one program.
    if (!pool_->claim_owner(this)) {
        error = "the CPU expert pool is already driven by another host task";
        return false;
    }
    std::function<void(int, int)> body;
    const int n = work_(call.step, call.point, body);
    if (n <= 0) return true;                                      // a point with nothing to do
    if (!body) {
        error = "the host point asked for " + std::to_string(n) + " jobs and gave no body to run";
        return false;
    }
    return pool_->run_phase(n, body, error, &last_);
}

}  // namespace monolith::basalt_cpu
