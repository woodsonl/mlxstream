// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) include/basalt/platform/direct_file.hpp and
// src/platform/direct_file.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the
// unbuffered reader, its thread pool, its completion protocol and its alignment contract.
//
// The n-gram table is 26.8 GiB and must stay on the device rather than occupy RAM, the operating system's file
// cache included. basalt states why a memory map will not do: a map cannot promise that, and an unbuffered read
// can (`direct_file.hpp:3-5`). The reads are random and 4 KiB, issued from a small pool of threads so that many
// are in flight at once, because one thread issuing every read would cap the table's prompt reads at the kernel
// time per read where the device serves several times that at depth.
//
// THE SUBSTITUTION THIS PORT MAKES, and it is the only one in the whole n-gram path that is not a transcription.
// macOS has no `O_DIRECT`. The nearest thing is `F_NOCACHE` set on the descriptor after opening, with
// `F_RDAHEAD` turned off because read-ahead on random 4 KiB reads is pure waste. Three differences follow, and
// none of them reaches a produced value:
//
//   * `F_NOCACHE` is ADVISORY where `O_DIRECT` is STRUCTURAL. It asks the unified buffer cache not to retain
//     this descriptor's pages. It does not promise the bytes never transit RAM, and it does not evict pages
//     already resident. So basalt's "never occupies RAM" becomes a strong tendency here rather than a
//     guarantee. That is a memory-pressure and timing difference.
//   * `F_NOCACHE` imposes no alignment, where `O_DIRECT` requires offset, length and buffer to be multiples of
//     4096. This port KEEPS the aligned scheme anyway. The two-page slot, the straddle handling, the in-page
//     offsets and the row dedupe are all built on it, so dropping it would be a redesign rather than a port.
//   * basalt warns that a live mapping on the same file serialises its unbuffered reads, and drops the mapping
//     after the header parse. That is a page-cache coherency behaviour `F_NOCACHE` has no contract to enforce,
//     so the warning may not apply here. Keeping the rule costs nothing, so the port keeps it.
//
// A SHORT READ AT THE END OF THE FILE IS NOT AN ERROR, and the table's end makes that reachable rather than
// theoretical. The table ends at byte 28,800,138,432, which is not a multiple of 4096: the final page holds
// 3,264 bytes. The last row begins 3,174 bytes into that page and is 90 long, so it ends exactly at 3,264. The
// reference's own last-row probe is therefore also its end-of-file tolerance test, and a port that treats a
// short read as a failure fails that probe.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace monolith::basalt_cpu {

/// One finished read. `tag` is whatever the caller passed to `submit`; the reader above passes its slot index.
/// A read that ran but stopped at the end of the file is `ok` with `bytes` short of the length asked for.
struct Completion {
    uint64_t tag = 0;
    uint32_t bytes = 0;
    bool ok = false;
};

/// The tag a `wake` posts, so a thread parked in `wait` returns without a real read completing
/// (`direct_file.hpp:44`).
inline constexpr uint64_t DIRECT_WAKE_TAG = ~0ull;

/// Offset, length and buffer are all multiples of this (`direct_file.hpp:33`).
inline constexpr uint64_t DIRECT_ALIGNMENT = 4096;

/// How many threads the pool runs by default. Each does one blocking read, so the thread count IS the device
/// queue depth (`direct_file.cpp:46-47`, :101).
inline constexpr int DIRECT_IO_THREADS = 16;

class DirectFile {
  public:
    ~DirectFile();

    DirectFile() = default;
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    /// Open unbuffered and start the pool. `threads` is clamped to 1 through 64, as basalt clamps it.
    bool open(const std::string& path, std::string& err, int threads = DIRECT_IO_THREADS);
    void close();

    bool is_open() const { return fd_ >= 0; }
    uint64_t size() const { return size_; }
    int threads() const { return (int) pool_.size(); }

    /// Queue one read. Returns false and sets `err` when the alignment contract is broken; queuing is all this
    /// does, and a pool thread issues it.
    bool submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err);

    /// Collect up to `max` completions. A negative timeout blocks, zero polls, a positive one waits that many
    /// milliseconds. Completions come back in the order the threads finished, which has nothing to do with the
    /// order they were submitted in, so a caller keys off the tag.
    int wait(Completion* out, int max, int timeout_ms);

    /// Post a synthetic completion so a parked `wait` returns.
    void wake();

    static void* alloc_aligned(size_t bytes);
    static void free_aligned(void* p);

    /// One driver at a time. `wait` pops from a single completion queue shared by every caller, so two drivers
    /// each take completions belonging to the other and a driver that discards what is not its own then waits
    /// forever for a completion already consumed. basalt never meets this because exactly one reader thread
    /// owns every call into this class; nothing here enforces that, so a caller holds this across a whole
    /// submit-and-collect cycle. A port addition, like the expert pool's in-phase flag.
    std::mutex& drive_mutex() { return drive_mu_; }

  private:
    struct Pending {
        uint64_t offset;
        void* buffer;
        uint32_t length;
        uint64_t tag;
    };

    void worker();

    int fd_ = -1;
    uint64_t size_ = 0;
    std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    std::deque<Pending> queue_;
    std::deque<Completion> done_;
    std::vector<std::thread> pool_;
    std::mutex drive_mu_;
    bool stop_ = false;
};

}  // namespace monolith::basalt_cpu
