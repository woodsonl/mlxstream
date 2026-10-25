// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/platform/direct_file.cpp @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): `worker`, `open`, `close`, `submit`,
// `wait`, `wake` and the aligned allocators, with `O_DIRECT` replaced by `F_NOCACHE` as the header explains.

#include "basalt_direct_file.h"

#include <algorithm>
#include <cstdlib>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace monolith::basalt_cpu {

namespace {

/// The issuing-thread count, clamped to 1 through 64 (`direct_file.cpp:48`). Each thread does one blocking
/// read, so this is the queue depth.
int io_threads(int dflt) { return std::clamp(dflt, 1, 64); }

}  // namespace

void* DirectFile::alloc_aligned(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, (size_t) DIRECT_ALIGNMENT, bytes) != 0) return nullptr;
    return p;
}

void DirectFile::free_aligned(void* p) { std::free(p); }

DirectFile::~DirectFile() { close(); }

bool DirectFile::open(const std::string& path, std::string& err, int threads) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        err = "DirectFile: cannot open " + path;
        return false;
    }
    // The substitution. F_NOCACHE asks the unified buffer cache not to retain this descriptor's pages, which is
    // the nearest macOS offers to O_DIRECT; F_RDAHEAD off because these are random 4 KiB reads and read-ahead
    // on them is waste. Both are advisory, so neither failing is fatal: the reads still return the right bytes,
    // they just may leave pages resident.
    ::fcntl(fd_, F_NOCACHE, 1);
    ::fcntl(fd_, F_RDAHEAD, 0);

    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    size_ = (uint64_t) st.st_size;

    stop_ = false;
    const int n = io_threads(threads);
    pool_.reserve((size_t) n);
    for (int i = 0; i < n; ++i) pool_.emplace_back([this] { worker(); });
    return true;
}

void DirectFile::close() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : pool_) {
        if (t.joinable()) t.join();
    }
    pool_.clear();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    std::lock_guard<std::mutex> lk(mu_);
    queue_.clear();
    done_.clear();
    size_ = 0;
    stop_ = false;
}

void DirectFile::worker() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_work_.wait(lk, [&] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        const Pending p = queue_.front();
        queue_.pop_front();
        lk.unlock();
        // One blocking read per thread, which is what makes the thread count the queue depth. A read whose
        // range crosses the end of the file comes back short rather than failing, and the caller checks the
        // byte count; only a negative return is an error.
        const ssize_t got = ::pread(fd_, p.buffer, p.length, (off_t) p.offset);
        lk.lock();
        done_.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
        cv_done_.notify_all();
    }
}

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    // Kept from the O_DIRECT contract even though F_NOCACHE does not require it: the two-page slot, the
    // straddle handling and the in-page offsets above are all built on page-aligned reads.
    if (offset % DIRECT_ALIGNMENT || length % DIRECT_ALIGNMENT ||
        ((uintptr_t) buffer) % DIRECT_ALIGNMENT || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    if (fd_ < 0) {
        err = "DirectFile: not open";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(Pending{offset, buffer, length, tag});
    }
    cv_work_.notify_one();
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    if (max <= 0) return 0;
    std::unique_lock<std::mutex> lk(mu_);
    if (done_.empty()) {
        if (timeout_ms < 0) {
            cv_done_.wait(lk, [&] { return !done_.empty(); });
        } else if (timeout_ms > 0) {
            cv_done_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !done_.empty(); });
        }
    }
    int n = 0;
    while (n < max && !done_.empty()) {
        out[n++] = done_.front();
        done_.pop_front();
    }
    return n;
}

void DirectFile::wake() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        done_.push_back(Completion{DIRECT_WAKE_TAG, 0, true});
    }
    cv_done_.notify_all();
}

}  // namespace monolith::basalt_cpu
