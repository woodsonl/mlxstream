// A ThreadSanitizer driver for the unbuffered reader: no Python, no Metal. The reader's correctness beside a
// close is synchronisation, and a wrong synchronisation here does not give different bytes: it gives a driver
// that waits forever for a completion already cleared, or an unsynchronised read of state `close` is writing.
// TSan sees the happens-before edges directly, where the Python cases can only hope to schedule into the window
// (`tests/kernels/test_basalt_direct_file.py` case D9 repeats its attempt twelve times for exactly that reason).
//
// The lock this models is the BINDING's, not the class's: `DirectFile` has no internal exclusion between a
// driver and a close, and `drive_mutex()` exists so the caller supplies it. So this driver takes the lock in
// the same three places `runtime/src/native.mm` does, and removing it from any one of them is the mutation.
#include "basalt_direct_file.h"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace monolith::basalt_cpu;

namespace {

constexpr int PAGE = 4096, PAGES = 64, ROUNDS = 40, READS = 256;

std::string write_fixture() {
    std::string path = "/tmp/direct_file_tsan.bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); _exit(2); }
    std::vector<unsigned char> page((size_t) PAGE);
    for (int p = 0; p < PAGES; ++p) {
        for (int i = 0; i < PAGE; ++i) page[(size_t) i] = (unsigned char) ((p * 31 + i) & 0xFF);
        std::fwrite(page.data(), 1, (size_t) PAGE, f);
    }
    std::fclose(f);
    return path;
}

}  // namespace

int main() {
    const std::string path = write_fixture();
    DirectFile f;
    std::string err;
    if (!f.open(path, err, 16)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 2; }

    std::atomic<bool> stop{false};
    std::atomic<long> rounds{0}, closes{0}, props{0};

    // The driver: one submit-and-collect cycle per round, under the lock for the whole cycle, as `read_many`
    // holds it. Collected by tag, because the queue is shared and a wake or another caller's read can arrive.
    std::thread driver([&] {
        std::vector<void*> bufs((size_t) READS, nullptr);
        for (size_t i = 0; i < bufs.size(); ++i) bufs[i] = DirectFile::alloc_aligned((size_t) PAGE);
        for (int r = 0; r < ROUNDS; ++r) {
            {
            std::lock_guard<std::mutex> drive(f.drive_mutex());
            std::string e;
            size_t submitted = 0;
            for (size_t i = 0; i < bufs.size(); ++i) {
                if (!f.submit((uint64_t) ((i % PAGES) * PAGE), bufs[i], (uint32_t) PAGE, (uint64_t) i, e)) break;
                ++submitted;
            }
            // Drained before the next round, never left queued: a worker still holds these pointers.
            for (size_t seen = 0; seen < submitted;) {
                Completion got[64];
                const int n = f.wait(got, 64, -1);
                for (int k = 0; k < n; ++k) if (got[k].tag < submitted) ++seen;
            }
            rounds.fetch_add(1);
            }                                     // released here, so the other two can take it
            std::this_thread::yield();
        }
        stop.store(true);
        for (void* b : bufs) DirectFile::free_aligned(b);
    });

    // The closer: `close` sets `stop_`, joins every worker and THEN clears the completion queue, so without the
    // lock it wipes the completion the driver is blocked on and no worker is left to post another.
    std::thread closer([&] {
        while (!stop.load()) {
            {
            std::lock_guard<std::mutex> drive(f.drive_mutex());
            f.close();
            std::string e;
            if (!f.open(path, e, 16)) { std::fprintf(stderr, "reopen: %s\n", e.c_str()); _exit(3); }
            closes.fetch_add(1);
            }
            std::this_thread::yield();        // std::mutex is unfair, and three hot threads starve one
        }
    });

    // The properties: plain reads of `size_`, `fd_` and `pool_`, every one of which `close` writes.
    std::thread properties([&] {
        while (!stop.load()) {
            {
            std::lock_guard<std::mutex> drive(f.drive_mutex());
            volatile uint64_t s = f.size();
            volatile int t = f.threads();
            volatile bool o = f.is_open();
            (void) s; (void) t; (void) o;
            props.fetch_add(1);
            }
            std::this_thread::yield();
        }
    });

    driver.join();
    closer.join();
    properties.join();
    f.close();
    std::remove(path.c_str());
    // Non-vacuity: std::mutex is unfair and a starved arm would report clean having barely run. The first
    // draft of this driver got ONE property read in forty rounds, which is no coverage of that arm at all.
    const long r = rounds.load(), c = closes.load(), pr = props.load();
    if (r < ROUNDS || c < 8 || pr < 8) {
        std::fprintf(stderr, "starved: %ld rounds (want %d), %ld closes, %ld property reads (want 8 each)\n",
                     r, ROUNDS, c, pr);
        return 5;
    }
    std::printf("ok: %ld read rounds, %ld close-reopen cycles, %ld property reads, no race\n", r, c, pr);
    return 0;
}
