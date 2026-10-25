// A ThreadSanitizer driver for the pool: no Python, no Metal. TSan finds the data race on the batch
// description and on `done_` directly, where a property test can only hope to schedule into the window.
#include "basalt_pool.h"
#include <atomic>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <thread>
#include <vector>
using namespace monolith::basalt_cpu;
int main() {
    ExpertPool pool(8, true, std::chrono::microseconds{200});
    const int n = 16, phases = 300;
    std::vector<int> count((size_t) n, 0);
    for (int k = 0; k < phases; ++k) {
        std::string err;
        // A fresh body each phase, so the plain `body_` pointer is genuinely republished: a port that
        // published the description after the epoch, or read it before the acquire, races here.
        std::vector<int> mark((size_t) n, 0);
        const bool ok = pool.run_phase(n, [&count, &mark](int job, int) {
            count[(size_t) job] += 1;
            mark[(size_t) job] = 1;
        }, err, nullptr);
        if (!ok) {
            // _exit, not return: the pool's contract forbids destroying it once a stall has poisoned it, and a
            // return here would unwind the automatic `pool` and join a worker that may never come back.
            std::fprintf(stderr, "phase %d failed: %s\n", k, err.c_str());
            std::fflush(stderr);
            _exit(2);
        }
        for (int j = 0; j < n; ++j) if (!mark[(size_t) j]) { std::fprintf(stderr, "job %d did not run\n", j); return 3; }
    }
    for (int j = 0; j < n; ++j)
        if (count[(size_t) j] != phases) { std::fprintf(stderr, "job %d ran %d times, not %d\n", j, count[(size_t) j], phases); return 4; }
    std::printf("ok: %d phases x %d jobs, each exactly once\n", phases, n);

    // Two drivers of one pool, which the Python bindings and the pump thread both make reachable: the bindings
    // release the GIL and PoolTask runs on the pump. Before run_phase took its in-phase flag, TSan named the race
    // on the batch description right here and neither driver was ever refused. The count is reported rather than
    // asserted: whether the two overlap at all is the scheduler's business, and the refusal itself is asserted
    // deterministically by tests/kernels/test_basalt_pool.py C4.
    std::atomic<int> refused{0};
    auto drive = [&pool, &refused, n]() {
        for (int k = 0; k < 60; ++k) {
            std::string err;
            if (!pool.run_phase(n, [](int, int) {}, err, nullptr)) {
                refused.fetch_add(1);
                if (pool.poisoned()) {
                    std::fprintf(stderr, "two drivers poisoned the pool: %s\n", err.c_str());
                    std::fflush(stderr);
                    _exit(5);
                }
            }
        }
    };
    std::thread ta(drive), tb(drive);
    ta.join();
    tb.join();
    std::printf("ok: two drivers, 120 phases, %d refused\n", refused.load());
    return 0;
}
