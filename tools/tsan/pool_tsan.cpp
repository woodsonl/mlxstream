// A ThreadSanitizer driver for the pool: no Python, no Metal. TSan finds the data race on the batch
// description and on `done_` directly, where a property test can only hope to schedule into the window.
#include "basalt_pool.h"
#include <cstdio>
#include <unistd.h>
#include <string>
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
    return 0;
}
