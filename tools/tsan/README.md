# ThreadSanitizer drivers

Two drivers live here, one per class whose correctness is synchronisation rather than arithmetic: the CPU expert
pool, and the unbuffered reader. Run the matching one after any change to `runtime/src/basalt_pool.cpp` or
`runtime/src/basalt_direct_file.cpp`, or to the bindings that drive them in `runtime/src/native.mm`.

## The CPU expert pool

The CPU expert pool's correctness is synchronisation, not arithmetic, and a wrong synchronisation does not give
different numbers: it gives a run that hangs or computes a job twice. The Python suite
(`tests/kernels/test_basalt_pool.py`) asserts the pool's properties, but it cannot schedule itself into the
windows those properties protect. ThreadSanitizer can see the happens-before edges directly.

```bash
clang++ -std=gnu++20 -O1 -g -fsanitize=thread -I runtime/src \
    tools/tsan/pool_tsan.cpp runtime/src/basalt_pool.cpp -o /tmp/pool_tsan && /tmp/pool_tsan
```

A clean run prints two `ok: ...` lines and nothing else. Run it after any change to
`runtime/src/basalt_pool.cpp`.

The driver has two parts. The first drives one pool from one thread for 300 phases, republishing a fresh body each
phase, so a port that published the batch description after the epoch or read it before the acquire races there.
The second drives one pool from **two** threads, which the Python bindings and the pump thread both make reachable:
the bindings release the GIL and the host task runs on the pump. It reports how many phases were refused rather
than asserting a count, since whether the two overlap at all is the scheduler's business; the refusal itself is
asserted deterministically by `tests/kernels/test_basalt_pool.py` case C4.

**What it caught.** The first draft of the pool read the batch description at the top of `drain`, before the claim
whose acquire is the only thing ordering it against the host's write. Every property test passed; ThreadSanitizer
named the file and line of both sides.

**What it does not catch.** Weakening `publish`'s epoch bump, `begin_batch`'s head store or `drain`'s done
increment from their written orders to `relaxed` produces no warning here and no wrong result, over 300 phases of
16 jobs [M]. Those orders are transcribed from basalt with their arguments recorded in
`docs/research/basalt-cpu-pool.md` section 1.5, and nothing in this tree measures them.

## The unbuffered reader

```bash
clang++ -std=gnu++20 -O1 -g -fsanitize=thread -I runtime/src \
    tools/tsan/direct_file_tsan.cpp runtime/src/basalt_direct_file.cpp -o /tmp/df_tsan && /tmp/df_tsan
```

A clean run prints one `ok: ...` line and nothing else.

The lock this driver models is the BINDING's, not the class's. `DirectFile` carries no internal exclusion between
a driver and a `close`: `drive_mutex()` exists so that the caller supplies it, and `runtime/src/native.mm` takes
it in three places. The driver takes it in the same three and runs them against each other: a read thread holding
one submit-and-collect cycle, a thread closing and reopening the file, and a thread reading the `size`, `threads`
and `is_open` properties.

**What it catches.** Removing the lock from any one of the three arms is the mutation, and each is reported [M]:
from the property arm, five data races and an abort; from the closing arm, two races and then a hang; from the
reading arm, one race and a hang. The hang is the failure `close` produces directly, because it sets `stop_`,
joins every worker and only then clears the completion queue, so the driver waits forever on a completion already
cleared and nothing is left to post another. `tests/kernels/test_basalt_direct_file.py` case D9 covers that same
arm from Python, where it can only be reached by repeating the attempt; here it is deterministic.

**Non-vacuity.** `std::mutex` is unfair, and three hot threads starve one: the first draft of this driver got a
single property read in forty rounds and reported clean. The driver now yields outside the lock in each arm and
fails with `starved:` if any arm ran too few times, so a clean line means every arm actually ran.
