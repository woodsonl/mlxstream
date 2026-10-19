# ThreadSanitizer drivers

The CPU expert pool's correctness is synchronisation, not arithmetic, and a wrong synchronisation does not give
different numbers: it gives a run that hangs or computes a job twice. The Python suite
(`tests/kernels/test_basalt_pool.py`) asserts the pool's properties, but it cannot schedule itself into the
windows those properties protect. ThreadSanitizer can see the happens-before edges directly.

```bash
clang++ -std=gnu++20 -O1 -g -fsanitize=thread -I runtime/src \
    tools/tsan/pool_tsan.cpp runtime/src/basalt_pool.cpp -o /tmp/pool_tsan && /tmp/pool_tsan
```

A clean run prints `ok: ...` and nothing else. Run it after any change to `runtime/src/basalt_pool.cpp`.

**What it caught.** The first draft of the pool read the batch description at the top of `drain`, before the claim
whose acquire is the only thing ordering it against the host's write. Every property test passed; ThreadSanitizer
named the file and line of both sides.

**What it does not catch.** Weakening `publish`'s epoch bump, `begin_batch`'s head store or `drain`'s done
increment from their written orders to `relaxed` produces no warning here and no wrong result, over 300 phases of
16 jobs [M]. Those orders are transcribed from basalt with their arguments recorded in
`docs/research/basalt-cpu-pool.md` section 1.9, and nothing in this tree measures them.
