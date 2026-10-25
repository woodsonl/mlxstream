# basalt's CPU expert pool: the queue, the orders, and where its host point goes

`runtime/src/basalt_pool.{h,cpp}` transcribes basalt's CPU expert pool (`src/kernels/cpu/pool.cpp` and
`include/basalt/kernels/cpu/pool.hpp` @ `b4c63b55754637f4c72ad53a0f791d425f71eb36`). The pool is the host half of
the expert tier: a set of spinning worker threads the host drives one batch at a time, so that an expert without a
resident GPU copy is computed on the CPU while the GPU runs the experts that do have one.

Nothing bitwise depends on the pool, which is section 2 below, so what a transcription has to reproduce exactly is
its **synchronisation**. Getting that wrong does not give different numbers. It gives a run that hangs, or one that
computes a job twice. This note records the three things the port rests on: the queue and its claim protocol, every
memory order with its argument, and where the pool's work sits in a Metal step.

Evidence tags: [M] measured, [S] vendor specification, [R] third-party report, [H] hypothesis. A line marked
**Inference** is read off basalt's source rather than stated by it.

## 1. The pool

### 1.1 The queue is one word

There is no ring and no lock. The header gives both reasons, verbatim (`pool.hpp:7-11`): "A lock-free SPMC ring is
what you need when jobs arrive while others are still being computed. Here they cannot: the host must SUM all ten
outputs before the next layer starts, so a layer is a barrier by construction and the queue never holds more than
one batch."

The whole queue is therefore one 64-bit word (`pool.cpp:22-24`, `pool.hpp:229`):

```cpp
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
alignas(64) std::atomic<uint64_t> head_{0};   // epoch << 32 | njobs << 16 | next index
```

Publishing is `begin_batch` (`pool.cpp:338-350`), in this order and no other: the completion counter is reset, the
epoch the host is about to use is computed, the word is stored with a release, and only then are sleepers notified.
basalt states why the reset is safe where it sits, verbatim: "Every job of the previous batch has completed
(`wait_done`), and a claim of it can no longer succeed, so nothing adds to `done` until this batch's first claim,
which the release below orders after the reset."

A claim (`pool.cpp:327-336`) is a compare-exchange on that word which **succeeds only for the epoch the worker woke
for**. A late worker's claim simply fails.

**Why the epoch is in the word**, which is the one thing a port must not simplify away: a worker that went to sleep
is still counted as parked when it wakes, so for a moment after it has seen a new epoch the host believes it is
idle. Without the epoch a late worker could claim from the next batch while the host was still writing it, run a
job twice, or add to `done` after the host had reset it, and `done != n` would then never end.

### 1.2 The park: spin, then sleep

A worker arrives at the park **before** its first wait (`pool.cpp:284`), with the reason verbatim (`:280-283`):
"ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction. Counting only on the
RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first `run()`, which
waits for `parked_ == n_` before publishing, then deadlocks."

The park loop (`pool.cpp:296-315`) spins on `epoch_` with an acquire load, checks `stop_` each iteration, reads the
clock once every 1024 iterations, and once it has spun for `spin_before_sleep_` takes the sleep mutex and blocks on
the condition variable.

### 1.3 A phase: two barriers around the batch

Verbatim (`pool.hpp:13-16`): "`run()` waits for `done == n` AND for every worker to PARK. Waiting only for `done` is
not enough: a worker can still be inside the drain loop after its last `done` increment, and the host resetting
`head` underneath it would let that worker claim a job from the NEXT batch before the next batch has been
published."

So a phase is `wait_parked`, publish, the host's own drain when it works, `wait_done`, `wait_parked`
(`pool.cpp:476-486`). A wait that sees no progress for `kStall` is a bug: basalt prints, releases its GPU waits and
calls `std::abort()` (`pool.cpp:362-368`).

### 1.4 The constants this port carries

| Constant | Value | Where | What it governs |
|---|---|---|---|
| `kSpinBeforeSleep` | 20 ms | `pool.hpp:184`, held as microseconds at `:239` | how long a parked worker spins before it blocks on the condition variable |
| `kStall` | 60 s | `pool.hpp:187` | "A pool wait that sees no completion for this long is a bug; the engine stops with a message instead of spinning forever, and the server starts it again" |
| spin sample period | 1024 iterations | `pool.cpp:301`, `:359`, `:382` | `if ((++spins & 1023u) != 0) continue;`, how often any spin reads the clock |
| `kMaxSplitMulti` | 96 | `pool.hpp:157` | the multi-token job limit; `run_split_multi_native` batches in chunks of it (`pool.cpp:547-548`) |
| `MAXT` | 8 | `expert.hpp:122` | tokens per job |
| `kParked`, `kSleeping`, `kBetween` | -1, -2, -3 | `pool.hpp:224-225` | worker diagnostic states |
| `kIdle`, `kWaitParked`, `kWaitDone` | -10, -11, -12 | `pool.hpp:224-225` | host diagnostic states |
| tasks per phase | `3 * threads` | `pool.cpp:524`, `:552` | "about three tasks per thread in each phase, so the tail is short" (`:494`) |
| batch size limit | `0xffff` | `pool.cpp:339` | the job count occupies 16 bits of the word, so a larger batch would overflow into the epoch |
| cache-line padding | `alignas(64)` | `pool.hpp:229-233`, `:236`, `:246`, `:255`, `:257` | each hot atomic on its own line; this chip's line is 128 [M] |

### 1.5 Every memory order

| Variable | Writes | Reads |
|---|---|---|
| `epoch_` | `fetch_add` **seq_cst** (`:271`), host only | **acquire** in the park test (`:298`) and after it (`:314`); **seq_cst** inside the condition-variable predicate (`:307`); **relaxed** in `begin_batch` (`:346`) |
| `head_` | **release** store (`:347`); CAS **acq_rel**, **acquire** on failure (`:333`) | **acquire** (`:328`) |
| `done_` | **relaxed** store 0 (`:345`); `fetch_add` **release** (`:472`, `:613`) | **acquire** (`:379`) |
| `parked_` | `fetch_add` and `fetch_sub` **acq_rel** (`:284`, `:315`, `:323`) | **acquire** (`:357`) |
| `stop_` | **release** (`:259`) | **relaxed** (`:299`, `:307`), **acquire** (`:312`) |
| `sleepers_` | `fetch_add` **seq_cst** (`:305`), `fetch_sub` **relaxed** (`:309`) | **seq_cst** (`:272`) |
| worker and host diagnostic states | **relaxed** (`:304`, `:310`, `:320`, `:322`, `:353-354`, `:402-403`) | default, in the diagnostic dump only |
| the batch description | plain stores by the host before `begin_batch` | plain loads by workers after the acquire |

Two of these carry arguments that break if weakened.

**The seq_cst pair is the lost-wakeup argument.** A worker going to sleep increments `sleepers_` and then reads
`epoch_`; the host increments `epoch_` and then reads `sleepers_`. Both pairs are seq_cst, so in one total order at
least one side sees the other's write: either the worker sees the new epoch and does not sleep, or the host sees
the sleeper and notifies under the mutex the worker holds until it is inside `wait`. The store-buffer interleaving
in which neither sees the other is exactly what seq_cst forbids. basalt's comment says this costs x86 nothing; on
arm64 it is not free, lowering to `ldaddal` and `dmb ish` rather than an already-fenced `lock xadd` [M]. It is
still required.

**The batch description is plain, and nothing else orders it.** **Inference** from the table's last row: the
description is made visible solely by the release store of `head_` and the matching acquire load in `claim`, plus
the seq_cst epoch bump. A port that publishes the description after the epoch, or that reads it before the
acquire, is a data race. The port's first draft read it at the top of `drain`, before the claim, and every property
test passed over it; ThreadSanitizer named both sides (`tools/tsan/`).

The header states why each hot atomic is padded, verbatim (`pool.hpp:217-221`): "`head_`, `done_`, `parked_` and
`epoch_` are the four that workers and the host actually contend on; sharing a cache line they would invalidate
each other on every access. The line the host must WRITE to publish work (`epoch_`) and READ to confirm the workers
are parked (`parked_`) must not be hammered by the very threads waiting for it, so the parked spin carries no
counter (a locked read-modify-write per iteration) and `alignas(64)` stops the four sharing."

## 2. What does not reach the output

The promise (`pool.hpp:54-55`), verbatim: "Every token's output is bitwise the single-token job's." So the thread
count, the thread placement, and the order in which jobs are claimed do not reach the numbers. Four independent
facts make that true, and a port has to reproduce all four:

1. **Per token, inside the kernel.** The accumulators are per token and the block loop order does not depend on the
   token count, so a row's value for a token is the same for every grouping of tokens into jobs
   (`src/kernels/cpu/expert.cpp:503-541`).
2. **Per row, across tasks.** The row ranges partition the expert's rows exactly and at row granularity, and the
   upper bound clamps at the expert boundary, so every (expert, row) pair is computed whole, by exactly one task,
   whatever the task count, the worker count, or the claim order are. **Inference** from `pool.cpp:418-421`,
   `:450`.
3. **The intermediate is quantized serially, from the whole vector.** The re-quantization of the gated activation
   runs on the host thread after the phase's park barrier, over all `FF` values at once (`pool.cpp:557-559`), so
   its chunk scales do not depend on how the phase was split. **Inference.**
4. **The activation is shared, not recomputed.** One quantized activation per token per layer, built by the caller
   and pointed at by every job (`src/core/expert_source.cpp:1588-1598`); the pool never re-quantizes its input.

The only arithmetic the pool itself performs is the SwiGLU at `pool.cpp:434`, which is elementwise and so
split-invariant by construction. It is also the one place a transcription can diverge from basalt, because
`std::exp` is libm and glibc's differs from Apple's: that divergence is per element, not per split, and
`docs/research/basalt-cpu-expf.md` settles it.

## 3. The Metal mapping

### 3.1 The two decode forms

basalt compiles its decode graph **twice** and picks between the two on one flag, `all_resident_`
(`src/core/verify.cpp:876-907`):

* With every expert resident it runs the **zero-doorbell** graph. The whole step is on the GPU and the host
  services nothing.
* When the cache comes up short it runs the **DecMoe** graph, whose window services every MoE layer, every window,
  whether or not that layer has a CPU entry.

This is not a performance detail. Each host handoff idles the GPU for a median of 136 to 137 microseconds as built,
against 0.5 microseconds at a plain pass boundary (`docs/design/flash-next.md`, `probes/p16_host_handoff`) [M]. At
48 MoE layers that is 6.5 to 6.6 milliseconds per step, spent whether or not a layer has any CPU work to do. So a
port that installs a point in every MoE layer unconditionally is not a slow exact port: it is a **different
program** from the one basalt runs in the identity configuration, and it spends that idle to be different.

The program is built per session from the pack and the profile, so the choice is made once at build time, not by a
predicate the step evaluates. `monolith/compiler/cpu_tier.py` is where it lives: `decode_host_points` returns an
empty list when every expert is resident, and one point per MoE layer when the cache is short.

### 3.2 Where the point goes

`HostPoint` (`monolith/runtime/program.py`) carries `task`, `publish`, `wait`, `writes` and `reads`, with the
semantics stated there verbatim: "The task starts once ops [0, publish) of the step have completed (publish 0: once
the previous step completed), and the GPU work from op `wait` on starts after it returns (wait = len(ops): the next
step's). publish == wait makes a sync point. `writes` and `reads` name the buffers the task touches; the ops that
may run beside it, [publish, wait), must not race them (Program.check_host_points)."

Mapping basalt's decode caller, the verify loop's per-layer flag wait (`verify.cpp:2549-2610`):

| basalt | the port |
|---|---|
| the per-layer flag wait, the pool call, the fence, the flag raise | one host point per MoE layer: `publish` after the layer's routing and planning op, `wait` at the combine, so the layer's **resident** experts are the `[publish, wait)` ops that run beside the pool. Without ops in that window the GPU idles for the whole of the task |
| the `seq_cst` fence and `_mm_sfence()` before raising the flag (`verify.cpp:2587-2588`) | not needed at the point, but **only because the pool's own release/acquire pair is kept**. The rows are written by the workers, not by the pump, so the chain is: a worker's stores to its output rows, then its `done_` increment with release (`pool.cpp:472`), then the pump's acquire load in `wait_done` (`:379`), then the task returns, then the pump commits the command buffer that reads them. basalt states the requirement for the host's own copy of it, verbatim (`pool.cpp:603-604`): "The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the device after `run()` returns, so the write must be published, not merely performed." A port that made `done_` a plain counter would break GPU visibility, not just the wait |
| `h_ymiss_`, the mapped staging the pool writes (`verify.cpp:750`) | the point's `writes` names that staging buffer, so `check_host_points` refuses any `[publish, wait)` op that reads it |

The prompt path's shape differs: its window spans the GPU's own expert ops, so the planning that builds the jobs
has to happen at an earlier point or on the host before the step, because the GPU ops in the window read those
tables.

### 3.3 What this port changes, and why none of it is a numeric difference

* **A stall reports instead of aborting.** basalt calls `release_gpu_waits` then `std::abort()`; the port returns
  false with a message, which the host task turns into an error the engine raises. A stalled pool is **poisoned**:
  its workers may still hold pointers into buffers the caller is about to free, so it is never destroyed and never
  reused, which is what `abort()` buys basalt for free.
* **The stall clock is armed once per wait** rather than every 2^32 spins. The transcribed spin body costs 0.422 ns
  here, so 2^32 iterations is 1.8 s against a 60 s limit and the abort could never fire [M].
* **The worker count and the spin-before-sleep come from the chip profile.** basalt derives the count from a Linux
  CPU topology this platform does not expose, and no placement control here is more than a hint.
* **Each worker's scratch is owned by the pool and indexed by worker id**, where basalt uses function-local
  `thread_local` arrays. On Darwin each such access goes through the thread-local-variable descriptor and allocates
  lazily on first use [M]; the buffers are scratch, so this is invisible to the output.
* **The hot atomics are padded to 128 bytes**, which is this chip's cache line, where basalt pads to 64 [M].
* **A second concurrent driver is refused.** basalt's pool has one host thread and needs no such flag; here the
  Python bindings release the GIL and the host task runs on the pump, so two native drivers of one pool are
  reachable, and two of them race on the batch description.

## 4. What is measured, and what rests on the transcription

The property suite (`tests/kernels/test_basalt_pool.py`) asserts that each job of a phase runs exactly once, under
the schedules that break a wrong implementation. It was mutated to find its limits, and the answer is sobering:
removing the epoch test from `claim`, removing either park barrier, removing both, and removing all three together
each leave every case passing, over 4,000 phases of 8 jobs [M]. The windows these protections cover are nanoseconds
wide and a trivial job never schedules into them, and the protections are layered, so no single removal is
observable even in principle there.

ThreadSanitizer (`tools/tsan/`) reads the happens-before edges directly and is the instrument for the rest. It
found the real ordering defect the property suite passed clean over. It does **not** catch weakening the written
memory orders to `relaxed`, which produces no warning and no wrong result over 300 phases of 16 jobs [M]. Those
orders rest on the transcription and on the arguments recorded in section 1.5, not on a test.
