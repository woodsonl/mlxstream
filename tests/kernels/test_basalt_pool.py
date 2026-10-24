# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/pool.cpp and include/basalt/kernels/cpu/pool.hpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): the properties the pool's claim protocol and
# its park barriers are built to give.
"""``ExpertPool`` of ``runtime/src/basalt_pool.h``, the worker pool that runs basalt's CPU expert tier.

**What this suite can and cannot pin.** Nothing bitwise depends on the pool: a job writes its own rows and the host
sums them in a fixed order, so the thread count, the placement and the order jobs are claimed in do not reach the
output (``docs/research/basalt-cpu-pool.md`` section 2). What the pool has to reproduce is its SYNCHRONISATION, and
getting that wrong does not give different numbers, it gives a run that hangs or computes a job twice. So this
suite asserts properties rather than values, and it asserts them under the schedules that break a wrong
implementation rather than under a quiet one.

The property every case rests on: **each job of a phase runs exactly once**. ``count_phase`` has job j do
``out[j] += 1``, so after k phases every element is k, and a job claimed twice or dropped shows up as an element
that is not.

Cases. **A**, one phase: A1 every job runs once, A2 the work is spread over the workers, A3 an empty phase, A4 a
phase with more jobs than workers. **B**, the claim protocol across batches, which is what the epoch in the batch
word exists for: B1 many phases back to back, B2 phases spaced so the workers sleep between them, B3 phases spaced
so some workers sleep and some do not, which is the schedule the header says "comes round constantly". **C**, the
pool's own guards: C1 the owner claim, C2 the batch-size refusal, C3 a stalled pool poisons itself and refuses
reuse. **D**, shapes: D1 a single worker, D2 without the host working, D3 the phase timers.

**The sleep boundary is the point.** A worker that went to sleep is still counted as parked when it wakes, so for a
moment after it has seen a new epoch the host believes it is idle. Group B spaces its phases around
``spin_before_sleep_ms`` so that moment keeps coming round.

**What this suite does NOT catch, measured.** It was mutated to find out, and the answer is sobering: removing the
epoch test from ``claim``, removing either park barrier, removing both, and removing all three together each leave
every case passing, over 4,000 phases of 8 jobs [M]. The windows these protections cover are nanoseconds wide and
a trivial job never schedules into them. Two things follow. The protections are layered, so no single removal is
observable even in principle here: with the park barriers in place a late claim cannot happen whatever the epoch
does, and with the epoch in place a late claim fails whatever the barriers do. And a property test is the wrong
instrument for the rest: ``tools/tsan/pool_tsan.cpp`` runs the same pool under ThreadSanitizer, which reads the
happens-before edges directly and found a real ordering defect this suite passed clean over (the batch description
was read before the claim that orders it). Weakening the written memory orders to ``relaxed`` is caught by neither
[M]; those rest on the transcription and the argument recorded with them, not on a test.
"""

import multiprocessing
import queue
import threading
import time

import numpy as np
import pytest

from monolith.runtime import _native as _nt

I32 = np.int32


def pool(**kw):
    """A pool with a short spin by default, so the sleep path is reachable inside a test's patience."""
    kw.setdefault("spin_before_sleep_ms", 2.0)
    return _nt.ExpertPool(**kw)


def run_phase(p, n):
    """One phase of n jobs. Returns (out, who, times)."""
    out = np.zeros(n, I32)
    who = np.full(n, -99, I32)
    times = p.count_phase(out, who)
    return out, who, times


# ---- A: one phase ------------------------------------------------------------------------------------------------

def test_a1_every_job_runs_exactly_once():
    """A1. The property the whole pool is built to give. A job claimed twice leaves a 2, a job dropped leaves a 0,
    and either is a claim protocol that does not work."""
    p = pool()
    out, who, _ = run_phase(p, 64)
    assert list(out) == [1] * 64, f"{int((out != 1).sum())} of 64 jobs did not run exactly once"
    assert (who >= -1).all() and (who < p.workers).all()


def test_a2_the_work_reaches_more_than_one_thread():
    """A2. A pool that ran every job on the host would pass A1. The jobs are claimed, not assigned, so which thread
    takes which is not fixed; what is asserted is only that more than one took part, over enough jobs that a pool
    distributing work at all cannot fail it."""
    p = pool(n_workers=4)
    seen = set()
    for _ in range(20):
        _, who, _ = run_phase(p, 256)
        seen.update(int(w) for w in who)
    assert len(seen) > 1, f"every job ran on one thread, {seen}"


def test_a3_an_empty_phase():
    """A3. Zero jobs is a phase that publishes a batch nothing can claim. It must still cross both park barriers
    and return, rather than waiting for a completion that cannot come."""
    p = pool()
    out, _, _ = run_phase(p, 0)
    assert len(out) == 0


def test_a4_more_jobs_than_workers():
    """A4. The index field of the batch word is what hands out jobs, so a phase deeper than the worker count is the
    case where it is actually used."""
    p = pool(n_workers=2)
    out, _, _ = run_phase(p, 1000)
    assert list(out) == [1] * 1000


# ---- B: the claim protocol across batches --------------------------------------------------------------------

def test_b1_many_phases_back_to_back():
    """B1. Phases with no gap, so the workers never reach the sleep and the park barrier is the only thing keeping
    one batch out of the next. The epoch in the batch word is what makes a late worker's claim fail."""
    p = pool()
    n, phases = 48, 200
    out = np.zeros(n, I32)
    who = np.full(n, -99, I32)
    for _ in range(phases):
        p.count_phase(out, who)
    assert list(out) == [phases] * n, f"after {phases} phases the counts are not all {phases}"


def test_b2_phases_spaced_so_every_worker_sleeps():
    """B2. A gap longer than the spin, so every worker is inside the condition variable when the next batch is
    published. This is the lost-wakeup path: the host bumps the epoch and then reads `sleepers_`, a worker
    increments `sleepers_` and then reads the epoch, and both sides are seq_cst so one of them sees the other. A
    port that weakened either side hangs here rather than returning a wrong number."""
    p = pool(spin_before_sleep_ms=1.0)
    n, phases = 16, 40
    out = np.zeros(n, I32)
    who = np.full(n, -99, I32)
    for _ in range(phases):
        time.sleep(0.004)                       # four times the spin, so the sleep is reached
        p.count_phase(out, who)
    assert list(out) == [phases] * n


def test_b3_phases_spaced_at_the_sleep_boundary():
    """B3. The schedule the header calls out: "tiny batches finish before a sleeper is awake, and that moment comes
    round constantly". The gap straddles the spin, so on any given phase some workers are spinning and some are
    inside the condition variable, which is the mix in which a late claim would be possible."""
    p = pool(spin_before_sleep_ms=1.0)
    n, phases = 8, 300
    out = np.zeros(n, I32)
    who = np.full(n, -99, I32)
    for i in range(phases):
        time.sleep(0.0004 + (i % 5) * 0.0004)   # 0.4 to 2.0 ms, straddling the 1 ms spin
        p.count_phase(out, who)
    assert list(out) == [phases] * n, f"{int((out != phases).sum())} of {n} counts are wrong"


# ---- C: the pool's own guards ----------------------------------------------------------------------------------

def test_c1_one_owner_at_a_time():
    """C1. Two distinct host tasks could otherwise drive one pool and overlap their phases, which no barrier in the
    pool protects against. The claim is a compare-exchange, as a task's runner claim is."""
    p = pool()
    a, b = 0x1000, 0x2000
    assert p.claim_owner(a)
    assert p.claim_owner(a)                     # the same owner may re-claim
    assert not p.claim_owner(b)
    p.release_owner(b)                          # a release by a non-owner changes nothing
    assert not p.claim_owner(b)
    p.release_owner(a)
    assert p.claim_owner(b)


def test_c2_a_batch_too_large_is_refused():
    """C2. The batch word carries the job count in 16 bits, so a larger batch would overflow into the epoch. basalt
    aborts here; this returns the refusal."""
    p = pool()
    out = np.zeros(0x10000, I32)
    who = np.zeros(0x10000, I32)
    with pytest.raises(ValueError, match="0 to 65535"):
        p.count_phase(out, who)


@pytest.mark.slow
def test_c3_a_stalled_pool_poisons_itself():
    """C3. A wait that sees no completion for kStall is a bug. basalt prints, releases the GPU waits and aborts;
    this reports instead, and the pool is poisoned: its workers may still hold pointers into buffers the caller is
    about to free, so it is never destroyed and never reused.

    The stall limit is 60 seconds, so this case costs the kernels tier about 70 of them. It carries the ``slow``
    marker for a run that wants to deselect it, and the tier does not: it runs, and it is the only case that
    proves the limit is armed at all, which matters because the transcribed spin wraps its clock every 1.8 s if
    armed as basalt arms it."""
    p = pool()
    ok, err = p.stall_phase(70.0)
    assert not ok and "stalled" in err
    assert p.poisoned
    out, who = np.zeros(4, I32), np.zeros(4, I32)
    with pytest.raises(ValueError, match="cannot be reused"):
        p.count_phase(out, who)
    globals().setdefault("_poisoned_pools", []).append(p)   # never destroyed: a worker still holds its body


# ---- D: shapes --------------------------------------------------------------------------------------------------

def test_d1_a_single_worker():
    """D1. One worker and the host. The park barrier and the claim still have to work with nothing to race."""
    p = pool(n_workers=1)
    assert p.workers == 1
    out, _, _ = run_phase(p, 100)
    assert list(out) == [1] * 100


def test_d2_without_the_host_working():
    """D2. basalt's host drains beside the workers unless --no-host-worker. With it off every job must still be
    claimed, by a worker, and the host waits."""
    p = pool(n_workers=2, host_works=False)
    assert not p.host_works
    out, who, _ = run_phase(p, 200)
    assert list(out) == [1] * 200
    assert (who >= 0).all(), "the host drained a job although host_works is off"


def test_d3_the_phase_timers():
    """D3. A slow pool poses one question, slow at the work or slow at the synchronisation, and only these answer
    it. What is asserted is what the timers are: three non-negative wall-clock milliseconds that together account
    for the phase, each measured around one of its three stages. Their ORDER is not a property - the first phase
    after construction spends longer waiting for the workers to reach the park than it spends on the work, which
    is exactly the kind of thing the timers exist to show."""
    p = pool()
    t0 = time.perf_counter()
    before, work, after = run_phase(p, 512)[2]
    wall_ms = (time.perf_counter() - t0) * 1000.0
    for t in (before, work, after):
        assert np.isfinite(t) and t >= 0.0
    total = before + work + after
    assert total <= wall_ms + 1.0, f"the three timers total {total:.3f} ms over a {wall_ms:.3f} ms phase"
    assert total >= wall_ms * 0.5, f"the three timers account for only {total:.3f} ms of a {wall_ms:.3f} ms phase"


def test_the_pool_refuses_a_mismatched_pair():
    p = pool()
    with pytest.raises(ValueError, match="as many ints"):
        p.count_phase(np.zeros(8, I32), np.zeros(7, I32))
    with pytest.raises(TypeError):
        p.count_phase(np.zeros(8, np.int64), np.zeros(8, np.int64))


def test_the_binding_refuses_what_the_equal_length_check_does_not_cover():
    """Equal lengths do not make two arrays distinct. The same array for both would let the worker-id write
    replace the count, and a shifted view of one buffer would let two jobs write one element at once; neither is
    a length error, and `.noconvert()` prevents a conversion copy rather than aliasing."""
    p = pool()
    a = np.zeros(8, I32)
    with pytest.raises(ValueError, match="share storage"):
        p.count_phase(a, a)
    b = np.zeros(9, I32)
    with pytest.raises(ValueError, match="share storage"):
        p.count_phase(b[:8], b[1:])
    p.count_phase(np.zeros(8, I32), np.zeros(8, I32))              # two distinct buffers are fine


def test_the_binding_refuses_an_unusable_spin():
    """The spin crosses into C++ as microseconds. A NaN, an infinity, or a value past the integer's range makes
    that conversion undefined rather than merely wrong, so it is refused before it happens."""
    for bad in (float("nan"), float("inf"), -1.0, 1e18):
        with pytest.raises(ValueError, match="finite"):
            _nt.ExpertPool(n_workers=1, spin_before_sleep_ms=bad)
    _nt.ExpertPool(n_workers=1, spin_before_sleep_ms=0.0)          # zero is a pool that never spins


def test_many_pools_live_and_die():
    """Construction and teardown are their own risk: the destructor sets the stop flag and bumps the epoch so a
    parked worker notices rather than sleeping through it, then joins. A worker that missed the bump would hang
    the interpreter here rather than fail a test."""
    for _ in range(20):
        p = pool(n_workers=3)
        run_phase(p, 32)
        del p


def _drive_from_a_thread(q):
    """Run in a child process: a pool driven from a thread that did not construct it, reporting the counts back."""
    import numpy as _np
    from monolith.runtime import _native as _n

    p = _n.ExpertPool(n_workers=0, host_works=True, spin_before_sleep_ms=2.0)
    out = _np.zeros(64, _np.int32)
    who = _np.full(64, -99, _np.int32)
    err = []

    def drive():
        try:
            for _ in range(20):
                p.count_phase(out, who)
        except Exception as e:                   # noqa: BLE001 - carried back through the queue below
            err.append(repr(e))

    t = threading.Thread(target=drive)
    t.start()
    t.join()
    q.put((list(out), err))


def test_a_pool_driven_from_another_thread():
    """The pump thread is whatever Python thread called Engine.run and can differ between runs, so the pool must
    work from a thread that did not construct it.

    In a child process, because the failure this guards against is a hang. A thread that never returns would
    outlive `join(timeout=...)` in this process and keep the whole test run alive after the assertion failed;
    a child can be terminated, so the deadline is real."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_drive_from_a_thread, args=(q,))
    proc.start()
    try:
        counts, err = q.get(timeout=120)
    except queue.Empty:
        proc.terminate()
        proc.join(10)
        pytest.fail("the pool did not finish when driven from another thread")
    proc.join(30)
    assert not err, err
    assert counts == [20] * 64
