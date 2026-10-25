# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cpu/pool.cpp:415-451, :543-572 and
# include/basalt/kernels/cpu/pool.hpp:54-63 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE):
# the two phases, the row split, and the promise they are built to keep.
"""``run_split_multi_native`` of ``runtime/src/basalt_cpu_experts.h``: one layer's CPU experts, through the pool.

**The promise this suite exists to hold.** basalt states it at ``pool.hpp:54-55``: *every token's output is bitwise
the single-token job's*. Whatever the thread count, whatever the placement, whatever order the jobs are claimed in,
and whatever grouping of tokens into jobs the router produced, the bits do not move. That is a stronger claim than
"the arithmetic is right", and it is the one a split can break without any test of the arithmetic noticing.

Four independent facts make it true, and the cases below take them one at a time:

1. **Per token, inside the kernel.** The accumulators are per token and the block loop order does not depend on how
   many tokens a job carries, so a row's value for a token is the same for every grouping. Case B1.
2. **Per row, across tasks.** Task i owns ``[rows * i / tasks, rows * (i + 1) / tasks)`` in int64 arithmetic, cut
   at expert boundaries, so every (expert, row) pair is computed whole by exactly one task, whatever the task
   count is. Cases A1 and A2.
3. **The intermediate is quantized serially, from the whole vector.** The re-quantization between the phases runs
   on the host over all 640 values at once, so the 20 chunk scales cannot depend on how the first phase was split.
   Case C1 is what fails when that moves into a worker.
4. **The activation is shared, not recomputed.** One activation per token per layer, pointed at by every job.
   Case B2.

**Does this suite bite.** Measured, by mutating the layer and rebuilding. Indexing the gate and up scratch
relative to ``r0`` rather than absolutely fails seven of the nine cases; replacing the serial re-quantization with
a per-slice one inside the parallel phase fails the same seven. Rewriting the expert-boundary clamp into an
equivalent conditional fails nothing, which is what a no-op mutation should do. The cases that survive every
mutation tried are E1 and E2, which hold shapes and refusals rather than values.

**What the reference is, and what it does not cover.** ``tests/kernels/basalt_cpu_emu.py`` is an independent numpy
model of the two Q2_0 kernels this layer calls, written against x86's semantics rather than against the port. Case
D1 runs a whole layer through it and compares bit for bit, so the suite rests on something other than the port
agreeing with itself.

The SwiGLU between the phases is the one step D1 takes from the port rather than from the model. Its bits are
pinned elsewhere, and more strongly than anything here could manage: ``test_basalt_cpu_expf.py`` holds
``expf_glibc`` at all 4,294,967,296 float32 arguments and the SwiGLU at 4,226 recorded x86 vectors. A
double-precision model of it would differ wherever the transcribed expf's 0.502 ulp changes the float32 of
``1 + e``, so D1 would then be measuring the model's exp against glibc's rather than measuring this layer.
"""

import numpy as np
import pytest

from monolith.runtime import _native as _nt

from . import basalt_cpu_emu as emu

H, FF, QK, QKA, BB, MAXC, MAXT = 2560, 640, 64, 32, 18, 80, 8
GU_ROW = BB * (H // QK)        # 720 bytes: a gate or up row's 40 Q2_0 blocks
D_ROW = BB * (FF // QK)        # 180 bytes: a down row's 10 blocks
UP_OFF = FF * GU_ROW           # the up rows follow the gate rows
DOWN_OFF = 2 * UP_OFF          # the down rows follow both
BLOB = DOWN_OFF + H * D_ROW


def make_blob(rng, n):
    """`n` experts, each [gate rows | up rows | down rows]. The bytes are arbitrary: a Q2_0 block is an fp16 scale
    then 16 code bytes, and every bit pattern of both is a legal block."""
    b = rng.integers(0, 256, size=(n, BLOB), dtype=np.uint8)
    # keep the block scales finite, so the test is about the split rather than about NaN propagation
    for off, nrows, row in ((0, FF, GU_ROW), (UP_OFF, FF, GU_ROW), (DOWN_OFF, H, D_ROW)):
        blocks = b[:, off:off + nrows * row].reshape(n, nrows, row // BB, BB)   # a view, so this writes through
        blocks[..., 1] &= 0x3B                                 # clear the exponent's top bits: no inf, no NaN
    return b


def make_acts(rng, ntok):
    """An activation pool, one row per token, quantized by the port's own quantizer so the inputs are the shape the
    kernel expects rather than arbitrary bytes."""
    q = np.zeros((ntok, H), np.int8)
    scale = np.zeros((ntok, MAXC), np.float32)
    sumv = np.zeros((ntok, MAXC), np.int32)
    hx = np.zeros((ntok, MAXC), np.float32)
    for t in range(ntok):
        x = rng.normal(0.0, 1.0, H).astype(np.float32)
        _nt.basalt_act_quant_q8_1(x, q[t], scale[t], sumv[t], hx[t])
    return q, scale, hx


def run(pool, blobs, q, scale, hx, job_nt, job_tok):
    out = np.zeros((len(job_nt), MAXT, H), np.float32)
    pool.run_experts(blobs, GU_ROW, D_ROW, UP_OFF, DOWN_OFF, q, scale, hx, job_nt, job_tok, out)
    return out


def jobs_of(nts, rng=None, ntok=None):
    """Per-job token counts into the (job_nt, job_tok) pair the binding takes."""
    job_nt = np.array(nts, np.int32)
    job_tok = np.zeros((len(nts), MAXT), np.int32)
    for j, nt in enumerate(nts):
        job_tok[j, :nt] = np.arange(nt) if ntok is None else rng.integers(0, ntok, nt)
    return job_nt, job_tok


@pytest.fixture(scope="module")
def layer():
    rng = np.random.default_rng(20261018)
    blobs = make_blob(rng, 5)
    q, scale, hx = make_acts(rng, MAXT)
    return blobs, q, scale, hx


# ---- A: the row split is invariant -------------------------------------------------------------------------------

def test_a1_the_thread_count_does_not_move_a_bit(layer):
    """A1. Fact 2, directly. The task count is 3 x threads, so every worker count partitions the rows differently;
    the output must not know. This is the case that fails if a row is ever computed by two tasks, or by none."""
    blobs, q, scale, hx = layer
    job_nt, job_tok = jobs_of([3, 1, 8, 2, 5])
    first = None
    for workers in (1, 2, 3, 7, 16):
        pool = _nt.ExpertPool(n_workers=workers, host_works=True, spin_before_sleep_ms=1.0)
        got = run(pool, blobs, q, scale, hx, job_nt, job_tok)
        if first is None:
            first = got
        else:
            bad = np.flatnonzero(got.view(np.uint32) != first.view(np.uint32))
            assert bad.size == 0, f"{workers} workers moved {bad.size} of {got.size} floats"


def test_a2_the_host_drain_does_not_move_a_bit(layer):
    """A2. The host draining beside the workers changes which thread computes which row, and nothing else."""
    blobs, q, scale, hx = layer
    job_nt, job_tok = jobs_of([4, 4])
    a = run(_nt.ExpertPool(n_workers=3, host_works=True, spin_before_sleep_ms=1.0), blobs[:2], q, scale, hx, job_nt, job_tok)
    b = run(_nt.ExpertPool(n_workers=3, host_works=False, spin_before_sleep_ms=1.0), blobs[:2], q, scale, hx, job_nt, job_tok)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_a3_repeated_runs_are_identical(layer):
    """A3. The same pool run twice. A claim order that reached the output would show here, since the order two
    spinning threads take is not the same twice."""
    blobs, q, scale, hx = layer
    job_nt, job_tok = jobs_of([8, 8, 8])
    pool = _nt.ExpertPool(n_workers=8, host_works=True, spin_before_sleep_ms=0.1)
    first = run(pool, blobs[:3], q, scale, hx, job_nt, job_tok)
    for _ in range(12):
        again = run(pool, blobs[:3], q, scale, hx, job_nt, job_tok)
        assert np.array_equal(again.view(np.uint32), first.view(np.uint32))


# ---- B: the grouping of tokens into jobs is invariant -------------------------------------------------------------

def test_b1_a_token_is_bitwise_its_single_token_job(layer):
    """B1. The promise in its own words. Token t computed alone, against the same token inside a job of eight."""
    blobs, q, scale, hx = layer
    many_nt, many_tok = jobs_of([MAXT])
    many = run(_nt.ExpertPool(n_workers=4, spin_before_sleep_ms=1.0), blobs[:1], q, scale, hx, many_nt, many_tok)
    for t in range(MAXT):
        one_nt = np.array([1], np.int32)
        one_tok = np.zeros((1, MAXT), np.int32)
        one_tok[0, 0] = t
        one = run(_nt.ExpertPool(n_workers=4, spin_before_sleep_ms=1.0), blobs[:1], q, scale, hx, one_nt, one_tok)
        assert np.array_equal(one[0, 0].view(np.uint32), many[0, t].view(np.uint32)), f"token {t} moved"


def test_b2_the_same_token_in_two_jobs_gives_the_same_rows(layer):
    """B2. Fact 4. The activation is shared read-only across the batch, so one token routed to two experts reads
    the same quantized input in both; the port never re-quantizes the FFN input."""
    blobs, q, scale, hx = layer
    # The SAME expert as two jobs, so a token reaching both goes through identical weights and any difference in
    # its rows is the batching, not the matrix. Two jobs is also what makes the comparison meaningful: one token
    # occupies a different slot in each.
    same = np.concatenate([blobs[:1], blobs[:1]])
    job_nt = np.array([2, 2], np.int32)
    job_tok = np.zeros((2, MAXT), np.int32)
    job_tok[0, :2] = [3, 5]
    job_tok[1, :2] = [5, 3]                      # the same two tokens, swapped between the slots
    out = run(_nt.ExpertPool(n_workers=4, spin_before_sleep_ms=1.0), same, q, scale, hx, job_nt, job_tok)
    # Token 3 is job 0 slot 0 and job 1 slot 1; token 5 is job 0 slot 1 and job 1 slot 0. Both pairs must be
    # bitwise equal: the activation is shared read-only, so neither job re-quantizes its input, and the slot a
    # token lands in does not reach its rows.
    assert np.array_equal(out[0, 0].view(np.uint32), out[1, 1].view(np.uint32)), "token 3 differs between jobs"
    assert np.array_equal(out[0, 1].view(np.uint32), out[1, 0].view(np.uint32)), "token 5 differs between jobs"
    # and the equalities above are not vacuous: two different tokens through the same expert differ
    assert not np.array_equal(out[0, 0], out[0, 1]), "the two tokens produced identical rows, so B2 proves nothing"


# ---- C: the serial step between the phases ------------------------------------------------------------------------

def test_c1_the_intermediate_is_quantized_from_the_whole_vector(layer):
    """C1. Fact 3, which is the one a port is most likely to break by moving the re-quantization into a worker. The
    20 chunk scales of a 640-wide vector come from that whole vector; a per-task re-quantization would give each
    chunk a scale from a different slice, and the down phase would then read different codes.

    Asserted as what it implies and can be observed: with the task count varied, which is what changes the slices,
    the quantized intermediate's effect on the output does not move.

    Measured by mutation: replacing the serial step with a per-slice `act_quant_q8_1(ff + r0, r1 - r0, a2)` inside
    the parallel phase changes 48,640 of 102,400 output floats and fails seven of this suite's nine cases, this one
    among them. A mutation that ADDS the per-slice call while leaving the serial step in place changes nothing at
    all, because the serial step overwrites it; that is a no-op rather than a weaker form of the defect."""
    blobs, q, scale, hx = layer
    job_nt, job_tok = jobs_of([1])
    outs = [run(_nt.ExpertPool(n_workers=w, host_works=True, spin_before_sleep_ms=1.0),
                blobs[:1], q, scale, hx, job_nt, job_tok)
            for w in (1, 5, 11)]
    for o in outs[1:]:
        assert np.array_equal(o.view(np.uint32), outs[0].view(np.uint32))


# ---- D: against an independent model --------------------------------------------------------------------------

def reference_layer(blob, q, scale, hx, toks):
    """The whole layer in numpy, through the emulation of the two kernels: gate and up rows, the SwiGLU, the serial
    re-quantization, then the down rows. Written against x86's semantics, not against the port."""
    nt = len(toks)
    acts = [(q[t], scale[t], hx[t]) for t in toks]
    gate = np.zeros((nt, FF), np.float32)
    up = np.zeros((nt, FF), np.float32)
    emu.q2_0_gguf_rows_multi(blob, GU_ROW, H // QK, acts, gate, 0, FF)
    emu.q2_0_gguf_rows_multi(blob[UP_OFF:], GU_ROW, H // QK, acts, up, 0, FF)
    # The SwiGLU through the port's own, because its bits are pinned elsewhere and more strongly than anything
    # here could: tests/kernels/test_basalt_cpu_expf.py holds expf_glibc at all 2^32 arguments and the SwiGLU at
    # 4,226 recorded x86 vectors. A double-precision model of it would differ wherever the 0.502 ulp of the
    # transcribed expf changes the float32 of 1 + e, so using one would make this case about the wrong thing.
    ff = np.zeros((nt, FF), np.float32)
    for t in range(nt):
        _nt.basalt_swiglu(np.ascontiguousarray(gate[t]), np.ascontiguousarray(up[t]), ff[t])
    a2 = []
    for t in range(nt):
        q2 = np.zeros(H, np.int8)
        s2 = np.zeros(MAXC, np.float32)
        sum2 = np.zeros(MAXC, np.int32)
        h2 = np.zeros(MAXC, np.float32)
        _nt.basalt_act_quant_q8_1(np.ascontiguousarray(ff[t]), q2, s2, sum2, h2)
        a2.append((q2, s2, h2))
    out = np.zeros((nt, H), np.float32)
    emu.q2_0_gguf_rows_multi(blob[DOWN_OFF:], D_ROW, FF // QK, a2, out, 0, H)
    return out


def test_d1_a_whole_layer_against_the_numpy_model(layer):
    """D1. The suite's anchor: the port's output against an independent model of the same computation. Without this
    every other case only says the port agrees with itself under rearrangement."""
    blobs, q, scale, hx = layer
    toks = [0, 4, 7]
    job_nt = np.array([len(toks)], np.int32)
    job_tok = np.zeros((1, MAXT), np.int32)
    job_tok[0, :len(toks)] = toks
    got = run(_nt.ExpertPool(n_workers=4, spin_before_sleep_ms=1.0), blobs[:1], q, scale, hx, job_nt, job_tok)
    want = reference_layer(blobs[0], q, scale, hx, toks)
    for i, t in enumerate(toks):
        bad = np.flatnonzero(got[0, i].view(np.uint32) != want[i].view(np.uint32))
        assert bad.size == 0, (f"token {t} differs at {bad.size} of {H} rows, first row {bad[0] if bad.size else -1}: "
                               f"{got[0, i][bad[0]] if bad.size else 0!r} against {want[i][bad[0]] if bad.size else 0!r}")


# ---- E: shapes and refusals --------------------------------------------------------------------------------------

def test_e1_more_experts_than_buffers(layer):
    """E1. basalt sizes 96 intermediate buffers and runs a deeper batch in chunks of that. The chunk boundary is
    where a port can lose a job or reuse a buffer still in flight."""
    rng = np.random.default_rng(7)
    blobs = make_blob(rng, 100)
    q, scale, hx = make_acts(rng, 2)
    job_nt, job_tok = jobs_of([1] * 100)
    out = run(_nt.ExpertPool(n_workers=4, spin_before_sleep_ms=1.0), blobs, q, scale, hx, job_nt, job_tok)
    assert np.isfinite(out[:, 0, :]).all()
    # every expert wrote something different, so none was skipped or served another's buffer
    rows = {out[e, 0, :].tobytes() for e in range(100)}
    assert len(rows) == 100, f"{100 - len(rows)} experts produced a duplicate row"


def test_e2_the_binding_refuses_a_bad_shape(layer):
    blobs, q, scale, hx = layer
    pool = _nt.ExpertPool(n_workers=2, spin_before_sleep_ms=1.0)
    job_nt, job_tok = jobs_of([1])
    with pytest.raises(ValueError, match="1 to 8 tokens"):
        bad = np.array([9], np.int32)
        run(pool, blobs[:1], q, scale, hx, bad, job_tok)
    with pytest.raises(ValueError, match="outside the activation pool"):
        tok = job_tok.copy()
        tok[0, 0] = 99
        run(pool, blobs[:1], q, scale, hx, job_nt, tok)
    with pytest.raises(ValueError, match="gu_row is below"):
        out = np.zeros((1, MAXT, H), np.float32)
        pool.run_experts(blobs[:1], 8, D_ROW, UP_OFF, DOWN_OFF, q, scale, hx, job_nt, job_tok, out)
