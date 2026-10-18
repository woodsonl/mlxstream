"""Single-kernel runs of basalt's sampler family (kernels/common/basalt_sampler.metal) for the GPU tests: one wrapper
per path, each running the path's dispatches in one command buffer on the family's bindings (the slot map at the top
of the .metal file), as the handler emits them.

- The source is ``basalt_sampler_source()`` with StepState for ``t_max = T_MAX + 1`` spliced in after the prelude
  (``pending_tokens`` holds a 70-row probe's queries), compiled with STEP_STATE=1: the kernels read the window's row
  count, position and seed there, as in a program.
- A window of T rows runs the grids of T + 1 rows with ``t_this_step = T``: the extra row's threadgroups must exit at
  the row guard, so its output slot keeps its prefill (asserted).
- Every output starts as ``fill`` bytes (0xFF: an unwritten row reads -1 and matches no id), every scratch as 0xFF;
  each buffer is exactly the size its rows need, so shader validation sees a store past it.
- A wrapper takes the reference module's ``SamplerParams`` (tests/basalt_sampler_ref.py) and packs ``BsSampleParams``
  with the mode its kernels gate on; the seed goes to StepState ``rng_lo`` / ``rng_hi``, the counter to ``ctr_base``.
"""

import struct

import numpy as np

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from monolith.runtime import _native as nt
from tests.kernels import basalt_gpu as gpu

T_MAX = 70                                     # the widest window a test runs (the probe's, sampler_parity.cpp:398)
LAYOUT = StepStateLayout(t_max=T_MAX + 1)
SOURCE = kernels.basalt_sampler_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
AM_CTAS = 8                                    # bs_greedy_partial's threadgroups per row (basalt's kAmCtas)


def pipeline(name):
    return gpu.pipeline(SOURCE, name, (("STEP_STATE", "1"),))


def threads(pso, n=1024):
    """basalt's ``n`` threads, or the widest multiple of 32 the pipeline allows (the kernel's result does not depend
    on it)."""
    return min(n, pso.max_threads_per_threadgroup // 32 * 32)


def record(p, vocab, rows, hist_len, mode, bias=False, probe=False):
    """``BsSampleParams`` for ``p`` (SamplerParams) on a window of ``rows`` grid rows."""
    return kernels.basalt_sampler_params(
        vocab, rows, -(-vocab // kernels.BS_BLOCK), hist_len, mode=mode, bias=bias, probe=probe, top_k=p.top_k,
        top_p=p.top_p, min_p=p.min_p, temperature=p.temperature, min_keep=p.min_keep, penalty_last_n=p.penalty_last_n,
        penalty_repeat=p.penalty_repeat, penalty_freq=p.penalty_freq, penalty_present=p.penalty_present)


def launch(steps, logits, p, *, mode, hist=None, hist_len=0, bias=None, position=0, fill=0xFF, scratch=None,
           lbuf=None):
    """Run ``steps``, (pipeline, grid, threads) in order, on one window: ``logits`` (T, V) float32, ``p`` the request,
    ``hist`` the flat (T, hist_len) int32 history rows, ``bias`` (V,) float32, ``position`` StepState's.
    ``scratch``: {slot: bytes}, each starting as 0xFF. ``lbuf``: a buffer already holding ``logits``, bound in place
    of a fresh upload (sampler_parity's DeviceRows, rows uploaded once and sampled many times: the test reads it back).
    Every slot the family binds is bound, a 16-byte placeholder where the request has no such datum. Returns the T
    picks and the T probabilities' bits; asserts the extra row of each output untouched."""
    l = np.ascontiguousarray(logits, np.float32)
    T, V = l.shape
    rows = T + 1
    dev = gpu.device()

    def buf(data, prefill=None):
        b = nt.Buffer(dev, data)
        if prefill is not None:
            b.fill(prefill)
        return b

    st = LAYOUT.pack({"t_this_step": T, "position": position, "rng_lo": p.seed & 0xFFFFFFFF,
                      "rng_hi": (p.seed >> 32) & 0xFFFFFFFF})
    slots = {0: buf(l.tobytes()) if lbuf is None else lbuf, 1: buf(struct.pack("<Q", p.counter & (2 ** 64 - 1))),
             2: buf(bytes(16)),
             3: buf(bytes(16) if bias is None else np.ascontiguousarray(bias, np.float32).tobytes()),
             4: buf(bytes(16) if hist is None else np.ascontiguousarray(hist, np.int32).tobytes()),
             5: buf(4 * rows, fill), 6: buf(4 * rows, fill),
             7: buf(record(p, V, rows, hist_len, mode, bias is not None)), 15: buf(st)}
    slots.update({k: buf(n, 0xFF) for k, n in (scratch or {}).items()})
    ds = []
    for pso, grid, tg in steps:
        d = nt.Dispatch().pipeline(pso)
        for k, b in slots.items():
            d = d.buffer(k, b)
        ds.append(d.grid(*grid).threadgroup(tg))
    r = nt.Queue(dev).run(ds)
    assert not r.error, r.error
    token = np.frombuffer(slots[5].read(0, 4 * rows), np.int32)
    qbits = np.frombuffer(slots[6].read(0, 4 * rows), np.uint32)
    unset = np.frombuffer(bytes([fill]) * 4, np.uint32)[0]
    assert token.view(np.uint32)[T] == unset and qbits[T] == unset, "the row past t_this_step was written"
    return token[:T].copy(), qbits[:T].copy()


def greedy_cl(logits, p, *, hist=None, hist_len=0, position=0, fill=0xFF, lbuf=None):
    """G-CL, basalt's sampler_greedy_cluster_kernel as two dispatches: ``bs_greedy_partial`` over (8, rows)
    threadgroups of 1,024 threads, each storing its (key, id) pair, then ``bs_greedy_merge``, one SIMD-group per row
    over its 8 pairs. Mode 0, the greedy request without a penalty window or a bias; a history, if given, is bound and
    never read. Returns the T picks."""
    part, merge = pipeline("bs_greedy_partial"), pipeline("bs_greedy_merge")
    rows = len(logits) + 1
    return launch([(part, (AM_CTAS, rows), threads(part)), (merge, (rows,), 32)], logits, p, mode=0, hist=hist,
                  hist_len=hist_len, position=position, fill=fill, scratch={8: rows * AM_CTAS * 8}, lbuf=lbuf)[0]


def greedy_1b(logits, p, *, hist=None, hist_len=0, bias=None, position=0, fill=0xFF, lbuf=None):
    """G-1B, basalt's sampler_greedy_kernel: ``bs_greedy_one_block`` alone, one threadgroup of up to 1,024 threads
    per row, in mode 1 whatever the request asks (the forced runs; the engine runs it after the cluster pair, which
    would hide a G-1B that never wrote). Its bitmap scratch is sized as basalt sizes its shared one: ceil(V / 32)
    words per row when the request has a window (penalty_last_n > 0 and a history), else a 16-byte placeholder, so
    validation sees a bitmap store the request did not size. Returns the T picks."""
    pso = pipeline("bs_greedy_one_block")
    T, V = np.shape(logits)
    rows = T + 1
    bits = rows * -(-V // 32) * 4 if p.penalty_last_n > 0 and hist_len > 0 else 16
    return launch([(pso, (rows,), threads(pso))], logits, p, mode=1, hist=hist, hist_len=hist_len, bias=bias,
                  position=position, fill=fill, scratch={10: bits}, lbuf=lbuf)[0]


def rows_alone(path, logits, p, *, hist=None, hist_len=0, position=0):
    """W.1, invariant 1 (a row's output never depends on the window width T or its place in it): ``path``'s window
    against each row t run alone, a one-row window of its logits and history row at ``position + t``. Returns the
    rows whose outputs (the picks; and the probabilities' bits, for a path that returns them) differ bitwise."""
    run = globals()[path]

    def outs(r):
        return [np.asarray(a).view(np.uint32) for a in (r if isinstance(r, tuple) else (r,))]

    batch = outs(run(logits, p, hist=hist, hist_len=hist_len, position=position))
    bad = []
    for t in range(len(logits)):
        h = None if hist is None else np.asarray(hist)[t * hist_len:(t + 1) * hist_len]
        one = outs(run(logits[t:t + 1], p, hist=h, hist_len=hist_len, position=position + t))
        if any(b[t] != o[0] for b, o in zip(batch, one)):
            bad.append(t)
    return bad
