"""Single-kernel runs of basalt's sampler family (kernels/common/basalt_sampler.metal) for the GPU tests: one wrapper
per path, each running the path's dispatches in one command buffer on the family's bindings (the slot map at the top
of the .metal file), as the handler emits them; ``split_part`` runs the split path's first stage alone and returns
its block lists.

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
from tests.basalt_sampler_ref import sampled_k
from tests.kernels import basalt_gpu as gpu

T_MAX = 70                                     # the widest window a test runs (the probe's, sampler_parity.cpp:398)
LAYOUT = StepStateLayout(t_max=T_MAX + 1)
SOURCE = kernels.basalt_sampler_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
AM_CTAS = 8                                    # bs_greedy_partial's threadgroups per row (basalt's kAmCtas)
TOP_PROB_THREADS = 1024                        # bs_greedy_top_prob's threadgroup, exactly (basalt's kAmThreads)
SPLIT_THREADS = 128                            # bs_split_part's threadgroup, exactly (basalt's kSplitWarps * 32)


def pipeline(name):
    return gpu.pipeline(SOURCE, name, (("STEP_STATE", "1"),))


def threads(pso, n=1024):
    """basalt's ``n`` threads, or the widest multiple of 32 the pipeline allows (the kernel's result does not depend
    on it)."""
    return min(n, pso.max_threads_per_threadgroup // 32 * 32)


def record(p, vocab, rows, hist_len, mode, bias=False, probe=False, id_map=False):
    """``BsSampleParams`` for ``p`` (SamplerParams) on a window of ``rows`` grid rows."""
    return kernels.basalt_sampler_params(
        vocab, rows, -(-vocab // kernels.BS_BLOCK), hist_len, mode=mode, bias=bias, probe=probe, id_map=id_map,
        top_k=p.top_k, top_p=p.top_p, min_p=p.min_p, temperature=p.temperature, min_keep=p.min_keep,
        penalty_last_n=p.penalty_last_n, penalty_repeat=p.penalty_repeat, penalty_freq=p.penalty_freq,
        penalty_present=p.penalty_present)


def launch(steps, logits, p, *, mode, hist=None, hist_len=0, bias=None, id_map=None, position=0, fill=0xFF,
           scratch=None, lbuf=None, queue=None):
    """Run ``steps``, (pipeline, grid, threads) in order, on one window: ``logits`` (T, V) float32, ``p`` the request,
    ``hist`` the flat (T, hist_len) int32 history rows, ``bias`` (V,) float32, ``id_map`` (V,) int32 (flags MAP),
    ``position`` StepState's.
    ``scratch``: {slot: a byte count, a fresh buffer starting as 0xFF; or an ``nt.Buffer`` the caller holds, to read
    back}. ``lbuf``: a buffer already holding ``logits``, bound in place of a fresh upload (sampler_parity's
    DeviceRows, rows uploaded once and sampled many times: the test reads it back). ``queue``: the ``nt.Queue`` to
    run on (basalt's stream), else a new one.
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
             7: buf(record(p, V, rows, hist_len, mode, bias is not None, id_map=id_map is not None)),
             11: buf(bytes(16) if id_map is None else np.ascontiguousarray(id_map, np.int32).tobytes()), 15: buf(st)}
    slots.update({k: n if isinstance(n, nt.Buffer) else buf(n, 0xFF) for k, n in (scratch or {}).items()})
    ds = []
    for pso, grid, tg in steps:
        d = nt.Dispatch().pipeline(pso)
        for k, b in slots.items():
            d = d.buffer(k, b)
        ds.append(d.grid(*grid).threadgroup(tg))
    r = (nt.Queue(dev) if queue is None else queue).run(ds)
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


def top_prob(logits, p, *, id_map=None, hist=None, hist_len=0, position=0, fill=0xFF, lbuf=None):
    """basalt's sampler_greedy_top_prob_cluster_kernel (the drafter's greedy tail) as two dispatches:
    ``bs_greedy_partial``, as in ``greedy_cl``, then ``bs_greedy_top_prob``, one threadgroup of exactly 1,024 threads
    per row: the merge, the pick's probability and, with ``id_map`` (V,) int32, the map. The request's parameters
    are not read (the kernel has no mode gate); a history, if given, is bound and never read. Refuses a pipeline that
    cannot run 1,024 threads, whose sums would not be basalt's. Returns the T picks and the probabilities' bits."""
    part, prob = pipeline("bs_greedy_partial"), pipeline("bs_greedy_top_prob")
    if prob.max_threads_per_threadgroup < TOP_PROB_THREADS:
        raise RuntimeError(f"bs_greedy_top_prob runs at most {prob.max_threads_per_threadgroup} threads per "
                           f"threadgroup; its probability needs exactly {TOP_PROB_THREADS}")
    rows = len(logits) + 1
    return launch([(part, (AM_CTAS, rows), threads(part)), (prob, (rows,), TOP_PROB_THREADS)], logits, p, mode=0,
                  hist=hist, hist_len=hist_len, id_map=id_map, position=position, fill=fill,
                  scratch={8: rows * AM_CTAS * 8}, lbuf=lbuf)


def split_part(logits, p, *, hist=None, hist_len=0, bias=None, position=0, fill=0xFF, lbuf=None):
    """S-SPLIT's first stage alone, basalt's sampler_split_part_kernel: ``bs_split_part`` over (n_blocks, rows)
    threadgroups of exactly 128 threads, in mode 2 whatever the request asks, each storing its 4,096-logit block's
    top-k list (k = sampled_k(top_k, V)) to ``cand`` at (t * n_blocks + b) * k as (id, value bits). ``cand`` holds
    exactly the rows' lists, starting as 0xFF bytes. Returns the T rows' lists, (T, n_blocks, k, 2) uint32; asserts
    the extra row's untouched."""
    pso = pipeline("bs_split_part")
    T, V = np.shape(logits)
    rows, nb, k = T + 1, -(-V // kernels.BS_BLOCK), sampled_k(p.top_k, V)
    cand = nt.Buffer(gpu.device(), rows * nb * k * 8)
    cand.fill(0xFF)
    launch([(pso, (nb, rows), SPLIT_THREADS)], logits, p, mode=2, hist=hist, hist_len=hist_len, bias=bias,
           position=position, fill=fill, scratch={9: cand}, lbuf=lbuf)
    lists = np.frombuffer(cand.read(0, cand.nbytes), np.uint32).reshape(rows, nb, k, 2)
    assert np.all(lists[T] == 0xFFFFFFFF), "the row past t_this_step was written"
    return lists[:T].copy()


def split(logits, p, *, hist=None, hist_len=0, bias=None, position=0, fill=0xFF, lbuf=None, queue=None):
    """S-SPLIT, basalt's sampler_split_part_kernel then sampler_split_merge_kernel: ``bs_split_part`` as in
    ``split_part``, then ``bs_split_merge``, one SIMD-group per row, which merges the row's block lists from ``cand``
    and runs the sampled tail. Mode 2 whatever the request asks, without the cluster pair the engine runs first (it
    would hide a row the split never wrote); ``cand`` holds exactly the rows' lists. The probe is off, so ``qprob``
    must keep its prefill (asserted). Returns the T picks."""
    part, merge = pipeline("bs_split_part"), pipeline("bs_split_merge")
    T, V = np.shape(logits)
    rows, nb = T + 1, -(-V // kernels.BS_BLOCK)
    picks, qbits = launch([(part, (nb, rows), SPLIT_THREADS), (merge, (rows,), 32)], logits, p, mode=2, hist=hist,
                          hist_len=hist_len, bias=bias, position=position, fill=fill,
                          scratch={9: rows * nb * sampled_k(p.top_k, V) * 8}, lbuf=lbuf, queue=queue)
    assert np.all(qbits == np.frombuffer(bytes([fill]) * 4, np.uint32)[0]), "qprob was written without the probe"
    return picks


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
