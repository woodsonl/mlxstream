"""basalt's sampler for the target's window (``kernels/common/basalt_sampler.metal``; basalt's
``src/kernels/cuda/sampler.cu``): greedy argmax, the sampled chain (top-k, penalties, top_p, min_p, temperature) and
its Philox draw, bit for bit basalt's.

* ``basalt_sample``: inputs ``(logits [T, V] F32, ctr_base [2] U32, nq [1] U32[, bias [V] F32][, history [T, H]
  I32])`` → outputs ``(token [T] I32, qprob [T] F32)``.
  - ``ctr_base``: the request's 64-bit draw-counter base, low word first; row t draws at ``ctr_base + position + t``
    (StepState's ``position``). 0 in the engine.
  - ``nq``: the probe's queried rows. With flags PROBE, row t < ``nq[0]`` writes to ``qprob[t]`` the probability of its
    query ``pending_tokens[t + 1]`` (basalt's ``sample_tokens_probe``); ``nq`` is read only then.
  - ``bias``, present with attr ``bias=True``: added to every logit first when the request sets flags BIAS. A program
    built without it compiles its kernels with ``BS_BIAS_BOUND`` 0, so a record's BIAS flag has no effect there.
  - ``history``, present with attr ``hist_len = H > 0``: row t's last H tokens, -1-padded, most recent last. A
    penalty window counts its last ``min(penalty_last_n, H)`` slots.
  - Per step, from StepState: the rows (``t_this_step``, in static programs too), the seed (``rng_hi:rng_lo``), the
    counter's ``position`` and the probe's queries.
  - Per request: the op's params record, ``BsSampleParams`` (``kernels.basalt_sampler_params``): ``mode``, flags and
    basalt's SamplerParams fields. The build writes mode 0, the greedy default; a request's writer rewrites it
    through the packer, which refuses what basalt's dispatcher exits on.
  - Attr ``greedy_only=True``: a program that serves mode 0 alone, as basalt's captured window graph does (greedy, no
    history, no bias): no ``bias`` or ``history`` input, and only the cluster pair is emitted, so no dispatch writes
    ``qprob``. Its record stays at mode 0; the other modes need a program built without the attr. The three gated
    dispatches it drops cost 5.6 to 5.7 µs a step at T = 1 and 7.9 to 10.2 µs at T = 8 on the M3 Max, over two runs of
    ``tools/bench/basalt_sampler.py`` [M].
* Dispatch order (every dispatch writes ``token`` after the ones before it, ordered by the barrier pass):
  1. ``bs_greedy_partial`` (8 threadgroups of 1,024 threads per row) then ``bs_greedy_merge``, always: the row's
     lowest id of its largest non-NaN logit above -inf, else 0 (basalt's cluster argmax).
  2. ``bs_greedy_one_block``, which runs only in mode 1.
  3. ``bs_split_part`` then ``bs_split_merge``, which run only in mode 2. When the build chose the one-block path
     (``kernels.basalt_sampler_path``: more than 64 rows, or more than 64 blocks of 4,096 logits),
     ``bs_sample_one_block`` replaces them.
* The ``mode`` word (``kernels.basalt_sampler_mode``, the branch basalt's verify window takes):

  ====  ====================================================================  ======================================
  mode  request                                                               dispatches that run
  ====  ====================================================================  ======================================
  0     not sampled (greedy, or temperature not above 0, NaN included), no    the cluster pair
        penalty window, no bias
  1     greedy, or temperature <= 0, with a penalty window                    the cluster pair, then the one-block
        (``penalty_last_n > 0``: needs ``history``) or a bias                 greedy kernel
  2     sampled (not greedy, temperature > 0), or not greedy with a NaN       the cluster pair, then the sampled path
        temperature and a penalty window or a bias
  ====  ====================================================================  ======================================

  The other dispatches return at their first instruction (``greedy_only`` programs do not emit them).
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_SAMPLE = register_op(OpDef("basalt_sample", OpClass.REDUCE, "span").bind("*", KernelBinding("bs_greedy_partial")))
