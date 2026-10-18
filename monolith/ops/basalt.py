"""basalt's GEMVs, q8_1 quantizers and Q5_K head (OI-013; docs/design/flash-next.md): one op kind per basalt host
launcher, whose kernels and grids follow from the shape alone, as the launcher picks them. Each op is the launcher's
dispatches of merged kernels (``kernels/common/basalt_*.metal``) in its order: one for a GEMV or a quantizer, two or
more for the Q5_K head's (below); the emitter (``compiler/emit_basalt.py``) takes their macros, grids, threadgroups,
params and scratch sizes from ``monolith/kernels.py``'s builders only, never from the profile's crew or core count.

Values. A weight is the GGUF tensor as shipped, ``[n_out, n_in]`` values: a weight value whose format is the ggml type's
name, or, for BF16 rows, a constant whose dtype is (a ``.basalt`` file's ``dense`` rows; ``packs/basalt.py``), never a
pack's block-lane-major slab. Activations are F32 ``[T, n]``; q8_1 rows are U8 ``[T, n_in / 32 * 36]``, column j's
36-byte blocks after column j - 1's (``basalt_q8_1.metal``). T, the window's columns, is the first output's rows.
Columns are contiguous: an output is exactly ``[T, n_out]``, an input exactly its op's width (basalt's kernels address
column j at j times the width).

Programs. basalt captures one graph per window width (T = 1 .. 8) and bakes T into every launch, so these ops compile T
into their macros and params: one static program per width, sharing the weight windows by name. A dynamic-T program
is refused (it binds StepState, which none of these kernels reads).

* ``basalt_q8_1``, attr ``quantizer``: ``native`` (``native_quantize_q8_1``) ``(x [T, n_in] F32)`` → ``xq``;
  ``swiglu`` (``native_swiglu_quantize_q8_1``) ``(gate, up [T, n_in] F32)`` → the q8_1 rows of their SwiGLU;
  ``ieee`` (``quantize_q8_1_rows``) ``(x)`` → ``xq`` in nvcc's IEEE arithmetic. n_in is a multiple of 32. The native
  and SwiGLU kinds refuse more than 8 columns (basalt's ``validate_shape``); the IEEE rows feed only the routed experts
  (basalt's F4), so any other reader is refused, and they are a buffer of their own: neither a view nor viewed.
* ``basalt_mmvq`` (``native_mmvq``): ``(xq, w [n_out, n_in])`` → ``y [T, n_out] F32``, w of a type of ``MMVQ_TRAITS``.
* ``basalt_dense_tc`` (``dense_tc``): the same values, w of a type of ``DENSE_TC_SB``, n_in a multiple of 512. The
  summation order is the one basalt's variant rule picks on its 170 multiprocessors for a 16-byte-aligned weight
  (``basalt_dense_tc_variant``'s defaults; every basalt weight is a 256-byte-aligned allocation), never the order this
  GPU's core count or a binding's offset would give.
* ``basalt_dense_tc_pair`` (``dense_tc_pair``): ``(xq, w0 [n_out0, n_in], w1 [n_out1, n_in])`` →
  ``(y0 [T, n_out0], y1 [T, n_out1])``, two projections of one input in the tile order. basalt runs it for every GDN
  layer's qkv and z projections whose weights are both 16-byte aligned (``pair_ok``), which every ``.basalt`` binding
  is.
* ``basalt_mmvf`` (``bf16_gemv_fp32_mmvf``, ``bf16_gemv_fp32_mmvf_multi``): ``(x [T, n_in] F32, w [n_out, n_in] bf16)``
  → ``y [T, n_out] F32``; the one-row kernel at T = 1, the multi-row kernel above, which gives every row the one-row
  kernel's bits.

Alignment, checked at every binding: 4 bytes for each q8_1, ``basalt_mmvq`` and ``basalt_dense_tc`` binding, and for
the BF16 GEMV's weight and output; 8 for the BF16 GEMV's activation (read as float2); 16 for both pair weights.
basalt's launchers check most of these (at b4c63b5): every pointer of ``native_mmvq`` and of the native and SwiGLU
quantizers at 4 (``validate_pointer``, ``native_mmvq.cu:1471-1475``, called by each type's launcher and at
``:1551-1552``, ``:1566-1568``), dense_tc's weight and rows at 4 (``dense_tc.cu:1078-1079``), the pair's weights at
16 and its rows at 4 (``:1109-1110``), the one-row BF16 GEMV's activation at 8 and its weight and output at 4
(``native_bf16.cu:167-171``), the multi-row one's activation at 8 (``:142-143``). The port adds the rest, which
basalt's allocations meet unchecked: the IEEE quantizer's (``quantize_q8_1_rows`` checks none,
``iq_kernels.cu:2450-2455``), dense_tc's output, the pair's outputs, the multi-row BF16 GEMV's weight and output.

The Q5_K head (``q5k_head.cu``). Its rows are Q5_K rows as stored: a ``q5_k`` weight ``[rows, n_in]`` (a ``.basalt``
tensor: the verify head's ``output.weight``) or a host input ``u8 [rows, n_in / 256 * 176]`` (``Graph.input(...,
host=True)``: the drafter's head rows, gathered for its draft vocabulary). An op reads the first n_out (n_rows) of them,
as basalt passes a pointer and a row count: its draft head's tier 0 reads the leading ``n_tier1_`` of the gathered
rows, tier 1 all of them (``mtp.cpp:1006``). A map is an i32 value of at least that many ids, negative for a row that
holds no token (basalt's ``dvocab_``).

* ``basalt_q5k_head`` (``q5k_head_mmvq``): ``(xq [T, n_in / 32 * 36], rows[, map])`` → ``y [T, n_out] F32``, the
  logits of rows 0 .. n_out - 1, n_out the output's width; n_in a multiple of 256 of at most 64 blocks and T 1 .. 8
  (``q5k_head_supported``). With a map, ``q5k_head_mask_unmapped`` follows on each column, as basalt masks the draft
  head's logits (``mtp.cpp:1022-1023``): a row whose id is negative becomes -inf.
* ``basalt_q5k_top1`` (``q5k_head_top1``): ``(xq [1, 2880], rows[, map])`` → ``(y [1, n_out] F32, id [1] I32,
  prob [1] F32)``: n_in 2560 and one column, as basalt calls it (``q5k_head.cu:374``; ``mtp.cpp:1016``). The first
  dispatch writes the logits and each threadgroup's partials, the merge the best row (its id through the map) and its
  probability. Both take the grid basalt's rule gives on its 170 multiprocessors (``basalt_q5k_head_grid``), the
  merge as its partial count: the probability sums in that order. Without a map a dummy buffer fills the map's slot.
* ``basalt_q5k_proxy_top1`` (``q5k_proxy_top1``), attrs ``n_rows`` and ``n_cap``: ``(xq [1, 2880], proxy, rows, map)``
  → ``(id [1] I32, prob [1] F32)``: the score over the first n_rows rows (1 <= n_rows <= n_cap) of the 2-bit copy, a
  host input ``u8 [q5k_proxy_bytes(n_cap)]``, then the final dispatch, which rescores the candidates on the Q5_K rows.
  n_in 2560 and one column (``mtp.cpp:1009-1014``); the map is required, as basalt's launcher requires it
  (``q5k_head.cu:518``).

The 2-bit copy's pack (``q5k_proxy_pack``) is no op: basalt runs it outside its graphs, at load over the copy's rows
and again at every flush over the context rows just gathered (``mtp.cpp:680``, ``:1439-1452``). The session issues it
as a direct dispatch and lets it complete before it submits a program that scores the copy; ``gather_rows``, the
id-map copy and ``begin_request``'s reset are the session's too. basalt captures the draft head per tier and width
(``round_exec_[tier][T]``), so each tier is a program of its own at each width, its n_out or n_rows compiled in, every
one binding the same host buffers. A follow-up dispatch (each mask, the merge, the final) reads what the dispatch
before it wrote, which its ``writes`` record declares, so the barrier pass gives it ``barrier_before``.

Alignment, the port's own (basalt's head launchers check only for null pointers): 2 bytes for the Q5_K rows (16-bit
loads), 16 for the 2-bit copy (read as uint4), 4 for every other binding.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_Q8_1 = register_op(OpDef("basalt_q8_1", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_q8_1")))
BASALT_MMVQ = register_op(OpDef("basalt_mmvq", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_mmvq")))
BASALT_MMVF = register_op(OpDef("basalt_mmvf", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_mmvf")))
BASALT_DENSE_TC = register_op(OpDef("basalt_dense_tc", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_dense_tc")))
BASALT_DENSE_TC_PAIR = register_op(OpDef("basalt_dense_tc_pair", OpClass.MAP, "rows")
                                   .bind("*", KernelBinding("basalt_dense_tc_pair")))
BASALT_Q5K_HEAD = register_op(OpDef("basalt_q5k_head", OpClass.MAP, "rows")      # basalt_q5k_head (+ mask_unmapped)
                              .bind("*", KernelBinding("basalt_q5k_head")))
BASALT_Q5K_TOP1 = register_op(OpDef("basalt_q5k_top1", OpClass.REDUCE, "rows")    # basalt_q5k_head_top1 + _merge
                              .bind("*", KernelBinding("basalt_q5k_top1")))
BASALT_Q5K_PROXY_TOP1 = register_op(OpDef("basalt_q5k_proxy_top1", OpClass.REDUCE, "rows")  # proxy_score + _final
                                    .bind("*", KernelBinding("basalt_q5k_proxy_top1")))
