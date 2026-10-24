"""basalt's GEMVs, q8_1 quantizers and routed and drafter experts (OI-013) and hyper-connection write (OI-005;
docs/design/flash-next.md): one op kind per basalt host launcher, whose kernels and grids follow from the shape alone,
as the launcher picks them. Each op is the launcher's dispatches of merged kernels (``kernels/common/basalt_*.metal``)
in its order: one for a GEMV, a quantizer or the write, three or four for the experts (below); the emitter
(``compiler/emit_basalt.py``) takes their macros, grids, threadgroups, params and scratch sizes from
``monolith/kernels.py``'s builders only, never from the profile's crew or core count.

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
* ``basalt_gr_post`` (``native_gr_post``): ``(r [T, hc * n_embd], block_out [T, n_embd], inject [T, hc] F32)`` →
  ``r_out [T, hc * n_embd] F32``, ``r_out = r + block_out * 2 sigmoid(inject / hc)`` per stream in basalt's fast-math
  rounding; n_embd and hc from ``block_out`` and ``inject``; any T >= 1 (the write has no width bound); ``r_out`` a
  fresh value whose bytes overlap none of the inputs.

Alignment, checked at every binding: 4 bytes for each q8_1, ``basalt_mmvq``, ``basalt_dense_tc`` and ``basalt_gr_post``
binding, and for the BF16 GEMV's weight and output; 8 for the BF16 GEMV's activation (read as float2); 16 for both pair
weights. basalt's launchers check most of these (at b4c63b5): every pointer of ``native_mmvq`` and of the native and
SwiGLU quantizers at 4 (``validate_pointer``, ``native_mmvq.cu:1471-1475``, called by each type's launcher and at
``:1551-1552``, ``:1566-1568``), dense_tc's weight and rows at 4 (``dense_tc.cu:1078-1079``), the pair's weights at 16
and its rows at 4 (``:1109-1110``), the one-row BF16 GEMV's activation at 8 and its weight and output at 4
(``native_bf16.cu:167-171``), the multi-row one's activation at 8 (``:142-143``), every pointer of ``native_gr_post``
at 4 (``check_pointer``, ``native_gr_postops.cu:75-78``, called at ``:110``). The port adds the rest, which basalt's
allocations meet unchecked: the IEEE quantizer's (``quantize_q8_1_rows`` checks none, ``iq_kernels.cu:2450-2455``),
dense_tc's output, the pair's outputs, the multi-row BF16 GEMV's weight and output.

The experts (``basalt_experts.metal``, ``basalt_s2_experts.metal``) run basalt's plan: ``n_groups[0]`` groups; group g's
expert blob starts at ``grp_ptr[g]``, a 64-bit GPU address (an I64 table), and owns the entries ``[grp_start[g],
grp_start[g + 1])``; entry e reads token ``ent_tok[e]``'s activation row and writes row ``ent_dst[e]`` of the output.
The tables are 1-D I32 values but for grp_ptr. The kernels reach the blobs only through those addresses, so each op's
gate/up and down dispatches name the buffers the blobs live in in ``meta["indirect_reads"]`` (``runtime/program.py``):
the barrier pass sees the reads, and ``Engine`` refuses a name that is no program buffer and makes each one resident.
T, the window's columns, is here the activation input's rows (an expert op's first output holds its entries). The
launchers take the plan's capacities as counts (cap_groups, cap_entries), which give the grids and the scratch;
basalt's calls pass both as T x K, the call's routed entries, K = 10 a token (``verify.cpp:1744``, ``:1764``;
``mtp.cpp:931-935``). The routed op derives them so from its attr ``k_per_tok``: its plan is the session's, with no
count the handler could check them against. The drafter's experts take them as attrs, checked against its device
plan's ids.

* ``basalt_experts`` (``native_expert_grouped``, Q2_0 gate/up and Q2_0 down, this pack's ``experts.layout``), attrs
  ``n_ff``, ``k_per_tok`` and ``blobs``: ``(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, xq [T, n_embd / 32 *
  36])`` → ``out [rows, n_embd] F32``, three dispatches (gate/up, SwiGLU + q8_1, down) with cap_groups = cap_entries =
  cap = T x k_per_tok, basalt's kernel choice for the shape (``basalt_expert_macros``) and its threadgroup rows,
  min(cap, 24) (its decode call leaves ``gy`` at 0, ``verify.cpp:1794``). xq is the IEEE q8_1 of the FFN input, a
  ``basalt_q8_1`` op's ``ieee`` rows (basalt's F4, ``verify.cpp:1731``). The tables hold the capacity (cap addresses,
  cap + 1 starts, cap destinations and tokens, at least one count) and out cap rows: an entry's destination
  is its routing index (basalt's ``resident_plan``, ``verify_kernels.cu:1002-1005``). ``blobs`` names the buffers the
  addresses point into: a graph value of at least one blob (its buffer; a ``.basalt`` file's ``experts`` section
  through its window, in tests) or a buffer the session allocates (the expert cache), which the program declares with
  role ``host`` and at least one blob's bytes, so that ``Engine`` requires the session to pass it. The plan is its
  producer's (basalt's ``resident_plan``, not ported yet; until then a test supplies it): groups in first-appearance
  order, their entries in routing order, each group's blob 2-byte aligned and wholly inside one named buffer (never
  split across buffers), ``grp_start`` from 0 and nondecreasing, at most cap groups and cap entries, each ``ent_tok``
  < T. A group's entries are taken six at a time however many there are (``GRP_NC``), so T needs no bound here.
* ``basalt_s2_plan`` (``moe_group_resident``), attrs ``k_per_tok`` and ``n_expert``: ``(ids [n] I32, base)`` →
  ``(grp_ptr, grp_start, counts, ent_dst, ent_tok)``, one threadgroup of 128 over the n routed ids, 1 <= n <= 128: a
  group per distinct id in first-appearance order, its entries in routing order, group g's blob at ``base + id *
  S2_BLOB_BYTES``, ``ent_dst`` the routing index, ``ent_tok`` that index / k_per_tok, ``counts`` (groups, entries).
  The tables hold n groups and n entries (n addresses, n + 1 starts, 2 counts, n destinations and tokens). ``base`` is
  the drafter's ``mtp/experts`` section (basalt's 512 blobs in one allocation, ``mtp.cpp:315``), a U8 value of at
  least n_expert blobs bound at an even offset, so that each address is 2-byte aligned; the router's ids are below
  n_expert.
* ``basalt_q8_0_scaled`` (``quantize_q8_0_scaled``): ``(x [T, n] F32)`` → ``(xq [T, n / 32 * 34] U8, xs [T, n / 32]
  F32)``, n a multiple of 32: the 34-byte block_q8_0 rows and their fp32 scales.
* ``basalt_s2_experts`` (``moe_grouped_s2``), attrs ``cap_groups`` and ``cap_entries``: ``(grp_ptr, grp_start, counts,
  ent_dst, ent_tok, xq [T, 2720], xs [T, 80])`` → ``out [rows, 2560] F32``, four dispatches (gate/up, SwiGLU, the
  intermediate's ``quantize_q8_0_scaled``, down) at 2560 x 640 only (``S2_H``, ``S2_FF``). The tables are one
  ``basalt_s2_plan``'s outputs, as basalt's drafter makes them (``mtp.cpp:931-935``), and the plan's base is the
  buffer the gate/up and down dispatches name in ``indirect_reads``. Over the plan's n ids: cap_groups >= n and
  cap_entries >= n (a threadgroup row per group, a scratch entry per entry), out holds n rows (``ent_dst`` < n) and xq
  the plan's tokens (``ent_tok`` < ceil(n / k_per_tok) <= T; basalt calls with n = T x k_per_tok). T <= 8: the kernels
  take at most 8 entries a group (basalt's ``static_assert(GMAX >= kVerifyMaxT)``, ``s2_expert_grouped.cu:745``), and
  a group holds one entry per token routed to its expert (the router's ids are distinct within a token). The fp32
  scales are required: basalt's drafter always passes them (without them it quantizes the intermediate with
  ``quantize_q8_0``, ``s2_expert_grouped.cu:1111-1112``).

Alignment of the expert ops: 8 bytes for grp_ptr (64-bit addresses), 2 for the plan's base, none for the q8_0 rows the
quantizer writes a byte at a time, 4 for every other binding (the drafter's kernels read the activation rows as
words: basalt sends a misaligned x or scratch to its previous kernels, ``s2_expert_grouped.cu:504-507``, which the
port does not have).
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_Q8_1 = register_op(OpDef("basalt_q8_1", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_q8_1")))
BASALT_MMVQ = register_op(OpDef("basalt_mmvq", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_mmvq")))
BASALT_MMVF = register_op(OpDef("basalt_mmvf", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_mmvf")))
BASALT_DENSE_TC = register_op(OpDef("basalt_dense_tc", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_dense_tc")))
BASALT_DENSE_TC_PAIR = register_op(OpDef("basalt_dense_tc_pair", OpClass.MAP, "rows")
                                   .bind("*", KernelBinding("basalt_dense_tc_pair")))
BASALT_EXPERTS = register_op(OpDef("basalt_experts", OpClass.MAP, "rows")            # gate/up, SwiGLU + q8_1, down
                             .bind("*", KernelBinding("basalt_experts")))
BASALT_S2_PLAN = register_op(OpDef("basalt_s2_plan", OpClass.MAP, "rows")            # one threadgroup over the ids
                             .bind("*", KernelBinding("basalt_s2_plan")))
BASALT_Q8_0_SCALED = register_op(OpDef("basalt_q8_0_scaled", OpClass.MAP, "rows")
                                 .bind("*", KernelBinding("basalt_q8_0_scaled")))
BASALT_S2_EXPERTS = register_op(OpDef("basalt_s2_experts", OpClass.MAP, "rows")      # gate/up, SwiGLU, quantizer, down
                                .bind("*", KernelBinding("basalt_s2_experts")))
BASALT_GR_POST = register_op(OpDef("basalt_gr_post", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_gr_post")))
