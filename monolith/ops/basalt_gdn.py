"""basalt's GDN decode window (``kernels/common/basalt_gdn.metal``; basalt's ``src/kernels/cuda/dec_gdn_rec.cu``): one
dispatch of ``dec_gdn_rec`` per GDN layer and window, its conv, gates, delta-rule recurrence and gated RMS norm bit for
bit basalt's (docs/research/basalt-nvcc/dec_gdn.md). The emitter's handler is ``compiler/emit_basalt.py``'s
``_gdn_window``; nothing lowers to the op here: a model package places it in its programs, between the pair projection
(``basalt_dense_tc_pair``: qkv and z) and the ``ssm_out`` projection that reads ``yq``. The commit, which folds the
window's accepted rows into the state after the host has chosen them, is a separate op.

* ``basalt_gdn_window``: HV value heads of 128, h_k key heads, C = 2 * 128 * h_k + 128 * HV channels (q, k, then v),
  N = n_embd = 2560. HV, C, h_k and N come from the shapes (HV from ``dt``, C from ``conv_w``, h_k = (C - 128 HV) /
  256, N from ``w_alpha``); T, the window's rows, is ``yq``'s.
  - Inputs, in kernel-slot order (input i binds buffer i): ``state (128 * HV * 128 + 3 * C,) F32``, ``qkv (T, C)
    F32``, ``conv_w (C, 4) F32``, ``x (T, N) F32``, ``w_alpha``, ``w_beta (HV, N) BF16``, ``dt``, ``ssm_a (HV,) F32``,
    ``z (T, HV * 128) F32``, ``gamma (128,) F32``, then the window stores ``win_qkv``, ``win_h (8, C) F32``,
    ``win_gate``, ``win_beta (8, HV) F32``. Output ``yq (T, HV * 128 / 32 * 36) U8``: the q8_1 rows of the gated
    RMS-normed output in ``ops/basalt.py``'s layout, which ``basalt_mmvq`` and ``basalt_dense_tc`` read as ``xq``.
    Attr ``eps`` (float): the norms' epsilon.
  - ``state``: the layer's recurrent state, S x HV x S (element (i, head, j) at (i * HV + head) * 128 + j), then the
    conv history [C][3], oldest first. The window reads it and never writes it (invariant 3); the commit does.
  - The stores, rows t < T written for the commit to replay, rows T..7 keeping their bits: ``win_qkv`` the qkv rows as
    they are (the commit's conv history), ``win_h`` the rows after the conv, the SiLU and the q/k L2 norm, ``win_gate``
    and ``win_beta`` the decay gate and the beta of each value head. The commit reads rows < n_keep <= T.
  - The rules the handler enforces: exactly these fourteen inputs and one output; the state and the four window stores
    are graph states (``g.state``) of exactly those shapes; ``updates=`` names the four window stores and never the
    state; the weights (``conv_w``, the gate rows, ``dt``, ``ssm_a``, ``gamma``) are values as shipped, a weight whose
    format, or a constant whose dtype, is ``f32`` or ``bf16`` (a ``.basalt`` file's entries), never a pack's
    block-lane-major slab, and a pack's aux entry holds exactly their bytes; one static program per width T = 1..8, a
    dynamic-T program refused: T is the program's width, in the params record, and the kernel reads no StepState.
    What host ``dec_gdn_rec`` exits on is refused too (``kernels.basalt_gdn_params``: n_embd other than 2560, no key
    head, HV not a multiple of h_k), and a C other than 2 * 128 * h_k + 128 * HV.
  - Alignment, checked at every binding: 16 bytes for ``x`` (read as float4) and the gate rows (read as uint4), 4 for
    every other binding, which basalt's allocations meet unchecked.
  - Dispatch: one threadgroup of 512 threads per value head (basalt's cluster of four 128-thread blocks), above the
    crew geometry as ``gqa_decode_v3``'s is; ``Engine`` refuses a pipeline that allows fewer.
* Names, for the model package and the prompt lane that writes the same state: ``gdn.{i}.state`` and
  ``gdn.{i}.win_qkv``, ``gdn.{i}.win_h``, ``gdn.{i}.win_gate``, ``gdn.{i}.win_beta``, i the layer's ordinal among the
  GDN layers. The stores hold MT = 8 rows (kVerifyMaxT) in every session, so every width's window program and the
  commit program share them by name.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_GDN_WINDOW = register_op(OpDef("basalt_gdn_window", OpClass.MAP, "heads")
                                .bind("*", KernelBinding("basalt_gdn_window")))
