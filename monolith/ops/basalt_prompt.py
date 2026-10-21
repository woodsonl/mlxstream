"""basalt's prompt path (docs/design/flash-next.md): one op kind per distinct signature of the kernels a prompt chunk
runs, ``basalt_prompt_<name>``, each lowered by ``compiler/emit_basalt_prompt.py`` to the dispatches of
``kernels/common/basalt_prompt_*.metal`` with the macros, grids, threadgroups and params of ``monolith/kernels.py``'s
builders. Each prompt PR appends its kinds here, with a bullet below.

Values. Activations are row-major ``[T, n]``, a token's row after the one before. A weight is the tensor as shipped,
``[N, K]`` (N outputs over K inputs): a weight value whose format is the ggml type's name, or, for BF16 rows, a
constant whose dtype is (a ``.basalt`` file's ``dense`` rows; ``packs/basalt.py``), never a pack's block-lane-major
slab (as ``ops/basalt.py`` says).

Rows. T, the first output's rows, is the chunk's row count: ``StepState.t_this_step`` clamped to the compiled bound B
in the dynamic-T prompt program (``kernels.PROMPT_ROWS``; every grid is encoded at B and a threadgroup past the chunk's
rows returns before its first barrier), the compiled count in a static one. Rows never come from an attr or a params
record (invariant 4). Outputs are exactly ``[T, N]``: no ``ldy`` and no ``beta`` (no prompt-path call passes another).

Programs. The dynamic-T prompt program at bound B (one per session; every chunk, the short last one included, runs
through it) and static programs (the tests') are both accepted; every dispatch binds StepState at slot 15 in both.

* ``basalt_prompt_gemm``, attr ``entry`` (``Gemm::<entry>``, ``src/prefill/gemm.cu:122-185``): ``"bf16"``: ``(x [T, K]
  BF16, w [N, K] BF16)`` → ``y [T, N] F32``; ``"native"``: ``(x [T, K] F16, w [N, K]`` GGUF blocks of one of the ten
  ggml types ``Gemm::native`` meets on this pack ``)`` → ``y [T, N] F32``, through one FP16 image of ``w`` per op.
  ``hc8`` adds ``"bf16_w"``. The entry must agree with ``w`` (a BF16 tensor for ``bf16``, GGUF blocks for
  ``native``, as ``native_proj`` and ``bf16_proj`` require, ``src/prefill/prefill.cpp:1188-1199``); a mismatch is
  refused. K is a multiple of 16. Each output is one fp32 ``fma`` chain from +0 in increasing k, so its bits depend
  only on its x row, its w row and K: never on the chunk's rows, the row's place in the chunk or the static slice
  that computes it (basalt's cuBLAS call promises none of these). Until dequant's kernels merge into this branch,
  ``native`` is refused.
* ``basalt_prompt_convert`` (``to_f16`` / ``to_bf16``, ``src/prefill/kernels.cu:818-833``): ``(x [T, n] F32)`` →
  ``y [T, n]`` F16 (``to_f16``) or BF16 (``to_bf16``); the output dtype picks the kernel. Element-local, so a row's
  bits never depend on the chunk. Until dequant's ``bf16_rne_nonan`` merges into this branch, a BF16 output is
  refused.
* ``basalt_prompt_hc_broadcast`` (``gr_broadcast``, ``src/prefill/kernels.cu:199-204``): ``(e [T, n] F32)`` →
  ``r [T, 4n] F32``, each of the four streams a 32-bit copy of ``e`` (NaN payloads included).
* ``basalt_prompt_hc_norm`` (``gr_norm_rs``, ``:85-102``), attr ``eps``: ``(r [T, 4n] F32, w [4n] F32)`` →
  ``(rs [T, 4] F32, xn [T, 4n] BF16)``.
* ``basalt_prompt_hc_silu`` (``gr_silu``, ``:161-170``): ``(lo [T, m] F32)`` → ``lo16 [T, m] BF16``, ``silu(lo / 4)``.
* ``basalt_prompt_hc_mix`` (``gr_mix_r``, ``:103-124``): ``(r [T, 4n], rs [T, 4], w [4n], gated [T, 4n])`` F32 →
  ``(mixed [T, n] F32, mixed_bf [T, n] BF16, mixed_h [T, n] F16)``, or ``(mixed, mixed_h)`` without the BF16 image,
  as the drafter's KV pass calls it (``src/prefill/prefill.cpp:1019``, ``mixed16 = nullptr``).
* ``basalt_prompt_hc_write`` (``gr_write``, ``:192-198``): ``(r [T, 4n], bo [T, n], inj [T, 4])`` F32 →
  ``r' [T, 4n] F32``.
* ``basalt_prompt_hc_write_norm`` (``gr_write_norm_rs``, ``:125-160``), attr ``eps``: ``(r, bo, inj, w_next [4n])`` →
  ``(r' [T, 4n] F32, rs [T, 4] F32, xn [T, 4n] BF16)``: ``r'`` is ``hc_write``'s, ``rs`` and ``xn`` are ``hc_norm``'s
  on ``r'`` with ``w_next``.

  The hyper-connection kinds: n is the model width (2560 on this pack), m the low-rank width (320), and ``r`` holds a
  token's four streams stream-major, ``r[t, c * n + d]``. Each output is per token, bitwise independent of T, of the
  row's index in the chunk and of the other rows. The stream count is basalt's HC = 4 (``kernels.cu:18-19``), fixed by
  the kernels' ``* 0.25f`` and read from the shapes, so no ``streams`` or ``hc`` attr exists and another count is
  refused. ``w`` and ``w_next`` are F32 tensors of 4n elements, ``[4n]`` or ``[1, 4n]``. ``eps`` is finite and at
  least 2^-102 as a float32, so the plain quotient and sum of the row norm give basalt's bits; any other value is
  refused. n is at least 256 (``gr_write_norm_rs`` keeps n / 256 values per thread, ``:128``, and every hc pipeline
  compiles it), a multiple of 256 for the two norms (thread k sweeps ``d = k + 256 j``) and of 4 for the others, as m
  is.

Alignment, checked at every binding: 2 bytes for 16-bit values, 4 for F32 ones.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_PROMPT_GEMM = register_op(OpDef("basalt_prompt_gemm", OpClass.MAP, "rows")
                                 .bind("*", KernelBinding("basalt_prompt_gemm")))
BASALT_PROMPT_CONVERT = register_op(OpDef("basalt_prompt_convert", OpClass.MAP, "rows")
                                    .bind("*", KernelBinding("basalt_prompt_convert")))
BASALT_PROMPT_HC_BROADCAST = register_op(OpDef("basalt_prompt_hc_broadcast", OpClass.MAP, "rows")
                                         .bind("*", KernelBinding("basalt_prompt_hc_broadcast")))
BASALT_PROMPT_HC_NORM = register_op(OpDef("basalt_prompt_hc_norm", OpClass.MAP, "rows")
                                    .bind("*", KernelBinding("basalt_prompt_hc_norm")))
BASALT_PROMPT_HC_SILU = register_op(OpDef("basalt_prompt_hc_silu", OpClass.MAP, "rows")
                                    .bind("*", KernelBinding("basalt_prompt_hc_silu")))
BASALT_PROMPT_HC_MIX = register_op(OpDef("basalt_prompt_hc_mix", OpClass.MAP, "rows")
                                   .bind("*", KernelBinding("basalt_prompt_hc_mix")))
BASALT_PROMPT_HC_WRITE = register_op(OpDef("basalt_prompt_hc_write", OpClass.MAP, "rows")
                                     .bind("*", KernelBinding("basalt_prompt_hc_write")))
BASALT_PROMPT_HC_WRITE_NORM = register_op(OpDef("basalt_prompt_hc_write_norm", OpClass.MAP, "rows")
                                          .bind("*", KernelBinding("basalt_prompt_hc_write_norm")))
