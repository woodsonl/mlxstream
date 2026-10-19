# basalt `qsa_select.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's QSA block scores and radix block top-k, read
from the sm_120 SASS (with the PTX where it shows what the front end chose), and the Metal form the port writes at
each site. This is the selection's page under invariant 5 of `docs/design/flash-next.md`; `matvec.md` is the q8_1
page and `sampler.md` (branch `woodsonl/basalt-sampler`) the model for this page's layout and notation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/qsa_select.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/qsa_select.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/qsa_select.hpp`, `qsa.hpp`, `qsa_seg.hpp`, `kv_stream.hpp`, `qsa_decode_attn.hpp`,
  `native_qsa_indexer.hpp`, `rope_scaling.hpp`, `qsa_dec_fused.hpp`, `pdl.hpp`), so every PTX and SASS line names its
  basalt file and line. CUDA's own headers come from the compiler's installation. Only basalt's public MIT source was
  sent.
- **Flags.** `qsa_select.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:160`) with
  nvcc's IEEE defaults: it is not in the `--use_fast_math` list (`:186-202`), and the only other fast-math flag
  (`:728`) belongs to the `mmq` target. A Release build (`:12-14`) passes CMake's `-O3 -DNDEBUG`,
  `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120 (`:3-4`,
  `:26-31`). No `-fmad`, `-ftz`, `-prec-div` or `-Xptxas` flag is set, so nvcc's defaults hold: `-fmad=true` (a
  multiply and an add in one expression may contract), `-ftz=false`, `-prec-div=true`, `-prec-sqrt=true`. CUDA
  therefore keeps subnormal operands and results in every float instruction of this file, and its ordered compares
  see a subnormal as itself.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`, which returns the PTX and the `SASS (sm_120)`
  listing. Compiler id `nvcc130`: NVCC 13.0.0, version string `Cuda compilation tools, release 13.0, V13.0.48`.
  Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`).
- **userArguments**, verbatim: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'`. Compiler Explorer
  adds `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them
  changes a kernel's code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026; every request below was posted again on October 18, 2026 and returned the
  same PTX and SASS, instruction for instruction and source line for source line (`.context/qsa/scripts-nvcc-c/recheck.sh`).
- **Translation units.** The whole file's SASS listing stops at Compiler Explorer's 5,000-line cut
  (`[truncated; too many lines]`) inside the first function it lists, `block_topk_reg_kernel<33>` (4,580
  instructions before the cut). So each kernel group was compiled as its own unit, the other kernels' bodies blanked
  line for line (an empty line keeps every line number; a blanked kernel compiles to 16 instructions, `EXIT` and
  padding) and the host launches of the template instances not kept replaced by `;` on the same line:

  | Unit | Kernel bodies kept | Launches kept | SASS of the kept kernels |
  |---|---|---|---|
  | `b1_rg2` | `block_scores_rg_kernel` (`:631-680`) | `<2>` (`:993-994`) | `block_scores_rg_kernel<2>` 672 |
  | `b1_rg4` | `block_scores_rg_kernel` | `<4>` (`:996-997`) | `block_scores_rg_kernel<4>` 968 |
  | `b1_rg8` | `block_scores_rg_kernel` | `<8>` (`:999-1000`) | `block_scores_rg_kernel<8>` 1,656 |
  | `b2_cluster` | `block_topk_cluster_kernel` (`:794-969`) | none | 1,576 |
  | `b3_topk_ref` | `block_topk_kernel` (`:61-153`) | none | 968 |
  | `b4_topk_reg` | `block_topk_reg_kernel` (`:335-421`) | `<TK_PER>`, `<TK_PER_MAX>` (`:1177`, `:1179`) | truncated, see below |
  | `b5_topk_wide` | `block_topk_wide_kernel` (`:433-538`) | none | 1,184 |
  | `b6_scores_plain` | `block_scores_kernel` (`:34-55`), `block_scores_tail_kernel` (`:274-293`), `block_scores_seg_kernel` (`:553-599`) | none | 120, 112, 672 |
  | `b7_scores_tc` | `block_scores_tc_kernel` (`:185-265`) | none | 1,000 |

  The requests and responses are `.context/qsa/nvcc/split/<unit>.{cu,request.json,response.json}` and the whole
  file's `.context/qsa/nvcc/qsa_select.{request,response_nvcc130}.json`; `.context/qsa/scripts-nvcc-c/blanked.py`
  lists each unit's blanked lines from the `#line` map, and `funcs.py` lists or dumps a response's functions (with
  `float`, only the float instructions, each with its source line). Every device function a kernel calls is inlined
  into it, so blanking another kernel's body cannot change a kept kernel's code.
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.f32`), which division it
  chose and which library sequence it inlined; the SASS shows what `ptxas` made of it. A row quotes the SASS of one
  copy where a site is unrolled; the copies were compared and differ only in registers. Instructions after a kernel's
  last `EXIT` are CUDA's out-of-line slow paths, which `-lineinfo` attributes to the line before them; the tables
  attribute each to the `CALL.REL.NOINC` that reaches it.
- **Not float arithmetic.** `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` at a strided loop's
  head are `ptxas`'s exact integer division; `HFMA2 R, -RZ, RZ, a, b` moves a constant; `IMAD.WIDE` forms
  addresses. The tables leave them out unless they read a stored value.

## 2. Kernels in the file and their paths

The decode path is the Flash-Next verify window (T = 1..8 rows, `.context/qsa/spec-engine.md` section 3; plain
decode is its T = 1 case) and the per-token layer forward; both score with `qsa_block_scores` and select with
`qsa_block_topk`. The window calls them per row group `[tb, te)` of `n = te - tb <= 8` rows (`src/core/verify.cpp:1037`)
with `active_blocks = -1` and `skip_identity = true` (`verify.cpp:1545-1550`), the layer forward with `nq = 1` and the
defaults (`src/core/layer.cpp:996-997`, under `g_fast_select`, which the window requires, `layer.cpp:434`). The
`batch_rec_` branch (`verify.cpp:1532-1534`, `qsa_block_scores_seg`) is the `--parallel` batch window, outside the
contract (`.context/qsa/spec-kernels-indexer-a.md` section 1.4). The prompt path (`src/prefill/prefill.cpp:1657-1670`)
is a v1 non-goal; it runs the kernels the tests below use as references.

| Kernel | Lines | Host entry and condition | Path | This page |
|---|---|---|---|---|
| `block_scores_rg_kernel<2>` | `:624-681` | `qsa_block_scores`, `nq <= MQ` (8, `:543`) and `active_blocks <= 0` (`:982`), `nq <= 2` (`:992-994`); grid `2 * SMs` (`:984-991`), block 256, PDL | ON: the window with n = 1, 2 and the layer forward (`verify.cpp:1545`, `layer.cpp:996`) | section 3.2 |
| `block_scores_rg_kernel<4>` | `:624-681` | the same, `nq` 3 or 4 (`:995-997`) | ON: the window with n = 3, 4 | section 3.2 |
| `block_scores_rg_kernel<8>` | `:624-681` | the same, `nq` 5 to 8 (`:998-1000`) | ON: the window with n = 5 to 8 | section 3.2 |
| `block_topk_cluster_kernel` | `:791-970` | `qsa_block_topk`, `nq <= CL_MAXQ` (16) and `topk_cluster_launch` returns true (`:1161`, `:1080-1147`); grid `(8, nq)`, block 1024, cluster `(8, 1, 1)` | ON: every window and layer-forward selection (`verify.cpp:1549`, `layer.cpp:997`; `nq <= 8`) | section 3.4 |
| `block_scores_kernel` | `:29-56` | `qsa_block_scores` otherwise: `nq > 8` or `active_blocks > 0` (`:1005-1008`); grid `(ceil(reach / 8), nq)`, block 256 | OFF the decode path: the prompt path's fallback (`prefill.cpp:1667`). `qsa_scores_parity`'s reference (`src/kernels/qsa_scores_parity.cpp:103`, "the reference kernel") | section 3.3 |
| `block_topk_kernel` | `:58-154` | `qsa_block_topk_ref` (`:1067-1077`); `qsa_block_topk` step 3 (`:1162-1168`, `active_blocks > 0` only); grid `(nq)`, block 256 | OFF the decode path: the prompt path's short batches. The one-CTA reference of `qsa_topk_parity` (`:69`), `qsa_scores_parity` (`:214`) and `decode_cluster_parity` (`:156-157`, `:290`) | section 3.4 |
| `block_topk_reg_kernel<33>` | `:331-422` | `qsa_block_topk`, `nq > 16` or no cluster, `reach = max_blocks <= 33,792` (`:1176-1177`); grid `(nq)`, block 1024. `<TK_PER_MAX>` (`:1179`) is the same instance (`TK_PER_MAX == TK_PER`, `:301`) | OFF: the prompt path; also a decode call whose cluster launch returns false (a capacity with `per_cta * 4 + 4 >= 65536`, `:1085`, or no resident cluster of 8 CTAs, `:1106-1126`) | section 3.4 |
| `block_topk_wide_kernel` | `:430-539` | `qsa_block_topk`, `reach > 33,792` (`:1173-1174`); grid `(nq)`, block 1024 | OFF: as the register kernel | section 3.4 |
| `block_scores_seg_kernel` | `:548-600` | `qsa_block_scores_seg` (`:1013-1024`); grid 256, block 256, PDL | OFF: the `--parallel` batch window (`verify.cpp:1533`). `qsa_scores_parity`'s per-segment check (`:160-166`) | section 3.5 |
| `block_scores_tc_kernel` | `:180-266` | `qsa_block_scores_tc` (`:1026-1065`); grid `(ceil(reach / (TC_NB * TC_ITER)), ceil(nq / TC_QT))`, block 128 | OFF: the prompt path's scores, 3xTF32 MMA ("FP32-level, not bitwise", `prefill.cpp:1663`) | section 3.5 |
| `block_scores_tail_kernel` | `:270-294` | `qsa_block_scores_tc`, after the tc kernel (`:1061`); grid `(nq)`, block 32 | OFF: as `block_scores_tc_kernel` | section 3.5 |

`qsa_select.cu` has no other `__global__` function. The decode path's top-k on Metal is not the cluster kernel's
mechanics (no thread-block clusters or distributed shared memory on this GPU) but separate dispatches that give the
one-CTA reference's ids (`spec-kernels-indexer-b.md` section 7); every top-k variant computes the same function of the
same keys (`:692-698`), so section 3.4 records the float sites of all four.

## 3. Per-kernel SASS tables

### 3.0 The Metal operations this page names

The file builds IEEE, so CUDA keeps subnormal operands and results at every instruction without `.FTZ`, and its
ordered comparisons see a subnormal as itself; this GPU flushes subnormals in arithmetic and compares them as zero.
The Metal column uses `sampler.md`'s names (section 2.0 there; the same set as `qsa_dec_attn.md` section 3.0), one
helper per operation at every site:

- `key32(b)` and `nan32(b)` on a float's bits, as in `sampler.md`: `key32` maps -0 to +0 and then sets the sign bit
  of a non-negative pattern or inverts a negative one; `nan32(b)` is `(b & 0x7fffffffu) > 0x7f800000u`. On the
  sampler branch they are `okey` and `fnan` (`kernels/common/basalt_sampler.metal:37-41`).
- `order_key32(b)`: basalt's `order_key`, `nan32(b) ? 0u : key32(b)` (section 3.1). It differs from `key32` only at
  NaN, which it sends to 0, below every other key (the least, -inf's, is `0x007fffff`). The sampler's split rounds
  already write this expression (`fnan(s[j]) ? 0u : okey(s[j])`, `basalt_sampler.metal:334` on that branch).
- `add32`, `mul32`, `fma32`: the correctly rounded IEEE operation with subnormals kept (`FADD`, `FMUL`, `FFMA`);
  Metal's `+`, `*` or `fma` whenever the operands and the result are normal, zero or infinite, otherwise the
  emulation. `add32` and `mul32` are main's `ieee_add` (`kernels/common/basalt_q8_1.metal:95`) and `ieee_mul`
  (`:108`). Main's `ieee_fma` (`:167`) takes an integer middle operand below 2^24 only ("fma(a, n, c) ... for an
  integer n, |n| < 2^24", `:158`), so the general `fma32` this file needs is the sampler branch's `ieee_fma_count`
  (`kernels/common/basalt_sampler.metal:80` on `origin/woodsonl/basalt-sampler`: the hardware's `fma` when every
  operand has a nonzero exponent field and the result is at least 2^-125, else the exact value rounded to odd in its
  software binary64 and then to binary32). Each row says whether a subnormal is reachable.
- `relu32(f)`: the source's `f > 0.0f ? f : 0.0f` (`:615`) with IEEE's ordered `>`, on the bits:
  `(as_type<uint>(f) - 1u) < 0x7f800000u ? f : 0.0f`. It keeps f from the least positive subnormal (bits 1) up to
  +inf (bits `0x7f800000`) and returns +0 for +0, -0, every negative value and every NaN. No float form of Metal
  gives that on this GPU: `f > 0.0f` reads a positive subnormal as zero and drops it, `fmax(f, 0.0f)` compares it as
  zero and returns the second operand of the tie (+0), and `fmax(0.0f, f)` keeps a negative subnormal and returns -0
  for -0. CUDA's `FMNMX R, RZ, f, !PT` (PTX `max.f32 f, 0`, no `.ftz`) computes the same function: NaN gives the
  other operand +0, a subnormal compares as itself, and `max.f32` orders +0 above -0 ([R] PTX ISA, `max`: "If
  values of both inputs are 0.0, then +0.0 > -0.0"); the Metal form follows the source's expression, which an IEEE
  build cannot change.

### 3.1 `order_key` (`:22-27`): the float-to-key transform

Every top-k kernel turns each score into a 32-bit key with `order_key` and selects on keys alone (the radix
histograms, the threshold digit, the ties by id; `.context/qsa/spec-kernels-indexer-b.md` section 7), so `order_key`
is the only floating-point code in the four top-k kernels (section 3.4). Source, verbatim (`:22-27`):

```cpp
__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
```

SASS from `block_topk_cluster_kernel` (unit `b2_cluster`), the copy in the strided loop of `:837` at 0xf70-0x1020:

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:24` (tested first) | `if (!(v == v)) return 0u;` | PTX `setp.nan.f32 %p1, %f1, %f1` on `s` itself, before the add (v is NaN exactly when s is); SASS `FSETP.NAN.AND P0, PT, R8, R8, PT`; the key register starts at 0 (`MOV R15, RZ`) and takes the transform only on !P0 (`@!P0 MOV R15, R16`) | `nan32(b) ? 0u : ...` on `b = as_type<uint>(s)` |
| `:23` | `const float v = s + 0.0f;` | PTX `add.f32 %f2, %f1, 0f00000000`; SASS `@!P0 FADD R15, RZ, R8` (IEEE: v = s for every non-NaN s except -0, which becomes +0; a subnormal s stays itself) | `b = (b == 0x80000000u) ? 0u : b` on the bits |
| `:25-26` | `(b & 0x80000000u) ? ~b : (b \| 0x80000000u)` | `ISETP.GE.AND P2, PT, R15, RZ, !P0` (sign bit clear, as a signed integer test); `@!P0 LOP3.LUT R16, RZ, R15, RZ, 0x33, !PT` (`~b`); `@!P0 FADD R18, -\|R15\|, -RZ` (the compiler writes `b \| 0x80000000u` as the float -\|v\| + (-0), PTX `abs.f32`, `neg.f32`); `@P2 MOV R16, R18` | `(b & 0x80000000u) ? ~b : (b \| 0x80000000u)` on `uint` |

- **CUDA's key is the integer transform.** `FADD R, -|v|, -RZ` without `.FTZ` returns -|v| exactly for every
  non-NaN v (-|v| + (-0) is -|v| in IEEE arithmetic; +0 gives -0), so its bits are `b | 0x80000000u`, a subnormal v
  included. With `:23`'s add, `order_key(s)` is `key32(s)` for every non-NaN s and 0 for NaN: `order_key32` on the
  bits gives CUDA's key for every input.
- **The Metal trap.** Written as float operations on this GPU, `s + 0.0f` reads a subnormal s of either sign as a
  zero of its sign and returns +0 (and `-abs(v)` would flush as well), so every subnormal score would get +0's key
  `0x80000000` and tie with zero and with each other, where CUDA orders them as themselves. The Metal form is integer
  code on `as_type<uint>(s)`, no float operation. The same source in a `--use_fast_math` file compiles to
  `FSETP.NAN.FTZ`, `FADD.FTZ R0, RZ, R2`, `FADD.FTZ R0, |R0|, -RZ`, `FADD.FTZ R6, -R0, -RZ`, which is that flushing
  key (`.context/qsa/nvcc/order_key_{ieee,fast}.{request,response}.json`, a 10-line kernel around `:22-27` compiled
  with and without the flag); `qsa_select.cu` is IEEE, so the subnormal-keeping form is basalt's.
- **The order the keys give.** For non-NaN a and b, a < b in IEEE order (subnormals as themselves, -0 equal to +0)
  exactly when `order_key(a) < order_key(b)`, and the keys are equal exactly when a == b. Every NaN, whatever its
  sign and payload, gets 0 and ranks below -inf; two NaNs tie. The selection breaks equal keys by id (section 3.4).
- **Copies.** Each top-k kernel inlines the same three float instructions (`FSETP.NAN`, `FADD RZ, s`,
  `FADD -|v|, -RZ`) per copy, differing only in registers and predicates (the copy at `:147` runs its first `FADD`
  unpredicated and selects afterwards): `block_topk_cluster_kernel` 5 copies at `:837` (0xf70 in the remainder loop,
  0x1130-0x1160 in the loop unrolled by 4); `block_topk_kernel` 7 at `:88`, 5 at `:113`, 1 at `:147`;
  `block_topk_reg_kernel<33>` 33 at `:353` (all before the SASS listing's cut, section 3.4); `block_topk_wide_kernel`
  7 at `:458`, 5 at `:489`, 1 at `:516` (`.context/qsa/scripts-nvcc-c/keysites.py` lists each copy's address and
  calling line).

### 3.2 `block_scores_rg_kernel<2>`, `<4>`, `<8>` (`:624-681`; `heads_score`, `:605-619`)

Units `b1_rg2`, `b1_rg4`, `b1_rg8`. A warp scores one block per trip of the loop `:661` (the CTA's 8 warps
take blocks `blockIdx.x * 8 + warp`, then step by `gridDim.x * 8`); lane l holds dims `4l .. 4l + 3` of the block's
key row and of every query's four heads, so per (block, query) the warp computes the 4 heads' dot products and lane 0
stores the score. SASS from `b1_rg2` (`<2>` unrolls the query loop `:665` twice; the copies differ only in
registers); the PTX is the same for every instance.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:667` | `const float4 k4 = (b == nbid[qi]) ? kd : cur;` | PTX `selp.f32` ×4; SASS `FSEL R68, R17, R2, !P1` (and `.x`, `.z`, `.w`), `P1` = `b != nbid[qi]` | integer select of the four words |
| `:672` | `dd[h] = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;` | PTX `mul.f32 %f59, %f57, %f6` (`k.y * q.y`); `fma.rn.f32 %f60, %f58, %f5, %f59` (`k.x`); `fma.rn.f32 %f61, %f56, %f7, %f60` (`k.z`); `fma.rn.f32 %f62, %f55, %f8, %f61` (`k.w`). SASS per head: `FMUL R77, R21, R68`; `FFMA R77, R20, R69, R77`; `FFMA R72, R22, R69, R77`; `FFMA R77, R23, R74, R72` | `fma32(k.w, q.w, fma32(k.z, q.z, fma32(k.x, q.x, mul32(k.y, q.y))))` |
| `:607-608` | `r0 = __shfl_xor_sync(.., hi ? a0 : a2, 16)`, `r1 = ...(hi ? a1 : a3, 16)` | `FSEL R73, R76, R77, !P6` (P6 = `lane & 16`); `SHFL.BFLY PT, R68, R73, 0x10, 0x1f`; the same for `r1` | `simd_shuffle_xor` of the selected word by 16 |
| `:609` | `e0 = (hi ? a2 : a0) + r0, e1 = (hi ? a3 : a1) + r1` | `FADD R76, R68, R76`; `FADD R75, R77, R74` | `add32(sel, r0)`, `add32(sel, r1)` |
| `:610-611` | `r2 = __shfl_xor_sync(.., b ? e0 : e1, 8); f = (b ? e1 : e0) + r2` | `SHFL.BFLY PT, R74, R74, 0x8, 0x1f`; `FADD R75, R74, R75` | `add32(sel, r2)` |
| `:612-614` | `f += __shfl_xor_sync(.., f, 4)`, then 2, then 1 | `SHFL.BFLY` 0x4, 0x2, 0x1, each followed by `FADD R, f, r` | `add32(f, simd_shuffle_xor(f, o))`, o = 4, 2, 1 |
| `:615` | `const float r = f > 0.0f ? f : 0.0f;` | PTX `max.f32 %f93, %f92, 0f00000000`; SASS `FMNMX R77, RZ, R68, !PT` | `relu32(f)`, section 3.0 |
| `:616-618` | `g1 = __shfl_sync(.., r, 8)`, `g2 = ...(r, 16)`, `g3 = ...(r, 24)`; `((r + g1) + g2) + g3` | `SHFL.IDX` 0x8, 0x10, 0x18; `FADD R76, R77, R76`; `FADD R69, R69, R76`; `FADD R73, R72, R69` | `add32(add32(add32(r, g1), g2), g3)` with `simd_shuffle` from lanes 8, 16, 24 |
| `:676` | `if (b == nbid[qi] && nkv[qi] % R != 0) score += 1e9f;` | PTX `add.f32 %f99, %f53, 0f4E6E6B28`, two `selp.f32`; SASS `@!P0 FADD R68, R73, 1e9` (computed on lane 0 before the test), `@!P4 FSEL R73, R68, R73, P5` | `add32(score, 1e9f)` under the same condition |
| `:677` | `out[qi * max_blocks + b] = score;` | `@!P0 STG.E` (P0 = lane != 0) | store the word |

- **The sum, exactly.** `IDX_DIM = 128`, `IDX_HEADS = 4` (`:18`). For one (block, query) and head h, lane l's
  `dd[h]` is the chain above over dims `4l .. 4l + 3` (the PTX multiplies the y words first: `mul.f32 %f59, %f57,
  %f6` with `%f57` the key's y word, `%f6` the query's, in `<8>` as in `<2>`). The transposed butterfly leaves head 0
  in lanes 0-7, head 1 in 8-15, head 2 in 16-23 and head 3 in 24-31 after the offset-8 step, and each lane's value is
  the plain XOR butterfly's over offsets 16, 8, 4, 2, 1 for its head: an IEEE add of two operands is the same in
  either order (bits included, NaN aside, and a NaN never reaches the store), which is what `:602-603` claims. Lane 0
  then adds the four relu'd head sums in head order (`:618`). A Metal kernel with a lane's dims `4l .. 4l + 3` can
  copy the transposed form with `simd_shuffle_xor` and `simd_shuffle`, or run one plain butterfly per head: the bits
  are the same.
- **Subnormals.** Reachable only through the inputs: a subnormal element of a query or pooled key, or a nonzero
  product `k_i * q_i` below 2^-77 in magnitude. When every nonzero element of the block's key row and the query is
  at least 2^-38 in magnitude, every exact product is a multiple of 2^-122, so is every rounded product, fma and sum,
  and each is zero or normal: Metal's `*`, `fma` and `+` give CUDA's bits. Below that, `mul32`, `fma32` and `add32`
  are required (a product or a cancelling partial sum can land on the subnormal grid, which this GPU flushes). The
  queries and pooled keys are RMS-normed rows (`native_qsa_indexer.md` for the keys) and `qsa_scores_parity` draws
  N(0, 1) values (`src/kernels/qsa_scores_parity.cpp:67`), so the emulation is not exercised there [H]. A guard that
  is uniform over the SIMD-group (every lane's loaded words zero or at least 2^-38 in magnitude, `simd_all`) can pick
  Metal's operators for a whole (block, query) and the helpers otherwise; the butterfly's adds need the guard of all
  32 lanes because they mix the lanes' partial sums.
- **No NaN and no -0 reaches the store.** `relu32` sends NaN (an infinite element's `inf - inf`, a NaN input) to +0
  and every non-positive sum to +0, so the score is a sum of four values in [+0, +inf] in head order: never NaN,
  never -0. The store at `:677` needs no `cuda_nan`.
- **The `1e9f` add is plain `+`.** The score is at least +0; a subnormal score is below half an ulp of 1e9f (64),
  so CUDA's IEEE sum is 1e9f, which is what Metal's `+` returns after reading the subnormal as zero. Every other
  score is zero or normal, where the two agree.
- **Instances.** `<4>` and `<8>` repeat `<2>`'s per-copy sequence (`.context/qsa/scripts-nvcc-c/fseq.py`, counts per
  source line): per query copy four `FMUL` and twelve `FFMA` at `:672`, one `FMNMX` per head at `:615`, nine `FADD`
  in `heads_score` and one `FADD` of `1e9`, with every count scaling as NQ (8 `FMUL` and 24 `FFMA` in `<2>`, 16 and
  48 in `<4>`, 32 and 96 in `<8>`). The butterfly and relu of each `heads_score` (`:607-615`) are listed twice,
  the second time in the out-of-line copy `ptxas` emits for a lane that arrives without the full warp
  (`SHFL.BFLY P, R, R, R, R`), whose float sequence is the same; the three `:618` adds are listed once.

### 3.3 `block_scores_kernel` (`:29-56`)

Unit `b6_scores_plain`. OFF the decode path (section 2); it is `qsa_scores_parity`'s reference, so its float
sequence is recorded against section 3.2's. One warp per (block, query) (`b = blockIdx.x * 8 + warp`, query
`blockIdx.y`); lane l holds dims `4l .. 4l + 3`; the four heads run one after another in the source and interleaved
in the SASS (the copies differ only in registers).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:40-41` | `key = (b == n_bid) ? dead : pooled + b * IDX_DIM`; the float4 load | address select, `LDG.E.128.CONSTANT R8, desc[UR4][R8.64]` (the query rows: four `LDG.E.128.CONSTANT` at `:46`) | the same row choice |
| `:47` | `float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;` | PTX `mul.f32 %f10, %f3, %f7` (k.y · q.y); `fma.rn.f32 %f11, %f2, %f6, %f10` (k.x); `fma.rn.f32 %f12, %f4, %f8, %f11` (k.z); `fma.rn.f32 %f13, %f5, %f9, %f12` (k.w). SASS per head: `FMUL R5, R9, R5`; `FFMA R5, R8, R4, R5`; `FFMA R6, R10, R6, R5`; `FFMA R6, R11, R7, R6` | `fma32(k.w, q.w, fma32(k.z, q.z, fma32(k.x, q.x, mul32(k.y, q.y))))`, section 3.2's chain |
| `:49` | `for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);` | `SHFL.BFLY` 0x10, 0x8, 0x4, 0x2, 0x1, each followed by `FADD R, d, r` (PTX `add.f32 %f15, %f13, %f14`: own value first) | `add32(d, simd_shuffle_xor(d, o))`, o = 16, 8, 4, 2, 1 |
| `:50` | `score += d > 0.0f ? d : 0.0f;` (score starts at `0.0f`, `:43`) | PTX `max.f32 %f24, %f23, 0f00000000` per head; the front end drops `0.0f +` and adds the first two heads directly (`add.f32 %f44, %f24, %f43`), then `add.f32 %f64, %f44, %f63`, `add.f32 %f1, %f64, %f83`. SASS `FMNMX R5, RZ, R5, !PT` per head; `FADD R0, R5, R0` (r0 + r1); `FADD R0, R0, R7` (+ r2); `FADD R5, R0, R5` (+ r3) | `add32(add32(add32(r0, r1), r2), r3)` with `r_h = relu32(d_h)` |
| `:53` | `if (b == n_bid && n_kv % R != 0) score += 1e9f;` | PTX `add.f32 %f84, %f1, 0f4E6E6B28` and two `selp.f32`; SASS `FADD R0, R5, 1e9` (every lane), `@!P0 FSEL R5, R0, R5, P1` | `score + 1e9f` (plain `+`, section 3.2) under the same condition |
| `:54` | `out[qi * max_blocks + b] = score;` | `STG.E` on lane 0 | store the word |

- **Bit for bit the rg kernel's score.** The chain at `:47` pairs the same words in the same order as `:672`
  (k.y · q.y first, then x, z, w); lane 0's butterfly sum at `:49` is section 3.2's transposed butterfly's value
  for each head (the same pairs at each offset, an IEEE add being the same in either order); and the head sum is
  `((r0 + r1) + r2) + r3` in both (`:618` adds `g1`, `g2`, `g3` to lane 0's `r` in head order). In IEEE arithmetic
  `0.0f + r0` is r0 for every r0 in [+0, +inf], so dropping it changes no bit on CUDA. This is the claim of `:622-623`, and
  `qsa_scores_parity`'s `memcmp` (`src/kernels/qsa_scores_parity.cpp:106`) rests on it.
- **The Metal trap here.** A Metal reference written from the source, `score = 0.0f; score += relu`, would flush a
  subnormal r0 to zero with Metal's `+` (CUDA keeps it). The reference writes the same `add32` chain from r0, as the
  PTX does, so the Metal decode kernel and the Metal reference share one sequence and agree bit for bit.
- **Subnormals, NaN, -0:** as section 3.2: reachable only through elements whose products fall below 2^-77; no NaN
  and no -0 reaches the store.

### 3.4 The block top-k: `block_topk_cluster_kernel` (`:791-970`), `block_topk_kernel` (`:58-154`), `block_topk_reg_kernel<33>` (`:331-422`), `block_topk_wide_kernel` (`:430-539`)

Units `b2_cluster`, `b3_topk_ref`, `b4_topk_reg`, `b5_topk_wide`. The four kernels compute one function of the keys
(`.context/qsa/spec-kernels-indexer-b.md` section 0: the threshold key by four 8-bit radix passes over weighted
histograms, every cell above it, then the cells at it in ascending cell order up to the budget, written in ascending
cell order). Their only floating-point code is `order_key` (section 3.1): every other operation is integer.

**Float census** (`.context/qsa/scripts-nvcc-c/ptxk.py <response> <kernel> float` for the PTX, `funcs.py ... float`
for the SASS; register numbers differ between copies, nothing else):

| Kernel | `order_key` call sites (copies) | PTX float instructions | SASS float instructions | SASS listed |
|---|---|---|---|---|
| `block_topk_cluster_kernel` | `:837` (5) | per copy `setp.nan.f32` (`:24`), `add.f32 s, 0f00000000` (`:23`), `abs.f32` and `neg.f32` (`:26`); 5 `ld.global.nc.f32` of the scores | per copy `FSETP.NAN.AND P, PT, s, s, PT`, `@!P FADD R, RZ, s`, `@!P FADD R, -\|v\|, -RZ`: 5 of each | 1,576, complete |
| `block_topk_kernel` | `:88` (7), `:113` (5), `:147` (1) | 13 copies of the four; 13 score loads | 13 of each three; the `:147` copy's `FADD R14, RZ, R14` unpredicated | 968, complete |
| `block_topk_reg_kernel<33>` | `:353` (33) | 33 copies of the four; 33 score loads (offsets 0 to +128 bytes) | 33 of each three, all before the listing's cut | 4,580 before `[truncated; too many lines]` |
| `block_topk_wide_kernel` | `:458` (7), `:489` (5), `:516` (1) | 13 copies of the four; 13 score loads | 13 of each three | 1,184, complete |

- **The truncated listing hides no float site.** The PTX of `block_topk_reg_kernel<33>` is complete and its only
  instructions with a float type are the 33 `order_key` copies and the 33 score loads; the PTX holds no `cvt` to or
  from a float (its `cvt`s are `s64.s32`, `u32.u64`, `u64.u32`), no float `setp` other than `.nan`, no `selp.f32`,
  no float `min`/`max`, no `call`, and no `div` or `rem`, so `ptxas` has no integer-division expansion (section 1,
  "Not float arithmetic") to add either; the SASS before the cut already holds all 33 copies of each of the three
  instructions and no `MUFU`, `I2F` or `F2I`. The same holds for the other three kernels' complete listings. No
  recompilation was needed.
- **Every compare of the selection is an integer compare.** `(k & hi_mask) == (prefix & hi_mask)` (`:89`), `k > thr`
  and `k == thr` (`:114-115`, `:148-150`), the digit scan `cum + hist[d] >= width` (`:95`), and their copies in the
  other three kernels compare keys, counts or ids (`ISETP`, `IMNMX`, `VIMNMX`); the census finds no `FSETP` but the NaN test,
  no `FMNMX` and no `FSEL`. So the IEEE order of the scores, subnormals included, enters the selection only through
  `order_key`, and the Metal kernels need no float compare at all: they load the scores as `uint`, apply
  `order_key32` (section 3.0), and run the integer logic of `spec-kernels-indexer-b.md` sections 4-5.
- **NaN.** `FSETP.NAN.AND P, PT, s, s, PT` is true exactly when s is a NaN (a subnormal is not), so every NaN score
  gets key 0, the least key, and ties with every other NaN; cells at the threshold are taken in ascending cell order
  (`spec-kernels-indexer-b.md` section 5). `decode_cluster_parity` checks NaN fixtures (`kNaNs`, `kAllNaN`,
  `src/kernels/decode_cluster_parity.cpp:104`, `:114`). Basalt's own scores are never NaN (section 3.2), so a NaN
  key needs a test-written score.
- **`active_blocks` chooses a kernel, not a function.** `qsa_block_topk` takes the one-CTA reference when the reach
  passes the register kernel's and `0 < active_blocks < 4608` (`:1165-1167`); the decode path passes -1 and takes the
  cluster kernel (section 2). Every variant gives the same ids, so a Metal port has one selection function and picks
  its dispatch geometry freely.
- **The cluster's mechanics.** Distributed shared memory, `st.async`, mbarriers and the cluster barrier
  (`:700-703`, `:807-817`) move integer counts only; on Metal the cluster becomes a per-slice histogram dispatch and a
  merge that reproduce the one-CTA reference's ids (`spec-kernels-indexer-b.md` sections 7.3 and 10.4). No float
  value crosses a CTA boundary.

### 3.5 Off the decode path: `block_scores_seg_kernel` (`:548-600`), `block_scores_tc_kernel` (`:180-266`), `block_scores_tail_kernel` (`:270-294`)

Units `b6_scores_plain` (seg, tail) and `b7_scores_tc`. Recorded for completeness and for the tests that call them;
no Metal kernel of the decode path needs them.

| Kernel | Line | basalt | PTX / SASS | Metal, if ported |
|---|---|---|---|---|
| `block_scores_seg_kernel` | `:575` | the queries staged in shared memory, `qs[i] = q_idx[i]` | `ld.global.nc.f32`, `st.shared.f32` (word copies; one `div.u32` for the loop's trip count) | word copies |
| | `:589` | `float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;` | `mul.f32 %f15, %f8, %f12` (k.y · q.y), then `fma.rn.f32` with x, z, w: section 3.3's chain | section 3.3's chain |
| | `:591`, `:592` | the butterfly 16 .. 1; `score += d > 0.0f ? d : 0.0f` | 5 `add.f32` per head; `max.f32 d, 0`; three head adds, the `0.0f +` dropped as at `:50` | `add32`, `relu32`, section 3.3 |
| | `:595` | `if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;` | `add.f32 %f, %f, 0f4E6E6B28`, `selp.f32` | plain `+` |
| `block_scores_tail_kernel` | `:285`, `:287`, `:288`, `:291` | `block_scores_kernel`'s body for the block `n_bid` on `dead` ("exactly block_scores_kernel's arithmetic for that block", `:269`) | `mul.f32 %f10, %f3, %f7` (k.y · q.y), three `fma.rn.f32`, 5 `add.f32` and one `max.f32` per head, three head adds, the `1e9` add | section 3.3 |
| `block_scores_tc_kernel` | `:170` (`tf32_hi`) | `cvt.rna.tf32.f32` (inline PTX), 144 copies | `FSETP.GEU.AND P, PT, \|x\|, +INF, PT`; `IADD3 r, x, 0x1000`; `SEL r, r, x, !P`; `LOP3.LUT r, r, 0xffffe000` (round half away on the bits, a non-finite x only masked) | integer code on the bits |
| | `:226-227`, `:235-238` | `x - __uint_as_float(hi)` (the TF32 low parts) | `sub.f32`; SASS `FADD R, x, -hi`, IEEE (the difference is exact, on x's ulp grid; it can be subnormal once \|x\| < 2^-102) | `add32(x, -hi)` |
| | `:177` (`mma_tf32`, three per head and k-step at `:239-241`: lo·hi, hi·lo, hi·hi) | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | `HMMA.1688.F32.TF32`, 48 in the listing | no Metal equivalent with the same bits (below) |
| | `:260` | `score += d > 0.0f ? d : 0.0f` per head | `FMNMX R, RZ, d, !PT`; three `FADD` per score, the `0.0f +` dropped | `relu32`, `add32` |

- **The seg and tail kernels repeat `block_scores_kernel`'s sequence**: the same chain (k.y · q.y first), the
  same butterfly, the same relu and head order, the same `1e9f` add, so `qsa_scores_parity`'s segment check
  (`memcmp` against one reference call per row, `src/kernels/qsa_scores_parity.cpp:160-172`) holds by construction.
  The seg kernel is the `--parallel` batch window (`src/core/verify.cpp:1533`), outside the contract.
- **The tc kernel is not bitwise by design.** Basalt's comment: "the summation order differs from the warp kernel,
  so a score can move in its last bits and a near-tie can select differently (not bitwise)" (`:157-160`); its one
  caller is the prompt path (`src/prefill/prefill.cpp:1663-1668`, a v1 non-goal), and no parity program calls
  `qsa_block_scores_tc` (`git grep` at b4c63b5). `HMMA.1688.F32.TF32`'s product alignment, rounding and summation
  order are undocumented (`native_qsa_score.md` section 4.3 records the same open point for that file's MMA), so a
  Metal port could match its bits only after measuring that model on sm_120. The port does not need it: the prompt
  path is outside v1, and the warp kernels give the reference's scores.

### 3.6 Library routines and slow paths

None. Over the PTX and SASS of every unit (`b1_rg2` to `b7_scores_tc`) and of the whole-file listing: no PTX
`call`, no `__nv_` or `__internal` symbol, no `.f64` instruction, no `.approx` instruction, no `div.rn.f32` and no
`rcp`; in the SASS no `CALL`, no `FCHK`, no `D*` instruction and no `F2F`. The one `MUFU` in all the listings is a
`MUFU.RCP` at `:575` in `block_scores_seg_kernel` (unit `b6_scores_plain`, 0x11b0), between `I2F.U32.RP` and
`F2I.FTZ.U32.TRUNC.NTZ`: `ptxas`'s exact unsigned division for that loop's trip count (section 1, "Not float
arithmetic"). The truncated `block_topk_reg_kernel<33>` is covered by its complete PTX (section 3.4). So nvcc
inlined no libdevice routine into any kernel of this file, and no kernel on the decode path has an approximate
operation: invariant 5's list of approximate operations has no site here.

## 4. Metal transcription rules for this file

### 4.1 Helpers by kernel

Every ported source starts with `#pragma STDC FP_CONTRACT OFF` (main's `kernels/common/basalt_fp.metal:43`) and
writes the fma of `:672` and `:47` explicitly. "Main" is `origin/main` at e7eb8b4 (its `kernels/` are unchanged at
50a6152); "sampler" is `origin/woodsonl/basalt-sampler` at 2acc918.

| Kernel | Path | Float helpers | Where they live | Subnormals reachable through |
|---|---|---|---|---|
| `block_scores_rg_kernel<2/4/8>` (3.2) | ON | `fma32`, `mul32`, `add32`, `relu32`; `score + 1e9f` plain; `simd_shuffle_xor` / `simd_shuffle` on the words (main's `shfl_xor`, `basalt_fp.metal:85`, for floats) | `add32` = `ieee_add` (main `kernels/common/basalt_q8_1.metal:95`; sampler `basalt_ieee.metal:32` after its move); `mul32` = `ieee_mul` (main `basalt_q8_1.metal:108`; sampler `basalt_ieee.metal:45`); `fma32` = `ieee_fma_count` (sampler only, `kernels/common/basalt_sampler.metal:80`, with `basalt_f64.metal`'s software binary64); `relu32` new (one line, section 3.0) | elements of the queries or pooled keys below 2^-38 (products below 2^-77) |
| `block_scores_kernel` (3.3), the test reference | OFF (`qsa_scores_parity`'s reference) | the same as the rg kernel, the head sum from r0 (no `0.0f +`) | as above | as above |
| `block_scores_seg_kernel`, `block_scores_tail_kernel` (3.5) | OFF | the same as `block_scores_kernel` | as above | as above |
| `block_scores_tc_kernel` (3.5) | OFF (prompt path) | not ported: `HMMA.1688.F32.TF32` has no Metal counterpart with known bits | - | - |
| `block_topk_cluster_kernel` and the three one-CTA kernels (3.4) | ON (cluster); OFF (the others, test references) | `order_key32` on the score's bits, then integer code only | new as `order_key32` = `spec-kernels-indexer-b.md`'s `qsa_order_key`, which is `fnan(b) ? 0u : okey(b)` with sampler `okey` (`basalt_sampler.metal:37`) and `fnan` (`:41`), the composition the sampler already writes (`:334`, `:753`) | none: no float arithmetic; the keys order subnormals as themselves |

No kernel of this file needs `cuda_nan` (no NaN reaches a float store, section 3.2), `ex2`/`exp`/`log` helpers or
binary64 beyond what `fma32`'s implementation uses internally.

### 4.2 What basalt's tests demand

- **Scores, bitwise between the decode kernel and the reference, and within 1e-5 of double.**
  `qsa_scores_parity` computes the five window cases (`src/kernels/qsa_scores_parity.cpp:81-87`, N(0, 1) keys and
  queries, `:67`) with the decode entry (`qsa_block_scores(..., -1, true)`, the rg kernel) and the reference entry
  (`qsa_block_scores(..., active, false)`, `block_scores_kernel`, `:102-103`), then requires
  `const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;` (`:106`) and
  `const bool ok = same && max_rel < 1e-5;` (`:123`), `max_rel` being |got - sc| / max(1, |sc|) against a host
  double sum over the blocks each query reaches (`:107-121`). On CUDA the first holds by construction (section
  3.3). The Metal test compares the Metal decode kernel with a Metal one-(block, query) reference that writes the
  same helper chain (bitwise), and both with the host double (`max_rel < 1e-5`). The bitwise half catches a chain
  or butterfly written differently in one kernel; only an adversarial fixture with elements below 2^-38 would
  exercise `fma32`, `mul32` and `add32`'s emulation, and the double half cannot see subnormal bits.
- **Top-k ids, identical to the one-CTA reference.** `qsa_scores_parity` (`:223`: `g_fail += same_n != total;`; the
  two calls `:213-214`; 6 contexts × nq {1, 3, 8, 16} × continuous, tied and all-equal scores, `:191-206`), `qsa_topk_parity`
  (`src/kernels/qsa_topk_parity.cpp:77`, `:98`: `const bool ok = same_cont == nq && same_tied == nq && same_equal
  == nq;`), `qsa_topk_active_parity` (`src/kernels/qsa_topk_active_parity.cpp:53-54`, `:62`: the dispatch with and
  without `active_blocks`), and `decode_cluster_parity`'s top-k half (`src/kernels/decode_cluster_parity.cpp:156-169`:
  `const bool ok = diff_old == 0 && diff_ref == 0;`). `decode_cluster_parity`'s fixtures include NaN scores
  (`kNaNs`, `:104`; `kAllNaN`, `:114`), infinities (`kInfs`, `:105-109`), signed zeros (`kSignedZero`, `:110`) and
  subnormal scores (`kDenormal`, `:111`: `(float) (pick(rng) % 7) * 1e-41f`, seven levels of which six are
  subnormal), so the keys must be computed on the bits: with a float compare or a float `s + 0.0f` on this GPU every
  `kDenormal` score would key as +0 and the ids would follow the cell order instead. On CUDA every check compares
  two GPU kernels that share `order_key`, so a key error common to both kernels would pass; the Metal tests add a
  host transcription (numpy: `order_key32` on the score bits, then the top `width` cells by key descending and cell
  ascending, written in ascending cell order, `spec-kernels-indexer-b.md` sections 0 and 5) on these fixtures.
- **No criterion reads a score's sign of zero or a NaN score's bits**: scores are compared bitwise only between
  two kernels of the same sequence, and the top-k reads keys, where -0 and +0 and all NaNs coincide.

### 4.3 Points for the contract and the kernel lane

- **No approximate operation and no library function** in `qsa_select.cu` (section 3.6): invariant 5's mapping of
  approximate operations has no site here, and the LIBRARY FUNCTIONS rule is not triggered.
- **The IEEE build is the whole cost.** Three places need the subnormal-keeping forms on this GPU: the `:672` chain
  and the butterfly adds (`fma32`, `mul32`, `add32`, unless a SIMD-group-uniform guard proves them plain, section
  3.2), the relu (`relu32` on the bits; no Metal float form gives it, section 3.0), and the top-k key (`order_key32`
  on the bits, section 3.1). The `1e9f` add is plain `+` (section 3.2).
- **Open: where `fma32` comes from.** Main's `ieee_fma` is restricted to an integer middle operand below 2^24
  (`kernels/common/basalt_q8_1.metal:158`); the only general correctly rounded fma with subnormals is the sampler
  branch's `ieee_fma_count` (`basalt_sampler.metal:80`), which needs `basalt_f64.metal`. Recommendation: the
  qsa-indexer pull request lands after the sampler branch (or carries the `basalt_ieee.metal` / `basalt_f64.metal`
  move and `ieee_fma_count` itself, renamed for its general use), and adds no second fma helper.
- **Open: one key helper.** `order_key32` (`spec-kernels-indexer-b.md`'s `qsa_order_key`) is the sampler's
  `fnan(b) ? 0u : okey(b)`. Recommendation: one shared definition next to `okey` and `fnan`, used by the sampler's
  top-k sites and the QSA top-k alike.
- **Invariant 1 holds in the kernel's arithmetic.** Each (block, query) score is the same sequence in `<2>`, `<4>`
  and `<8>` (section 3.2, "Instances") and depends only on that query's row and step record, so a token's scores
  and ids do not depend on the window width or the row's place in it; the Metal kernel keeps one per-(block, query)
  sequence for every row count.
- **Order.** The scores have a fixed order per (block, query): the chain, the butterfly (any lane mapping with the
  same pairs), the relu and the head order. The top-k is a pure function of the keys and weights, so its geometry
  (one threadgroup, slices and a merge) is free.
