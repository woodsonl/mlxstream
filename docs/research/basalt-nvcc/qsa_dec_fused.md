# basalt `qsa_dec_fused.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's `qsa_dec_fused.cu` (the drafter's fused query
post `q_post_kernel<bool>` behind `qsa_dec_q_post`; the fused KV post `kv_post_kernel` / `kv_post_seg_kernel` is
recorded only as far as its path verdict), read from the sm_120 SASS (with the PTX where it shows what the front end
chose), and the Metal form the port writes at each site. This is the file's page under invariant 5 of
`docs/design/flash-next.md`; `sampler.md` (branch `woodsonl/basalt-sampler`) is the model for its layout and
notation; `native_qsa.md` and `native_rope.md` are the pages of the separate norm and rotation this kernel fuses.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/qsa_dec_fused.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/qsa_dec_fused.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/qsa_dec_fused.hpp`, `f16_bits.hpp`, `kv_q8.hpp`, `kv_stream.hpp`, `mrope.hpp`,
  `native_qsa_indexer.hpp`, `pdl.hpp`, `qsa.hpp`, `qsa_decode_attn.hpp`, `qsa_seg.hpp`, `rope_scaling.hpp`), so every
  PTX and SASS line names its basalt file and line. CUDA's own headers come from the compiler's installation. Only
  basalt's public MIT source was sent.
- **Flags.** `qsa_dec_fused.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:154`) and is
  in the `--use_fast_math` list (`:186-202`, the file at `:199`, the option at `:202`). A Release build (`:12-14`)
  passes CMake's `-O3 -DNDEBUG`, `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one
  architecture is 120 (`:3-4`, `:26-31`). `--use_fast_math` means `-ftz=true -prec-div=false -prec-sqrt=false
  -fmad=true` and the fast intrinsic for each single-precision library call; no other `-fmad`, `-ftz`, `-prec-div`
  or `-Xptxas` flag is set. So every single-precision `FADD`, `FMUL` and `FFMA` below carries `.FTZ`.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48` (`Build cuda_13.0.r13.0/compiler.36260728_0`). Basalt requires
  CUDA 13.0 or newer (`CMakeLists.txt:32-33`). The query-post unit below was also compiled with `nvcc133` (`Cuda
  compilation tools, release 13.3, V13.3.33`, `Build cuda_13.3.r13.3/compiler.37862127_0`) as a cross-check
  (section 3).
- **userArguments**, verbatim, every request: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'
  --use_fast_math` (the responses' `compilationOptions` echo it). Compiler Explorer adds `-g -lineinfo
  --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them changes a kernel's
  code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026 (both compilers).
- **Translation units.** The whole file's SASS listing stops at Compiler Explorer's 5,000-line cut
  (`[truncated; too many lines]`) inside `kv_post_kernel<true>`, the last kernel of the listing; both
  `q_post_kernel` instantiations come before the cut and are complete. The query post was also compiled as its own
  unit (`qsa_dec_fused.q_post`): the file with the bodies of the KV-post kernels blanked line for line (`:226` and
  `:240-254`; an empty line keeps every line number, and a blanked kernel compiles to a 16-instruction stub). The
  unit's `q_post_kernel<false>` and `q_post_kernel<true>` SASS equal the whole file's instruction for instruction and
  source line for source line once registers and labels are renamed (298 and 346 listing lines). The tables quote
  the unit, which the 13.3 cross-check also compiled.
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.ftz.f32`) and which
  approximation it chose (`div.approx.ftz.f32`, `rcp.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `sin.approx.ftz.f32`,
  `cos.approx.ftz.f32`, `lg2.approx.ftz.f32`, `ex2.approx.ftz.f32`); the SASS shows what `ptxas` made of it. A row
  quotes the SASS of one copy where a site is unrolled; the copies were compared and differ only in registers. The
  instructions after a kernel's last `EXIT` are CUDA's out-of-line subroutines, which `-lineinfo` attributes to the
  line before them; the tables attribute each to the `CALL.REL.NOINC` that reaches it.
- **Not float arithmetic.** `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` at a strided loop's
  head are `ptxas`'s exact integer division; `HFMA2 R, -RZ, RZ, a, b` moves a constant; `IMAD.WIDE` forms addresses.
  The tables leave them out unless they read a stored value.

## 2. Kernels in the file and their paths

`qsa_dec_fused.cu` fuses the per-token work between a QSA layer's projections "with the arithmetic of the separate
kernels bit for bit" (`include/basalt/kernels/qsa_dec_fused.hpp:1-2`): `qsa_dec_kv_post` (K norm and RoPE, the INT8
K/V appends, the indexer appends) and `qsa_dec_q_post` (the query heads out of `q_full` with norm and RoPE, and
optionally the indexer query heads). With the fp16 KV cache only the drafter's query post runs: the drafter takes
`qsa_dec_q_post` under `!st_.kv_rot && native_rope_enabled() && qsa_dec_fused_supported(s)`, with no `kv_int8`
condition, while every other fused call needs INT8 K/V.

| Kernel or function | Lines | Path | Evidence |
|---|---|---|---|
| `q_post_kernel<false>` via `qsa_dec_q_post`, body `q_post_dev<false>`, the query-head blocks | `:257-260`, `:198-213`, launch `:322-335` | ON: drafter, the query split, norm and RoPE of its one row's `n_head` = 24 heads (`include/basalt/core/layout.hpp:41`; `T` = 1 on the full layer, `src/core/mtp.cpp:743`) | `mtp.cpp:861-864`; `qp.q_idx` is left null (`qsa_dec_fused.hpp:46`), so the grid is `T * n_head` blocks (`:329`) and every block takes `b < n * n_head` (`:205`) |
| the same, the indexer-query blocks | `:214-221` | OFF: only verify's fused path sets `q_idx`, and it needs `st.kv_int8` (`src/core/verify.cpp:1352-1353`; the calls at `:1433`, `:1499`) | run by `qsa_dec_fused_parity` (`src/kernels/qsa_dec_fused_parity.cpp`) |
| `q_post_kernel<true>` | `:257-260` | OFF: `rt.cos` is always null (`src/kernels/cuda/native_rope.cu:99`) | `:330-331` |
| `kv_post_kernel<TAB>`, `kv_post_seg_kernel<TAB>`, with `kv_post_dev`, `append_q8_row`, `indexer_append_row`, `take_slot` | `:224-255`, `:77-195` | OFF: INT8 K/V only, "INT8 K/V without rotation only" (`qsa_dec_fused.hpp:9`): verify's `fused_pre` (`verify.cpp:1352-1353`, calls `:1369-1370`), the drafter's `kv_fused` (`mtp.cpp:750-751`, call `:838`) | with fp16 KV the decode's K post is `norm<256>` and `apply<false>` (`native_qsa.md`, `native_rope.md`), its appends `qsa.cu`'s `kv_append_kernel` (`qsa.md`) and the indexer's `native_qsa_indexer_append` |
| `norm_scale256` | `:34-46` | ON inside the query post: "native_qsa.cu's norm<256> for one row of n_cols <= 256 (thread d holds column d, x = 0 past n_cols)" | `:207` (and `:175`, `:217`, both OFF) |
| `rope_row<false>` | `:54-75` | ON inside the query post: "native_rope.cu's apply<TAB> on a row held in shared memory" | `:210` (and `:178`, `:220`, both OFF) |
| `warp_sum` | `:24-28` | ON, inlined twice into `norm_scale256` | `:38`, `:43` (`:131`, `:136` in the OFF indexer append) |
| `abs_ieee`, `max_ieee`, `div_ieee` | `:29-32` | OFF: only `append_q8_row` uses them (`:82-91`) | grep |
| `rope_args` (host) | `:262-271` | ON (host): `ra.theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);` (`:264`), the same host call and value as `native_rope.cu:118` | `:328` |
| `pdl_launch_dependents` | `:204` (`include/basalt/kernels/pdl.hpp:73-77`) | ON: `griddepcontrol.launch_dependents`, a launch-order hint with no arithmetic | none in the Metal port (one command per dispatch, ordered by the program) |

Basalt's `qsa_dec_fused_parity` holds `q_post_kernel` bitwise against the separate kernels: `norm<256>` then
`native_rope_apply` on the query rows (`src/kernels/qsa_dec_fused_parity.cpp:121-124`) and on the indexer-query rows
(`:125-126`). Section 3 compares the fused kernel's SASS with theirs.

## 3. Per-kernel SASS tables

### 3.1 `q_post_kernel<false>` (`:257-260`; `q_post_dev<false>`, `:198-222`, with `norm_scale256` and `rope_row<false>`)

288 SASS instructions in both the unit and the whole-file compile, no subroutine. Launch (`:329`, `:333`):
`n * (n_head + (q_idx ? idx_n_head : 0))` blocks of `THREADS` = 256 threads, no dynamic shared memory (static
`row[256]` and `sums[32]`, 1,152 bytes), no cluster. Block `b < n * n_head` is query head `b % n_head` of row
`b / n_head`; thread `d` holds column `d` of the 256. The rows below are the query-head branch, the drafter's path;
the indexer-query branch (`:214-221`, OFF) is the same sequence on 128 columns (end of the section).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:204` | `pdl_launch_dependents();` | `PREEXIT` (`pdl.hpp:75`) | none |
| `:205-206` | `if (b < n * n_head) { const float x = a.q_full[(size_t) b * 2 * HD + d];` | `ISETP.GE.AND P0, PT, R5, UR4, PT`; `@P0 BRA`; `LDG.E R0` | the query half of `q_full`'s head (`[n][n_head][2 * 256]`: query, then gate) |
| `:36-37` | `float partial = 0.0f; if (d < n_cols) partial += x * x;` | PTX `fma.rn.ftz.f32 %f112, %f8, %f8, 0f00000000`, `selp.f32 ..., 0f00000000, %p30`; SASS `FFMA.FTZ R10, R0, R0, RZ`; `ISETP.GE.U32.AND P0, PT, R2, 0x100, PT`; `FSEL R10, R10, RZ, !P0` | `partial = d < n_cols ? fma_ftz(x, x, 0.0f) : 0.0f`: `norm<256>`'s one-column chain, `fma(x, x, +0)` (`native_qsa.md` section 3.1) |
| `:38` (`:24-28`) | `partial = warp_sum(partial);` | `SHFL.BFLY PT, R11, R10, 0x10, 0x1f`; `FADD.FTZ R11, R10, R11`; the same with `0x8`, `0x4`, `0x2`, `0x1` | `warp_sum_ftz(partial)` |
| `:39-41` | `lane = d % 32; if (lane == 0) sums[d / 32] = partial; __syncthreads();` | `@!P0 STS [R12], R9`; `BAR.SYNC.DEFER_BLOCKING 0x0` | as `norm<256>` |
| `:42-43` | `partial = lane < THREADS / 32 ? sums[lane] : 0.0f; partial = warp_sum(partial);` | `ISETP.GT.U32.AND P0, PT, R15, 0x7, PT`; `@!P0 LDS R13, [R15]`; the butterfly as `:38` | as `norm<256>`: lanes 8-31 add +0.0 |
| `:44-45` | `const float mean = partial / n_cols; return rsqrtf(mean + epsilon);` | PTX `mov.f32 %f135, 0f43800000`, `div.approx.ftz.f32 %f136, %f134, %f135`, `add.ftz.f32 %f137, %f52, %f136`, `rsqrt.approx.ftz.f32`; SASS `FFMA.FTZ R12, R12, 0.00390625, R19` (`ptxas` evaluated the approximate reciprocal of the constant 256 at compile time, 2^-8, and fused the multiply with the add); `MUFU.RSQ R13, R12` | `fast::rsqrt(fma_ftz(partial, rcp_ftz(float(n_cols)), eps))`, `norm<256>`'s form: `rcp_ftz(256.0f)` is 2^-8 exactly, the constant here; so the fused and separate Metal kernels compute the same `fma` |
| `:208` | `row[d] = scale * x * a.q_norm[d];` | `LDG.E R16` (gamma); `FMUL.FTZ R13, R0, R13` (x · scale); `FMUL.FTZ R13, R13, R16`; `STS [R0], R13` | `row[d] = mul_ftz(mul_ftz(scale, x), g)`: the first product is commutative, so this is `norm<256>`'s `(scale · x) · gamma` |
| `:209` | `__syncthreads();` | `BAR.SYNC.DEFER_BLOCKING 0x0` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `:59`, `:62` | `if (pair < ROT / 2) { ... theta_extrap = mrope_pos(ra.mtab, pos, pair) * powf(ra.theta_scale, float(pair));` | `ISETP.GT.U32.AND P0, PT, R2, 0x1f, PT`; `LDG.E R13` (`pos_q[b]`); `MUFU.LG2 R12, UR4`; `I2FP.F32.U32 R9, R2` (`float(pair)`, exact; unsigned here, signed in `apply<false>`, the same value for 0-31); `FMUL.FTZ R12, R9, R12`; `MUFU.EX2 R12, R12`; `I2FP.F32.S32 R13, R13`; `FMUL.FTZ R14, R13, R12` | `rope_cs(pos, pair, theta_scale)`, as `native_rope.md` section 3.1 |
| `rope_scaling.hpp:153-162` | `theta = freq_scale * theta_extrap;` the `ext_factor` test; `cos_out = cosf(theta) * mscale; sin_out = sinf(theta) * mscale;` | `FMUL.FTZ R2, R14, UR5`; `UFSETP.NEU.FTZ.AND UP0, UPT, UR4, URZ, UPT`; `BRA.U !UP0` (YaRN skipped; its body is `native_rope.md`'s, OFF); `FMUL.RZ R10, R2, 0.15915493667125701904`; `MUFU.SIN R11, R10`; `MUFU.COS R9, R10`; `FMUL.FTZ R11, R11, R4`; `FMUL.FTZ R9, R9, R4` (`mscale`) | inside `rope_cs` |
| `:65-67` | `const float a = row[pair], b = row[pair + ROT / 2]; r0 = a * c - b * s; r1 = a * s + b * c;` | `LDS R3, [R0]`; `LDS R8, [R0+0x80]`; `FMUL.FTZ R2, R11, R8` (s·b); `FMUL.FTZ R8, R9, R8` (c·b); `FFMA.FTZ R9, R9, R3, -R2` (c·a − s·b); `FFMA.FTZ R11, R11, R3, R8` (s·a + c·b) | `r0 = fma_ftz(c, a, -mul_ftz(s, b))`, `r1 = fma_ftz(s, a, mul_ftz(c, b))`: `apply<false>`'s pairing (`native_rope.md` section 3.1) |
| `:69-74` | `__syncthreads(); if (pair < ROT / 2) { row[pair] = r0; row[pair + ROT / 2] = r1; } __syncthreads();` | `BAR.SYNC`; `@!P0 STS [R0], R9`; `@!P0 STS [R0+0x80], R11`; `BAR.SYNC` | two barriers around the in-row exchange (or, per thread, read `a` and `b` and write both outputs in a geometry where one thread owns a pair) |
| `:210-211` | `rope_row<TAB>(row, a.pos_q[b], d, ra); a.qcur[(size_t) b * HD + d] = row[d];` | `LDS R13, [R0]`; `STG.E` | `qcur[...] = cuda_nan(row[d])` |

**Against the separate kernels.** Read in order, the query branch's float instructions are those of `norm<256>` at
`n_cols` = 256 (`native_qsa.md` section 3.1: one `FFMA.FTZ` of `x · x` onto +0, the two butterflies with offsets 16 to
1, the slot write and barrier, the select of lanes below 8, `FFMA.FTZ(total, rcp(256), eps)`, `MUFU.RSQ`, two
`FMUL.FTZ`) followed by those of `apply<false>` (`native_rope.md` section 3.1: `MUFU.LG2`, `FMUL.FTZ`, `MUFU.EX2`,
`FMUL.FTZ`, `FMUL.FTZ` by `freq_scale`, `FMUL.RZ` by `0x3E22F983`, `MUFU.SIN`, `MUFU.COS`, two `FMUL.FTZ` by
`mscale`, then `FMUL.FTZ` s·b and c·b and the two `FFMA.FTZ` with a·c and a·s fused), with the same operand pairing
at every `FFMA`. The one difference is the reciprocal at `:44`: the constant 2^-8 here, `MUFU.RCP` of
`float(256)` there; the two are the same bits exactly when `MUFU.RCP(256.0f)` returns 2^-8, which
`qsa_dec_fused_parity`'s bitwise comparison requires of the hardware. The values move between the stages through
shared memory instead of global memory, which rounds nothing. The query post therefore introduces no arithmetic of
its own.

**The indexer-query branch** (`:214-221`, OFF; 128 columns): `x = d < ID ? qp[d] : 0.0f` (`ISETP.GT.U32.AND P0, PT,
R2, 0x7f, PT`; `@!P0 LDG.E`), the same `FFMA.FTZ R8, R0, R0, RZ` with `FSEL` on `d < 128`, the same butterflies,
`FFMA.FTZ R12, R15, 0.0078125, R20` (2^-7 = `rcp(128)`), `@!P0 MUFU.RSQ`, the two `@!P0 FMUL.FTZ` and the same
rotation (`FMUL.FTZ R9, R2, R5` s·b; `FMUL.FTZ R11, R4, R5` c·b; `FFMA.FTZ R5, R4, R3, -R9`; `FFMA.FTZ R11, R2, R3,
R11`), stored in place to `q_idx` for `d < 128` (`:221`).

### 3.2 `q_post_kernel<true>`, `kv_post_kernel`, `kv_post_seg_kernel`: OFF-PATH

`q_post_kernel<true>` (328 instructions) has `<false>`'s float opcodes per source line plus the table read; the
KV-post kernels are INT8-only (section 2). None is tabulated. `append_q8_row`'s explicit IEEE operations (`abs.f32`,
`max.f32`, `div.rn.f32`) are recorded with the INT8 KV format, which the contract leaves out.

### 3.3 nvcc 13.3

The query-post unit compiled by `nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) gives
`q_post_kernel<false>` 288 instructions and `q_post_kernel<true>` 336. Per source line, the float-class opcodes with
their immediates are the same on all 18 float lines of each instance, and the rotation's four instructions are
register for register the same in both branches (`FMUL.FTZ R2, R11, R8`, `FMUL.FTZ R8, R9, R8`,
`FFMA.FTZ R9, R9, R3, -R2`, `FFMA.FTZ R11, R11, R3, R8` in the query branch), so the pairing does not depend on the
toolkit. The integer differences are spelling only (`IADD3` to `IADD`, `ISETP.NE.S64.AND` to `ISETP.NE.U64.AND`,
`IMAD.WIDE` to `IMAD.WIDE.U32`, `NOP` padding).

## 4. What the record implies for the Metal port

- **No libdevice routine.** The unit's PTX has no `call` and no `__nv_` or `__cudart_` symbol; its only `.f64`
  operations are the uncalled `std::isfinite(double)` body that `--keep-device-functions` emits; the SASS of both
  `q_post_kernel` instances has no `CALL`, no `FCHK` and no `D*` instruction. Its approximate instructions are
  `MUFU.RSQ` (`rsqrtf`), `MUFU.LG2` and `MUFU.EX2` (`__powf`), `FMUL.RZ` with `MUFU.SIN`/`MUFU.COS` (`__sinf`,
  `__cosf`), and `MUFU.RCP` only in the OFF YaRN body (`rope_scaling.hpp:138`, `:159`): the norm's reciprocal is the
  compile-time constant. Nothing here falls under the libdevice rule.
- **The fused query post adds no arithmetic.** Section 3.1 shows it is `norm<256>` followed by `apply<false>`
  instruction for instruction, with the same pairing at every `FFMA`. So the Metal port gives the drafter's query
  the same bits whether it runs one fused kernel or the separate norm and rotation dispatches (after copying the
  query halves out of `q_full`), provided both call the same helpers: `fma_ftz`, `warp_sum_ftz`,
  `fma_ftz(partial, rcp_ftz(float(n_cols)), eps)` (which is `fma_ftz(partial, 2^-8, eps)` at 256 columns, so the
  constant may be written either way), `fast::rsqrt`, `mul_ftz(mul_ftz(scale, x), g)`, `rope_cs` and the
  rotation's `fma_ftz(c, a, -mul_ftz(s, b))`, `fma_ftz(s, a, mul_ftz(c, b))`. A Metal `qsa_dec_fused_parity` then
  compares the Metal fused kernel with the Metal separate kernels bitwise, as basalt does on CUDA.
- **What differs from CUDA.** Only the approximations: `MUFU.RSQ` (every row), the angle's `MUFU.LG2`, `MUFU.EX2`,
  `MUFU.SIN`, `MUFU.COS` (`native_rope.md` section 4: within 1.7e-7 of a float64 model of CUDA's chain at every
  position). The double-precision tests that hold these rows are the contract's query and key norm test
  (`qsa_parity`'s `if (!(worst <= 1e-6)) {`, `src/kernels/qsa_parity.cpp:611`) and `rope_parity`'s `3e-3`
  (`src/kernels/rope_parity.cpp:400`).
- **Geometry.** One threadgroup of 256 threads per (row, head); the column a thread owns, the shuffles and the
  `sums` slot must follow basalt's lane mapping (`native_qsa.md` section 4: `t = sg * 32 + lane`). The rotation reads
  `a` and `b` from the normed row in threadgroup memory: two barriers as in `:69-74`, or the pair's thread reads both
  halves before anyone writes (any order gives the same values, since every output depends only on its own pair).
- **Stores and NaN.** The `qcur` store (and the OFF branch's `q_idx` store) can carry a NaN from a NaN or infinite
  input: `cuda_nan`.
- **Per-step data.** `pos_q[b]` (a buffer of positions per row and head, invariant 4); `theta_scale`
  (`0x3F1AB32B`, from the host's `powf` at `:264`, the value `native_rope.md` section 4 settles), `eps` (1e-6,
  `mtp.cpp:63`) and the head counts are per-program constants.
- **Not ported.** `pdl_launch_dependents` (`PREEXIT`) is a CUDA launch-order hint; Metal's dispatch order in the step
  program already serializes the drafter's chain.

**Helper inventory.**

| Helper | Where it lives | Used at |
|---|---|---|
| `fma_ftz` | `kernels/common/basalt_fp.metal:73` (origin/main) | `:37`, `:45`, `:66`, `:67` |
| `warp_sum_ftz` (with `shfl_xor`, `add_ftz`) | `basalt_fp.metal:87` (`:85`, `:49`) | `:38`, `:43` |
| `rcp_ftz` | `basalt_fp.metal:83` | `:44` |
| `fast::rsqrt` | MSL standard library | `:45` |
| `mul_ftz` | `basalt_fp.metal:80` | `:62`, `:66`, `:67`, `:208` |
| `rope_cs`, `mul_rz` | new; shared with `native_rope` (`native_rope.md` section 4) | `:62`, `rope_scaling.hpp:153-162` |
| `ftz` | `basalt_fp.metal:45` | inside `rope_cs` |
| `cuda_nan` | `basalt_fp.metal:120` | `:211` (and `:221`, OFF) |
