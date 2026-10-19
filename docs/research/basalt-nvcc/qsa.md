# basalt `qsa.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of the on-path part of basalt's `qsa.cu` (the fp16 KV
append `kv_append_kernel` and the software converter `f16_from_f32` it stores through), read from the sm_120 SASS
(with the PTX where it shows what the front end chose), and the Metal form the port writes at each site. This is the
file's page under invariant 5 of `docs/design/flash-next.md`; `sampler.md` (branch `woodsonl/basalt-sampler`) is the
model for its layout and notation, and `nan.md` records the converter's NaN encoding.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/qsa.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/qsa.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/qsa.hpp`, `f16_bits.hpp`, `kv_stream.hpp`, `qsa_decode_attn.hpp`, `rope.hpp`,
  `rope_scaling.hpp`, `mrope.hpp`), so every PTX and SASS line names its basalt file and line. CUDA's own headers come
  from the compiler's installation. Only basalt's public MIT source was sent.
- **Flags.** `qsa.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:129`) with nvcc's IEEE
  defaults: it is not in the `--use_fast_math` list (`:186-202`), and the only other fast-math flag (`:728`) belongs
  to the `mmq` target. A Release build (`:12-14`) passes CMake's `-O3 -DNDEBUG`, `add_compile_definitions` passes
  `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120 (`:3-4`, `:26-31`). No `-fmad`, `-ftz`,
  `-prec-div` or `-Xptxas` flag is set, so nvcc's defaults hold: `-fmad=true`, `-ftz=false`, `-prec-div=true`,
  `-prec-sqrt=true`.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48`. Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`). The
  unit below was also compiled with `nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) as a cross-check
  (section 3).
- **userArguments**, verbatim. The whole file: `-arch=sm_120 -lineinfo -O3 -DNDEBUG`. The append unit (the one this
  page's tables quote): `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'`. The whole-file request
  lacks the version define; nothing in `qsa.cu` or the headers it includes names `BASALT_VERSION` (a grep finds no
  use under `src/kernels/cuda/qsa.cu` or `include/basalt`), so the define changes no code. Compiler Explorer adds
  `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them changes
  a kernel's code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026 (both compilers).
- **Translation units.** The whole file's SASS listing stops at Compiler Explorer's 5,000-line cut
  (`[truncated; too many lines]`) inside `indexer_key_append_kernel`, the sixth kernel of the listing, before
  `kv_append_kernel`. So the append was also compiled as its own unit (`qsa.kv_append`): `qsa.cu` with the bodies of
  the other kernels blanked line for line (an empty line keeps every line number; a blanked kernel compiles to a
  16-instruction stub): `:160-163`, `:165-167`, `:169-199`, `:201-202`, `:204-212`, `:214-216`, `:218-225`,
  `:227-241`, `:252-264`, `:266-272`, `:274-287`, `:308-317`, `:319-336`, `:338-369`, `:371-391`, `:393-400`,
  `:409-422`, `:442-459`, `:461-467`, `:469-481`, `:483-499`, `:501-507`, `:514-519` and `:793-798`. The whole
  file's PTX is complete (4,202 lines), and its `kv_append_kernel` is the unit's instruction for instruction and
  source line for source line (370 PTX lines each) once registers and labels are renamed. Equal PTX does not by
  itself prove equal SASS, but neither PTX holds a floating-point instruction for this kernel (only `div.s32`,
  `div.s64`, `rem.s64`, integer `cvt` and the `cvt.u16.u32` truncations of the converter), so the property this page
  records, that the append is integer work, holds for basalt's build whatever `ptxas` schedules.
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.f32`) and which conversion
  it chose; the SASS shows what `ptxas` made of it. The instructions after a kernel's last `EXIT` are CUDA's
  out-of-line subroutines (here only 64-bit integer division and remainder), which `-lineinfo` attributes to the line
  before them; the tables attribute each to the `CALL.REL.NOINC` that reaches it.
- **Not float arithmetic.** `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` at a strided loop's
  head are `ptxas`'s exact integer division; `HFMA2 R, -RZ, RZ, a, b` moves a constant; `IMAD.WIDE` forms addresses.
  The tables leave them out unless they read a stored value.

## 2. Kernels in the file and their paths

`qsa.cu` is basalt's first ("table") QSA implementation. With an fp16 KV cache the contract's decode path (the verify
window, `Verifier::run`) and the drafter use only its append; every other kernel is reached from the per-token layer
forward `qsa_layer` (`src/core/layer.cpp:903`, from `block_layer_pre` at `:1280`), which the contract's decode does
not run (`docs/design/flash-next.md:291-292`), from a flag-off branch, or from tests. The engine forces the native
QSA set on (`src/program/generate.cpp:1893`, applied at `:2399-2401`).

| Kernel or function | Lines | Path | Evidence |
|---|---|---|---|
| `kv_append_kernel` via `kv_append_step` | `:121-142`, `:594-605` | ON: decode, every row of every window | `src/core/verify.cpp:1474-1476` (the `else` after `kv_q4` and `kv_int8`, `:1468-1473`) |
| the same | | ON: drafter, every row it appends | `src/core/mtp.cpp:853-855`; the fused `qsa_dec_kv_post` needs `st_.kv_int8` (`:750-751`), so fp16 KV takes the loop at `:839-856` |
| the same, host-copy branch `:137-141` | `:137-141` | OFF: `host.k_pool` is null when KV is fully resident | `include/basalt/kernels/kv_stream.hpp:30-31`; `src/core/layer.cpp:672-673` allocates the host pools only for `p.mode != 0`, and `kv_plan` returns mode 0 when `--kv-resident` is 0 (`layer.cpp:482`; the default at `src/program/generate.cpp:374`); `docs/design/flash-next.md:416-417` |
| `f16_from_f32` (`include/basalt/kernels/f16_bits.hpp:37-65`) | inlined at `:134`, `:135` (and `:139`, `:140`) | ON with the append | the K and V stores |
| `kv_append` (host-scalar wrapper) | `:714-718` | tests only | `src/kernels/qsa_parity.cpp:476`, `:1022`, `:1233`, `:1278` |
| `qsa_step_fill` (host) | `:697-712` | ON (host): the step record the append reads at `kStepPos` | `src/core/verify.cpp:2166`; the drafter fills its own (`src/core/mtp.cpp:1480-1486`) |
| `indexer_key_append_kernel` | `:154-242` | OFF: the window appends indexer keys with `native_qsa_indexer_append` | `src/core/verify.cpp:1478-1482`; `layer.cpp:975` only |
| `qsa_index_kernel` | `:248-288` | OFF: the path scores with `native_qsa_score` | `layer.cpp:999`; tests |
| `topk_kernel` (with `order_key`) | `:296-401` | OFF: the path selects with the radix block top-k | `layer.cpp:999`; tests |
| `kv_gather_kernel` | `:405-423` | OFF: `qsa_dec_attn` reads the pools through the page table | `layer.cpp:1015`; tests |
| `qsa_attend_kernel` | `:438-508` | OFF | `layer.cpp:1028`; tests |
| `qsa_gate_apply_kernel` | `:512-520` | OFF: tests only (`qsa_parity.cpp:1142`) | no engine caller |
| `qsa_gate_apply_f32_kernel` | `:791-799` | OFF: the `else` of `native_qsa_enabled()` | `verify.cpp:1577-1581`; with fp16 pools `fused_gate` skips the separate gate (`verify.cpp:1558-1559`, `:1572`) |

This page records the on-path kernel only; the off-path kernels' SASS is in the whole-file compile up to Compiler
Explorer's cut and is not tabulated.

## 3. Per-kernel SASS tables

### 3.1 `kv_append_kernel` (`qsa.cu:121-142`)

Unit `qsa.kv_append`, 388 SASS instructions before the two subroutines. Launch (`:594-605`): block `THREADS`,
grid `grid_for(n_head_kv * head_dim, THREADS)`, no shared memory, no cluster; one thread per (head, dim) element of
one cell. The kernel holds **no floating-point instruction**: its SASS has no `FADD`, `FMUL`, `FFMA`, `FSETP`,
`FMNMX`, `F2F`, `F2FP` or `MUFU` other than the integer divisions' `MUFU.RCP` (`:128`, `:131`, `:133`, `:138`), and
its PTX none either (section 1). K and V are loaded as 32-bit words (`LDG.E.CONSTANT`) and reach the converter's
integer operations unchanged.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:125` | `const long long pos = (long long) __ldg(step + kStepPos);` | `LDG.E.CONSTANT R6, desc[UR6][R6.64]` (`sm_32_intrinsics.hpp:112`); `SHF.R.S32.HI R7, RZ, 0x1f, R6` | integer; `pos` from the step record (invariant 4) |
| `:126-127` | `i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= kv_heads * head_dim) return;` | `IMAD`; `IMAD`; `ISETP.GE.AND P0, PT, R0, R5, PT`; `@P0 EXIT` | integer |
| `:128` | `h = i / head_dim, d = i - h * head_dim` | `I2F.RP`, `MUFU.RCP`, `IADD3 R4, PT, PT, R2, 0xffffffe, RZ`, `F2I.FTZ.U32.TRUNC.NTZ`, `IMAD.HI.U32` and the two corrections: `ptxas`'s exact signed 32-bit division | integer `/` |
| `:131` | `table[pos / page_size]` | when the high words of `pos` and `page_size` are zero (`LOP3.LUT R2, R7, UR5, RZ, 0xfc, !PT`; `ISETP.NE.U32.AND`), the unsigned 32-bit division as at `:128`; else `CALL.REL.NOINC` to `__internal_0_$__cuda_sm20_div_s64` | integer |
| `:132` | `if (page >= 0)` | `ISETP.GE.AND P0, PT, R11, RZ, PT`; `@!P0 BRA` | integer |
| `:133` | `(page * kv_heads + h) * page_size + (pos % page_size)` | `IMAD.WIDE.U32`, `IMAD`; the remainder as at `:131`, or `CALL.REL.NOINC` to `__internal_1_$__cuda_sm20_rem_s64` | integer (64-bit; section 4) |
| `:134` | `k_pool[row * head_dim + d] = f16_from_f32(kcur[i]);` | `LDG.E.CONSTANT R16` (kcur[i]'s bits); section 3.2 inline; `STG.E.U16` | `k_pool[...] = f16_from_f32(as_type<uint>(kcur[i]))`, the word loaded as `uint` |
| `:135` | `v_pool[row * head_dim + d] = f16_from_f32(vcur[i]);` | as `:134`, a second inlined copy | as `:134` |
| `:137` | `if (host.k_pool != nullptr)` | `ISETP.NE.S64.AND P0, PT, RZ, UR8, PT`; `@!P0 EXIT` | none: the host copy is off-path (section 2) |
| `:138-140` | the host-pool row and two more converter copies | as `:131`, `:133`, `:134` | none |

### 3.2 `f16_from_f32` (`include/basalt/kernels/f16_bits.hpp:37-65`), the copy inlined at `:134`

Every instruction is an integer one; the `HFMA2 R18, -RZ, RZ, 0, 5.9604644775390625e-08` at `:58` writes the half
pair (0, 2^-24), whose bits are the integer 1, as the operand of `1u << (sh - 1)`. The other three copies
(`:135`, `:139`, `:140`) have the same instructions with other registers.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:39-43` | `memcpy(&x, &f, 4)`; `sign = (x >> 16) & 0x8000u`; `rawexp = (x >> 23) & 0xFFu`; `exp = rawexp - 127 + 15`; `man = x & 0x7FFFFFu` | no instruction for the `memcpy` (the loaded word); `SHF.R.U32.HI R12, RZ, 0x10, R16`; `LOP3.LUT R12, R12, 0x8000, RZ, 0xc0, !PT`; `SHF.R.U32.HI R12, RZ, 0x17, R16`; `LOP3.LUT R13, R12, 0xff, RZ, 0xc0, !PT`; `LOP3.LUT R15, R16, 0x7fffff, RZ, 0xc0, !PT` (`exp` is never formed: every test below compares `rawexp`) | `uint` operations on the bits |
| `:50` | `if (rawexp == 0xFFu) return sign \| 0x7C00u \| (man ? 0x200u : 0u);` | `ISETP.NE.AND P0, PT, R13, 0xff, PT`; `@!P0 ISETP.NE.AND P1, PT, R15, RZ, PT`; `@!P0 SEL R9, RZ, 0x200, !P1`; `@!P0 LOP3.LUT R14, R9, 0x7c00, R12, 0xfe, !PT` (three-way OR) | a NaN gives `sign \| 0x7E00`, an infinity `sign \| 0x7C00` |
| `:51` | `if (exp >= 31) return sign \| 0x7C00u;` | `ISETP.GE.U32.AND P0, PT, R13, 0x8f, PT` (rawexp ≥ 143); `@P0 LOP3.LUT R17, R12, 0x7c00, RZ, 0xfc, !PT` | integer |
| `:52-53` | `if (exp <= 0) { if (exp < -10) return sign; ...` | `ISETP.GT.U32.AND P0, PT, R13, 0x70, PT` (rawexp > 112: the normal branch); `ISETP.GT.U32.AND P0, PT, R13, 0x65, PT` (rawexp > 101); `@!P0 PRMT R17, R12, 0x7610, R17` | integer: zero, every fp32 subnormal and every \|x\| < 2^-25 give the signed zero |
| `:54-58` | `man \|= 0x800000u; sh = 14 - exp; h = (man >> sh) & 0x3FFu; rem = man & ((1u << sh) - 1u); if (rem > (1u << (sh - 1)) \|\| (rem == (1u << (sh - 1)) && (h & 1u))) ++h;` | `IADD3 R14, PT, PT, -R13, 0x7e, RZ` (sh = 126 - rawexp); `LOP3.LUT R15, R15, 0x800000, RZ, 0xfc, !PT`; `SHF.R.U32.HI R15, RZ, R14, R15` (the `& 0x3FF` is dropped: sh ≥ 14 leaves at most 10 bits); `MOV R17, 0xffffffff`; `SHF.L.U32 R16, R17, R14, RZ`; `LOP3.LUT R16, R16, 0x800000, R15, 0xe, !PT` (rem); `IADD3 R13, PT, PT, -R13, 0x7d, RZ`; `SHF.L.U32 R13, R18, R13, RZ` (the half point); `ISETP.GT.U32.AND P1, PT, R16, R13, PT`; `@!P1 LOP3.LUT R14, R15, 0x1, RZ, 0xc0, !PT`; `@!P1 ISETP.NE.U32.AND P0, PT, R14, 0x1, PT`; `@!P1 ISETP.EQ.AND P0, PT, R16, R13, !P0`; `@!P1 SEL R14, RZ, 0x1, !P0` | integer round to nearest even into the fp16 subnormals (a carry into 0x0400 gives the smallest normal) |
| `:59` | `return (uint16_t) (sign \| h);` | `IADD3 R14, PT, PT, R12, R14, R15` (sign + increment + h; h + 1 ≤ 0x400 never reaches the sign bit); `PRMT R17, R14, 0x7610, R17` | integer |
| `:61` | `h = (uint16_t) (sign \| ((uint32_t) exp << 10) \| (man >> 13));` | `SHF.R.U32.HI R14, RZ, 0xd, R15`; `IMAD R14, R13, 0x400, R14` (rawexp · 1024 + man >> 13); `IADD3 R13, PT, PT, R14, 0x4000, RZ` (−112 · 1024 = −0x1C000, which is 0x4000 modulo 2^16); `LOP3.LUT R13, R13, R12, RZ, 0xfc, !PT` | integer |
| `:62-63` | `rem = man & 0x1FFFu; if (rem > 0x1000u \|\| (rem == 0x1000u && (h & 1u))) ++h;` | `LOP3.LUT R15, R16, 0x3fff, RZ, 0xc0, !PT` (x & 0x3fff); `LOP3.LUT R16, R16, 0x1fff, RZ, 0xc0, !PT` (rem); `ISETP.NE.AND P0, PT, R15, 0x3000, PT` (rem == 0x1000 with h odd); `ISETP.GT.U32.OR P0, PT, R16, 0x1000, !P0`; `IADD3 R17, PT, PT, R13, 0x1, RZ`; `@!P0 IADD3 R17, PT, PT, R13, RZ, RZ` | integer: round to nearest even; a carry out of the mantissa raises the exponent, and 0x7BFF + 1 gives the infinity 0x7C00 |
| `:64` | `return h;` | the low 16 bits reach `STG.E.U16` | `ushort` store |

### 3.3 nvcc 13.3

The unit compiled by `nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) gives `kv_append_kernel` 389
instructions with the same float-class instructions on every line (the integer divisions' `I2F.RP`/`UI2F.U32.RP`,
`MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ`; compared per source line as opcode multisets). The integer instructions differ
in spelling only, on 12 of the kernel's 32 source lines: `IADD3` becomes the two-input `IADD`, `ISETP.NE.AND`
becomes `ISETP.NE.U32.AND`, `ISETP.NE.S64.AND` becomes `ISETP.NE.U64.AND` (a test against zero), and the converter's
`:58` and `:63` tests become `ISETP.NE.U32.AND`/`ISETP.EQ.U32.AND` of the same operands. No stored bit depends on the
toolkit version.

## 4. What the record implies for the Metal port

- **The append is integer work, so its bits are exact by construction.** No float instruction touches K or V
  between the load and the 16-bit store (section 3.1), so neither CUDA's IEEE build nor this GPU's subnormal flush
  enters: every fp32 subnormal and every \|x\| < 2^-25 becomes the signed zero (`:53`), fp16 subnormals are rounded
  to nearest even in integer arithmetic (`:54-58`), finite overflow saturates to ±infinity (`:51`), and a NaN keeps
  its sign as `sign | 0x7E00` (`:50`, `nan.md`). The Metal kernel loads `kcur` and `vcur` as `device const uint*`
  (or reinterprets with `as_type<uint>` before any float operation; register moves keep subnormal and NaN bits) and
  converts with `f16_from_f32(uint)`.
- **`half()` is an admissible body for the finite and infinite inputs, not for NaN.** Main measures `f16_bits(x)`
  (`as_type<ushort>(half(x))`, `kernels/common/basalt_fp.metal:115`) equal to the integer transcription
  `f16_from_f32` (`tests/kernels/basalt_emu.py:350-367`) on every non-NaN input of its sweep: the fp16 subnormal
  range, 65504..65536 of both signs, fp32 subnormals and 2^20 random patterns
  (`tests/kernels/test_basalt_s2_experts.py:65-106`). For a NaN this GPU's `half()` gives 0x7e00 whatever the sign,
  and `cuda_nan_f16` gives sm_120's hardware 0x7fff; neither is `f16_from_f32`'s `sign | 0x7E00`. So the helper is
  either the integer transcription line for line (section 3.2) or `fnan ? sign | 0x7E00 : f16_bits(x)`; the
  `cuda_nan_f16` rule of invariant 5 does not apply here, because basalt converts in software, not with
  `__float2half`.
- **Index arithmetic.** `pos / page_size`, `pos % page_size` and the row are 64-bit in basalt (`:125`, `:131`,
  `:133`); for this model's pools (`max_cells` up to 262,144 cells × 2 KV heads × 256 dims, 2^27 halves per pool)
  32-bit `uint` arithmetic gives the same indices. `page < 0` skips the store (`:132`); the port keeps that skip and
  adds the store guard the repository's rules require (the row against the pool binding's size).
- **Per-step values.** `pos` comes from the step record (`step[kStepPos]`, `:125`), never a kernel argument;
  the port reads it from the window's step binding (invariant 4).
- **Not on the path.** The host-pool branch (`:137-141`) has no Metal counterpart (KV is fully resident).

**Helper inventory.**

| Helper | Where it lives | Used at |
|---|---|---|
| `f16_from_f32(uint)`: integer transcription of `f16_bits.hpp:37-65`, or `f16_bits` plus the `sign \| 0x7E00` NaN branch | new (proposed for `kernels/common/basalt_fp.metal`, next to `f16_bits` at `:115`) | `:134`, `:135` |
| `f16_bits` | `kernels/common/basalt_fp.metal:115` (origin/main) | inside the helper above, if it takes the `half()` body |
| `fnan`-style NaN test on bits (`(b & 0x7fffffff) > 0x7f800000`) | the sampler branch's `fnan` (`kernels/common/basalt_sampler.metal`, branch `woodsonl/basalt-sampler` only); one integer expression, so the helper may inline it | inside the helper |

No float helper (`*_ftz`, `ieee_*`, `cuda_nan`) is needed by this kernel.
