# basalt `native_qsa_score.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's MMA block scorer, read from the sm_120 SASS
(with the PTX where it shows what the front end chose), and the Metal form a port would write at each site. The
kernel has no caller at basalt's commit (section 2), so this page records it for completeness; the block scores the
engine runs are `qsa_block_scores` in `qsa_select.cu` (`qsa_select.md`). `sampler.md` (branch
`woodsonl/basalt-sampler`) is the model for this page's layout and notation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/native_qsa_score.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/native_qsa_score.cu` (151 lines) with basalt's local includes inlined under `#line`
  directives (`include/basalt/kernels/native_qsa_score.hpp`, `qsa.hpp`), so every PTX and SASS line names its basalt
  file and line. CUDA's own headers come from the compiler's installation. Only basalt's public MIT source was sent.
  The stored request's source equals that inlining of b4c63b5 line for line, 548 lines with no blanked or replaced
  line (`.context/qsa/nvcc-a/check_src.py` against `.context/qsa/nvcc/native_qsa_score.request.json`).
- **Flags.** The file builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:145`) and is in the
  `--use_fast_math` list (`CMakeLists.txt:185-202`: the comment at `:185`, `set_source_files_properties` at `:186`,
  the file at `:198`, `PROPERTIES COMPILE_OPTIONS "--use_fast_math"` at `:202`). `--use_fast_math` implies
  `-ftz=true -prec-div=false -prec-sqrt=false -fmad=true`. A Release build (`:12-14`) passes CMake's `-O3 -DNDEBUG`,
  `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120 (`:3-4`,
  `:26-31`).
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48` (`.context/qsa/nvcc/version.response_nvcc130.json`). Basalt
  requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`).
- **userArguments**, verbatim: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"' --use_fast_math`.
  Compiler Explorer adds `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler
  (GCC 14.1); none of them changes a kernel's code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026 (`.context/qsa/nvcc/native_qsa_score.response_nvcc130.json`, exit code 0, no
  diagnostics); listings dumped October 18 with `.context/qsa/nvcc-b/dump.py` to `nvcc-b/score.ptx.txt` (667 lines)
  and `nvcc-b/score.sass.txt` (655 lines). The whole file is one translation unit; the listing is complete (no
  `[truncated; too many lines]`).
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end chose (`add.rn.ftz.f32`,
  `max.ftz.f32`, the inline `ldmatrix` and `mma` PTX the source writes); the SASS shows what `ptxas` made of it. A
  row quotes the SASS of one copy where a site is unrolled; the copies were compared and differ only in registers.

## 2. Kernels in the file and their paths

The file holds one kernel, `score_kernel` (`:51-121`, `__launch_bounds__(64,1)`), and its host entry
`native_qsa_score` (`:134-150`, declared at `include/basalt/kernels/native_qsa_score.hpp:26-29`). The entry throws
unless the stream is non-null, `idx_dim == 128`, `idx_n_head == 4`, `idx_block == 4`, `idx_top_k == 2048`,
`1 <= max_cells <= INT32_MAX - 3` and `max_blocks == max_cells/4 + 1` (`:137-139`), checks 4 or 5 spans for
alignment and overlap (`:140-145`), and launches
`score_kernel<<<unsigned((max_blocks+ROWS-1)/ROWS),dim3(32,WARPS),0,static_cast<cudaStream_t>(stream)>>>`
(`:146-147`): ⌈`max_blocks`/32⌉ CTAs of 2 warps, no dynamic shared memory (the static `shared[2*16*36]` floats,
4,608 bytes, `:61`).

| Kernel | Lines | Launch | Path |
|---|---|---|---|
| `score_kernel` | `:51-121` | grid ⌈`max_blocks`/32⌉, block (32, 2), static shared 4,608 bytes (`:146-147`) | OFF: no caller |

**No caller.** `git grep -n native_qsa_score b4c63b5` finds `CMakeLists.txt:145`, `:198`, the header
(`native_qsa_score.hpp:7`, `:8`, `:26`) and the file itself (`:21`, `:132-134`), nothing else: no engine path and no
parity program calls `native_qsa_score`, and `native_qsa_score_set_enabled` is never called, so the switch keeps its
`false` initializer (`:30`). The block scores every path runs are `qsa_block_scores` in `src/kernels/cuda/qsa_select.cu`,
an IEEE-built file: the window (`src/core/verify.cpp:1545`), the layer forward (`src/core/layer.cpp:996`) and the
prompt path (`src/prefill/prefill.cpp:1667`); their record is `qsa_select.md`. The kernel spec
(`.context/qsa/spec-kernels-indexer-a.md` sections 1.1 and 1.7) records the same.

Device functions inlined into the kernel: `load_a` (`:35-39`, inline PTX `ldmatrix.sync.aligned.m8n8.x4.b16`),
`load_b` (`:40-44`, `ldmatrix ... .x2.b16`) and `mma` (`:45-50`, inline PTX
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` on the raw fp32 bits, no `cvt.rn.tf32`, `:46`).

## 3. Per-kernel SASS tables

### 3.0 The Metal operations this page names

The file builds with `--use_fast_math`, so every scalar float instruction in the listing carries `.FTZ`: CUDA reads a
subnormal operand as zero (sign kept) and writes a subnormal result as zero, which is what this GPU's arithmetic does
in `MTLMathModeSafe`. The Metal column uses main's helpers in `kernels/common/basalt_fp.metal`: `add_ftz` (`:49`),
`max_ftz` (`:81`), with the file's `#pragma STDC FP_CONTRACT OFF` (`:43`). `max_ftz` is Metal's `fmax` on flushed
operands: like `FMNMX.FTZ ... !PT` it returns the other operand when one is NaN.

### 3.1 `score_kernel` (`:51-121`)

SASS from the whole-file unit (655 lines, one function). The CTA is 2 warps (`threadIdx.y` = warp); CTA b covers the
32 pooled rows (blocks) `row0 = 32·b` to `row0 + 31` (`:58`). The 4 step words are read from device memory and
validated before any work (`:55-57`: `LDG.E.CONSTANT` at `step + 0x4`, `+0x8`, `+0x0`, `+0xc`; `VIMNMX.U32 R5, R86,
0x803` is `min(n, 2051)`), and a CTA past `full` exits (`:59`). The K loop (`:66`) runs twice per warp,
`col = warp·32 + lane` then `+ 64`, so warp w accumulates K = [32w, 32w + 32) and then [32w + 64, 32w + 96).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:74` | `tile[i*STRIDE+lane]=row<=full?pooled[size_t(row)*D+col]:0.0f;` (16 rows per `ia`) | `LDG.E` + `STS` (rows past `full` store `RZ`); no arithmetic | copy the word (+0 for rows past `full`) |
| `:82` | `tile[h*STRIDE+lane]=h<HEADS?query[h*D+col]:0.0f;` | `STS` of the 4 query words, `STS ..., RZ` for h = 4..7 | copy (+0 for h = 4..7) |
| `:36-38`, `:41-43` | `ldmatrix.sync.aligned.m8n8.x4.b16` / `.x2.b16` from the tile (inline PTX) | `LDSM.16.M88.4` ×4 per `ia`, `LDSM.16.M88.2` ×4 per pass | register moves: each 32-bit register receives two adjacent b16 halves, i.e. one whole fp32 word of the tile; the fragment map decides which (row, k) a lane holds |
| `:45-49` | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {c}, {a}, {b}, {c}` on the raw fp32 bits (no `cvt.rn.tf32`, `:46`) | `HMMA.1688.F32.TF32` ×8 per pass: for each `ia` a chain k = 0, 1, 2, 3 (`HMMA R40, R40, R190, R32`; `HMMA R24, R24, R192, R40`; `HMMA R16, R16, R28, R24`; `HMMA R32, R4, R30, R16`), the two `ia` chains interleaved | no Metal operation gives these bits (section 4); the order to keep is the chain: per warp, k-blocks of 8 ascending over its first 32 columns, then its second 32 |
| `:98` | `shared[h*COMBINE+i]=c[ia].x[l];` | `STS` ×8 per warp | store the partials |
| `:110` | `float v=__fadd_rn(0.0f,shared[j*COMBINE+lane]);` (warp 0's partial) | PTX `add.rn.ftz.f32 %f195, %f194, %f193` with `%f194 = 0f00000000`; SASS `FADD.FTZ R4, RZ, R4` (×4, one per head) | `add_ftz(0.0f, p0)` (turns -0 into +0; the add is not folded away) |
| `:111` | `v=__fadd_rn(v,shared[j*COMBINE+ROWS+lane]);` (warp 1's partial) | `FADD.FTZ R4, R4, R5` (×4) | `add_ftz(v, p1)` |
| `:112` | `h[j]=fmaxf(v,0.0f);` | PTX `max.ftz.f32 %f198, %f197, 0f00000000`; SASS `FMNMX.FTZ R4, RZ, R4, !PT` (×4) | `max_ftz(v, 0.0f)`: a NaN `v` gives +0; -0 cannot occur (`:110` gave +0) |
| `:114` | `float sum=__fadd_rn(__fadd_rn(__fadd_rn(h[0],h[1]),h[2]),h[3]);` | `FADD.FTZ R3, R4, R3` (h0 + h1); `FADD.FTZ R3, R8, R3` (+ h2); `FADD.FTZ R3, R10, R3` (+ h3) | `add_ftz(add_ftz(add_ftz(h0, h1), h2), h3)`: heads in order 0, 1, 2, 3 |
| `:115` | `if(bias)sum=__fadd_rn(sum,bias[row]);` | `@P1 FADD.FTZ R3, R3, R2` (P1 = bias non-null) | `add_ftz(sum, bias[row])` when the bias binding is present |
| `:116` | `sum=__fadd_rn(sum,row==full&&n%R?1e9f:0.0f);` | PTX `selp.f32 %f217, 0f4E6E6B28, 0f00000000, %p45`, `selp.f32 %f218, %f217, 0f00000000, %p46`; SASS `FSEL R0, RZ, 1e9, P2`; `FSEL R0, R0, RZ, P0`; `FADD.FTZ R0, R0, R3` | `add_ftz(sum, (row == full && n % 4 != 0) ? 1e9f : 0.0f)` (`1e9f` = `0x4E6E6B28`) |
| `:118` | `sum=__fadd_rn(sum,0.0f);` | `FADD.FTZ R7, RZ, R0` | `add_ftz(sum, 0.0f)` |
| `:119` | `for(int i=row*R;i<n&&i<(row+1)*R;++i)cells[i]=sum;` | `STG.E` (the PTX unrolls the loop by 16 with remainders of 8, 4 and 1: 29 `st.global.f32`; at most 4 iterations run) | store `sum` to the block's live cells |

Notes:

- **The whole float record is 8 `HMMA` (per pass), 14 `FADD.FTZ`, 4 `FMNMX.FTZ` and 2 `FSEL`.** `fops.py` lists
  nothing else: no `FFMA`, `FMUL`, `MUFU`, conversion or division, so no CUDA slow path, no `CALL` and no libdevice
  routine (the PTX has no `__nv_` symbol and no `call`; its only `.func` is `dim3::dim3`, which
  `--keep-device-functions` emits and the kernel does not call).
- **`__fadd_rn` keeps every add.** The `0.0f +` adds at `:110` and `:118` survive into the SASS (`FADD.FTZ R, RZ,
  R`): `__fadd_rn` is never contracted or folded. The Metal port writes them as calls, not as `x + 0.0f` that a
  compiler could drop (under `MTLMathModeSafe` it may not, because of -0, but the helper call states the intent).
- **NaN.** The header requires finite inputs (`include/basalt/kernels/native_qsa_score.hpp:23`). A NaN in `pooled`
  or `query` reaches only the per-head partials, which `fmaxf(v, 0)` turns into +0 (`:112`); a NaN `bias[row]`
  reaches the store at `:119` through `:115`, where sm_120's `FADD` writes its canonical NaN (`nan.md`), so a port
  stores through `cuda_nan` (`basalt_fp.metal:120`) if it accepts a non-finite bias at all.
- **Subnormals.** Every scalar add flushes (`.FTZ`), as this GPU does. Whether the tensor core flushes subnormal
  `tf32` operands or accumulator values is not documented and was not measured (section 4).

## 4. Metal transcription rules for this file

### 4.1 Helpers by kernel

| Kernel | Path | Scalar float helpers (origin/main) | Tensor-core step | Library sequences | Subnormals |
|---|---|---|---|---|---|
| `score_kernel` (3.1) | OFF: no caller | `add_ftz` (`kernels/common/basalt_fp.metal:49`), `max_ftz` (`:81`); `cuda_nan` (`:120`) at the store if a non-finite bias is accepted | `HMMA.1688.F32.TF32`: no Metal helper exists or is proposed (below) | none | flushed at every scalar site, as on this GPU; the tensor core's handling unrecorded |

No helper is new for the scalar part; the port would need nothing from the sampler branch.

### 4.2 What basalt's tests demand

Nothing: no parity program and no engine path calls `native_qsa_score` (`git grep` at b4c63b5 over `src/` finds no
caller, section 2), so no basalt criterion constrains this kernel's bits. The block scores a test checks are
`qsa_block_scores`' (`src/kernels/qsa_scores_parity.cpp`, `src/kernels/qsa_seg_parity.cpp`; `qsa_select.md`).

### 4.3 Points for the contract and the kernel lane

- **Leave the kernel out.** It is off-path at b4c63b5 and the unit's block scores are `qsa_block_scores`
  (`.context/qsa/spec-kernels-indexer-a.md` sections 1.1 and 1.7 make the same call). Nothing here constrains
  the `qsa-indexer` pull request.
- **If a later basalt revision calls it, the TF32 MMA is the open question.** The scalar epilogue is exact on this
  GPU with main's helpers (every site is `.FTZ`). The score itself comes from `HMMA.1688.F32.TF32` on raw fp32 bits
  (`:46`; the header says "MMA consumes the same raw F32 register bits as the oracle",
  `include/basalt/kernels/native_qsa_score.hpp:14`), whose bits depend on what sm_120's tensor core does with the
  low 13 mantissa bits of each operand, how it aligns, rounds and orders the 8 products and the accumulator, and how
  it treats subnormals. NVIDIA documents none of these; third-party measurements on earlier generations report
  exact products and a truncating multi-term addition [R], and sm_120 was not measured. A Metal port of this kernel
  would need that model measured on an sm_120 device before it could claim bitwise equality, and the 8-element
  k-block chain order of section 3.1 (per warp, k-blocks ascending over its first 32 columns, then its second 32;
  the two warps' partials added to +0 in warp order, `:110-111`).
- **Per-step values** are already where invariant 4 wants them: the kernel reads `n`, `full`, `pos` and `width`
  from the device `step` words (`:55-57`) and writes nothing when they disagree.
- **No approximate operation and no library function** occurs in the file (section 3.1 notes).
