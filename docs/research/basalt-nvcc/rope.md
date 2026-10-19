# basalt `rope.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's `rope.cu` (the table rotation
`rope_neox_kernel`; the table itself is built on the host), read from the sm_120 SASS (with the PTX where it shows
what the front end chose), and the Metal form a port would write at each site. This is the file's page under
invariant 5 of `docs/design/flash-next.md`; `sampler.md` (branch `woodsonl/basalt-sampler`) is the model for its
layout and notation, and `native_rope.md` is the page of the rotation the engine runs.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/rope.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/rope.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/rope.hpp`, `rope_scaling.hpp`, `mrope.hpp`), so every PTX and SASS line names its basalt
  file and line. CUDA's own headers come from the compiler's installation. Only basalt's public MIT source was sent.
- **Flags.** `rope.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:122`) with nvcc's
  IEEE defaults: it is not in the `--use_fast_math` list (`:186-202`), and the only other fast-math flag (`:728`)
  belongs to the `mmq` target. A Release build (`:12-14`) passes CMake's `-O3 -DNDEBUG`, `add_compile_definitions`
  passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120 (`:3-4`, `:26-31`). No `-fmad`, `-ftz`,
  `-prec-div` or `-Xptxas` flag is set, so nvcc's defaults hold: `-fmad=true`, `-ftz=false`, `-prec-div=true`,
  `-prec-sqrt=true`.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48`. Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`). The
  file was also compiled with `nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) as a cross-check
  (section 3).
- **userArguments**, verbatim: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'`. Compiler Explorer
  adds `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them
  changes a kernel's code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026 (both compilers).
- **Translation unit.** The whole file in one unit; its SASS listing is complete (404 lines, one kernel).
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.f32`); the SASS shows what
  `ptxas` made of it. A row quotes the SASS of one copy where a site is unrolled; the copies were compared and differ
  only in registers.
- **Not float arithmetic.** `IMAD.WIDE` forms addresses; `HFMA2 R, -RZ, RZ, a, b` moves a constant. The tables leave
  them out.

## 2. Kernels in the file and their paths

`rope.cu` is basalt's first RoPE: a host builder of float64 cos/sin tables and the table rotation
`rope_neox_kernel`. The engine forces native RoPE on (`src/program/generate.cpp:1893`, applied at `:2401`), so every
engine call of `rope_neox_apply` is the `else` of a `native_rope_enabled()` branch and never runs. The session still
builds and registers the table, but `rope_table_for` returns no table (`src/kernels/cuda/native_rope.cu:98-99`: "the
kernels compute the angles (the stored table is not read)"), so no decode or drafter kernel reads it
(`native_rope.md`, `qsa_dec_fused.md`).

| Kernel or function | Lines | Path | Evidence |
|---|---|---|---|
| `build_rope_table(n_rot, RopeScaling, ...)` (host, float64) | `:60-85` | ON (host, session init), type `None`: `:61-63` calls the loop below; the Linear/YaRN body `:65-84` is OFF | `src/core/layer.cpp:586` (elastic arena) and `:713` (resident pools) with `rope_scaling()`; the configuration is `None` with base 1e7 (`src/program/generate.cpp:2135-2147`, set at `:2158`) |
| `build_rope_table(n_rot, theta, ...)` (host, float64) | `:47-58` | ON (host): the table is built, uploaded and registered, and read by no on-path kernel | registered at `layer.cpp:719-720` (`n_rot == 64`); `native_rope.cu:99` returns `{}` for every configuration |
| `rope_scaling_set`, `rope_scaling` (host) | `:44-45` | ON (host): the process configuration every RoPE site reads | `generate.cpp:2158`; read by `native_rope_apply` at `verify.cpp:1343` and `mtp.cpp:765` |
| `rope_neox_kernel` via `rope_neox_apply` | `:96-112`, `:116-133` | OFF: the `else` of `native_rope_enabled()` | `src/core/verify.cpp:1344`, `src/core/mtp.cpp:766`, `layer.cpp:909`; tests only: `src/kernels/rope_parity.cpp:119`, `:165`, `:330`, `:384`, `:417` |
| `rope_neox_pair` (`include/basalt/kernels/rope.hpp:42-45`) | inlined at `:110` | OFF with the kernel; its other inlined copy, the table indexer's pooled-key rotation (`qsa.cu:238-240`), is OFF too (`qsa.md` section 2) | the native kernels inline their own rotation (`native_rope.cu:65-67`) |

The M-RoPE table `mtab` (`include/basalt/kernels/mrope.hpp:48-50`) is null unless `--vision` is given
(`generate.cpp:2406-2417`); with it, a text cell holds `(c, c, c)` (`:2410`), so `mrope_pos` returns the cell index
either way and the contract's text-only decode never reads it.

## 3. Per-kernel SASS tables

No kernel of `rope.cu` is on the path (section 2). The rotation is recorded because `rope_parity` runs it
(`rope_parity.cpp:119`, `:330`) and holds the native rotation against it (`:384-400`, `:410-429`); the host table is
recorded because the session builds it.

### 3.1 `rope_neox_kernel` (`rope.cu:96-112`), OFF-PATH

392 SASS instructions (nvcc 13.0). Launch (`:123-126`): block 128, grid `(rows + 127) / 128`, no shared memory, no
cluster; one thread per row. The tail copy `:106` is unrolled by 16, 8, 4 and 1 and holds no float instruction. The
rotation `:107-111` is unrolled by four with a remainder loop; every pair compiles to two `FMUL` and two `FFMA`, none
with `.FTZ` (IEEE build). The PTX fuses only `:44`'s add (`fma.rn.f32`); at `:43` it writes `mul.f32`, `mul.f32`,
`sub.f32`, and `ptxas` contracts the first product into the subtraction (PTX `mul`/`sub` without a rounding modifier
permit it), so the SASS, not the PTX, shows the fused form of `oa`.

Registers in the remainder loop (the rows below): `R10` = `xr[i]` = a, `R16` = `xr[half + i]` = b, `R19` =
`cos_tab[toff + i]` = c and `R21` = `sin_tab[toff + i]` = s (the table pointers are the parameters at `0x3a0` and
`0x3a8`). The four unrolled copies are the same pattern (`FMUL R19, R35, R28`; `FMUL R28, R31, R28`;
`FFMA R31, R31, R18, -R19`; `FFMA R35, R35, R18, R28` with `R31` = a, `R35` = b, `R18` = c, `R28` = s).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:100-101` | `const long long r = (long long) blockIdx.x * blockDim.x + threadIdx.x; if (r >= rows) return;` | `IMAD.WIDE.U32 R10, R3, UR4, R10`; `ISETP.GE.S64.AND P0, PT, R10, UR6, PT`; `@P0 EXIT` | integer |
| `:106` | `for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];` | 32-bit `LDG.E.CONSTANT` / `STG.E` pairs, unrolled by 16, 8, 4 and 1 | a 32-bit word copy (`uint`, or float register moves, which keep subnormal and NaN bits) |
| `:108` | `toff = (size_t) mrope_pos(mtab, pos[r], i) * half` | `LDG.E.CONSTANT R0` (`pos[r]`); with `mtab` set, `pair % 3` by multiply-high (`UIMAD.WIDE.U32 ..., -0x55555555`) and `LDG.E.CONSTANT` of `tab[pos * 3 + pair % 3]` (`mrope.hpp:49`); `IMAD.WIDE.U32` by `half` | integer; `mtab` is null on the contract's path (section 2) |
| `:110` (`rope.hpp:43`) | `oa = a * c - b * s;` | PTX `mul.f32` (a·c), `mul.f32` (b·s), `sub.f32`; SASS `FMUL R25, R16, R21` (b·s, rounded); `FFMA R25, R10, R19, -R25` (a·c − b·s, one rounding) | `fma(a, c, -(b * s))`; IEEE subnormals: `ieee_fma(a, c, -ieee_mul(b, s))` |
| `:110` (`rope.hpp:44`) | `ob = a * s + b * c;` | PTX `mul.f32` (a·s), `fma.rn.f32` (b, c, a·s); SASS `FMUL R27, R10, R21` (a·s, rounded); `FFMA R27, R16, R19, R27` | `fma(b, c, a * s)`; IEEE subnormals: `ieee_fma(b, c, ieee_mul(a, s))` |
| `:110` | the stores of `orow[i]`, `orow[half + i]` | `STG.E` of the two `FFMA` results | stores of the two results |

No comparison, reduction, conversion or special function: the kernel is the two contractions per pair above, so a
transcription in this operand order gives basalt's bits wherever no subnormal occurs, and the `ieee_*` forms give them
everywhere.

### 3.2 The host table (`rope.cu:47-85`): host code, no device instruction

`build_rope_table` is host C++, built by the host compiler (GCC on basalt's Linux build); the device listing holds
nothing of it. Per entry of the configuration on the path (type `None`, `:61-63` into `:52-55`),
in order:

| Line | basalt | Operation | What the bits depend on |
|---|---|---|---|
| `:52` | `const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);` | `-2.0 * i` and the division by 64 are exact (`-i / 32`); `std::pow` | the host libm's `pow` (glibc) |
| `:53` | `const double ang = (double) p * inv;` | one IEEE binary64 multiply (`p` < 2^53 converts exactly) | none beyond IEEE |
| `:54` | `cos_tab[(size_t) p * half + i] = (float) std::cos(ang);` | libm `cos`, then binary64 to binary32, round to nearest even | the host libm's `cos` |
| `:55` | `sin_tab[(size_t) p * half + i] = (float) std::sin(ang);` | libm `sin`, then the same conversion | the host libm's `sin` |

The Linear and YaRN body `:65-84` (`rope_yarn_ramp` in binary32, `std::log` in `RopeScaling::mscale` and
`rope_yarn_corr_dim`) is off the path and reached only by `rope_parity`'s scaled-table checks.

### 3.3 nvcc 13.3

The unit compiled by `nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) gives `rope_neox_kernel` 392
instructions with the same float instructions on every source line (compared per source line as opcode multisets:
the two `FMUL` and two `FFMA` per pair at `rope.hpp:43-44`). The integer instructions differ in spelling on 7 of the
kernel's 15 source lines: `IADD3` becomes `IADD`, `UISETP.NE.AND` becomes `UISETP.NE.U32.AND`, `ISETP.NE.S64.AND`
becomes `ISETP.NE.U64.AND` (a test of `mtab` against null), `IMAD.WIDE` address forms become `IADD.64`, and the
`MOV`/`NOP` padding differs. No stored bit depends on the toolkit version.

## 4. What the record implies for the Metal port

- **No on-path Metal kernel comes from this file.** The decode and drafter rotations are the native ones
  (`native_rope.md`; the drafter's query post, `qsa_dec_fused.md`), which compute their angles on the GPU; the
  table is built and registered but no on-path kernel reads it (section 2).
- **The table.** The contract counts one 256-byte RoPE row per cell (`docs/design/flash-next.md:498`, `:516`). A port
  that keeps it builds it on the host in binary64 by a transcription of `:47-58`, `inv = pow(1e7, -i / 32)`,
  `ang = p * inv`, `float32(cos(ang))`, `float32(sin(ang))`, and binds it to no Metal kernel. Its bits depend on
  the host libm's `pow`, `cos` and `sin` (section 3.2). `rope_parity` compares the builder with a transcription on
  the same host (`rope_parity.cpp:72-77`, `memcmp` at `:76-77`), so a port's check passes on any libm by
  construction; whether the port's table equals the one glibc builds is not something basalt tests, and is an
  open question of this page.
- **The table rotation, if a test needs it.** The contract keeps three `rope_parity` criteria
  (`docs/design/flash-next.md:376`): the table bitwise (`:76`), the table rotation against a float host reference,
  `if (!(rel <= 1e-6)) {` (`:145`), and the native rotation against the table rotation,
  `if (!(rel <= 3e-3)) ++agree_bad;` (`:400`, and `:429` for the unscaled table). A Metal table rotation
  that gives basalt's bits is section 3.1 transcribed: `oa = ieee_fma(a, c, -ieee_mul(b, s))`,
  `ob = ieee_fma(b, c, ieee_mul(a, s))`, the tail copied as 32-bit words. Each output depends only on its own pair,
  so any geometry gives the same bits; basalt's one thread per row (`:89-95`) exists to avoid a race, not for the
  arithmetic. The plain `fma(a, c, -(b * s))` in `MTLMathModeSafe` differs only where an input, a product or a
  result is subnormal, by less than 2^-126 absolute, inside `:145`'s `1e-6` of the row's largest input for any
  row with a normal element; the `ieee_*` forms are the exact transcription and cost little in a kernel no decode
  step runs. For `:400` the port may instead hold the native Metal rotation against a binary64 rotation of the
  binary64 table in numpy, which is the contract's double-precision form of the same bound (`flash-next.md:464`).
- **NaN.** The rotation stores what the `FFMA` writes. A NaN arises only from a NaN or infinite input or an
  overflowing product, and the store then holds sm_120's NaN, 0x7fffffff, where Metal's arithmetic gives 0x7fc00000
  (`nan.md`): a port that must give basalt's NaN bits passes the two results through `cuda_nan`
  (`kernels/common/basalt_fp.metal:120`). `rope_parity` feeds Gaussian rows (`:86-88`), so no test reaches it.

**Helper inventory** (only if the table rotation is ported):

| Helper | Where it lives | Used at |
|---|---|---|
| `ieee_mul` | `kernels/common/basalt_q8_1.metal:108` (origin/main); moved to `kernels/common/basalt_ieee.metal` on the sampler branch (`woodsonl/basalt-sampler` only) | `rope.hpp:43-44`, the products b·s and a·s |
| `ieee_fma` | `kernels/common/basalt_q8_1.metal:167` (origin/main); `basalt_ieee.metal` on the sampler branch | `rope.hpp:43-44`, the fused a·c and b·c |
| `cuda_nan` | `kernels/common/basalt_fp.metal:120` (origin/main) | the two stores, if NaN bits matter |

The host table needs no Metal helper.
