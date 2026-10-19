# basalt `qsa_dec_attn.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's decode attention, read from the sm_120 SASS
(with the PTX where it shows what the front end chose), and the Metal form the port writes at each site. This is the
attention's page under invariant 5 of `docs/design/flash-next.md`; `matvec.md` is the q8_1 page and `sampler.md`
(branch `woodsonl/basalt-sampler`) the model for this page's layout and notation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/qsa_dec_attn.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/qsa_dec_attn.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/qsa_dec_attn.hpp`, `qsa.hpp`, `qsa_decode_attn.hpp`, `qsa_seg.hpp`, `kv_stream.hpp`,
  `native_qsa_indexer.hpp`, `rope_scaling.hpp`, `qsa_dec_fused.hpp`, `kv_q8.hpp`, `pdl.hpp`, `q8_1_finite.hpp`), so
  every PTX and SASS line names its basalt file and line. CUDA's own headers come from the compiler's installation.
  Only basalt's public MIT source was sent.
- **Flags.** `qsa_dec_attn.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:153`) with
  nvcc's IEEE defaults: it is not in the `--use_fast_math` list (`:186-202`, under the comment at `:185`), and the
  only other fast-math flag (`:728`) belongs to the `mmq` target. A Release build (`:12-14`) passes CMake's `-O3
  -DNDEBUG`, `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120
  (`:3-4`, `:26-31`). No `-fmad`, `-ftz`, `-prec-div` or `-Xptxas` flag is set, so nvcc's defaults hold:
  `-fmad=true`, `-ftz=false`, `-prec-div=true`, `-prec-sqrt=true`. Every `.FTZ` and every approximate instruction
  below therefore comes from what the source writes by hand (`__expf`, the PTX of `gate_fast` and `q8_1_store_fm`),
  not from a flag.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48`. Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`).
- **userArguments**, verbatim: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'`. Compiler Explorer
  adds `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them
  changes a kernel's code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026.
- **Translation units.** The whole file's SASS listing stops at Compiler Explorer's 5,000-line cut
  (`[truncated; too many lines]`) after the four `attn_chunk_v2` instances (`<0, true>`, `<1, true>`, `<0, false>`,
  `<1, false>`) and the first 395 instructions of `attn_merge_v2`. So each kernel group was also compiled as its own
  unit, the other kernels' bodies blanked line for line (an empty line keeps every line number), and the host
  launches of the template instances not kept (`:778-779`, `:818-819`, `:822-823`) blanked so they are not
  instantiated:
  - unit `a1_chunk_v2_0f`: `attn_chunk_v2<0, false>` alone (`:455-628`, `:635-696` and `:710-713` blanked);
  - unit `a2_merge_v2`: `attn_merge_v2` alone (`:455-628`, `:635-696`, `:703-706` and `:782-783` blanked as well);
  - unit `a3_draft`: `draft_attn_chunk` and `draft_merge` (`:703-706`, `:710-713` and every `attn_chunk_v2` launch
    blanked).

  The cut changes nothing in a kept kernel, because every device function it calls is inlined into it: the unit's
  `attn_chunk_v2<0, false>` and the full compile's are identical, 1,032 instructions with no difference in
  instruction or source line, and the unit's `attn_merge_v2` equals the full compile's 395 instructions before the
  cut, once labels and slow-path subroutine numbers are renamed (`.context/qsa/nvcc-b/cmp.py`).
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.f32`), which division it
  chose and the inline PTX `gate_fast` and `q8_1_store_fm` write; the SASS shows what `ptxas` made of it. A row
  quotes the SASS of one copy where a site is unrolled; the copies were compared and differ only in registers. The
  instructions after a kernel's last `EXIT` are CUDA's out-of-line slow paths, which `-lineinfo` attributes to the
  line before them; the tables attribute each to the `CALL.REL.NOINC` that reaches it.
- **Not float arithmetic.** `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` at a strided loop's
  head are `ptxas`'s exact integer division; `HFMA2 R, -RZ, RZ, a, b` moves a constant; `IMAD.WIDE` forms
  addresses; `HADD2.F32 R, -RZ, Rh.H0_H0` widens an fp16 value to fp32 (exact). The tables leave them out unless
  they read a stored value.

## 2. Kernels in the file and their paths

One host entry runs on the contract path, `qsa_dec_attn` (`:743-793`). It picks the kernel pair from the pools'
format, `kv_mode_of` (`:717-723`): with the fp16 KV cache `qsa_attn_pools` fills only `k_pool`, `v_pool` and
`page_table` (`src/core/layer.cpp:779-787`), so the mode is 0, and the drafter's wide window needs `mode == 1`
(`:764`). Both engine callers therefore launch the same two kernels:

| Kernel | Lines | Launch | Path |
|---|---|---|---|
| `attn_chunk_v2<0, false>`, body `attn_chunk_dev<0, false>` | `:701-707`, `:143-306` | grid (`n_chunks` = ⌈cap/64⌉, `n_head_kv`, `n_q`), 256 threads, `sizeof(ChunkSmem<0>)` = 73,216 bytes dynamic (`:781-783`) | ON: decode and verify windows (`src/core/verify.cpp:1557-1566`, `fused_gate`), the drafter's layer (`src/core/mtp.cpp:881-883`) |
| `attn_merge_v2`, body `attn_merge_dev` | `:709-714`, `:347-421` | grid (`n_head`, `n_q`), 256 threads, (2·`n_chunks` + 8)·4 bytes dynamic (`:785`, `:787`) | ON: both callers; with `q_full` and `attn_gated` set it also writes the gated row (`gate_fast`, `:308-319`), and with `xq` set that row's q8_1 blocks (`q8_1_store_fm`, `:321-345`). Both callers pass all three (`verify.cpp:1563-1565` with `attn_q8` true on the path, `mtp.cpp:883`) |
| `attn_chunk_v2<1, false>` | `:701-707` | as `<0, false>` | OFF: int8 KV (`:776-779`) |
| `attn_chunk_v2<0, true>`, `<1, true>` | `:701-707` | `qsa_dec_attn_seg` (`:795-832`) | OFF: verify's batch windows, the only caller passing `int8 = true` (`verify.cpp:1539`) |
| `draft_attn_chunk`, `draft_merge` | `:454-629`, `:634-697` | `mode == 1 && n_q == 1 && cap >= 8192` (`:764-775`, `:786`) | OFF: int8 KV only, so the fp16 drafter takes the pair above with `a.chunk` = 64 |

So the contract's "drafter's wide window" (`flash-next.md`, the approximate-operations list: `:576-577`, `:667`) is
not on the fp16 path; its `__expf` sites run in the int8 kernels only. The drafter's fp16 attention uses decode's
`__expf` sites (`:259-260`, `:380`).

Device functions inlined into the on-path pair: `warp_sum` (`:32-36`), `warp_max` (`:37-41`), `reduce12`
(`:46-68`), `k8_from<0>` (`:113-131`), `v1_from<0>` (`:132-140`), `gate_fast` (`:308-319`), the `fm_*` PTX wrappers
(`:326-329`), `q8_1_store_fm` (`:330-345`) with `q8_1_finite` and `q8_1_ds` (`include/basalt/kernels/q8_1_finite.hpp`),
and `pdl_launch_dependents` / `pdl_wait` (`include/basalt/kernels/pdl.hpp:67`, `:75`: `ACQBULK` and `PREEXIT`, no
float work). `a.scale` is computed on the host (`:757`, `1.0f / sqrtf(256.0f)` = 0.0625 exactly) and read from the
argument block.

The SASS of the on-path pair comes from units `a1_chunk_v2_0f` (1,032 instructions) and `a2_merge_v2`; the
`a3_draft` unit lists the int8 drafter kernels and is not tabulated here (off-path).

## 3. Per-kernel SASS tables

### 3.0 The Metal operations this page names

The file builds IEEE, so CUDA keeps subnormal operands and results at every instruction without `.FTZ`, and its
ordered comparisons see a subnormal as itself; this GPU flushes subnormals in arithmetic and compares them as zero.
The Metal column uses `sampler.md`'s names (section 2.0 there), one helper per operation at every site:

- `add32`, `mul32`, `fma32`, `div32`: the correctly rounded IEEE operation with subnormals kept (`FADD`, `FMUL`,
  `FFMA`, `div.rn.f32`). Each is Metal's `+`, `*`, `fma` or `/` whenever its operands and result are normal, zero or
  infinite; otherwise the integer emulation (`ieee_add`, `ieee_mul`, `ieee_div` on main; a general `fma32` only on the
  sampler branch, see section 4). Each row says whether a subnormal is reachable.
- `fmax32(a, b)`: CUDA's `fmaxf` (`FMNMX ... !PT` without `.FTZ`): the other operand when one is NaN, else the larger
  in IEEE order with subnormals distinct. `key32` and the ordered/unordered compares `gt32`, `geu32`, `neu32`, ... as
  in `sampler.md` section 2.0.
- `ex2_ieee(t)`: `ex2.approx.f32` (no `.ftz`) as `ptxas` expands it on sm_120:
  `FSETP.GEU.AND P, PT, t, -126, PT`; `@!P FMUL t, t, 0.5`; `MUFU.EX2 r, t`; `@!P FMUL r, r, r`. Below -126 it halves
  the argument and squares the result with an IEEE multiply, so CUDA's `__expf` returns subnormals for arguments
  from about -87.3 down to about -103.3 (2^-149). The contract maps `MUFU.EX2` to `fast::exp2`; the Metal form is
  `t < -126.0f ? mul32(e, e) : fast::exp2(t)` with `e = fast::exp2(t * 0.5f)` (the halving is exact; a NaN `t` takes
  the plain branch on both sides; `t = -inf` gives +0 on both). `expf_ieee(x)` below is `ex2_ieee(mul32(x,
  0x3FB8AA3B))`, CUDA's `__expf` in an IEEE-built file; main's `expf_neg` (`kernels/common/basalt_fp.metal:104`) is the
  same without the halving branch, which changed no result at its one site (`experts.md` section 2).
- The `.FTZ` forms the PTX of `gate_fast` and `q8_1_store_fm` write: main's `add_ftz`, `mul_ftz`, `max_ftz`,
  `abs_ftz`, `rcp_ftz`, `expf_ftz`-style `fast::exp2` (`kernels/common/basalt_fp.metal:45-107`).

### 3.1 `attn_chunk_v2<0, false>` (`:701-707`; `attn_chunk_dev<0, false>`, `:143-306`)

SASS from unit `a1_chunk_v2_0f` (identical to the full compile's). The 256 threads are 8 warps; a CTA is one
(chunk, KV head, query). Integer and address work (`:157-206`: the ids, the width `n_ids`, the page lookup
`cell / page_size` through `ptxas`'s exact integer division at `:174`, the `cp.async` copies) is not float
arithmetic. The scores loop (`:211-249`, `#pragma unroll 1`) runs 4 passes of two cells per warp; the per-head
softmax (`:252-273`) and the value loop (`:278-305`, unrolled by 8) follow.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:116-123` | `k8_from<0>`: `__half22float2` of the 16-byte K slice, `out[2j] = f.x`, `out[2j+1] = f.y` | `HADD2.F32 R, -RZ, Rh.H0_H0` and `.H1_H1` (attributed to `cuda_fp16.hpp:980`; PTX `cvt.f32.f16`), 16 per pass | `h2f` (exact) |
| `:161-166` | `qr[h][0..7]` = the query's 8 floats at `lane * 8` | `LDG.E.128.CONSTANT` ×2 per head (PTX `ld.global.nc.v4.f32`); no arithmetic | loads |
| `:234-235` | `pp[j][h] = kk[j][0] * qr[h][0] + kk[j][1] * qr[h][1] + ... + kk[j][7] * qr[h][7]` | PTX per (j, h): `mul.f32 p, k1, q1`; `fma.rn.f32 p, k0, q0, p`; then `fma.rn.f32 p, k_i, q_i, p` for i = 2..7 in order. SASS `FMUL R101, R97, R126` (q1·k1); `FFMA R101, R96, R127, R101` (q0·k0 + p); `FFMA R100, R98, R125, R101` (q2·k2 + p); ... 24 chains (12 heads × 2 cells): 24 `FMUL` and 168 `FFMA`, attributed to `:234-235` | `p = mul32(k1, q1); p = fma32(k0, q0, p); p = fma32(k2, q2, p); ...; p = fma32(k7, q7, p)`: the **second** product is the plain multiply, the first is fused into the first add |
| `:240` | `pp[j][h] = 0.0f`, h = 12..15 | constants | `0.0f` |
| `:46-68` | `reduce12(pp[j], lane)`: at each level `(hi ? a : b) + __shfl_xor_sync(.., hi ? b : a, o)`, o = 16, 8, 4, 2, then `r1 + __shfl_xor_sync(.., r1, 1)` | `FSEL` pairs on the lane bit (`FSEL R133, R121, R122, !P2`), `SHFL.BFLY`, `FADD R100, R100, R125`: 16 + 8 + 4 + 2 + 2 `FADD` per pass of two cells (8 + 4 + 2 + 1 + 1 per call, `:241`), attributed to `:51`, `:57`, `:63`, `:66`, `:67` | the same selects (on bits), `simd_shuffle_xor`, `add32` in the same pairing |
| `:243` | `sm.sp[hd][cc[j]] = sj * a.scale` | `FMUL R101, R101, UR7` (`a.scale`, constant bank) | `mul32(sj, scale)`; scale = 0.0625 (host, `:757`), so the product is exact unless |sj| < 2^-122 (subnormal) |
| `:245` | masked: `sm.sp[lane][cc[j]] = -FLT_MAX` | constant store | `-FLT_MAX` (`0xff7fffff`) |
| `:257` | `m = warp_max(fmaxf(x0, x1))` | `FMNMX R4, R9, R10, !PT`; then `:39` `FMNMX R7, R4, R7, !PT` after `SHFL.BFLY` 16, 8, 4, 2, 1 | `fmax32` in the same order (xor 16, 8, 4, 2, 1; own operand first) |
| `:258` | `n = warp_max(fmaxf(y0, y1))` (y = 0 for warps 4..7, `two` false) | `FMNMX R8, R12, R11, !PT`, the same butterfly | as `:257` |
| `:259` | `e0 = ok0 ? __expf(x0 - m) : 0.0f`, `e1` likewise | PTX `sub.f32`, `mul.f32 .., 0f3FB8AA3B`, `ex2.approx.f32`. SASS `@P6 FADD R9, R9, -R15`; `@P6 FMUL R9, R9, 1.4426950216293334961`; `@P6 FSETP.GEU.AND P1, PT, R9, -126, PT`; `@!P2 FMUL R9, R9, 0.5`; `@P6 MUFU.EX2 R9, R9`; `@!P2 FMUL R9, R9, R9` (P6 = ok0; P2 = P1 under P6); 0 when not ok | `ok0 ? ex2_ieee(mul32(add32(x0, -m), log2e)) : 0.0f` |
| `:260` | `f0 = ok0 ? __expf(y0 - n) : 0.0f`, `f1` likewise | as `:259` (`@P6 FADD R11, -R19, R11`, ...) | as `:259` |
| `:261-265` | `sm.spt[lane][h0] = e0`, ... | `STS` | stores |
| `:267` | `l = warp_sum(e0 + e1)` | `FADD R6, R5, R4`; then `:34` `FADD R9, R6, R9` after each `SHFL.BFLY` 16, 8, 4, 2, 1 | `add32`, then the butterfly with `add32` (`warp_sum_ieee`, `basalt_q8_1.metal:206`) |
| `:268` | `l2 = warp_sum(f0 + f1)` | `FADD R10, R12, R7`; the same butterfly | as `:267` |
| `:270-271` | `part_m[...] = m; part_l[...] = l` (lane 0) | `STG.E` | stores (scratch: not an output) |
| `:285-286` | `__half22float2(&sm.v[c][t * 4])` | `HADD2.F32 R, -RZ, Rh.H0_H0` / `.H1_H1` | `h2f` |
| `:293-296` | `w[i]` = `sm.spt[c][0..11]` | `LDS.128` ×3 | loads |
| `:299-300` | `acc[i][0] = fmaf(w[i], v0, acc[i][0]); acc[i][1] = fmaf(w[i], v1, acc[i][1])`, c = 0..63 ascending | PTX `fma.rn.f32 acc, w, v, acc`; SASS `FFMA` (96 + 96: 12 heads × 2 dims × 8 cells per unrolled pass), the cells in ascending order | `fma32(w, v, acc)`, cells ascending |
| `:304-305` | `part_acc[...] = (acc[i][0], acc[i][1])` | `STG.E.64` | stores (scratch) |

Notes:

- **The dot product's first add fuses the first product.** The front end contracts `a·b + c·d + ...` as
  `fma(a, b, fl(c·d))`, then `fma(e, f, ·)` for each further term (PTX `mul.f32 %f324, %f960, %f211`: element 1
  times `qr[h][1]`; `fma.rn.f32 %f325, %f961, %f210, %f324`: element 0 times `qr[h][0]`). A port that rounds k0·q0
  first and fuses k1·q1 gives different bits; the order must be written as in the table.
- **Where a subnormal is reachable.** `fp16` K and V values are normal floats once widened (the smallest fp16
  subnormal is 2^-24), and q is an fp32 activation, so the products and sums of `:234-235` meet a subnormal only for
  a query component or a sum below 2^-126 in magnitude; `:243` only for |sj| below 2^-122; `:259-260` give a
  subnormal `e` for a score more than about 87.3 below the chunk maximum (`ex2_ieee`'s halving branch), and such a
  weight reaches the sums at `:267` and the accumulators at `:299-300`, where w·v can also underflow for normal w
  below 2^-126/|v|. Each of these needs its IEEE helper for CUDA's bits. With flushing instead, a dropped term is
  below about 2^-110 in magnitude (a weight below 2^-126 times |v| ≤ 65504), but it can also move a later fma off a
  rounding tie: `fma(a, b, c)` whose exact a·b lies halfway between two floats rounds to the even one for c = +0 and
  away from it for a positive subnormal c, so a flushed partial (the dot's first product, an accumulator still
  subnormal after its first cells) can leave the next normal partial one ulp from CUDA's. Either is far inside the
  tests' bounds, but not bitwise.
- **The comparisons.** `fmaxf` and `warp_max` are `FMNMX` without `.FTZ`: NaN is dropped in favour of the other
  operand, and a subnormal score orders as itself. `fmax32` must order subnormals by `key32`; which zero `FMNMX`
  returns for +0 against -0 is not recorded, and no output depends on it (m feeds only x − m, and `__expf` of either
  zero is 1, and the merge's own maximum, section 3.2). `ok0`/`ok1` and the masks are integer tests.
- **No libdevice routine** is inlined into this kernel: no `CALL`, no slow path; `__expf` and `__half22float2` are
  intrinsics.

### 3.2 `attn_merge_v2` (`:709-714`; `attn_merge_dev`, `:347-421`, with `gate_fast` and `q8_1_store_fm`)

SASS from unit `a2_merge_v2` (equal to the full compile's instructions before its cut, section 1). A CTA is one
(query head h, query qi); thread d (0..255) owns output dimension d, so warp w holds dimensions 32w..32w+31, exactly
one q8_1 block. `n_ids` comes from the step record (`:352`), and `nc` = min(`n_chunks`, ⌈`n_ids`/64⌉) is `ptxas`'s
exact integer division (`:354`: `I2F.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ`), not float arithmetic. The float
instructions by source line (`.context/qsa/nvcc-b/fops.py`): `:363` `FMNMX` ×29; `:39` `FMNMX` ×5; `:377` `FMNMX` ×7;
`:380` `FSETP.NEU` ×5, `FADD` ×5, `FMUL` ×15, `FSETP.GEU` ×5, `MUFU.EX2` ×5; `:389-391` `FSETP.GEU` ×32, `FFMA` ×64;
`:401-403` `FSETP.GEU` ×16, `FFMA` ×32; `:408-410` `FSETP.GEU` ×15, `FFMA` ×30; `:413` `MUFU.RCP`, `FCHK`, `FFMA` ×5,
`FSETP.GT`, `FSEL`, and the division's slow path (attributed to `:714`); `:317` `FMUL.FTZ` ×2, `MUFU.EX2`, `FADD.FTZ`,
`MUFU.RCP`; `:326-329`, `:339-341`, `q8_1_finite.hpp:19` and `cuda_fp16.hpp:264`: the q8_1 rows (section 3.3).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:357-363` | `mx = -FLT_MAX`; `mx = fmaxf(mx, m)` for c = d, d + 256, ... < nc | PTX `max.f32 %f618, %f618, %f397` (mx first); SASS `FMNMX R4, R14, R4, !PT`, ... 29 `FMNMX ... !PT` without `.FTZ` (the strided loop unrolled) | `fmax32(mx, m)` in c order |
| `:371` | `xp[j] = j < nc ? pa[j * cs] : 0.0f`, j < 32 | `LDG.E`; no arithmetic | loads |
| `:372` (`:39`) | `mx = warp_max(mx)` | `FMNMX R10, R5, R4, !PT` after each `SHFL.BFLY` 16, 8, 4, 2, 1 (×5) | `fmax32` butterfly, xor 16, 8, 4, 2, 1 |
| `:375-377` | `M = s_red[0]`, then `M = fmaxf(M, s_red[w])`, w = 1..7 | PTX `max.f32` ×7 in w order; SASS 7 `FMNMX ... !PT` | `fmax32(M, s_red[w])`, w ascending |
| `:380` | `s_w[c] = m == -FLT_MAX ? -1.0f : __expf(m - M)` | PTX `setp.eq.f32 %p49, %f74, 0fFF7FFFFF`; `sub.f32`; `mul.f32 .., 0f3FB8AA3B`; `ex2.approx.f32`. SASS `FSETP.NEU.AND P1, PT, R26, -3.40282346638528859812e+38, PT`; `@P1 FADD R7, -R5, R26`; `@P1 FMUL R25, R7, 1.4426950216293334961`; `@P1 FSETP.GEU.AND P2, PT, R25, -126, PT`; `@!P0 FMUL R25, R25, 0.5`; `@P1 MUFU.EX2 R26, R25`; `@!P0 FMUL R26, R26, R26` (P0 = P2 under P1); −1.0 (`HFMA2 R7, -RZ, RZ, -1.875, 0`) otherwise; 5 copies | `m == -FLT_MAX ? -1.0f : ex2_ieee(mul32(add32(m, -M), log2e))`, log2e = `0x3FB8AA3B`. The compare is exact in Metal: −FLT_MAX is normal, and a NaN m is unequal on both sides, so it takes `__expf` |
| `:389`, `:401`, `:408` | `if (w < 0.0f) continue;` | PTX `setp.lt.f32 %p58, %f89, 0f00000000`; SASS `FSETP.GEU.AND P0, PT, R4, RZ, PT`, the two `FFMA` predicated on P0 | `w < 0.0f` (exact: w is −1.0, NaN, or `ex2`'s result ≥ +0, whose subnormals this GPU compares as +0, still not below 0) |
| `:390`, `:402`, `:409` | `L = fmaf(s_l[c], w, L)` from `L = 0.0f`, c ascending | PTX `fma.rn.f32`; SASS `@P0 FFMA R26, R4, R35, RZ` (first chunk, the +0 addend kept), then `@P0 FFMA R26, R3, R5, R26`, ... | `fma32(s_l[c], w, L)`, c ascending from `L = 0.0f` |
| `:391`, `:403`, `:410` | `acc = fmaf(xp[c], w, acc)` from `acc = 0.0f`, c ascending | `@P0 FFMA R25, R4, R3, RZ`, then `@P0 FFMA R25, R56, R5, R25`, ... | `fma32(x_c, w, acc)`, c ascending from `acc = 0.0f` |
| `:413` | `o = L > 0.0f ? acc / L : 0.0f` | PTX `setp.gt.f32`; `div.rn.f32 %f556, %f719, %f718`; `selp.f32`. SASS `MUFU.RCP R3, R26`; `FCHK P0, R25, R26`; `FFMA R0, R3, -R26, 1`; `FFMA R0, R3, R0, R3`; `FFMA R3, R0, R25, RZ`; `FFMA R4, R3, -R26, R25`; `FFMA R0, R0, R4, R3`; `@!P0 BRA` past `CALL.REL.NOINC $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath`; `FSETP.GT.AND P2, PT, R26, RZ, PT`; `FSEL R15, R0, RZ, P2` | `L > 0.0f ? div32(acc, L) : 0.0f`, `div32` = main's `ieee_div` (`kernels/common/basalt_q8_1.metal:181`), whose precondition b ≥ 1 holds wherever the quotient is kept (notes) |
| `:714` (slow path) | `div.rn.f32` when `FCHK` flags the operands | `FSETP.GTU.FTZ` ×2 and `FSETP.NEU.FTZ` ×3 against `+INF` (the NaN and infinity tests); `@!P0 FFMA R0, R0, 1.84467440737095516160e+19, RZ` and `@!P1 FFMA R3, R3, 1.84467440737095516160e+19, RZ` (each operand whose exponent field is zero scaled by 2^64); `MUFU.RCP`, `FADD.FTZ R7, -R6, -RZ` and `FFMA` ×7 (the iteration); `FFMA.RZ`, `FFMA.RM`, `FFMA.RP` and `FSETP.NEU.FTZ` of the `.RP` result against the `.RM` one (the rounding of a subnormal quotient); `MUFU.RSQ R5, -QNAN` (0/0, ∞/∞); `FADD.FTZ R5, R0, R3` (a NaN operand) | `ptxas`'s IEEE division, not libdevice: `ieee_div` gives the same correctly rounded quotient; its NaN is 0x7fffffff (`cuda_nan`) |
| `:414` | `a.attn[(qi·n_head + h)·256 + d] = o` | `STG.E` | `cuda_nan(o)` |
| `:416` | `raw = a.q_full[(qi·n_head + h)·512 + 256 + d]` | `LDG.E R6, desc[UR12][R6.64+0x400]` | load |
| `:417` (`gate_fast`, `:308-319`) | inline PTX `mul.ftz.f32 t, raw, 0fBFB8AA3B`; `ex2.approx.ftz.f32 e, t`; `add.ftz.f32 d, e, 0f3F800000`; `rcp.approx.ftz.f32 r, d`; `mul.ftz.f32 y, o, r` | `FMUL.FTZ R0, R6, -1.4426950216293334961`; `MUFU.EX2 R0, R0` (no range branch: `.ftz`); `FADD.FTZ R8, R0, 1`; `MUFU.RCP R8, R8`; `FMUL.FTZ R15, R15, R8` | `gv = mul_ftz(o, rcp_ftz(add_ftz(expf_ftz(-raw), 1.0f)))` (`basalt_fp.metal:80`, `:83`, `:49`, `:107`): the helpers and order `native_qsa.md` section 3.2 gives the standalone `gate` kernel, whose five instructions these are. `expf_ftz(-raw)` is `ftz(fast::exp2(mul_ftz(raw, −log2 e)))` bit for bit (negation is exact, rounding to nearest even symmetric) |
| `:418` | `a.attn_gated[(qi·n_head + h)·256 + d] = gv` | `STG.E` | `cuda_nan(gv)` |
| `:419` | `q8_1_store_fm(gv, a.xq, i / 32, d & 31)`, i = (qi·n_head + h)·256 + d | section 3.3 | `q8_1_store_native(gv, xq, i)` (`basalt_q8_1.metal:52`) |

Notes:

- **The fma chains run in chunk order.** The three loops (`:386-392`, the 32 prefetched chunks; `:394-405`, blocks of
  8; `:406-411`, the rest) visit c = 0, 1, ..., nc − 1 once each, ascending, and skip the chunks marked −1, so L and
  acc are two fma chains in ascending c from +0. The prefetch (`:371`) and the unrolling change the loads only; the
  port writes one loop. The max chains (`:363`, `:372`, `:377`) give the same bits in any order except a zero's
  sign, which no output reads (section 3.1 notes: m − M is ±0 only where `__expf` gives 1 either way).
- **Why main's `ieee_div` fits.** A chunk's l is at least 1 unless it is NaN (its maximal cell adds `__expf(0)` = 1
  exactly, `:259`, `:267`, and every e is ≥ 0), and the chunk holding M gets w = `ex2(0)` = 1, so L is +0 (nc = 0 or
  every chunk skipped), at least 1, or NaN (a NaN chunk maximum gives w = NaN). `L > 0.0f` is false for +0 and NaN,
  so o = +0 there: NaN weights give a zero output, not NaN. The quotient is kept only for L ≥ 1 (or +∞), inside
  `ieee_div`'s precondition, and L is never subnormal, so Metal's `>` is exact.
- **Where a subnormal is reachable.** w is subnormal for a chunk maximum more than about 87.3 below M (`ex2_ieee`'s
  halving branch); such a w reaches both chains, and xp·w can underflow for a normal w too. acc, and so o, is
  subnormal only through such products; `ieee_div` then returns CUDA's subnormal quotient where Metal's `/` gives
  ±0. m − M and its product with log2(e) are subnormal only for chunk maxima within 2^-125 of each other. With
  flushing instead, o differs from CUDA's by about 2^-110 absolute at most, or by an ulp of a normal partial where a
  flushed subnormal L or acc would have moved the next chunk's fma off a rounding tie (section 3.1 notes); gv's
  `FMUL.FTZ` flushes a subnormal o anyway.
- **NaN at the stores.** A NaN that `FFMA`, the division (`FADD.FTZ R5, R0, R3`, `MUFU.RSQ R5, -QNAN`), `FMUL.FTZ` or
  `MUFU` produce or pass is NVIDIA's canonical 0x7fffffff; the port stores `cuda_nan(o)` and `cuda_nan(gv)` (`nan.md`).
- **The gate is approximate.** `gate_fast` writes `ex2.approx.ftz` and `rcp.approx.ftz` by hand in an IEEE-built file,
  so `ptxas` emits `MUFU.EX2` without the range branch and `MUFU.RCP` without a Newton step; invariant 5 maps them to
  `fast::exp2` and the correctly rounded `1.0f / d`, both flushed. The gated row and its q8_1 blocks are therefore not
  bit-exact to CUDA, and the contract's double test (rel < 2e-6) is their check. o (`:414`) does not pass through
  the gate: given the softmax weights it is IEEE-exact with the helpers above, and its only approximate inputs are the
  `__expf` weights (`:259-260`, `:380`).

### 3.3 The q8_1 rows against main's native sequence (`matvec.md`)

`q8_1_store_fm` (`:330-345`) is `native_quantize_q8_1_kernel`'s block written with inline `.ftz` PTX (`fm_max`,
`fm_abs`, `fm_add`, `fm_div`, `:326-329`) so that an IEEE-built file reproduces the `--use_fast_math` sequence.
Site by site against `matvec.md` section 1 (`native_mmvq.cu:146-159`, nvcc 13.2 there, 13.0 here):

| Site | `qsa_dec_attn.cu` | SASS here | SASS of the native quantizer (`matvec.md` §1) | Metal (`q8_1_store_native`) |
|---|---|---|---|---|
| \|xi\| | `:331` `fm_abs(xi)`: `abs.ftz.f32` | `FADD.FTZ R0, \|R15\|, -RZ`, and `\|R15\|` folded into the first max | `FADD.FTZ R4, \|R0\|, -RZ`, `\|R0\|` folded the same way | `abs_ftz(xi)` |
| amax butterfly | `:333` `fm_max(amax, __shfl_xor_sync(.., amax, o))`, o = 16..1: `max.ftz.f32` | `FMNMX.FTZ R4, \|R15\|, R0, !PT`, then `FMNMX.FTZ R5, R4, R5, !PT`, ... (×5, own operand first) | `FMNMX.FTZ` ×5 | `warp_max_ftz` |
| sum butterfly | `:335` `fm_add(sum, __shfl_xor_sync(.., sum, o))`: `add.ftz.f32` | `FADD.FTZ R8, R15, R6`, then `FADD.FTZ R9, R8, R7`, ... (×5) | `FADD.FTZ` ×5 | `warp_sum_ftz` |
| d | `:336` `fm_div(amax, 127.0f)`: `div.approx.ftz.f32 %f592, %f574, %f594` (`%f594` = `0f42FE0000`) | `FMUL.FTZ R5, R0, 0.0078740157186985015869` | `FMUL.FTZ R6, R4, 0.0078740157186985015869` | `mul_ftz(amax, as_type<float>(0x3C010204u))` |
| finite clamp | `:337` `q8_1_finite(d)` (`q8_1_finite.hpp:19`) | `FSETP.GT.AND P0, PT, \|R5\|, 65504, PT`; `LOP3.LUT R0, R6, 0x80000000, R5, 0xb8` (copysign of `0x477fe000`); `FSEL R9, R0, R5, P0` | `FSETP.GT.FTZ.AND`, `LOP3.LUT`, `FSEL` | `q8_1_finite` (`basalt_q8_1.metal:34`) |
| all-zero block | `:339` `amax != 0.0f` | `FSETP.NEU.AND P1, PT, R0, RZ, PT` | `FSETP.NEU.FTZ.AND P2, PT, R4, RZ` | `amax == 0.0f ? 0 : ...` |
| xi / d | `:340` `fm_div(xi, d)`: `div.approx.ftz.f32 %f598, %f557, %f392` | `MUFU.RCP R0, R9`; `FMUL.FTZ R0, R15, R0` | `MUFU.RCP R9, R7`; `FMUL.FTZ R0, R0, R9` | `mul_ftz(xi, rcp_ftz(d))` |
| `roundf` | `:340`: `copysign.f32` with `0f3F000000`, `add.rz.f32`, `cvt.rzi.f32.f32` | `LOP3.LUT R5, R5, 0x80000000, R0, 0xb8` (±0.5); `FADD.RZ R0, R0, R5`; `FRND.TRUNC R0, R0` | `LOP3.LUT`, `FADD.FTZ.RZ`, `FRND.TRUNC` | `round` |
| clamp, int8 | `:341`: `setp.gt/lt.f32` with ±127, `selp.f32` ×2, `cvt.rzi.s32.f32` | `FSETP.GT.AND P0, PT, R0, 127, PT`; `FMNMX.NAN R4, R0, -127, !PT`; `FSEL R4, R4, 127, !P0`; `F2I.TRUNC.NTZ R4, R4` | `FSETP.GT.FTZ`, `FMNMX.NAN R0, R9, -127`, `FSEL ... 127`, `F2I.FTZ.TRUNC.NTZ` | `q8_1_code` (NaN to 0) |
| qs store | `:343` `y[ib].qs[lane] = q` | `STG.E.U8 [R10.64+0x4]`, ib·36 by `IMAD.WIDE.U32 R2, R5, 0x24, R2` | the same layout | `q8_1_put` |
| ds | `:344` `if (lane == 0) y[ib].ds = q8_1_ds(d, sum)`: `q8_1_finite` on both, `cvt.rn.f16.f32` ×2 | `FSETP.GT.AND` ×2 with 65504, `LOP3.LUT` ×2, `FSEL` ×2; `F2FP.F16.F32.PACK_AB R5, R5, R0` (d low, sum high); `STG.E [R2.64]` | `F2FP.F16.F32.PACK_AB R13, R13, R0` | `q8_1_put`: `cuda_nan_f16(f16_bits(q8_1_finite(v)))`, d in the low half |

The sequences agree instruction for instruction. Every difference is a missing `.FTZ` on a compare, the round's add
or the integer conversion (`FSETP.GT`, `FSETP.NEU`, `FADD.RZ`, `F2I.TRUNC.NTZ` here; the `.FTZ` forms in the native
kernel), and none changes a bit: each of those instructions reads only flushed values (the outputs of `FMUL.FTZ`,
`FMNMX.FTZ` and `FADD.FTZ`), ±0.5 added to such a value (at least 0.5 in magnitude), or an integral value. The
block layout is the same too: `AttnQ81` (`:322-325`) is `half2 ds` then `int8_t qs[32]`, 36 bytes, the layout
`q8_1_put` writes. The block of dimension d is i / 32 with i = (qi·n_head + h)·256 + d and the lane d & 31, which is
`q8_1_store_native`'s indexing, so the merge calls main's device function unchanged: `q8_1_store_native(gv, xq, i)`.
Its value gv is the gate's `FMUL.FTZ` output, so it is never subnormal.

As in the native quantizer, the codes are exact only where `MUFU.RCP(d)` is the correctly rounded reciprocal
(`matvec.md` section 1 notes); here they are approximate before that, because gv itself is (section 3.2).

### 3.4 Library routines inlined into the on-path kernels

None. Neither kernel's PTX has a `call`, a `__nv_` or `__cudart_` symbol or an `.f64` operation, and their SASS has
no `D*` instruction; the one `CALL` (merge, `:413`) reaches `$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath`,
which is `ptxas`'s expansion of the IEEE division `div.rn.f32`, not library code. The CUDA math functions the two
kernels call each became PTX instructions whose result is fixed by IEEE rules, or the contract's one approximate
instruction:

| Function | Call sites | PTX | What its bits depend on |
|---|---|---|---|
| `fmaxf` | chunk `:39` (`warp_max`), `:257`, `:258`; merge `:39`, `:363`, `:377` | `max.f32` | the operands only: IEEE order, a NaN dropped, two NaNs 0x7fffffff (`fmax32`) |
| `fmaf` | chunk `:299`, `:300`; merge `:390-391`, `:402-403`, `:409-410` | `fma.rn.f32` | the exact a·b + c rounded once (`fma32`) |
| `__expf` | chunk `:259`, `:260`; merge `:380` | `mul.f32` by `0f3FB8AA3B`, `ex2.approx.f32` | `MUFU.EX2`, the contract's approximate operation (invariant 5: `fast::exp2`), inside `ptxas`'s range branch (section 3.0, section 4.3) |
| `roundf` | merge `:340` | `copysign.f32` with `0f3F000000`, `add.rz.f32`, `cvt.rzi.f32.f32` | exactly round half away from zero (`matvec.md` section 1): Metal's `round` |
| `fabsf`, `copysignf` | merge `q8_1_finite.hpp:19` (twice for d, once for the sum) | `abs.f32`, `copysign.f32` | sign-bit operations: `q8_1_finite` |

The rest is not library code either: `__half22float2` and `__float2half` are `cuda_fp16.hpp`'s inline PTX
(`cvt.f32.f16`, `cvt.rn.f16.f32`), `gate_fast` and the `fm_*` wrappers are basalt's own inline PTX, and `/` is
`div.rn.f32`. No routine with a polynomial or table body (the library `expf`, `logf`, `log1pf`, `sinf`, `cosf`,
`powf`, ...) is reached, so nothing here falls under the libdevice rule.

## 4. What the record implies for the Metal port

### 4.1 Helpers by kernel

The helper inventory of both on-path kernels: what each page name in section 3 is, where it lives today, and its
sites. File paths are under `kernels/common/`; "main" is origin/main e7eb8b4, "sampler" the unmerged branch
`origin/woodsonl/basalt-sampler`.

| Page name | Helper | main | sampler branch | `attn_chunk_v2<0, false>` | `attn_merge_v2` |
|---|---|---|---|---|---|
| widen fp16 | `h2f` | `basalt_fp.metal:110` | same | `:116-123` (K), `:285-286` (V) | |
| `add32` | `ieee_add` | `basalt_q8_1.metal:95` | moved to `basalt_ieee.metal:32` | `reduce12` (`:46-68`), x − m (`:259-260`), e0 + e1 (`:267-268`) | m − M (`:380`) |
| warp sum | `warp_sum_ieee` | `basalt_q8_1.metal:206` | `basalt_q8_1.metal:146` | `:267-268` | |
| `mul32` | `ieee_mul` | `basalt_q8_1.metal:108` | `basalt_ieee.metal:45` | the dot's plain product (`:234-235`), `:243`, inside `ex2_ieee` | inside `ex2_ieee` (`:380`) |
| `fma32` | fma with IEEE subnormals, any operands | none: main's `ieee_fma` (`basalt_q8_1.metal:167`) needs an integer middle operand below 2^24, which no site here has | `ieee_fma_count` (`basalt_sampler.metal:80`), general despite its name; its slow path is `basalt_f64.metal`'s `f64_fma_odd` (`:231`) | the dot's 7 fused terms (`:234-235`), `:299-300` | `:390-391`, `:402-403`, `:409-410` |
| `div32` | `ieee_div` | `basalt_q8_1.metal:181`; its precondition b ≥ 1, ∞ or NaN holds (section 3.2) | `basalt_q8_1.metal:121` | | `:413` |
| `fmax32` | CUDA's `fmaxf` on bits | none | `fmaxf_bits` with `okey` and `fnan` (`basalt_sampler.metal:37-52`) | `:257-258`, `warp_max` (`:39`) | `:363`, `warp_max` (`:39`), `:377` |
| `ex2_ieee` | `ptxas`'s `ex2.approx.f32` expansion around `fast::exp2` (section 3.0); new, or one of the two forms beside it (section 4.3) | none: `expf_neg` (`basalt_fp.metal:104`) has no range branch | none: `expf_approx` (`basalt_sampler.metal:93`) has no range branch | `:259-260` | `:380` |
| `reduce12` | the 12-head reduce-scatter (section 3.1); new, kernel-local: the selects on the lane bit, `shfl_xor`, `ieee_add` | `shfl_xor` (`basalt_fp.metal:85`) | same | `:46-68` | |
| gate | `mul_ftz`, `rcp_ftz`, `add_ftz`, `expf_ftz` | `basalt_fp.metal:80`, `:83`, `:49`, `:107` | same | | `:417`, in `native_qsa.md` section 3.2's order |
| q8_1 rows | `q8_1_store_native` (with `abs_ftz`, `warp_max_ftz`, `warp_sum_ftz`, `mul_ftz`, `rcp_ftz`, `q8_1_finite`, `q8_1_code`, `q8_1_put`) | `basalt_q8_1.metal:52` | `basalt_q8_1.metal:53` | | `:419` |
| NaN at stores | `cuda_nan` | `basalt_fp.metal:120` | same | | `:414`, `:418` |

Notes:

- **Only the merge's stores need `cuda_nan`.** The chunk kernel writes scratch (`part_m`, `part_l`, `part_acc`,
  `:270-271`, `:304-305`) that only the merge reads, and no NaN encoding there changes an output: the merge's tests
  on those values (`m == -FLT_MAX`, `w < 0.0f`, `L > 0.0f`) are false for every NaN, `fmax32` drops one, arithmetic
  passes one on, and the merge's own stores canonicalize. The q8_1 block's NaN goes through `q8_1_put`'s
  `cuda_nan_f16`.
- **`fma32`'s fast path and this kernel's zeros.** `ieee_fma_count` takes the hardware's `fma` only when all three
  operands have a nonzero exponent field and the result is at least 2^-125, so every zero operand takes the software
  binary64 path. The attention has many: each chain's +0 start (`FFMA ... RZ`, section 3.2; `acc` from 0.0f at
  `:299-300`), the +0 weight of every masked cell (`:259-260`), and every zero element of K, V or q. The result is
  correct, but the kernel lane should add a fast path for exact zeros (a zero product leaves c, with IEEE's signed-zero
  rule when c is a zero) and test it as the sampler's helpers are (`tests/kernels/test_basalt_sampler_helpers.py`).
- **No ordered compare needs `key32`.** The only float compares outside the max chains (`m == -FLT_MAX`,
  `w < 0.0f`, `L > 0.0f` and the q8_1 rows' compares on flushed values) are exact with Metal's operators (section 3.2),
  so neither kernel needs `f_gt`/`f_lt`.

### 4.2 What basalt's tests demand

`src/kernels/qsa_dec_attn_parity.cpp` (399 lines) is the file's test. It runs eight cases per pool kind
(`:202-210`: one cell; widths 63, 64, 65; the identity window of 1,250 to 1,252 cells; two full selections; five
rows with every 7th page of the shuffled page table masked to −1; eight full rows; one 4,096-cell row; 1,250 cells
in a 4,096 cap), with `cap = qsa_selection_width(kTopkMaxCells, s)` = 2,048 + 4 − 1 = 2,051 (`:201`,
`include/basalt/kernels/qsa.hpp:130-133`), so 33 chunks of 64 cells, and the real shapes (24 query heads, 2 KV heads,
`head_dim` 256, `page_size` 4, `qsa.hpp:105-113`). Its fixture is `std::mt19937 rng(7)` (`:53`): fp16 K and V
`f16(nd(rng) * 0.7f)` (`:68`), q `nd(rng) * 2.0f` ("scores of a few units: many probabilities far below the max",
`:80`), the gate half of `q_full` `nd(rng) * 3.0f` (`:81`). Mode 1 (int8 pools) runs first, mode 0 (fp16) second
(`:202`); only mode 0 is on the contract path (section 2). Spec part D (`.context/qsa/spec-tests-d.md` sections 2-3)
gives every check an id and the Metal test that replaces it; this section records what each demands of the sites above.

- **The attention output, bitwise against the older kernel (A1 `b0`, A2 `bad`).** `b0 += std::memcmp(&r0[i], &x0[i],
  4) != 0;` (`:142`) and `if (std::memcmp(&r[i], &x[i], 4) != 0) { ++bad; ...` (`:155`), failing through
  `if (b0 || b1 || b2) ++g_fail;` (`:146`) and `if (bad || gbad) ++g_fail;` (`:158`): basalt requires
  `qsa_decode_attn_batch` (`qsa_decode_attn.cu`, which no engine path runs) and this file's pair to give the same bits,
  and the same bits again when the merge also writes the gated row and the q8_1 rows. The Metal port does not carry
  the older kernel (spec part D section 3.1); what the record fixes is that the gated and q8_1 variants change
  nothing before the store of o (`:414`), because the gate reads o after it is computed (section 3.2), so the port's
  three calls of a case agree bit for bit, and every row of an `n_q`-row call equals its one-row call, since a CTA
  of either kernel reads one query's ids, step record and q (sections 3.1 and 3.2, invariant 1). The CUDA bits
  themselves depend on `MUFU.EX2` at `:259`, `:260` and `:380` and, below 2^-126, on IEEE subnormals (section 3.1
  and 3.2 notes): no basalt criterion compares the fp16 output with anything but the older CUDA kernel, so the
  port's check on o against CUDA's arithmetic is the contract's double test (`flash-next.md:461`, `qsa_parity`'s
  rule, below), which neither difference approaches.
- **The gated row, bitwise against `native_qsa_gate_apply` (A1 `b1`, A2 `gbad`).** `b1 += std::memcmp(&g0[i],
  &g1[i], 4) != 0;` (`:143`) and `if (std::memcmp(&gr[i], &gx[i], 4) != 0) ++gbad;` (`:156`), the reference gated
  by `k::native_qsa_gate_apply(out_ref, qfull, gated_ref, NH * n_q, D, st);` (`:131`). The merge's five `gate_fast`
  instructions (`:417`) are the standalone gate kernel's (`native_qsa.md` section 3.2), so one helper per operation
  (`mul_ftz`, `rcp_ftz`, `add_ftz`, `expf_ftz`, in that kernel's order) keeps the Metal merge's gated row bitwise
  equal to the Metal gate applied to the same o, which is what the Metal test compares (spec part D section 3.5). The
  gate is approximate (`ex2.approx.ftz`, `rcp.approx.ftz`), so its bits against CUDA's are the contract's double
  test, `rel < 2e-6` (`flash-next.md:462`, `src/kernels/shared_expert_parity.cpp:372`).
- **The q8_1 rows, bitwise against `native_quantize_q8_1` (A1 `b2`).** `for (size_t i = 0; i < nb; ++i) b2 += q0[i]
  != q1[i];` (`:145`), the reference being `k::native_quantize_q8_1(gated_ref, xq_ref, NH * D, (int) n_q, st);`
  (`:133`) over rows of 24 · 256 = 6,144 values, 192 blocks of 36 bytes each. Section 3.3 shows `q8_1_store_fm` and the
  native quantizer are the same instruction sequence and the same block indexing, so the merge calling main's
  `q8_1_store_native` (`kernels/common/basalt_q8_1.metal:52`), the device function `native_quantize_q8_1`
  (`:60`) runs, meets this check by construction, whatever `MUFU.RCP` does on CUDA.
- **The wide window (A3, A4): off the fp16 path.** The repeat check `if (std::memcmp(again.data(), out[1].data(),
  again.size() * 4) != 0) ++unstable;` (`:292`) and the double check `const bool ok = e_new <= 4e-3 *
  std::max(max_ref, 1e-6);` (`:296`) run only on `pools_of(1)` (`:225`), the int8 tensor-core window (`:764-775`).
  On fp16 pools the same call takes this file's pair (section 2), which has no atomics and no order that depends on
  timing (every reduction above has a fixed order), so repeated calls give the same bits; spec part D (section 3.6)
  keeps both checks on fp16 pools.

- **`qsa_parity`'s attention rule, which the contract borrows.** `qsa_parity` never runs this file: its attention
  check runs `qsa_attend` on cells gathered by `kv_gather` (`src/kernels/qsa_parity.cpp:1091`, `:1094`) against
  `ref_attend` (`:314-343`) on the same fp16 cache (`:1100`), `num += std::fabs(attn[i] - attn_f16[i]);` and
  `den += std::fabs(attn_f16[i]);` (`:1103-1104`), failing on `if (!(num / den <= 1e-5)) { std::printf("    *** over
  1e-5 ***\n"); ++g_bad; }` (`:1117`). The contract applies that rule to this pair's o (`flash-next.md:461`; spec
  part D section 7.1). `ref_attend` is plain double arithmetic: the scale `1.0 / std::sqrt((double) head_dim)`, the
  KV head `h / (n_head / n_head_kv)`, each score a sum over d in order, then `std::exp(s - mx)`, the weights divided
  by their sum, and o as a sum over the cells in order (`:318-341`). The bound is about 170 times fp32's unit
  roundoff (2^-24), so a numpy float64 transcription serves (no host library result is needed bit for bit), and the
  port's departures from CUDA's bits recorded above (the last bits of the `fast::exp2` weights; wherever a subnormal
  is flushed rather than emulated, about 2^-110 absolute per value or one ulp of a partial at a rounding tie) stay far
  below it. `qsa_parity`'s gate
  check is the fp16 `qsa_gate_apply`'s, `if (!(gnum / gden <= 1e-3))` (`:1142`, `:1154`), another kernel; the
  merge's gate takes `shared_expert_parity`'s `rel < 2e-6` instead (`src/kernels/shared_expert_parity.cpp:372`,
  `flash-next.md:462`).
- **`qsa_seg_parity` uses the pair only as a reference.** It runs `qsa_dec_attn` once per slot (`:263`) against the
  segment entry `qsa_dec_attn_seg` (`:281`), comparing `attn`, `gated` and the q8_1 rows bitwise (`:363-365`,
  `if (tot) ++g_fail;` at `:382`), on int8 and fp16 pools. The segment kernels run only on verify's batch-record path
  with int8 pools (`src/core/verify.cpp:1539`, section 2), so the check constrains no kernel the port carries, and
  the test is not in the unit's list.
- **Nothing else.** The engine calls (`src/core/verify.cpp:1563`, `src/core/mtp.cpp:883`) and these two programs are
  the only calls of `qsa_dec_attn` at b4c63b5 (`git grep -F 'qsa_dec_attn('`); `qsa_dec_fused_parity` does not call
  it. So no basalt criterion compares this pair's fp16 output, gated row or q8_1 rows with a host value: on CUDA every
  check of them is kernel against kernel.

### 4.3 Points for the contract and the kernel lane

- **The attention's `__expf` needs `ex2_ieee`, and invariant 5 should say so.** Invariant 5 maps `MUFU.EX2` to
  `fast::exp2` (`docs/design/flash-next.md:135`); the `FSETP.GEU` against −126, the halving `FMUL` and the squaring
  `FMUL` around it at `:259`, `:260` and `:380` are `ptxas`'s IEEE instructions (section 3.0), and a subnormal
  weight reaches o (sections 3.1 and 3.2 notes), so the port writes them: `ex2_ieee(t) = t < -126.0f ? ieee_mul(e,
  e) : fast::exp2(t)` with `e = fast::exp2(t * 0.5f)`, one helper at all three sites, so the plain, gated and q8_1
  calls stay bitwise equal. The kernel spec already writes it (`.context/qsa/spec-kernels-attn.md`, section 8 row F3
  and section 4's `:380` row). Inside it, `fast::exp2` meets only arguments whose result is normal (t ≥ −126 on the
  plain branch, t/2 in [−126, −63) on the other for t ≥ −252) or whose square rounds to +0 whatever it returns (t <
  −252), so the helper does not depend on what `fast::exp2` returns for a subnormal result, the [H] of `sampler.md`
  section 3.3. The one exception is an argument just above −126, where a `fast::exp2` that undershoots 2^-126 would
  give a flushed value [H]; that is the approximate operation's own error, which invariant 5 accepts.
- **One helper for `ex2.approx.f32`.** Main's `expf_neg` (`kernels/common/basalt_fp.metal:104`) and the sampler
  branch's `expf_approx` (`basalt_sampler.metal:93`) are the same PTX operation without the branch; at their sites
  the branch changes no output (`experts.md` section 2; `sampler.md` section 3.3, which recommends leaving it out
  there). Invariant 5's "One helper serves each operation at every site" then holds only if those two become
  `ex2_ieee` of their products, which leaves their outputs as they are. Recommendation: the contract names
  `ex2_ieee` as the one helper of the non-`.ftz` `ex2.approx.f32` (`__expf` in an IEEE-built file), placed beside
  `ieee_mul`, whose square it calls; `expf_ftz` stays the `.ftz` form (`MUFU.EX2` alone). If merged
  `basalt_fp.metal` is not to change, invariant 5 records the two branch-free helpers as an exception instead.
- **`qsa-attn` depends on the sampler branch.** `fma32` (`ieee_fma_count`, `basalt_sampler.metal:80`, with
  `basalt_f64.metal`'s `f64_fma_odd`) and `fmax32` (`fmaxf_bits`, `okey`, `fnan`, `:37-52`) exist only on
  `origin/woodsonl/basalt-sampler`, unmerged and cut from an older main; everything else section 4.1 lists is on
  main. So `qsa-attn` either stacks on that branch or the helpers land on main first, with `basalt_ieee.metal`,
  `basalt_f64.metal` and their tests (`tests/kernels/test_basalt_sampler_helpers.py`, H.3 and H.6). The attention
  calls the fma helper at every fma, so it wants a general name; `ieee_fma` is taken by main's integer-operand
  helper (`basalt_q8_1.metal:167`), and `ieee_fma_any` matches the sampler branch's `ieee_div_any`.
- **`fma32`'s zeros, exactly.** The hardware path requires every operand's exponent field to be nonzero, so each
  chain's +0 start (`FFMA ... RZ`, `:390-391`; `acc` from +0, `:280`), each masked cell's zero weight and value
  (`:259-260`, `:187`) and each zero element of K, V or q takes the binary64 software path. Two relaxations keep the
  result exact: (a) c = ±0 with a and b as now (nonzero exponent fields) is the hardware fma under the same result
  test (|r| ≥ 2^-125), because a true zero is read as zero and a nonzero product decides the sign; this sends every
  chain start to the hardware, the main saving. (b) a = ±0 or b = ±0 with the other factor finite makes the product
  an exact zero, so the result is c's bits when c is nonzero and not NaN (a subnormal c included), and for c = ±0 it
  is −0 only when c and the product (sign(a) XOR sign(b)) are both negative, else +0. 0 · ∞ is NaN, so (b) leaves an
  infinite or NaN factor to the existing path. The tests add these cases (each zero sign, a subnormal c, 0 · ∞)
  beside H.3.
- **No IEEE helper can be dropped for being small.** A flushed subnormal addend moves the next fma off a rounding
  tie (section 3.1 notes) and leaves a normal partial one ulp from CUDA's, so a cheaper form must give the same bits
  on every input, not only within 2^-110. What the checks cost in the two hot loops (per thread, 192 fma-class
  operations per two-cell scores pass and 192 `fmaf` per eight cells of the value loop) is for the kernel lane to
  measure.
- **The drafter's wide window is int8-only, and three contract rows cite it.** The approximate-operations list
  (`flash-next.md:64-65`, "the drafter's wide window, `:576-577`, `:667`"), the parity table's `qsa_dec_attn_parity`
  row (`flash-next.md:373`, "the draft-only wide window against double", criterion `:296`) and the double-test
  table's attention row (`flash-next.md:461`, "checks only the drafter's wide window against double") describe
  kernels that run only with int8 pools (section 2: `:764`; section 4.2: `pools_of(1)`,
  `src/kernels/qsa_dec_attn_parity.cpp:225`). On the fp16 path the drafter runs the decode pair with `a.chunk` = 64.
  Recommendation: the contract marks those sites and that check off-path (int8 KV), the wide window leaves
  `qsa-attn`'s scope, and the repeat and double checks run on fp16 pools as spec part D section 3.6 keeps them.
- **Per-step values (invariant 4)** are already where the invariant wants them: the width `n_ids` comes from the
  query's step record (`:150`, `:159`, `:352`), the cell ids, page table and query from buffers (`:148-149`,
  `:174`); a chunk CTA past the width returns (`:168-169`) and the merge stops at `nc` (`:354`). The launch carries
  only static values: the grid from `cap` and `n_q` (`:755`, `:763`, `:787`), `a.scale` (`:757`), `a.chunk` = 64,
  `page_size`, `n_chunks` and the scratch stride.
- **Invariant 1.** Row qi's o reads only its own ids[0, `n_ids`), q, step record and the pools: the chunk partials
  cover fixed 64-cell spans of its selection, and the merge visits c < nc in ascending order, with nc = ⌈`n_ids`/64⌉
  whenever cap ≥ `n_ids`. So neither cap nor `n_q` nor the row's index enters a row's bits, which is what section
  4.2's one-row comparison checks.
- **The gate and the q8_1 rows need nothing new.** The gated row's CUDA check is the double test `rel < 2e-6`
  (`flash-next.md:462`), and its Metal check the standalone gate on the same o (section 4.2). The q8_1 rows are
  main's `q8_1_store_native` unchanged (section 3.3).
- **Nothing falls under the libdevice rule** (section 3.4): the pair inlines no library routine, so this page leaves
  no open question for NVIDIA's code.
