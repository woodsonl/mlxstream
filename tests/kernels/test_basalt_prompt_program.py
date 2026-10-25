"""basalt's prompt-path op kinds (monolith/ops/basalt_prompt.py) in compiled programs on the GPU. A session compiles
the dynamic-T prompt program once, at its chunk bound B, and replays it for every chunk with StepState.t_this_step set
to the chunk's rows (monolith/generate.py:604); this tier runs the handlers' dispatches (compiler/emit_basalt_prompt.py)
that way, through Engine, over a synthetic .basalt file (tests/contract/test_basalt_ops.py ``write_file``) and the
copies pack ``write_pack`` (monolith/packs/basalt_hc8.py) makes of it, and checks what the single-kernel
tier (tests/kernels/test_basalt_prompt_gemm.py) cannot see: that the handlers bind what the kernels read, and that a
program's rows are StepState data (dynamic-T) or its compiled count (static). Each prompt kind joins ``graph``.

- GM-PROG: Gemm::bf16 (src/prefill/gemm.cu:122-132) of BF16 rows by dense/a, 48 BF16 rows over 2560 inputs
  (ssm_alpha's shape, src/prefill/prefill.cpp:1587); Gemm::native (:169-185) of FP16 rows by gguf/b, Q2_0 256 x
  2560, then gguf/c, Q4_K 64 x 2560; to_f16 and to_bf16 (src/prefill/kernels.cu:818-833) of F32 rows of 2560 (the
  PLE batch's, prefill.cpp:1531-1536); then Gemm::bf16_w (gemm.cu:134-154) of BF16 rows by an int8 hyper-connection
  copy, that of hc8_parity's down matrix (tests/basalt_hc_ref.py; 320 x 10240, the pack's largest image), stored in
  the file as the BF16 row blk.0.hc_attn_down.weight, which write_pack quantizes into the rows hc8/<name>/q and /s
  the op reads. The three images (gguf/b's and gguf/c's dequantizations, then the copy's BF16 image) share the
  program's one scratch, sized to the copy's. The programs: dynamic-T at B 64 and B 128 on one StepState layout
  (t_max 128), each replayed at t_this_step 1, 17 and 64, and static at T 17, replayed at t_this_step 1 (what the
  advance leaves after a step, kernels/common/advance.metal:34, so a static program that read its rows from
  StepState would compute one). Each replay runs twice, into outputs prefilled 0xFF before each run: the rows below
  the chunk's are the single-kernel helpers' (tests/kernels/basalt_prompt_gpu.py ``prompt_gemm``, ``native``,
  ``convert`` and ``prompt_gemm_copy``) on the same bytes, one whole run of the 128 rows at B 128 each, so both
  bounds and the static program give the same rows (at B 128 the handler splits the copy's product into two column
  slices); gguf/b's rows show that gguf/c's dequantization did not overwrite the scratch while gguf/b's products
  read it, and gguf/c's that the copy's image did not while gguf/c's did (the reuse is ordered); the rest keep the
  fill; and the second run gives the first's bits. The file is bound through its windows, and again through exact
  per-tensor copies (``BasaltSource(copy=True)``), so that shader validation bounds every read of the file by its
  own tensor (gguf/b and gguf/c share a page); the copies pack binds through its one window in both modes (GM-DQ8
  binds a copy exactly)."""

import numpy as np
import pytest

from monolith.core import DType, Graph
from monolith.core.shapes import T
from monolith.core.step_state import StepStateLayout
from monolith.packs.basalt import BasaltSource
from monolith.packs.basalt_hc8 import write_pack
from monolith.runtime import Engine
from tests.basalt_hc_ref import hc8_parity_matrices
from tests.contract.test_basalt_ops import emit, op, weight, write_file
from tests.contract.test_basalt_prompt_ops import copy_op
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_prompt_gpu as gpu
from tests.kernels.test_basalt_mmvf import bf16_weights
from tests.kernels.test_basalt_mmvq import weights

F32 = np.float32
LAYOUT = StepStateLayout(t_max=128)     # the session's one layout: every program compiles against it
N, K = 48, 2560                         # dense/a
GGUF = {"b": (42, 256), "c": (12, 64)}  # gguf/<name>: (ggml type, rows) over K: Q2_0, Q4_K
CONVERTS = {"f16": DType.F16, "bf16": DType.BF16}   # y.<name>: to_f16, to_bf16 of xf [T, K] F32
HC, HN, HK = "blk.0.hc_attn_down.weight", 320, 10240   # the copy: its table name, rows and columns


def graph(g, src):
    """The prompt kinds over the file and its copies: Gemm::bf16 of xb [T, K] by dense/a into y.a [T, N], Gemm::native
    of x16 [T, K] by gguf/b and gguf/c into y.b and y.c, to_f16 and to_bf16 of xf [T, K] into y.f16 and y.bf16, then
    Gemm::bf16_w of x.<HC> [T, HK] by HC's copy into y.<HC> [T, HN]."""
    xb = g.input("xb", (T, K), DType.BF16)
    op(g, "basalt_prompt_gemm", [xb, weight(g, "dense/a", src)], [g.value("y.a", (T, N), DType.F32)], entry="bf16")
    x16 = g.input("x16", (T, K), DType.F16)
    for name, (_, n) in GGUF.items():
        op(g, "basalt_prompt_gemm", [x16, weight(g, f"gguf/{name}", src)], [g.value(f"y.{name}", (T, n), DType.F32)],
           entry="native")
    xf = g.input("xf", (T, K), DType.F32)
    for name, dtype in CONVERTS.items():
        op(g, "basalt_prompt_convert", [xf], [g.value(f"y.{name}", (T, K), dtype)])
    copy_op(g, HC, HN, HK)


@pytest.mark.parametrize("copy", [False, True], ids=["windows", "exact-copies"])
def test_prompt_gemm_in_a_program(tmp_path, copy):
    """GM-PROG: (t, dynamic-T, [(t_this_step, the rows computed)]) per program, against one whole product of the 128
    rows at B 128 per weight (each asserted live: finite, one output nonzero), the copy's on the copies pack's own
    bytes, and one conversion of xf's 128 rows at B 128 per 16-bit dtype (each asserted live: every element written,
    one nonzero); each program's one scratch is asserted to be the copy's image's size, so the three images share it."""
    rng = np.random.default_rng(79)
    wb = bf16_weights(rng, N, K)
    xb = emu.bf16_bits(rng.standard_normal((128, K)).astype(F32))
    blocks = {name: weights(tid, K, n, rng) for name, (tid, n) in GGUF.items()}
    x16 = emu.half_bits(rng.standard_normal((128, K)).astype(F32))
    xf = rng.standard_normal((128, K)).astype(F32)
    xc = emu.bf16_bits(rng.standard_normal((128, HK)).astype(F32))
    gguf = {name: (tid, n, K, blocks[name].tobytes()) for name, (tid, n) in GGUF.items()}
    dense = {"a": (N, K, wb.tobytes()), HC: (HN, HK, hc8_parity_matrices()[0].tobytes())}
    src = BasaltSource(write_file(tmp_path / "prompt.basalt", gguf, dense), copy=copy)
    copies = write_pack(src.file, tmp_path / "copies")
    q, s = copies.aux_array(f"hc8/{HC}/q"), copies.aux_array(f"hc8/{HC}/s").view(np.uint16)
    ref = {"a": gpu.prompt_gemm(xb, wb, bf16=True, B=128),            # y.<name>: the rows of <name>'s product
           **{name: gpu.native(tid, blocks[name], x16, B=128)[0] for name, (tid, _) in GGUF.items()},
           HC: gpu.prompt_gemm_copy(xc, q, s, slices=[(0, 128, 0, HN)])[0]}
    for name, want in ref.items():
        assert np.isfinite(want.view(F32)).all() and np.count_nonzero(want), f"a vacuous reference (y.{name})"
    for name in CONVERTS:                                             # y.<name>: xf's rows converted
        ref[name] = gpu.convert(xf, bf16=name == "bf16", B=128, rows=128)
        assert (ref[name] != 0xFFFF).all() and np.count_nonzero(ref[name]), f"a vacuous reference (y.{name})"
    programs = [(b, True, [(r, r) for r in (1, 17, 64)]) for b in (64, 128)] + [(17, False, [(1, 17)])]
    for t, dynamic, replays in programs:
        g = Graph("step")
        graph(g, src)
        prog = emit(g, src, t, [copies], dynamic_t=dynamic, layout=LAYOUT)
        scratch = {b: prog.buffers[b].nbytes for b in prog.buffers if "prompt_gemm_f16" in b}
        assert list(scratch.values()) == [2 * HN * HK], f"one scratch for the three images, the copy's size: {scratch}"
        eng = Engine(prog, gpu.device())
        eng.buffers["xb"].write(xb[:t].tobytes(), 0)
        eng.buffers["x16"].write(x16[:t].tobytes(), 0)
        eng.buffers["xf"].write(xf[:t].tobytes(), 0)
        eng.buffers[f"x.{HC}"].write(xc[:t].tobytes(), 0)
        for r, m in replays:
            eng.buffers[eng.program.step_state].write(LAYOUT.pack({"t_this_step": r}), 0)
            runs = []
            for _ in range(2):
                for name in ref:
                    eng.buffers[f"y.{name}"].fill(0xFF)
                assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
                runs.append({name: np.frombuffer(eng.read(f"y.{name}"), want.dtype).reshape(t, -1)
                             for name, want in ref.items()})
            for name, want in ref.items():
                y, again, fill = runs[0][name], runs[1][name], np.iinfo(want.dtype).max   # the 0xFF prefill's word
                assert np.array_equal(y[:m], want[:m]) and (y[m:] == fill).all() and np.array_equal(again, y), (
                    "port: rows < r bitwise equal to the kernel-tier helpers' rows, rows >= r keep 0xFF, a second "
                    "replay identical, both bounds and the static program (at r 17) giving the same rows "
                    f"({'dynamic-T at B' if dynamic else 'static at T'} {t}, t_this_step {r}, y.{name}, "
                    f"{'exact copies' if copy else 'windows'}): {np.count_nonzero(y[:m] != want[:m])} outputs below "
                    f"row {m} differ from the helper's, {np.count_nonzero(y[m:] != fill)} at or past it written, "
                    f"{np.count_nonzero(again != y)} differ in the second run")
