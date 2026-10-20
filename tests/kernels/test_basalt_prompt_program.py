"""basalt's prompt-path op kinds (monolith/ops/basalt_prompt.py) in compiled programs on the GPU. A session compiles
the dynamic-T prompt program once, at its chunk bound B, and replays it for every chunk with StepState.t_this_step set
to the chunk's rows (monolith/generate.py:604); this tier runs the handlers' dispatches (compiler/emit_basalt_prompt.py)
that way, through Engine, over a synthetic .basalt file (tests/contract/test_basalt_ops.py ``write_file``), and checks
what the single-kernel tier (tests/kernels/test_basalt_prompt_gemm.py) cannot see: that the handlers bind what the
kernels read, and that a program's rows are StepState data (dynamic-T) or its compiled count (static). Each prompt
kind joins ``graph``.

- GM-PROG: Gemm::bf16 (src/prefill/gemm.cu:122-132) of BF16 rows by dense/a, 48 BF16 rows over 2560 inputs
  (ssm_alpha's shape, src/prefill/prefill.cpp:1587), in dynamic-T programs at B 64 and B 128 on one StepState layout
  (t_max 128), each replayed at t_this_step 1, 17 and 64, and in a static program at T 17, replayed at t_this_step 1
  (what the advance leaves after a step, kernels/common/advance.metal:34, so a static program that read its rows from
  StepState would compute one). Each replay runs twice, into an output prefilled 0xFF before each run: the rows below
  the chunk's are the single-kernel helper's (tests/kernels/basalt_prompt_gpu.py ``prompt_gemm``) on the same bytes,
  one whole product at B 128, so both bounds and the static program give the same rows; the rest keep the fill; and
  the second run gives the first's bits."""

import numpy as np

from monolith.core import DType, Graph
from monolith.core.shapes import T
from monolith.core.step_state import StepStateLayout
from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.contract.test_basalt_ops import emit, op, weight, write_file
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_prompt_gpu as gpu
from tests.kernels.test_basalt_mmvf import bf16_weights

F32 = np.float32
FILL = 0xFFFFFFFF                       # an output word no dispatch wrote (the 0xFF prefill)
LAYOUT = StepStateLayout(t_max=128)     # the session's one layout: every program compiles against it
N, K = 48, 2560                         # dense/a


def graph(g, src):
    """The prompt kinds over the file: Gemm::bf16 of xb [T, K] by dense/a into y.a [T, N]."""
    xb = g.input("xb", (T, K), DType.BF16)
    op(g, "basalt_prompt_gemm", [xb, weight(g, "dense/a", src)], [g.value("y.a", (T, N), DType.F32)], entry="bf16")


def test_prompt_gemm_in_a_program(tmp_path):
    """GM-PROG on dense/a: (t, dynamic-T, [(t_this_step, the rows computed)]) per program, against one whole product
    of the 128 rows at B 128 (asserted live: finite, one output nonzero)."""
    rng = np.random.default_rng(79)
    wb = bf16_weights(rng, N, K)
    xb = emu.bf16_bits(rng.standard_normal((128, K)).astype(F32))
    src = BasaltSource(write_file(tmp_path / "prompt.basalt", {}, {"a": (N, K, wb.tobytes())}))
    ref = gpu.prompt_gemm(xb, wb, bf16=True, B=128)
    assert np.isfinite(ref.view(F32)).all() and np.count_nonzero(ref), "a vacuous reference"
    programs = [(b, True, [(r, r) for r in (1, 17, 64)]) for b in (64, 128)] + [(17, False, [(1, 17)])]
    for t, dynamic, replays in programs:
        g = Graph("step")
        graph(g, src)
        eng = Engine(emit(g, src, t, dynamic_t=dynamic, layout=LAYOUT), gpu.device())
        eng.buffers["xb"].write(xb[:t].tobytes(), 0)
        for r, m in replays:
            eng.buffers[eng.program.step_state].write(LAYOUT.pack({"t_this_step": r}), 0)
            ys = []
            for _ in range(2):
                eng.buffers["y.a"].fill(0xFF)
                assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
                ys.append(np.frombuffer(eng.read("y.a"), np.uint32).reshape(t, N))
            y, again = ys
            assert np.array_equal(y[:m], ref[:m]) and (y[m:] == FILL).all() and np.array_equal(again, y), (
                "port: rows < r bitwise equal to the kernel-tier helper's rows, rows >= r keep 0xFF, a second replay "
                "identical, both bounds and the static program (at r 17) giving the same rows "
                f"({'dynamic-T at B' if dynamic else 'static at T'} {t}, t_this_step {r}, dense/a): "
                f"{np.count_nonzero(y[:m] != ref[:m])} outputs below row {m} differ from the helper's, "
                f"{np.count_nonzero(y[m:] != FILL)} at or past it written, {np.count_nonzero(again != y)} differ in "
                "the second run")
