"""Single-kernel runs of basalt's prompt path (kernels/common/basalt_prompt_*.metal, basalt_mmq.metal) for the GPU
tests. Every prompt kernel reads its rows from StepState at slot 15 (monolith/kernels.py's PROMPT_ROWS), so ``run_rows``
is basalt_gpu.run with a StepState record bound there. A source compiles as the emitter registers it (the StepState
struct spliced in after the prelude) with the dynamic-T program's row macros: STEP_STATE 1, T_SRC 0 and ROWS_MAX the
compiled bound B; the static form (T_SRC 2) is the contract and program tiers' concern."""

from types import SimpleNamespace

import numpy as np

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from monolith.runtime import _native as nt
from tests.contract.test_basalt_ops import spliced
from tests.kernels.basalt_gpu import device, pipeline


def prompt_pipeline(source, name, B, macros=()):
    """Kernel ``name`` of ``source`` as a dynamic-T program at bound ``B`` compiles it, with ``macros`` beside."""
    m = {"STEP_STATE": "1", "T_SRC": "0", "ROWS_MAX": f"{B}u", **dict(macros)}
    return pipeline(spliced(source), name, tuple(sorted(m.items())))


def run_rows(pso, buffers, params, slot, grid, threads, rows, position, fill=None, done=0):
    """basalt_gpu.run with a StepState record (the layout ``spliced`` gives the source; ``t_this_step`` = ``rows``,
    ``position``, ``done``) bound at slot 15 and returned last. ``grid`` (threadgroups) and ``threads`` are 3-tuples."""
    dev = device()
    state = StepStateLayout().pack({"t_this_step": rows, "position": position, "done": done})
    srcs = [*buffers, state]
    bufs = [nt.Buffer(dev, b) for b in srcs]
    for buf, src in zip(bufs, buffers):
        if fill is not None and not isinstance(src, (bytes, bytearray)):
            buf.fill(fill)
    d = nt.Dispatch().pipeline(pso)
    for k, b in enumerate(bufs[:-1]):
        d = d.buffer(k, b)
    d = d.buffer(15, bufs[-1])
    if params is not None:
        d = d.bytes(slot, params)
    r = nt.Queue(dev).run([d.grid(*grid).threadgroup(*threads)])
    assert not r.error, r.error
    return [b.read(0, len(src) if isinstance(src, (bytes, bytearray)) else src) for b, src in zip(bufs, srcs)]


def prompt_gemm(xb, wb, *, bf16, B, rows=None, slices=None, fill=0xFF, done=0):
    """basalt_prompt_gemm: ``Gemm::bf16``'s product (``bf16``) or ``Gemm::native``'s FP16 product of the 16-bit bits
    ``xb`` [B, K] and ``wb`` [N, K] (gemm.cu:122-132, :158-185), compiled at bound ``B`` and run with ``rows`` (default
    B) and ``done`` in StepState: one run_rows per slice ``(row0, nrows, col0, ncols)`` into one y of exactly B * N
    floats whose bytes start as ``fill``. ``slices`` default to the handler's rule, kernels.basalt_prompt_gemm_slices(B,
    N, K). Returns y's bits, uint32 [B, N]."""
    xb, wb = np.ascontiguousarray(xb, np.uint16), np.ascontiguousarray(wb, np.uint16)
    (nx, k), (n, kw) = xb.shape, wb.shape
    assert nx == B and kw == k, f"x {xb.shape} and w {wb.shape} at bound {B}"
    pso = prompt_pipeline(kernels.basalt_prompt_gemm_source(), "basalt_prompt_gemm", B, {"GEMM_BF16": str(int(bf16))})
    y = bytes([fill]) * (4 * B * n)
    for row0, nrows, col0, ncols in kernels.basalt_prompt_gemm_slices(B, n, k) if slices is None else slices:
        y = run_rows(pso, [xb.tobytes(), wb.tobytes(), y], kernels.basalt_prompt_gemm_params(n, k, row0, col0), 3,
                     (-(-ncols // 64), -(-nrows // 64), 1), (256, 1, 1), B if rows is None else rows, 0, done=done)[2]
    return np.frombuffer(y, np.uint32).reshape(B, n)


def convert(x, *, bf16, B, rows, fill=0xFF):
    """basalt_prompt_convert: ``to_bf16`` (``bf16``) or ``to_f16`` of the floats ``x`` [B, n] (kernels.cu:818-833),
    compiled at bound ``B`` and run with ``rows`` in StepState, one thread per element, into one y of exactly B * n
    halves whose bytes start as ``fill``. ``x``'s bits reach the kernel as they are (NaN payloads included). Returns
    y's bits, uint16 [B, n]."""
    x = np.ascontiguousarray(x)
    assert x.dtype == np.float32 and x.ndim == 2 and len(x) == B, f"x {x.dtype} {x.shape} at bound {B}"
    n = x.shape[1]
    pso = prompt_pipeline(kernels.basalt_prompt_gemm_source(), "basalt_prompt_convert", B, {"CV_BF16": str(int(bf16))})
    y = run_rows(pso, [x.tobytes(), 2 * B * n], kernels.basalt_prompt_convert_params(n), 2, (-(-B * n // 256), 1, 1),
                 (256, 1, 1), rows, 0, fill=fill)[1]
    return np.frombuffer(y, np.uint16).reshape(B, n)


def plan_words(plan):
    """basalt_emu.moe_plan's ``plan`` as the MMQ kernels' plan buffer: kernels.MMQ_PLAN_WORDS int32 words, n,
    rows_local, order[512], bounds[513], wpre[513] (kernels/common/basalt_mmq.metal), each array's words past its
    entries -1."""
    w = np.full(kernels.MMQ_PLAN_WORDS, -1, np.int32)
    w[:2] = plan.n, plan.rows_local
    for at, a in ((2, plan.order), (514, plan.bounds), (1027, plan.wpre)):
        w[at: at + len(a)] = a
    return w


def mmq_plan(ids, *, B, n_expert=512, cpu_mask=None, hist=None, done=0):
    """The plan kernels basalt_mmq_plan_count, _scan and _fill alone (basalt's host expert walk, prefill.cpp:1840-1932,
    :1993-1996): moe_mmq without x, its arguments and its slot, src, plan, hist and state."""
    return moe_mmq(None, ids, B=B, n_expert=n_expert, cpu_mask=cpu_mask, hist=hist, done=done)


def mmq_quantize(x, plan, *, B):
    """basalt_mmq_quantize (kernel 1, quantize_mmq_q8_1<D4>) of the float32 rows ``x`` [B, 2560] through ``plan``
    (basalt_emu.moe_plan's for its R tokens: slot p reads row src[p], slots p < bounds[n]), compiled at bound ``B`` with
    the gate/up product's macros and run with R rows in StepState, into an Xq of the 10 B slots' blocks whose bytes
    start as 0xFF. Returns Xq's bytes, uint8 [10 B, 20, 144]."""
    macros = kernels.basalt_mmq_macros("gate_up")
    k, slots = int(macros["K"]), B * plan.slot.shape[1]
    x = np.ascontiguousarray(x, np.float32)
    assert x.shape == (B, k), f"x {x.shape} at bound {B}"
    src = np.full(slots, -1, np.int32)
    src[: len(plan.src)] = plan.src
    pso = prompt_pipeline(kernels.basalt_mmq_source(), "basalt_mmq_quantize", B, macros)
    nbytes = slots * k // 128 * kernels.MMQ_BLOCK_BYTES
    xq = run_rows(pso, [x.tobytes(), src.tobytes(), nbytes, plan_words(plan).tobytes()], None, 0, (slots, k // 512, 1),
                  (kernels.MMQ_THREADS, 1, 1), len(plan.slot), 0, fill=0xFF)[2]
    return np.frombuffer(xq, np.uint8).reshape(slots, k // 128, kernels.MMQ_BLOCK_BYTES)


def mmq(xq, plan, buf, offsets, *, B, product="gate_up", sm_count=kernels.DENSE_TC_SM_COUNT):
    """basalt_mmq (kernel 2, mul_mat_q<Q2_0> with its stream-k fixup folded per output element, every launch of the
    plan in one dispatch) with ``product``'s macros, on the activation blocks ``xq`` [10 B, pad512(K) / 128, 144]
    (kernel 1's slots) through ``plan`` (basalt_emu.moe_plan's for its R tokens: n, order, bounds, wpre). Expert e's
    blob is at byte ``offsets[e]`` of ``buf``, one buffer made resident with Dispatch.use, and the kernel reads it
    through an I64 [512] table of GPU addresses (basalt_gpu.experts' grp_ptr; ids without a blob 0). Compiled at bound
    ``B`` with MMQ_SM_COUNT ``sm_count`` (basalt's 170 unless a test asks), run with R rows in StepState on
    kernels.basalt_mmq_grids' grid, into an output of the 10 B slots' LD floats whose bytes start as 0xFF. Returns its
    bits, uint32 [10 B, LD]."""
    macros = kernels.basalt_mmq_macros(product, sm_count=sm_count)
    k, ld, slots = int(macros["K"]), int(macros["LD"]), 10 * B
    xq = np.ascontiguousarray(xq, np.uint8)
    assert xq.shape == (slots, -(-k // 512) * 4, kernels.MMQ_BLOCK_BYTES), f"xq {xq.shape} at bound {B}"
    pso = prompt_pipeline(kernels.basalt_mmq_source(), "basalt_mmq", B, macros)
    dev = device()
    b_blobs = nt.Buffer(dev, np.ascontiguousarray(buf, np.uint8).tobytes())
    tab = np.zeros(512, np.uint64)
    tab[: len(offsets)] = b_blobs.gpu_address + np.asarray(offsets, np.uint64)
    bufs = [nt.Buffer(dev, b) for b in (xq.tobytes(), tab.tobytes(), plan_words(plan).tobytes(), 4 * slots * ld)]
    bufs[3].fill(0xFF)
    b_st = nt.Buffer(dev, StepStateLayout().pack({"t_this_step": len(plan.slot)}))
    d = nt.Dispatch().pipeline(pso)
    for slot, b in enumerate(bufs):
        d = d.buffer(slot, b)
    d = d.buffer(15, b_st).use(b_blobs).grid(*kernels.basalt_mmq_grids(B, product)["basalt_mmq"], 1)
    r = nt.Queue(dev).run([d.threadgroup(kernels.MMQ_THREADS, 1, 1)])
    assert not r.error, r.error
    return np.frombuffer(bufs[3].read(0, bufs[3].nbytes), np.uint32).reshape(slots, ld)


def mmq_swiglu_quantize(gu, plan, *, B):
    """basalt_mmq_swiglu_quantize (kernel 3, mmq::swiglu fused with quantize_mmq_q8_1<D4> on H) of the float32 GU rows
    ``gu`` [10 B, 1280] (slot p's gate outputs, then its up outputs) through ``plan`` (basalt_emu.moe_plan's for its R
    tokens: slots p < bounds[n]), compiled at bound ``B`` with the down product's macros (K = n_ff = 640) and run with R
    rows in StepState, into an Hq of the 10 B slots' pad512(640) / 128 = 8 blocks whose bytes start as 0xFF. Slot 1,
    which kernel 3 does not bind, holds a 4-byte placeholder. Returns Hq's bytes, uint8 [10 B, 8, 144]."""
    macros = kernels.basalt_mmq_macros("down")
    k, slots = int(macros["K"]), B * plan.slot.shape[1]
    kp = -(-k // 512) * 512
    gu = np.ascontiguousarray(gu, np.float32)
    assert gu.shape == (slots, 2 * k), f"gu {gu.shape} at bound {B}"
    pso = prompt_pipeline(kernels.basalt_mmq_source(), "basalt_mmq_swiglu_quantize", B, macros)
    nbytes = slots * kp // 128 * kernels.MMQ_BLOCK_BYTES
    hq = run_rows(pso, [gu.tobytes(), 4, plan_words(plan).tobytes(), nbytes], None, 0, (slots, kp // 512, 1),
                  (kernels.MMQ_THREADS, 1, 1), len(plan.slot), 0, fill=0xFF)[3]
    return np.frombuffer(hq, np.uint8).reshape(slots, kp // 128, kernels.MMQ_BLOCK_BYTES)


def moe_mmq(x, ids, buf=None, offsets=None, *, B, n_expert=512, cpu_mask=None, hist=None, prior=None, done=0):
    """One MoE half's routed experts through basalt's MMQ chain in one command buffer, compiled at bound ``B`` with
    N_EXPERT ``n_expert``, every dispatch on kernels.basalt_mmq_grids' grid and one StepState at slot 15 (R = len(ids)
    rows, ``done``): the plan kernels basalt_mmq_plan_count, _scan and _fill (basalt's host expert walk,
    prefill.cpp:1840-1932, :1993-1996) on the routed ids ``ids`` [R, 10], whose rows from R on hold 512 and -1, ids no
    kernel may read; then, unless ``x`` is None, kernels 1-4 (prefill.cpp:1990, :2241-2264): basalt_mmq_quantize of the
    float32 rows ``x`` [B, 2560] through the plan's src, basalt_mmq with the gate/up product's macros,
    basalt_mmq_swiglu_quantize and basalt_mmq with the down product's. Expert e's blob is at byte ``offsets[e]`` of
    ``buf``, one buffer made resident with Dispatch.use and read through an I64 [512] table of GPU addresses (ids past
    ``offsets`` 0). ``cpu_mask`` U32 [n_expert] (nonzero: basalt's on_cpu; default all zero); ``hist`` the mmq_hist
    scratch's first bytes (default 0xFF: every range a kernel must not read counts absent pairs); ``prior`` an earlier
    chunk's plan (basalt_emu.moe_plan's) whose slot, src and plan words those buffers hold before the step, else their
    bytes start as 0xFF, as Xq's, GU's, Hq's and D's do. Returns slot int32 [B, 10], src int32 [10 B], plan int32
    [MMQ_PLAN_WORDS], hist's bytes, StepState's fields (``state``) and, with ``x``, Xq uint8 [10 B, 20, 144], GU uint32
    [10 B, 1280], Hq uint8 [10 B, 8, 144] and D uint32 [10 B, 2560]."""
    return moe_mmq_program(buf, offsets, B=B, n_expert=n_expert, cpu_mask=cpu_mask, hist=hist, prior=prior,
                           experts=x is not None)(x, ids, done=done)


def moe_mmq_program(buf=None, offsets=None, *, B, n_expert=512, cpu_mask=None, hist=None, prior=None, experts=True):
    """moe_mmq's buffers and dispatches made once (its arguments; without ``experts`` the plan kernels alone), for one
    program over several chunks (spec-tests-moe.md 5.6's addition 3). Returns ``run(x, ids, done=0)``, which writes
    the routed ids ``ids`` [R, 10] (rows from R on 512 and -1), ``x`` and StepState (R rows, ``done``) into their
    buffers, fills Xq, GU, Hq and D with 0xFF, runs the same dispatches in one command buffer and returns moe_mmq's
    namespace: the row count and the ids reach the kernels through buffers, nothing of a chunk through the encode.
    slot, src, plan and mmq_hist keep what the previous run left, as a program's scratch does."""
    mask = np.zeros(n_expert, np.uint32) if cpu_mask is None else np.asarray(cpu_mask, np.uint32)
    assert mask.shape == (n_expert,), mask.shape
    mg, md = (kernels.basalt_mmq_macros(p, n_expert=n_expert) for p in ("gate_up", "down"))
    gg, gd = (kernels.basalt_mmq_grids(B, p) for p in ("gate_up", "down"))
    slot, src = np.full((B, 10), -1, np.int32), np.full(10 * B, -1, np.int32)
    plan = np.full(kernels.MMQ_PLAN_WORDS, -1, np.int32)
    if prior is not None:
        slot[:len(prior.slot)], src[:len(prior.src)], plan = prior.slot, prior.src, plan_words(prior)
    dev = device()
    b_ids = nt.Buffer(dev, slot.nbytes)
    b_mask, b_slot, b_src, b_plan = (nt.Buffer(dev, a.tobytes()) for a in (mask, slot, src, plan))
    b_hist = nt.Buffer(dev, 4 * gg["basalt_mmq_plan_count"][0] * (512 + 1) if hist is None else hist)
    if hist is None:
        b_hist.fill(0xFF)
    layout = StepStateLayout()
    b_st = nt.Buffer(dev, layout.size)

    def dispatch(name, macros, grid, binds, threads=kernels.MMQ_THREADS, use=None):
        d = nt.Dispatch().pipeline(prompt_pipeline(kernels.basalt_mmq_source(), name, B, macros))
        for s, b in binds:
            d = d.buffer(s, b)
        d = d.buffer(15, b_st).grid(*grid, 1).threadgroup(threads, 1, 1)
        return d if use is None else d.use(use)

    ds = [dispatch("basalt_mmq_plan_count", mg, gg["basalt_mmq_plan_count"], [(0, b_ids), (1, b_hist)]),
          dispatch("basalt_mmq_plan_scan", mg, gg["basalt_mmq_plan_scan"], [(1, b_hist), (2, b_plan), (5, b_mask)],
                   512),
          dispatch("basalt_mmq_plan_fill", mg, gg["basalt_mmq_plan_fill"],
                   [(0, b_ids), (1, b_hist), (3, b_slot), (4, b_src)], 10 * kernels.MMQ_RANGE)]
    outs, b = {}, {}
    if experts:
        slots, kd = 10 * B, int(md["K"])
        b_blobs = nt.Buffer(dev, np.ascontiguousarray(buf, np.uint8).tobytes())
        tab = np.zeros(512, np.uint64)
        tab[: len(offsets)] = b_blobs.gpu_address + np.asarray(offsets, np.uint64)
        b_x, b_tab = nt.Buffer(dev, 4 * B * int(mg["K"])), nt.Buffer(dev, tab.tobytes())
        outs = {"xq": (np.uint8, (slots, int(mg["K"]) // 128, kernels.MMQ_BLOCK_BYTES)),
                "gu": (np.uint32, (slots, int(mg["LD"]))),
                "hq": (np.uint8, (slots, -(-kd // 512) * 4, kernels.MMQ_BLOCK_BYTES)),
                "d": (np.uint32, (slots, int(md["LD"])))}
        b = {n: nt.Buffer(dev, int(np.prod(s)) * np.dtype(t).itemsize) for n, (t, s) in outs.items()}
        ds += [dispatch("basalt_mmq_quantize", mg, gg["basalt_mmq_quantize"],
                        [(0, b_x), (1, b_src), (2, b["xq"]), (3, b_plan)]),
               dispatch("basalt_mmq", mg, gg["basalt_mmq"], [(0, b["xq"]), (1, b_tab), (2, b_plan), (3, b["gu"])],
                        use=b_blobs),
               dispatch("basalt_mmq_swiglu_quantize", md, gd["basalt_mmq_swiglu_quantize"],
                        [(0, b["gu"]), (2, b_plan), (3, b["hq"])]),
               dispatch("basalt_mmq", md, gd["basalt_mmq"], [(0, b["hq"]), (1, b_tab), (2, b_plan), (3, b["d"])],
                        use=b_blobs)]
    i32 = lambda b: np.frombuffer(b.read(0, b.nbytes), np.int32)

    def run(x, ids, done=0):
        ids = np.asarray(ids, np.int32)
        R, k = ids.shape
        assert k == 10 and R <= B, f"ids {ids.shape} at bound {B}"
        assert (x is not None) == experts, "x is the experts' input, and only theirs"
        rows = np.tile(np.array([512, -1], np.int32), 5 * B).reshape(B, 10)
        rows[:R] = ids
        b_ids.write(rows.tobytes())
        b_st.write(layout.pack({"t_this_step": R, "done": done}))
        if experts:
            x = np.ascontiguousarray(x, np.float32)
            assert x.shape == (B, int(mg["K"])), f"x {x.shape} at bound {B}"
            b_x.write(x.tobytes())
            for o in b.values():
                o.fill(0xFF)
        r = nt.Queue(dev).run(ds)
        assert not r.error, r.error
        return SimpleNamespace(slot=i32(b_slot).reshape(B, 10), src=i32(b_src), plan=i32(b_plan),
                               hist=b_hist.read(0, b_hist.nbytes), state=layout.unpack(b_st.read(0, b_st.nbytes)),
                               **{n: np.frombuffer(b[n].read(0, b[n].nbytes), t).reshape(s)
                                  for n, (t, s) in outs.items()})

    return run
