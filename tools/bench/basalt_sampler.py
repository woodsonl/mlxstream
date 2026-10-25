# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/sampler_parity.cpp:512-571 (bench_sampled) and :654-690 (the
# coupled draft's per-call time) @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""basalt's sampler on this GPU, per call: sampler_parity's ``--bench`` timings and the cost of the dispatches a
greedy request gates off. An opt-in script, never collected by pytest; run it with the GPU to itself and without
shader validation:

    python tools/bench/basalt_sampler.py [--iters 50] [--coupled-iters 200] [--pairs 40] [--steps 20]

- B.1, bench_sampled: the sampled path at basalt's vocabulary (248,320 logits of normal(0, 3)) for T = 1, 4, 8 and
  top_k = 20, 64 (top_p 0.95, temperature 0.7, seed 1): a ``basalt_sample`` program emitted static at T, mode 2 (the
  split path), 3 warm-up replays, then ``--iters`` replays with the counter base set to the call's index, one replay
  per command buffer; then the same with the probe (flags PROBE, every query 0, nq = T). Printed: the GPU time per
  call (StepReport.gpu_ms), its mean, as basalt prints it, and its minimum.
- B.2, the coupled draft: bs_coupled_penalize, bs_coupled_part and bs_coupled_merge on one draft row of 106,299
  logits (top_k 20, temperature 0.6, seed 5, the window of the ring's last 64 slots at repeat 1.1, step record 9;
  coupled_parity's subset maps, cap 256), 5 warm-up calls, then ``--coupled-iters`` calls, one command buffer each.
- The gated dispatches' cost in mode 0 (greedy): the program's five dispatches, of which the one-block greedy kernel
  and the split pair return at their first instruction, against the ``greedy_only`` build (the cluster pair alone), at
  T = 1 and 8, in ``--pairs`` alternating pairs of ``--steps`` replays in one command buffer. Printed per step: each
  side's minimum, their difference, and each side's spread (largest less smallest).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith import kernels  # noqa: E402
from monolith.compiler import emit_program  # noqa: E402
from monolith.core import BlockDomain, DType, Graph, OpClass, T  # noqa: E402
from monolith.core.profile import Profile  # noqa: E402
from monolith.core.step_state import StepStateLayout  # noqa: E402
from monolith.runtime import Engine  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402

V = 248320                                     # basalt's vocabulary (sampler_parity.cpp:514)
M = 0xFFFFFFFF
# The handler takes no value from the profile (fixed geometries); the emitter needs one for its family.
PROF = Profile.from_dict("bench", {"gpu_cores": 40, "nominal_gbps": 400.0, "engine": {
    "family": "Apple9", "lane_order": "interleaved16", "threadgroups_per_core": 2}})
CLUSTER = ("bs_greedy_partial", "bs_greedy_merge")


def program(t, greedy_only=False):
    """``basalt_sample`` over (T, V) logits, no bias or history, emitted static at T = t (``greedy_only``: the
    greedy-only build); StepState's ``pending_tokens`` holds t + 1 entries, so the probe queries every row."""
    g = Graph()
    ins = [g.input("logits", (T, V), DType.F32), g.input("ctr_base", (2,), DType.U32), g.input("nq", (1,), DType.U32)]
    token, qprob = g.value("token", (T,), DType.I32), g.value("qprob", (T,), DType.F32)
    g.op("basalt_sample", ins, [token, qprob], domain=BlockDomain("span", 1), klass=OpClass.REDUCE, hist_len=0,
         **(dict(greedy_only=True) if greedy_only else {}))
    return emit_program(g, pack=[], profile=PROF, t=t, layout=StepStateLayout(t_max=max(8, t + 1)), tail=None,
                        token=token)


def functions(prog):
    return [prog.kernels[o.kernel].function for o in prog.ops]


def engine(prog, dev, cache, logits, t, *, mode, probe=False, top_k=20, top_p=0.95, temperature=0.7, seed=1):
    """The program instantiated with ``logits`` and the request: the op's record repacked in ``mode`` (the build's
    structure kept), nq = t with the probe, the seed, the rows and the probe's queries (all 0) in StepState."""
    eng = Engine(prog, dev, pipeline_cache=cache)
    eng.buffers["logits"].write(np.ascontiguousarray(logits, np.float32).tobytes(), 0)
    o = next(o for o in prog.ops if prog.kernels[o.kernel].function == "bs_greedy_partial")
    rec = next(name for index, name, _ in o.bindings if index == 7)
    vocab, rows, n_blocks, hist_len = struct.unpack_from("<IIII", prog.buffers[rec].init)
    eng.buffers[rec].write(kernels.basalt_sampler_params(vocab, rows, n_blocks, hist_len, mode=mode, probe=probe,
                                                         top_k=top_k, top_p=top_p, temperature=temperature), 0)
    if "nq" in eng.buffers:                                       # a greedy-only program binds no nq
        eng.buffers["nq"].write(struct.pack("<I", t if probe else 0), 0)
    st = eng.state()
    st.update(t_this_step=t, rng_lo=seed & M, rng_hi=seed >> 32 & M, pending_tokens=[0] * (t + 1))
    eng.buffers[prog.step_state].write(prog.layout.pack(st), 0)
    return eng


def per_call(eng, iters, warm=3):
    """GPU ms of each of ``iters`` single-replay command buffers, the counter base set to the call's index first
    (basalt's ``p.counter = it``), after ``warm`` untimed calls."""
    ms = []
    for it in list(range(warm)) + list(range(iters)):
        eng.buffers["ctr_base"].write(struct.pack("<II", it & M, it >> 32 & M), 0)
        ms.append(eng.run(1, steps_per_cb=1, in_flight=1).gpu_ms)
    return ms[warm:]


def us(ms):
    return f"mean {1000 * float(np.mean(ms)):8.1f} us, min {1000 * float(np.min(ms)):8.1f} us"


def bench_sampled(dev, cache, iters):
    rng = np.random.default_rng(20)
    for t in (1, 4, 8):
        l = (rng.standard_normal((t, V)) * 3).astype(np.float32)
        prog = program(t)
        assert functions(prog) == list(CLUSTER) + ["bs_greedy_one_block", "bs_split_part", "bs_split_merge"]
        for k in (20, 64):
            plain = per_call(engine(prog, dev, cache, l, t, mode=2, top_k=k), iters)
            probed = per_call(engine(prog, dev, cache, l, t, mode=2, probe=True, top_k=k), iters)
            print(f"  B.1: n_vocab {V}, rows {t}, top_k {k:2d}, top_p 0.95: {us(plain)} per call; "
                  f"with the probe: {us(probed)}")


def bench_coupled(dev, iters):
    nv, cap, max_t = 106299, 256, 8
    layout = StepStateLayout(t_max=8)
    src = kernels.basalt_sampler_source().replace(kernels.PRELUDE, kernels.PRELUDE + layout.to_msl() + "\n", 1)
    lib = nt.Library(dev, src, {"STEP_STATE": "1"})
    sub = np.arange(nv, dtype=np.int32) + 25                      # coupled_parity's subset: slot i is token i + 25
    inv = np.full(nv + 50, -1, np.int32)
    inv[sub] = np.arange(nv, dtype=np.int32)
    ring = np.full(cap + max_t, -1, np.int32)
    ring[cap - 64:cap] = sub[np.arange(cap - 64, cap) % 40]
    nb = -(-nv // kernels.BS_BLOCK)
    image = kernels.basalt_coupled_params(top_k=20, temperature=0.6, seed=5, penalty_last_n=64, penalty_repeat=1.1)
    bufs = {0: (np.random.default_rng(77).standard_normal(nv) * 3).astype(np.float32).tobytes(), 5: bytes(4),
            6: bytes(4), 7: kernels.basalt_coupled_node(nv, nv + 50, cap, 0, id_to_sub=True, sub_to_id=True),
            9: bytes(nb * kernels.BS_SEL_MAX * 8), 11: sub.tobytes(), 12: inv.tobytes(), 13: image,
            14: ring.tobytes(), 15: layout.pack({}), 16: bytes(16), 17: bytes(16), 18: np.int32(9).tobytes()}
    slots = {k: nt.Buffer(dev, b) for k, b in bufs.items()}
    ds = []
    for name, grid, tg in (("bs_coupled_penalize", 1, 1024), ("bs_coupled_part", nb, 512), ("bs_coupled_merge", 1, 32)):
        d = nt.Dispatch().pipeline(nt.Pipeline(lib, name))
        for k, b in slots.items():
            d = d.buffer(k, b)
        ds.append(d.grid(grid).threadgroup(tg))
    q = nt.Queue(dev)
    ms = []
    for i in range(5 + iters):
        r = q.run(ds)
        assert not r.error, r.error
        ms.append(r.gpu_ms)
    print(f"  B.2: coupled sample nv {nv}: {us(ms[5:])} per call (penalize + part + merge, "
          f"{nv * 4 / 1024:.0f} KB read)")


def bench_gated(dev, cache, pairs, steps):
    rng = np.random.default_rng(5)
    for t in (1, 8):
        l = (rng.standard_normal((t, V)) * 3).astype(np.float32)
        full, cut = program(t), program(t, greedy_only=True)
        assert functions(cut) == list(CLUSTER)
        sides = {"all five": engine(full, dev, cache, l, t, mode=0),
                 "cluster pair": engine(cut, dev, cache, l, t, mode=0)}
        times = {name: [] for name in sides}
        for e in sides.values():
            e.run(steps, steps_per_cb=steps, in_flight=1)                  # warm-up
        for i in range(pairs):
            order = list(sides) if i % 2 == 0 else list(sides)[::-1]     # alternate which side runs first
            for name in order:
                times[name].append(sides[name].run(steps, steps_per_cb=steps, in_flight=1).gpu_ms / steps)
        lo = {name: 1000 * min(v) for name, v in times.items()}
        spread = {name: 1000 * (max(v) - min(v)) for name, v in times.items()}
        print(f"  mode 0, rows {t}: all five dispatches {lo['all five']:7.1f} us per step (spread "
              f"{spread['all five']:.1f}), greedy_only (the cluster pair) {lo['cluster pair']:7.1f} us (spread "
              f"{spread['cluster pair']:.1f}); the gated three cost {lo['all five'] - lo['cluster pair']:+.1f} us")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=50, help="B.1's timed calls per case (basalt's 50)")
    ap.add_argument("--coupled-iters", type=int, default=200, help="B.2's timed calls (basalt's 200)")
    ap.add_argument("--pairs", type=int, default=40, help="alternating pairs of the mode-0 comparison")
    ap.add_argument("--steps", type=int, default=20, help="replays per command buffer in the mode-0 comparison")
    a = ap.parse_args()
    dev, cache = nt.Device(), {}
    info = dev.info()
    print(f"basalt sampler bench on {info.name} ({info.gpu_cores} GPU cores)")
    bench_sampled(dev, cache, a.iters)
    bench_coupled(dev, a.coupled_iters)
    bench_gated(dev, cache, a.pairs, a.steps)


if __name__ == "__main__":
    main()
