# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dec_commit_parity.cpp:57-74, :147-153 and :196-245 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): make_gdn's draws, set_commit's record and
# bench's block sweep with its GB/s figure.
"""basalt's GDN commit (kernels/common/basalt_gdn_commit.metal's ``gdn_commit_layers``) on this GPU, for two choices of
the port: the threadgroup count and the cost of the IEEE fma guard. An opt-in script, never collected by pytest; run it
with the GPU to itself and without shader validation:

    python tools/bench/basalt_gdn_commit.py [--rounds 20] [--warm 2]

Each time is the GPU time (RunResult.gpu_ms) of one command buffer holding a session commit's 36 one-layer dispatches
at Flash-Next's GDN shapes (C 10240, h_v 48, the window's MT = 8 rows; each layer its own state and window buffers, one
record), in Queue.run's serial encoder, the test harness's form, which runs them one after another. The commit program
replays them with no barrier after the first (the contract test's CT.3), where they may overlap, so these times may
overstate what a low threadgroup count costs there. Every run starts from the same state, written from the host first
(the minima are the same without the reset, measured on an M3 Max): a commit replays rows into the state it reads, so
without it the heads drift from run to run and a decaying one reaches the slow path. A table's sides run once per
round, in an order reversed every other round, after --warm rounds; printed: each side's minimum and maximum over
--rounds rounds, in microseconds per command buffer.

- (a) The threadgroup count (plan-commit B5): make_gdn's fixture (dec_commit_parity.cpp:57-74 at MT = 8, seed 11 on
  numpy's generator) committed at bench's n_keep 1, 2, 3 and 5 from 32, 68, 136 and 272 threadgroups per dispatch
  (272: one per item, the shipped count; basalt runs 32 persistent blocks), with bench's GB/s at the minimum: the
  recurrent states read and written.
- (b) The guard (B12): n_keep 8 at 272 threadgroups, the window's v rows zero so that the state alone sets the path:
  the shipped build on the fixture's state against a bench-local copy whose kernel text has ``fma_ieee(`` replaced by
  ``fma(`` (the fast path's cost), and against a state of 0x1p-110 in every word, which puts all 65 ``fma_ieee`` calls
  per thread and row (32 kv, the delta, 32 updates) on the software binary64 path (the worst case). Asserted before
  the timing: the copy stores the shipped build's bits, and the small state is still below 2^-100 after the 8 rows.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith import kernels  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402

S, HK, HV, MT, NL = 128, 16, 48, 8, 36        # Flash-Next's GDN layers; MT: the contract's window rows (D18)
C = 2 * HK * S + HV * S                        # 10240 conv channels: q, k, then v
REC = S * HV * S                               # a layer's recurrent state; its C x 3 conv history follows
SF = REC + 3 * C
GB = 2.0 * NL * REC * 4 / 1e9                  # bench's figure (:226): the recurrent states read and written
TGS = (32, 68, 136, 272)                       # threadgroups per dispatch; 272 = 4 * HV + C / 128, one per item
SMALL = 2.0 ** -110                            # below fma_ieee's 2^-100: its software binary64 path


def record(n):
    """set_commit(commit, n, 100) (dec_commit_parity.cpp:147-153), as bench calls it: n, n - 1, then 100 + t for the
    kept rows t, else -1; the commit reads word 0."""
    return np.array([n, n - 1] + [100 + t if t < n else -1 for t in range(MT)], np.int32).tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=20, help="timed rounds per table row (bench's 20 per block count)")
    ap.add_argument("--warm", type=int, default=2, help="untimed rounds first")
    a = ap.parse_args()
    dev = nt.Device()
    q = nt.Queue(dev)
    rng = np.random.default_rng(11)
    draw = lambda n, sd: (rng.standard_normal(NL * n, dtype=np.float32) * np.float32(sd)).reshape(NL, n)
    # make_gdn's order (z, y and gamma, which follow, the commit never reads)
    st0, h, qkv = draw(SF, 0.1), draw(MT * C, 0.1), draw(MT * C, 1.0)
    gate, beta = draw(MT * HV, 0.3), draw(MT * HV, 0.3)
    state = [nt.Buffer(dev, SF * 4) for _ in range(NL)]
    rows = [[nt.Buffer(dev, x.tobytes()) for x in layer] for layer in zip(h, qkv, gate, beta)]
    commit, used = nt.Buffer(dev, (2 + MT) * 4), nt.Buffer(dev, 4)
    prm = kernels.basalt_gdn_commit_params(C, HV, MT)

    def layers(pso, tgs):
        """The commit's 36 one-layer dispatches (the op's record) from ``tgs`` threadgroups of 128 threads."""
        out = []
        for st, rw in zip(state, rows):
            d = nt.Dispatch().pipeline(pso).buffer(0, st)
            for k, b in enumerate(rw, 1):
                d = d.buffer(k, b)
            out.append(d.buffer(5, commit).buffer(6, used).bytes(7, prm).grid(tgs).threadgroup(128))
        return out

    def run(ds, image, n):
        """One command buffer of ``ds`` from the per-layer state bytes ``image``: its GPU ms (used must read n)."""
        for b, im in zip(state, image):
            b.write(im, 0)
        used.fill(0xFF)
        r = q.run(ds)
        assert not r.error, r.error
        got = int(np.frombuffer(used.read(0, 4), np.uint32)[0])
        assert got == n, f"the commit used {got} rows, not {n}"
        return r.gpu_ms

    def timed(sides, n):
        """{label: (dispatches, image)}: each side's (min, max) microseconds over the rounds, paired and alternating."""
        us = {k: [] for k in sides}
        for i in range(a.warm + a.rounds):
            for k in list(sides)[:: 1 if i % 2 == 0 else -1]:
                t = run(*sides[k], n)
                if i >= a.warm:
                    us[k].append(1000 * t)
        return {k: (min(v), max(v)) for k, v in us.items()}

    src = kernels.basalt_gdn_commit_source()
    shipped = nt.Pipeline(nt.Library(dev, src, {}), "gdn_commit_layers")
    normal = [x.tobytes() for x in st0]
    info = dev.info()
    print(f"basalt GDN commit on {info.name} ({info.gpu_cores} GPU cores): {NL} one-layer dispatches per command "
          f"buffer (serial encoder), C {C}, h_v {HV}, MT {MT}; microseconds per command buffer, min-max over "
          f"{a.rounds} alternating rounds after {a.warm} untimed\n")
    print("(a) threadgroups per dispatch (B5), make_gdn's fixture; GB/s at the min: the recurrent states read and "
          "written\n")
    print("| n_keep | " + " | ".join(f"{t} threadgroups" for t in TGS) + " |")
    print("|---:|" + "---:|" * len(TGS))
    for n in (1, 2, 3, 5):
        commit.write(record(n), 0)
        res = timed({t: (layers(shipped, t), normal) for t in TGS}, n)
        print(f"| {n} | " + " | ".join(f"{lo:.1f}-{hi:.1f} ({GB / (lo * 1e-6):.0f} GB/s)" for lo, hi in res.values())
              + " |")

    hz = h.reshape(NL, MT, C).copy()
    hz[:, :, 2 * HK * S:] = 0                                    # v, after q and k: the state alone sets the path
    for rw, x in zip(rows, hz):
        rw[0].write(x.tobytes(), 0)
    k = kernels.template("basalt_gdn_commit.metal")
    assert src.endswith(k) and k.count("fma_ieee(") == 3, "the commit's three fma_ieee sites (kv, delta, update) moved"
    plain = nt.Pipeline(nt.Library(dev, src[: -len(k)] + k.replace("fma_ieee(", "fma("), {}), "gdn_commit_layers")
    small = [np.full(SF, SMALL, np.float32).tobytes()] * NL
    commit.write(record(MT), 0)
    ship8, plain8 = layers(shipped, TGS[-1]), layers(plain, TGS[-1])
    snap = lambda: [b.read(0, SF * 4) for b in state]
    run(ship8, normal, MT)
    want = snap()
    run(plain8, normal, MT)
    assert snap() == want, "the fma copy stores other bits than the shipped build on the fixture's state"
    run(ship8, small, MT)
    assert all(np.abs(np.frombuffer(b.read(0, REC * 4), np.float32)).max() < 2.0 ** -100 for b in state), \
        "the small state reached 2^-100: its later rows left the slow path"
    res = timed({"fma": (plain8, normal), "fma_ieee": (ship8, normal), "small": (ship8, small)}, MT)
    base = res["fma_ieee"][0]
    print(f"\n(b) the guard (B12): n_keep {MT}, {TGS[-1]} threadgroups, the window's v rows zero\n")
    print("| build, state | us per command buffer | us per dispatch (min / 36) | min against fma_ieee, fixture state |")
    print("|---|---:|---:|---:|")
    for label, key in (("fma (bench-local copy), fixture state", "fma"),
                       ("fma_ieee (shipped), fixture state", "fma_ieee"),
                       ("fma_ieee (shipped), 0x1p-110 everywhere: every call on the slow path", "small")):
        lo, hi = res[key]
        print(f"| {label} | {lo:.1f}-{hi:.1f} | {lo / NL:.1f} | {lo - base:+.1f} ({lo / base:.2f}x) |")


if __name__ == "__main__":
    main()
