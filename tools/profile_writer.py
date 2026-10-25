#!/usr/bin/env python3
"""Write (or refresh) this machine's chip profile from the kernel harnesses — the autotuner at install time (design
§5.7, #49). It measures what the compiler reads from the selected backend's ``config.json`` / ``engine`` block:

* the pack's lane order (the T = 1 GEMV rate, interleaved16 vs contiguous) and the threadgroups per core (x1 vs x2),
* ``cost_T`` per format: the shader GEMV's pass cost at T = 1, 2, 4, 8 relative to T = 1, at the best geometry per T,
* ``accelerator_<fmt>``: the tensor-ops tile (gemm_tile) at 8 / 16 / 32 token rows in the same unit, each row in the
  geometry the engine's autotuner's rule picks among those that pass the oracle, which the autotuner itself does not
  check (one or two threadgroups per core, the K-splits ``TILE_KSPLITS`` and their cacheless twins; recorded as
  ``tile_mode``; a geometry that fails the oracle is recorded under ``tile_oracle_fail``), each timed with BF16 output
  as the engine compiles the tile and checked once more with 32-bit output under the strict rule, and from the two the
  accelerator switch and ``accelerator_min_t`` per format (the smallest T the tile takes); every run after the probes
  takes the lane order and scale placement the profile will hold,
* the attention kernel (v1, v2 and v3 over context lengths and T).

A kernel that fails the oracle (a tile or shader GEMV geometry, or a T = 1 probe) or a tile-step error other than a
refusal stops the writer: it lists each and writes no profile (exit 1). A chip without tensor ops runs ``--no-accelerator``.

The tile rows price the autotuner's pick. An autotuning session dispatches it, and serve does on the shapes the
configuration's kernel choices list; serve runs every other shape on the unsplit tile (the crew). A refresh that moves
``accelerator_min_t`` for a format those choices do not cover needs a serve A/B before it ships. The writer and the
autotuner time crew and crew2 at one and two threadgroups per core; compiler.emit dispatches them at the
configuration's ``threadgroups_per_core`` times that, so on a chip with two per core a crew or crew2 row prices a
geometry the engine does not run.

Everything else in a profile is the probes' record (``./probes/run_all.sh``): ``sibling_order`` (p11) and
``max_cb_ms`` (p6/p6b) carry over from the existing file, or take the safe defaults on a new chip. Min-of-N over
>= 2 GB streamed per point (``--quick`` for a short run). The decisions are ``monolith.backends.metal.calibration``'s.

    python tools/profile_writer.py --dry-run                 # measure and print the engine block
    python tools/profile_writer.py                           # write the selected backend config.json (merged into an existing one)
    python tools/profile_writer.py --nominal-gbps 307        # a new chip: the spec bandwidth (else a measured stand-in)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

import gemm_bench  # noqa: E402
import gemv_bench  # noqa: E402
import gqa_bench  # noqa: E402

from monolith.backends.metal import ChipConfig as Profile, config_path, config_for_device, get_backend  # noqa: E402
from monolith.backends.metal.calibration import (COST_TS, NOISE, TILE_KSPLITS, TILE_TMS, choose_lane_order,  # noqa: E402
                                                 choose_scale_placement, decide, merge_profile, profile_name, tile_rows)
from monolith.bench import RANDOM_FORMATS  # noqa: E402
from monolith.formats import FORMATS  # noqa: E402
from monolith.kernels import GemmRefused  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402

GEOMETRIES: Tuple[Tuple[str, Dict[str, Any]], ...] = (("crew x1", {"tg_per_core": 1}), ("crew x2", {"tg_per_core": 2}),
                                                      ("1blk/SG tg64", {"one_block_per_sg": True, "tg": 64}))
DEFAULT_FORMATS = ("nvfp4", "fp8_e4m3", "int4_affine", "bf16")
DEFAULT_ATTENTION = (32, 4, 256)            # heads, kv heads, head_dim (the target's)


def device_facts(info: Any, *, target_gb: float = 21.0, nominal_gbps: Optional[float] = None) -> Dict[str, Any]:
    """The top-level facts of the profile from the Metal device (and the machine)."""
    ws, mb = info.recommended_working_set / 2 ** 30, info.max_buffer_length / 2 ** 30
    mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    d = {"chip": info.name, "gpu_family": f"Apple{info.apple_family}", "gpu_cores": int(info.gpu_cores), "memory_gb": int(round(mem)),
         "os": f"macOS {platform.mac_ver()[0]}", "gpu_working_set_gb": round(ws, 2), "max_buffer_gb": round(mb, 2),
         "hosts_target_model": bool(ws >= target_gb),
         "hosts_target_model_note": f"{target_gb:g} GB of weights against recommendedMaxWorkingSetSize {ws:.2f} GB, maxBufferLength {mb:.2f} GB"}
    if nominal_gbps:
        d["nominal_gbps"] = float(nominal_gbps)
    return d


def int_list(s: str) -> List[int]:
    """An argparse type: comma-separated integers (a malformed list is a usage error)."""
    return [int(x) for x in s.split(",")]


def _failed_checks(bf16: Dict[str, Any], strict: Optional[Dict[str, Any]]) -> str:
    """The checks a tile geometry failed, each with the numbers its own rule reads: the timed run's BF16 output
    (OracleCheck.ok_rounded) and the untimed run's 32-bit output (OracleCheck.ok)."""
    failed = []
    if bf16["ok"] is False:
        failed.append(f"BF16 output: {bf16['max_ulp_at_scale']} ulp at scale, rel err {bf16['max_rel_err']:.1e}")
    if strict is not None and strict["ok"] is False:
        failed.append(f"32-bit output: {strict['max_ulp_at_rms']} ulp at RMS, rel err {strict['max_rel_err']:.1e}")
    return "; ".join(failed)


def measure(*, shape: Tuple[int, int] = (17408, 5120), formats: Sequence[str] = DEFAULT_FORMATS, ts: Sequence[int] = COST_TS,
            tms: Sequence[int] = TILE_TMS, copies: Optional[int] = None, reps: int = 3, accelerator: bool = True,
            attention: Optional[Tuple[int, int, int]] = DEFAULT_ATTENTION, ctxs: Sequence[int] = (1024, 4096), attn_ts: Sequence[int] = (1, 4),
            current_lane_order: Optional[str] = None, current_placement: Optional[str] = None,
            log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run the harnesses and return the measurement mapping ``monolith.backends.metal.calibration.decide`` reads.
    ``current_lane_order`` and ``current_placement`` are the profile's values, which a tie in their probes keeps."""
    b = gemv_bench.Bench()
    n, k = shape
    m: Dict[str, Any] = {"family": f"Apple{b.info.apple_family}", "chip": b.info.name, "shape": f"{n}x{k}", "copies": copies, "reps": reps,
                         "oracle_fail": []}                         # shader geometries and probes that fail the oracle: no profile
    formats = [f for f in dict.fromkeys(formats) if f in RANDOM_FORMATS]      # once each: a second sweep would reset the first's record
    ref = "fp8_e4m3" if "fp8_e4m3" in formats else formats[0]
    # 1. the lane order: the reference format at T = 1, crew x1
    m["lane_order_gbps"] = {}
    for order in ("interleaved16", "contiguous"):
        r = b.run(ref, n, k, t=1, lane_order=order, copies=copies, reps=reps)
        m["lane_order_gbps"][order] = r["gbps"]
        if r["ok"] is False:
            m["oracle_fail"].append(f"lane order {order} ({ref}, T = 1)")
        log(f"lane order {order:13s} {ref} T=1: {r['ms']:.3f} ms {r['gbps']:.0f} GB/s{'' if r['ok'] in (None, True) else '  ORACLE FAIL'}")
    lane = choose_lane_order(m["lane_order_gbps"], current_lane_order)[0]     # decide()'s pick: the packs serve builds
    m["lane_order_measured_with"] = lane
    # 1b. the scale placement: a format with block scales at T = 1, the scales inline in the unit or in the block's
    # region — on a K = 4096 stripe, where a lane's scales are one word (the packer keeps wider runs inline anyway)
    scaled = next((f for f in ("nvfp4", "int8", "int4_affine") if f in formats), None)
    m["scale_placement_gbps"] = {}
    if scaled is not None:
        for placement in ("inline", "block"):
            r = b.run(scaled, n, 4096, t=1, lane_order=lane, copies=copies, reps=reps, placement=placement)
            m["scale_placement_gbps"][placement] = r["gbps"]
            if r["ok"] is False:
                m["oracle_fail"].append(f"scale placement {placement} ({scaled}, T = 1)")
            log(f"scale placement {placement:7s} {scaled} T=1: {r['ms']:.3f} ms {r['gbps']:.0f} GB/s ({r['unit_bytes']}-byte units)"
                f"{'' if r['ok'] in (None, True) else '  ORACLE FAIL'}")
        m["scale_placement_measured_with"] = scaled
    placement = choose_scale_placement(m["scale_placement_gbps"], current_placement)[0]     # decide()'s pick: the packs serve builds
    m["scale_placement_timed"] = placement
    log(f"scale placement timed: {placement}")
    # 2. threadgroups per core at T = 1
    m["threadgroups_ms"] = {}
    for g in (1, 2):
        r = b.run(ref, n, k, t=1, lane_order=lane, placement=placement, tg_per_core=g, copies=copies, reps=reps)
        m["threadgroups_ms"][g] = r["ms"]
        if r["ok"] is False:
            m["oracle_fail"].append(f"threadgroups x{g} ({ref}, T = 1)")
        log(f"threadgroups x{g} {ref} T=1: {r['ms']:.3f} ms {r['gbps']:.0f} GB/s{'' if r['ok'] in (None, True) else '  ORACLE FAIL'}")
    # 3. the shader GEMV's cost per format and T at the best geometry
    m["shader_ms"], m["shader_geometry"], m["shader_gbps"] = {}, {}, {}
    for fmt in formats:
        m["shader_ms"][fmt], m["shader_geometry"][fmt], m["shader_gbps"][fmt] = {}, {}, {}
        for t in ts:
            best = None
            for name, geo in GEOMETRIES:
                try:
                    r = b.run(fmt, n, k, rows=16, t=t, lane_order=lane, placement=placement, copies=copies, reps=reps, **geo)
                except Exception as e:  # noqa: BLE001 — a geometry the kernel refuses (register budget) is simply not a candidate
                    if "shader validation:" in str(e):   # an out-of-bounds access is not a refusal
                        raise
                    log(f"shader {fmt} T={t} {name}: skipped ({e})")
                    continue
                if r["ok"] is False:
                    m["oracle_fail"].append(f"shader {fmt} T={t}: {name}")
                    log(f"shader {fmt} T={t} {name}: ORACLE FAIL (ulp {r['max_ulp_at_rms']}) — recorded; no profile is written")
                    continue
                if best is None or r["ms"] < best[0]:
                    best = (r["ms"], name, r["gbps"])
            if best is None:
                if any(f.startswith(f"shader {fmt} T={t}: ") for f in m["oracle_fail"]):
                    continue                                        # every geometry that ran is wrong: main() lists them
                raise RuntimeError(f"profile_writer: no geometry ran for {fmt} at T = {t}")
            m["shader_ms"][fmt][t], m["shader_geometry"][fmt][t], m["shader_gbps"][fmt][t] = best
            base = m["shader_ms"][fmt].get(ts[0])
            log(f"shader {fmt:12s} T={t}: {best[0]:.3f} ms ({best[1]}, {best[2]:.0f} GB/s)" + (f" = {best[0] / base:.2f}x T={ts[0]}" if base else ""))
    # 4. the tensor-ops tile in each geometry compiler.autotune.tune_gemm tries, and per TM its pick (the module docstring): a
    # refused crew2 or K-split is skipped; a refused crew or any other error ends the step (tile_error), tile_oracle_fail kept
    m["tile_ms"], m["tile_gbps"], m["tile_mode"], m["tile_oracle_fail"] = {}, {}, {}, {}
    if accelerator:
        ran = False                                                 # a tile variant ran: the tensor ops build on this chip
        try:
            gb = gemm_bench.GemmBench()
            for fmt in formats:
                m["tile_ms"][fmt], m["tile_gbps"][fmt], m["tile_mode"][fmt], m["tile_oracle_fail"][fmt] = {}, {}, {}, {}
                block_scaled = FORMATS.get(fmt).scale_group > 0
                for tm in tms:
                    ms, gbps = {}, {}                               # the geometries that pass the oracle

                    def run(mode, **kw):
                        nonlocal ran
                        r = None
                        try:
                            r = gb.run(fmt, n, k, tm=tm, lane_order=lane, placement=placement, copies=copies, reps=reps, out_bf16=True, **kw)
                            ran = True
                            strict = gb.run(fmt, n, k, tm=tm, lane_order=lane, placement=placement, copies=1, reps=1, **kw)   # untimed
                        except GemmRefused as e:                    # the kernel refuses this geometry for this format, TM and K
                            if mode == "crew":                      # the unsplit tile refused: the shape is not the tile's
                                raise GemmRefused(f"{fmt} TM={tm} crew: {e}") from e
                            log(f"tile {fmt} TM={tm} {mode}: skipped ({e})")
                            return None
                        except Exception as e:                      # names the geometry whose error ends the tile step
                            if r is not None and r["ok"] is False:  # the timed run's oracle failure stays recorded
                                m["tile_oracle_fail"][fmt].setdefault(tm, []).append(mode)
                                log(f"tile {fmt} TM={tm} {mode}: ORACLE FAIL ({_failed_checks(r, None)}) — recorded; no profile is written")
                            unbuilt = not ran and str(e).startswith(("MSL compile failed", "pipeline failed"))   # the first tile kernel
                            raise RuntimeError(f"{fmt} TM={tm} {mode}: {e}" + (" (a chip without tensor ops runs --no-accelerator)" if unbuilt else "")) from e
                        if r["ok"] is False or strict["ok"] is False:
                            m["tile_oracle_fail"][fmt].setdefault(tm, []).append(mode)
                            log(f"tile {fmt} TM={tm} {mode}: ORACLE FAIL ({_failed_checks(r, strict)}) — recorded; no profile is written")
                        else:
                            ms[mode], gbps[mode] = r["ms"], r["gbps"]
                        return r

                    run("crew")
                    run("crew2", tg_per_core=2)
                    for s in TILE_KSPLITS:
                        r = run(f"ksplit{s}", ksplit=s)
                        if block_scaled and (r is None or "SCALE_CACHE" in r["macros"]):     # a cacheless twin only where it differs
                            run(f"ksplit{s}nc", ksplit=s, scale_cache=False)
                    if "crew" not in ms:                            # a wrong unsplit tile gives this TM no row
                        continue
                    best = min(ms, key=ms.get)
                    mode = "crew" if ms[best] > ms["crew"] * (1 - NOISE) else best
                    m["tile_ms"][fmt][tm], m["tile_gbps"][fmt][tm], m["tile_mode"][fmt][tm] = ms[mode], gbps[mode], mode
                    shader1 = m["shader_ms"][fmt].get(1)
                    log(f"tile   {fmt:12s} TM={tm}: {ms[mode]:.3f} ms ({mode}, {gbps[mode]:.0f} GB/s)"
                        + (f" = {ms[mode] / shader1:.2f}x the T=1 shader pass" if shader1 else ""))
        except Exception as e:  # noqa: BLE001
            if "shader validation:" in str(e):
                raise
            # never an empty message: the stop check reads the key; a wrong geometry stays recorded in tile_oracle_fail
            m["tile_ms"], m["tile_gbps"], m["tile_mode"], m["tile_error"] = {}, {}, {}, str(e) or type(e).__name__
            m["tile_refused"] = isinstance(e, GemmRefused)                  # the unsplit tile refused the shape: no tile rows
            log(f"tile: no measurement ({e})")
    # 5. the attention kernels v1, v2 and v3 over the contexts and T
    if attention is not None:
        heads, kv, d = attention
        m["attention_ms"] = {"v1": {}, "v2": {}, "v3": {}}
        m["attention_rep"] = heads // kv
        for ctx in ctxs:
            for t in attn_ts:
                for kind in ("v1", "v2", "v3"):
                    try:
                        r = gqa_bench.run(b.dev, heads, kv, d, ctx, t, 64, min(4, (heads // kv) * t) if kind == "v2" else 4, reps,
                                          v2=kind == "v2", v2_tg=2, v3=kind == "v3")
                    except Exception as e:  # noqa: BLE001
                        if "shader validation:" in str(e):
                            raise
                        log(f"attention {kind} ctx={ctx} T={t}: skipped ({e})")
                        continue
                    m["attention_ms"][kind][(ctx, t)] = r["ms"]
                    log(f"attention {kind} ctx={ctx} T={t}: {r['ms']:.3f} ms ({r['kv_gbps']:.0f} GB/s of KV)")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="17408x5120", help="the GEMV shape measured (the target's largest projection)")
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    ap.add_argument("--ts", default=",".join(str(t) for t in COST_TS), type=int_list)
    ap.add_argument("--tms", default=",".join(str(t) for t in TILE_TMS), type=int_list)
    ap.add_argument("--copies", type=int, help="packs streamed per measurement (default: >= 2 GB)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--attention", default=",".join(str(x) for x in DEFAULT_ATTENTION), help="heads,kv,head_dim or 'none'")
    ap.add_argument("--ctx", default="1024,4096", type=int_list)
    ap.add_argument("--attn-ts", default="1,4", type=int_list)
    ap.add_argument("--no-accelerator", action="store_true", help="skip the tile: the profile's accelerator is off, and so is every engine "
                    "path gated on it (the T = 1 rows GEMVs and MMA attention among them)")
    ap.add_argument("--quick", action="store_true", help="a short run: 4 packs per point, 2 reps, the first --tms value only, one context")
    ap.add_argument("--nominal-gbps", type=float, help="the chip's spec bandwidth (kept from an existing profile; else a measured stand-in)")
    ap.add_argument("--target-gb", type=float, default=21.0, help="the target model's resident weight bytes for hosts_target_model")
    ap.add_argument("--name", help="the profile name (default <chip>-<cores>c)")
    ap.add_argument("--out", type=Path, help="write here instead of the registered backend configuration")
    ap.add_argument("--dry-run", action="store_true", help="measure and print; write nothing")
    a = ap.parse_args(argv)
    n, k = (int(x) for x in a.shape.lower().split("x"))
    formats = [f for f in a.formats.split(",") if f]
    ts, tms, ctxs, attn_ts = a.ts, a.tms, a.ctx, a.attn_ts
    if not formats or set(formats) - set(RANDOM_FORMATS):
        ap.error(f"--formats takes names from {', '.join(RANDOM_FORMATS)}: the formats the harnesses build")
    if 1 not in ts or min(ts) < 1:
        ap.error("--ts takes token counts of 1 or more and must include 1: the costs are relative to the T = 1 pass")
    if set(tms) - set(TILE_TMS):
        ap.error(f"--tms takes {', '.join(map(str, TILE_TMS))}: the tile's token rows")
    attention = None if a.attention.strip().lower() == "none" else tuple(int(x) for x in a.attention.split(","))
    copies, reps = a.copies, a.reps
    if a.quick:
        copies, reps, tms, ctxs = (copies or 4), min(reps, 2), tms[:1], ctxs[:1]
    compared = [t for t in ts if 2 <= t <= max(TILE_TMS)]            # the T the tile is compared at, each against its tile_rows(T) row
    missing = sorted({tile_rows(t) for t in compared} - set(tms))
    if not a.no_accelerator and (not compared or missing):
        ap.error(f"the tile is compared at each --ts value from 2 to {max(TILE_TMS)} against its tile row: "
                 + (f"--tms lacks {', '.join(map(str, missing))} (--quick keeps the first --tms value)" if compared else "--ts has none"))
    info = nt.Device().info()
    name = a.name or profile_name(info.name, info.gpu_cores)
    try:
        src = config_path(name)
    except ValueError:
        if a.out is None and not a.dry_run:
            ap.error("an unregistered chip needs --out; register its backend after calibration")
        src = a.out
    path = a.out or src
    existing = json.loads(src.read_text()) if src is not None and src.exists() else None
    if existing:
        get_backend(existing.get("backend", "common")).validate_device(Profile.from_dict(name, existing), info)
    print(f"{info.name}: {info.gpu_cores} cores, Apple{info.apple_family}; profile {name} ({'merging ' + str(src) if existing else 'new'}) -> {path}", flush=True)
    selected = config_for_device(info.gpu_cores, info.apple_family, info.name)
    t0 = time.time()
    m = measure(shape=(n, k), formats=formats, ts=ts, tms=tms, copies=copies, reps=reps, accelerator=not a.no_accelerator,
                attention=attention, ctxs=ctxs, attn_ts=attn_ts, current_lane_order=(existing or {}).get("engine", {}).get("lane_order"),
                current_placement=(existing or {}).get("engine", {}).get("scale_placement"), log=lambda s: print(s, flush=True))
    fails = [f"tile {fmt} TM={tm}: {', '.join(modes)}" for fmt, rows in m["tile_oracle_fail"].items() for tm, modes in sorted(rows.items())]
    problems = [f"fails the oracle: {f}" for f in fails + m["oracle_fail"]]
    if "tile_error" in m and not m["tile_refused"]:
        problems.append(f"the tile step failed: {m['tile_error']}")
    if problems:                                                    # a wrong or failing kernel: no profile until it is fixed
        print("\nno profile written:", *(f"  {p}" for p in problems), sep="\n")
        return 1
    engine, notes = decide(m, existing.get("engine") if existing else None)
    device = device_facts(info, target_gb=a.target_gb, nominal_gbps=a.nominal_gbps)
    doc = merge_profile(existing, device=device, engine=engine, measurements=m, notes=notes, written=time.strftime("%Y-%m-%d"),
                        command="python " + " ".join(sys.argv))
    doc.setdefault("name", name)
    doc.setdefault("backend", selected.backend if selected else "common")
    doc.setdefault("validation", "unmeasured")
    Profile.from_dict(name, doc)                                    # the loader's validation before anything is written
    print(f"\nmeasured in {time.time() - t0:.0f} s\nengine: {json.dumps(engine, indent=2)}\ndecisions:")
    for key, why in notes.items():
        print(f"  {key}: {why}")
    if a.dry_run:
        return 0
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
