"""The profile writer's decisions (monolith.core.profile_writer, #49): pure functions over harness numbers, tested
here without a GPU; tools/profile_writer.py runs the harnesses and hands them these numbers."""

import json
from types import SimpleNamespace

import pytest

from monolith import kernels
from monolith.backends.metal.calibration import NOISE, TILE_KSPLITS
from monolith.bench import RANDOM_FORMATS
from monolith.core.profile import Profile, load_profiles
from monolith.core.profile_writer import (NEVER, accelerator_plan, attention_choice, choose_lane_order, choose_threadgroups, cost_table, decide,
                                          merge_profile, profile_name, tile_rows)


def test_profile_name_and_tile_rows():
    assert profile_name("Apple M5 Pro", 20) == "apple-m5-pro-20c" and profile_name("Apple M3 Pro", 18) == "apple-m3-pro-18c"
    assert [tile_rows(t) for t in (2, 8, 9, 16, 17, 32)] == [8, 8, 16, 16, 32, 32]
    with pytest.raises(ValueError):
        tile_rows(33)


def test_lane_order_and_threadgroups_follow_the_noise_rule():
    assert choose_lane_order({"interleaved16": 275, "contiguous": 238})[0] == "interleaved16"      # the M5 Pro (p13)
    assert choose_lane_order({"contiguous": 134, "interleaved16": 132}, "contiguous")[0] == "contiguous"   # the M3 Pro's tie keeps its value
    assert choose_lane_order({"contiguous": 134, "interleaved16": 132})[0] == "interleaved16"      # a tie on a new chip: the default
    assert choose_lane_order({"contiguous": 150, "interleaved16": 132})[0] == "contiguous"
    assert choose_threadgroups({1: 1.0, 2: 0.99})[0] == 1 and choose_threadgroups({1: 1.0, 2: 0.9})[0] == 2 and choose_threadgroups({1: 1.0, 2: 1.2})[0] == 1
    assert choose_threadgroups({1: 1.0, 2: 0.99}, current=2)[0] == 2
    with pytest.raises(ValueError):
        choose_lane_order({})


def test_cost_table_is_relative_to_t1():
    assert cost_table({1: 2.0, 2: 2.2, 4: 2.4, 8: 7.2}) == {1: 1.0, 2: 1.1, 4: 1.2, 8: 3.6}
    with pytest.raises(ValueError):
        cost_table({2: 1.0})


def test_accelerator_plan_reproduces_the_m5_pro_decision():
    m5 = load_profiles()["apple-m5-pro-20c"]
    shader = {"fp8": m5.cost_t["fp8"], "nvfp4": m5.cost_t["nvfp4"]}
    tile = {"fp8": m5.cost_t["accelerator_fp8"], "nvfp4": m5.cost_t["accelerator_nvfp4"]}
    on, min_t, why = accelerator_plan(shader, tile)
    assert on == "on" and min_t == {"fp8": 2, "nvfp4": 2} and "fp8: the tile from T = 2" in why      # FP8's 1.09 vs 1.08 is a tie: the tile
    # a format the tile loses to at every compared T takes NEVER, its note listing those T; a chip without the tile is off
    never = accelerator_plan({"bf16": {1: 1.0, 2: 1.1, 4: 1.3, 8: 3.0}}, {"bf16": {8: 4.0, 16: 4.1, 32: 5.0}})
    assert never[:2] == ("off", {"bf16": NEVER}) and never[2].startswith("bf16: the tile loses at T = 2, 4, 8 (")
    # a T whose tile row was not measured is not compared: T = 16 here (no TM 16 row); with no row to compare, no claim
    assert accelerator_plan({"bf16": {1: 1.0, 2: 1.1, 4: 1.3, 8: 3.0, 16: 6.0}}, {"bf16": {8: 4.0}})[2].startswith("bf16: the tile loses at T = 2, 4, 8 (")
    assert accelerator_plan({"bf16": {1: 1.0, 2: 1.1}}, {"bf16": {16: 1.0}})[2].startswith("bf16: no measured T has a tile row (")
    assert accelerator_plan({"bf16": {1: 1.0, 2: 1.1, 4: 1.3, 8: 3.0}}, {"bf16": {8: 1.3, 16: 1.5, 32: 2.5}})[:2] == ("on", {"bf16": 4})   # 1.3 > 1.1 × 1.03, ≤ 1.3 × 1.03
    assert accelerator_plan(shader, {}) == ("off", {}, "no tile measurement (the tile step was skipped)")


def test_attention_choice():
    assert attention_choice({(1024, 1): 0.10, (4096, 4): 0.30}, {(1024, 1): 0.08, (4096, 4): 0.25})[0] == "v2"
    assert attention_choice({(1024, 1): 0.10, (4096, 4): 0.30}, {(1024, 1): 0.08, (4096, 4): 0.30})[0] == "v1"    # not everywhere
    assert attention_choice({(1024, 1): 0.10}, {})[0] == "v1"
    # v3: auto (= v3) when it beats v1 and v2 at every point measured; otherwise the v1 / v2 rule
    v1m, v2m = {(1024, 1): 0.10, (4096, 1): 0.30, (4096, 4): 0.60}, {(1024, 1): 0.08, (4096, 1): 0.25, (4096, 4): 0.50}
    assert attention_choice(v1m, v2m, v3={(1024, 1): 0.05, (4096, 1): 0.20, (4096, 4): 0.40})[0] == "auto"
    assert attention_choice(v1m, v2m, v3={(1024, 1): 0.05, (4096, 1): 0.29})[0] == "v2"                     # not faster than v2 at (4096, 1)
    assert attention_choice(v1m, {**v2m, (4096, 4): 0.61}, v3={(1024, 1): 0.05, (4096, 1): 0.20, (4096, 4): 0.40})[0] == "auto"  # v3 wins everywhere; v2's loss is moot
    assert attention_choice(v1m, {**v2m, (4096, 4): 0.61}, v3={(1024, 1): 0.05, (4096, 1): 0.31, (4096, 4): 0.40})[0] == "v1"    # v3 loses a point, and so does v2: v1
    assert attention_choice(v1m, v2m, v3={(8192, 8): 0.1})[0] == "v2"                                       # no common point
    assert attention_choice(v1m, v2m, v3={(1024, 1): 0.05, (4096, 1): 0.20})[0] == "v2"                     # a v1 point v3 could not run: not auto


def _measurements():
    return {"family": "Apple10", "chip": "Apple M5 Pro", "shape": "17408x5120",
            "lane_order_gbps": {"interleaved16": 275.0, "contiguous": 238.0}, "threadgroups_ms": {1: 0.324, 2: 0.330},
            "shader_ms": {"fp8_e4m3": {1: 0.324, 2: 0.350, 4: 0.360, 8: 1.166}, "nvfp4": {1: 0.196, 2: 0.251, 4: 0.351, 8: 1.039}},
            "tile_ms": {"fp8_e4m3": {8: 0.353, 16: 0.376, 32: 0.680}, "nvfp4": {8: 0.202, 16: 0.204, 32: 0.455}},
            "attention_ms": {"v1": {(1024, 1): 0.05, (4096, 4): 0.2}, "v2": {(1024, 1): 0.06, (4096, 4): 0.21}}}


def test_leaf_writer_preserves_layer_fusion_selection():
    cfg = load_profiles()["apple-m5-max-40c"].gdn_mixer_fusion
    engine, notes = decide(_measurements(), {"gdn_mixer_fusion": cfg})
    assert engine["gdn_mixer_fusion"] == cfg
    assert "layer study" in notes["gdn_mixer_fusion"]


def test_decide_and_merge_produce_a_loadable_profile(tmp_path):
    engine, notes = decide(_measurements(), {"sibling_order": "alu_first", "max_cb_ms": 16, "lane_order": "contiguous", "attention_rows": 8})
    assert engine["attention_rows"] == 8                                                  # not measured: the file's value carries over
    assert decide(_measurements(), {"sibling_order": "alu_first"})[0]["attention_rows"] == 4
    assert engine["family"] == "Apple10" and engine["lane_order"] == "interleaved16" and engine["threadgroups_per_core"] == 1
    assert engine["sibling_order"] == "alu_first" and engine["max_cb_ms"] == 16 and engine["attention"] == "v1"
    assert engine["accelerator"] == "on" and engine["accelerator_min_t"] == {"fp8": 2, "nvfp4": 2}
    assert engine["cost_T"]["fp8"] == {"1": 1.0, "2": 1.08, "4": 1.111, "8": 3.599} and engine["cost_T"]["nvfp4"]["4"] == pytest.approx(1.791)
    assert engine["cost_T"]["accelerator_fp8"] == {"8": 1.09, "16": 1.16, "32": 2.099} and set(notes) == {"lane_order", "scale_placement", "threadgroups_per_core", "accelerator", "attention"}
    assert engine["scale_placement"] == "inline"                                          # no placement measurement: the default stays
    existing = {"chip": "Apple M5 Pro", "gpu_cores": 20, "nominal_gbps": 307.0, "measured": "2026-09-22", "streaming": {"probe": "p5b"},
                "engine": {"family": "Apple10", "lane_order": "contiguous", "sibling_order": "alu_first"}}
    device = {"chip": "Apple M5 Pro", "gpu_family": "Apple10", "gpu_cores": 20, "memory_gb": 24, "os": "macOS 26.5.1", "gpu_working_set_gb": 19.07,
              "max_buffer_gb": 14.3, "hosts_target_model": False, "hosts_target_model_note": "21 GB against 19.07"}
    doc = merge_profile(existing, device=device, engine=engine, measurements=_measurements(), notes=notes, written="2026-09-25", command="python tools/profile_writer.py")
    assert doc["streaming"] == {"probe": "p5b"} and doc["measured"] == "2026-09-22" and doc["nominal_gbps"] == 307.0 and "nominal_note" not in doc
    assert doc["writer"]["previous_engine"]["lane_order"] == "contiguous" and doc["writer"]["measurements"]["attention_ms"]["v1"]["(1024, 1)"] == 0.05
    p = Profile.from_dict("apple-m5-pro-20c", json.loads(json.dumps(doc)))
    assert p.lane_order == "interleaved16" and p.accelerator == "on" and p.cost("accelerator_nvfp4", 16) == pytest.approx(1.041) and p.cost("fp8", 3) == pytest.approx((1.08 + 1.111) / 2)
    # a new chip without a spec figure: the measured stand-in, flagged
    fresh = merge_profile(None, device=device, engine=engine, measurements=_measurements(), notes=notes, written="2026-09-25", command="x")
    assert fresh["nominal_gbps"] == 280.0 and "stand-in" in fresh["nominal_note"] and fresh["measured"] == "2026-09-25"
    assert Profile.from_dict("new", fresh).threadgroups_per_core == 1
    # no tile measurement (Apple9): the shader path, an empty min_t
    no_tile = dict(_measurements(), tile_ms={}, attention_ms={})
    eng9, notes9 = decide(dict(no_tile, family="Apple9"), None)
    assert eng9["accelerator"] == "off" and eng9["accelerator_min_t"] == {} and eng9["attention"] == "v1" and eng9["sibling_order"] == "either"
    assert notes9["accelerator"] == "no tile measurement (the tile step was skipped)" and "accelerator_fp8" not in eng9["cost_T"]


class _Bench:                                                               # the shader GEMV harness: 0.5 ms per token row, every check passes
    info = SimpleNamespace(apple_family=10, name="fake")

    def run(self, fmt, n, k, t=1, **kw):
        return {"ms": 0.5 * t, "gbps": 100.0, "ok": True, "unit_bytes": 16}


class _Stub:                                                                # nt.Buffer / Library / Pipeline / Dispatch: every call chains
    max_threads_per_threadgroup = 1024

    def __init__(self, *a, **kw):
        pass

    def __getattr__(self, name):
        return lambda *a, **kw: self


def _mode(ksplit=1, tg_per_core=1, scale_cache=None):
    """A tile geometry's name, as the writer's tile_mode and the autotuner's variants give it."""
    return ("crew2" if tg_per_core == 2 else "crew") if ksplit == 1 else f"ksplit{ksplit}" + ("nc" if scale_cache is False else "")


def _fake_device(monkeypatch, pw):
    """main()'s device without Metal: a 20-core M5 Pro."""
    info = SimpleNamespace(name="Apple M5 Pro", gpu_cores=20, apple_family=10, recommended_working_set=19 * 2 ** 30, max_buffer_length=14 * 2 ** 30)
    monkeypatch.setattr(pw, "nt", SimpleNamespace(Device=lambda: SimpleNamespace(info=lambda: info)))


def test_writer_records_the_autotuners_tile_pick(monkeypatch):
    """measure()'s tile step on fake harnesses: per TM it times each geometry the autotuner tries (crew, crew2, the
    K-splits TILE_KSPLITS and a cacheless twin where the split caches its scales or is refused) and records the
    autotuner's pick among those that pass the oracle (the fastest, but crew unless that wins by at least the noise
    margin) under its name and rate, at the measured shape, every timed run with BF16 output as the engine compiles the
    tile and each followed by one untimed 32-bit-output run of the same geometry, checked by the strict rule. A refused
    geometry is no candidate; one that fails either check is recorded (through a later error too) and is no candidate
    either; a TM whose unsplit tile fails is still swept and gets no row; an unsplit refusal ends the tile step and clears
    every tile row, and so does any other error, a ValueError that is no refusal included; the tile error names the
    geometry, and the note then states that error alone."""
    import tools.profile_writer as pw

    # ms per TM and geometry: absent or None = refused for the shape, negative = fails the oracle, an exception = raised;
    # TM 8's ksplit4 caches its scales, so its cacheless twin is timed too
    tile = {8: {"crew": 1.0, "crew2": 0.9, "ksplit2": -0.3, "ksplit4": 0.7, "ksplit4nc": 0.55, "ksplit8": -0.6},
            16: {"crew": 1.2, "crew2": 1.19, "ksplit2": 1.17, "ksplit4": 1.18},           # 1.17 is within the noise of 1.2
            32: {"crew": -2.0, "ksplit2": 1.5}}
    strict_fail = {(8, "crew2")}                                            # its timed BF16 run passes, its 32-bit check fails
    bf16_fail = {(8, "ksplit4")}                                            # its 32-bit check passes, its timed BF16 run fails
    tried, checked, shapes = [], [], set()                                  # the timed runs and the check runs, in order

    class GemmBench:
        def run(self, fmt, n, k, *, tm, copies, reps, ksplit=1, tg_per_core=1, scale_cache=None, out_bf16=False, **kw):
            mode = _mode(ksplit, tg_per_core, scale_cache)
            (tried if out_bf16 else checked).append((tm, mode, copies, reps))
            shapes.add((n, k))
            ms = tile[tm].get(mode)
            if ms is None:
                raise kernels.GemmRefused(f"gemm_tile: KSPLIT={ksplit} must divide the K tiles")
            if isinstance(ms, Exception):
                raise ms
            macros = {"SCALE_CACHE": "1"} if (tm, mode) == (8, "ksplit4") else {}
            ok = ms > 0 and (tm, mode) not in (bf16_fail if out_bf16 else strict_fail)
            if out_bf16:                                                     # the timed run: the row's time, the BF16 rule's numbers
                return {"ms": abs(ms), "gbps": 1.0 / abs(ms), "ok": ok, "max_ulp_at_scale": 1.5, "max_ulp_at_rms": 9.0, "max_rel_err": 2e-3, "macros": macros}
            return {"ms": 10 * abs(ms), "gbps": 0.1 / abs(ms), "ok": ok,     # the untimed check: no row takes its time
                    "max_ulp_at_scale": 0.5, "max_ulp_at_rms": 7.0, "max_rel_err": 4e-3, "macros": macros}

    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    lines = []
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[8, 32, 16], attention=None, log=lines.append)
    assert [r[:2] for r in tried] == ([(8, md) for md in ("crew", "crew2", "ksplit2", "ksplit4", "ksplit4nc", "ksplit8", "ksplit16", "ksplit16nc")]
                                      + [(32, md) for md in ("crew", "crew2", "ksplit2", "ksplit4", "ksplit4nc", "ksplit8", "ksplit8nc", "ksplit16", "ksplit16nc")]
                                      + [(16, md) for md in ("crew", "crew2", "ksplit2", "ksplit4", "ksplit8", "ksplit8nc", "ksplit16", "ksplit16nc")])
    # each timed run (BF16 output, the caller's copies and reps) that was not refused has one untimed 32-bit check, same geometry
    assert {r[2:] for r in tried} == {(None, 3)} and checked == [(tm, md, 1, 1) for tm, md, *_ in tried if tile[tm].get(md) is not None]
    assert m["tile_ms"] == {"nvfp4": {8: 0.55, 16: 1.2}} and m["tile_mode"] == {"nvfp4": {8: "ksplit4nc", 16: "crew"}}
    assert m["tile_gbps"] == {"nvfp4": {8: 1 / 0.55, 16: 1 / 1.2}}          # the picked geometry's rate: TM 16's is crew's
    assert m["tile_oracle_fail"] == {"nvfp4": {8: ["crew2", "ksplit2", "ksplit4", "ksplit8"], 32: ["crew"]}} and shapes == {(17408, 5120)}   # measure()'s shape
    # each failing check with the numbers its own rule reads: ulp at scale for BF16 output, ulp at RMS for 32-bit output
    assert all(any(s in line for line in lines) for s in (
        "TM=8 ksplit2: ORACLE FAIL (BF16 output: 1.5 ulp at scale, rel err 2.0e-03; 32-bit output: 7.0 ulp at RMS, rel err 4.0e-03)",
        "TM=8 crew2: ORACLE FAIL (32-bit output: 7.0 ulp at RMS, rel err 4.0e-03)", "TM=8 ksplit4: ORACLE FAIL (BF16 output: 1.5 ulp at scale, rel err 2.0e-03)",
        "TM=16 ksplit8: skipped", "TM=32 crew: ORACLE FAIL"))
    engine = decide(m)[0]                                                   # the rows of the geometries that pass
    assert engine["accelerator_min_t"] == {"nvfp4": 2} and engine["cost_T"]["accelerator_nvfp4"] == pytest.approx({"8": 1.1, "16": 2.4})
    assert "/".join(map(str, TILE_KSPLITS)) in engine["note"]
    for change, error, refused in (({"crew": None}, "nvfp4 TM=16 crew: gemm_tile: KSPLIT=1", True),   # an unsplit refusal, a split's dispatch error,
                                   ({"crew": 1.2, "ksplit2": RuntimeError("dispatch failed")}, "nvfp4 TM=16 ksplit2: dispatch failed", False),
                                   ({"ksplit2": 1.17, "ksplit4": ValueError("pack failed")}, "nvfp4 TM=16 ksplit4: pack failed", False)):   # a ValueError, no refusal
        tile[16].update(change)
        m = pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[8, 32, 16], attention=None, log=lines.append)
        assert m["tile_ms"] == m["tile_gbps"] == m["tile_mode"] == {} and error in m["tile_error"] and m["tile_refused"] is refused
        engine, notes = decide(m)
        assert m["tile_oracle_fail"] == {"nvfp4": {8: ["crew2", "ksplit2", "ksplit4", "ksplit8"], 32: ["crew"]}} and engine["accelerator_min_t"] == {}
        assert notes["accelerator"] == (f"no tile rows for any format: the unsplit tile refused the measured shape: {m['tile_error']}" if refused
                                        else f"no tile measurement: the tile step ended on an error: {m['tile_error']}")
    assert any("TM=16 ksplit2: dispatch failed" in line for line in lines)  # the log names the geometry whose error ended the step


def test_writer_tile_pick_crew2_unscaled_and_duplicate_formats(monkeypatch):
    """Two threadgroups per core win when faster than crew by at least the noise margin; a format without block scales
    gets no cacheless twins; every run takes the measured lane order, each timed run the caller's copies and reps, each
    32-bit check one copy and one rep; the rows are
    in units of the same run's T = 1 shader pass; a geometry that fails the oracle is no candidate for its format's row;
    a format named twice is measured once."""
    import tools.profile_writer as pw

    class Bench:                                                            # contiguous streams faster: the measured lane order
        info = SimpleNamespace(apple_family=10, name="fake")

        def run(self, fmt, n, k, t=1, lane_order="interleaved16", **kw):
            return {"ms": 0.5 * t, "gbps": 200.0 if lane_order == "contiguous" else 100.0, "ok": True, "unit_bytes": 16}

    tile = {"fp8_e4m3": {"crew": 1.0, "crew2": 0.9, "ksplit2": 0.95},        # crew2 beats crew by more than the noise
            "nvfp4": {"crew": 1.0}, "int4_affine": {"crew": 1.0, "ksplit2": -0.8}}
    tried = []

    class GemmBench:
        def run(self, fmt, n, k, *, tm, lane_order, copies, reps, ksplit=1, tg_per_core=1, scale_cache=None, out_bf16=False, **kw):
            mode = _mode(ksplit, tg_per_core, scale_cache)
            tried.append((fmt, mode, lane_order, copies, reps, out_bf16))
            ms = tile[fmt].get(mode)
            if ms is None:
                raise kernels.GemmRefused(f"gemm_tile: KSPLIT={ksplit} refused")
            return {"ms": abs(ms), "gbps": 1.0 / abs(ms), "ok": ms > 0, "max_ulp_at_scale": 1.5, "max_ulp_at_rms": 9.0, "max_rel_err": 2e-3, "macros": {}}

    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(formats=["fp8_e4m3", "nvfp4", "int4_affine", "int4_affine"], ts=[1, 2], tms=[8], copies=4, reps=2, attention=None, log=lambda s: None)
    assert [f for f, md, *_, bf16 in tried if md == "crew" and bf16] == ["fp8_e4m3", "nvfp4", "int4_affine"]
    assert m["tile_mode"] == {"fp8_e4m3": {8: "crew2"}, "nvfp4": {8: "crew"}, "int4_affine": {8: "crew"}}
    assert {f for f, md, *_ in tried if md.endswith("nc")} == {"nvfp4", "int4_affine"}   # only block-scaled formats get cacheless twins
    # the measured lane order everywhere; the timed runs take the caller's copies and reps and BF16 output, the checks one copy, one rep, 32 bits
    assert {r[2:] for r in tried} == {("contiguous", 4, 2, True), ("contiguous", 1, 1, False)}
    engine = decide(m)[0]
    assert engine["accelerator"] == "on" and engine["accelerator_min_t"] == {"fp8": 2, "nvfp4": 2, "int4_affine": 2}
    assert engine["cost_T"]["accelerator_fp8"] == pytest.approx({"8": 1.8}) and engine["cost_T"]["accelerator_nvfp4"] == pytest.approx({"8": 2.0})
    assert engine["cost_T"]["accelerator_int4_affine"] == pytest.approx({"8": 2.0})                    # crew's row: ksplit2 failed


def test_writer_writes_no_profile_on_a_kernel_failure(monkeypatch, tmp_path, capsys):
    """A tile or shader geometry or probe that fails the oracle, or a tile-step error other than a refusal, stops main()
    before anything is written, a dry run included, and the output names each failure; empty failure records (measure()
    gives every swept format one) write the profile, and a refusal writes it without tile rows."""
    import tools.profile_writer as pw

    _fake_device(monkeypatch, pw)
    m = dict(_measurements(), tile_oracle_fail={"fp8_e4m3": {}, "nvfp4": {32: ["crew"], 8: ["ksplit2", "ksplit8nc"]}}, oracle_fail=["shader nvfp4 T=4: crew x2"])
    monkeypatch.setattr(pw, "measure", lambda **kw: dict(m))
    out, args = tmp_path / "p.json", ["--name", "x-20c", "--out", str(tmp_path / "p.json")]
    for extra in ([], ["--dry-run"]):
        assert pw.main(args + extra) == 1 and not out.exists()
    printed = capsys.readouterr().out
    assert all(f"fails the oracle: {f}" in printed for f in ("tile nvfp4 TM=8: ksplit2, ksplit8nc", "tile nvfp4 TM=32: crew", "shader nvfp4 T=4: crew x2"))
    assert "tile fp8_e4m3" not in printed
    m.update(tile_oracle_fail={"nvfp4": {}}, oracle_fail=[], tile_ms={}, tile_error="nvfp4 TM=16 ksplit2: dispatch failed", tile_refused=False)
    assert pw.main(args) == 1 and not out.exists()
    printed = capsys.readouterr().out
    assert "the tile step failed: nvfp4 TM=16 ksplit2: dispatch failed" in printed
    m.update(tile_refused=True)                                             # a refusal: the profile without tile rows
    assert pw.main(args) == 0 and Profile.from_dict("x-20c", json.loads(out.read_text())).accelerator == "off"
    del m["tile_error"], m["tile_refused"]
    m["tile_ms"] = _measurements()["tile_ms"]
    assert pw.main(args) == 0 and Profile.from_dict("x-20c", json.loads(out.read_text())).accelerator == "on"


def test_writer_records_shader_and_probe_oracle_failures(monkeypatch):
    """A shader GEMV geometry or a T = 1 probe (lane order, scale placement, threadgroups) that fails the oracle is
    recorded under oracle_fail, which stops main(); the shader cost keeps the passing geometries."""
    import tools.profile_writer as pw

    class Bench:                    # failing the oracle: the contiguous lane-order probe, the block scale-placement probe,
        info = SimpleNamespace(apple_family=10, name="fake")      # the x2 threadgroups probe and crew x2 at T = 2

        def run(self, fmt, n, k, t=1, lane_order="interleaved16", **kw):
            probe = "rows" not in kw
            bad = ((probe and lane_order == "contiguous" and not kw.keys() & {"tg_per_core", "placement"}) or (probe and kw.get("placement") == "block")
                   or (kw.get("tg_per_core") == 2 and (probe or t == 2)))
            return {"ms": 0.5 * t, "gbps": 100.0, "ok": not bad, "unit_bytes": 16, "max_ulp_at_rms": 9.0}

    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], accelerator=False, attention=None, log=lambda s: None)
    assert m["oracle_fail"] == ["lane order contiguous (nvfp4, T = 1)", "scale placement block (nvfp4, T = 1)", "threadgroups x2 (nvfp4, T = 1)",
                                "shader nvfp4 T=2: crew x2"] and m["shader_ms"]["nvfp4"] == {1: 0.5, 2: 1.0}


def test_writer_stops_on_a_tile_error_without_a_message(monkeypatch, tmp_path):
    """A tile-step error whose message is empty still stops the writer: the record names the error's type. A plain
    ValueError is no refusal (only GemmRefused is)."""
    import tools.profile_writer as pw

    class Broken:                                                            # the tile bench fails before any run
        def __init__(self):
            raise ValueError()

    _fake_device(monkeypatch, pw)
    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", Broken)
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], attention=None, log=lambda s: None)
    assert m["tile_error"] == "ValueError" and m["tile_refused"] is False
    out = tmp_path / "p.json"
    assert pw.main(["--attention", "none", "--formats", "nvfp4", "--name", "x-20c", "--out", str(out)]) == 1 and not out.exists()


@pytest.mark.parametrize("failing_t", [1, 2])
def test_writer_lists_a_token_count_where_every_shader_geometry_fails(monkeypatch, tmp_path, capsys, failing_t):
    """When every shader GEMV geometry fails the oracle at one T (T = 1 included), measure() skips that T and main()
    lists each failure, writes no profile and returns 1; the tile step still runs."""
    import tools.profile_writer as pw

    class Bench:
        info = SimpleNamespace(apple_family=10, name="fake")

        def run(self, fmt, n, k, t=1, **kw):
            return {"ms": 0.5 * t, "gbps": 100.0, "ok": not ("rows" in kw and t == failing_t), "unit_bytes": 16, "max_ulp_at_rms": 9.0}

    tile_runs = []

    class GemmBench:                                                         # crew only; every split refused
        def run(self, fmt, n, k, *, tm, ksplit=1, tg_per_core=1, **kw):
            tile_runs.append((tm, ksplit, tg_per_core))
            if ksplit != 1 or tg_per_core != 1:
                raise kernels.GemmRefused("refused")
            return {"ms": 1.0, "gbps": 1.0, "ok": True, "max_ulp_at_rms": 0.0, "macros": {}}

    _fake_device(monkeypatch, pw)
    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    out = tmp_path / "p.json"
    assert pw.main(["--attention", "none", "--formats", "nvfp4", "--ts", "1,2", "--tms", "8", "--name", "x-20c", "--out", str(out)]) == 1
    printed = capsys.readouterr().out
    assert not out.exists() and "the tile step failed" not in printed and tile_runs                 # the tile step still ran
    assert all(f"fails the oracle: shader nvfp4 T={failing_t}: {name}" in printed for name, _ in pw.GEOMETRIES)


def test_writer_pick_follows_the_noise_rule(monkeypatch):
    """The writer's pick is tune_gemm's under the same per-geometry times: crew unless another geometry is at least
    NOISE faster. Twice as fast and exactly NOISE faster win, 2 percent faster keeps crew; integer times keep the
    autotuner's per-copy division exact."""
    import numpy as np
    import tools.profile_writer as pw
    from monolith.bench import pack_spec, random_spec
    from monolith.compiler.autotune import Autotuner
    from monolith.formats import PackLayout

    assert 100 * (1 - NOISE) == 97
    table = {8: {"crew": 100, "crew2": 50}, 16: {"crew": 100, "crew2": 98}, 32: {"crew": 100, "crew2": 97}}   # every other geometry 1000
    info = pack_spec(random_spec("fp8_e4m3", 256, 4096, np.random.default_rng(0)), PackLayout(rows=16, lane_order="interleaved16"))[1]

    timing, geometry = {}, kernels.gemm_geometry                            # tune_gemm sizes each variant's grid just before timing it
    monkeypatch.setattr(kernels, "gemm_geometry", lambda mode, *a, **kw: timing.update(mode=mode) or geometry(mode, *a, **kw))
    tuner = object.__new__(Autotuner)
    tuner.nt, tuner.dev, tuner.cores, tuner.choices = SimpleNamespace(Buffer=_Stub, Library=_Stub, Pipeline=_Stub, Dispatch=_Stub), None, 20, {}
    picked = {}
    for tm, row in table.items():                                            # one x_permute precedes the copies: len(ds) - 1 of them
        tuner._time = lambda ds, row=row: row.get(timing["mode"], 1000) * (len(ds) - 1)
        picked[tm] = tuner.tune_gemm(info, tm, None).grid_mode

    class GemmBench:                                                         # the kernel's own refusals, the table's times
        def run(self, fmt, n, k, *, tm, ksplit=1, tg_per_core=1, scale_cache=None, **kw):
            macros = kernels.gemm_macros(info, tm=tm, ksplit=ksplit, scale_cache=scale_cache)
            ms = table[tm].get(_mode(ksplit, tg_per_core), 1000)
            return {"ms": ms, "gbps": 1.0 / ms, "ok": True, "max_ulp_at_rms": 0.0, "macros": macros}

    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(shape=(256, 4096), formats=["fp8_e4m3"], ts=[1, 2], tms=[8, 16, 32], attention=None, log=lambda s: None)
    assert m["tile_mode"]["fp8_e4m3"] == picked == {8: "crew2", 16: "crew", 32: "crew2"}


def test_writer_without_the_tile_writes_a_shader_profile(monkeypatch, tmp_path):
    """--no-accelerator skips the tile step (its bench is never built) and writes a profile with the tile off."""
    import tools.profile_writer as pw

    class NoTile:
        def __init__(self):
            raise AssertionError("--no-accelerator built the tile bench")

    _fake_device(monkeypatch, pw)
    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", NoTile)
    out = tmp_path / "p.json"
    assert pw.main(["--no-accelerator", "--tms", "32", "--attention", "none", "--name", "x-20c", "--out", str(out)]) == 0   # no row check
    p = Profile.from_dict("x-20c", json.loads(out.read_text()))
    assert p.accelerator == "off" and not any(key.startswith("accelerator_") for key in p.cost_t)


def test_the_tile_kernel_accepts_exactly_the_shared_splits(monkeypatch):
    """gemm_tile's split check reads TILE_KSPLITS: at K = 16384 (64 K tiles) it accepts 1 and exactly those splits, and
    a narrower list narrows the kernel."""
    import numpy as np
    from monolith.bench import pack_spec, random_spec
    from monolith.formats import PackLayout

    info = pack_spec(random_spec("fp8_e4m3", 64, 16384, np.random.default_rng(0)), PackLayout(rows=16, lane_order="interleaved16"))[1]

    def accepted():
        out = set()
        for s in range(1, 33):
            try:
                kernels.gemm_macros(info, tm=8, ksplit=s)
                out.add(s)
            except kernels.GemmRefused:
                pass
        return out

    assert accepted() == {1, *TILE_KSPLITS}
    monkeypatch.setattr(kernels, "TILE_KSPLITS", (2, 4))
    assert accepted() == {1, 2, 4}


@pytest.mark.parametrize("backend", ["m4_pro", "m5_pro"])                     # the Apple9 and Apple10 tile layouts, whatever the host
@pytest.mark.parametrize("placement", ["inline", "block"])
@pytest.mark.parametrize("fmt", sorted(RANDOM_FORMATS))                       # every format --formats takes
@pytest.mark.parametrize("k", [768, 1024, 4096, 5120, 17408])
@pytest.mark.parametrize("tm", [8, 16, 32])
def test_writer_sweep_is_tune_gemms_candidate_set(monkeypatch, fmt, k, tm, placement, backend):
    """Value: protects=the writer times exactly tune_gemm's tile candidates; fails_when=the twin rule, the splits or
    the crew modes change on one side only; why_new=the only tie is GPU-tier at one point, and without a pinned backend the contract tier checks the
    host's tile layout alone (Apple9 here, Apple10 without Metal); seam=using_backend."""
    import numpy as np
    import tools.profile_writer as pw
    from monolith.bench import pack_spec, random_spec
    from monolith.compiler.autotune import Autotuner, gemm_key
    from monolith.formats import PackLayout
    from monolith.backends.metal.context import using_backend

    with using_backend(backend):                              # the tuner and the writer see one layout
        n = 256                                                   # the candidate rule reads K, TM, the format and the pack, not N
        infos = {p: pack_spec(random_spec(fmt, n, k, np.random.default_rng(0)), PackLayout(rows=16, lane_order="interleaved16", scale_placement=p))[1]
                 for p in ("inline", "block")}
        info = infos[placement]

        tuner = object.__new__(Autotuner)
        tuner.nt, tuner.dev, tuner.cores, tuner.choices = SimpleNamespace(Buffer=_Stub, Library=_Stub, Pipeline=_Stub, Dispatch=_Stub), None, 20, {}
        tuner._time = lambda ds: 1.0
        try:
            tuner.tune_gemm(info, tm, None)
            expected = {mode for _, mode in tuner.choices[gemm_key(info, tm, None)]["variants"]}
        except kernels.GemmRefused:
            expected = None                                       # the unsplit tile refused: the writer's tile step ends
        assert expected is None or k != 768 or fmt == "bf16"      # K = 768: every quantized format's unsplit tile refuses

        timed = set()

        class GemmBench:                                          # the real refusals and SCALE_CACHE, from gemm_macros, on the pack timed
            def run(self, fmt, n, k, *, tm, lane_order, placement="inline", ksplit=1, tg_per_core=1, scale_cache=None, **kw):
                macros = kernels.gemm_macros(infos[placement], tm=tm, ksplit=ksplit, scale_cache=scale_cache)
                timed.add(_mode(ksplit, tg_per_core, scale_cache))
                return {"ms": 1.0, "gbps": 1.0, "ok": True, "max_ulp_at_rms": 0.0, "macros": macros}

        monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
        monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
        m = pw.measure(shape=(n, k), formats=[fmt], ts=[1], tms=[tm], attention=None, current_placement=placement, log=lambda s: None)
        if expected is None:
            assert m["tile_refused"] is True and m["tile_ms"] == {}, m.get("tile_error")
        else:
            assert "tile_error" not in m and timed == expected


@pytest.mark.parametrize("fmt,k,layout,kw", [("fp8_e4m3", 4096, {"rows": 32}, {"tm": 8}),                       # R
                                             ("nvfp4", 512, {"scale_placement": "block"}, {"tm": 8}),          # sub-word units
                                             ("fp8_e4m3", 4096, {}, {"tm": 8, "tn": 48, "tk": 64}),           # tile shape
                                             ("nvfp4", 768, {}, {"tm": 8}),                                   # K multiple
                                             ("fp8_e4m3", 4096, {}, {"tm": 12})])                             # TM
def test_unsplit_tile_refusals_are_gemm_refused(fmt, k, layout, kw):
    """Each reachable refusal of the unsplit tile raises GemmRefused, which the writer reads as a shape the tile does
    not take (a profile without tile rows); a plain ValueError would stop the writer as a defect."""
    import numpy as np
    from monolith.bench import pack_spec, random_spec
    from monolith.formats import PackLayout

    info = pack_spec(random_spec(fmt, 64, k, np.random.default_rng(0)), PackLayout(**{"rows": 16, "lane_order": "interleaved16", **layout}))[1]
    with pytest.raises(kernels.GemmRefused):
        kernels.gemm_macros(info, **kw)


def test_cached_and_emitted_splits_stay_inside_tile_ksplits():
    """gemm_tile accepts only TILE_KSPLITS (and 1): every split a shipped kernel-choices file caches and every split
    compiler.emit names must be one of them, or a narrower list fails those programs when they compile."""
    import re
    from pathlib import Path

    from monolith.compiler import emit

    root = Path(kernels.__file__).parent / "backends" / "metal"
    cached = {c["grid_mode"] for f in root.rglob("kernel-choices*.json") for c in json.loads(f.read_text())["choices"].values()
              if str(c.get("grid_mode", "")).startswith("ksplit")}
    emitted = set(re.findall(r'"(ksplit\d+)"', Path(emit.__file__).read_text()))
    assert cached and emitted and {kernels.gemm_ksplit(mode) for mode in cached | emitted} <= set(TILE_KSPLITS)


@pytest.mark.parametrize("args", [["--tms", "8,24"], ["--ts", "2,4"], ["--formats", "fp8"], ["--formats", ","],
                                  ["--tms", "16,32"], ["--ts", "1"], ["--quick", "--tms", "16,8"],          # no compared T has its row
                                  ["--tms", "16", "--ts", "1,2,4,8,16"], ["--quick", "--ts", "1,2,4,8,16"],   # one compared T lacks it
                                  ["--tms", "8,"], ["--tms", ""], ["--ts", "1,x"], ["--ctx", "1024,"], ["--attn-ts", "1,"],  # not integer lists
                                  ["--ts", "1,2,4,8,32", "--tms", "8,16"], ["--ts", "1,64"],   # T = 32 needs TM 32; T = 64 is never compared
                                  ["--formats", "q4_0"], ["--formats", "nvfp4,f16"],   # registered formats the harnesses cannot build
                                  ["--ts", "0,1,2"], ["--ts", "1,2,-4"]])              # token counts below 1
def test_writer_rejects_bad_arguments_before_measuring(monkeypatch, args):
    """A token-row count the tile has no variant for, a cost list without T = 1 (the costs' unit) or with a T below 1,
    a format name the harnesses cannot build (an unregistered one or one monolith.bench.random_spec does not make), an
    empty format list, a tile sweep that leaves a compared T (2 to 32) without its tile row (tile_rows, after --quick
    keeps the first --tms value) or compares at no T, and a malformed integer list stop the writer at argument parsing
    (exit 2), before the device is opened."""
    import tools.profile_writer as pw

    monkeypatch.setattr(pw, "nt", SimpleNamespace(Device=lambda: pytest.fail("the device opened before the arguments were checked")))
    with pytest.raises(SystemExit) as e:
        pw.main(args)
    assert e.value.code == 2


def test_random_formats_are_the_formats_random_spec_makes():
    """The writer's --formats check reads monolith.bench.RANDOM_FORMATS: it names exactly the registered formats
    random_spec makes, else the check admits a format the harnesses cannot build or refuses one they can."""
    import numpy as np
    from monolith.bench import RANDOM_FORMATS, random_spec
    from monolith.formats import FORMATS

    made = set()
    for f in FORMATS.names():
        try:
            random_spec(f, 16, 1024, np.random.default_rng(0))
            made.add(f)
        except KeyError:
            pass
    assert made == set(RANDOM_FORMATS)


@pytest.mark.parametrize("fail", ["lane order contiguous (fp8_e4m3, T = 1)", "shader nvfp4 T=2: crew x2"])
def test_writer_stops_on_a_shader_or_probe_failure_alone(monkeypatch, tmp_path, capsys, fail):
    """A shader geometry or a T = 1 probe that fails the oracle stops main() while every tile geometry passes, a dry
    run included, and the output names it."""
    import tools.profile_writer as pw

    _fake_device(monkeypatch, pw)
    m = dict(_measurements(), tile_oracle_fail={"fp8_e4m3": {}, "nvfp4": {}}, oracle_fail=[fail])
    monkeypatch.setattr(pw, "measure", lambda **kw: dict(m))
    out = tmp_path / "p.json"
    for extra in ([], ["--dry-run"]):
        assert pw.main(["--name", "x-20c", "--out", str(out), *extra]) == 1 and not out.exists()
        assert f"fails the oracle: {fail}" in capsys.readouterr().out


@pytest.mark.parametrize("block_gbps,current,timed", [(110.0, None, "block"), (90.0, "block", "inline"),      # a win and a loss
                                                      (101.0, "block", "block"), (101.0, None, "inline")])     # a tie keeps the current value
def test_writer_times_on_the_scale_placement_it_writes(monkeypatch, block_gbps, current, timed):
    """The shader GEMV and the tile run on the scale placement the profile will hold: the placement probe's pick under
    choose_scale_placement's rule with the profile's current value, recorded as scale_placement_timed; decide()
    writes that placement."""
    import tools.profile_writer as pw

    placements = set()

    class Bench:                                                            # the placement probe: K = 4096, no rows argument
        info = SimpleNamespace(apple_family=10, name="fake")

        def run(self, fmt, n, k, t=1, placement="inline", **kw):
            if "rows" in kw or "tg_per_core" in kw:
                placements.add(("shader" if "rows" in kw else "threadgroups", placement))
            return {"ms": 0.5 * t, "gbps": block_gbps if placement == "block" and "rows" not in kw else 100.0, "ok": True, "unit_bytes": 16}

    class GemmBench:                                                         # crew only; every other geometry refused
        def run(self, fmt, n, k, *, tm, placement="inline", ksplit=1, tg_per_core=1, **kw):
            placements.add(("tile", placement))
            if ksplit != 1 or tg_per_core != 1:
                raise kernels.GemmRefused("refused")
            return {"ms": 1.0, "gbps": 1.0, "ok": True, "max_ulp_at_rms": 0.0, "max_rel_err": 0.0, "macros": {}}

    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[8], attention=None, current_placement=current, log=lambda s: None)
    assert placements == {("shader", timed), ("threadgroups", timed), ("tile", timed)} and m["scale_placement_timed"] == timed
    assert decide(m, {"scale_placement": current} if current else None)[0]["scale_placement"] == timed


def test_writer_main_times_on_the_profiles_current_placement(monkeypatch, tmp_path):
    """main() hands measure() the existing profile's lane order and scale placement (none for a new profile): a tie
    keeps them."""
    import tools.profile_writer as pw

    _fake_device(monkeypatch, pw)
    calls = []
    monkeypatch.setattr(pw, "measure", lambda **kw: calls.append(kw) or dict(_measurements(), tile_oracle_fail={}, oracle_fail=[]))
    out = tmp_path / "p.json"
    args = ["--name", "x-20c", "--out", str(out)]
    assert pw.main(args) == 0 and calls[-1]["current_placement"] is None and calls[-1]["current_lane_order"] is None
    doc = json.loads(out.read_text())
    doc["engine"].update(scale_placement="block", lane_order="contiguous")
    out.write_text(json.dumps(doc))
    assert pw.main(args + ["--dry-run"]) == 0 and calls[-1]["current_placement"] == "block" and calls[-1]["current_lane_order"] == "contiguous"


def test_a_crew_refusal_names_its_format_and_tm(monkeypatch):
    """A refusal of the unsplit tile, the one tile error a written profile records, names its format and TM."""
    import tools.profile_writer as pw

    class GemmBench:
        def run(self, fmt, n, k, *, tm, **kw):
            raise kernels.GemmRefused("gemm_tile: R=8 must divide the 16-row tile")

    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[16], attention=None, log=lambda s: None)
    assert m["tile_refused"] is True and m["tile_error"] == "nvfp4 TM=16 crew: gemm_tile: R=8 must divide the 16-row tile"
    assert decide(m)[1]["accelerator"].endswith("nvfp4 TM=16 crew: gemm_tile: R=8 must divide the 16-row tile")


@pytest.mark.parametrize("tm,mode,bf16,error,hint", [(8, "crew", True, "MSL compile failed: no tensor ops", True),
                                                     (8, "crew", True, "pipeline failed: no tensor ops", True),
                                                     (8, "crew", True, "Caused GPU Timeout Error", False),     # a fault, not a build failure
                                                     (8, "crew", False, "MSL compile failed: x", False),       # its BF16 tile built and ran
                                                     (8, "ksplit2", True, "MSL compile failed: x", False),     # the unsplit tile built and ran
                                                     (16, "crew", True, "MSL compile failed: x", False)])      # TM 8's variants ran
def test_writer_hints_no_accelerator_only_when_the_unsplit_tile_fails_to_compile(monkeypatch, tm, mode, bf16, error, hint):
    """The tile error suggests --no-accelerator only when the unsplit tile fails to compile before any tile kernel
    ran (bf16: the failing run is the timed BF16 one, else its 32-bit check): a chip without tensor ops. Any other
    tile-step error names its geometry without the hint."""
    import tools.profile_writer as pw

    class GemmBench:
        def run(self, fmt, n, k, *, tm, ksplit=1, tg_per_core=1, out_bf16=False, **kw):
            if (tm, _mode(ksplit, tg_per_core)) == fault and out_bf16 is bf16:
                raise RuntimeError(error)
            return {"ms": 1.0, "gbps": 1.0, "ok": True, "max_ulp_at_rms": 0.0, "max_rel_err": 0.0, "macros": {}}

    fault = (tm, mode)
    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[8, 16], attention=None, log=lambda s: None)
    assert m["tile_refused"] is False and m["tile_error"].startswith(f"nvfp4 TM={tm} {mode}: {error}")
    assert ("--no-accelerator" in m["tile_error"]) is hint


@pytest.mark.parametrize("where", ["shader", "crew", "ksplit2"])
def test_writer_raises_a_shader_validation_report(monkeypatch, where):
    """A shader-validation report is no refusal and no tile error to record: measure() raises it from a shader GEMV
    geometry and from the tile step (its unsplit tile or a split), so no profile comes from a kernel that stores out
    of bounds."""
    import tools.profile_writer as pw

    report = RuntimeError("shader validation: Invalid store")

    class Bench:
        info = SimpleNamespace(apple_family=10, name="fake")

        def run(self, fmt, n, k, t=1, **kw):
            if where == "shader" and "rows" in kw and t == 2:
                raise report
            return {"ms": 0.5 * t, "gbps": 100.0, "ok": True, "unit_bytes": 16}

    class GemmBench:
        def run(self, fmt, n, k, *, tm, ksplit=1, tg_per_core=1, **kw):
            if _mode(ksplit, tg_per_core) == where:
                raise report
            return {"ms": 1.0, "gbps": 1.0, "ok": True, "max_ulp_at_rms": 0.0, "max_rel_err": 0.0, "macros": {}}

    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    with pytest.raises(RuntimeError, match="shader validation: Invalid store"):
        pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[8], attention=None, log=lambda s: None)


def test_writer_a_crew_failing_only_its_32_bit_check_gets_no_row(monkeypatch):
    """The unsplit tile passing its timed BF16 run but failing its 32-bit check gives its TM no row, like any oracle
    failure of the unsplit tile, and the sweep goes on to every other TM and format: no tile error."""
    import tools.profile_writer as pw

    swept = set()

    class GemmBench:                                                         # crew only; every other geometry refused
        def run(self, fmt, n, k, *, tm, ksplit=1, tg_per_core=1, out_bf16=False, **kw):
            swept.add((fmt, tm))
            if ksplit != 1 or tg_per_core != 1:
                raise kernels.GemmRefused("refused")
            ok = out_bf16 or (fmt, tm) != ("nvfp4", 8)                       # nvfp4 TM 8: only the 32-bit check fails
            return {"ms": 1.0, "gbps": 1.0, "ok": ok, "max_ulp_at_rms": 9.0, "max_rel_err": 2e-3, "macros": {}}

    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(formats=["nvfp4", "fp8_e4m3"], ts=[1, 2], tms=[8, 16], attention=None, log=lambda s: None)
    assert "tile_error" not in m, m.get("tile_error")
    assert m["tile_oracle_fail"] == {"nvfp4": {8: ["crew"]}, "fp8_e4m3": {}}
    assert m["tile_ms"] == {"nvfp4": {16: 1.0}, "fp8_e4m3": {8: 1.0, 16: 1.0}}
    assert swept == {(f, tm) for f in ("nvfp4", "fp8_e4m3") for tm in (8, 16)}


def test_writer_keeps_a_timed_runs_oracle_failure_through_its_checks_error(monkeypatch, tmp_path, capsys):
    """A geometry whose timed BF16 run fails the oracle stays recorded when its 32-bit check then raises: main()
    lists both the oracle failure and the tile error and writes no profile."""
    import tools.profile_writer as pw

    class GemmBench:                                                         # crew: BF16 output wrong, then its check faults
        def run(self, fmt, n, k, *, tm, out_bf16=False, **kw):
            if not out_bf16:
                raise RuntimeError("dispatch failed")
            return {"ms": 1.0, "gbps": 1.0, "ok": False, "max_ulp_at_scale": 3.0, "max_ulp_at_rms": 9.0, "max_rel_err": 2e-2, "macros": {}}

    _fake_device(monkeypatch, pw)
    monkeypatch.setattr(pw.gemv_bench, "Bench", _Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    out = tmp_path / "p.json"
    assert pw.main(["--attention", "none", "--formats", "fp8_e4m3", "--ts", "1,2", "--tms", "8", "--name", "x-20c", "--out", str(out)]) == 1
    printed = capsys.readouterr().out
    assert not out.exists() and "TM=8 crew: ORACLE FAIL (BF16 output: 3.0 ulp at scale, rel err 2.0e-02)" in printed
    assert "fails the oracle: tile fp8_e4m3 TM=8: crew" in printed and "the tile step failed: fp8_e4m3 TM=8 crew: dispatch failed" in printed


def test_writer_raises_when_every_shader_geometry_errors_at_one_t(monkeypatch):
    """Every shader geometry at one T raising, with none failing the oracle, leaves that T unmeasured: measure()
    raises rather than return a cost table without it."""
    import tools.profile_writer as pw

    class Bench:                                                            # every shader geometry at T = 2 raises
        info = SimpleNamespace(apple_family=10, name="fake")

        def run(self, fmt, n, k, t=1, **kw):
            if "rows" in kw and t == 2:
                raise RuntimeError("pipeline failed: register budget")
            return {"ms": 0.5 * t, "gbps": 100.0, "ok": True, "unit_bytes": 16}

    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    with pytest.raises(RuntimeError, match="no geometry ran for nvfp4 at T = 2"):
        pw.measure(formats=["nvfp4"], ts=[1, 2], accelerator=False, attention=None, log=lambda s: None)


@pytest.mark.parametrize("gbps,current,timed", [({"interleaved16": 100.0, "contiguous": 101.0}, "interleaved16", "interleaved16"),  # ties keep
                                                ({"interleaved16": 101.0, "contiguous": 100.0}, "contiguous", "contiguous"),        # the profile's
                                                ({"interleaved16": 100.0, "contiguous": 101.0}, None, "interleaved16"),            # a new chip: the default
                                                ({"interleaved16": 100.0, "contiguous": 120.0}, "interleaved16", "contiguous")])   # a clear win
def test_writer_times_on_the_lane_order_it_writes(monkeypatch, gbps, current, timed):
    """Every run after the lane-order probe takes the lane order the profile will hold: the probe's pick under
    choose_lane_order's rule with the profile's current value, which decide() writes."""
    import tools.profile_writer as pw

    lanes = set()

    class Bench:
        info = SimpleNamespace(apple_family=10, name="fake")

        def run(self, fmt, n, k, t=1, lane_order="interleaved16", **kw):
            probe = "rows" not in kw and not kw.keys() & {"tg_per_core", "placement"}
            if not probe:
                lanes.add(lane_order)
            return {"ms": 0.5 * t, "gbps": gbps[lane_order] if probe else 100.0, "ok": True, "unit_bytes": 16}

    class GemmBench:                                                         # crew only; every other geometry refused
        def run(self, fmt, n, k, *, tm, lane_order, ksplit=1, tg_per_core=1, **kw):
            lanes.add(lane_order)
            if ksplit != 1 or tg_per_core != 1:
                raise kernels.GemmRefused("refused")
            return {"ms": 1.0, "gbps": 1.0, "ok": True, "max_ulp_at_rms": 0.0, "max_rel_err": 0.0, "macros": {}}

    monkeypatch.setattr(pw.gemv_bench, "Bench", Bench)
    monkeypatch.setattr(pw.gemm_bench, "GemmBench", GemmBench)
    m = pw.measure(formats=["nvfp4"], ts=[1, 2], tms=[8], attention=None, current_lane_order=current, log=lambda s: None)
    assert lanes == {timed} and m["lane_order_measured_with"] == timed
    assert decide(m, {"lane_order": current} if current else None)[0]["lane_order"] == timed


def test_gemm_bench_judges_each_output_by_its_rule(monkeypatch):
    """GemmBench.run judges 32-bit output by OracleCheck.ok (the writer's strict check) and BF16 output by
    ok_rounded (one rounding of the FP32 result, the timed run), and reports the metric each rule reads."""
    import tools.profile_writer as pw                                        # puts tools/bench on sys.path
    from monolith.bench import OracleCheck

    gemm_bench = pw.gemm_bench

    class Buf(_Stub):
        def read(self, off, size):
            return bytes(size)

    info = SimpleNamespace(gpu_cores=20, apple_family=10, name="fake")
    monkeypatch.setattr(gemm_bench, "nt", SimpleNamespace(Device=lambda: SimpleNamespace(info=lambda: info), Buffer=Buf, Library=_Stub,
                                                          Pipeline=_Stub, Dispatch=_Stub,
                                                          Queue=lambda dev: SimpleNamespace(run=lambda ds: SimpleNamespace(error=None, gpu_ms=1.0))))
    # within one BF16 rounding (1 ulp at scale, rel 5e-3) but past the strict rule (3 ulp at RMS, rel 5e-3 > 1e-4)
    monkeypatch.setattr(gemm_bench, "check_against_oracle", lambda y, ref: OracleCheck(3.0, 4, 5e-3, 1.0))
    gb = gemm_bench.GemmBench()
    bf16 = gb.run("fp8_e4m3", 64, 1024, tm=8, copies=1, reps=1, out_bf16=True)
    strict = gb.run("fp8_e4m3", 64, 1024, tm=8, copies=1, reps=1)
    assert bf16["ok"] is True and strict["ok"] is False and bf16["max_ulp_at_scale"] == strict["max_ulp_at_scale"] == 1.0


def test_gemm_bench_refuses_a_geometry_before_allocating(monkeypatch):
    """GemmBench.run asks gemm_macros for the geometry before it allocates the >= 2 GiB of weight copies, so a refused
    K-split (the writer's sweep meets dozens per run) costs no allocation."""
    import tools.profile_writer as pw                                        # puts tools/bench on sys.path

    gemm_bench, allocs = pw.gemm_bench, []

    class Buffer:
        def __init__(self, dev, arg):
            allocs.append(arg)

        def __getattr__(self, name):
            return lambda *a, **kw: None

    monkeypatch.setattr(gemm_bench, "nt", SimpleNamespace(Buffer=Buffer))
    bench = object.__new__(gemm_bench.GemmBench)
    bench.dev, bench.info, bench.profile, bench._pipelines = None, SimpleNamespace(gpu_cores=40, name="fake"), None, {}
    with pytest.raises(kernels.GemmRefused):
        bench.run("fp8_e4m3", 64, 5120, tm=8, ksplit=16)                     # 20 or 40 K tiles: 16 divides neither
    assert allocs == []


@pytest.mark.parametrize("mode,kw", [("crew", {}), ("crew2", {"tg_per_core": 2}), ("ksplit4", {"ksplit": 4})])
def test_gemm_bench_dispatches_each_mode_in_the_tuners_geometry(monkeypatch, mode, kw):
    """GemmBench.run dispatches gemm_tile for a mode in kernels.gemm_geometry's (SIMD-groups, threadgroups, threadgroup
    size), the geometry tune_gemm times under that name, so the writer's row for a mode prices the tuner's geometry."""
    import struct
    import tools.profile_writer as pw                                        # puts tools/bench on sys.path

    gemm_bench, tiles = pw.gemm_bench, []

    class Pipeline(_Stub):
        def __init__(self, lib, function):
            self.function = function

    class Dispatch(_Stub):                                                   # records the tile dispatch's params, grid and threadgroup
        def pipeline(self, pso):
            self.function = pso.function
            return self

        def bytes(self, index, b):
            self.params = b
            return self

        def grid(self, n):
            self.n_tg = n
            return self

        def threadgroup(self, n):
            self.tg = n
            if self.function == "gemm_tile":
                tiles.append(self)
            return self

    info = SimpleNamespace(gpu_cores=40, apple_family=10, name="fake")
    monkeypatch.setattr(gemm_bench, "nt", SimpleNamespace(Device=lambda: SimpleNamespace(info=lambda: info), Buffer=_Stub, Library=_Stub,
                                                          Pipeline=Pipeline, Dispatch=Dispatch,
                                                          Queue=lambda dev: SimpleNamespace(run=lambda ds: SimpleNamespace(error=None, gpu_ms=1.0))))
    r = gemm_bench.GemmBench().run("fp8_e4m3", 4096, 4096, tm=8, copies=1, reps=1, check=False, **kw)
    (tile,) = tiles
    n_sg = struct.unpack("<IIIIfIII", tile.params)[2]
    assert (n_sg, tile.n_tg, tile.tg) == kernels.gemm_geometry(mode, kernels.gemm_tiles(4096, r["tn"]), 40, 384)


def test_the_no_accelerator_hint_prefixes_are_the_runtimes():
    """measure() adds the --no-accelerator hint when the unsplit tile's error starts with the runtime's compile or pipeline
    failure message (runtime/src/metal_core.mm), so a reworded native message would drop the hint with every test green."""
    from pathlib import Path

    src = (Path(kernels.__file__).resolve().parents[1] / "runtime" / "src" / "metal_core.mm").read_text()
    assert 'std::string("MSL compile failed: ")' in src and 'std::string("pipeline failed: ")' in src


def test_the_tile_row_counts_stay_inside_tile_tms():
    """gemm_tile accepts exactly calibration.TILE_TMS as token rows, and every row count the emitter (gemm_tm) and the
    profile's cost table (tile_rows) ask for is one of them, the same for each T: the writer's --tms guard, the kernel
    and the rows the engine dispatches agree."""
    import numpy as np
    from monolith.backends.metal.calibration import TILE_TMS
    from monolith.bench import pack_spec, random_spec
    from monolith.compiler.emit import gemm_tm
    from monolith.formats import PackLayout

    info = pack_spec(random_spec("fp8_e4m3", 64, 4096, np.random.default_rng(0)), PackLayout(rows=16, lane_order="interleaved16"))[1]

    def accepted(tm):
        try:
            kernels.gemm_macros(info, tm=tm)
            return True
        except kernels.GemmRefused:
            return False

    assert {tm for tm in range(1, 129) if accepted(tm)} == set(TILE_TMS)
    assert {gemm_tm(t) for t in range(1, 65)} <= set(TILE_TMS) and {tile_rows(t) for t in range(1, 33)} <= set(TILE_TMS)
    assert all(gemm_tm(t) == tile_rows(t) for t in range(1, 33))
