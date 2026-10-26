"""tools/profile_writer.py on the device: a short measurement at a small shape produces a profile the loader accepts
and the decisions' inputs are all present (the real run takes minutes and writes the selected backend configuration)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith.core.profile import Profile, load_profile  # noqa: E402
from monolith.core.profile_writer import decide, merge_profile  # noqa: E402
from monolith.runtime import is_available  # noqa: E402

pytestmark = pytest.mark.skipif(not is_available(), reason="monolith.runtime._native is not built")


def test_writer_measures_and_the_profile_loads(tmp_path):
    from tools.profile_writer import device_facts, measure
    from monolith.runtime import _native as nt

    lines = []
    m = measure(shape=(2048, 1024), formats=["fp8_e4m3", "nvfp4"], ts=[1, 2], tms=[8], copies=4, reps=1, attention=(8, 2, 64), ctxs=[256],
                attn_ts=[1], log=lines.append)
    assert set(m["lane_order_gbps"]) == {"interleaved16", "contiguous"} and set(m["threadgroups_ms"]) == {1, 2}
    assert set(m["shader_ms"]["fp8_e4m3"]) == {1, 2} and set(m["shader_ms"]["nvfp4"]) == {1, 2} and all(v > 0 for v in m["shader_ms"]["nvfp4"].values())
    assert not any("ORACLE FAIL" in l for l in lines), [l for l in lines if "FAIL" in l]
    engine, notes = decide(m, None)
    info = nt.Device().info()
    doc = merge_profile(None, device=device_facts(info, target_gb=21.0), engine=engine, measurements=m, notes=notes, written="2026-09-25", command="test")
    p = Profile.from_dict("x", doc)
    assert p.family == f"Apple{info.apple_family}" and p.gpu_cores == info.gpu_cores and p.cost("fp8", 1) == 1.0 and 2 in p.cost_t["fp8"]
    assert p.lane_order in ("contiguous", "interleaved16") and p.threadgroups_per_core in (1, 2) and p.attention in ("v1", "v2", "auto")
    assert "tile_error" not in m, m["tile_error"]                         # the kernel tier needs the tile on every device it runs on
    assert not any(m["tile_oracle_fail"].values()), m["tile_oracle_fail"]  # a failing geometry stops the writer
    assert not m["oracle_fail"], m["oracle_fail"]                         # a failing shader geometry or probe stops it too
    assert p.cost_t["accelerator_fp8"][8] > 0 and p.accelerator in ("on", "off")
    assert {"crew", "crew2"} <= set(m["tile_geometry_ms"]["fp8_e4m3"][8])  # every geometry recorded, the unsplit tile at both counts
    assert all(m["tile_mode"][f][8] in ("crew", "crew2", "ksplit2", "ksplit2nc", "ksplit4", "ksplit4nc")    # K = 1024: 4 K tiles,
               for f in ("fp8_e4m3", "nvfp4"))                                                                 # so splits 8 and 16 are refused
    out = tmp_path / "x.json"
    out.write_text(json.dumps(doc, indent=2))
    assert load_profile(out).nominal_gbps == p.nominal_gbps and doc["writer"]["decisions"]["lane_order"]


def test_writer_times_every_tile_geometry_the_autotuner_tries(monkeypatch):
    """The writer's tile sweep is the autotuner's candidate set (compiler.autotune.tune_gemm) on the same pack, and its
    pick is one of them: INT4 at K = 4096, TM 8, where the cacheless twins are candidates."""
    import numpy as np
    import tools.profile_writer as pw
    from monolith.bench import pack_spec, random_spec
    from monolith.compiler.autotune import Autotuner, gemm_key
    from monolith.formats import PackLayout
    from monolith.runtime import _native as nt

    timed, real = set(), pw.gemm_bench.GemmBench.run

    def spy(self, fmt, n, k, *, tm, ksplit=1, tg_per_core=1, scale_cache=None, **kw):
        r = real(self, fmt, n, k, tm=tm, ksplit=ksplit, tg_per_core=tg_per_core, scale_cache=scale_cache, **kw)
        timed.add(("crew2" if tg_per_core == 2 else "crew") if ksplit == 1 else f"ksplit{ksplit}" + ("nc" if scale_cache is False else ""))
        return r

    monkeypatch.setattr(pw.gemm_bench.GemmBench, "run", spy)
    m = pw.measure(shape=(2048, 4096), formats=["int4_affine"], ts=[1], tms=[8], copies=2, reps=1, attention=None, log=lambda s: None)
    assert "tile_error" not in m and not any(m["tile_oracle_fail"].values()), (m.get("tile_error"), m["tile_oracle_fail"])  # the only run of the
    # 8- and 16-way splits without their scale cache through the reference check
    _, info, _ = pack_spec(random_spec("int4_affine", 2048, 4096, np.random.default_rng(0)),
                           PackLayout(rows=16, lane_order=m["lane_order_measured_with"], scale_placement=m["scale_placement_timed"]))
    dev = nt.Device()
    tuner = Autotuner(dev, dev.info().gpu_cores, reps=1, warmup_ms=1.0)
    tuner.tune_gemm(info, 8, None)
    variants = {mode for _, mode in tuner.choices[gemm_key(info, 8, None)]["variants"]}
    assert timed == variants and any(v.endswith("nc") for v in variants) and m["tile_mode"]["int4_affine"][8] in variants
