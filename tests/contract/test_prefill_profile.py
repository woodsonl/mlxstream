"""tools/bench/prefill_profile.py's record: which profiled prompt programs it calls exact."""
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")                # main() imports monolith.serve, which needs the serve extra
from tools.bench import prefill_profile


@pytest.mark.parametrize("prefill_exact,macros,exact", [
    (True, {}, True),                         # the large exact graph, whose matrix attention takes no EXACT_SUB
    (False, {"EXACT_SUB": "128u"}, True),     # a 128-row exact program: v3's exact form
    (False, {}, False),                       # a 128-row program the plain one matches (v1, the matrix attention)
])
def test_the_record_calls_the_exact_programs_exact(tmp_path, monkeypatch, prefill_exact, macros, exact):
    program = SimpleNamespace(kernels={"attention": SimpleNamespace(macros=macros)}, buffers={}, ops=[], backend_id="b",
                              context_capacity=1024, step_state="st", layout=SimpleNamespace(pack=lambda state: b""))
    engine = SimpleNamespace(program=program, state=lambda: {"error": 0}, run=lambda *a, **k: SimpleNamespace(gpu_ms=1.0),
                             buffers={"st": SimpleNamespace(write=lambda data, offset: None)})
    session = SimpleNamespace(prefill_exact=prefill_exact, prefill_engine=lambda: engine, reset=lambda **k: None)
    assets = SimpleNamespace(model_dir="m", pack_dir="p", options=lambda tokens: ("recipe", {}))
    monkeypatch.setattr("monolith.serving.setup.prepare", lambda args: assets)
    monkeypatch.setattr("monolith.generate.load_session", lambda *a, **k: session)
    out = tmp_path / "profile.json"
    monkeypatch.setattr(sys, "argv", ["prefill_profile.py", "--model", "m", "--exact", "--positions", "0",
                                      "--repeats", "1", "--out", str(out)])
    prefill_profile.main()
    assert json.loads(out.read_text())["exact"] is exact
