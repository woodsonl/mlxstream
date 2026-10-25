"""The source distribution carries every file the native module compiles."""
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_manifest_ships_every_native_source():
    cmake = (ROOT / "runtime" / "CMakeLists.txt").read_text()
    sources = re.search(r"nanobind_add_module\(\s*_native\s+NB_STATIC\s+([^)]*)\)", cmake).group(1).split()
    assert sources and all((ROOT / "runtime" / s).is_file() for s in sources)
    patterns = [p for line in (ROOT / "MANIFEST.in").read_text().splitlines()
                if line.split()[:2] == ["recursive-include", "runtime"] for p in line.split()[2:]]
    missing = [s for s in sources if not any(fnmatch.fnmatch(Path(s).name, p) for p in patterns)]
    assert not missing, f"MANIFEST.in's runtime rule leaves out {missing}"
