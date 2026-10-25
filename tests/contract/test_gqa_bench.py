"""tools/bench/gqa_bench.py's refusals: check_exact's EXACT_SUB row counts, and the command line's, before a device opens."""
import pytest

from tools.bench import gqa_bench
from tools.bench.gqa_bench import check_exact


@pytest.mark.parametrize("exact,ts,kw,reason", [
    (-1, [512], {}, "uint"), (2 ** 32, [2 ** 32 + 1], {}, "uint"),            # EXACT_SUB is a uint
    (128, [512], {"v2": True}, "v1 or v3"), (128, [512], {"steal": True}, "v1 or v3"),   # no exact form
    (128, [128], {}, "one sub-pass"), (128, [512, 4], {}, "one sub-pass"),    # v1's pass of T <= 128 rows: the plain kernel
])
def test_check_exact_refuses(exact, ts, kw, reason):
    with pytest.raises(ValueError, match=reason):
        check_exact(exact, ts, **kw)


@pytest.mark.parametrize("exact,ts,kw", [
    (0, [1, 4], {}), (0, [1], {"v2": True}), (0, [1], {"steal": True}),
    (128, [129, 512], {}), (1, [2], {}), (2 ** 32 - 1, [2 ** 32], {}),
    (128, [128, 1], {"v3": True}),                 # v3's rows take the 128-row grid from token 0, wherever a pass starts
])
def test_check_exact_accepts(exact, ts, kw):
    check_exact(exact, ts, **kw)


@pytest.mark.parametrize("argv", [
    ["--exact", "128", "--t", "1,4"],            # every pass inside one sub-pass
    ["--v2", "--exact", "128", "--t", "512"],    # no exact form
    ["--v3", "--steal"], ["--v2", "--v3"],       # one kernel at a time
])
def test_cli_refuses_before_a_device_opens(argv, monkeypatch):
    monkeypatch.setattr(gqa_bench, "nt", None)   # a device access is an AttributeError, not a GPU run
    with pytest.raises(SystemExit) as e:
        gqa_bench.main(argv)
    assert e.value.code == 2
