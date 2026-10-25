"""tools/bench/gqa_bench.py's check_exact: the EXACT_SUB row counts the bench refuses before it compiles anything."""
import pytest

from tools.bench.gqa_bench import check_exact


@pytest.mark.parametrize("exact,ts,kw,reason", [
    (-1, [512], {}, "uint"), (2 ** 32, [2 ** 32 + 1], {}, "uint"),            # EXACT_SUB is a uint
    (128, [512], {"v2": True}, "v1 or v3"), (128, [512], {"steal": True}, "v1 or v3"),   # no exact form
    (128, [128], {}, "one sub-pass"), (128, [512, 4], {}, "one sub-pass"),    # a pass of T <= 128 rows: the plain kernel
])
def test_check_exact_refuses(exact, ts, kw, reason):
    with pytest.raises(ValueError, match=reason):
        check_exact(exact, ts, **kw)


@pytest.mark.parametrize("exact,ts,kw", [
    (0, [1, 4], {}), (0, [1], {"v2": True}), (0, [1], {"steal": True}),
    (128, [129, 512], {}), (1, [2], {}), (2 ** 32 - 1, [2 ** 32], {}),
])
def test_check_exact_accepts(exact, ts, kw):
    check_exact(exact, ts, **kw)
