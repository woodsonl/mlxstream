"""Explicit direct-cache tuning does not broaden automatic attention selection."""
from types import SimpleNamespace

import pytest

from monolith import kernels
from monolith.backends.metal.context import using_backend
from monolith.compiler.emit import _gqa_kernel, _gqa_mma_direct


def context(**changes):
    values = dict(t=8, attention="mma-direct", cores=40, dynamic_t=False,
                  speculative=False, commute_norm=True, accelerator="on")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("d,heads,kv,norm", [(64, 32, 8, False),
                          (128, 24, 8, False), (256, 8, 2, True)])
def test_measured_static_shapes_select_direct_cache(d, heads, kv, norm):
    c = context()
    attrs = dict(head_dim=d, heads=heads, kv_heads=kv, qk_norm=norm)
    with using_backend("m5_max_40c"):
        assert _gqa_kernel(c, heads, kv, d=d, qk_norm=norm) == "mma"
        assert _gqa_mma_direct(c, attrs, 8, 33024)
        with pytest.raises(ValueError, match="divisible by 256"):
            _gqa_mma_direct(c, attrs, 8, 33025)


@pytest.mark.parametrize("changes", [dict(t=1), dict(t=4), dict(dynamic_t=True),
    dict(speculative=True), dict(commute_norm=False), dict(accelerator="off"),
    dict(cores=32)])
def test_unmeasured_modes_retain_automatic_d64_route(changes):
    with using_backend("m5_max_40c"):
        c = context(**changes)
        assert _gqa_kernel(c, 32, 8, d=64, qk_norm=False) == "v3"
        assert not _gqa_mma_direct(c, dict(head_dim=64, heads=32, kv_heads=8,
                                         qk_norm=False), c.t, 33024)


def test_automatic_and_other_backends_are_unchanged():
    with using_backend("m5_max_40c"):
        assert _gqa_kernel(context(attention="auto"), 32, 8, d=64, qk_norm=False) == "v3"
        assert _gqa_kernel(context(), 32, 8, lm_mode=1, d=64, qk_norm=False) == "v3"
        # The MoE and the 0.6B shape are not promoted by this explicit route.
        for heads, kv in ((32, 4), (16, 8)):
            assert not _gqa_mma_direct(context(), dict(head_dim=128, heads=heads,
                kv_heads=kv, qk_norm=True), 8, 33024)
    with using_backend("m5_max_32c"):
        assert _gqa_kernel(context(), 32, 8, d=64, qk_norm=False) == "v3"


@pytest.mark.parametrize("d,nsg", [(32, 8), (64, 16), (96, 8), (128, 32), (256, 16), (960, 4), (1632, 4)])
def test_automatic_v3_geometry_builds_for_each_head_dim(d, nsg):
    # auto runs v3 wherever mma does not; head_dim 96 made 24 SIMD-groups, which gqa_v3_macros rejects
    assert _gqa_kernel(context(attention="auto", t=1), 16, 8, d=d) == "v3"
    assert kernels.gqa_v3_simdgroups(d) == nsg
    assert kernels.gqa_v3_macros(d)["NSG3"] == f"{nsg}u"


def test_v3_threadgroup_budget_counts_the_query():
    # gqa_decode_v3 declares query[D] beside the fold buffer: head_dim 960 at 8 SIMD-groups needs 34,688 bytes, and
    # head_dim 1664 needs 33,344 even at 4, so v3 fits head dims up to 1632
    with pytest.raises(ValueError):
        kernels.gqa_v3_macros(960, nsg=8)
    with pytest.raises(ValueError):
        kernels.gqa_v3_macros(1664)
