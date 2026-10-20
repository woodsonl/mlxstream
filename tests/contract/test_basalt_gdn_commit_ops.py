"""The packer of basalt's GDN commit (monolith/kernels.py ``basalt_gdn_commit_params`` and ``basalt_gdn_commit_grid``):
the record's 48 bytes at the contract's shapes and at basalt's multi-layer strides, the threadgroup count, and the
refusals, each a valid input with exactly one defect refused by that defect's own check (its message). No GPU."""

import re
import struct

import pytest

from monolith import kernels

C, HV, MT = 10240, 48, 8          # Flash-Next's GDN layer (h_k = 16) and the contract's window rows (D18)
STATE = 817152                    # a layer's 128 x 48 x 128 state then its 10240 x 3 history (D17; dcp's GDN_FLOATS)
RECORD = "<12I"                   # GdnCommitParams (basalt_gdn_commit.metal)


def test_basalt_gdn_commit_params_record_and_grid():
    """The record's fields in GdnCommitParams' order (state_stride, h_stride, gb_stride, C, h_k, h_v, n_layers,
    conv_blocks, mt, three zero words): the dense defaults at the contract's shapes and at the smallest layer (one key
    and one value head, one row); dec_commit_parity.cpp:83-89's three layers at its MT = 7, strides given; and
    dec_gdn_parity.cpp:519-521's form, two layers whose rows sit in a wider window (20 rows here), packed as given. The
    threadgroup count is one per item, 272 a layer."""
    rec = kernels.basalt_gdn_commit_params(C, HV, MT)
    assert len(rec) == 48 and struct.unpack(RECORD, rec) == (817152, 81920, 384, 10240, 16, 48, 1, 80, 8, 0, 0, 0)
    small = kernels.basalt_gdn_commit_params(384, 1, 1)
    assert struct.unpack(RECORD, small) == (17536, 384, 1, 384, 1, 1, 1, 3, 1, 0, 0, 0)
    dcp = kernels.basalt_gdn_commit_params(C, HV, 7, n_layers=3, state_stride=STATE, h_stride=7 * C, gb_stride=7 * HV)
    assert struct.unpack(RECORD, dcp) == (STATE, 71680, 336, C, 16, HV, 3, 80, 7, 0, 0, 0)
    dgp = kernels.basalt_gdn_commit_params(C, HV, MT, n_layers=2, state_stride=STATE, h_stride=20 * C,
                                           gb_stride=20 * HV)
    assert struct.unpack(RECORD, dgp) == (STATE, 204800, 960, C, 16, HV, 2, 80, MT, 0, 0, 0)
    assert kernels.basalt_gdn_commit_grid(C, HV) == 272 and kernels.basalt_gdn_commit_grid(C, HV, 36) == 9792


def _packer(bad, good=None):
    """The packer at the contract's shapes with the keywords ``good`` (valid: none, or each one step inside its
    limit), or with ``bad`` instead (the one defect)."""
    return lambda defect: kernels.basalt_gdn_commit_params(
        **{"c": C, "h_v": HV, "mt": MT, **(bad if defect else (good or {}))})


# (id, build, why): build(False) must pass and build(True) raise a ValueError whose message holds why, literally
REFUSALS = [
    ("packer h_v of 0", _packer({"h_v": 0}), "h_v must be at least 1, got 0"),
    ("packer C - 128 h_v of 0", _packer({"c": 128 * HV}),
     "C - 128 * h_v must be a positive multiple of 256, got 0 (C 6144, h_v 48)"),
    ("packer C - 128 h_v not whole 256s", _packer({"c": C + 128}),
     "C - 128 * h_v must be a positive multiple of 256, got 4224 (C 10368, h_v 48)"),
    ("packer mt of 0", _packer({"mt": 0}, {"mt": 1}), "mt must be 1 to 8, got 0"),
    ("packer mt of 9", _packer({"mt": 9}, {"mt": 8}), "mt must be 1 to 8, got 9"),
    ("packer n_layers of 0", _packer({"n_layers": 0}, {"n_layers": 1}), "n_layers must be at least 1, got 0"),
    ("packer state_stride short", _packer({"state_stride": STATE - 1}, {"state_stride": STATE}),
     "state_stride 817151 is below its dense value 817152"),
    ("packer h_stride short", _packer({"h_stride": 81919}, {"h_stride": 81920}),
     "h_stride 81919 is below its dense value 81920"),
    ("packer gb_stride short", _packer({"gb_stride": 383}, {"gb_stride": 384}),
     "gb_stride 383 is below its dense value 384"),
    # 5256 layers of 817,152 floats end at 4,294,950,912 < 2^32; 5257 end at 4,295,768,064
    ("packer n_layers past the uint index", _packer({"n_layers": 5257}, {"n_layers": 5256}),
     "n_layers 5257 times the widest stride 817152 reaches 2^32 floats"),
    ("packer h_stride past the uint index", _packer({"h_stride": 2 ** 32}, {"h_stride": 2 ** 32 - 1}),
     "n_layers 1 times the widest stride 4294967296 reaches 2^32 floats"),
    ("packer gb_stride past the uint index", _packer({"gb_stride": 2 ** 32}, {"gb_stride": 2 ** 32 - 1}),
     "n_layers 1 times the widest stride 4294967296 reaches 2^32 floats"),
]


@pytest.mark.parametrize("build, why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(build, why):
    """CT.4: each case passes without its defect, then its defect is refused by its own check (the message)."""
    build(False)
    with pytest.raises(ValueError, match=re.escape(why)):
        build(True)
