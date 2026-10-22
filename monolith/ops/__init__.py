"""The op registry: for each op kind, its class, block domain, cost model and the kernel binding per chip profile.
A missing binding for the target profile fails the build (the coverage guard). Importing this package registers the
decode op kinds the layer library lowers to (``embed``, ``rmsnorm_stat``, ``norm_apply``, ``gemv``, ``lm_head``,
``gqa_decode``, ``gdn_mixer``, ``argmax``, ``sample``), the DSpark round's (``tap_concat``, ``draft_attn``,
``confidence``, ``verify_select``, ``accept_scan``), basalt's GEMVs and q8_1 quantizers (``ops/basalt.py``),
basalt's sampler (``basalt_sample``) and basalt's GDN decode window (``basalt_gdn_window``); kernels bind to them per
profile."""

from .registry import OPS, CostModel, KernelBinding, OpDef, register_op
from . import attention, basalt, basalt_gdn, basalt_ple, basalt_sample, draft, embed, gdn, gemv, moe, norm, sample  # noqa: F401  (registration)

from . import basalt_prompt  # noqa: F401  (registration: basalt's prompt path, ops/basalt_prompt.py)

__all__ = ["OPS", "CostModel", "KernelBinding", "OpDef", "register_op"]
