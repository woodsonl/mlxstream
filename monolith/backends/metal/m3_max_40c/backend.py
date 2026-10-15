from ..base import MetalBackend


class Backend(MetalBackend):
    """Native backend for the 40-core M3 Max: its own measured configuration and kernel choices."""
    id = "m3_max_40c"

    def serving_prefill_chunk(self, recipes, model=None):
        # pins serve's fallback of 128 rows: the prompt-pass kernel choices were measured in 128-row programs, and a
        # 512-row exact pass (the M3 Ultra's) with the DSpark drafter resident ran out of GPU memory on the measured
        # 48 GB machine
        return 128
