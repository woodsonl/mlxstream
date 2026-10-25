"""``monolith/compiler/ple_tier.py``: the host point that drives the n-gram row gather.

The hash and the gather are host work, and a program is encoded once and replayed, so placing them inside it
would run them once at encode time and replay that one token's rows for every later step: a finite, stable,
wrong answer rather than a crash. The split keeps them at a host point between passes.

These cases hold the three things a port can get wrong about that point, each of which leaves a program that
runs: one point per step rather than one per layer, a window wide enough to hide the reads, and the staging
buffer declared so no op of the window can read it under the gather. Contract tier: no GPU, no native module.
"""

import pytest

from monolith.compiler.ple_tier import (
    HANDOFF_IDLE_US,
    PLE_N_HEADS,
    PLE_ROW_BYTES,
    PleWindow,
    decode_ple_point,
    gather_bytes_per_token,
    handoff_cost_ms,
)
from monolith.runtime.program import HostPoint

# The embedding is op 0; layer 0's ops are 1 and 2; the PLE block opens layer 1 at op 3.
WINDOW = PleWindow(publish=1, wait=3)


def test_one_point_per_step_not_one_per_layer():
    """The PLE block runs at layer 1 and nowhere else, so a step carries exactly one of these. This is the
    difference from the CPU expert tier, which carries one per MoE layer, and getting it wrong costs a handoff
    per layer for a gather that happens once."""
    points = decode_ple_point(WINDOW)
    assert len(points) == 1
    assert points[0] == HostPoint(task="ple_gather", publish=1, wait=3,
                                  writes=["ple_staging"], reads=["step_state"])


def test_the_window_has_ops_in_it():
    """A point whose publish equals its wait is a sync point: the GPU idles for the whole gather. The window
    exists because sixteen rows on sixteen pages all depend on the token, so the embedding and layer 0 are the
    only work there is to hide them behind."""
    p = decode_ple_point(WINDOW)[0]
    assert p.wait > p.publish, "the window is empty, so nothing runs beside the gather"


def test_an_inverted_window_is_refused():
    with pytest.raises(ValueError, match="publish <= wait"):
        PleWindow(publish=5, wait=2)


def test_a_sync_point_window_is_allowed_but_is_the_degenerate_case():
    """publish == wait is legal and is what a program with nothing to overlap would produce. It is allowed so a
    caller can express it deliberately, and the test above is what keeps the real one from becoming it."""
    p = decode_ple_point(PleWindow(publish=4, wait=4))[0]
    assert p.publish == p.wait


def test_the_staging_buffer_is_declared_written():
    """What stands in for basalt's fence. The gather writes the staging buffer and the pump commits the command
    buffer that reads it only after the point returns; naming the buffer in ``writes`` is what makes
    ``Program.check_host_points`` refuse an op of the window that reads it under the gather."""
    p = decode_ple_point(WINDOW)[0]
    assert p.writes == ["ple_staging"]
    assert p.reads == ["step_state"]


def test_the_buffer_names_follow_the_program():
    """A program that renamed its step state would otherwise get a point naming a buffer it does not have,
    which ``check_host_points`` rejects, or one that is not the buffer its ops write."""
    p = decode_ple_point(WINDOW, task="g", staging="s", step_state="ss")[0]
    assert (p.task, p.writes, p.reads) == ("g", ["s"], ["ss"])


def test_one_tokens_gather_is_1440_bytes():
    """Sixteen rows of 90. This is why the tier is cheap next to the experts, and it is stated as a number
    rather than a claim so a change to either constant shows up here."""
    assert PLE_N_HEADS == 16 and PLE_ROW_BYTES == 90
    assert gather_bytes_per_token() == 1440


def test_the_handoff_price_is_one_handoff():
    """One point, so one handoff: about 0.14 ms of GPU idle per step, against the 6.5 to 6.6 ms the expert
    tier's 48 points cost when the cache comes up short."""
    assert handoff_cost_ms(decode_ple_point(WINDOW)) == pytest.approx(HANDOFF_IDLE_US / 1000.0)
    assert handoff_cost_ms([]) == 0.0
