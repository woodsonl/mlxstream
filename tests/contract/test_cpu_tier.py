"""``monolith/compiler/cpu_tier.py``: which decode program form the CPU expert tier gets, and what it costs.

basalt compiles its decode graph twice and picks on ``all_resident_``: the zero-doorbell graph, where the host
services nothing, or DecMoe, whose window services every MoE layer. The engineering review of the pool plan made
this a blocking finding, because a port that installs 48 points unconditionally is not a slow exact port but a
different program, one that spends 6.5 to 6.6 ms per step of GPU idle to be different.

These cases hold the choice itself, the shape of the points it produces, and the price it reports. They are
contract-tier: no GPU, no native module, nothing measured here.
"""

import pytest

from monolith.compiler.cpu_tier import HANDOFF_IDLE_US, MoeLayerWindow, decode_host_points, handoff_cost_ms
from monolith.runtime.program import HostPoint

# 48 MoE layers, each with its point after the routing op and its wait at the combine, as the Qwen3.8 decode
# program has them (docs/research/basalt-cpu-pool.md section 6.2).
LAYERS = [MoeLayerWindow(publish=5 * i + 2, wait=5 * i + 4) for i in range(48)]


def test_every_expert_resident_installs_no_points():
    """The zero-doorbell form. This is the configuration the identity tests run in, and the whole point of the
    finding: the host services nothing, so there is nothing to install."""
    assert decode_host_points(LAYERS, all_resident=True) == []


def test_a_short_cache_installs_one_point_per_layer():
    """The DecMoe form. Every MoE layer gets a point, whether or not that layer has a CPU entry, because that is
    what basalt's window does."""
    points = decode_host_points(LAYERS, all_resident=False)
    assert len(points) == len(LAYERS)
    assert all(isinstance(p, HostPoint) for p in points)
    assert [(p.publish, p.wait) for p in points] == [(w.publish, w.wait) for w in LAYERS]


def test_the_points_name_what_the_task_touches():
    """``writes`` and ``reads`` are what stop an op of the window racing the task: Program.check_host_points
    refuses any op of [publish, wait) that reads the staging the pool writes, or writes the step state it
    reads."""
    p = decode_host_points(LAYERS[:1], all_resident=False)[0]
    assert p.task == "cpu_experts"
    assert p.writes == ["cpu_staging"]
    assert p.reads == ["step_state"]


def test_the_names_are_the_caller_s():
    points = decode_host_points(LAYERS[:2], all_resident=False, task="experts", staging="staging")
    assert [p.task for p in points] == ["experts"] * 2
    assert points[0].writes == ["staging"]


def test_no_layers_gives_no_points_either_way():
    """A model with no MoE layer has nothing to service, so neither form installs anything."""
    assert decode_host_points([], all_resident=False) == []
    assert decode_host_points([], all_resident=True) == []


def test_a_window_cannot_run_backwards():
    with pytest.raises(ValueError, match="publish <= wait"):
        MoeLayerWindow(publish=7, wait=6)


def test_a_sync_point_is_a_window_of_no_ops():
    """publish == wait is allowed and means no GPU op runs beside the task: the handoff then idles the GPU for the
    whole of the task rather than overlapping it with the layer's resident experts."""
    w = MoeLayerWindow(publish=4, wait=4)
    assert decode_host_points([w], all_resident=False)[0].wait == 4


def test_the_price_of_the_form_is_reported():
    """48 handoffs at the measured median is the 6.5 to 6.6 ms per step the review priced, and it is why the
    zero-doorbell form exists. A builder reports this rather than leaving it implicit."""
    assert handoff_cost_ms(decode_host_points(LAYERS, all_resident=True)) == 0.0
    cost = handoff_cost_ms(decode_host_points(LAYERS, all_resident=False))
    assert 6.5 <= cost <= 6.6, f"48 points cost {cost:.3f} ms, not the measured 6.5 to 6.6"
    assert handoff_cost_ms([]) == 0.0
    assert handoff_cost_ms(decode_host_points(LAYERS[:1], all_resident=False)) == pytest.approx(HANDOFF_IDLE_US / 1000)


def test_the_points_pass_the_program_s_own_check():
    """The points this module builds have to be installable, not merely well shaped: publish boundaries never
    decrease, and no op of a window touches what the task does."""
    from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program

    layers = [MoeLayerWindow(publish=2 * i, wait=2 * i + 1) for i in range(3)]
    prog = Program(
        kernels={"k": KernelSpec(source="", function="k")},
        buffers={
            "step_state": BufferSpec(nbytes=256, role="step_state"),
            "cpu_staging": BufferSpec(nbytes=256, role="arena"),
            "x": BufferSpec(nbytes=256, role="arena"),
        },
        ops=[
            OpSpec(kernel="k", bindings=[(0, "x", 0)], grid=(1, 1, 1), threadgroup=(1, 1, 1), name=f"op{i}")
            for i in range(6)
        ],
    )
    prog.host_points = decode_host_points(layers, all_resident=False)
    prog.check_host_points()                      # raises if any point is ill formed or races its window
