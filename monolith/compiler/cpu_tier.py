"""The host points that drive the CPU expert tier, and the choice of whether to install any.

basalt compiles its decode graph in two forms and picks between them on one flag. With every expert resident it
runs the **zero-doorbell** graph: the whole step is on the GPU and the host services nothing. When the cache comes
up short it runs the **DecMoe** graph instead, whose window services every MoE layer, every window, whether or not
that layer has a CPU entry (``src/core/verify.cpp:876-907`` on ``all_resident_``).

The distinction is not a performance detail, which is why this module exists rather than a flag somewhere in a
builder. A port that installs a point in every MoE layer unconditionally is not a slow exact port: it is a
**different program** from the one basalt runs in the identity configuration, and it spends the price below of GPU
idle to be different.

**The price.** Each host handoff idles the GPU for a median of 136 to 137 microseconds as built, against 0.5
microseconds at a plain pass boundary (``docs/design/flash-next.md:616-622`` [M]). At 48 MoE layers that is 6.5 to
6.6 milliseconds per step, spent whether or not a layer has any CPU work to do. The decode budget it comes out of
is the whole reason the zero-doorbell form exists.

So: ``decode_host_points`` returns an empty list when every expert is resident, and one point per MoE layer when
the cache is short. The choice is made once, when the program is built from the pack and the profile, not by a
predicate the step evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from monolith.runtime.program import HostPoint

# What one handoff costs, measured on the 48 GB 40-core M3 Max (docs/design/flash-next.md:616-622, probes/p16).
HANDOFF_IDLE_US = 136.5


@dataclass(frozen=True)
class MoeLayerWindow:
    """Where one MoE layer's host point sits, as op boundaries of the step.

    ``publish`` goes after the layer's routing and planning op, so the tables the task reads are written before it
    starts; ``wait`` goes at the combine, so the layer's **resident** experts are the ops of ``[publish, wait)``
    that run on the GPU beside the pool. The window is what buys the handoff back: without ops in it the GPU idles
    for the whole of the task.
    """
    publish: int
    wait: int

    def __post_init__(self) -> None:
        if self.publish > self.wait:
            raise ValueError(f"a layer window needs publish <= wait, got {self.publish} > {self.wait}")


def decode_host_points(
    layers: Sequence[MoeLayerWindow],
    *,
    all_resident: bool,
    task: str = "cpu_experts",
    staging: str = "cpu_staging",
    step_state: str = "step_state",
) -> List[HostPoint]:
    """The decode program's host points, in program order.

    ``all_resident`` is basalt's own flag: true when the cache holds every expert, which is the configuration the
    identity tests run in. It selects between the two forms and nothing else does.

    ``staging`` is the buffer the pool writes its rows into. It is named in each point's ``writes`` so that
    ``Program.check_host_points`` refuses any op of the window that reads it; the step-state buffer is named in
    ``reads`` for the same reason, since the task reads the step's own state. ``step_state`` takes the program's
    own ``Program.step_state`` name rather than assuming the default, because a program that renamed it would get
    points declaring a buffer it does not have, which ``check_host_points`` rejects, or a buffer that is not the
    one its ops actually write.
    """
    if all_resident:
        return []                    # the zero-doorbell graph: the host services nothing
    return [
        HostPoint(task=task, publish=w.publish, wait=w.wait, writes=[staging], reads=[step_state])
        for w in layers
    ]


def handoff_cost_ms(points: Sequence[HostPoint]) -> float:
    """What installing these points costs the step in GPU idle, in milliseconds, at the measured median handoff.

    This is a floor and not an estimate of the step: it counts only the idle each handoff imposes, and says
    nothing about the CPU work the task then does. It exists so a builder can report the price of the form it
    chose rather than leave it implicit.
    """
    return len(points) * HANDOFF_IDLE_US / 1000.0
