"""The host point that drives the n-gram (PLE) tier, and the window it has to span.

**Why the host half cannot be inside the program.** The hash and the row gather are both host operations: the
hash because it is a 64-bit multiply and xor, which the device graph has no integer type for, and the gather
because the table is a file the host reads. A program is encoded once and replayed, so anything host-side placed
inside it runs ONCE, at encode time, and every later step replays that one result. The engine would then gather
the encode-time token's rows forever and produce a perfectly finite, perfectly stable, completely wrong answer
(``src/core/layer.cpp:1203-1211``). That failure is silent, which is why the split exists at all: the host runs
the hash and the gather between passes, and only the device half is encoded.

**The window is the whole point.** A token needs sixteen rows on sixteen different pages, and all sixteen depend
on the token itself, so there is nothing to prefetch until the token id is known. The only work available to
hide those reads behind is the embedding and layer 0 (``include/basalt/ngram/ple_reader.hpp:5-10``). So the
point publishes after the embedding op and waits at the PLE block, and the ops of ``[publish, wait)`` are layer
0's, which run on the GPU while the device reads are in flight. A point whose window is empty idles the GPU for
the whole gather.

**The PLE block runs at layer 1 and nowhere else** (``src/core/layer.cpp:1197-1201``). Its output is a residual
update applied before layer 1's own read, which is what the architecture means by the rows arriving "before
layer 2". So there is exactly one of these points per step, never one per layer, which is the difference from
the CPU expert tier's points in ``cpu_tier.py``.

**What carries the ordering.** basalt publishes its staging buffer with a sequentially consistent fence and a
store fence before raising a flag the device spins on, and the device fences after the wait
(``src/core/verify.cpp:2521-2523``; ``src/kernels/cuda/verify_kernels.cu:919-922``). Dropping either half is
the one substitution in this tier that can change produced values: the consumer could observe the flag before
some of the row data and read a wrong-but-finite vector. Here that ordering is carried by the host-point
machinery rather than by a hand-written fence: the task writes the staging buffer, the point returns, and only
then does the pump commit the command buffer that reads it. Naming the staging buffer in ``writes`` is what
makes ``Program.check_host_points`` refuse any op of the window that reads it, which is the structural form of
the same guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from monolith.runtime.program import HostPoint

# What one handoff costs the step in GPU idle, measured on the 48 GB 40-core M3 Max
# (docs/design/flash-next.md, probes/p16_host_handoff).
HANDOFF_IDLE_US = 136.5

# Sixteen rows per token, 90 bytes each: the traffic one token's gather moves.
PLE_N_HEADS = 16
PLE_ROW_BYTES = 90


@dataclass(frozen=True)
class PleWindow:
    """Where the gather's host point sits, as op boundaries of the step.

    ``publish`` goes after the embedding op, which is the earliest the token id is known, and ``wait`` goes at
    the PLE block in layer 1, which is the latest the rows can arrive. The ops between them are layer 0's, and
    they are what the device reads hide behind.
    """
    publish: int
    wait: int

    def __post_init__(self) -> None:
        if self.publish > self.wait:
            raise ValueError(f"a PLE window needs publish <= wait, got {self.publish} > {self.wait}")


def decode_ple_point(window: PleWindow, *, task: str = "ple_gather", staging: str = "ple_staging",
                     step_state: str = "step_state") -> List[HostPoint]:
    """The decode program's PLE host point, as a list so a program with no PLE gets an empty one.

    One point per step, not per layer: the block runs at layer 1 alone. ``staging`` is the buffer the gather
    writes the token's 2560 floats into, named in ``writes`` so ``Program.check_host_points`` refuses any op of
    the window that reads it; ``step_state`` is named in ``reads`` because the task reads the step's own state
    to learn the token.
    """
    return [HostPoint(task=task, publish=window.publish, wait=window.wait,
                      writes=[staging], reads=[step_state])]


def gather_bytes_per_token() -> int:
    """What one token's gather moves off the device, which is what makes this tier cheap: sixteen 90-byte rows
    is 1,440 bytes, against the experts' hundreds of megabytes (``include/basalt/core/layer.hpp:532-535``)."""
    return PLE_N_HEADS * PLE_ROW_BYTES


def handoff_cost_ms(points: Sequence[HostPoint]) -> float:
    """What installing these points costs the step in GPU idle, in milliseconds, at the measured median handoff.

    A floor rather than an estimate of the step: it counts the idle each handoff imposes and says nothing about
    the gather the task then does, which is the part the window is there to hide.
    """
    return len(points) * HANDOFF_IDLE_US / 1000.0
