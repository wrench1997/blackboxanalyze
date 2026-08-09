"""Read-only GPU contention check for long local training runs.

The gate is deliberately advisory: it never kills or reprioritizes another
application.  A training launcher can use the decision to defer a new run or
continue as a background job when a user's renderer is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ComputeProcess:
    pid: int
    name: str
    sm_percent: float
    mem_percent: float


def parse_pmon(text: str) -> list[ComputeProcess]:
    rows: list[ComputeProcess] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # nvidia-smi pmon: gpu pid type sm mem enc dec jpg ofa fb ccpm command
        if len(fields) < 12 or fields[2] not in {"C", "C+G"}:
            continue
        try:
            rows.append(ComputeProcess(pid=int(fields[1]), name=fields[-1], sm_percent=float(fields[3]), mem_percent=float(fields[4])))
        except (TypeError, ValueError):
            continue
    return rows


def foreign_compute(rows: Iterable[ComputeProcess], *, allowed_names: Iterable[str] = ("python.exe", "python", "torchrun.exe", "torchrun")) -> list[ComputeProcess]:
    allowed = {str(name).lower() for name in allowed_names}
    return [row for row in rows if row.name.lower() not in allowed]


def should_defer(rows: Iterable[ComputeProcess], *, threshold: float = 50.0) -> bool:
    return any(row.sm_percent >= float(threshold) for row in foreign_compute(rows))


__all__ = ["ComputeProcess", "foreign_compute", "parse_pmon", "should_defer"]
