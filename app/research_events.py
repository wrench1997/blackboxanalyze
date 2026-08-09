from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENT_FILE = PROJECT_ROOT / "artifacts" / "research-events.jsonl"
_LOCK = threading.RLock()


def emit_event(
    *,
    actor: str,
    tool: str,
    phase: str,
    status: str,
    message: str,
    payload: dict[str, Any] | None = None,
    artifact: str | None = None,
) -> dict[str, Any]:
    event = {
        "schema_version": "sift-research-event-v1",
        "id": f"{time.time_ns()}-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "tool": tool,
        "phase": phase,
        "status": status,
        "message": message,
        "payload": payload or {},
        "artifact": artifact,
    }
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _LOCK:
        EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return event


def list_events(limit: int = 100) -> list[dict[str, Any]]:
    if not EVENT_FILE.exists():
        return []
    with _LOCK:
        lines = EVENT_FILE.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-max(1, min(limit, 500)):]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
