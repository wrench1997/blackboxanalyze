from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, scenario: dict[str, Any]) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "observations": [],
            "candidates": [],
            "query_count": 0,
            "closure_history": [],
            "last_closure_report": None,
        }
        with self._lock:
            self._sessions[session_id] = session
        return deepcopy(session)

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._sessions.get(session_id)
            return deepcopy(value) if value else None

    def mutate(self, session_id: str, callback):
        with self._lock:
            if session_id not in self._sessions:
                return None
            result = callback(self._sessions[session_id])
            return deepcopy(result if result is not None else self._sessions[session_id])

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
