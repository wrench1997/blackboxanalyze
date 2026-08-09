"""Workspace-scoped, hash-chained ledger for reproducible research runs."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_LEDGER_SCHEMA = "sift-experiment-ledger-v1"
_LOCK = threading.RLock()
_FORBIDDEN_KEYS = frozenset({
    "raw_body", "body_preview", "request_body", "password", "passwd", "secret", "token",
    "authorization", "cookie", "credential", "credentials", "session_cookie",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class ExperimentLedger:
    """Append-only ledger kept inside one workspace root."""

    def __init__(self, path: Path, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.path = Path(path).resolve()
        if not self.path.is_relative_to(self.workspace_root):
            raise ValueError("experiment ledger must stay inside workspace")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous_hash = "0" * 64
        if self.path.exists():
            verified = self.verify()
            self.previous_hash = verified["head"]

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("experiment ledger record must be an object")
        forbidden = sorted(set(_walk_keys(record)) & _FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError("experiment ledger contains forbidden raw/secret keys: " + ", ".join(forbidden))
        if not bool(record.get("local_only", True)):
            raise ValueError("experiment ledger requires local_only=true")
        if not str(record.get("protocol_id", "")):
            raise ValueError("experiment ledger protocol_id is required")
        envelope = {
            "schema_version": EXPERIMENT_LEDGER_SCHEMA,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "previous_hash": self.previous_hash,
            **json.loads(_canonical(record)),
        }
        envelope["record_hash"] = _sha256({key: value for key, value in envelope.items() if key != "record_hash"})
        line = _canonical(envelope)
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        self.previous_hash = envelope["record_hash"]
        return envelope

    def verify(self) -> dict[str, Any]:
        previous = "0" * 64
        count = 0
        if not self.path.exists():
            return {"valid": True, "record_count": 0, "head": previous}
        with _LOCK:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != EXPERIMENT_LEDGER_SCHEMA:
                raise ValueError("unsupported experiment ledger schema")
            if record.get("previous_hash") != previous:
                raise ValueError(f"experiment ledger previous hash mismatch at record {count + 1}")
            declared = str(record.get("record_hash", ""))
            body = dict(record)
            body.pop("record_hash", None)
            if declared != _sha256(body):
                raise ValueError(f"experiment ledger record hash mismatch at record {count + 1}")
            previous = declared
            count += 1
        return {"valid": True, "record_count": count, "head": previous}


__all__ = ["EXPERIMENT_LEDGER_SCHEMA", "ExperimentLedger"]
