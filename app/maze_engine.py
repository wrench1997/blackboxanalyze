"""Versioned, replayable engineering runner for rule-maze episodes.

This layer does not choose probes or infer vulnerability labels.  It owns the
boring but essential lifecycle: safety validation, sanitized step records,
hash-chained evidence, graph snapshots, and a manifest that can be replayed by
the API or a CI job.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .juice_shop_adapter import EvidenceLedger
from .detection_payload import validate_detection_payload
from .maze_solver import MazeGraph, action_identity


MAZE_RUN_SCHEMA = "sift-maze-run-v1"
MAZE_EVIDENCE_SCHEMA = "sift-maze-evidence-v1"
DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "maze-runs"
FORBIDDEN_EVIDENCE_KEYS = frozenset({
    "body_preview",
    "raw_body",
    "request_body",
    "password",
    "secret",
    "token",
    "authorization",
    "cookie",
})
SAFETY_FLAGS = (
    "script_execution",
    "network_access",
    "navigation",
    "database_touched",
    "real_sleep_performed",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def validate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate a bounded evidence object before it enters a run artifact."""

    if not isinstance(evidence, dict):
        raise ValueError("maze evidence must be an object")
    forbidden = sorted(set(_walk_keys(evidence)) & FORBIDDEN_EVIDENCE_KEYS)
    if forbidden:
        raise ValueError(f"maze evidence contains forbidden secret/raw keys: {', '.join(forbidden)}")
    unsafe = [flag for flag in SAFETY_FLAGS if bool(evidence.get(flag, False))]
    if unsafe:
        raise ValueError("maze safety invariant failed: " + ", ".join(unsafe))
    declared_hash = evidence.get("evidence_hash")
    if declared_hash:
        without_hash = dict(evidence)
        without_hash.pop("evidence_hash", None)
        expected = sha256_json(without_hash)
        # Browser fallback hashes are intentionally accepted as a separate
        # non-cryptographic mode; Python oracles use canonical SHA-256.
        if evidence.get("evidence_hash_algorithm") != "non-cryptographic-fallback" and declared_hash != expected:
            raise ValueError("maze evidence hash mismatch")
    return {
        "schema_version": MAZE_EVIDENCE_SCHEMA,
        "evidence_hash": str(declared_hash or sha256_json(evidence)),
        "evidence_hash_algorithm": evidence.get("evidence_hash_algorithm", "sha256-canonical-json"),
        "safety_flags": {flag: bool(evidence.get(flag, False)) for flag in SAFETY_FLAGS},
        "body": json.loads(canonical_json(evidence)),
    }


def verify_ledger(path: Path) -> dict[str, Any]:
    """Verify an EvidenceLedger hash chain without mutating it."""

    if not path.exists():
        return {"valid": True, "record_count": 0, "head": "0" * 64}
    previous = "0" * 64
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_hash = str(record.get("record_hash", ""))
        if record.get("previous_hash") != previous:
            raise ValueError(f"maze ledger previous hash mismatch at record {count + 1}")
        envelope = dict(record)
        envelope.pop("record_hash", None)
        expected = hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
        if record_hash != expected:
            raise ValueError(f"maze ledger record hash mismatch at record {count + 1}")
        previous = record_hash
        count += 1
    return {"valid": True, "record_count": count, "head": previous}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


class MazeRunRecorder:
    """Record one deterministic run without exposing evaluator state."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        artifact_root: Path | None = None,
        run_id: str | None = None,
        protocol_id: str = "sift-rule-maze-loop-1",
        target_kind: str = "synthetic_local",
        seed: int | None = None,
    ) -> None:
        self.workspace_root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
        self.artifact_root = (artifact_root or self.workspace_root / "artifacts" / "maze-runs").resolve()
        if not self.artifact_root.is_relative_to(self.workspace_root):
            raise ValueError("maze artifact root must stay inside workspace")
        self.run_id = run_id or f"maze-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        self.run_dir = self.artifact_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.ledger_path = self.run_dir / "evidence.jsonl"
        self.graph_path = self.run_dir / "graph.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.ledger = EvidenceLedger(self.ledger_path, self.workspace_root)
        self.graph = MazeGraph()
        self.steps: list[dict[str, Any]] = []
        self.status = "running"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.protocol_id = protocol_id
        self.target_kind = target_kind
        self.seed = seed

    def record_transition(
        self,
        before: dict[str, Any] | None,
        action: dict[str, Any],
        after: dict[str, Any] | None,
        *,
        evidence: dict[str, Any] | None = None,
        goal: dict[str, Any] | None = None,
        detection_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.status != "running":
            raise RuntimeError("maze run is already finalized")
        checked_evidence = validate_evidence(evidence) if evidence is not None else None
        checked_payload = validate_detection_payload(detection_payload) if detection_payload is not None else None
        transition = self.graph.record_transition(
            before,
            action,
            after,
            goal_status=str(goal.get("status")) if isinstance(goal, dict) and goal.get("status") else None,
        )
        record = {
            "schema_version": MAZE_RUN_SCHEMA,
            "step": len(self.steps) + 1,
            "transition": transition,
            "action": action_identity(action),
            "goal": goal,
            "evidence": checked_evidence,
            "detection_payload": checked_payload,
        }
        stored = self.ledger.append(record)
        step = {
            "step": record["step"],
            "record_hash": stored["record_hash"],
            "edge": transition["edge"],
            "kind": transition["kind"],
            "goal_status": transition["goal_status"],
            "evidence_hash": checked_evidence["evidence_hash"] if checked_evidence else None,
            "payload_sha256": checked_payload["payload_sha256"] if checked_payload else None,
        }
        self.steps.append(step)
        return step

    def mark_reset(self, *, reset_kind: str = "fresh", evaluator_state_hidden: bool = True) -> None:
        if self.steps:
            raise RuntimeError("reset metadata must be recorded before the first maze step")
        self.reset = {
            "kind": str(reset_kind),
            "evaluator_state_hidden": bool(evaluator_state_hidden),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

    def finalize(self, *, status: str = "complete", notes: list[str] | None = None) -> dict[str, Any]:
        if self.status != "running":
            raise RuntimeError("maze run is already finalized")
        self.status = str(status)
        _atomic_write_json(self.graph_path, self.graph.to_dict())
        ledger = verify_ledger(self.ledger_path)
        graph_body = json.loads(self.graph_path.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": MAZE_RUN_SCHEMA,
            "run_id": self.run_id,
            "protocol_id": self.protocol_id,
            "target_kind": self.target_kind,
            "seed": self.seed,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "reset": getattr(self, "reset", {"kind": "unspecified", "evaluator_state_hidden": True}),
            "safety": {
                "local_only": True,
                "script_execution": False,
                "network_access": False,
                "navigation": False,
                "database_touched": False,
                "real_sleep_performed": False,
            },
            "step_count": len(self.steps),
            "steps": list(self.steps),
            "artifacts": {
                "ledger": str(self.ledger_path.relative_to(self.workspace_root)),
                "ledger_head": ledger["head"],
                "graph": str(self.graph_path.relative_to(self.workspace_root)),
                "graph_sha256": sha256_json(graph_body),
            },
            "notes": list(notes or []),
        }
        _atomic_write_json(self.manifest_path, manifest)
        return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MAZE_RUN_SCHEMA:
        raise ValueError("unsupported maze run manifest schema")
    if not isinstance(manifest.get("steps"), list):
        raise ValueError("maze run manifest steps must be a list")
    return manifest


def latest_manifest(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path | None:
    if not artifact_root.exists():
        return None
    manifests = [path for path in artifact_root.glob("*/manifest.json") if path.is_file()]
    return max(manifests, key=lambda path: path.stat().st_mtime) if manifests else None
