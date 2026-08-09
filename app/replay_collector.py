"""Local replay collector for safe detection probes.

The collector talks to the application's read-only replay adapters through an
ASGI transport whose base URL is still fixed to ``127.0.0.1:3100``.  It
captures bounded response projections and hashes, never raw bodies, cookies,
credentials, or evaluator-only fields.  Each run gets a fresh reset record
before the probe is sent.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable
from urllib.parse import urlsplit

import httpx

from .detection_payload import ALLOWED_PROBE_KINDS, build_detection_payload
from .maze_engine import sha256_json, validate_evidence
from .rule_ir import canonical as canonical_rule_ir, complexity as rule_ir_complexity, evaluate as evaluate_rule_ir


COLLECTOR_SCHEMA = "sift-local-replay-collector-v1"
LOCAL_BASE_URL = "http://127.0.0.1:3100"
ALLOWED_REPLAY_PATHS = frozenset({
    "/api/health",
    "/api/maze/labs",
    "/api/maze/replay/dom",
    "/api/maze/replay/sql",
})
FORBIDDEN_RESPONSE_KEYS = frozenset({
    "body",
    "raw_body",
    "body_preview",
    "cookie",
    "set-cookie",
    "authorization",
    "password",
    "secret",
    "token",
})
ORACLE_PROJECTION_KEYS = frozenset({
    "oracle",
    "sink",
    "transforms",
    "candidate_signal",
    "browser_sink_observed",
    "dom_change",
    "marker_hits",
    "tag_shape",
    "script_like_present",
    "controlled_differential",
    "interpreter_boundary",
    "modality",
    "syntax_error_observed",
    "error_differential",
    "blind_boolean_differential",
    "row_shape_differential",
    "timing_differential",
    "simulated_latency_ms",
    "timeout_budget_ms",
    "timeout_observed",
    "real_sleep_performed",
    "local_callback_observed",
    "local_callback_only",
    "external_network",
    "database_touched",
    "network_access",
    "navigation",
    "script_execution",
    "evidence_hash",
})


def _assert_local_target(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port not in {None, 3100}:
        raise ValueError("replay collector target must be exactly http://127.0.0.1:3100")
    if parsed.username or parsed.password or parsed.path not in {"", "/"}:
        raise ValueError("replay collector target may not contain credentials or a path")
    return LOCAL_BASE_URL


def _shape(value: Any, *, depth: int = 0) -> Any:
    """Produce a bounded JSON shape without retaining values."""

    if depth >= 3:
        return "leaf"
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": sorted(str(key) for key in value.keys())[:64],
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "item_shapes": [_shape(item, depth=depth + 1) for item in value[:4]],
        }
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _response_projection(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    # The SQL replay adapter wraps its bounded evidence under ``evidence``;
    # the DOM adapter returns the evidence object directly.  Never retain the
    # AST or any other unbounded response field.
    source = body.get("evidence") if isinstance(body.get("evidence"), dict) else body
    projection: dict[str, Any] = {}
    for key, value in source.items():
        if str(key) not in ORACLE_PROJECTION_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            projection[str(key)] = value
        elif isinstance(value, list):
            # Tag shapes/transforms are bounded structural arrays, not raw
            # response content.
            projection[str(key)] = [str(item)[:64] for item in value[:32]]
    return projection


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = set(str(key).casefold() for key in value) & FORBIDDEN_RESPONSE_KEYS
        if forbidden:
            raise ValueError(f"replay response contains forbidden keys: {', '.join(sorted(forbidden))}")
        for item in value.values():
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def _rule_ir_for(spec: dict[str, Any]) -> dict[str, Any]:
    path = str(spec["path"])
    if path == "/api/maze/replay/dom":
        return {
            "op": "and",
            "args": [
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.browser_sink_observed"}, "right": {"op": "const", "value": True}},
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.dom_change"}, "right": {"op": "const", "value": True}},
            ],
        }
    if path == "/api/maze/replay/sql":
        return {
            "op": "and",
            "args": [
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.controlled_differential"}, "right": {"op": "const", "value": True}},
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.interpreter_boundary"}, "right": {"op": "const", "value": True}},
            ],
        }
    return {
        "op": "eq",
        "left": {"op": "field", "path": "response.status_code"},
        "right": {"op": "const", "value": 200},
    }


def validate_replay_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("replay spec must be an object")
    target = _assert_local_target(str(spec.get("target", LOCAL_BASE_URL)))
    method = str(spec.get("method", "GET")).upper()
    if method != "GET":
        raise ValueError("local replay collector permits only GET")
    path = str(spec.get("path", ""))
    if path not in ALLOWED_REPLAY_PATHS or "://" in path or path.startswith("//"):
        raise ValueError("replay path is not an allow-listed read-only local adapter")
    params = dict(spec.get("params") or {})
    if len(params) > 8:
        raise ValueError("replay query parameter count exceeds the local bound")
    for key, value in params.items():
        if len(str(key)) > 64 or len(str(value)) > 2048:
            raise ValueError("replay query parameter exceeds the local bound")
        if str(key).casefold() in {"cookie", "authorization", "token", "password"}:
            raise ValueError("replay query may not contain credentials")
    if any(part in path.casefold() for part in ("/api/challenges", "/snippets")):
        raise ValueError("evaluator-only replay paths are not permitted")
    probe_kind = str(spec.get("probe_kind", "http_canary"))
    if probe_kind not in ALLOWED_PROBE_KINDS:
        raise ValueError("unknown replay probe kind")
    marker = str(spec.get("marker", "sift-replay"))
    probe = str(spec.get("probe", marker))
    payload = build_detection_payload(
        target=target,
        method="GET",
        path=path,
        marker=marker,
        probe_kind=probe_kind,
        probe=probe,
        expected=dict(spec.get("expected") or {}),
    )
    family = str(spec.get("family", ""))
    if family not in {"xss", "injection", "access_control", "url_redirect", "logic"}:
        raise ValueError("replay spec family is not supported")
    source_id = str(spec.get("source_id", ""))
    if not source_id or len(source_id) > 96:
        raise ValueError("replay source_id is required")
    return {
        "schema_version": "sift-local-replay-spec-v1",
        "target": target,
        "method": method,
        "path": path,
        "params": json.loads(json.dumps(params, ensure_ascii=False)),
        "marker": marker,
        "probe_kind": probe_kind,
        "probe": probe,
        "encoding": str(spec.get("encoding", "validated_probe")),
        "expected": json.loads(json.dumps(dict(spec.get("expected") or {}), ensure_ascii=False)),
        "family": family,
        "source_id": source_id,
        "lab_id": str(spec.get("lab_id", "local-replay")),
        "surface": str(spec.get("surface", path.strip("/").replace("/", "_"))),
        "expected_oracle": str(spec.get("expected_oracle", "synthetic_rule_surface_v1")),
        "expected_signal": str(spec.get("expected_signal", "status_code_200")),
        "payload": payload,
    }


class LocalReplayCollector:
    """Collect bounded replay records from the local FastAPI application."""

    def __init__(self, application: Any, *, base_url: str = LOCAL_BASE_URL, timeout_seconds: float = 3.0) -> None:
        self.base_url = _assert_local_target(base_url)
        self.application = application
        self.timeout_seconds = min(max(float(timeout_seconds), 0.1), 5.0)
        self.reset_count = 0

    def _fresh_reset(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.reset_count += 1
        reset_id = hashlib.sha256(
            f"{spec['source_id']}:{spec['lab_id']}:{self.reset_count}".encode("utf-8")
        ).hexdigest()[:20]
        return {
            "kind": "fresh_local_asgi",
            "reset_id": reset_id,
            "sequence": self.reset_count,
            "evaluator_state_hidden": True,
            "external_network": False,
        }

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_replay_spec(raw_spec)
        reset = self._fresh_reset(spec)
        transport = httpx.ASGITransport(app=self.application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            follow_redirects=False,
        ) as client:
            baseline = await client.get("/api/health", headers={"accept": "application/json"})
            response = await client.get(
                spec["path"],
                params=spec["params"],
                headers=spec["payload"]["headers"],
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        _assert_no_forbidden_keys(body)
        projection = _response_projection(body)
        body_bytes = response.content
        response_summary = {
            "status_code": int(response.status_code),
            "headers": {
                key.casefold(): str(response.headers[key])
                for key in ("content-type", "content-length")
                if key in response.headers
            },
            "body_length": len(body_bytes),
            "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
            "json_shape": _shape(body),
        }
        baseline_summary = {
            "status_code": int(baseline.status_code),
            "body_length": len(baseline.content),
            "body_sha256": hashlib.sha256(baseline.content).hexdigest(),
        }
        envelope = {
            "collector": COLLECTOR_SCHEMA,
            "target": self.base_url,
            "path": spec["path"],
            "method": "GET",
            "reset": reset,
            "baseline": baseline_summary,
            "response": response_summary,
            "oracle_projection": projection,
            "script_execution": False,
            "network_access": False,
            "navigation": False,
            "database_touched": False,
            "real_sleep_performed": False,
            "credentials_accessed": False,
        }
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        rule_ir = _rule_ir_for(spec)
        rule_ir_canonical = canonical_rule_ir(rule_ir)
        rule_ir_result = evaluate_rule_ir(rule_ir, {
            "response": response_summary,
            "oracle_projection": projection,
        })
        return {
            "schema_version": COLLECTOR_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": spec["family"],
            "payload": spec["payload"],
            "probe_artifact": {
                "original": spec["probe"],
                "encoding": str(spec.get("encoding", "validated_probe")),
                "probe_sha256": hashlib.sha256(spec["probe"].encode("utf-8")).hexdigest(),
            },
            "semantic": {
                "family": spec["family"],
                "surface": spec["surface"],
                "expected_oracle": spec["expected_oracle"],
                "expected_signal": spec["expected_signal"],
            },
            "evaluator_state_visible": False,
            "replay": {
                "target": self.base_url,
                "method": "GET",
                "path": spec["path"],
                "params": spec["params"],
                "fresh_reset": reset,
                "transport": "in_process_asgi",
            },
            "response_projection": response_summary,
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir": rule_ir,
            "rule_ir_canonical": rule_ir_canonical,
            "rule_ir_complexity": rule_ir_complexity(rule_ir),
            "rule_ir_result": bool(rule_ir_result),
            "safety": {
                "local_only": True,
                "fresh_reset": True,
                "external_network": False,
                "script_execution": False,
                "database_touched": False,
                "real_sleep_performed": False,
                "raw_body_stored": False,
                "credentials_stored": False,
            },
        }

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in specs:
            records.append(await self.collect(spec))
        return records
