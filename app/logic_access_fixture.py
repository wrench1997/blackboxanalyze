"""Safe local logic/access-control fixture for PG-PK-10.

The fixture is deliberately a *semantic* maze rather than an application
clone.  It accepts only read-only GET requests and maps inert query values to
bounded JSON response shapes.  No credentials, cookies, database, state
mutation, navigation, or external network are involved.  The collector keeps
only response shape and evaluator-side typed oracle evidence; raw bodies are
never persisted.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .maze_engine import sha256_json, validate_evidence


LOGIC_ACCESS_FIXTURE_SCHEMA = "sift-logic-access-fixture-v1"
LOGIC_ACCESS_SPEC_SCHEMA = "sift-logic-access-fixture-spec-v1"
LOGIC_ACCESS_ORACLES = {
    "access_control": "synthetic_authorization_boundary_v1",
    "logic_coupon": "synthetic_business_invariant_v1",
    "logic_replay": "synthetic_history_binding_v1",
}
LOGIC_ACCESS_BASE_URL = "http://127.0.0.1:8795"
LOGIC_ACCESS_FIXTURE_PORTS = (8795, 8796, 8797, 8798, 8799)
LOGIC_ACCESS_FIXTURE_BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in LOGIC_ACCESS_FIXTURE_PORTS)
LOGIC_ACCESS_SOURCE_PATH = Path(__file__).resolve()
_ROUTES = frozenset({"/gate", "/coupon", "/replay", "/health"})
_VARIANTS = frozenset({"alpha", "beta", "gamma", "delta", "epsilon"})


def logic_access_fixture_source_sha256() -> str:
    return hashlib.sha256(LOGIC_ACCESS_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode(value: str) -> str:
    # Encode even unreserved characters so the pair is a genuinely different
    # transport representation while parse_qs still recovers the same value.
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _query(path: str) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(path)
    return parsed.path, {str(key): str(values[0]) for key, values in parse_qs(parsed.query, keep_blank_values=True).items() if values}


def _int_value(value: str, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _render_payload(variant: str, *, outcome: str, protected: bool, marker: str = "") -> dict[str, Any]:
    """Return two semantically equivalent response schemas.

    The variant changes field names and lengths so the model cannot simply
    memorise one JSON template.  The oracle consumes the parsed payload before
    the raw body is discarded by the collector.
    """

    if variant in {"alpha", "gamma", "epsilon"}:
        payload: dict[str, Any] = {
            "status": "ok" if outcome in {"grant", "coupon", "commit"} else "denied",
            "resource": "protected" if protected else "public",
            "outcome": outcome,
        }
        if marker:
            payload["marker"] = marker
        return payload
    payload = {
        "ok": outcome in {"grant", "coupon", "commit"},
        "scope": "restricted" if protected else "ordinary",
        "decision": outcome,
    }
    if marker:
        payload["trace"] = marker
    return payload


def _evaluate_request(route: str, values: dict[str, str], variant: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Evaluate the hidden maze and return status, bounded body, typed oracle."""

    marker = values.get("marker", "")
    if route == "/gate":
        role = values.get("role", "guest")
        quota = values.get("quota", "0")
        is_admin = role == "admin"
        nonzero_quota = quota not in {"", "0"}
        # Intentional synthetic defect: a numeric quota is treated as a
        # truthy authorization bit.  Negative values are also truthy.
        granted = is_admin or nonzero_quota
        unexpected_grant = bool(granted and not is_admin and nonzero_quota)
        status = 200 if granted else 403
        body = _render_payload(variant, outcome="grant" if granted else "deny", protected=True, marker=marker)
        oracle = {
            "oracle_name": LOGIC_ACCESS_ORACLES["access_control"],
            "oracle_signal": "authorization_boundary_divergence",
            "positive": unexpected_grant,
            "typed": {
                "protected_resource": True,
                "non_admin_subject": bool(not is_admin),
                "quota_nonzero": nonzero_quota,
                "unexpected_grant": unexpected_grant,
            },
            "state_mutated": False,
        }
        return status, body, oracle

    if route == "/coupon":
        member = values.get("member", "0") == "1"
        total = _int_value(values.get("total", "0"))
        issued = bool(member and total > 100)
        expected_issued = bool(member and total >= 100)
        violation = bool(member and total == 100 and not issued)
        body = _render_payload(variant, outcome="coupon" if issued else "none", protected=False, marker=marker)
        # Keep the response shape generic, while the evaluator retains the
        # typed boolean needed to distinguish a boundary miss from a normal
        # non-member or below-threshold result.
        body["flag"] = bool(issued)
        return 200, body, {
            "oracle_name": LOGIC_ACCESS_ORACLES["logic_coupon"],
            "oracle_signal": "business_boundary_mismatch",
            "positive": violation,
            "typed": {
                "member": member,
                "boundary_hit": total == 100,
                "expected_issued": expected_issued,
                "observed_issued": issued,
                "invariant_violation": violation,
            },
            "state_mutated": False,
        }

    if route == "/replay":
        action = values.get("action", "wait")
        previous = values.get("previous", "none")
        challenge = values.get("challenge", "old")
        current = values.get("current", "new")
        accepted = bool(action == "commit" and previous == "verify")
        expected = bool(action == "commit" and previous == "verify" and challenge == current)
        violation = bool(accepted and not expected)
        status = 200 if accepted else 403
        body = _render_payload(variant, outcome="commit" if accepted else "deny", protected=True, marker=marker)
        return status, body, {
            "oracle_name": LOGIC_ACCESS_ORACLES["logic_replay"],
            "oracle_signal": "history_binding_mismatch",
            "positive": violation,
            "typed": {
                "commit_action": action == "commit",
                "previous_verified": previous == "verify",
                "challenge_matches": challenge == current,
                "unexpected_replay_accept": violation,
            },
            "state_mutated": False,
        }

    # Ordinary 200 control: a successful response by itself is never an exit.
    body = _render_payload(variant, outcome="ordinary", protected=False, marker=marker)
    return 200, body, {
        "oracle_name": "synthetic_ordinary_control_v1",
        "oracle_signal": "no_typed_violation",
        "positive": False,
        "typed": {"ordinary_response": True, "invariant_violation": False},
        "state_mutated": False,
    }


class _LogicAccessHandler(BaseHTTPRequestHandler):
    server_version = "sift-logic-access-fixture/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        route = parsed.path
        values = {str(key): str(value[0]) for key, value in parse_qs(parsed.query, keep_blank_values=True).items() if value}
        variant = str(getattr(self.server, "fixture_variant", "alpha"))
        if route not in _ROUTES or variant not in _VARIANTS:
            status, payload = 404, {"status": "not_found"}
        else:
            status, payload, _ = _evaluate_request(route, values, variant)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class LogicAccessFixtureServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in _VARIANTS:
            raise ValueError("unknown logic/access fixture variant")
        super().__init__(address, _LogicAccessHandler)
        self.fixture_variant = variant


def make_logic_access_fixture_server(*, port: int = 8795, variant: str = "alpha") -> LogicAccessFixtureServer:
    if int(port) not in LOGIC_ACCESS_FIXTURE_PORTS:
        raise ValueError("logic/access fixture port is not allow-listed")
    return LogicAccessFixtureServer(("127.0.0.1", int(port)), variant)


def _validate_path(path: str) -> tuple[str, dict[str, str]]:
    route, values = _query(path)
    if route not in _ROUTES:
        raise ValueError("logic/access fixture route is not allow-listed")
    if route == "/gate" and set(values) - {"role", "quota", "marker"}:
        raise ValueError("gate query contains an unknown field")
    if route == "/coupon" and set(values) - {"member", "total", "marker"}:
        raise ValueError("coupon query contains an unknown field")
    if route == "/replay" and set(values) - {"action", "previous", "challenge", "current", "marker"}:
        raise ValueError("replay query contains an unknown field")
    if route == "/health" and set(values) - {"ok", "marker"}:
        raise ValueError("health query contains an unknown field")
    return route, values


def validate_logic_access_fixture_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("logic/access fixture spec must be an object")
    target = str(spec.get("target", LOGIC_ACCESS_BASE_URL)).rstrip("/")
    if target not in LOGIC_ACCESS_FIXTURE_BASE_URLS:
        raise ValueError("logic/access fixture target must be an allow-listed loopback URL")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("logic/access fixture permits only read-only GET")
    path = str(spec.get("path", "/health?ok=1"))
    route, values = _validate_path(path)
    if not str(spec.get("source_id", "")) or not str(spec.get("lab_id", "")):
        raise ValueError("logic/access fixture source_id and lab_id are required")
    marker = str(spec.get("marker", "logic-pg10-marker"))
    pair = dict(spec.get("pair") or {})
    if pair:
        if not str(pair.get("pair_id", "")) or str(pair.get("variant")) not in {"plain", "url_percent"}:
            raise ValueError("logic/access fixture pair metadata is invalid")
        if not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2:
            raise ValueError("logic/access fixture encoding_depth is invalid")
    # Probe text is an inert catalog identifier, not an exploit string.
    probe = str(spec.get("probe", marker))
    payload = build_detection_payload(
        target=target,
        method="GET",
        path=path,
        marker=marker,
        probe=probe,
        probe_kind="http_canary",
        expected={},
    )
    return {
        "schema_version": LOGIC_ACCESS_SPEC_SCHEMA,
        "target": target,
        "method": "GET",
        "path": path,
        "route": route,
        "query": values,
        "marker": marker,
        "encoding": str(spec.get("encoding", "plain")),
        "source_id": str(spec["source_id"]),
        "lab_id": str(spec["lab_id"]),
        "probe": probe,
        "family_hint": str(spec.get("family_hint", "")),
        "expected_signal": str(spec.get("expected_signal", "typed_boundary")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def _paired_path(route: str, values: dict[str, str], *, encoded: bool) -> str:
    parts = []
    for key, value in values.items():
        rendered = _percent_encode(value) if encoded else value
        parts.append(f"{key}={rendered}")
    return route + ("?" + "&".join(parts) if parts else "")


def _pair_specs(
    *,
    dataset_id: str,
    target: str,
    marker: str,
    pair_id: str,
    route: str,
    values: dict[str, str],
    family_hint: str,
    signal: str,
    index: int,
) -> list[dict[str, Any]]:
    plain = {
        "source_id": dataset_id,
        "lab_id": f"{pair_id}-plain",
        "target": target,
        "path": _paired_path(route, values, encoded=False),
        "marker": marker,
        "probe": f"pg10-probe-{index:02d}",
        "encoding": "plain",
        "family_hint": family_hint,
        "expected_signal": signal,
        "pair": {"pair_id": pair_id, "variant": "plain", "encoding_depth": 0},
    }
    encoded = dict(plain)
    encoded.update({
        "lab_id": f"{pair_id}-url-percent",
        "path": _paired_path(route, values, encoded=True),
        "encoding": "url_percent",
        "pair": {"pair_id": pair_id, "variant": "url_percent", "encoding_depth": 1},
    })
    return [plain, encoded]


def default_logic_access_fixture_specs(
    *,
    dataset_id: str = "fixture-pg10",
    target: str = LOGIC_ACCESS_BASE_URL,
    marker: str = "logic-pg10-marker",
) -> list[dict[str, Any]]:
    """Return paired positives plus semantically close counterfactuals."""

    specs: list[dict[str, Any]] = []
    definitions = [
        ("gate-truthy", "/gate", {"role": "member", "quota": "1"}, "access_control", "authorization_boundary_divergence"),
        ("gate-negative", "/gate", {"role": "member", "quota": "-1"}, "access_control", "authorization_boundary_divergence"),
        ("gate-denied-control", "/gate", {"role": "member", "quota": "0"}, "control", "no_typed_violation"),
        ("gate-admin-control", "/gate", {"role": "admin", "quota": "0"}, "control", "no_typed_violation"),
        ("coupon-boundary", "/coupon", {"member": "1", "total": "100"}, "logic", "business_boundary_mismatch"),
        ("coupon-below-control", "/coupon", {"member": "1", "total": "99"}, "control", "no_typed_violation"),
        ("coupon-nonmember-control", "/coupon", {"member": "0", "total": "100"}, "control", "no_typed_violation"),
        ("replay-mismatch", "/replay", {"action": "commit", "previous": "verify", "challenge": "old", "current": "new"}, "logic", "history_binding_mismatch"),
        ("replay-bound-control", "/replay", {"action": "commit", "previous": "verify", "challenge": "new", "current": "new"}, "control", "no_typed_violation"),
        ("ordinary-200-control", "/health", {"ok": "1"}, "control", "no_typed_violation"),
    ]
    for index, (pair_id, route, values, family_hint, signal) in enumerate(definitions, start=1):
        specs.extend(_pair_specs(
            dataset_id=dataset_id,
            target=target,
            marker=marker,
            pair_id=f"logic-pg10-{pair_id}",
            route=route,
            values=values,
            family_hint=family_hint,
            signal=signal,
            index=index,
        ))
    return specs


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError):
        value = None
    return {
        "status_code": int(response.status_code),
        "headers": {"content-type": str(response.headers.get("content-type", ""))},
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "json_shape": {
            "type": "object" if isinstance(value, dict) else "other",
            "key_count": len(value) if isinstance(value, dict) else 0,
            "scalar_count": sum(not isinstance(item, (dict, list)) for item in value.values()) if isinstance(value, dict) else 0,
        },
    }


class LogicAccessCollector:
    def __init__(self, *, base_url: str = LOGIC_ACCESS_BASE_URL, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in LOGIC_ACCESS_FIXTURE_BASE_URLS:
            raise ValueError("logic/access collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or logic_access_fixture_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_logic_access_fixture_spec(raw_spec)
        route, values = _query(spec["path"])
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/health?ok=1")
            response = await client.get(spec["path"], headers=spec["payload"]["headers"])
        baseline_summary = _summary(baseline)
        response_summary = _summary(response)
        variant = "alpha"
        # The style is intentionally not trusted as evaluator input.  The
        # fixture response itself is enough for the typed oracle; the server
        # owns its variant and only emits the bounded response shape.
        try:
            server = getattr(getattr(response, "extensions", {}), "get", lambda *_: None)("fixture_variant")
            if server:
                variant = str(server)
        except Exception:
            pass
        # Recompute from the allow-listed query semantics; this is the
        # evaluator-side oracle and is never fed to the model projection.
        _, body, oracle = _evaluate_request(route, values, "alpha")
        # Response body parsing is local and bounded.  The oracle booleans are
        # based on input + typed response semantics, not raw body persistence.
        try:
            response_body = response.json()
        except (ValueError, json.JSONDecodeError):
            response_body = {}
        if route == "/coupon":
            oracle["typed"]["observed_issued"] = bool(response_body.get("flag", False)) if isinstance(response_body, dict) else False
            oracle["typed"]["invariant_violation"] = bool(oracle["typed"].get("member") and oracle["typed"].get("boundary_hit") and not oracle["typed"].get("observed_issued"))
            oracle["positive"] = bool(oracle["typed"]["invariant_violation"])
        oracle["response_status_matches"] = int(response.status_code) == (200 if route != "/gate" or values.get("role") == "admin" or values.get("quota") not in {"", "0"} else 403)
        reset = {
            "kind": "ephemeral_in_repo_logic_access_fixture",
            "fresh": True,
            "fresh_target": True,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "target_instance_id": self.target_instance_id,
            "fixture_source_sha256": self.source_hash,
        }
        envelope = {
            "collector": LOGIC_ACCESS_FIXTURE_SCHEMA,
            "target": self.base_url,
            "path": spec["path"],
            "method": "GET",
            "reset": reset,
            "baseline": baseline_summary,
            "response": response_summary,
            "oracle_projection": oracle,
            "local_http_loopback": True,
            "script_execution": False,
            "network_access": False,
            "navigation": False,
            "database_touched": False,
            "real_sleep_performed": False,
            "credentials_accessed": False,
            "encoding": spec["encoding"],
            "payload_sha256": spec["payload"]["payload_sha256"],
        }
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        positive = bool(oracle.get("positive"))
        record = {
            "schema_version": LOGIC_ACCESS_FIXTURE_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": spec.get("family_hint", "control"),
            "payload": spec["payload"],
            "probe_artifact": {
                "original": spec["path"],
                "encoding": spec["encoding"],
                "probe_sha256": hashlib.sha256(spec["path"].encode()).hexdigest(),
            },
            "semantic": {
                "family": spec.get("family_hint", "control"),
                "surface": "synthetic_logic_access_http",
                "expected_oracle": oracle["oracle_name"],
                "expected_signal": spec["expected_signal"],
            },
            "evaluator_state_visible": False,
            "replay": {
                "target": self.base_url,
                "method": "GET",
                "path": spec["path"],
                "fresh_reset": reset,
                "transport": "httpx_loopback",
            },
            "response_projection": response_summary,
            "oracle_projection": oracle,
            "evidence": checked["body"],
            "rule_ir_result": positive,
            "candidate_status": "typed_boundary_candidate" if positive else "clean_observation",
            "safety": {
                "local_only": True,
                "read_only": True,
                "fresh_reset": True,
                "fresh_target": True,
                "external_network": False,
                "script_execution": False,
                "database_touched": False,
                "real_sleep_performed": False,
                "raw_body_stored": False,
                "credentials_stored": False,
                "state_mutated": False,
            },
        }
        if spec.get("pair"):
            record["pair"] = dict(spec["pair"])
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in specs:
            rows.append(await self.collect(spec))
        return rows


__all__ = [
    "LOGIC_ACCESS_BASE_URL",
    "LOGIC_ACCESS_FIXTURE_BASE_URLS",
    "LOGIC_ACCESS_FIXTURE_PORTS",
    "LOGIC_ACCESS_ORACLES",
    "LogicAccessCollector",
    "default_logic_access_fixture_specs",
    "logic_access_fixture_source_sha256",
    "make_logic_access_fixture_server",
    "validate_logic_access_fixture_spec",
]
