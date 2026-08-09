"""Third independent logic/access maze source for durable Rule IR promotion."""

from __future__ import annotations

import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .logic_access_fixture import LOGIC_ACCESS_ORACLES
from .maze_engine import sha256_json, validate_evidence


LOGIC_ACCESS_V3_SCHEMA = "sift-logic-access-fixture-v3"
LOGIC_ACCESS_V3_SPEC_SCHEMA = "sift-logic-access-fixture-spec-v3"
LOGIC_ACCESS_V3_BASE_URL = "http://127.0.0.1:8815"
LOGIC_ACCESS_V3_PORTS = (8815, 8816, 8817)
LOGIC_ACCESS_V3_BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in LOGIC_ACCESS_V3_PORTS)
LOGIC_ACCESS_V3_SOURCE_PATH = Path(__file__).resolve()
_ROUTES = frozenset({"/permit", "/benefit", "/finalize", "/heartbeat"})
_VARIANTS = frozenset({"red", "blue", "green"})
_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def logic_access_v3_source_sha256() -> str:
    return hashlib.sha256(LOGIC_ACCESS_V3_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _query(path: str) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(path)
    return parsed.path, {str(key): str(values[0]) for key, values in parse_qs(parsed.query, keep_blank_values=True).items() if values}


def _int_value(value: str, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _render(variant: str, *, outcome: str, protected: bool, marker: str) -> dict[str, Any]:
    if variant == "red":
        payload: dict[str, Any] = {"decision": outcome, "zone": "private" if protected else "open", "ticket": len(outcome)}
    elif variant == "blue":
        payload = {"granted": outcome in {"grant", "coupon", "commit"}, "scope": "sealed" if protected else "common", "phase": outcome}
    else:
        payload = {"ok": outcome in {"grant", "coupon", "commit"}, "area": "member" if protected else "public", "result": outcome}
    if marker:
        payload["trace"] = marker
    return payload


def _evaluate(route: str, values: dict[str, str], variant: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
    marker = values.get("marker", "")
    if route == "/permit":
        actor = values.get("actor", "visitor")
        credit = values.get("credit", "0")
        owner = actor == "owner"
        credit_nonzero = credit not in {"", "0"}
        granted = owner or credit_nonzero
        unexpected = bool(granted and not owner and credit_nonzero)
        status = 200 if granted else 403
        return status, _render(variant, outcome="grant" if granted else "deny", protected=True, marker=marker), {
            "oracle_name": LOGIC_ACCESS_ORACLES["access_control"],
            "oracle_signal": "authorization_boundary_divergence",
            "positive": unexpected,
            "typed": {"protected_resource": True, "non_admin_subject": not owner, "quota_nonzero": credit_nonzero, "unexpected_grant": unexpected},
            "state_mutated": False,
        }
    if route == "/benefit":
        account = values.get("account", "0") == "1"
        value = _int_value(values.get("value", "0"))
        issued = bool(account and value > 100)
        expected = bool(account and value >= 100)
        violation = bool(account and value == 100 and not issued)
        body = _render(variant, outcome="coupon" if issued else "none", protected=False, marker=marker)
        body["issued"] = issued if variant == "red" else ("yes" if issued and variant == "blue" else (1 if issued else 0))
        return 200, body, {
            "oracle_name": LOGIC_ACCESS_ORACLES["logic_coupon"],
            "oracle_signal": "business_boundary_mismatch",
            "positive": violation,
            "typed": {"member": account, "boundary_hit": value == 100, "expected_issued": expected, "observed_issued": issued, "invariant_violation": violation},
            "state_mutated": False,
        }
    if route == "/finalize":
        verb = values.get("verb", "wait")
        prior = values.get("prior", "none")
        stamp = values.get("stamp", "old")
        fresh = values.get("fresh", "new")
        accepted = bool(verb == "commit" and prior == "verified")
        matches = stamp == fresh
        violation = bool(accepted and not matches)
        status = 200 if accepted else 403
        return status, _render(variant, outcome="commit" if accepted else "deny", protected=True, marker=marker), {
            "oracle_name": LOGIC_ACCESS_ORACLES["logic_replay"],
            "oracle_signal": "history_binding_mismatch",
            "positive": violation,
            "typed": {"commit_action": verb == "commit", "previous_verified": prior == "verified", "challenge_matches": matches, "unexpected_replay_accept": violation},
            "state_mutated": False,
        }
    return 200, _render(variant, outcome="ordinary", protected=False, marker=marker), {
        "oracle_name": "synthetic_ordinary_control_v1",
        "oracle_signal": "no_typed_violation",
        "positive": False,
        "typed": {"ordinary_response": True, "invariant_violation": False},
        "state_mutated": False,
    }


class _LogicAccessV3Handler(BaseHTTPRequestHandler):
    server_version = "sift-logic-access-fixture-v3/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        route = parsed.path
        values = {str(key): str(value[0]) for key, value in parse_qs(parsed.query, keep_blank_values=True).items() if value}
        variant = str(getattr(self.server, "fixture_variant", "red"))
        if route not in _ROUTES or variant not in _VARIANTS:
            status, payload = 404, {"error": "not_found"}
        else:
            status, payload, _ = _evaluate(route, values, variant)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-Sift-Protocol", f"logic-v3-{variant}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class LogicAccessV3Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in _VARIANTS:
            raise ValueError("unknown logic/access v3 fixture variant")
        super().__init__(address, _LogicAccessV3Handler)
        self.fixture_variant = variant


def make_logic_access_v3_fixture_server(*, port: int = 8815, variant: str = "red") -> LogicAccessV3Server:
    if int(port) not in LOGIC_ACCESS_V3_PORTS:
        raise ValueError("logic/access v3 fixture port is not allow-listed")
    return LogicAccessV3Server(("127.0.0.1", int(port)), variant)


def _validate_path(path: str) -> tuple[str, dict[str, str]]:
    route, values = _query(path)
    if route not in _ROUTES:
        raise ValueError("logic/access v3 route is not allow-listed")
    allowed = {
        "/permit": {"actor", "credit", "marker"},
        "/benefit": {"account", "value", "marker"},
        "/finalize": {"verb", "prior", "stamp", "fresh", "marker"},
        "/heartbeat": {"alive", "marker"},
    }
    if set(values) - allowed[route]:
        raise ValueError("logic/access v3 query contains an unknown field")
    return route, values


def validate_logic_access_v3_fixture_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("logic/access v3 fixture spec must be an object")
    target = str(spec.get("target", LOGIC_ACCESS_V3_BASE_URL)).rstrip("/")
    if target not in LOGIC_ACCESS_V3_BASE_URLS or str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("logic/access v3 target/method is not authorized")
    path = str(spec.get("path", "/heartbeat?alive=1"))
    route, values = _validate_path(path)
    source_id, lab_id = str(spec.get("source_id", "")), str(spec.get("lab_id", ""))
    marker = str(spec.get("marker", "logic-pg18-marker"))
    if not source_id or not lab_id or not _MARKER_RE.fullmatch(marker):
        raise ValueError("logic/access v3 provenance or marker is invalid")
    pair = dict(spec.get("pair") or {})
    if pair and (not str(pair.get("pair_id", "")) or str(pair.get("variant")) not in {"plain", "url_percent"} or not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2):
        raise ValueError("logic/access v3 pair metadata is invalid")
    probe = str(spec.get("probe", marker))
    payload = build_detection_payload(target=target, method="GET", path=path, marker=marker, probe=probe, probe_kind="http_canary", expected={})
    return {"schema_version": LOGIC_ACCESS_V3_SPEC_SCHEMA, "target": target, "method": "GET", "path": path, "route": route, "query": values, "marker": marker, "encoding": str(spec.get("encoding", "plain")), "source_id": source_id, "lab_id": lab_id, "probe": probe, "family_hint": str(spec.get("family_hint", "control")), "expected_signal": str(spec.get("expected_signal", "typed_boundary")), "payload": payload, **({"pair": pair} if pair else {})}


def _paired_path(route: str, values: dict[str, str], *, encoded: bool) -> str:
    return route + "?" + "&".join(f"{key}={_percent_encode(value) if encoded else value}" for key, value in values.items())


def _pair_specs(*, dataset_id: str, target: str, marker: str, pair_id: str, route: str, values: dict[str, str], family_hint: str, signal: str, index: int) -> list[dict[str, Any]]:
    plain = {"source_id": dataset_id, "lab_id": f"{pair_id}-plain", "target": target, "path": _paired_path(route, values, encoded=False), "marker": marker, "probe": f"pg18-probe-{index:02d}", "encoding": "plain", "family_hint": family_hint, "expected_signal": signal, "pair": {"pair_id": pair_id, "variant": "plain", "encoding_depth": 0}}
    encoded = dict(plain)
    encoded.update({"lab_id": f"{pair_id}-url-percent", "path": _paired_path(route, values, encoded=True), "encoding": "url_percent", "pair": {"pair_id": pair_id, "variant": "url_percent", "encoding_depth": 1}})
    return [plain, encoded]


def default_logic_access_v3_specs(*, dataset_id: str = "fixture-pg18-v3", target: str = LOGIC_ACCESS_V3_BASE_URL, marker: str = "logic-pg18-marker") -> list[dict[str, Any]]:
    definitions = [
        ("permit-boundary", "/permit", {"actor": "member", "credit": "1"}, "access_control", "authorization_boundary_divergence"),
        ("permit-negative", "/permit", {"actor": "member", "credit": "-1"}, "access_control", "authorization_boundary_divergence"),
        ("permit-owner-control", "/permit", {"actor": "owner", "credit": "0"}, "control", "no_typed_violation"),
        ("benefit-boundary", "/benefit", {"account": "1", "value": "100"}, "logic", "business_boundary_mismatch"),
        ("benefit-below-control", "/benefit", {"account": "1", "value": "99"}, "control", "no_typed_violation"),
        ("benefit-nonmember-control", "/benefit", {"account": "0", "value": "100"}, "control", "no_typed_violation"),
        ("finalize-mismatch", "/finalize", {"verb": "commit", "prior": "verified", "stamp": "old", "fresh": "new"}, "logic", "history_binding_mismatch"),
        ("finalize-bound-control", "/finalize", {"verb": "commit", "prior": "verified", "stamp": "new", "fresh": "new"}, "control", "no_typed_violation"),
        ("heartbeat-ordinary-control", "/heartbeat", {"alive": "1"}, "control", "no_typed_violation"),
        ("heartbeat-zero-control", "/heartbeat", {"alive": "0"}, "control", "no_typed_violation"),
    ]
    specs: list[dict[str, Any]] = []
    for index, (pair_id, route, values, family_hint, signal) in enumerate(definitions, start=1):
        specs.extend(_pair_specs(dataset_id=dataset_id, target=target, marker=marker, pair_id=f"logic-pg18-{pair_id}", route=route, values=values, family_hint=family_hint, signal=signal, index=index))
    return specs


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError):
        value = None
    return {"status_code": int(response.status_code), "headers": {"content-type": str(response.headers.get("content-type", ""))}, "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest(), "json_shape": {"type": "object" if isinstance(value, dict) else "other", "key_count": len(value) if isinstance(value, dict) else 0, "scalar_count": sum(not isinstance(item, (dict, list)) for item in value.values()) if isinstance(value, dict) else 0}}


class LogicAccessV3Collector:
    def __init__(self, *, base_url: str = LOGIC_ACCESS_V3_BASE_URL, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in LOGIC_ACCESS_V3_BASE_URLS:
            raise ValueError("logic/access v3 collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or logic_access_v3_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_logic_access_v3_fixture_spec(raw_spec)
        route, values = _query(spec["path"])
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/heartbeat?alive=1")
            response = await client.get(spec["path"], headers=spec["payload"]["headers"])
        baseline_summary, response_summary = _summary(baseline), _summary(response)
        _, _, oracle = _evaluate(route, values, "red")
        try:
            response_body = response.json()
        except (ValueError, json.JSONDecodeError):
            response_body = {}
        if route == "/benefit" and isinstance(response_body, dict):
            raw_issued = response_body.get("issued", False)
            observed = raw_issued is True or raw_issued == 1 or str(raw_issued).casefold() == "yes"
            oracle["typed"]["observed_issued"] = observed
            oracle["typed"]["invariant_violation"] = bool(oracle["typed"].get("member") and oracle["typed"].get("boundary_hit") and not observed)
            oracle["positive"] = bool(oracle["typed"]["invariant_violation"])
        reset = {"kind": "ephemeral_in_repo_logic_access_v3_fixture", "fresh": True, "fresh_target": True, "state_change_allowed": False, "evaluator_state_hidden": True, "external_network": False, "target_instance_id": self.target_instance_id, "fixture_source_sha256": self.source_hash}
        envelope = {"collector": LOGIC_ACCESS_V3_SCHEMA, "target": self.base_url, "path": spec["path"], "method": "GET", "reset": reset, "baseline": baseline_summary, "response": response_summary, "oracle_projection": oracle, "local_http_loopback": True, "script_execution": False, "network_access": False, "navigation": False, "database_touched": False, "real_sleep_performed": False, "credentials_accessed": False, "encoding": spec["encoding"], "payload_sha256": spec["payload"]["payload_sha256"]}
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        positive = bool(oracle.get("positive"))
        record = {"schema_version": LOGIC_ACCESS_V3_SCHEMA, "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}", "source_id": spec["source_id"], "lab_id": spec["lab_id"], "family": spec.get("family_hint", "control"), "payload": spec["payload"], "probe_artifact": {"original": spec["probe"], "encoding": spec["encoding"], "probe_sha256": hashlib.sha256(spec["probe"].encode()).hexdigest()}, "semantic": {"family": spec.get("family_hint", "control"), "surface": "synthetic_logic_access_http_v3", "expected_oracle": oracle["oracle_name"], "expected_signal": spec["expected_signal"]}, "evaluator_state_visible": False, "replay": {"target": self.base_url, "method": "GET", "path": spec["path"], "fresh_reset": reset, "transport": "httpx_loopback"}, "response_projection": response_summary, "oracle_projection": oracle, "evidence": checked["body"], "rule_ir_result": positive, "candidate_status": "typed_boundary_candidate" if positive else "clean_observation", "safety": {"local_only": True, "read_only": True, "fresh_reset": True, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "raw_body_stored": False, "credentials_stored": False, "state_mutated": False}}
        if spec.get("pair"):
            record["pair"] = dict(spec["pair"])
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.collect(spec) for spec in specs]


__all__ = ["LOGIC_ACCESS_V3_BASE_URL", "LOGIC_ACCESS_V3_BASE_URLS", "LOGIC_ACCESS_V3_PORTS", "LOGIC_ACCESS_V3_SCHEMA", "LogicAccessV3Collector", "default_logic_access_v3_specs", "logic_access_v3_source_sha256", "make_logic_access_v3_fixture_server", "validate_logic_access_v3_fixture_spec"]

