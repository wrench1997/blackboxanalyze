"""PG-198 AI payload grounding for an authorized local replay.

The model may choose a parameterized probe class, but the request executor
only binds that choice to observed, non-secret GET/POST fields.  Runtime
values are kept in memory for the single loopback request and are never
written to the report.  The persisted record contains hashes, method/path
binding, bounded response projections, and the oracle decision only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import httpx

from .detection_payload import build_detection_payload, validate_detection_payload
from .maze_engine import sha256_json
from .payload_learner import PayloadLearner, generate_payload_candidates
from .pg185_pikachu_dom_adapter import inert_dom_probe, validate_marker
from .pg193_browser_dom_oracle import run_browser_dom_oracle
from .pg197_alt_dom_oracle import run_alt_dom_oracle
from .pg195_request_surface_adapter import project_surface_response


SCHEMA_VERSION = "sift-pg198-payload-grounding-v1"
_ALLOWED_METHODS = frozenset({"GET", "POST"})
_FORBIDDEN_FIELDS = frozenset({
    "password", "passwd", "secret", "token", "csrf", "cookie",
    "session", "authorization", "file", "upload",
})
_ALLOWED_FIELDS = frozenset({
    "message", "text", "content", "name", "username", "id", "title",
    "url", "filename", "submit", "probe", "src",
})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _route_fields(fields: list[str] | tuple[str, ...]) -> list[str]:
    normalized = sorted({str(item) for item in fields if str(item)})
    if not normalized or len(normalized) > 16:
        raise ValueError("PG-198 requires a bounded observed field set")
    for field in normalized:
        if field.casefold() in _FORBIDDEN_FIELDS:
            raise ValueError("PG-198 refuses credential or session fields")
        if field.casefold() not in _ALLOWED_FIELDS:
            raise ValueError("PG-198 field was not observed in the safe route inventory")
    return normalized


def _runtime_values(*, payload: Mapping[str, Any], fields: list[str]) -> dict[str, str]:
    """Bind a validated abstract candidate to a runtime-only field map."""

    marker = validate_marker(str(payload["marker"]))
    probe_kind = str(payload["probe_kind"])
    values: dict[str, str] = {}
    for field in fields:
        lowered = field.casefold()
        if lowered == "submit":
            values[field] = "submit"
        elif lowered == "id":
            values[field] = "1"
        elif probe_kind in {"inert_dom_markup", "encoded_dom_markup"}:
            encoding = "identity" if probe_kind == "inert_dom_markup" else "html_entity"
            values[field] = inert_dom_probe(marker, encoding=encoding)
        else:
            # SQL and HTTP candidates remain abstract classes; the network
            # request carries only an inert marker, never an SQL operator or
            # executable fragment.
            values[field] = marker
    return values


def generate_grounded_candidates(
    *,
    family: str,
    target: str,
    path: str,
    method: str,
    fields: list[str] | tuple[str, ...],
    marker: str,
) -> list[dict[str, Any]]:
    """Generate AI-visible candidate manifests for one observed route."""

    method = str(method).upper()
    if method not in _ALLOWED_METHODS:
        raise ValueError("PG-198 supports only GET and safe POST")
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("PG-198 path must be origin-relative")
    marker = validate_marker(marker)
    field_names = _route_fields(fields)
    base = generate_payload_candidates(str(family), path=path, marker=marker)
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(base):
        source = dict(row["payload"])
        form = {field: marker for field in field_names if field.casefold() != "submit"}
        if method == "POST" and "submit" in field_names:
            form["submit"] = "submit"
        payload = build_detection_payload(
            target=target,
            method=method,
            path=path,
            marker=marker,
            probe=source["probe"],
            probe_kind=source["probe_kind"],
            form=form if method == "POST" else None,
            expected=dict(source.get("expected") or {}),
        )
        candidates.append({
            "candidate_id": _digest({"family": family, "index": index, "payload": payload})[:20],
            "family": str(family),
            "payload": payload,
            "generation": {
                "generator": "pg198-payload-learner-grounder-v1",
                "family_hidden_from_policy": True,
                "candidate_index": index,
                "field_names": field_names,
            },
        })
    return candidates


def candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the persisted candidate view without the raw probe value."""

    payload = validate_detection_payload(dict(candidate.get("payload") or {}))
    generation = dict(candidate.get("generation") or {})
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "family": str(candidate.get("family", "unknown")),
        "method": payload["method"],
        "path": payload["path"],
        "probe_kind": payload["probe_kind"],
        "probe_sha256": hashlib.sha256(str(payload["probe"]).encode("utf-8")).hexdigest(),
        "payload_sha256": payload["payload_sha256"],
        "expected_keys": sorted(str(key) for key in dict(payload.get("expected") or {})),
        "candidate_index": int(generation.get("candidate_index", -1)),
        "family_hidden_from_policy": bool(generation.get("family_hidden_from_policy", False)),
    }


def send_grounded_candidate(
    client: httpx.Client,
    *,
    candidate: Mapping[str, Any],
    fields: list[str] | tuple[str, ...],
    layout_variant: str,
    baseline_status: int | None = None,
    typed_available: bool = False,
) -> dict[str, Any]:
    """Send one AI-selected candidate and return only bounded evidence."""

    payload = validate_detection_payload(dict(candidate.get("payload") or {}))
    field_names = _route_fields(fields)
    values = _runtime_values(payload=payload, fields=field_names)
    method = str(payload["method"]).upper()
    path = str(payload["path"])
    if method == "GET":
        response = client.get(path, params=values, follow_redirects=False)
    elif method == "POST":
        response = client.post(path, data=values, follow_redirects=False)
    else:  # validate_detection_payload already rejects this; keep fail-closed.
        raise ValueError("PG-198 request method is not allow-listed")

    projected = project_surface_response(
        response,
        marker=str(payload["marker"]),
        layout_variant=str(layout_variant),
        baseline_status=baseline_status,
        run_browser=typed_available,
    )
    body_text = str(projected.pop("body_text", ""))
    signal = dict(projected.pop("signal", {}) or {})
    projection = dict(projected["response_projection"])
    projection_marker = dict(projection.get("marker") or {})
    signal["candidate_signal"] = bool(
        signal.get("marker_reflected")
        or signal.get("sql_error_shape")
        or signal.get("redirect_present")
        or projection.get("status_changed")
        or projection_marker.get("reflected")
    )
    binding = {
        "method": method,
        "path": path,
        "placement": "query" if method == "GET" else "form",
        "field_names": field_names,
        "field_count": len(field_names),
        "values_sha256": _digest(values),
        "runtime_only": True,
    }
    binding["binding_sha256"] = sha256_json(binding)
    oracle: dict[str, Any]
    if typed_available:
        browser = dict(projected.get("oracle_projection") or {})
        browser_effect = bool(browser.get("typed_surface_effect"))
        alternate = run_alt_dom_oracle(f"<main>{body_text}</main>", marker=str(payload["marker"]))
        agreement = bool(browser_effect == bool(alternate["dom_change"]))
        oracle = {
            "typed_available": True,
            "browser_effect": browser_effect,
            "alternate_effect": bool(alternate["dom_change"]),
            "dual_agreement": agreement,
            "confirmed_positive": False,
            "vulnerability_claim_allowed": False,
            "browser_evidence_hash": str((browser.get("signals") or {}).get("evidence_hash", "")),
            "alternate_evidence_hash": str(alternate["evidence_hash"]),
        }
    else:
        oracle = {
            "typed_available": False,
            "dual_agreement": False,
            "confirmed_positive": False,
            "vulnerability_claim_allowed": False,
            "abstain_reason": "pikachu_surface_oracle_unknown",
        }
    evidence = {
        "candidate_sha256": _digest(candidate_summary(candidate)),
        "payload_sha256": str(payload["payload_sha256"]),
        "binding_sha256": binding["binding_sha256"],
        "projection_sha256": str(projection.get("projection_sha256", "")),
        "oracle_sha256": _digest(oracle),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate_summary(candidate),
        "binding": binding,
        "response_projection": projected["response_projection"],
        "oracle": oracle,
        "signal": {
            key: signal[key]
            for key in (
                "marker_reflected", "marker_in_script_source", "marker_in_attribute",
                "sql_error_shape", "redirect_present", "external_redirect",
                "status_changed", "candidate_signal",
            )
            if key in signal
        },
        "evidence": {**evidence, "evidence_sha256": _digest(evidence)},
        "raw_probe_stored": False,
        "raw_response_stored": False,
    }


def choose_and_ground(
    learner: PayloadLearner,
    candidates: list[dict[str, Any]],
    *,
    client: httpx.Client,
    fields: list[str] | tuple[str, ...],
    layout_variant: str,
    baseline_status: int | None,
    typed_available: bool,
) -> dict[str, Any]:
    """Let the learner choose, send, and receive a feedback record."""

    chosen = learner.select(candidates)
    result = send_grounded_candidate(
        client,
        candidate=chosen,
        fields=fields,
        layout_variant=layout_variant,
        baseline_status=baseline_status,
        typed_available=typed_available,
    )
    signal = bool((result.get("signal") or {}).get("candidate_signal", False))
    status = "candidate" if signal else "dead_end"
    feedback = learner.observe(chosen, status=status, evaluator_confirmed=False)
    result["ai_decision"] = {
        "candidate_id": str(chosen["candidate_id"]),
        "selection_score": float(chosen.get("selection_score", 0.0)),
        "status_feedback": feedback["status"],
        "model_used_evaluator": False,
    }
    result["promotion"] = {
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return result


__all__ = [
    "SCHEMA_VERSION",
    "candidate_summary",
    "choose_and_ground",
    "generate_grounded_candidates",
    "send_grounded_candidate",
]
