"""Provenance and authorization contracts for safe payload samples.

The catalog deliberately stores *detection manifests*, not unrestricted
exploit strings.  Every sample carries a source attestation, the original
abstract probe, its encoding description, the expected local oracle, and a
hash that binds those fields together.  Policy code may use the attestation
and a structural feature key, but family/target labels stay in evaluator-side
metadata during the holdout experiments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .detection_payload import payload_digest, validate_detection_payload
from .rule_ir import canonical as canonical_rule_ir, complexity as rule_ir_complexity


CATALOG_SCHEMA = "sift-authorized-payload-catalog-v1"
SOURCE_SCHEMA = "sift-authorized-payload-source-v1"
ALLOWED_FAMILIES = frozenset({"xss", "injection", "access_control", "url_redirect", "logic", "workflow_invariant"})
ALLOWED_ORACLES = frozenset({
    "controlled_detached_dom_v1",
    "synthetic_sql_ast_differential_v1",
    "synthetic_rule_surface_v1",
    "pikachu_bounded_http_projection_v1",
    "fixture_heterogeneous_surface_oracle_v1",
    "fixture_heterogeneous_surface_oracle_v2",
    "pg69_typed_workflow_invariant_v1",
})
REQUIRED_AUTHORIZED_USE = frozenset({"training", "local_replay", "holdout_evaluation"})
LOCAL_SCOPE = "http://127.0.0.1:3100"
AUTHORIZED_LOCAL_SCOPES = frozenset({
    LOCAL_SCOPE,
    "http://127.0.0.1:8766",
    "http://127.0.0.1:8767",
    "http://127.0.0.1:8768",
    "http://127.0.0.1:8795",
    "http://127.0.0.1:8796",
    "http://127.0.0.1:8797",
    "http://127.0.0.1:8798",
    "http://127.0.0.1:8799",
    "http://127.0.0.1:8800",
    "http://127.0.0.1:8801",
    "http://127.0.0.1:8802",
    "http://127.0.0.1:8803",
    "http://127.0.0.1:8804",
    "http://127.0.0.1:8805",
    "http://127.0.0.1:8806",
    "http://127.0.0.1:8807",
    "http://127.0.0.1:8808",
    "http://127.0.0.1:8809",
    "http://127.0.0.1:8810",
    "http://127.0.0.1:8811",
    "http://127.0.0.1:8812",
    "http://127.0.0.1:8813",
    "http://127.0.0.1:8814",
    "http://127.0.0.1:8815",
    "http://127.0.0.1:8816",
    "http://127.0.0.1:8817",
    "http://127.0.0.1:8818",
})
AUTHORIZED_SOURCE_TYPES = frozenset({"in_repo_synthetic", "authorized_local_container"})
AUTHORIZED_LICENSES = frozenset({"in_repo_synthetic", "internal-research", "local_container"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def source_digest(provenance: dict[str, Any]) -> str:
    body = dict(provenance)
    body.pop("source_sha256", None)
    return _digest(body)


def catalog_digest(catalog: dict[str, Any]) -> str:
    body = dict(catalog)
    body.pop("catalog_sha256", None)
    return _digest(body)


def probe_digest(probe: str) -> str:
    return hashlib.sha256(str(probe).encode("utf-8")).hexdigest()


def structural_feature_key(payload: dict[str, Any]) -> str:
    """Return a family-free feature used for source transfer.

    The key is deliberately structural: a DOM encoding depth, an abstract SQL
    channel class, or the generic HTTP canary surface.  It is not a
    vulnerability-family label and cannot identify an evaluator target.
    """

    normalized = validate_detection_payload(dict(payload))
    kind = normalized["probe_kind"]
    if kind in {"inert_dom_markup", "encoded_dom_markup"}:
        return f"dom:{kind}"
    if kind in {"sql_fragment_class", "sql_channel_class"}:
        return f"sql:{kind}:{normalized['probe']}"
    return f"surface:{kind}"


def validate_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("payload source provenance must be an object")
    source_id = str(provenance.get("source_id", ""))
    if not source_id or len(source_id) > 96:
        raise ValueError("payload source_id must be a short non-empty identifier")
    source_type = str(provenance.get("source_type", ""))
    if source_type not in AUTHORIZED_SOURCE_TYPES:
        raise ValueError("payload source type is not authorized for local research")
    origin = str(provenance.get("origin", ""))
    if not origin.startswith("app/") and not origin.startswith("research/"):
        raise ValueError("payload source origin must point inside the repository")
    license_name = str(provenance.get("license", ""))
    if license_name not in AUTHORIZED_LICENSES:
        raise ValueError("payload source license is not approved")
    authorization = str(provenance.get("authorization", ""))
    if authorization != "workspace_local_only":
        raise ValueError("payload source authorization must be workspace_local_only")
    scope = [str(item) for item in provenance.get("scope", [])]
    if len(scope) != 1 or scope[0] not in AUTHORIZED_LOCAL_SCOPES:
        raise ValueError("payload source scope must be an authorized loopback target")
    authorized_for = {str(item) for item in provenance.get("authorized_for", [])}
    if not REQUIRED_AUTHORIZED_USE.issubset(authorized_for):
        raise ValueError("payload source is missing an authorized use purpose")
    if bool(provenance.get("external_network", True)):
        raise ValueError("payload source may not authorize external network access")
    if bool(provenance.get("evaluator_state_visible", True)):
        raise ValueError("payload source may not expose evaluator state")
    if not str(provenance.get("captured_at", "")):
        raise ValueError("payload source captured_at is required")
    normalized = {
        "schema_version": SOURCE_SCHEMA,
        "source_id": source_id,
        "source_type": source_type,
        "origin": origin,
        "license": license_name,
        "authorization": authorization,
        "scope": scope,
        "captured_at": str(provenance["captured_at"]),
        "authorized_for": sorted(authorized_for),
        "external_network": False,
        "evaluator_state_visible": False,
    }
    if source_type == "authorized_local_container":
        image_digest = str(provenance.get("container_image_digest", ""))
        if not image_digest.startswith("sha256:") or len(image_digest) > 80:
            raise ValueError("container source must carry a pinned image digest")
        normalized["container_image_digest"] = image_digest
    expected_hash = source_digest(normalized)
    declared_hash = provenance.get("source_sha256")
    if declared_hash is not None and str(declared_hash) != expected_hash:
        raise ValueError("payload source provenance hash mismatch")
    normalized["source_sha256"] = expected_hash
    return normalized


def validate_sample(sample: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise ValueError("payload catalog sample must be an object")
    sample_id = str(sample.get("sample_id", ""))
    if not sample_id or len(sample_id) > 128:
        raise ValueError("payload catalog sample_id must be a short non-empty identifier")
    payload = validate_detection_payload(dict(sample.get("payload") or {}))
    target_scope = provenance["scope"][0]
    if payload["target"] != target_scope:
        raise ValueError("payload catalog sample target must match its authorized source scope")
    artifact = dict(sample.get("probe_artifact") or {})
    if str(artifact.get("original", "")) != payload["probe"]:
        raise ValueError("probe_artifact.original must match the validated probe")
    if str(artifact.get("probe_sha256", "")) != probe_digest(payload["probe"]):
        raise ValueError("probe artifact hash mismatch")
    encoding = str(artifact.get("encoding", ""))
    if not encoding or len(encoding) > 80:
        raise ValueError("probe artifact encoding is required")
    semantic = dict(sample.get("semantic") or {})
    family = str(semantic.get("family", ""))
    if family not in ALLOWED_FAMILIES:
        raise ValueError(f"unsupported payload catalog family: {family}")
    expected_oracle = str(semantic.get("expected_oracle", ""))
    if expected_oracle not in ALLOWED_ORACLES:
        raise ValueError("payload catalog expected_oracle is not approved")
    expected_signal = str(semantic.get("expected_signal", ""))
    if not expected_signal or len(expected_signal) > 160:
        raise ValueError("payload catalog expected_signal is required")
    if bool(sample.get("evaluator_state_visible", True)):
        raise ValueError("payload sample may not expose evaluator state")
    optional: dict[str, Any] = {}
    pair = sample.get("pair")
    if pair is not None:
        if not isinstance(pair, dict):
            raise ValueError("payload pair metadata must be an object")
        pair_id = str(pair.get("pair_id", ""))
        variant = str(pair.get("variant", ""))
        surface_role = str(pair.get("surface_role", ""))
        encoding_depth = pair.get("encoding_depth")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{4,96}", pair_id):
            raise ValueError("payload pair_id must be a short structural identifier")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", variant):
            raise ValueError("payload pair variant must be a short structural identifier")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", surface_role):
            raise ValueError("payload pair surface_role must be a short structural identifier")
        if isinstance(encoding_depth, bool) or not isinstance(encoding_depth, int) or not 0 <= encoding_depth <= 3:
            raise ValueError("payload pair encoding_depth must be an integer from 0 to 3")
        optional["pair"] = {
            "pair_id": pair_id,
            "variant": variant,
            "surface_role": surface_role,
            "encoding_depth": encoding_depth,
        }
    counterfactual = sample.get("counterfactual")
    if counterfactual is not None:
        if not isinstance(counterfactual, dict):
            raise ValueError("payload counterfactual metadata must be an object")
        kind = str(counterfactual.get("kind", ""))
        intervention = str(counterfactual.get("intervention", ""))
        source_sample_id = str(counterfactual.get("source_sample_id", ""))
        if kind not in {"negative_control", "positive_control", "oracle_flip"}:
            raise ValueError("payload counterfactual kind is not supported")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{4,96}", intervention):
            raise ValueError("payload counterfactual intervention is invalid")
        if source_sample_id and not re.fullmatch(r"[A-Za-z0-9_.-]{4,128}", source_sample_id):
            raise ValueError("payload counterfactual source_sample_id is invalid")
        optional["counterfactual"] = {
            "kind": kind,
            "intervention": intervention,
            **({"source_sample_id": source_sample_id} if source_sample_id else {}),
        }
    replay = sample.get("replay")
    if replay is not None:
        if not isinstance(replay, dict):
            raise ValueError("payload replay metadata must be an object")
        replay_target = str(replay.get("target", target_scope))
        replay_method = str(replay.get("method", payload["method"])).upper()
        if replay_target != target_scope or replay_method not in {"GET", "HEAD", "OPTIONS", "POST"}:
            raise ValueError("payload replay metadata must remain within its authorized local scope")
        replay_path = str(replay.get("path", payload["path"]))
        if not replay_path.startswith("/") or any(part in replay_path.casefold() for part in ("/api/challenges", "/snippets")):
            raise ValueError("payload replay path is not permitted")
        params = dict(replay.get("params") or {})
        if len(params) > 8:
            raise ValueError("payload replay metadata has too many query parameters")
        for key, value in params.items():
            if str(key).casefold() in {"cookie", "authorization", "token", "password"}:
                raise ValueError("payload replay metadata may not contain credentials")
            if len(str(value)) > 2048:
                raise ValueError("payload replay parameter is too large")
        replay_form = replay.get("form", payload.get("form", {}) if replay_method == "POST" else {})
        if replay_method == "POST":
            if not isinstance(replay_form, dict) or not replay_form:
                raise ValueError("POST replay metadata requires a non-empty safe form")
            if dict(replay_form) != dict(payload.get("form") or {}):
                raise ValueError("POST replay form must match the validated payload form")
        elif replay_form:
            raise ValueError("replay form data is only valid for POST metadata")
        optional["replay"] = {
            "target": target_scope,
            "method": replay_method,
            "path": replay_path,
            "params": json.loads(_canonical(params)),
            "fresh_reset": json.loads(_canonical(dict(replay.get("fresh_reset") or {}))),
            "transport": str(replay.get("transport", "local")),
        }
        if replay_method == "POST":
            optional["replay"]["form"] = json.loads(_canonical(dict(replay_form)))
    for key in ("response_projection", "oracle_projection", "evidence"):
        if key in sample:
            value = sample[key]
            if not isinstance(value, dict):
                raise ValueError(f"payload {key} must be an object")
            lowered_keys = {str(item).casefold() for item in _walk(value) if isinstance(item, str)}
            if lowered_keys & {"raw_body", "body_preview", "password", "secret", "token", "authorization", "cookie"}:
                raise ValueError(f"payload {key} contains raw/secret fields")
            optional[key] = json.loads(_canonical(value))
    rule_ir = sample.get("rule_ir")
    if rule_ir is not None:
        if not isinstance(rule_ir, dict) or "op" not in rule_ir:
            raise ValueError("payload rule_ir must be a Rule IR expression")
        canonical_ir = canonical_rule_ir(rule_ir)
        optional["rule_ir"] = json.loads(canonical_ir)
        optional["rule_ir_canonical"] = canonical_ir
        optional["rule_ir_complexity"] = rule_ir_complexity(rule_ir)
        if "rule_ir_result" in sample:
            optional["rule_ir_result"] = bool(sample["rule_ir_result"])
    normalized = {
        "schema_version": "sift-authorized-payload-sample-v1",
        "sample_id": sample_id,
        "source_id": provenance["source_id"],
        "payload": payload,
        "probe_artifact": {
            "original": payload["probe"],
            "encoding": encoding,
            "probe_sha256": probe_digest(payload["probe"]),
        },
        "semantic": {
            "family": family,
            "surface": str(semantic.get("surface", "")),
            "expected_oracle": expected_oracle,
            "expected_signal": expected_signal,
        },
        "structural_feature": structural_feature_key(payload),
        "evaluator_state_visible": False,
        **optional,
    }
    return normalized


def validate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        raise ValueError("payload catalog must be an object")
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError("unsupported payload catalog schema")
    sources = catalog.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("payload catalog must contain at least one source")
    normalized_sources: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_samples: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("payload catalog source must be an object")
        provenance = validate_provenance(dict(source.get("provenance") or {}))
        if provenance["source_id"] in seen_sources:
            raise ValueError("duplicate payload source_id")
        seen_sources.add(provenance["source_id"])
        samples = source.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("payload catalog source must contain samples")
        normalized_samples: list[dict[str, Any]] = []
        for sample in samples:
            normalized = validate_sample(dict(sample), provenance)
            if normalized["sample_id"] in seen_samples:
                raise ValueError("duplicate payload catalog sample_id")
            seen_samples.add(normalized["sample_id"])
            normalized_samples.append(normalized)
        normalized_sources.append({"provenance": provenance, "samples": normalized_samples})
    normalized = {
        "schema_version": CATALOG_SCHEMA,
        "catalog_id": str(catalog.get("catalog_id", "sift-authorized-payload-catalog")),
        "sources": normalized_sources,
        "safety": {
            "local_only": True,
            "external_network": False,
            "script_execution": False,
            "database_touched": False,
            "evaluator_state_visible": False,
            "real_exploit_strings": False,
        },
    }
    expected_hash = catalog_digest(normalized)
    declared_hash = catalog.get("catalog_sha256")
    if declared_hash is not None and str(declared_hash) != expected_hash:
        raise ValueError("payload catalog hash mismatch")
    normalized["catalog_sha256"] = expected_hash
    return normalized


def flatten_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = validate_catalog(catalog)
    rows: list[dict[str, Any]] = []
    for source in normalized["sources"]:
        provenance = source["provenance"]
        for sample in source["samples"]:
            row = {
                "sample_id": sample["sample_id"],
                "source_id": provenance["source_id"],
                "payload": copy.deepcopy(sample["payload"]),
                "provenance": copy.deepcopy(provenance),
                "semantic": copy.deepcopy(sample["semantic"]),
                "structural_feature": sample["structural_feature"],
                "probe_artifact": copy.deepcopy(sample["probe_artifact"]),
            }
            for key in (
                "pair",
                "counterfactual",
                "replay",
                "response_projection",
                "oracle_projection",
                "evidence",
                "rule_ir",
                "rule_ir_canonical",
                "rule_ir_complexity",
                "rule_ir_result",
            ):
                if key in sample:
                    row[key] = copy.deepcopy(sample[key])
            rows.append(row)
    return rows


def policy_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Return a policy-visible candidate without family/target labels."""

    provenance = validate_provenance(dict(record["provenance"]))
    payload = validate_detection_payload(dict(record["payload"]))
    attestation = {
        "source_id": provenance["source_id"],
        "source_sha256": provenance["source_sha256"],
        "source_type": provenance["source_type"],
        "origin": provenance["origin"],
        "license": provenance["license"],
        "authorization": provenance["authorization"],
        "scope": list(provenance["scope"]),
        "captured_at": provenance["captured_at"],
        "authorized_for": list(provenance["authorized_for"]),
        "external_network": False,
        "evaluator_state_visible": False,
    }
    if "container_image_digest" in provenance:
        attestation["container_image_digest"] = provenance["container_image_digest"]
    return {
        "candidate_id": str(record["sample_id"]),
        "payload": payload,
        "source_attestation": attestation,
    }


def validate_policy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if a policy proposes a sample without valid provenance."""

    if not isinstance(candidate, dict):
        raise ValueError("policy candidate must be an object")
    payload = validate_detection_payload(dict(candidate.get("payload") or {}))
    attestation = validate_provenance(dict(candidate.get("source_attestation") or {}))
    target_scope = str(attestation["scope"][0])
    if payload["target"] != target_scope:
        raise ValueError("policy candidate target is outside its authorized local scope")
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "payload": payload,
        "source_attestation": attestation,
    }


def write_catalog(path: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_catalog(catalog)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def load_catalog(path: Path) -> dict[str, Any]:
    return validate_catalog(json.loads(path.read_text(encoding="utf-8")))
