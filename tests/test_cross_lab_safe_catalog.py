import copy
import hashlib
import json
from pathlib import Path

import pytest

from app.cross_lab_safe_catalog import (
    ReadOnlySafeCatalogCollector,
    build_catalog,
    registry_status,
    validate_sample,
    validate_source,
)


ROOT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _registry() -> dict:
    return json.loads((ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json").read_text(encoding="utf-8"))


def _source(*, target_id: str = "juice_shop_loop12", host: str = "127.0.0.1") -> dict:
    registry = _registry()
    entry = next((item for item in registry["targets"] if item["target_id"] == target_id), None)
    image = (
        "sha256:" + entry["container_image"].rsplit("@sha256:", 1)[1]
        if entry and "container_image" in entry
        else "sha256:" + _hash(target_id + "-image")
    )
    return {
        "target_id": target_id,
        "app_family": target_id.replace("_loop12", ""),
        "source_id": target_id + "-pg25b",
        "source_type": "authorized_local_container",
        "origin_ref": "research.pg25b.local-manifest",
        "license": "internal-research",
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": "http", "host": host, "port": 3100},
        "container_image_digest": image,
        "collector_sha256": _hash("collector"),
        "reset_adapter_sha256": _hash("reset-adapter"),
        "oracle_contract_sha256": _hash("oracle-contract"),
        "read_only": True,
        "external_network": False,
    }


def _reset(source: dict, suffix: str) -> dict:
    return {
        "reset_id": "reset-" + suffix,
        "kind": "container_recreate",
        "target_instance_id": "instance-" + suffix,
        "state_epoch": "epoch-" + suffix,
        "reset_adapter_sha256": source["reset_adapter_sha256"],
        "baseline_projection_sha256": _hash("baseline-" + suffix),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }


def _manifest(suffix: str, *, method: str = "GET") -> dict:
    manifest = {
        "manifest_id": "manifest-" + suffix,
        "payload_sha256": _hash("payload-" + suffix),
        "probe_ref": "safe-canary-v1",
        "probe_kind": "http_canary",
        "route_template_id": "abstract-route-v1",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": _hash("marker-" + suffix),
        "max_bytes": 64,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    if method == "POST":
        manifest["form_field_names"] = ["probe"]
        manifest["form_content_type"] = "application/x-www-form-urlencoded"
    return manifest


def _response(suffix: str) -> dict:
    return {
        "status_code": 200,
        "status_class": "2xx",
        "content_type_class": "json",
        "body_length_bucket": "1-255",
        "body_sha256": _hash("body-" + suffix),
        "semantic_body_sha256": _hash("semantic-" + suffix),
        "shape": {"kind": "object", "field_count": 2, "scalar_count": 2},
        "header_names": ["content-type"],
        "marker": {"reflected": False, "location": "none", "count": 0},
        "transport_error": False,
        "status_changed": False,
        "state_changed": False,
        "location_origin_changed": False,
    }


def _oracle(source: dict, *, positive: bool) -> dict:
    return {
        "oracle_id": "typed-boundary-v1",
        "oracle_contract_sha256": source["oracle_contract_sha256"],
        "family": "access_control",
        "modality": "authorization_boundary" if positive else "negative_control",
        "candidate_signal": positive,
        "positive": positive,
        "positive_authority": True,
        "confirmed_effect": "authorization_boundary" if positive else "none",
        "signals": {"unexpected_transition": positive, "state_effect": positive},
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def _rule_ir() -> dict:
    return {
        "rule_key": "access-control.protected-transition",
        "grammar_version": "rule-ir-v1",
        "family_candidate": "access_control",
        "operator_set": ["and", "eq"],
        "required_slots": ["baseline_denied", "followup_accepted"],
        "bound_slots": ["baseline_denied", "followup_accepted"],
        "executable": False,
    }


def test_pg25b_builds_evidence_bound_evaluation_only_catalog():
    registry = _registry()
    collector = ReadOnlySafeCatalogCollector(_source(), registry=registry)
    control = collector.collect(
        sample_id="control-sample-01",
        sample_role="negative_control",
        sampling_seed=11,
        reset=_reset(collector.source, "control"),
        payload_manifest=_manifest("control"),
        response_projection=_response("control"),
        oracle_projection=_oracle(collector.source, positive=False),
        rule_ir=_rule_ir(),
    )
    positive = collector.collect(
        sample_id="candidate-sample-01",
        sample_role="candidate",
        sampling_seed=17,
        reset=_reset(collector.source, "candidate"),
        payload_manifest=_manifest("candidate"),
        response_projection=_response("candidate"),
        oracle_projection=_oracle(collector.source, positive=True),
        rule_ir=_rule_ir(),
        negative_control={
            "control_sample_id": control["sample_id"],
            "control_evidence_hash": control["evidence"]["evidence_hash"],
            "intervention": "state-unchanged",
            "verdict": "confirmed_negative",
            "same_source": True,
            "same_surface": True,
        },
    )
    catalog = build_catalog("pg25b-safe-catalog", collector.source, [control, positive])
    assert positive["decision"]["evidence_status"] == "confirmed_positive"
    assert positive["decision"]["training_action"] == "abstain"
    assert positive["decision"]["abstain_reasons"] == ["source_evaluation_only"]
    assert catalog["training_eligible"] is False
    assert catalog["safety"]["raw_body_stored"] is False
    assert catalog["catalog_sha256"]


def test_pg25b_fails_closed_for_non_loopback_or_nonfresh_source():
    registry = _registry()
    with pytest.raises(ValueError, match="loopback"):
        validate_source(_source(host="192.0.2.1"), registry=registry)
    collector = ReadOnlySafeCatalogCollector(_source(), registry=registry)
    reset = _reset(collector.source, "stale")
    reset["fresh_target"] = False
    with pytest.raises(ValueError, match="fresh reset"):
        collector.collect(
            sample_id="stale-sample-01",
            sample_role="candidate",
            sampling_seed=3,
            reset=reset,
            payload_manifest=_manifest("stale"),
            response_projection=_response("stale"),
            oracle_projection=_oracle(collector.source, positive=False),
            rule_ir=_rule_ir(),
        )


def test_pg25b_rejects_raw_projection_and_unbound_mutation():
    registry = _registry()
    collector = ReadOnlySafeCatalogCollector(_source(), registry=registry)
    projection = _response("raw")
    projection["raw_body"] = "not-permitted"
    with pytest.raises(ValueError, match="non-bounded"):
        collector.collect(
            sample_id="raw-sample-01",
            sample_role="negative_control",
            sampling_seed=5,
            reset=_reset(collector.source, "raw"),
            payload_manifest=_manifest("raw"),
            response_projection=projection,
            oracle_projection=_oracle(collector.source, positive=False),
            rule_ir=_rule_ir(),
        )
    record = collector.collect(
        sample_id="bound-sample-01",
        sample_role="negative_control",
        sampling_seed=7,
        reset=_reset(collector.source, "bound"),
        payload_manifest=_manifest("bound"),
        response_projection=_response("bound"),
        oracle_projection=_oracle(collector.source, positive=False),
        rule_ir=_rule_ir(),
    )
    tampered = copy.deepcopy(record)
    tampered["oracle_projection"]["signals"]["state_effect"] = True
    with pytest.raises(ValueError, match="oracle projection hash mismatch|not bound to evidence"):
        validate_sample(tampered, collector.source)


def test_pg25b_unregistered_labs_remain_evaluation_only():
    registry = _registry()
    assert registry_status(registry, "juice_shop_loop12")["training_eligible"] is False
    for target_id in ("webgoat", "crapi", "dvwa"):
        status = registry_status(registry, target_id)
        assert status["registered"] is False
        assert status["training_eligible"] is False
        assert status["training_role"] == "unregistered_evaluation_only"


def test_pg25b_accepts_bounded_safe_post_form_metadata():
    registry = _registry()
    collector = ReadOnlySafeCatalogCollector(_source(), registry=registry)
    record = collector.collect(
        sample_id="post-control-01",
        sample_role="negative_control",
        sampling_seed=23,
        reset=_reset(collector.source, "post"),
        payload_manifest=_manifest("post", method="POST"),
        response_projection=_response("post"),
        oracle_projection=_oracle(collector.source, positive=False),
        rule_ir=_rule_ir(),
    )
    assert record["payload_manifest"]["method"] == "POST"
    assert record["payload_manifest"]["form_field_names"] == ["probe"]
