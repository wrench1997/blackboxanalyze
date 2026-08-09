from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.build_pg361_payload_shape_slot_dataset import build


def _fixture_rows():
    # The builder is tested with a strict-shaped synthetic row.  It is not a
    # live target and intentionally carries no raw request/response fields.
    ontology = json.loads(open("research/pg331_web_token_ontology_v1.json", encoding="utf-8-sig").read())
    manifest = {axis: {field: "observed" for field in spec["fields"]} for axis, spec in ontology["axes"].items()}
    presence_alias = {
        "document_structure": "document_presence",
        "navigation": "navigation_presence",
        "request_transport": "request_transport_presence",
        "response_transport": "response_transport_presence",
        "javascript_surface": "javascript_presence",
        "failure_feedback": "failure_feedback_presence",
        "belief_and_replay": "belief_replay_presence",
    }
    context = []
    for axis in ontology["axes"]:
        context.append(f"{presence_alias[axis]}=observed")
        for field in ontology["axes"][axis]["fields"]:
            context.append(f"{axis}_field_{field}=observed")
    source_digest = hashlib.sha256(b"surface").hexdigest()
    row = {
        "schema_version": "pg331-whole-web-source-row-v1",
        "record_id": "row-1",
        "split": "train",
        "source_meta": {
            "source_id": "src",
            "implementation": "impl",
            "family_id": "fam",
            "surface_id": "surface",
            "collector_id": "collector",
            "authorization_id": "auth",
            "image_digest": hashlib.sha256(b"image").hexdigest(),
            "source_digest": source_digest,
        },
        "reset": {
            "fresh_reset": True,
            "reset_id": "reset-1",
            "target_instance_digest": hashlib.sha256(b"target").hexdigest(),
            "network_mode": "loopback",
            "external_network": False,
            "loopback_only": True,
            "state_clean": True,
        },
        "context_tokens": context,
        "target_projection": {
            "question": "none",
            "next_action": "select_probe_variant",
            "repair_action": "none",
            "transport_ref": "get_query",
            "field_role_ref": "query_term",
            "encoding_ref": "identity",
            "probe_variant_ref": "source_attested_candidate",
            "payload_shape_ref": "query_marker",
            "oracle_ref": "typed_effect",
            "negative_control_presence_ref": "matched_triplet",
            "safe_to_send": True,
        },
        "field_capture_manifest": manifest,
        "evaluator_sidecar": {
            "typed_available": True,
            "negative_control": True,
            "reference_present": True,
            "candidate_present": True,
            "fresh_reset": True,
            "evidence_hash": hashlib.sha256(b"evidence").hexdigest(),
            "confirmed_positive": True,
            "effect_class": "typed_effect",
            "evaluator_version": "fixture",
        },
        "operator_reviewed": False,
        "hard_negative": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "training_eligible": False,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "failures": [],
        "tokenizer": {"schema_version": "fixture", "ontology_sha256": hashlib.sha256(b"ontology").hexdigest(), "loss_report": {}},
        "axis_presence": {presence_alias[axis]: "observed" for axis in ontology["axes"]},
        "target_tokens": [],
    }
    from app.pg331_source_row import _target_tokens, sha256_json

    row["target_tokens"] = _target_tokens(row["target_projection"])
    row["record_sha256"] = sha256_json(row)
    registry = {"records": [{
        "source_hash": source_digest,
        "implementation_group": "impl",
        "surface_template_id": "surface-template",
        "transport_method": "GET",
        "parameter_role": "query_term",
        "response_shape": "html_form_get",
        "script_surface": "none",
    }]}
    return {"records": [row]}, registry


def test_builder_adds_syntax_slot_without_raw_values():
    source, registry = _fixture_rows()
    result = build(source, registry)
    row = result["records"][0]
    assert row["target_projection"]["syntax_category_ref"] == "marker"
    assert "syntax_category_ref=marker" in row["target_tokens"]
    assert row["training_eligible"] is False
    assert result["information_gate"]["raw_payload_in_context"] is False


def test_builder_fails_closed_when_source_hash_is_not_attested():
    source, registry = _fixture_rows()
    source["records"][0]["source_meta"]["source_digest"] = hashlib.sha256(b"missing").hexdigest()
    with pytest.raises(ValueError, match="validation failed|missing_registry_source"):
        build(source, registry)
