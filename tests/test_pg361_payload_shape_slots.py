from __future__ import annotations

import hashlib

import pytest

from app.pg350_runtime_payload_binder import bind_runtime_probe
from app.pg361_payload_shape_slots import syntax_attestation, target_tokens, validate_slots


def _slots(**overrides):
    result = {
        "transport_ref": "get_query",
        "field_role_ref": "query_term",
        "encoding_ref": "url_percent",
        "syntax_category_ref": "delimiter_boundary",
        "payload_shape_ref": "query_marker",
        "probe_variant_ref": "source_attested_candidate",
        "oracle_ref": "typed_effect",
        "negative_control_presence_ref": "matched_triplet",
        "safe_to_send": True,
    }
    result.update(overrides)
    return result


def _runtime():
    return {
        "target_origin": "http://127.0.0.1:8080",
        "route": {"method": "GET", "path": "/lab/search", "field_name": "q"},
        "loopback_only": True,
        "external_network": False,
        "source_attested": True,
        "route_attested": True,
        "field_attested": True,
        "fresh_reset": True,
        "candidate_reference_negative": True,
        "replay_consistency": True,
        "authorization_id": "pg361_local_lab",
        "allowed_template_ids": ["pg361_query_v1"],
        "stateful_evaluator": False,
    }


def _catalog(category: str = "delimiter_boundary"):
    template = "{{MARKER}}'"
    return {
        "templates": [
            {
                "template_id": "pg361_query_v1",
                "shape": "query_marker",
                "syntax_category_ref": category,
                "template": template,
                "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "local_only": True,
                "non_destructive": True,
            }
        ]
    }


def test_slots_are_abstract_and_have_stable_token_order():
    canonical = validate_slots(_slots())
    tokens = target_tokens(_slots())
    assert canonical["syntax_category_ref"] == "delimiter_boundary"
    assert tokens[0] == "[TARGET_BOS]"
    assert tokens[4] == "syntax_category_ref=delimiter_boundary"
    assert tokens[-1] == "[TARGET_EOS]"
    assert all("http" not in token and "<script" not in token for token in tokens)


def test_new_candidate_requires_syntax_category():
    with pytest.raises(ValueError, match="syntax_category_ref"):
        validate_slots({key: value for key, value in _slots().items() if key != "syntax_category_ref"})


def test_raw_literal_is_rejected_but_payload_shape_slot_is_allowed():
    with pytest.raises(ValueError, match="raw"):
        validate_slots({**_slots(), "raw_value": "literal"})
    assert validate_slots(_slots())["payload_shape_ref"] == "query_marker"


def test_source_attestation_hashes_only_bounded_surface_metadata():
    attestation = syntax_attestation(
        {
            "implementation_group": "fixture_impl",
            "surface_template_id": "query_surface",
            "transport_method": "GET",
            "parameter_role": "query_term",
            "response_shape": "html_text",
            "script_surface": "none",
        },
        "delimiter_boundary",
    )
    assert len(attestation["syntax_attestation_sha256"]) == 64
    assert "query_surface" not in attestation


def test_binder_requires_template_category_for_pg361_rule():
    with pytest.raises(ValueError, match="syntax category"):
        bind_runtime_probe(
            _slots(),
            _runtime(),
            _catalog("marker"),
            marker="PG361_CANARY",
        )


def test_binder_generates_ephemeral_wire_from_abstract_slots():
    probe = bind_runtime_probe(_slots(), _runtime(), _catalog(), marker="PG361_CANARY")
    assert "PG361_CANARY%27" in probe.human_review_wire()
    persisted = probe.persisted_projection()
    assert persisted["abstract_slots"]["syntax_category_ref"] == "delimiter_boundary"
    assert "PG361_CANARY" not in str(persisted)
    assert persisted["raw_wire_stored"] is False
