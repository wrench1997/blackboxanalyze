from __future__ import annotations

import hashlib
import json

import pytest

from app.pg350_runtime_payload_binder import (
    bind_runtime_probe,
    validate_template_catalog,
)


def _catalog(*, shape: str = "sql_string_marker", template: str = "{{MARKER}}'") -> dict:
    return {
        "templates": [
            {
                "template_id": "pg350_sql_quote_v1",
                "shape": shape,
                "template": template,
                "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "local_only": True,
                "non_destructive": True,
            }
        ]
    }


def _runtime(*, method: str = "GET", path: str = "/lab/search.php", stateful: bool = False) -> dict:
    result = {
        "target_origin": "http://127.0.0.1:8080",
        "route": {"method": method, "path": path, "field_name": "q"},
        "loopback_only": True,
        "external_network": False,
        "source_attested": True,
        "route_attested": True,
        "field_attested": True,
        "fresh_reset": True,
        "candidate_reference_negative": True,
        "replay_consistency": True,
        "authorization_id": "pg350_local_lab",
        "allowed_template_ids": ["pg350_sql_quote_v1"],
        "stateful_evaluator": stateful,
    }
    if stateful:
        result.update({"state_reset_before": True, "state_reset_after": True, "database_clean": True, "teardown": True})
    return result


def _rule(*, transport: str = "get_query", shape: str = "sql_string_marker", encoding: str = "identity") -> dict:
    return {
        "transport_ref": transport,
        "field_role_ref": "query_term",
        "encoding_ref": encoding,
        "payload_shape_ref": shape,
        "probe_variant_ref": "source_attested_candidate",
        "oracle_ref": "response_shape",
        "safe_to_send": "1",
    }


def test_binds_abstract_shape_to_ephemeral_get_wire_and_hides_persisted_raw() -> None:
    probe = bind_runtime_probe(_rule(), _runtime(), _catalog(), marker="BB_CANARY_42")
    wire = probe.human_review_wire()
    assert "GET http://127.0.0.1:8080/lab/search.php?q=BB_CANARY_42%27" in wire
    assert "BB_CANARY_42'" in probe.raw_value
    persisted = probe.persisted_projection()
    serialized = json.dumps(persisted, ensure_ascii=False)
    assert "BB_CANARY_42'" not in serialized
    assert "GET http://127.0.0.1:8080" not in serialized
    assert persisted["raw_payload_stored"] is False
    assert persisted["raw_wire_stored"] is False
    assert persisted["training_context_allowed"] is False


def test_post_form_requires_post_and_uses_wire_encoding() -> None:
    template = "{{MARKER}}"
    catalog = _catalog(shape="html_form_marker", template=template)
    runtime = _runtime(method="POST", path="/lab/submit.php")
    rule = _rule(transport="post_form", shape="html_form_marker", encoding="form_urlencoded")
    probe = bind_runtime_probe(rule, runtime, catalog, marker="BB_FORM_1")
    assert probe.method == "POST"
    assert "Content-Type: application/x-www-form-urlencoded" in probe.human_review_wire()
    assert "q=BB_FORM_1" in (probe.body or "")


def test_stateful_lane_requires_reset_clean_and_teardown() -> None:
    runtime = _runtime(stateful=True)
    runtime.pop("database_clean")
    with pytest.raises(ValueError, match="stateful evaluator"):
        bind_runtime_probe(_rule(), runtime, _catalog(), marker="BB_STATE_1")


def test_rejects_public_target_and_raw_model_slot() -> None:
    runtime = _runtime()
    runtime["target_origin"] = "https://example.invalid"
    with pytest.raises(ValueError, match="loopback"):
        bind_runtime_probe(_rule(), runtime, _catalog(), marker="BB_CANARY_1")
    with pytest.raises(ValueError, match="raw/evaluator"):
        bind_runtime_probe({**_rule(), "raw_value": "literal"}, _runtime(), _catalog(), marker="BB_CANARY_1")


def test_rejects_missing_oracle_and_unapproved_template() -> None:
    with pytest.raises(ValueError, match="oracle_ref"):
        bind_runtime_probe({**_rule(), "oracle_ref": "unknown"}, _runtime(), _catalog(), marker="BB_CANARY_1")
    runtime = _runtime()
    runtime["allowed_template_ids"] = ["not_in_catalog"]
    with pytest.raises(ValueError, match="exactly one"):
        bind_runtime_probe(_rule(), runtime, _catalog(), marker="BB_CANARY_1")


def test_template_catalog_projection_never_returns_raw_template() -> None:
    normalized = validate_template_catalog(_catalog(), shape="sql_string_marker")
    assert all("template" not in entry for entry in normalized["templates"])
    assert normalized["templates"][0]["template_sha256"]
