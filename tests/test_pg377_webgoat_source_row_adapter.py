from __future__ import annotations

from copy import deepcopy

import pytest

from app.pg331_evaluator_sidecar import build_pg331_evaluator_sidecar, sha256_json
from app.pg377_webgoat_source_row_adapter import (
    AXES,
    FIELD_COUNT,
    METHODS,
    ROLES,
    capture_pg377_webgoat_source_row,
    validate_pg377_webgoat_source_row,
)


HTML = "<!doctype html><html lang='en'><head><script>fetch('/api')</script></head><body><form method='post'><input name='q'></form></body></html>"


def _reset() -> dict[str, object]:
    return {
        "reset_id": "pg377-reset-01",
        "fresh_reset": True,
        "target_instance_digest": sha256_json("target"),
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "volume_mount_count": 0,
        "container_restart_used": False,
    }


def _sidecar() -> dict[str, object]:
    digest = sha256_json("evidence")
    projection = {
        "status_class": "2xx",
        "content_type_class": "html",
        "body_shape": "html",
        "shape_sha256": digest,
        "non_destructive": True,
    }

    def role(name: str, typed: bool) -> dict[str, object]:
        return {
            "sent": True,
            "available": True,
            "executed": True,
            "typed_effect_confirmed": typed,
            "effect_class": "result_shape" if typed else "none",
            "projection": projection,
            "evidence_sha256": sha256_json(name),
        }

    return build_pg331_evaluator_sidecar(
        record_id="pg377:fixture",
        reset=_reset(),
        candidate=role("candidate", True),
        reference=role("reference", True),
        negative=role("negative", False),
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
    )


def test_full_page_projection_emits_seven_axes_and_107_fields() -> None:
    row = capture_pg377_webgoat_source_row(
        html=HTML,
        headers={"Content-Type": "text/html"},
        request_projection={"method": "POST", "parameters": [{"role": "query_term"}]},
        response_projection={"status": 302, "body_length": 0, "body_shape": "empty"},
        role="candidate",
        reset=_reset(),
        evaluator_sidecar=_sidecar(),
    )
    assert set(row["observation"]) == set(AXES)
    assert row["method"] == "POST"
    assert row["field_capture_manifest_count"] == FIELD_COUNT == 107
    assert all(len(row["field_capture_manifest"][axis]) > 0 for axis in AXES)
    assert row["wire_created"] is False and row["target_contacted"] is False
    assert row["typed_projection"]["evaluator_only"] is True
    assert HTML not in str(row)
    assert validate_pg377_webgoat_source_row(row)["valid"] is True


def test_missing_page_transport_and_sidecar_remain_not_observed_and_ask_safe() -> None:
    row = capture_pg377_webgoat_source_row(html=None, headers=None, request_projection=None, response_projection=None, role="reference")
    assert row["method"] == "unknown"
    assert all(status == "not_observed" for section in row["field_capture_manifest"].values() for status in section.values())
    assert row["target_projection"]["question"] == "ask_typed"
    assert row["target_projection"]["safe_to_send"] is False
    assert row["training_eligible"] is False
    # Without a page there is no claim of a complete source row.
    result = validate_pg377_webgoat_source_row(row)
    assert result["valid"] is False and "full_page_observation" in result["failures"]


def test_failure_without_action_change_forces_repair_target() -> None:
    row = capture_pg377_webgoat_source_row(
        html=HTML,
        headers={"Content-Type": "text/html"},
        request_projection={"method": "GET"},
        response_projection={"status": 200, "body_length": 10, "body_shape": "html"},
        failure_projection={"failure_class": "response_shape_mismatch", "previous_action": "candidate_probe", "next_action": "candidate_probe"},
        role="negative",
    )
    assert row["target_projection"]["question"] == "ask_failure"
    assert row["target_projection"]["next_action"] == "repair"
    assert row["target_projection"]["repair_action"] == "observe"
    assert row["target_projection"]["safe_to_send"] is False


@pytest.mark.parametrize("bad", [{"url": "http://example.test"}, {"payload": "marker"}, {"response_body": "literal"}, {"wire": "bytes"}])
def test_raw_projection_fields_are_rejected(bad: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="raw/literal"):
        capture_pg377_webgoat_source_row(
            html="<html></html>",
            headers=bad,
            request_projection={"method": "GET"},
            response_projection={"status": 200},
        )


def test_pg368_method_shape_rows_cannot_be_relabelled_as_full_source_rows() -> None:
    with pytest.raises(ValueError, match="PG-368"):
        capture_pg377_webgoat_source_row(
            html=HTML,
            request_projection={"method": "GET"},
            response_projection={"status": 200},
            method_shape_only=True,
        )


def test_evaluator_sidecar_is_not_copied_into_model_context() -> None:
    sidecar = _sidecar()
    row = capture_pg377_webgoat_source_row(
        html=HTML,
        headers={"Content-Type": "text/html"},
        request_projection={"method": "GET"},
        response_projection={"status": 200, "body_length": 64, "body_shape": "html"},
        role="replay",
        reset=_reset(),
        evaluator_sidecar=sidecar,
    )
    assert row["evaluator_sidecar"] is not None
    context = " ".join(row["context_tokens"])
    assert "typed_effect_confirmed" not in context
    # ``belief_candidate_present`` is an allowed abstract availability token;
    # the evaluator's role object/effect answer is not copied.
    assert "candidate_present" in context
    assert "confirmed_positive" not in context
    assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}


def test_javascript_overlay_accepts_bounded_script_count() -> None:
    row = capture_pg377_webgoat_source_row(
        html="<script>ignored</script>",
        javascript_context_projection={
            "script_count": 1,
            "source_text_stored": False,
            "js_semantic_tokens": ["js_script_count=present"],
        },
    )
    assert row["javascript_context_overlay"]["script_count"] == 1


def test_role_and_method_allowlists_are_explicit() -> None:
    assert set(METHODS) == {"GET", "POST"}
    assert set(ROLES) == {"candidate", "reference", "negative", "replay"}
    with pytest.raises(ValueError, match="role"):
        capture_pg377_webgoat_source_row(html="<html></html>", role="route")
