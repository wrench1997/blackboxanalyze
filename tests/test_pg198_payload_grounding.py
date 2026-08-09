import json
from pathlib import Path

import httpx
import pytest

from app.payload_learner import PayloadLearner
from app.pg198_payload_grounding import (
    candidate_summary,
    choose_and_ground,
    generate_grounded_candidates,
    send_grounded_candidate,
)


TARGET = "http://127.0.0.1:8833"
ROOT = Path(__file__).resolve().parents[1]


def _transport(request: httpx.Request) -> httpx.Response:
    marker = request.url.params.get("message") or request.url.params.get("name") or "pg198-probe"
    return httpx.Response(
        200,
        headers={"content-type": "text/html"},
        content=f'<main><span data-sift-marker="{marker}">{marker}</span></main>'.encode(),
        request=request,
    )


def test_pg198_generates_get_and_post_candidates_without_leaking_raw_probe() -> None:
    get_rows = generate_grounded_candidates(
        family="xss",
        target=TARGET,
        path="/vul/xss/xss_01.php",
        method="GET",
        fields=["message", "submit"],
        marker="pg198-get-a1",
    )
    post_rows = generate_grounded_candidates(
        family="xss",
        target=TARGET,
        path="/vul/xss/xsspost/post_login.php",
        method="POST",
        fields=["username", "submit"],
        marker="pg198-post-a1",
    )
    assert len(get_rows) == len(post_rows) == 2
    assert all(row["payload"]["method"] == "GET" for row in get_rows)
    assert all(row["payload"]["method"] == "POST" for row in post_rows)
    assert post_rows[0]["payload"]["form"] == {"submit": "submit", "username": "pg198-post-a1"}
    summary = candidate_summary(get_rows[0])
    assert "probe" not in summary
    assert "<span" not in json.dumps(summary, ensure_ascii=False)
    assert summary["family_hidden_from_policy"] is True


def test_pg198_ai_selection_sends_bound_request_and_dual_oracle() -> None:
    rows = generate_grounded_candidates(
        family="xss",
        target=TARGET,
        path="/vul/xss/xss_01.php",
        method="GET",
        fields=["message", "submit"],
        marker="pg198-send-a1",
    )
    learner = PayloadLearner(seed=198)
    with httpx.Client(transport=httpx.MockTransport(_transport), base_url=TARGET) as client:
        result = choose_and_ground(
            learner,
            rows,
            client=client,
            fields=["message", "submit"],
            layout_variant="inline_html",
            baseline_status=200,
            typed_available=True,
        )
    assert result["schema_version"] == "sift-pg198-payload-grounding-v1"
    assert result["binding"]["method"] == "GET"
    assert result["binding"]["placement"] == "query"
    assert result["binding"]["runtime_only"] is True
    assert result["oracle"]["typed_available"] is True
    assert result["oracle"]["dual_agreement"] is True
    assert result["oracle"]["confirmed_positive"] is False
    assert result["raw_probe_stored"] is False
    assert result["raw_response_stored"] is False
    serialized = json.dumps(result, ensure_ascii=False)
    assert "<span" not in serialized
    assert "body_text" not in serialized
    assert result["promotion"]["training_eligible"] is False


def test_pg198_unknown_oracle_is_not_a_positive() -> None:
    rows = generate_grounded_candidates(
        family="injection",
        target=TARGET,
        path="/vul/sqli/sqli_search.php",
        method="POST",
        fields=["name", "submit"],
        marker="pg198-sql-a1",
    )
    with httpx.Client(transport=httpx.MockTransport(_transport), base_url=TARGET) as client:
        result = send_grounded_candidate(
            client,
            candidate=rows[0],
            fields=["name", "submit"],
            layout_variant="table_cell",
            typed_available=False,
        )
    assert result["binding"]["method"] == "POST"
    assert result["oracle"]["typed_available"] is False
    assert result["oracle"]["confirmed_positive"] is False
    assert result["oracle"]["vulnerability_claim_allowed"] is False
    assert result["oracle"]["abstain_reason"] == "pikachu_surface_oracle_unknown"


def test_pg198_rejects_secret_fields() -> None:
    with pytest.raises(ValueError, match="credential"):
        generate_grounded_candidates(
            family="xss",
            target=TARGET,
            path="/vul/xss/xsspost/post_login.php",
            method="POST",
            fields=["username", "password"],
            marker="pg198-secret-a1",
        )


def test_pg198_local_report_records_ai_send_and_quarantines_raw_material() -> None:
    report = json.loads((ROOT / "research" / "pg198_payload_grounding_report_v1.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((ROOT / "research" / "pg198_payload_grounding_protocol_v1.json").read_text(encoding="utf-8-sig"))
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["status"] == "completed_ai_selected_local_get_post_grounding"
    assert report["counts"]["ai_candidate_send_count"] == 6
    assert report["counts"]["grounded_payload_hash_match_count"] == 6
    assert report["counts"]["method_binding_match_count"] == 6
    assert report["counts"]["dom_dual_agreement_count"] == 2
    assert report["counts"]["unknown_oracle_abstain_count"] == 4
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert protocol["ai_role"] == "select_candidate_bind_runtime_values_send_request_receive_projection"
    assert protocol["training_promotion_allowed"] is False


def test_pg198_rule_is_registered_with_hard_promotion_gate() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))
    rule = rules["pg198_payload_grounding"]
    assert rule["get_post_coverage"] is True
    assert rule["grounded_payload_hash_match_count"] == 6
    assert rule["method_binding_match_count"] == 6
    assert rule["raw_payload_strings_stored"] is False
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
