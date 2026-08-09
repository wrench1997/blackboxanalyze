from __future__ import annotations

from scripts.build_pg362_full_rule_ir_dataset import SLOTS, build
from scripts.audit_pg362_full_rule_ir_dataset import audit


def _source_row(question: str = "ask_failure") -> dict[str, object]:
    return {
        "record_id": "source-1",
        "split": "train",
        "context_tokens": ["document_presence=observed", "request_method=get"],
        "target_tokens": [
            "[TARGET_BOS]",
            f"question={question}",
            "next_action=repair",
            "repair_action=method",
            "transport_ref=get_query",
            "field_role_ref=display_text",
            "encoding_ref=identity",
            "probe_variant_ref=none",
            "safe_to_send=0",
            "syntax_category_ref=delimiter_boundary",
            "payload_shape_ref=html_text_marker",
            "oracle_ref=unknown",
            "negative_control_presence_ref=matched_triplet",
            "[TARGET_EOS]",
        ],
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
    }


def test_build_normalizes_legacy_target_and_derives_bounded_ask_reason() -> None:
    result = build({"records": [_source_row()]}, source_sha256="a" * 64, source_path="fixture.json")
    assert result["status"] == "diagnostic_candidate_only"
    row = result["records"][0]
    assert [token.split("=", 1)[0] for token in row["target_tokens"][1:-1]] == list(SLOTS)
    assert "ask_reason=failure_feedback" in row["target_tokens"]
    assert row["training_eligible"] is False


def test_audit_keeps_full_target_and_no_raw_material() -> None:
    result = build({"records": [_source_row("none")]}, source_sha256="b" * 64, source_path="fixture.json")
    report = audit(result)
    assert report["status"] == "diagnostic_candidate_only"
    assert report["counts"]["raw_hits"] == 0
    assert report["counts"]["missing_vocabulary_tokens"] == 0
    assert report["capacity"]["required_context_window"] > 0

