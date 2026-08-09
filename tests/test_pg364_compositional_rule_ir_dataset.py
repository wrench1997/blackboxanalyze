from __future__ import annotations

from scripts.audit_pg364_compositional_rule_ir_dataset import audit
from scripts.build_pg364_compositional_rule_ir_dataset import DEFAULT_HOLDOUT_IMPLEMENTATIONS, build


def _row(implementation: str, record_id: str, *, syntax: str = "marker") -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_meta": {"implementation": implementation},
        "context_tokens": ["document_presence=observed", "request_method=get", f"syntax_hint={syntax}"],
        "target_tokens": [
            "[TARGET_BOS]",
            "question=none",
            "next_action=select_probe_variant",
            "repair_action=none",
            "transport_ref=get_query",
            "field_role_ref=query_term",
            "encoding_ref=identity",
            "syntax_category_ref=marker",
            "probe_variant_ref=source_attested_candidate",
            "safe_to_send=1",
            "payload_shape_ref=html_text_marker",
            "oracle_ref=typed_effect",
            "negative_control_presence_ref=matched_triplet",
            "[TARGET_EOS]",
        ],
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
    }


def test_compositional_split_hashes_groups_and_preserves_slot_coverage() -> None:
    source = {"records": [_row("train_impl", "a"), _row("holdout_impl", "b")]}
    result = build(source, source_sha256="a" * 64, source_path="fixture.json", holdout_implementations=("holdout_impl",))
    assert result["status"] == "diagnostic_candidate_only"
    assert result["counts"]["train_rows"] == 1
    assert result["counts"]["implementation_holdout_rows"] == 1
    assert result["split_contract"]["group_disjoint"] is True
    report = audit(result)
    assert report["status"] == "diagnostic_candidate_only"
    assert report["split_contract"]["target_value_coverage"] is True


def test_unseen_holdout_target_value_is_blocked_instead_of_hidden() -> None:
    train = _row("train_impl", "a")
    holdout = _row("holdout_impl", "b")
    holdout["target_tokens"] = [token.replace("field_role_ref=query_term", "field_role_ref=unseen_role") for token in holdout["target_tokens"]]
    result = build({"records": [train, holdout]}, source_sha256="b" * 64, source_path="fixture.json", holdout_implementations=("holdout_impl",))
    report = audit(result)
    assert report["status"] == "blocked_incomplete"
    assert any(item.startswith("target_coverage:field_role_ref") for item in report["failures"])


def test_default_holdout_is_explicit_and_promotion_remains_closed() -> None:
    assert len(DEFAULT_HOLDOUT_IMPLEMENTATIONS) == 5
    result = build({"records": [_row("train_impl", "a")]}, source_sha256="c" * 64, source_path="fixture.json")
    assert all(value is False for value in result["promotion"].values())
