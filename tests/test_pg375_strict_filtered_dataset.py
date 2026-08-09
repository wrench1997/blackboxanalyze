from __future__ import annotations

import json

from scripts.audit_pg375_strict_dataset import audit
from scripts.build_pg362_full_rule_ir_dataset import SLOTS
from scripts.build_pg375_strict_dataset import build


def _raw(record_id: str, split: str, context: list[str], *, implementation: str) -> dict[str, object]:
    values = {
        "question": "none",
        "ask_reason": "none",
        "next_action": "select_probe_variant",
        "repair_action": "none",
        "transport_ref": "get_query",
        "field_role_ref": "query_term",
        "encoding_ref": "identity",
        "syntax_category_ref": "marker",
        "probe_variant_ref": "source_attested_candidate",
        "safe_to_send": "1",
        "payload_shape_ref": "html_text_marker",
        "oracle_ref": "typed_effect",
        "negative_control_presence_ref": "matched_triplet",
    }
    return {
        "record_id": record_id,
        "source_meta": {"implementation": implementation},
        "split": split,
        "context_tokens": context,
        "target_tokens": ["[TARGET_BOS]", *[f"{slot}={values[slot]}" for slot in SLOTS], "[TARGET_EOS]"],
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
    }


def _source() -> dict[str, object]:
    return {
        "schema_version": "fixture",
        "records": [
            _raw("train-a", "train", ["document_presence=observed", "request_method=get", "surface=alpha"], implementation="train"),
            _raw("train-b", "train", ["document_presence=observed", "request_method=get", "surface=beta"], implementation="train"),
            _raw("holdout-a", "implementation_holdout", ["document_presence=observed", "request_method=get", "surface=alpha"], implementation="holdout"),
            _raw("holdout-b", "implementation_holdout", ["request_method=get", "document_presence=observed", "surface=beta"], implementation="holdout"),
            _raw("holdout-unknown", "implementation_holdout", ["request_method=get", "surface=unseen"], implementation="holdout"),
        ],
        "split_contract": {"train_group_hashes": ["train-group"], "holdout_group_hashes": ["holdout-group"]},
    }


def test_pg375_builder_holdout_precedence_and_quarantine() -> None:
    result = build(_source(), source_sha256="a" * 64, source_path="fixture.json")
    assert result["status"] == "candidate_only"
    assert result["counts"]["filtered_train_rows"] == 1
    assert result["counts"]["active_holdout_rows"] == 1
    assert result["counts"]["excluded_train_rows"] == 1
    # Removing train-a under holdout precedence also removes its only source
    # for ``surface=alpha``; both the duplicate and the explicitly unseen
    # holdout row are therefore quarantined rather than silently relabelled.
    assert result["counts"]["quarantined_holdout_rows"] == 2
    assert result["split_contract"]["active_cross_split_context_overlap"] == 0
    assert result["split_contract"]["active_cross_split_exact_overlap"] == 0
    assert result["coverage"]["filtered_train"]["token_entropy_bits"] > 0
    assert result["excluded"][0].keys() == {"record_ref_sha256", "original_split", "reasons"}
    assert "context_tokens" not in result["quarantine"][0]


def test_pg375_audit_recomputes_zero_active_gaps_and_keeps_promotion_closed() -> None:
    result = build(_source(), source_sha256="b" * 64, source_path="fixture.json")
    report = audit(result)
    assert report["status"] == "passed_candidate_audit"
    assert report["counts"]["unknown_context_tokens"] == 0
    assert report["counts"]["unknown_target_tokens"] == 0
    assert report["counts"]["active_cross_split_context_overlap"] == 0
    assert report["quarantine_contract"]["hash_only_refs"] is True
    assert report["quarantine_contract"]["rows_not_relabelled"] is True
    assert all(value is False for value in report["promotion"].values())


def test_pg375_builder_never_emits_raw_material() -> None:
    encoded = json.dumps(build(_source(), source_sha256="c" * 64, source_path="fixture.json"), ensure_ascii=False)
    assert "raw_payload=" not in encoded
    assert "response_body=" not in encoded
    assert "http://" not in encoded
