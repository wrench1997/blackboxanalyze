from __future__ import annotations

from scripts.audit_pg361_payload_shape_slots import audit


def test_audit_reports_slot_entropy_and_keeps_promotion_closed():
    dataset = {
        "records": [
            {
                "schema_version": "not-a-real-row",
                "record_id": "diagnostic-only",
                "split": "train",
                "context_tokens": ["document_presence=observed"],
                "target_tokens": ["syntax_category_ref=marker"],
                "training_eligible": False,
            }
        ]
    }
    report = audit(dataset, dataset_sha256="a" * 64)
    assert report["status"] == "blocked"
    assert report["gates"]["predictive_entropy_holdout"] == "not_run"
    assert report["promotion"]["training_allowed"] is False

