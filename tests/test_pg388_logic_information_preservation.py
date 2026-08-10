from __future__ import annotations

import json

from scripts.audit_pg388_logic_information_preservation import audit


def test_information_audit_is_bounded_and_blocks_missing_train_split() -> None:
    report = audit()
    assert report["status"] == "blocked_information_gate"
    assert report["sources"]["implementation_count"] == 2
    assert report["sources"]["split_counts"] == {"implementation_holdout": 280}
    assert report["sequence_diversity"]["context"]["count"] == 280
    assert report["capacity"]["context_length_max"] > 0
    assert report["information_gate"]["passed"] is False
    assert report["training_eligible"] == 0
    assert report["promotion"]["training_allowed"] is False
    assert report["surface_dataset"]["row_count"] == 144
    assert report["surface_dataset"]["split_counts"] == {"implementation_holdout": 72, "train": 72}
    assert report["surface_dataset"]["sequence_diversity"]["cross_split_context_overlap"] == 0
    assert report["surface_dataset"]["source_contract"]["operator_reviewed"] is False
    assert report["surface_dataset"]["contract_passed"] is False


def test_information_audit_does_not_emit_rows_or_token_sequences() -> None:
    serialized = json.dumps(audit(), ensure_ascii=False).casefold()
    for marker in ("\"rows\"", "context_tokens", "target_tokens", "payload=", "wire=", "response_body=", "http://", "https://"):
        assert marker not in serialized
