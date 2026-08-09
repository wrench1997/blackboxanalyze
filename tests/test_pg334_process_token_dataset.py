import json

from scripts.build_pg334_process_token_dataset import build_dataset
from scripts.audit_pg334_process_token_dataset import audit


def _source():
    return {
        "records": [
            {
                "record_id": "fixture:a",
                "pair_id": "pair:a",
                "split": "implementation_train",
                "pre_question_context_tokens": ["[BOS]", "family=dom_effect", "method=GET", "unknown_slot=dom_x", "unknown_slot_count=1", "[CTX_END]"],
                "post_observation_context_tokens": ["[BOS]", "family=dom_effect", "bound_slot=dom_x", "observed_status=2xx", "[CTX_END]"],
                "labels": {"expected_positive": True},
                "source": {"fixture": "x"},
            },
            {
                "record_id": "fixture:b",
                "pair_id": "pair:a",
                "split": "implementation_holdout",
                "pre_question_context_tokens": ["[BOS]", "family=sql", "method=POST", "unknown_slot=sql_x", "unknown_slot_count=1", "[CTX_END]"],
                "post_observation_context_tokens": ["[BOS]", "family=sql", "bound_slot=sql_x", "observed_status=4xx", "[CTX_END]"],
                "labels": {"expected_positive": False},
                "source": {"fixture": "y"},
            },
        ]
    }


def test_builder_removes_family_and_slot_literals_and_preserves_process_targets():
    data = build_dataset(_source())
    assert len(data["records"]) == 4
    assert all(not any(token.startswith(("family=", "unknown_slot=", "bound_slot=")) for token in row["context_tokens"]) for row in data["records"])
    assert any(row["target_projection"]["question"] == "ask_missing_observation" for row in data["records"])
    assert all(row["training_eligible"] is False for row in data["records"])


def test_audit_is_diagnostic_only_and_negative_post_abstains():
    report = audit(build_dataset(_source()))
    # This tiny fixture intentionally has a cross-split pair, so the audit
    # may fail closed on pair completeness while still checking the process
    # labels we care about.
    assert report["status"] in {"diagnostic_only", "blocked"}
    assert report["checks"]["context_firewall"]
    assert report["checks"]["negative_abstain"]
    assert report["promotion"]["training_allowed"] is False
