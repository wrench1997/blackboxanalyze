from __future__ import annotations

import json
from pathlib import Path

import torch

from app.pg283_feedback_policy import FeedbackPolicy, build_vocab, encode, evaluate


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg283_dataset_is_audited_and_split():
    data = _load("pg283_feedback_policy_dataset_v1.json")
    audit = _load("pg283_feedback_policy_dataset_audit_v1.json")
    assert audit["status"] == "passed"
    assert data["counts"] == {"train": 1161, "route_dev": 108, "family_holdout": 135, "hard_negative": 324, "total": 1728}
    assert data["training_contract"]["family_hidden_in_context"] is True
    assert data["training_contract"]["literal_payload_values_out_of_context"] is True
    assert data["scientific_contract"]["cross_template_generalization_claim_allowed"] is False


def test_pg283_hard_negative_is_abstain_only():
    data = _load("pg283_feedback_policy_dataset_v1.json")
    hard = data["hard_negative_records"]
    assert hard
    assert all(row["target"]["next_action"] == "abstain" for row in hard)
    assert all(row["target"]["safe_to_send"] is False for row in hard)
    assert all("oracle_label_in_context=0" in row["context_tokens"] for row in hard)


def test_pg283_model_contract_forward_and_evaluate_without_training():
    data = _load("pg283_feedback_policy_dataset_v1.json")
    rows = data["records"][:8]
    vocab = build_vocab(rows)
    values, lengths, targets = encode(rows, vocab)
    assert values.shape[0] == len(rows)
    assert lengths.shape[0] == len(rows)
    model = FeedbackPolicy(len(vocab))
    output = model(values, lengths)
    assert set(output) == {"action", "probe", "channel", "encoding", "safe"}
    metrics = evaluate(model, rows, vocab, torch.device("cpu"))
    assert metrics["count"] == len(rows)
    assert "false_allow_count" in metrics


def test_pg283_report_keeps_scientific_gate_blocked_after_remote_training():
    report = _load("pg283_feedback_policy_report_v1.json")
    audit = _load("pg283_feedback_policy_audit_v1.json")
    assert report["status"] == "completed_remote_pg283_feedback_policy"
    assert report["device"]["cuda_visible_devices"] == "0"
    assert report["scientific_gate"]["status"] == "blocked"
    assert report["source"]["live_send"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert audit["status"] == "passed"
