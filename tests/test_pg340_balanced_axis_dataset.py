from __future__ import annotations

import json

from scripts.audit_pg340_balanced_axis_dataset import audit
from scripts.build_pg340_balanced_axis_dataset import build
from scripts.build_pg340_balanced_axis_vocabulary import build as build_vocabulary


def test_balanced_split_adds_an_implementation_without_leaking_context_labels() -> None:
    data = build()
    assert data["status"] == "diagnostic_only_pending_information_gate"
    assert data["counts"]["train_implementation_count"] == 2
    assert data["counts"]["holdout_implementation_count"] == 1
    assert data["counts"]["accepted_training_rows"] == 0
    assert all(row["training_eligible"] is False for row in data["records"])
    assert all(len(row["source_implementation_hash"]) == 64 for row in data["records"])
    rendered = json.dumps(data["records"], ensure_ascii=False).casefold()
    assert "pikachu-fixed" not in rendered and "webgoat" not in rendered and "vulnerables-web-dvwa" not in rendered
    assert all(value is False for value in data["promotion"].values())


def test_audit_reports_axis_entropy_and_disjoint_implementation_holdout() -> None:
    result = audit(build())
    assert result["status"] == "diagnostic_only_information_gate_pending"
    assert result["information_gate"]["passed"] is False
    assert result["scientific_gate"]["accepted_training_rows"] == 0
    assert result["split_implementation_isolation"]["passed"] is True
    assert len(result["axis_entropy"]) == 7
    assert result["axis_entropy"]["response_transport"]["train_sequence_entropy"]["unique"] >= 2
    assert "context_tokens" not in json.dumps(result, ensure_ascii=False)


def test_vocabulary_is_append_only_and_does_not_read_holdout() -> None:
    vocab = build_vocabulary()
    assert vocab["status"] == "diagnostic_only"
    assert vocab["append_only"] is True
    assert vocab["holdout_rows_used_for_vocabulary"] is False
    assert vocab["forbidden_tokens"] == []
    assert all(value is False for value in vocab["promotion"].values())
