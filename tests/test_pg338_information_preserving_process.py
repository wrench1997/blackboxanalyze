from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg338_information_preserving_process_dataset import audit
from scripts.build_pg338_information_preserving_process_dataset import build
from scripts.build_pg338_information_preserving_vocabulary import build as build_vocab


ROOT = Path(__file__).resolve().parents[1]


def _dataset() -> dict:
    return json.loads((ROOT / "research" / "pg338_information_preserving_process_token_v1.json").read_text(encoding="utf-8-sig"))


def test_pg338_build_keeps_full_axis_context_and_explicit_split_policy():
    result = build()
    assert result["status"] == "diagnostic_only_full_axis_cross_implementation"
    assert result["counts"] == {
        "total": 27,
        "train": 18,
        "implementation_holdout": 9,
        "probe_observed": 18,
        "failure_repair": 6,
        "negative_review": 3,
        "ask_preflight": 0,
        "full_axis_rows": 27,
    }
    assert result["source"]["independent_implementation_holdout"] is True
    assert result["process_policy"]["accepted_training_rows"] == 0
    assert min(len(row["context_tokens"]) for row in result["records"]) >= 32
    assert all(row["context_firewall"]["forbidden_token_count"] == 0 for row in result["records"])
    assert all(row["training_eligible"] is False for row in result["records"])


def test_pg338_information_audit_measures_all_axes_and_ablation():
    result = audit(_dataset())
    assert result["status"] == "diagnostic_only"
    assert result["record_count"] == 27
    assert result["context_target_alignment"]["failed_rows"] == 0
    assert all(details["coverage"] == 1.0 for details in result["axis_entropy"].values())
    assert all(details["field_ablation"]["changed_rate"] == 1.0 for details in result["axis_entropy"].values())
    assert result["scientific_gate"]["accepted_training_rows"] == 0


def test_pg338_vocabulary_is_context_target_separate_and_firewalled():
    result = build_vocab(_dataset())
    assert result["status"] == "diagnostic_only"
    assert result["context_vocabulary_size"] > 400
    assert result["target_vocabulary_size"] >= 10
    assert result["forbidden_tokens"] == []
    assert result["append_only"] is True
    assert all(value is False for value in result["promotion"].values())
