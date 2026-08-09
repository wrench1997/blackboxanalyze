from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_pg337_cross_impl_has_real_dvwa_holdout():
    module = importlib.import_module("scripts.build_pg337_cross_impl_process_token_dataset")
    data = module.build_dataset()
    assert data["source"]["independent_implementation_holdout"] is True
    assert data["counts"]["train"] > 0
    assert data["counts"]["implementation_holdout"] > 0
    assert data["counts"]["real_dvwa_failure_rows"] >= 1
    assert all(row["context_firewall"]["forbidden_token_count"] == 0 for row in data["records"])
    assert all(row["training_eligible"] is False for row in data["records"])


def test_pg337_audit_passes_diagnostic_contract():
    builder = importlib.import_module("scripts.build_pg337_cross_impl_process_token_dataset")
    auditor = importlib.import_module("scripts.audit_pg337_cross_impl_process_token_dataset")
    result = auditor.audit(builder.build_dataset())
    assert result["status"] == "diagnostic_only"
    assert result["scientific_gate"]["accepted_training_rows"] == 0
    assert result["promotion"]["training_allowed"] is False


def test_pg337_vocabulary_is_append_only():
    builder = importlib.import_module("scripts.build_pg337_cross_impl_process_token_dataset")
    vocab = importlib.import_module("scripts.build_pg337_cross_impl_process_vocabulary")
    data = builder.build_dataset()
    base = {"context_tokens": ["legacy=token"], "target_tokens": ["legacy_target=token"], "vocabulary_sha256": "a" * 64}
    result = vocab.build(data, base=base)
    assert "legacy=token" in result["context_tokens"]
    assert "legacy_target=token" in result["target_tokens"]
    assert result["coverage"]["independent_implementation_holdout"] is True
