from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_pg385_variant_selector_candidate import (
    DEFAULT_DATASET,
    _safe_rows,
    run_candidate,
)


def test_variant_selector_cpu_candidate_is_abstract_only() -> None:
    report = run_candidate(
        dataset_path=DEFAULT_DATASET,
        device="cpu",
        epochs=1,
        microbatch=8,
        d_model=32,
        n_layers=1,
        experts=2,
        expert_hidden=64,
        max_length=128,
        row_limit=16,
        checkpoint_dir=None,
    )
    assert report["status"] == "abstract_variant_selector_candidate_only"
    assert report["data"]["vocabulary_scope"] == "train_context_only"
    assert report["data"]["unknown_gaps"]["blocked"] is False
    assert report["evaluator_binding"]["model_emits_raw_string"] is False
    assert report["evaluator_binding"]["model_emits_variant_reference"] is True
    assert report["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def test_loader_rejects_raw_model_fragment() -> None:
    data = json.loads(Path(DEFAULT_DATASET).read_text(encoding="utf-8"))
    data["records"][0]["context_tokens"].append("payload=forbidden")
    with pytest.raises(ValueError, match="raw/evaluator marker"):
        _safe_rows(data)


def test_loader_requires_train_and_holdout() -> None:
    data = json.loads(Path(DEFAULT_DATASET).read_text(encoding="utf-8"))
    for row in data["records"]:
        row["split"] = "train"
    with pytest.raises(ValueError, match="split is empty"):
        _safe_rows(data)
