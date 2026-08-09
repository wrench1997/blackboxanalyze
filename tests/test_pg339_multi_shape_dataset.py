from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg339_multi_shape_dataset import audit
from scripts.build_pg339_multi_shape_dataset import ROOT, build


def test_build_preserves_hashed_provenance_and_never_promotes_holdout() -> None:
    data = build()
    assert data["status"] == "diagnostic_only_pending_information_gate"
    assert data["counts"]["accepted_training_rows"] == 0
    assert data["counts"]["shape_holdout_rows"] > 0
    assert data["counts"]["duplicate_rows"] > 0
    assert all(row["training_eligible"] is False for row in data["records"])
    assert all(row["split"] != "train" or row["source_split"] == "train" for row in data["records"])
    assert all(len(row["source_implementation_hash"]) == 64 for row in data["records"])
    assert all("context_tokens" not in item and "target_tokens" not in item for item in data["duplicate_manifest"])
    assert all(value is False for value in data["promotion"].values())


def test_audit_measures_isolation_entropy_and_ablation_without_token_examples() -> None:
    result = audit(build())
    assert result["information_gate"]["passed"] is False
    assert result["scientific_gate"]["accepted_training_rows"] == 0
    assert result["split_implementation_isolation"]["passed"] is True
    assert len(result["axis_entropy"]) == 7
    assert all("field_ablation" in value and "field_status_entropy" in value for value in result["axis_entropy"].values())
    rendered = json.dumps(result, ensure_ascii=False).casefold()
    assert "context_tokens" not in rendered and "target_tokens" not in rendered


def test_builder_dedupes_context_target_and_prioritizes_holdout(tmp_path: Path) -> None:
    base = json.loads((ROOT / "research" / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json").read_text(encoding="utf-8-sig"))
    row = dict(base["records"][0])
    first = dict(row); first["split"] = "train"
    second = dict(row); second["split"] = "implementation_holdout"
    pg333 = {"records": [first, second]}
    pg338 = {"records": []}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(pg333), encoding="utf-8"); b.write_text(json.dumps(pg338), encoding="utf-8")
    data = build(pg333_path=a, pg338_path=b)
    assert len(data["records"]) == 1
    assert data["records"][0]["split"] == "shape_holdout"
    assert data["counts"]["duplicate_rows"] == 1
