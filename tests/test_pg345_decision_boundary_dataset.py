from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg343_full_axis_target_conditioned import audit_sources
from scripts.build_pg345_decision_boundary_dataset import BOUNDARY, build


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/pg344_cross_impl_role_bound_dataset_v1.json"


def test_pg345_adds_only_abstract_boundary_and_preserves_rows() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8-sig"))
    result = build(base, dataset_path=BASE)
    assert result["counts"]["accepted_rows"] == result["counts"]["input_rows"] == 30
    assert result["information_preservation"]["seven_axis_tokens_deleted"] == 0
    assert all(row["context_tokens"][-1] == BOUNDARY for row in result["records"])
    assert all(row["target_tokens"] == base["records"][index]["target_tokens"] for index, row in enumerate(result["records"]))
    assert all(row["training_eligible"] is False for row in result["records"])


def test_pg345_audit_accepts_explicit_schema_without_relaxing_promotion(tmp_path: Path) -> None:
    base = json.loads(BASE.read_text(encoding="utf-8-sig"))
    result = build(base, dataset_path=BASE)
    path = tmp_path / "pg345.json"
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    report = audit_sources((path,))
    assert report["counts"]["implementation_split_leaks"] == 0
    assert report["promotion"]["training_allowed"] is False
