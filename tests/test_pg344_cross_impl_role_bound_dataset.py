from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg343_full_axis_target_conditioned import audit_sources
from scripts.build_pg344_cross_impl_role_bound_dataset import build_cross_impl


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "research/pg333_three_impl_get_post_diagnostic_source_rows_v1.json",
    ROOT / "research/pg337_dvwa_failure_repair_source_rows_v1.json",
    ROOT / "research/pg342_webgoat_failure_repair_source_rows_v1.json",
)


def test_pg344_split_is_implementation_disjoint_and_keeps_source_split() -> None:
    dataset = build_cross_impl(SOURCES)
    assert dataset["counts"]["accepted_rows"] == 30
    assert dataset["counts"]["train_rows"] == 21
    assert dataset["counts"]["implementation_holdout_rows"] == 9
    assert dataset["source_split_preserved"] is True
    assert all(row["training_eligible"] is False for row in dataset["records"])
    groups = dataset["implementation_hash_groups"]
    assert all(set(counts).issubset({"train"}) or set(counts).issubset({"implementation_holdout"}) for counts in groups.values())


def test_pg344_audit_passes_context_and_implementation_isolation(tmp_path: Path) -> None:
    dataset = build_cross_impl(SOURCES)
    path = tmp_path / "pg344.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    report = audit_sources((path,))
    assert report["counts"]["implementation_split_leaks"] == 0
    assert report["counts"]["context_split_leaks"] == 0
    assert report["promotion"]["training_allowed"] is False
