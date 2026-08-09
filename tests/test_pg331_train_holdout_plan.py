from __future__ import annotations

from pathlib import Path

from app.pg331_source_row import collect_pg331_source_row
from scripts.plan_pg331_train_holdout import plan
from tests.test_pg331_source_row import _evaluator, _field_capture_manifest, _observation, _reset, _source_meta, _target


def _row(record_id: str, implementation: str, split: str, *, reviewed: bool) -> dict[str, object]:
    meta = _source_meta()
    meta.update({"implementation": implementation, "family_id": "fixture"})
    return collect_pg331_source_row(
        record_id=record_id,
        observation=_observation(),
        source_meta=meta,
        reset=_reset(),
        evaluator=_evaluator(),
        field_capture_manifest=_field_capture_manifest(),
        target_projection=_target(),
        split=split,
        operator_reviewed=reviewed,
    )


def test_plan_passes_only_with_explicit_disjoint_eligible_train_and_holdout(tmp_path: Path) -> None:
    import json

    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps({"records": [_row("train-1", "impl-a", "train", reviewed=True), _row("hold-1", "impl-b", "implementation_holdout", reviewed=False)]}), encoding="utf-8")
    report = plan(dataset, train_implementation="impl-a", holdout_implementation="impl-b")
    assert report["status"] == "passed"
    assert report["counts"]["eligible_train_rows"] == 1
    assert report["counts"]["holdout_rows"] == 1
    assert report["promotion"]["training_allowed"] is False


def test_plan_blocks_current_all_holdout_diagnostic_dataset() -> None:
    report = plan(Path("research/pg331_cross_impl_multiseed_diagnostic_v1.json"), train_implementation="pikachu-fixed", holdout_implementation="bkimminich-juice-shop")
    assert report["status"] == "blocked"
    assert "empty:eligible_train_rows" in report["failures"]
    assert report["counts"]["holdout_rows"] > 0


def test_plan_rejects_same_implementation() -> None:
    import pytest

    with pytest.raises(ValueError, match="must differ"):
        plan(Path("research/pg331_cross_impl_multiseed_diagnostic_v1.json"), train_implementation="pikachu-fixed", holdout_implementation="pikachu-fixed")
