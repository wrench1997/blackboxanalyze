from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_pg331_diagnostic_datasets import merge


def _dataset(path: Path, record_id: str, implementation: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "pg331-source-row-collection-v1",
                "dataset_sha256": f"reported-{record_id}",
                "records": [
                    {
                        "record_id": record_id,
                        "training_eligible": False,
                        "source_meta": {"implementation": implementation, "family_id": "fixture"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_merge_preserves_rows_and_records_input_hashes(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a.json", "a", "impl-a")
    second = _dataset(tmp_path / "b.json", "b", "impl-b")
    result = merge([first, second])
    assert result["counts"] == {
        "input_datasets": 2,
        "records": 2,
        "implementations": 2,
        "families": 1,
        "training_eligible": 0,
        "duplicate_record_ids": 0,
    }
    assert len(result["input_manifests"]) == 2
    assert all(len(item["file_sha256"]) == 64 for item in result["input_manifests"])
    assert result["promotion"]["training_allowed"] is False


def test_merge_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a.json", "same", "impl-a")
    second = _dataset(tmp_path / "b.json", "same", "impl-b")
    with pytest.raises(ValueError, match="duplicate record_id"):
        merge([first, second])


def test_merge_requires_two_inputs(tmp_path: Path) -> None:
    first = _dataset(tmp_path / "a.json", "a", "impl-a")
    with pytest.raises(ValueError, match="at least two"):
        merge([first])
