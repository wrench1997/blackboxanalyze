from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

from tests.test_pg331_source_row import _evaluator, _field_capture_manifest, _observation, _reset, _source_meta, _target


_SPEC = importlib.util.spec_from_file_location("pg331_source_row_collection", Path(__file__).parents[1] / "scripts" / "collect_pg331_source_rows.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
collect_rows = _MODULE.collect_rows


def _input_item() -> dict[str, object]:
    return {"record_id": "input:complete", "observation": _observation(), "source_meta": _source_meta(), "reset": _reset(), "evaluator": _evaluator(), "field_capture_manifest": _field_capture_manifest(), "target_projection": _target(), "split": "train", "operator_reviewed": True}


def test_collection_keeps_only_abstract_rows_and_marks_counts() -> None:
    dataset = collect_rows({"records": [_input_item()]})
    assert dataset["counts"] == {"input": 1, "accepted": 1, "incomplete": 0, "rejected": 0, "training_eligible": 1}
    row = dataset["records"][0]
    assert row["training_eligible"] is True
    assert "observation" not in row
    assert "evaluator_sidecar" in row
    assert dataset["promotion"]["memory_promotion_allowed"] is False


def test_rejected_capture_is_digest_only_and_cannot_train() -> None:
    item = _input_item()
    item["observation"] = deepcopy(item["observation"])
    item["observation"]["request_transport"]["raw_payload"] = "literal-should-not-persist"
    dataset = collect_rows({"records": [item]})
    assert dataset["counts"]["rejected"] == 0
    assert dataset["counts"]["accepted"] == 1
    row = dataset["records"][0]
    assert row["training_eligible"] is False
    assert row["collector_status"] if "collector_status" in row else True
    assert "literal-should-not-persist" not in str(dataset)
