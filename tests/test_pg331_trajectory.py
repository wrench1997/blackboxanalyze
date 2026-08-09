from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

from app.pg331_source_row import collect_pg331_source_row
from app.pg331_trajectory import audit_pg331_trajectory


_SPEC = importlib.util.spec_from_file_location("pg331_source_row_test_helpers", Path(__file__).with_name("test_pg331_source_row.py"))
assert _SPEC and _SPEC.loader
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def _row(*, method: str, record_id: str, evidence: str, failure: bool = False):
    observation = deepcopy(_HELPERS._observation())
    observation["request_transport"]["method"] = method
    observation["request_transport"]["placement"] = "query" if method == "GET" else "form"
    if failure:
        observation["failure_feedback"] = {
            "failure_class": "parse_error",
            "failure_stage": "request",
            "error_shape": "alpha",
            "previous_action": "select_probe_variant",
            "next_action": "select_probe_variant",
            "repair_delta_axis": "none",
            "repair_outcome": "failed",
        }
    evaluator = deepcopy(_HELPERS._evaluator())
    evaluator["evidence_hash"] = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
    target = deepcopy(_HELPERS._target())
    return collect_pg331_source_row(
        record_id=record_id,
        observation=observation,
        source_meta=_HELPERS._source_meta(),
        reset=_HELPERS._reset(),
        evaluator=evaluator,
        field_capture_manifest=_HELPERS._field_capture_manifest(),
        target_projection=target,
        split="train",
        operator_reviewed=True,
    )


def test_trajectory_accepts_ordered_get_post_triplet_without_promotion() -> None:
    steps = [
        {"step_index": 0, "action_role": "candidate_request", "row": _row(method="GET", record_id="r0", evidence="e0")},
        {"step_index": 1, "action_role": "reference_request", "row": _row(method="POST", record_id="r1", evidence="e1")},
        {"step_index": 2, "action_role": "negative_request", "row": _row(method="POST", record_id="r2", evidence="e2")},
    ]
    result = audit_pg331_trajectory(steps, require_get_post=True, require_triplet=True)
    assert result["valid"] is True
    assert result["get_post_pair"] is True
    assert result["typed_evidence_unique"] is True
    assert result["trajectory_training_eligible"] is True
    assert result["promotion"]["training_allowed"] is False


def test_trajectory_rejects_duplicate_typed_evidence_within_role() -> None:
    steps = [
        {"step_index": 0, "action_role": "candidate_request", "row": _row(method="GET", record_id="r0", evidence="same")},
        {"step_index": 1, "action_role": "candidate_request", "row": _row(method="POST", record_id="r1", evidence="same")},
    ]
    result = audit_pg331_trajectory(steps)
    assert result["valid"] is False
    assert "evidence_reused_same_role:1" in result["failures"]


def test_trajectory_marks_missing_evaluator_as_ask_and_not_trainable() -> None:
    evaluator = deepcopy(_HELPERS._evaluator())
    evaluator["typed_available"] = False
    row = collect_pg331_source_row(
        record_id="missing-evaluator",
        observation=_HELPERS._observation(),
        source_meta=_HELPERS._source_meta(),
        reset=_HELPERS._reset(),
        evaluator=evaluator,
        field_capture_manifest=_HELPERS._field_capture_manifest(),
        target_projection=_HELPERS._target(),
        operator_reviewed=True,
    )
    result = audit_pg331_trajectory([{"step_index": 0, "action_role": "ask", "row": row}])
    assert result["valid"] is False
    assert result["ask_count"] == 1
    assert result["trajectory_training_eligible"] is False


def test_trajectory_rejects_stuck_failure_and_step_gap() -> None:
    row = _row(method="GET", record_id="r0", evidence="e0", failure=True)
    result = audit_pg331_trajectory([{"step_index": 1, "action_role": "repair", "row": row}])
    assert result["valid"] is False
    assert "step_index:0" in result["failures"]
    assert "failure_action_not_changed:0" in result["failures"]

