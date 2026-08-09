from __future__ import annotations

from copy import deepcopy
import json

from app.pg331_source_row import collect_pg331_source_row
from scripts.select_pg331_audited_rows import materialize_valid_rows, select_audited_rows


def _row(record_id: str, split: str) -> dict[str, object]:
    from tests.test_pg331_source_row import _evaluator, _field_capture_manifest, _observation, _reset, _source_meta, _target
    return collect_pg331_source_row(record_id=record_id, observation=_observation(), source_meta=_source_meta(), reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), split=split, operator_reviewed=True)


def test_selection_keeps_only_valid_references_and_preserves_split_groups() -> None:
    valid = _row("valid-row", "train")
    invalid = deepcopy(_row("incomplete-row", "implementation_holdout"))
    invalid["context_tokens"] = []
    document = {"records": [valid, invalid]}
    selected = select_audited_rows(document, {"schema_version": "pg331a-source-row-audit-v1", "status": "blocked"}, dataset_sha256="a" * 64, source_audit_sha256="b" * 64)
    assert selected["status"] == "diagnostic_valid_rows_only"
    assert selected["counts"]["valid_rows"] == 1
    assert selected["counts"]["excluded_rows"] == 1
    assert selected["counts"]["split_counts"] == {"implementation_holdout": 1, "train": 1}
    assert selected["split_relabelled"] is False
    assert all(value is False for value in selected["promotion"].values())
    encoded = json.dumps(selected, ensure_ascii=False)
    assert "valid-row" not in encoded and "incomplete-row" not in encoded and "context_tokens" not in encoded
    assert selected["excluded_reason_counts"]["missing_presence:belief_replay_presence"] == 1


def test_selection_does_not_accept_incomplete_ask_rows_as_valid() -> None:
    row = _row("ask-row", "train")
    row["context_tokens"] = [token for token in row["context_tokens"] if not token.startswith("javascript_presence=")]
    selected = select_audited_rows({"records": [row]}, {})
    assert selected["counts"]["valid_rows"] == 0
    assert selected["counts"]["excluded_rows"] == 1
    assert selected["training_eligible"] is False


def test_explicit_materialization_keeps_only_valid_abstract_rows() -> None:
    valid = _row("valid-materialized", "train")
    invalid = deepcopy(_row("invalid-materialized", "implementation_holdout"))
    invalid["context_tokens"] = []
    default = select_audited_rows({"records": [valid, invalid]}, {})
    materialized = materialize_valid_rows({"records": [valid, invalid]}, {}, dataset_sha256="e" * 64, source_audit_sha256="f" * 64)
    assert "records" not in default
    assert materialized["status"] == "diagnostic_valid_rows_materialized"
    assert materialized["counts"]["materialized_valid_rows"] == 1
    assert len(materialized["records"]) == 1
    assert materialized["records"][0]["split"] == "train"
    encoded = json.dumps([row["context_tokens"] + row["target_tokens"] for row in materialized["records"]], ensure_ascii=False).casefold()
    for forbidden in ("payload", "raw_", "response_body=", "oracle=", "evaluator="):
        assert forbidden not in encoded
    assert all(value is False for value in materialized["promotion"].values())
    assert materialized["records"][0]["training_eligible"] is False
