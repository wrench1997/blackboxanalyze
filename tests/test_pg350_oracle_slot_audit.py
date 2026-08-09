from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg350_oracle_slots import audit


def test_oracle_slot_audit_is_complete_but_not_a_training_license() -> None:
    root = Path(__file__).parents[1]
    dataset = json.loads((root / "research/pg350_oracle_slot_source_rows_v1.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / "research/pg350_oracle_slot_vocabulary_v1.json").read_text(encoding="utf-8"))
    report = audit(dataset, vocabulary, dataset_sha256="a" * 64, vocabulary_sha256="b" * 64)
    assert report["status"] == "diagnostic_only"
    assert report["failures"] == []
    assert report["target_slot_coverage"] == {"payload_shape_ref": True, "oracle_ref": True, "negative_control_presence_ref": True}
    assert report["information_gate"]["accepted_training_rows"] == 0
    assert report["promotion"]["training_allowed"] is False
