import json
from pathlib import Path

from scripts.run_pg349_a800_payload_probe_candidate import _decision_gate, constrained_summary


ROOT = Path(__file__).parents[1]


def test_decision_gate_accepts_only_clean_audit():
    audit = json.loads((ROOT / "research" / "pg349_decision_boundary_audit_v1.json").read_text(encoding="utf-8"))
    gate = _decision_gate(audit)
    assert gate["passed"] is True
    assert not gate["failures"]


def test_constrained_summary_is_abstract_and_fail_closed():
    dataset = json.loads((ROOT / "research" / "pg349_dynamic_typed_source_rows_v5.json").read_text(encoding="utf-8"))
    summary = constrained_summary(dataset)
    assert summary["rows"] == 2600
    assert summary["raw_payload_in_output"] is False
    assert summary["constrained_false_allow"] == 0
    assert summary["promotion"]["payload_catalog_promotion_allowed"] is False
