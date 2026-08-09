from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg351_ask_oracle_composition import audit
from scripts.build_pg351_ask_oracle_composition_dataset import build


ROOT = Path(__file__).resolve().parents[1]


def test_pg351_merge_preserves_ask_and_typed_targets_without_raw_literals() -> None:
    typed_path = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
    ask_path = ROOT / "research" / "pg348_dynamic_context_dataset_v1.json"
    typed = json.loads(typed_path.read_text(encoding="utf-8"))
    ask = json.loads(ask_path.read_text(encoding="utf-8"))
    result = build(typed, ask, typed_sha256="a" * 64, ask_sha256="b" * 64)
    assert result["status"] == "diagnostic_candidate_only"
    assert result["counts"]["records"] == 1832
    assert result["supervision_lanes"] == {"ask_missing_observation": 992, "typed_replay": 840}
    assert result["counts"]["training_eligible_rows"] == 0
    assert any(token == "question=ask_typed" for row in result["records"] for token in row["target_tokens"])
    assert any(token == "oracle_ref=unknown" for row in result["records"] for token in row["target_tokens"])
    ask_rows = [row for row in result["records"] if row["supervision_lane"] == "ask_missing_observation"]
    assert ask_rows and all("safe_to_send=0" in row["target_tokens"] and row["safe_to_send"] is False for row in ask_rows)
    typed_rows = [row for row in result["records"] if row["supervision_lane"] == "typed_replay"]
    assert typed_rows and any("safe_to_send=0" in row["target_tokens"] for row in typed_rows)
    serialized = json.dumps(result, ensure_ascii=False)
    assert "raw_payload=" not in serialized
    assert "http://" not in serialized


def test_pg351_audit_requires_both_ask_and_rule_ir_action_coverage() -> None:
    typed_path = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
    ask_path = ROOT / "research" / "pg348_dynamic_context_dataset_v1.json"
    result = build(
        json.loads(typed_path.read_text(encoding="utf-8")),
        json.loads(ask_path.read_text(encoding="utf-8")),
        typed_sha256="a" * 64,
        ask_sha256="b" * 64,
    )
    audit_result = audit(result, dataset_sha256="c" * 64)
    assert audit_result["status"] == "diagnostic_candidate_only"
    assert audit_result["information_gate"]["ask_supervision_present"] is True
    assert audit_result["information_gate"]["typed_rule_ir_present"] is True
    assert audit_result["information_gate"]["accepted_training_rows"] == 0
    assert audit_result["failures"] == []
