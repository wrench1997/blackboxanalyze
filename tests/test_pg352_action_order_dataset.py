from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg351_ask_oracle_composition import audit
from scripts.build_pg352_action_order_dataset import TARGET_ORDER, build


ROOT = Path(__file__).resolve().parents[1]


def test_pg352_moves_only_decision_prefix_and_preserves_all_slots() -> None:
    source_path = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = build(source, source_sha256="a" * 64)
    assert result["status"] == "diagnostic_candidate_only"
    assert result["counts"] == {
        "records": 1832,
        "train_rows": 1152,
        "implementation_holdout_rows": 680,
        "training_eligible_rows": 0,
        "raw_payload_in_context": False,
    }
    for row in result["records"]:
        keys = [token.split("=", 1)[0] for token in row["target_tokens"][1:-1]]
        assert keys == [key for key in TARGET_ORDER if key in keys]
        assert len(keys) == len(set(keys))
        assert all("payload=" not in token.casefold() for token in row["context_tokens"] + row["target_tokens"])


def test_pg352_audit_keeps_safe_negative_and_ask_boundary() -> None:
    source_path = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = build(source, source_sha256="a" * 64)
    audited = audit(result, dataset_sha256="b" * 64)
    assert audited["status"] == "diagnostic_candidate_only"
    ask = [row for row in result["records"] if row["supervision_lane"] == "ask_missing_observation"]
    assert ask and all(row["safe_to_send"] is False and "safe_to_send=0" in row["target_tokens"] for row in ask)
    assert audited["failures"] == []
