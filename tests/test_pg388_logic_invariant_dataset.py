from __future__ import annotations

import json
from pathlib import Path

from app.pg388_logic_invariant_projection import LOGIC_CASES, ROLES


DATASET = Path("research/pg388_logic_invariant_dataset_v1.json")


def _rows() -> list[dict]:
    artifact = json.loads(DATASET.read_text(encoding="utf-8"))
    return artifact["rows"]


def test_dataset_has_balanced_logic_matrix_and_is_candidate_only() -> None:
    artifact = json.loads(DATASET.read_text(encoding="utf-8"))
    counts = artifact["counts"]
    assert counts["records"] == len(LOGIC_CASES) * 2 * 4 * 5 * len(ROLES)
    assert counts["train"] == counts["implementation_holdout"]
    assert counts["cases"] >= 40
    assert counts["roles"] == len(ROLES)
    assert artifact["status"] == "abstract_logic_candidate_only"
    assert artifact["training_eligible"] == 0
    assert artifact["promotion"]["training_allowed"] is False


def test_dataset_split_and_context_firewall_are_explicit() -> None:
    artifact = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = _rows()
    assert {row["split"] for row in rows} == {"train", "implementation_holdout"}
    assert {row["implementation_ref"] for row in rows} == {"logic_fixture_a", "logic_fixture_b"}
    for row in rows[:64]:
        assert row["raw_source_stored"] is False
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert row["oracle_answer_in_context"] is False
        assert row["training_eligible"] is False
        text = json.dumps({"context": row["context_tokens"], "target": row["target_tokens"], "logic": row["logic_context"]}, ensure_ascii=False).casefold()
        for marker in ("http://", "https://", "payload", "wire", "response_body", "credential", "<script"):
            assert marker not in text
    assert artifact["context_firewall"]["external_network"] is False


def test_logic_dataset_preserves_feedback_and_negative_roles() -> None:
    rows = _rows()
    assert {row["feedback_state"] for row in rows} == {"baseline", "missing", "invariant_mismatch", "state_mismatch", "typed_effect"}
    negatives = [row for row in rows if row["role"] == "negative"]
    assert negatives
    assert all("next_action=abstain" in row["target_tokens"] for row in negatives)
    repairs = [row for row in rows if row["feedback_state"] == "invariant_mismatch" and row["role"] == "candidate"]
    assert any("next_action=repair" in row["target_tokens"] for row in repairs)
