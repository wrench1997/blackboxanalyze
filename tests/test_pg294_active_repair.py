import json
from pathlib import Path

from app.pg294_active_repair import audit_records, project_observable_context, state_target


ROOT = Path(__file__).resolve().parents[1]


def test_missing_observation_is_a_question_not_a_final_answer():
    result = state_target(False, typed_available="unknown", feedback_state="unknown", replay_ready="unknown", evidence_present="unknown")
    assert result == ("recheck_oracle", "recheck_oracle", "ask_typed_availability", False)


def test_oracle_verdict_shortcuts_are_removed_from_context():
    projected = project_observable_context([
        "[BOS]",
        "method=POST",
        "result_verified=1",
        "replay_expected=typed",
        "failure=typed_effect",
        "lane=gold",
        "typed_available=unknown",
        "[EOS]",
    ])
    joined = " ".join(projected)
    assert "result_verified" not in joined
    assert "replay_expected" not in joined
    assert "typed_effect" not in joined
    assert "lane=" not in joined
    assert "method=POST" in joined


def test_pg294_dataset_audit_and_state_cells_pass():
    dataset = json.loads((ROOT / "research" / "pg294_active_repair_dataset_v1.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "research" / "pg294_active_repair_dataset_audit_v1.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert set(dataset["counts"]["state_cells"]) >= {"missing_key", "progress", "unavailable"}
    assert dataset["contract"]["context_excludes_oracle_verdict"] is True
    assert all(row["oracle_label_in_context"] is False for row in dataset["records"])
    assert all(not row["hard_negative"] or row["training_eligible"] is False for row in dataset["records"])
    assert audit_records(dataset["records"])["status"] == "passed"
