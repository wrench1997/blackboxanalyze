from __future__ import annotations

from copy import deepcopy

from scripts.build_pg388_logic_rule_ir_composition_dataset import (
    SLOT_ORDER,
    audit_dataset,
    build_dataset,
)


def test_rule_ir_builder_emits_ordered_slots_and_keeps_summary_off_context() -> None:
    artifact = build_dataset()
    assert artifact["status"] == "abstract_rule_ir_composition_candidate_only"
    assert artifact["counts"]["records"] == 840
    assert artifact["counts"]["train"] == 420
    assert artifact["counts"]["implementation_holdout"] == 420
    assert artifact["counts"]["slot_count"] == 11
    assert artifact["training_eligible"] == 0
    assert artifact["source_contract"]["row_bound_typed_evidence"] is False
    for row in artifact["rows"]:
        assert row["target_tokens"] == [f"{slot}=" + row["target_tokens"][i].split("=", 1)[1] for i, slot in enumerate(SLOT_ORDER)]
        assert set(row["evaluator_summary"]) <= {"effect_shape", "state_delta", "invariant_result"}
        assert all("effect_shape=" not in token for token in row["target_tokens"])
        assert all("http://" not in token and "payload=" not in token for token in row["context_tokens"])


def test_rule_ir_audit_passes_but_keeps_promotion_closed() -> None:
    report = audit_dataset(build_dataset())
    assert report["status"] == "passed_candidate_rule_ir_audit"
    assert report["invalid_rows"] == 0
    assert report["training_eligible"] == 0
    assert report["promotion"]["training_allowed"] is False


def test_rule_ir_audit_rejects_reordered_or_tampered_target() -> None:
    artifact = deepcopy(build_dataset())
    artifact["rows"][0]["target_tokens"] = list(reversed(artifact["rows"][0]["target_tokens"]))
    report = audit_dataset(artifact)
    assert report["status"] == "blocked_rule_ir_audit"
    assert report["invalid_rows"] >= 1


def test_rule_ir_builder_rejects_raw_marker_in_source_context() -> None:
    artifact = build_dataset()
    artifact["rows"][0]["context_tokens"].append("payload=forbidden")
    report = audit_dataset(artifact)
    assert report["status"] == "blocked_rule_ir_audit"
    assert report["invalid_rows"] >= 1
