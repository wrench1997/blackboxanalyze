import hashlib
import json
import re
from pathlib import Path

from app.active_goal_label_inducer import (
    ActiveGoalLabelInducer,
    REQUIRED_COMPOSITION_ATOMS,
    compose_rule_ir,
)
from app.probe_binding_attestation import (
    CANONICAL_BINDING_SHA256,
    add_binding_attestation,
    binding_attestation_valid,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg104_keeps_real_observability_gap_and_passes_composition_audits():
    report = _load("pg104_probe_binding_ablation_report_v1.json")
    proposal = _load("pg104_probe_binding_ablation_proposal_v1.json")
    assert report["status"] == "blocked"
    guarded = report["metrics"]["guarded_proposal"]["all_evaluation"]
    assert guarded["known_confirm_recall"] == 1.0
    assert guarded["known_label_consistency"] == 1.0
    assert guarded["false_accept_count"] == 0
    assert report["metrics"]["guarded_proposal"]["pg69_additional_unseen_implementation"]["unknown_family_strict_abstain"] is False
    observability = report["metrics"]["unknown_observability"]["pg69_additional_unseen_implementation"]
    assert observability["observable_unknown_strict_abstain"] is True
    assert observability["unobservable_unknown_positive_count"] == 2
    assert observability["unobservable_unknown_nonconfirm"] is True
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False

    composition = report["metrics"]["compositional_rule_ir_ablation"]
    assert composition["required_atoms"] == list(REQUIRED_COMPOSITION_ATOMS)
    assert composition["copy_paste_order_invariant"] is True
    assert composition["cross_sample_recombination_order_invariant"] is True
    assert composition["cross_sample_recombination_executable"] is False
    assert composition["family_free"] is True
    assert composition["candidate_waits_for_typed_oracle_rate"] == 1.0
    assert composition["candidate_promotion_eligible_count"] == 0

    contract = proposal["composition_contract"]
    assert contract["family_classification_forbidden"] is True
    assert contract["typed_oracle_required"] is True
    assert contract["executable"] is False
    assert contract["rule_ir"]["executable"] is False
    assert contract["rule_ir"]["atoms"] == sorted(REQUIRED_COMPOSITION_ATOMS)
    text = json.dumps({"report": report, "proposal": proposal}, ensure_ascii=False).casefold()
    assert "workflow_invariant" not in text
    assert "xss" not in text
    assert "union select" not in text


def test_pg104_visible_replay_is_attested_and_evaluator_blind():
    report = _load("pg104_probe_binding_ablation_report_v1.json")
    dataset = _load("pg104_probe_binding_ablation_visible_dataset_v1.json")
    trace = _load("pg104_probe_binding_ablation_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert trace["evaluator_labels_in_trace"] is False
    assert len(dataset["rows"]) == 562
    assert len(trace["steps"]) == 562
    assert sorted({row["method"] for row in dataset["rows"]}) == ["GET", "POST"]
    assert len({row["fresh_reset"]["target_instance_id"] for row in dataset["rows"]}) == 562
    assert all(row["negative_control_matched"] for row in dataset["rows"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"]) for row in dataset["rows"])
    assert all(binding_attestation_valid(row["model_input"]) for row in dataset["rows"])
    assert all(not row["raw_probe_strings_stored"] and not row["raw_response_body_stored"] for row in dataset["rows"])
    assert dataset["long_term_memory_write"] is False
    assert trace["long_term_memory_write"] is False
    for row in dataset["rows"]:
        assert not any(key in row["model_input"] for key in ("family", "oracle", "raw_body", "route_template_id"))
        assert row["guarded_proposal"]["promotion_eligible"] is False
    for name, digest in report["source"]["source_hashes"].items():
        path = {
            "pg101_dataset": "research/pg101_active_probe_signature_visible_dataset_v1.json",
            "pg103_dataset": "research/pg103_auto_goal_label_active_probe_visible_dataset_v1.json",
            "pg103_trace": "research/pg103_auto_goal_label_active_probe_trace_v1.json",
            "pg69_fixture": "app/pg69_workflow_fixture.py",
            "binding_module": "app/probe_binding_attestation.py",
            "inducer_module": "app/active_goal_label_inducer.py",
            "runner": "scripts/run_pg104_probe_binding_ablation.py",
        }[name]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_composition_is_copy_paste_order_invariant_and_missing_binding_does_not_promote():
    pg101 = _load("pg101_active_probe_signature_visible_dataset_v1.json")
    train = [
        {"model_input": add_binding_attestation(row["model_input"])}
        for row in pg101["rows"]
        if row["role"] == "train"
    ]
    inducer = ActiveGoalLabelInducer(require_binding_attestation=True).fit(train)
    source = train[0]["model_input"]
    output = inducer.predict(source, guarded=True)
    assert output["promotion_eligible"] is False
    if output["decision"] == "confirm_candidate":
        assert output["composition_decision"] == "await_typed_oracle"
        assert "get_post_repeat" in output["composition"]["missing_atoms"]
        assert "negative_control_clear" in output["composition"]["missing_atoms"]

    missing = json.loads(json.dumps(source, ensure_ascii=False))
    missing.pop("probe_binding", None)
    guarded = inducer.predict(missing, guarded=True)
    assert guarded["decision"] == "abstain"
    assert guarded["composition_decision"] == "abstain"
    assert guarded["promotion_eligible"] is False

    first = compose_rule_ir(["effect_present", "probe_binding_valid", "supported_active_slot:p0"])
    second = compose_rule_ir(["supported_active_slot:p0", "effect_present", "probe_binding_valid", "effect_present"])
    assert first == second
    assert first["executable"] is False
    assert first["canonical_sha256"] == hashlib.sha256(
        json.dumps(first["expression"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

