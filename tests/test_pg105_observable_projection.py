import hashlib
import json
import re
from pathlib import Path

from app.active_goal_label_inducer import ActiveGoalLabelInducer
from app.pg105_observable_projection import (
    SCHEMA_VERSION,
    attach_causal_extension,
    make_causal_projection,
)
from app.probe_binding_attestation import add_binding_attestation, binding_attestation_valid


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg105_repairs_opaque_abstention_without_claiming_vulnerability_detection():
    report = _load("pg105_observable_projection_report_v1.json")
    assert report["status"] == "passed_observable_projection_diagnostic"
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    guarded = report["metrics"]["guarded_proposal"]["all_evaluation"]
    assert guarded["known_confirm_recall"] == 1.0
    assert guarded["false_accept_count"] == 0
    assert report["metrics"]["guarded_proposal"]["pg69_observable_projection"]["unknown_family_strict_abstain"] is True
    causal = report["metrics"]["causal_extension"]
    assert causal["positive_opaque_count"] == 2
    assert causal["positive_opaque_anomaly_present"] is True
    assert causal["negative_anomaly_count"] == 0
    assert causal["opaque_relation_is_generic_code_1"] is True
    assert report["safety"]["input_change_is_not_effect_atom"] is True
    composition = report["metrics"]["compositional_rule_ir_ablation"]
    assert composition["candidate_promotion_eligible_count"] == 0
    assert composition["cross_sample_recombination_executable"] is False


def test_pg105_artifacts_are_bounded_fresh_and_oracle_blind():
    report = _load("pg105_observable_projection_report_v1.json")
    dataset = _load("pg105_observable_projection_visible_dataset_v1.json")
    trace = _load("pg105_observable_projection_trace_v1.json")
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
    assert all(
        row["model_input"]["causal_extension"]["schema_version"] == SCHEMA_VERSION
        for row in dataset["rows"]
        if row["source"] == "pg69"
    )
    assert all(not row["raw_probe_strings_stored"] and not row["raw_response_body_stored"] for row in dataset["rows"])
    assert dataset["long_term_memory_write"] is False
    text = json.dumps({"report": report, "dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("workflow_invariant", "xss", "union select", "<script", "onerror", "amount", "member"):
        assert forbidden not in text
    for name, digest in report["source"]["source_hashes"].items():
        path = {
            "pg101_dataset": "research/pg101_active_probe_signature_visible_dataset_v1.json",
            "pg103_dataset": "research/pg103_auto_goal_label_active_probe_visible_dataset_v1.json",
            "pg103_trace": "research/pg103_auto_goal_label_active_probe_trace_v1.json",
            "pg69_fixture": "app/pg69_workflow_fixture.py",
            "causal_module": "app/pg105_observable_projection.py",
            "inducer_module": "app/active_goal_label_inducer.py",
            "runner": "scripts/run_pg105_observable_projection.py",
        }[name]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_pg105_input_response_relation_is_bounded_and_does_not_become_effect_present():
    opaque = make_causal_projection({"amount": "99"}, {"amount": "100"}, response_changed=False)
    assert opaque["input_changed"] is True
    assert opaque["response_changed"] is False
    assert opaque["input_changed_response_unchanged"] is True
    assert opaque["relation_code"] == 1
    assert "amount" not in json.dumps(opaque, ensure_ascii=False)
    ordinary = make_causal_projection({"value": "foo"}, {"value": "bar"}, response_changed=False)
    assert ordinary["input_changed"] is False
    assert ordinary["relation_code"] == 0

    pg101 = _load("pg101_active_probe_signature_visible_dataset_v1.json")
    source = next(row["model_input"] for row in pg101["rows"] if row["role"] == "train")
    extension = [
        make_causal_projection({"x": "0"}, {"x": "0"}, response_changed=bool(changed))
        for changed in source["delta_pattern"]
    ]
    extended = add_binding_attestation(attach_causal_extension(source, extension))
    assert binding_attestation_valid(extended)
    assert extended["attention_pattern"] == [bool(value) for value in source["delta_pattern"]]
    inducer = ActiveGoalLabelInducer(require_binding_attestation=True).fit([{"model_input": extended}])
    assert inducer.predict(extended, guarded=True)["promotion_eligible"] is False


def test_pg105_opaque_rows_abstain_with_generic_anomaly_atom():
    dataset = _load("pg105_observable_projection_visible_dataset_v1.json")
    pg101 = _load("pg101_active_probe_signature_visible_dataset_v1.json")
    train = [
        {"model_input": add_binding_attestation(row["model_input"])}
        for row in pg101["rows"]
        if row["role"] == "train"
    ]
    inducer = ActiveGoalLabelInducer(require_binding_attestation=True).fit(train)
    opaque_rows = [
        row for row in dataset["rows"]
        if row["source"] == "pg69"
        and not any(row["model_input"]["delta_pattern"])
        and any(row["model_input"]["causal_extension"]["input_changed_response_unchanged_pattern"])
    ]
    assert len(opaque_rows) == 2
    for row in opaque_rows:
        output = inducer.predict(row["model_input"], guarded=True)
        assert output["decision"] == "abstain"
        assert "candidate_without_surface_delta" in output["composition"]["observed_atoms"]
        assert "effect_present" not in output["composition"]["observed_atoms"]
        assert output["promotion_eligible"] is False


def test_pg105_supported_slot_decoy_input_change_cannot_confirm_without_surface_effect():
    pg101 = _load("pg101_active_probe_signature_visible_dataset_v1.json")
    train = [
        {"model_input": add_binding_attestation(row["model_input"])}
        for row in pg101["rows"]
        if row["role"] == "train"
    ]
    source = json.loads(json.dumps(train[0]["model_input"], ensure_ascii=False))
    source["delta_pattern"] = [False] * 9
    source["geometry_sign_pattern"] = [[0] * 11 for _ in range(9)]
    projections = [
        make_causal_projection({"value": "9"}, {"value": "10"}, response_changed=False),
        *[
            make_causal_projection({"value": "0"}, {"value": "0"}, response_changed=False)
            for _ in source["delta_pattern"][1:]
        ],
    ]
    decoy = add_binding_attestation(attach_causal_extension(source, projections))
    inducer = ActiveGoalLabelInducer(require_binding_attestation=True).fit(train)
    output = inducer.predict(decoy, guarded=True)
    assert output["active_slots"] == ["p0"]
    assert output["decision"] == "abstain"
    assert output["reason"] == "input_changed_without_surface_effect"
    assert output["promotion_eligible"] is False
