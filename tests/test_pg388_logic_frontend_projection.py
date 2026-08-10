import json
from pathlib import Path

from scripts.project_pg388_logic_frontend_summary import build_summary


def test_pg388_frontend_projection_is_bounded_and_fail_closed():
    summary = build_summary()
    assert summary["status"] == "diagnostic_rule_ir_candidate"
    assert summary["dataset"]["records"] == 840
    assert summary["dataset"]["split_counts"] == {"implementation_holdout": 420, "train": 420}
    assert summary["dataset"]["slot_order"][-1] == "safe_to_send"
    assert summary["audit"]["status"] == "passed_candidate_rule_ir_audit"
    assert summary["audit"]["context_firewall_passed"] is True
    assert summary["gates"]["capability_training_allowed"] is False
    assert summary["gates"]["training_eligible"] is False
    assert summary["live_evidence"]["row_bound"] is True
    assert summary["live_evidence"]["audit_status"] == "passed_candidate_logic_rule_ir_source_row_audit"
    assert summary["gates"]["fresh_role_reset_attested"] is True
    assert summary["gates"]["operator_reviewed"] is False
    assert summary["plan"]["optimizer_started"] is False
    assert summary["independent_holdout"]["source_rows"]["strict_valid"] == 140
    assert summary["independent_holdout"]["docker_smoke"]["health_http_status"] == 200
    assert summary["independent_holdout"]["docker_smoke"]["negative_control_clean"] is True
    assert summary["independent_holdout"]["gates"]["docker_smoke_observed"] is True
    assert summary["independent_holdout"]["gates"]["training_eligible"] is False
    assert summary["candidate_model"]["status"] == "candidate_only_projection"
    assert len(summary["candidate_model"]["runs"]) == 7
    assert summary["candidate_model"]["runs"][0]["vocabulary_scope"] == "train_context_only"
    assert summary["candidate_model"]["runs"][0]["execution"]["device"] == "cpu"
    assert summary["candidate_model"]["runs"][3]["label"] == "11-slot composition"
    assert summary["candidate_model"]["runs"][3]["holdout_count"] == 128
    assert summary["candidate_model"]["runs"][4]["label"] == "11-slot composition (full CPU)"
    assert summary["candidate_model"]["runs"][4]["holdout_count"] == 420
    assert summary["candidate_model"]["runs"][5]["label"] == "11-slot composition (full CPU e8)"
    assert summary["candidate_model"]["runs"][5]["holdout_count"] == 420
    assert summary["candidate_model"]["runs"][5]["weakest_head"]["accuracy"] >= 0.96
    assert summary["candidate_model"]["runs"][5]["holdout_ask_recall"] == 1.0
    assert summary["candidate_model"]["runs"][5]["holdout_composition_exact"] == 0.964286
    assert summary["candidate_model"]["runs"][5]["holdout_slot_accuracy"] == 0.994156
    assert summary["candidate_model"]["runs"][5]["holdout_repair_recall"] == 1.0
    assert summary["candidate_model"]["runs"][6]["label"] == "11-slot composition (local CUDA e8)"
    assert summary["candidate_model"]["runs"][6]["holdout_composition_exact"] == 0.992857
    assert summary["candidate_model"]["runs"][6]["holdout_slot_accuracy"] == 0.998701
    assert summary["candidate_model"]["runs"][6]["execution"]["device"] == "cuda:0"
    assert summary["candidate_model"]["runs"][6]["execution"]["gpu_touched"] is True
    assert summary["candidate_model"]["latest_label"] == "11-slot composition (local CUDA e8)"
    assert summary["candidate_model"]["training_allowed"] is False
    assert summary["taxonomy_coverage"]["case_count"] == 66
    assert summary["taxonomy_coverage"]["category_count"] == 10
    assert summary["taxonomy_coverage"]["missing_anchor_count"] == 0
    assert summary["taxonomy_coverage"]["candidate_only_count"] == 10
    assert summary["supplemental_canary"]["audit_status"] == "passed_candidate_only"
    assert summary["supplemental_canary"]["counts"]["role_rows"] == 120
    assert summary["supplemental_canary"]["counts"]["fresh_resets_before"] == 120
    assert summary["supplemental_canary"]["counts"]["fresh_resets_after"] == 120
    assert summary["supplemental_canary"]["counts"]["negative_violation"] == 0
    assert summary["supplemental_canary"]["execution"]["in_process_only"] is True
    assert summary["supplemental_canary"]["execution"]["docker_started"] is False
    assert summary["gates"]["supplemental_canary_audit"] is True
    assert summary["typed_supplement"]["audit_status"] == "passed_diagnostic_only"
    assert summary["typed_supplement"]["counts"]["records"] == 120
    assert summary["typed_supplement"]["counts"]["train"] == 0
    assert summary["typed_supplement"]["counts"]["typed_evaluator_observed"] == 120
    assert summary["typed_supplement"]["context_firewall_passed"] is True
    assert summary["gates"]["typed_supplement_audit"] is True
    assert summary["information_preservation"]["status"] == "blocked_information_gate"
    assert summary["information_preservation"]["sequence_diversity"]["context"]["unique_ratio"] == 0.007143
    assert summary["information_preservation"]["axis_presence"]["zero_entropy_axis_count"] == 7
    assert summary["gates"]["information_preservation_audit"] is False
    assert summary["information_preservation"]["composition_dataset"]["split_counts"] == {"implementation_holdout": 420, "train": 420}
    assert summary["information_preservation"]["composition_dataset"]["context_unique_ratio"]["train"] == 0.333333
    assert summary["information_preservation"]["composition_dataset"]["cross_split_target_overlap"] == 140
    assert summary["information_preservation"]["composition_dataset"]["contract_passed"] is False
    assert summary["information_preservation"]["surface_dataset"]["split_counts"] == {"implementation_holdout": 72, "train": 72}
    assert summary["information_preservation"]["surface_dataset"]["context_unique_ratio"]["train"] == 0.166667
    assert summary["information_preservation"]["surface_dataset"]["cross_split_context_overlap"] == 0
    assert summary["information_preservation"]["surface_dataset"]["axis_observed_count"] == 7
    assert summary["information_preservation"]["surface_dataset"]["contract_passed"] is False


def test_pg388_frontend_projection_contains_no_rows_or_raw_markers():
    serialized = json.dumps(build_summary(), ensure_ascii=False).casefold()
    for marker in (
        '"rows"',
        "context_tokens",
        "target_tokens",
        "record_ref_sha256",
        "row_sha256",
        "payload=",
        "wire=",
        "response_body",
        "oracle_answer",
        "evaluator_answer",
        "http://",
        "https://",
        "container_id",
        "image_id",
        '"context_tokens"',
    ):
        assert marker not in serialized
