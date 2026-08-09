import json
from pathlib import Path

from app.pg230_next_token_quality_funnel import event_tokens, prepare_record, quality_lane


ROOT = Path(__file__).resolve().parents[1]


def _base_row() -> dict:
    return {
        "source": "unit-test-local-lab",
        "method": "GET",
        "seed": 23001,
        "evidence_hash": "a" * 64,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "candidate_sent": True,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "status_class": "2xx",
        "oracle_available": True,
        "typed_effect_confirmed": True,
        "candidate_reference_agreement": True,
        "negative_clean": True,
    }


def test_quality_funnel_separates_gold_silver_and_self_error() -> None:
    lane, reasons = quality_lane(_base_row())
    assert lane == "gold"
    assert reasons == ["typed_oracle_reference_negative_complete"]

    silver = _base_row()
    silver["oracle_available"] = False
    silver["typed_effect_confirmed"] = False
    silver["candidate_reference_agreement"] = False
    silver["negative_clean"] = False
    silver_lane, _ = quality_lane(silver)
    assert silver_lane == "silver"

    self_error = _base_row()
    self_error["model_self_error_detected"] = True
    self_error["model_gate_corrected_diagnosis"] = "confirmed_local_effect"
    self_error_lane, _ = quality_lane(self_error)
    assert self_error_lane == "hard_negative"
    record = prepare_record(self_error)
    assert record["repair_action"] == "gate_correction"
    assert "failure=model_self_error" in event_tokens(self_error, self_error_lane)
    assert "repair=gate_correction" in event_tokens(self_error, self_error_lane)


def test_pg230_report_keeps_quality_gate_honest() -> None:
    report = json.loads((ROOT / "research" / "pg230_next_token_quality_funnel_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg230_next_token_quality_funnel_dataset_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_next_token_quality_funnel_frozen_xxl_training"
    assert report["funnel"]["raw_records"] == 176
    assert report["funnel"]["unique_records"] == 33
    assert report["funnel"]["duplicate_records"] == 143
    assert report["funnel"]["lane_counts"] == {"gold": 2, "silver": 12, "hard_negative": 7, "quarantine": 12}
    assert report["selected"]["holdout"]["self_error_recall"] == 0.5
    assert report["selected"]["holdout"]["lane_accuracy"] < 0.5
    assert report["selected"]["holdout"]["repair_accuracy"] < 0.5
    assert report["metrics"]["classification_context_excludes_lane_and_repair_targets"] is True
    assert any(item["failure"] == "lane_repair_target_leakage_through_last_hidden" for item in report["engineering_repairs"])
    assert report["frozen_body_changed"] is False
    assert report["promotion"]["quarantine_training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["metrics"]["next_token_loss_is_not_quality_gate"] is True
    assert "research\\pg222_problem_diagnoser_dataset_v1.json" in report["source_reports"]
    assert "research\\pg222_problem_diagnoser_dataset_v1.json" in dataset["source_reports"]
    assert dataset["contract"]["evaluator_targets_as_features"] is False
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False


def test_next_token_rule_requires_self_check_repair_lineage_and_append_only_storage() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))
    policy = rules["next_token_quality_funnel_policy"]
    required = policy["required_record"]
    assert required["prediction_top_k_token_ids_and_logprobs"] is True
    assert required["teacher_forcing_leakage_check"] is True
    assert required["self_check_trace"]["independent_checker_version"] is True
    assert required["failure_is_model_or_environment"] is True
    assert required["repair_delta_projection"] is True
    assert required["counterfactual_negative_projection"] is True
    assert required["parent_record_id"] is True
    assert policy["dataset_storage"]["record_immutable_after_ingest"] is True
    assert "replay_repair_or_abstention_on_fresh_reset" in policy["sieve_order"]
    assert policy["promotion_rules"]["next_token_loss_alone_can_promote"] is False


def test_next_token_sample_sieve_keeps_loss_and_quality_separate() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))
    policy = rules["next_token_quality_funnel_policy"]
    sieve = policy["sample_value_sieve"]
    selection = policy["model_selection_contract"]
    storage = policy["dataset_storage"]

    assert sieve["quality_score_contract"]["score_is_audit_only"] is True
    assert sieve["quality_score_contract"]["score_must_not_be_a_training_feature"] is True
    assert sieve["hard_gates"]["missing_core_field"] == "quarantine_incomplete"
    assert "minimal_repair_and_fresh_replay" in sieve["sieve_order"]
    assert "cross_family_and_implementation_holdout" in selection["primary_selection_metrics"]
    assert selection["next_token_role"].startswith("表示学习")
    assert selection["catastrophic_forgetting_gate"]["replay_old_canary_after_each_update"] is True
    assert selection["catastrophic_forgetting_gate"]["any_guardrail_regression_blocks_promotion"] is True
    assert storage["record_immutable_after_ingest"] is True
    assert "quality_sieve_report" in storage["required_artifacts"]
    assert storage["partition_files"]["hard_negative"] == "hard_negative.jsonl"


def test_explicit_oracle_gap_abstain_is_trainable_hard_negative() -> None:
    row = _base_row()
    row.update({
        "candidate_sent": False,
        "oracle_available": False,
        "typed_effect_confirmed": False,
        "candidate_reference_agreement": False,
        "negative_clean": True,
        "abstention_required": True,
        "failure_signature": "timing_channel_forbidden",
    })
    lane, reasons = quality_lane(row)
    assert lane == "hard_negative"
    assert reasons == ["explicit_oracle_gap_abstain"]
