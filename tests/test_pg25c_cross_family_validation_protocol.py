import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "pg_pk_25c_cross_family_validation_protocol_v1.json"


def _protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_pg25c_is_offline_preregistered_and_fail_closed():
    protocol = _protocol()
    assert protocol["protocol_id"] == "pg-pk-25c-cross-family-validation-v1"
    assert protocol["schema_version"] == "sift-pg25c-cross-family-validation-protocol-v1"
    assert protocol["status"] == "preregistered_no_target_execution"
    assert protocol["scope"]["protocol_design_is_offline"] is True
    assert protocol["scope"]["target_execution_performed_by_this_protocol_commit"] is False
    assert protocol["scope"]["external_network"] is False
    assert protocol["scope"]["free_form_generation"] is False
    assert protocol["upstream_contracts"]["threshold_tuning_on_pg25c"] is False


def test_pg25c_record_contract_contains_only_bounded_references():
    protocol = _protocol()
    contract = protocol["record_contract"]
    required = set(contract["required_provenance_fields"])
    assert {
        "source_hash",
        "container_or_fixture_digest",
        "target_instance_id",
        "sampling_seed",
        "manifest_entry_id",
        "manifest_sha256",
        "reset_receipt_hash",
        "evidence_hash",
    } <= required
    forbidden_keys = {
        "raw_request",
        "raw_request_body",
        "raw_response_body",
        "secret",
        "credential",
        "hidden_evaluator_state",
    }
    assert forbidden_keys.isdisjoint(set(_walk_keys(protocol)))
    assert set(contract["forbidden_record_fields"]) == forbidden_keys


def test_pg25c_source_target_and_seed_isolation_is_explicit():
    isolation = _protocol()["isolation_policy"]
    source = isolation["source_hash"]
    target = isolation["target_instance"]
    seed = isolation["sampling_seed"]
    assert source["new_source_hash_must_not_match_training_or_calibration"] is True
    assert source["same_source_hash_variants_count_as_one_source"] is True
    assert source["minimum_distinct_source_hashes_for_memory_promotion"] == 3
    assert target["minimum_independent_instances_per_source_hash"] == 3
    assert target["training_or_calibration_instance_overlap_allowed"] is False
    assert target["instance_identifier_reuse_across_split_allowed"] is False
    assert seed["minimum_seeds_per_target_instance"] == 2
    assert seed["renaming_identical_samples_is_not_independent"] is True
    assert isolation["split_before_augmentation"] is True
    assert isolation["duplicate_evidence_policy"].startswith("deduplicate")


def test_pg25c_family_holdout_never_becomes_posthoc_calibration():
    family = _protocol()["family_isolation_policy"]
    assert family["known_family_set_frozen_before_collection"] is True
    assert family["known_family_cross_target_rows_may_not_enter_calibration"] is True
    assert family["unseen_family_rows_may_not_enter_training_or_calibration"] is True
    assert family["unseen_family_expected_final_action"] == "abstain"
    assert family["candidate_route_for_unseen_family_is_diagnostic_only"] is True
    assert family["expected_family_is_evaluator_side_only"] is True


def test_pg25c_fresh_reset_is_per_observation_and_fail_closed():
    reset = _protocol()["fresh_reset_policy"]
    assert reset["fresh_reset_required_before_every_independent_observation"] is True
    assert reset["fresh_target_required"] is True
    assert reset["paired_views_use_separate_reset_receipts"] is True
    assert reset["reset_failure_action"] == "invalidate_sample_and_report_engineering_failure"
    assert reset["reset_failures_count_in_model_denominator"] is False
    assert reset["batch_reuse_as_independent_episode_allowed"] is False
    assert {"source_hash", "target_instance_id", "reset_nonce_hash"} <= set(reset["reset_receipt_required_fields"])


def test_pg25c_acceptance_sample_plan_has_independent_statistical_support():
    plan = _protocol()["minimum_sampling_plan"]["acceptance_run"]
    cells = plan["target_instances"] * plan["seeds_per_instance"]
    positives = cells * plan["known_family_positive_cases_per_cell"]
    negatives = cells * plan["known_family_negative_cases_per_cell"]
    unseen = cells * plan["unseen_family_ood_cases_per_cell"]
    assert cells == 6
    assert positives == plan["independent_positive_cases_per_known_family"] == 24
    assert negatives == plan["independent_negative_cases_per_known_family"] == 60
    assert positives + negatives == plan["independent_cases_per_known_family"] == 84
    assert unseen == plan["independent_cases_per_unseen_family"] == 60
    controls = _protocol()["minimum_sampling_plan"]["negative_control_allocation_per_known_family_cell"]
    assert sum(value for key, value in controls.items() if key != "total_independent_negatives") == 10
    assert controls["total_independent_negatives"] == plan["known_family_negative_cases_per_cell"]


def test_pg25c_abstain_and_statistical_gates_are_executable():
    gates = _protocol()["abstention_and_acceptance_gates"]
    known = gates["known_family_cross_target"]
    unseen = gates["unseen_family_ood"]
    risk = gates["statistical_risk"]
    assert known["unsafe_accept_count_max"] == 0
    assert known["negative_false_accept_count_max"] == 0
    assert known["aggregate_positive_recall_min"] == 0.60
    assert known["per_target_positive_recall_min"] == 0.50
    assert known["seed_recall_range_max"] == 0.15
    assert unseen["final_abstain_rate_min"] == 0.95
    assert unseen["rule_ir_emission_count_max"] == 0
    assert risk["minimum_independent_negative_count"] == 60
    assert risk["required_observed_false_accept_count"] == 0
    assert risk["false_accept_upper_bound_max"] == 0.05
    computed_upper = 1.0 - math.pow(1.0 - risk["one_sided_confidence_level"], 1.0 / risk["minimum_independent_negative_count"])
    assert abs(risk["zero_event_upper_bound_at_n_60"] - computed_upper) < 1e-6
    assert computed_upper < risk["false_accept_upper_bound_max"]


def test_pg25c_reports_breakdowns_and_does_not_auto_promote():
    protocol = _protocol()
    reporting = protocol["statistical_reporting"]
    integration = protocol["pg23_pg24_integration"]
    assert {"source_hash", "target_instance_id", "sampling_seed", "negative_control_kind"} <= set(reporting["required_breakdowns"])
    assert {"deduplicated", "reset_failures", "false_accepts", "family_errors"} <= set(reporting["required_counts"])
    assert reporting["all_failures_reported_without_exclusion"] is True
    registry = integration["pg24_registry_transition"]
    assert registry["automatic_training_eligibility_on_pass"] is False
    assert registry["automatic_memory_promotion_on_pass"] is False
    assert registry["acceptance_cohort_may_later_enter_training"] is False
    assert registry["fresh_source_and_target_cohort_required_for_training"] is True
    memory = integration["memory_promotion_contract"]
    assert memory["minimum_distinct_datasets"] == 3
    assert memory["minimum_distinct_source_hashes"] == 3
    assert memory["pg25c_acceptance_alone_is_sufficient"] is False
