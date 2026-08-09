import json
from pathlib import Path

from app.research_ops import _pg324_contract_projection, _pg325_contract_projection, _pg326_contract_projection, _pg327_training_projection, _pg327b_replay_projection, _pg331_readonly_source_projection, _pg331_source_collection_projection, _pg331_train_holdout_diagnostic_v2_projection, _pg331_typed_capacity_projection, _pg331_typed_source_rows_projection, _pg332_extended_diagnostic_projection, _pg333_cross_impl_projection, _pg333_webgoat_projection, _pg334_process_token_projection, _pg335_real_process_projection, _pg336_real_failure_process_projection, _pg337_cross_impl_process_projection, _pg338_information_preserving_projection, _pg339_multi_shape_projection, _pg340_balanced_axis_projection, _pg341_target_conditioned_projection, _pg342_full_axis_failure_repair_projection, build_payload_review, build_research_ops_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_payload_review_is_bounded_and_human_readable():
    result = build_payload_review()

    assert result["review_only"] is True
    assert result["persisted"] is False
    assert result["training_eligible"] is False
    assert result["target_scope"]["kind"] == "loopback_only"
    assert result["target_scope"]["arbitrary_target_input"] is False
    assert len(result["entries"]) >= 15
    assert any(str(entry.get("source", "")).startswith("PG-179") for entry in result["entries"])

    routes = {entry["route"]: entry for entry in result["entries"]}
    blind = routes["/vul/sqli/sqli_blind_b.php"]
    assert "kobe%27%20AND%20%271%27%3D%271" in blind["ai"]["true"]["wire"]
    assert "kobe%27%20AND%20%271%27%3D%272" in blind["ai"]["false"]["wire"]
    assert blind["oracle_evidence"]["matched"] is True
    assert blind["oracle_evidence"]["pattern_id"]
    assert blind["oracle_evidence"]["evidence_sha256"]
    assert blind["oracle_evidence"]["oracle_facts"]["candidate_boolean_differential"] is True
    assert blind["oracle_evidence"]["candidate_true_projection"]["result_shape"] == "record_present"
    assert blind["oracle_evidence"]["candidate_false_projection"]["result_shape"] == "record_absent"

    wide = routes["/vul/sqli/sqli_widebyte.php"]
    assert wide["validation_status"] == "validated_local_effect"
    assert "kobe%df%27%20OR%201%3D1%23" in wide["reference"]["request"]["wire"]
    assert wide["ai"]["status"] == "model_replayed_on_3_of_4_fresh_seeds"
    assert "kobe%df%27%20OR%201%3D1%23" in wide["ai"]["request"]["wire"]
    assert wide["oracle_evidence"]["evidence_sha256"]
    assert wide["oracle_evidence"]["raw_response_body_stored"] is False

    xss = routes["/vul/xss/xss_01.php"]
    assert "%3Cspan%20data-sift-marker%3D%22sift-review-canary%22%3E" in xss["ai"]["request"]["wire"]
    assert "<script" not in xss["ai"]["request"]["wire"].lower()

    for entry in result["entries"]:
        assert "oracle_evidence" in entry
        assert entry["oracle_evidence"]["raw_payload_stored"] is False
        assert entry["oracle_evidence"]["raw_response_body_stored"] is False
        for channel in (entry["ai"], entry["reference"], entry["negative"]):
            if channel.get("request"):
                assert "<LOOPBACK_ORIGIN>" in channel["request"]["wire"]


def test_research_snapshot_does_not_expose_raw_payloads():
    result = build_research_ops_snapshot()
    encoded = str(result)
    assert result["judge"]["vulnerability_claim_allowed"] is False
    assert "kobe%df" not in encoded
    assert "<script" not in encoded.lower()
    assert all(task["raw_material_available"] is False for task in result["tasks"]["all"])
    assert any(task["id"].startswith("pg256-") for task in result["tasks"]["all"])
    assert next(metric for metric in result["capability"]["metrics"] if metric["id"] == "widebyte")["value"] == "3/4"
    assert next(metric for metric in result["capability"]["metrics"] if metric["id"] == "boolean")["value"] == "2/2"
    assert next(metric for metric in result["capability"]["metrics"] if metric["id"] == "xss")["value"] == "12/16"
    assert next(metric for metric in result["capability"]["metrics"] if metric["id"] == "pg257")["value"] == "100%"
    pg258 = next(metric for metric in result["capability"]["metrics"] if metric["id"] == "pg258")
    assert pg258["value"] == "61%"
    assert pg258["status"] == "blocked"
    assert result["capability"]["model"]["pg258"]["promotion_blocked"] is True
    pg259 = next(metric for metric in result["capability"]["metrics"] if metric["id"] == "pg259")
    assert pg259["value"] == "50%"
    assert pg259["status"] == "blocked"
    assert result["capability"]["model"]["pg259"]["promotion_blocked"] is True
    assert result["capability"]["model"]["pg262"]["record_count"] == 20
    assert result["capability"]["model"]["pg262"]["audit_complete"] is True
    assert result["capability"]["model"]["pg262"]["training_eligible"] is False
    pg263_model = result["capability"]["model"]["pg263"]
    pg263_metric = next(metric for metric in result["capability"]["metrics"] if metric["id"] == "pg263")
    assert pg263_model["status"] == "completed_pg263_pg262_augmented_masked_capacity_training"
    assert pg263_model["judge_pass"] is True
    assert pg263_model["promotion_blocked"] is True
    assert pg263_metric["value"] == "97%"
    assert result["capability"]["model"]["pg257"]["promotion_blocked"] is True
    assert any(task["id"] == "pg257-rule-ir-capacity" for task in result["tasks"]["trainer"])
    assert any(task["id"] == "pg258-unified-rule-ir" for task in result["tasks"]["trainer"])
    assert any(task["id"] == "pg259-active-belief" for task in result["tasks"]["trainer"])
    assert any(task["id"].startswith("pg242-") for task in result["tasks"]["all"])
    assert any(layer["id"] == "fresh-augmentation" for layer in result["architecture"])
    assert len(result["process_traces"]) == 59
    trace = result["process_traces"][0]
    assert trace["route"].startswith("GET /vul/sqli/")
    assert [stage["id"] for stage in trace["stages"]] == ["observe", "decide", "candidate", "reference", "negative", "oracle", "next"]
    assert trace["target_hash"]
    assert "kobe%27" not in str(result["process_traces"])
    assert any(item["id"].startswith("pg259-") for item in result["process_traces"])

    pg277 = result["capability"]["model"]["pg277_question_composition"]
    assert pg277["status"] == "completed_question_composition_ablation"
    assert pg277["audit_pass"] is True
    assert pg277["coarse_conflicting_record_count"] == 36
    assert pg277["coarse_positive_recall"] == 0.0
    assert pg277["final_only_pre_question_accuracy"] == 0.0
    assert pg277["process_positive_recall"] == 1.0
    assert pg277["process_ask_recovery"] == 1.0
    assert pg277["process_question_recovery_min"] == 0.0
    assert pg277["conservative_question_recovery_min"] == 1.0
    assert pg277["dpo_question_recovery_min"] == 0.0
    assert pg277["promotion_blocked"] is True
    pg277_metric = next(metric for metric in result["capability"]["metrics"] if metric["id"] == "pg277")
    assert pg277_metric["value"] == "0%"
    assert pg277_metric["status"] == "blocked"

    brief = result["learning_requirements"]
    assert brief["evidence"]["audit_pass"] is True
    assert brief["evidence"]["dataset_audit_pass"] is True
    assert brief["evidence"]["model_audit_pass"] is True
    assert brief["evidence"]["source_audit"] == "pg287_identifiability_dataset_audit_v1.json"
    assert brief["evidence"]["controlled_rows"] == 52
    assert brief["evidence"]["real_multifamily_gold_rows"] == 0
    assert brief["evidence"]["coarse_conflicting_rows"] == 288
    assert brief["evidence"]["pg278_post_conflict_groups"] == 0
    assert brief["evidence"]["pg279_post_conflict_groups"] == 0
    assert brief["evidence"]["pg279_get_rows"] == 216
    assert brief["evidence"]["pg279_post_rows"] == 72
    assert brief["evidence"]["pg280_conditional_entropy_bits"] == 1.0
    assert brief["evidence"]["pg280_bayes_error_lower_bound"] == 0.5
    assert brief["evidence"]["pg280_final_only_pre_supervision_rows"] == 0
    assert brief["evidence"]["pg280_process_pre_supervision_rows"] == 288
    assert brief["evidence"]["pg280_hard_negative_rows"] == 48
    assert brief["evidence"]["pg280_docker_status"] == "unavailable"
    assert brief["evidence"]["pg280_remote_adapter_status"] == "unavailable"
    assert brief["evidence"]["pg280_remote_adapter_audit_pass"] is True
    assert brief["evidence"]["pg281_route_positive_recall_min"] == 1.0
    assert brief["evidence"]["pg281_family_positive_recall_min"] == 1.0
    assert brief["evidence"]["pg281_hard_negative_reject_min"] == 1.0
    assert brief["evidence"]["pg281_hard_negative_false_allow_max"] == 0
    assert brief["evidence"]["pg281_selected_variant"] == "plain_sft"
    assert brief["evidence"]["pg281_risk_weight_variant_count"] == 5
    assert brief["latest_experiment"]["controlled_rows"] == 9576
    assert brief["latest_experiment"]["id"] == "PG-287"
    assert brief["promotion_gate"]["current_status"].startswith("PG-287 safety metrics recorded")
    assert {queue["id"] for queue in brief["queues"]} == {
        "Q1-observation-counterfactuals",
        "Q2-failure-repair-trajectories",
        "Q3-missing-question-recovery",
        "Q4-ood-and-forgetting",
    }
    assert all(queue["minimum_quota"] for queue in brief["queues"])
    assert all(queue["collect"] for queue in brief["queues"])
    assert all(queue["acceptance"] for queue in brief["queues"])
    assert any(item["id"] == "final-label-only" for item in brief["forbidden"])
    assert any(item["id"] == "rl-on-incomplete" for item in brief["forbidden"])
    assert any(task["id"] == "pg277-question-composition-audit" for task in result["tasks"]["all"])
    assert any(task["id"] == "pg278-multifamily-question-policy" for task in result["tasks"]["all"])
    assert any(task["id"] == "pg279-remote-replay-policy" for task in result["tasks"]["all"])
    assert any(task["id"] == "pg280-ontology-policy" for task in result["tasks"]["all"])
    assert result["capability"]["model"]["pg278"]["gate_pass"] is True
    assert result["capability"]["model"]["pg278"]["promotion_blocked"] is True
    assert result["capability"]["model"]["pg279"]["operational_audit_pass"] is True
    assert result["capability"]["model"]["pg279"]["scientific_gate_status"] == "blocked"
    assert result["capability"]["model"]["pg279"]["retention_status"] == "passed"
    assert result["capability"]["model"]["pg280"]["operational_audit_pass"] is True
    assert result["capability"]["model"]["pg280"]["remote_adapter_status"] == "unavailable"
    assert result["capability"]["model"]["pg280"]["remote_adapter_audit_pass"] is True
    assert result["capability"]["model"]["pg280"]["scientific_gate_status"] == "blocked"
    assert result["capability"]["model"]["pg280"]["process_ask_rate"] == 1.0
    assert result["capability"]["model"]["pg280"]["final_only_ask_rate"] == 0.0
    assert result["capability"]["model"]["pg280"]["promotion_blocked"] is True
    assert result["capability"]["model"]["pg281"]["operational_audit_pass"] is True
    assert result["capability"]["model"]["pg281"]["route_positive_recall"] == 1.0
    assert result["capability"]["model"]["pg281"]["hard_negative_false_allow"] == 0
    assert result["capability"]["model"]["pg281"]["selected_variant"] == "plain_sft"
    assert result["capability"]["model"]["pg281"]["risk_weight_variant_count"] == 5
    assert result["capability"]["model"]["pg281"]["literal_payload_generation"] is False
    assert any(task["id"] == "pg281-abstract-payload-policy" for task in result["tasks"]["all"])


def test_pg286_observation_gate_is_visible_and_quarantines_incomplete_evidence():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg286"]
    assert model["status"] == "catalog_audited_collection_only"
    assert model["total_rows"] == 28
    assert model["complete_rows"] == 12
    assert model["incomplete_rows"] == 16
    assert model["sql_ast_available_rows"] == 0
    assert model["training_eligible_rows"] == 0
    assert model["hard_negative_rows"] == 28
    assert model["family_hidden_in_context"] is True
    assert model["oracle_label_in_context"] is False
    assert model["remote_docker_status"] == "unavailable"
    assert model["operational_audit_pass"] is True
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["collector"] if item["id"] == "pg286-observation-token-gate")
    assert task["status"] == "needs_authorized_remote_evaluator"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg286" and metric["value"] == "12/28" for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")
    assert result["learning_requirements"]["promotion_gate"]["next_experiment"].startswith("PG-331")


def test_pg287_identifiability_gate_surfaces_family_resolved_failure():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg287"]
    assert model["status"] == "completed_remote_pg287_identifiability_training"
    assert model["train_count"] == 6678
    assert model["route_dev_count"] == 756
    assert model["family_holdout_count"] == 630
    assert model["hard_negative_count"] == 1512
    assert model["family_ambiguous_ask_recall"] == 1.0
    assert model["family_resolved_encoding_accuracy"] is None
    assert model["family_resolved_encoding_count"] == 0
    assert model["hard_negative_ask_recall"] == 1.0
    assert model["hard_negative_false_allow"] == 0
    assert model["remote_docker_status"] == "unavailable"
    assert model["operational_audit_pass"] is False
    assert model["promotion_blocked"] is True
    assert model["live_batch_status"] == "blocked"
    assert model["live_batch_record_count"] == 0
    assert model["live_batch_family_resolved_count"] == 0
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg287-identifiability")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg287" and metric["value"] == "N/A resolved" for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")
    assert result["learning_requirements"]["promotion_gate"]["next_experiment"].startswith("PG-331")


def test_pg292_feature_gate_is_reported_but_not_promoted():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg292"]
    assert model["status"] == "completed_remote_pg292_feature_gate"
    assert model["selected_variant"] == "guarded"
    assert model["selected_threshold"] == 0.8
    assert model["route_positive_recall"] == 1.0
    assert model["family_positive_recall"] == 1.0
    assert model["hard_negative_false_allow"] == 0
    assert model["hard_negative_safe_reject"] == 1.0
    assert model["engineering_gate_status"] == "passed"
    assert model["scientific_gate_status"] == "blocked"
    assert model["remote_docker_status"] == "unavailable"
    assert model["real_application_gold_rows"] == 0
    assert model["live_adapter"] == "app/pg292_live.py"
    assert model["live_endpoint"] == "/api/maze/remote-docker/pg292-live"
    assert model["wire_emission_allowed"] is False
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg292-feature-gate")
    assert task["status"] == "promotion_blocked"
    assert task["confirmed_positive"] is False
    assert any(metric["id"] == "pg292" and metric["status"] == "blocked" for metric in result["capability"]["metrics"])


def test_pg293_greedy_failure_decoder_exposes_same_context_hard_negative_failure():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg293"]
    assert model["status"] == "completed_remote_pg293_failure_next_action"
    assert model["autoregressive_eval"] is True
    assert model["source_holdout_count"] == 40
    assert model["seed_holdout_count"] == 12
    assert model["hard_negative_eval_count"] == 8
    assert model["holdout_positive_recall"] == 1.0
    assert model["hard_negative_false_allow"] == 8
    assert model["engineering_gate_status"] == "blocked"
    assert model["scientific_gate_status"] == "blocked"
    assert model["local_morning_status"] == "completed_local_morning_pg293_failure_next_action"
    assert model["local_morning_device"] == "NVIDIA GeForce RTX 3060"
    assert model["local_morning_holdout_positive_recall"] == 1.0
    assert model["local_morning_hard_negative_false_allow"] == 8
    assert model["local_morning_engineering_gate_status"] == "blocked"
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg293-failure-next-action")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg293" and metric["value"] == "100% / 8" for metric in result["capability"]["metrics"])


def test_pg295_causal_moe_separates_missing_question_from_answer_only_control():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg295"]
    assert model["status"] == "completed_local_morning_pg295_causal_moe"
    assert model["architecture"] == "causal_transformer_moe"
    assert model["causal_next_token_only"] is True
    assert model["experts"] == 4
    assert model["seed_missing_question_recall"] == 1.0
    assert model["answer_only_missing_question_recall"] == 0.0
    assert model["hard_negative_false_allow"] == 8
    assert model["engineering_gate_status"] == "blocked"
    assert model["scientific_gate_status"] == "blocked"
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg295-causal-moe-question-composition")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg295" and metric["value"] == "100% / 8" for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg300_question_policy_requires_recall_and_no_unnecessary_question():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg300"]
    assert model["status"] == "completed_local_morning_pg300_question_policy"
    assert model["architecture"] == "causal_transformer_moe_question_only"
    assert model["question_recall_min"] == 1.0
    assert model["hard_negative_false_allow_max"] == 0
    assert model["hard_negative_unnecessary_question_max"] == 1.0
    assert model["engineering_gate_status"] == "blocked"
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg300-question-policy")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg300" and metric["status"] == "blocked" for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg301_pg302_and_pg303_assembly_progress_remains_blocked_and_visible():
    result = build_research_ops_snapshot()
    pg301 = result["capability"]["model"]["pg301"]
    assert pg301["status"] == "completed_local_morning_pg301_payload_assembly"
    assert pg301["holdout_question_recall_min"] == 0.75
    assert pg301["holdout_assembly_slot_exact_min"] == 0.583333
    assert pg301["promotion_blocked"] is True
    pg302 = result["capability"]["model"]["pg302"]
    assert pg302["status"] == "completed_local_morning_pg302_symbolic_assembly"
    assert pg302["holdout_question_recall_min"] == 0.25
    assert pg302["holdout_unnecessary_question_max"] == 1.0
    pg302b = result["capability"]["model"]["pg302b"]
    assert pg302b["status"] == "completed_local_morning_pg302b_symbolic_curriculum"
    assert pg302b["holdout_question_recall_min"] == 0.25
    pg303 = result["capability"]["model"]["pg303"]
    assert pg303["status"] == "completed_local_morning_pg303_guarded_eval"
    assert pg303["raw_holdout_missing_question_recall"] == 0.75
    assert pg303["guarded_holdout_missing_question_recall"] == 1.0
    assert pg303["guarded_hard_negative_false_allow"] == 0
    assert pg303["neural_claim_allowed"] is False
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg301-302-assembly-composition")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg304_fixture_replay_contract_is_visible_but_not_promoted():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg304"]
    assert model["status"] == "completed_loopback_evaluator_only"
    assert model["fixture_episode_count"] == 3
    assert model["fixture_typed_positive_count"] == 2
    assert model["fixture_blocked_count"] == 1
    assert model["get_post_pair"] is True
    assert model["loopback_only"] is True
    assert model["external_network_disabled"] is True
    assert model["training_eligible_count"] == 0
    assert model["memory_promotion_allowed_count"] == 0
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg304-loopback-replay-contract")
    assert task["status"] == "ready_for_authorized_adapter"
    assert task["confirmed_positive"] is False
    assert task["raw_material_available"] is False
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg314_independent_variant_replay_is_visible_but_scientific_promotion_stays_closed():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg314"]
    assert model["status"] == "completed_real_local_docker_independent_variant_replay"
    assert model["route_count"] == 2
    assert model["get_count"] == 1
    assert model["post_count"] == 1
    assert model["variant_role_count"] == 6
    assert model["variant_exact_count"] == 6
    assert model["model_variant_send_count"] == 6
    assert model["model_typed_effect_count"] == 2
    assert model["negative_lane_violation_count"] == 0
    assert model["preflight_question_recall"] == 1.0
    assert model["preflight_unsafe_allow"] == 0
    assert model["docker_network_none"] is True
    assert model["fresh_reset_per_route"] is True
    assert model["typed_evidence_hash_per_route"] is True
    assert model["source_grounded_wire"] is True
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg314-independent-variant-replay")
    assert task["status"] == "promotion_blocked"
    assert task["confirmed_positive"] is False
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg314" and "variant 6/6" in metric["value"] for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg315_worst_seed_failure_repair_regression_is_visible():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg315"]
    assert model["status"] == "completed_real_local_docker_all_seed_replay"
    assert model["seed_count"] == 3
    assert model["route_count"] == 6
    assert model["variant_role_count"] == 18
    assert model["variant_exact_count"] == 14
    assert model["model_typed_effect_count"] == 4
    assert model["negative_lane_violation_count"] == 2
    assert model["repair_row_count"] == 6
    assert model["repair_abstain_correct_count"] == 0
    assert model["worst_question_recall"] == 1.0
    assert model["worst_variant_exact"] < 0.9
    assert model["worst_repair_abstain"] == 0.0
    assert model["worst_negative_violation"] == 2
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg315-worst-seed-replay")
    assert task["status"] == "promotion_blocked"
    assert task["confirmed_positive"] is False
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg315" and "repair 0/6" in metric["value"] for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg316_repair_anchor_and_live_replay_are_visible_but_question_gate_remains_blocked():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg316"]
    live = result["capability"]["model"]["pg316_live"]
    assert model["status"] == "completed_local_morning_pg316_failure_repair"
    assert model["question_recall_min"] == 0.88
    assert model["variant_exact_min"] == 1.0
    assert model["repair_exact_min"] == 1.0
    assert model["repair_safe_allow_max"] == 0.0
    assert model["hard_false_allow_max"] == 0.0
    assert model["repair_train_rows"] == 128
    assert model["promotion_blocked"] is True
    assert live["status"] == "completed_real_local_docker_pg316_live_replay"
    assert live["variant_exact_count"] == 6
    assert live["model_typed_effect_count"] == 2
    assert live["failure_repair_correct"] == 2
    assert live["failure_repair_count"] == 2
    assert live["negative_lane_violation_count"] == 0
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg316-failure-repair-anchor")
    assert task["status"] == "promotion_blocked"
    live_task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg316-live-independent-variant")
    assert live_task["status"] == "promotion_blocked"
    assert any(metric["id"] == "pg316" and "repair 100%" in metric["value"] for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg317_question_anchor_and_live_replay_are_visible_but_promotion_stays_closed():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg317"]
    live = result["capability"]["model"]["pg317_live"]
    assert model["status"] == "completed_local_morning_pg317_question_anchor"
    assert model["anchor_question_min"] >= 0.95
    assert model["anchor_safe_allow_max"] == 0.0
    assert model["anchor_unnecessary_question_max"] == 0.0
    assert model["question_recall_min"] >= 0.9
    assert model["variant_exact_min"] >= 0.9
    assert model["repair_exact_min"] >= 0.9
    assert model["hard_false_allow_max"] == 0.0
    assert model["dataset_audit_status"] == "passed"
    assert model["promotion_blocked"] is True
    assert live["status"] == "completed_real_local_docker_pg317_live_replay"
    assert live["variant_exact_count"] == 6
    assert live["model_typed_effect_count"] == 2
    assert live["preflight_question_recall"] == 1.0
    assert live["failure_repair_correct"] == 2
    assert live["failure_repair_count"] == 2
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg317-question-anchor")
    assert task["status"] == "promotion_blocked"
    live_task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg317-live-independent-variant")
    assert live_task["status"] == "promotion_blocked"
    assert any(metric["id"] == "pg317" and "ASK 98.9%" in metric["value"] for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg318_family_holdout_is_visible_and_next_step_is_pg322():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg318_family_holdout"]
    assert model["status"] == "completed_real_local_docker_pg318_family_holdout"
    assert model["seed_count"] == 3
    assert model["route_count"] == 18
    assert model["variant_exact_min"] == 1.0
    assert model["typed_effect_route_rate_min"] == 1.0
    assert model["multi_missing_question_recall_min"] == 1.0
    assert model["multi_missing_unsafe_allow"] == 0
    assert model["failure_repair_rate_min"] == 1.0
    assert model["docker_network_none"] is True
    assert model["promotion_blocked"] is True
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg318-family-heldout-replay")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg318" and "typed 18/18" in metric["value"] for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg319_pg320_pg321_progress_keeps_scientific_gate_closed():
    result = build_research_ops_snapshot()
    pg319 = result["capability"]["model"]["pg319_cross_impl"]
    assert pg319["status"] == "completed_local_morning_pg319_cross_impl_moe"
    assert pg319["implementation_route_question_min"] == 1.0
    assert pg319["family_question_min"] == 0.0
    assert pg319["promotion_blocked"] is True
    pg320 = result["capability"]["model"]["pg320_observation_lattice"]
    assert pg320["live_variant_exact_min"] == 0.3333333333333333
    assert pg320["live_negative_lane_violation_max"] == 6
    pg321 = result["capability"]["model"]["pg321_variant_role"]
    assert pg321["live_variant_exact_min"] == 1.0
    assert pg321["live_typed_effect_route_rate_min"] == 1.0
    assert pg321["live_negative_lane_violation_max"] == 0
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg321-family-heldout-replay")
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert any(metric["id"] == "pg321" and "typed 18/18" in metric["value"] for metric in result["capability"]["metrics"])
    pg322 = result["capability"]["model"]["pg322_cross_impl_decoy"]
    assert pg322["ask_question_min"] == 0.933333
    assert pg322["hard_false_allow_max"] == 2.0
    pg323 = result["capability"]["model"]["pg323_decoy_ask_anchor"]
    assert pg323["ask_question_min"] == 0.983333
    assert pg323["hard_false_allow_max"] == 0.0
    assert pg323["live_variant_exact_min"] == 1.0
    assert pg323["live_typed_effect_route_rate_min"] == 1.0
    assert pg323["live_negative_lane_violation_max"] == 0
    task323 = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg323-vulnerableapp-fresh-replay")
    assert task323["status"] == "promotion_blocked"
    assert task323["raw_material_available"] is False
    assert any(metric["id"] == "pg323" and "typed 6/6" in metric["value"] for metric in result["capability"]["metrics"])
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg324_projection_rejects_pre_v2_artifacts_without_claiming_capability():
    projection = _pg324_contract_projection(
        {"schema_version": "pg324-juice-shop-source-heldout-report-v1", "status": "completed_real_local_docker_pg324_juice_shop_source_heldout"},
        {"schema_version": "pg324-juice-shop-source-heldout-catalog-v1"},
        {"schema_version": "pg324-juice-shop-source-heldout-trace-v1"},
        {"schema_version": "pg324-juice-shop-source-heldout-protocol-v1"},
    )
    assert projection["artifact_status"] == "stale_contract"
    assert projection["counts"] == {}
    assert projection["claim_allowed"] is False
    assert projection["model_capability_claim_allowed"] is False
    assert projection["promotion_blocked"] is True
    assert projection["raw_material_available"] is False


def test_pg324_snapshot_keeps_current_artifact_evaluation_only():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg324_juice_shop_source_heldout"]
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg324-juice-shop-source-heldout-fresh-replay")
    metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg324")
    assert model["promotion_blocked"] is True
    assert model["model_capability_claim_allowed"] is False
    assert model["training_eligible"] is False
    assert model["memory_promotion_allowed"] is False
    assert model["vulnerability_claim_allowed"] is False
    assert task["typed_effect"] is False
    assert task["confirmed_positive"] is False
    assert task["raw_material_available"] is False
    assert metric["status"] == "blocked"


def test_pg325_snapshot_exposes_sql_family_evidence_without_promotion():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg325_sql_family_holdout"]
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg325-sql-family-holdout-fresh-replay")
    metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg325")
    assert model["artifact_status"] == "completed_evaluation_only"
    assert model["audit_status"] == "passed"
    assert model["counts"]["positive_typed_effect_count"] == 9
    assert model["counts"]["belief_duplicate_evidence_count"] == 0
    assert model["family_ood"] is True
    assert model["role_bound_belief_evidence"] is True
    assert model["model_capability_claim_allowed"] is False
    assert model["training_eligible"] is False
    assert task["status"] == "promotion_blocked"
    assert task["confirmed_positive"] is False
    assert metric["status"] == "blocked"


def test_pg325_projection_marks_missing_documents_incomplete():
    projection = _pg325_contract_projection({}, {}, {}, {}, {})
    assert projection["artifact_status"] == "awaiting_fresh_replay"
    assert projection["claim_allowed"] is False
    assert projection["promotion_blocked"] is True


def test_pg326_snapshot_separates_observed_matrix_from_forgetting_gate():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg326_cross_impl_forgetting_matrix"]
    task = next(item for item in result["tasks"]["reviewer"] if item["id"] == "pg326-cross-implementation-forgetting-matrix")
    metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg326")
    assert model["artifact_status"] == "completed_evaluation_matrix_blocked"
    assert model["implementation_count"] == 3
    assert model["counts"]["positive_typed_effect_count"] == 18
    assert model["forgetting"]["paired_replay_present"] is True
    assert model["matrix_gate_checks"]["uniform_observation_contract"] is False
    assert model["audit_status"] == "passed"
    assert model["training_eligible"] is False
    assert task["status"] == "promotion_blocked"
    assert metric["status"] == "blocked"
    assert result["research_goal"]["next_experiment"].startswith("PG-331")


def test_pg326_projection_marks_missing_matrix_incomplete():
    projection = _pg326_contract_projection({}, {}, {})
    assert projection["artifact_status"] == "awaiting_matrix"
    assert projection["claim_allowed"] is False


def test_pg327_remote_candidate_is_visible_but_not_promoted():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg327_a800_replay_training"]
    metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg327")
    task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg327-a800-replay")
    assert model["artifact_status"] == "completed_remote_a800_candidate"
    assert model["gpu_name"] == "NVIDIA A800-SXM4-80GB"
    assert model["visible_cuda_devices"] == "0"
    assert model["memory_promotion_allowed"] is False
    assert metric["status"] == "blocked"
    assert "GPU0" in metric["value"]
    assert task["status"] == "promotion_blocked"
    assert task["raw_material_available"] is False
    assert result["capability"]["next"].startswith("PG-331A")


def test_pg327_projection_rejects_incomplete_report():
    projection = _pg327_training_projection({})
    assert projection["artifact_status"] == "awaiting_training"
    assert projection["training_allowed"] is False
    assert projection["promotion_blocked"] is True


def test_pg327b_paired_replay_is_visible_but_remains_evaluator_only():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg327b_paired_fresh_replay"]
    metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg327b")
    assert model["artifact_status"] == "completed_paired_fresh_replay"
    assert model["paired_replay_present"] is True
    assert model["same_canary_route_set"] is True
    assert model["audit_status"] == "passed"
    assert model["training_eligible"] is False
    assert metric["status"] == "blocked"
    assert "before/after typed 9/9" in metric["value"]
    assert result["capability"]["next"].startswith("PG-331A")


def test_pg327b_projection_requires_passed_audit():
    report = json.loads((ROOT / "research" / "pg327b_paired_fresh_replay_report_v1.json").read_text(encoding="utf-8-sig"))
    projection = _pg327b_replay_projection(report, {"status": "blocked", "failures": ["test"]})
    assert projection["artifact_status"] == "audit_blocked"
    assert projection["paired_replay_present"] is False
    assert projection["promotion_blocked"] is True


def test_pg331_whole_web_token_contract_is_visible_and_blocks_incomplete_data():
    result = build_research_ops_snapshot()
    model = result["capability"]["model"]["pg331_information_preservation"]
    metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg331")
    capacity_metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg331-capacity")
    source_row_metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg331a-source-row")
    loopback_metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg331a-loopback-smoke")
    legacy_metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg331-legacy-manifest")
    a800_metric = next(item for item in result["capability"]["metrics"] if item["id"] == "pg331-a800-resource")
    task = next(item for item in result["tasks"]["collector"] if item["id"] == "pg331a-whole-web-token-collection")
    capacity_task = next(item for item in result["tasks"]["trainer"] if item["id"] == "pg331b-model-capacity-audit")
    assert model["status"] == "blocked_missing_information"
    assert model["record_count"] == 195
    assert model["axis_count"] == 7
    assert len(model["missing_axes"]) == 7
    assert model["context_target_alignment"] == 0.082051
    assert model["vocabulary_training_allowed"] is False
    assert model["promotion_blocked"] is True
    assert model["source_row_audit_status"] == "blocked"
    assert model["source_row_record_count"] == 0
    assert model["loopback_smoke_status"] == "target_unavailable"
    assert model["loopback_smoke_target_count"] == 2
    assert model["loopback_smoke_target_contacted_count"] == 0
    assert model["legacy_web_manifest"]["page_count"] == 63
    assert model["legacy_web_manifest"]["route_count"] == 73
    assert model["legacy_web_manifest"]["training_allowed"] is False
    assert model["remote_a800_readonly_preflight"]["gpu0_resource_status"] == "idle"
    assert model["remote_a800_readonly_preflight"]["training_allowed_now"] is False
    assert model["readonly_evidence_training_allowed"] is False
    assert metric["status"] == "blocked"
    assert "7/7 axes missing" in metric["value"]
    assert model["legacy_max_length"] == 72
    assert model["required_context_window"] > 72
    assert model["legacy_capacity_pass"] is False
    assert capacity_metric["status"] == "blocked"
    assert "legacy max72 FAIL" in capacity_metric["value"]
    assert source_row_metric["status"] == "blocked"
    assert "ASK only" in source_row_metric["value"]
    assert loopback_metric["status"] == "blocked"
    assert legacy_metric["status"] == "blocked"
    assert "63 pages" in legacy_metric["value"]
    assert a800_metric["status"] == "blocked"
    assert "GPU0 idle" in a800_metric["value"]
    assert task["status"] == "blocked_on_information_audit"
    assert task["training_eligible"] is False
    assert capacity_task["status"] == "blocked_on_capacity_contract"
    assert capacity_task["training_eligible"] is False
    assert result["capability"]["next"].startswith("PG-331A")


def test_pg331_readonly_source_reports_and_live_collection_stay_blocked():
    legacy = json.loads((ROOT / "research" / "pg331_legacy_web_manifest_audit_v1.json").read_text(encoding="utf-8-sig"))
    preflight = json.loads((ROOT / "research" / "pg331_remote_a800_readonly_preflight_v1.json").read_text(encoding="utf-8-sig"))
    readonly = _pg331_readonly_source_projection(legacy, preflight)
    assert readonly["legacy_web_manifest"]["status"] == "diagnostic_only_blocked"
    assert readonly["legacy_web_manifest"]["page_count"] == 63
    assert readonly["legacy_web_manifest"]["route_count"] == 73
    assert "parameterized_get_response" in readonly["legacy_web_manifest"]["missing_observations"]
    assert readonly["remote_a800_readonly_preflight"]["status"] == "gpu_ready_data_gate_blocked"
    assert readonly["remote_a800_readonly_preflight"]["gpu0"]["name"] == "NVIDIA A800-SXM4-80GB"
    assert readonly["remote_a800_readonly_preflight"]["gpu0_resource_status"] == "idle"
    assert readonly["training_allowed"] is False
    snapshot_model = build_research_ops_snapshot()["capability"]["model"]["pg331_information_preservation"]
    assert snapshot_model["legacy_manifest_status"] == "diagnostic_only_blocked"
    assert snapshot_model["legacy_manifest_page_count"] == 63
    assert snapshot_model["legacy_manifest_route_count"] == 73
    assert snapshot_model["remote_a800_preflight_status"] == "gpu_ready_data_gate_blocked"
    assert snapshot_model["remote_a800_gpu0"]["resource_status"] == "idle"
    assert snapshot_model["training_allowed"] is False

    report = {
        "status": "completed_diagnostic_only",
        "runtime": {"route_count": 3},
        "counts": {"route_count": 3, "target_contacted": 0, "ask_rows": 3, "training_eligible": 0},
        "promotion": {"training_allowed": True, "memory_promotion_allowed": True},
    }
    dataset = {"counts": {"input": 3, "training_eligible": 0}, "records": []}
    live = _pg331_source_collection_projection(report, dataset)
    assert live["route_count"] == 3
    assert live["get_count"] == 0
    assert live["post_count"] == 0
    assert live["parameterized_get_count"] == 0
    assert live["target_contacted"] == 0
    assert live["ask_count"] == 3
    assert live["training_eligible"] is False
    assert live["training_eligible_count"] == 0
    assert all(value is False for value in live["promotion"].values())
    assert live["training_allowed"] is False


def test_pg331_live_collection_missing_artifacts_are_pending_and_blocked():
    live = _pg331_source_collection_projection({}, {})
    assert live["artifact_status"] == "pending"
    assert live["report_status"] == "pending"
    assert live["dataset_status"] == "pending"
    assert live["route_count"] == 0
    assert live["training_allowed"] is False
    assert all(value is False for value in live["promotion"].values())


def test_pg331_typed_source_rows_projection_is_evaluator_only():
    report = json.loads((ROOT / "research" / "pg331_pikachu_typed_source_rows_report_v1.json").read_text(encoding="utf-8-sig"))
    audit = json.loads((ROOT / "research" / "pg331_pikachu_typed_source_rows_audit_v1.json").read_text(encoding="utf-8-sig"))
    sidecars = json.loads((ROOT / "research" / "pg331_pikachu_typed_evaluator_sidecars_v1.json").read_text(encoding="utf-8-sig"))
    typed = _pg331_typed_source_rows_projection(report, audit, sidecars)
    assert typed["status"] == "completed_diagnostic_only"
    assert typed["route_count"] == 3
    assert typed["row_count"] == 9
    assert typed["typed_positive"] == 3
    assert typed["training_eligible"] is False
    assert typed["training_eligible_count"] == 0
    assert typed["audit_status"] == "blocked"
    assert typed["operator_reviewed"] is False
    assert typed["get_count"] == 2
    assert typed["post_count"] == 1
    assert typed["training_allowed"] is False
    assert all(value is False for value in typed["promotion"].values())

    model = build_research_ops_snapshot()["capability"]["model"]["pg331_information_preservation"]
    assert model["typed_source_rows"]["audit_status"] == "blocked"
    assert model["typed_source_rows"]["typed_positive"] == 3


def test_pg331_typed_source_rows_missing_artifacts_are_pending():
    typed = _pg331_typed_source_rows_projection({}, {}, {})
    assert typed["status"] == "pending"
    assert typed["audit_status"] == "pending"
    assert typed["sidecar_status"] == "pending"
    assert typed["route_count"] == 0
    assert typed["row_count"] == 0
    assert typed["training_eligible"] is False
    assert all(value is False for value in typed["promotion"].values())


def test_pg332_extended_get_post_and_a800_projection_is_bounded_and_blocked():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg332_extended_diagnostic_projection(
        load("pg332_dvwa_typed_get_report_v1.json"),
        load("pg332_dvwa_typed_get_source_rows_audit_v1.json"),
        load("pg332_dvwa_typed_get_sidecars_v1.json"),
        load("pg332_dvwa_typed_get_source_rows_v1.json"),
        load("pg332_dvwa_typed_stored_post_report_v1.json"),
        load("pg332_dvwa_typed_stored_post_source_audit_v1.json"),
        load("pg332_dvwa_typed_stored_post_sidecars_v1.json"),
        load("pg332_dvwa_typed_stored_post_source_rows_v1.json"),
        load("pg332_dvwa_pikachu_get_post_cross_impl_source_audit_v1.json"),
        load("pg332_dvwa_pikachu_get_post_cross_impl_information_audit_v3.json"),
        load("pg332_dvwa_pikachu_get_post_cross_impl_capacity_v2.json"),
        load("pg332_dvwa_pikachu_get_post_cross_impl_a800_representation_smoke_v1.json"),
    )
    assert projected["status"] == "diagnostic_blocked"
    assert projected["get"]["route_count"] == 1
    assert projected["get"]["get_count"] == 1
    assert projected["get"]["typed_positive"] == 3
    assert projected["post"]["post_count"] == 1
    assert projected["post"]["typed_positive"] == 3
    assert projected["post"]["stateful_disposable"] is True
    assert projected["post"]["database_clean_before"] is True
    assert projected["post"]["state_delta_evaluator_only"] is True
    assert projected["cross_impl"]["record_count"] == 27
    assert projected["cross_impl"]["implementation_count"] == 2
    assert projected["cross_impl"]["typed_complete_count"] == 27
    assert projected["information"]["accepted_training_eligible_count"] == 0
    assert projected["capacity"]["required_context_window"] == 4145
    assert projected["a800_representation"]["context_only"] is True
    assert projected["a800_representation"]["target_tokens_read"] is False
    assert projected["a800_representation"]["information_promotion_gate_passed"] is False
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())
    encoded = json.dumps(projected, ensure_ascii=False)
    for forbidden in ('"context_tokens"', '"target_tokens"', '"response_body"', '"payload"', '"oracle"', '"records"'):
        assert forbidden not in encoded

    snapshot_model = build_research_ops_snapshot()["capability"]["model"]
    assert snapshot_model["pg332_diagnostic"]["cross_impl"]["record_count"] == 27
    assert snapshot_model["pg331_information_preservation"]["pg332_extended"]["post"]["stateful_disposable"] is True
    assert snapshot_model["pg332_diagnostic"]["a800_representation"]["epochs"] == 4
    assert snapshot_model["pg332_diagnostic"]["a800_representation"]["seeds"] == [33121, 33122, 33123]
    assert next(item for item in build_research_ops_snapshot()["capability"]["metrics"] if item["id"] == "pg332-cross-impl")["status"] == "blocked"


def test_pg332_extended_missing_artifacts_is_pending_and_fail_closed():
    projected = _pg332_extended_diagnostic_projection({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    assert projected["status"] == "pending"
    assert projected["cross_impl"]["record_count"] == 0
    assert projected["a800_representation"]["status"] == "pending"
    assert projected["training_eligible"] is False
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg333_webgoat_projection_stays_diagnostic_only():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg333_webgoat_projection(
        load("pg333_webgoat_typed_method_shape_report_v1.json"),
        load("pg333_webgoat_typed_method_shape_source_audit_v1.json") if (ROOT / "research" / "pg333_webgoat_typed_method_shape_source_audit_v1.json").exists() else {},
        load("pg333_webgoat_typed_method_shape_sidecars_v1.json"),
        load("pg333_webgoat_typed_method_shape_source_rows_v1.json"),
    )
    assert projected["source_row_count"] == 18
    assert projected["methods"]["GET"] == 1
    assert projected["methods"]["POST"] == 1
    assert projected["typed_positive_route_seed_count"] == 6
    assert projected["negative_violation_count"] == 0
    assert projected["network_none"] is True
    assert projected["no_bind_or_volume"] is True
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())
    snapshot_model = build_research_ops_snapshot()["capability"]["model"]
    assert snapshot_model["pg333_diagnostic"]["source_row_count"] == 18
    assert next(item for item in build_research_ops_snapshot()["capability"]["metrics"] if item["id"] == "pg333-webgoat")["status"] == "blocked"


def test_pg333_webgoat_projection_missing_artifacts_is_pending():
    projected = _pg333_webgoat_projection({}, {}, {}, {})
    assert projected["status"] == "pending"
    assert projected["source_row_count"] == 0
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg333_cross_impl_projection_is_bounded_and_context_only():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg333_cross_impl_projection(
        load("pg333_three_impl_get_post_diagnostic_source_rows_v1.json"),
        load("pg333_three_impl_get_post_diagnostic_source_audit_v1.json"),
        load("pg333_three_impl_get_post_diagnostic_information_audit_v1.json"),
        load("pg333_three_impl_get_post_diagnostic_vocabulary_v1.json"),
        load("pg333_three_impl_get_post_diagnostic_capacity_v1.json"),
        load("pg333_three_impl_a800_representation_e1_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["merged_record_count"] == 45
    assert projected["implementation_count"] == 3
    assert projected["source_split_counts"]["train"] == 9
    assert projected["information_audit_status"] == "diagnostic"
    assert projected["required_context_window"] == 4145
    assert projected["a800_train_rows"] == 9
    assert projected["a800_holdout_rows"] == 36
    assert projected["a800_target_tokens_read"] is False
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg333_cross_impl_projection_missing_artifacts_is_pending():
    projected = _pg333_cross_impl_projection({}, {}, {}, {}, {}, {})
    assert projected["status"] == "pending"
    assert projected["merged_record_count"] == 0
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg334_process_token_projection_is_bounded_and_candidate_only():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg334_process_token_projection(
        load("pg334_process_token_diagnostic_v1.json"),
        load("pg334_process_token_diagnostic_audit_v1.json"),
        load("pg334_process_token_vocabulary_v1.json"),
        load("pg334_a800_process_representation_e1_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["record_count"] == 576
    assert projected["pre_question_count"] == 288
    assert projected["negative_count"] == 288
    assert projected["a800_target_tokens_read"] is False
    assert projected["training_allowed"] is False
    assert "context_tokens" not in projected


def test_pg334_process_token_projection_missing_artifacts_is_pending():
    projected = _pg334_process_token_projection({}, {}, {}, {})
    assert projected["status"] == "pending"
    assert projected["promotion"]["memory_promotion_allowed"] is False


def test_pg335_real_process_projection_is_bounded_and_tracks_ask_failure_negative():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg335_real_process_projection(
        load("pg335_real_process_token_diagnostic_v1.json"),
        load("pg335_real_process_token_diagnostic_audit_v1.json"),
        load("pg335_real_process_token_vocabulary_v1.json"),
        load("pg335_a800_process_representation_e1_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["source_row_count"] == 45
    assert projected["ask_count"] == 315
    assert projected["failure_count"] == 15
    assert projected["negative_review_count"] == 15
    assert projected["a800_target_tokens_read"] is False
    assert projected["training_allowed"] is False
    assert "context_tokens" not in projected


def test_pg335_real_process_projection_missing_artifacts_is_pending():
    projected = _pg335_real_process_projection({}, {}, {}, {})
    assert projected["status"] == "pending"
    assert all(value is False for value in projected["promotion"].values())


def test_pg336_real_failure_process_projection_is_bounded_and_blocked():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg336_real_failure_process_projection(
        load("pg336_real_failure_process_token_v1.json"),
        load("pg336_real_failure_process_token_audit_v1.json"),
        load("pg336_real_failure_process_vocabulary_v1.json"),
        {},
    )
    assert projected["status"] == "blocked_incomplete"
    assert projected["record_count"] == 180
    assert projected["ask_preflight_count"] == 135
    assert projected["failure_repair_count"] == 9
    assert projected["negative_review_count"] == 9
    assert projected["independent_implementation_holdout"] is False
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())
    encoded = json.dumps(projected, ensure_ascii=False)
    for forbidden in ('"records"', '"context_tokens"', '"target_tokens"', '"payload"', '"response_body"', '"oracle"'):
        assert forbidden not in encoded


def test_pg336_real_failure_process_projection_missing_artifacts_is_pending():
    projected = _pg336_real_failure_process_projection({}, {}, {}, {})
    assert projected["status"] == "pending"
    assert projected["record_count"] == 0
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg336_real_failure_process_a800_smoke_is_visible_but_not_promoted():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg336_real_failure_process_projection(
        load("pg336_real_failure_process_token_v1.json"),
        load("pg336_real_failure_process_token_audit_v1.json"),
        load("pg336_real_failure_process_vocabulary_v1.json"),
        load("pg336_a800_real_failure_representation_e1_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["a800_train_rows"] == 60
    assert projected["a800_holdout_rows"] == 120
    assert projected["a800_target_tokens_read"] is False
    assert projected["independent_implementation_holdout"] is False
    assert projected["training_allowed"] is False
    model = build_research_ops_snapshot()["capability"]["model"]["pg336_real_failure_process_token_diagnostic"]
    assert model["a800_status"] == "representation_pretrain_candidate_only"
    assert model["a800_target_tokens_read"] is False
    metric = next(item for item in build_research_ops_snapshot()["capability"]["metrics"] if item["id"] == "pg336-real-failure-process-tokens")
    assert metric["status"] == "blocked"


def test_pg337_cross_impl_process_projection_is_bounded_and_blocked():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg337_cross_impl_process_projection(
        load("pg337_cross_impl_process_token_v1.json"),
        load("pg337_cross_impl_process_token_audit_v1.json"),
        load("pg337_cross_impl_process_vocabulary_v1.json"),
        load("pg337_a800_cross_impl_representation_e1_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["record_count"] == 183
    assert projected["train_count"] == 60
    assert projected["implementation_holdout_count"] == 123
    assert projected["real_dvwa_failure_rows"] == 2
    assert projected["a800_train_rows"] == 60
    assert projected["a800_holdout_rows"] == 123
    assert projected["a800_target_tokens_read"] is False
    assert projected["independent_implementation_holdout"] is True
    assert projected["training_allowed"] is False
    encoded = json.dumps(projected, ensure_ascii=False)
    for forbidden in ('"records"', '"context_tokens"', '"target_tokens"', '"payload"', '"response_body"', '"oracle"'):
        assert forbidden not in encoded


def test_pg337_cross_impl_projection_missing_artifacts_is_pending():
    projected = _pg337_cross_impl_process_projection({}, {}, {}, {})
    assert projected["status"] == "pending"
    assert projected["record_count"] == 0
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg337_snapshot_prefers_latest_a800_candidate_without_promotion():
    model = build_research_ops_snapshot()["capability"]["model"]["pg337_cross_impl_process_token_diagnostic"]
    assert model["a800_epochs"] == 8
    assert model["a800_target_tokens_read"] is False
    assert model["training_allowed"] is False
    assert all(value is False for value in model["promotion"].values())


def test_pg338_information_preserving_projection_is_bounded_and_blocked():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg338_information_preserving_projection(
        load("pg338_information_preserving_process_token_v1.json"),
        load("pg338_information_preserving_process_audit_v1.json"),
        load("pg338_information_preserving_vocabulary_v1.json"),
        load("pg338_a800_information_preserving_representation_e1_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["record_count"] == 27
    assert projected["full_axis_rows"] == 27
    assert projected["train_count"] == 18
    assert projected["implementation_holdout_count"] == 9
    assert projected["a800_target_tokens_read"] is False
    assert projected["training_allowed"] is False
    assert projected["axis_entropy"]["response_transport"]["ablation_changed_rate"] == 1.0
    encoded = json.dumps(projected, ensure_ascii=False)
    for forbidden in ('"records"', '"context_tokens"', '"target_tokens"', '"payload"', '"response_body"', '"oracle"'):
        assert forbidden not in encoded


def test_pg338_snapshot_exposes_full_axis_candidate_without_promotion():
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg338_information_preserving_process_token_diagnostic"]
    assert model["full_axis_rows"] == 27
    assert model["a800_epochs"] == 2
    assert model["a800_target_tokens_read"] is False
    assert model["training_allowed"] is False
    metric = next(item for item in snapshot["capability"]["metrics"] if item["id"] == "pg338-information-preserving-process-tokens")
    assert metric["status"] == "blocked"


def test_pg339_multi_shape_projection_is_bounded_and_blocked():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg339_multi_shape_projection(
        load("pg339_multi_shape_diagnostic_dataset_v1.json"),
        load("pg339_multi_shape_diagnostic_audit_v1.json"),
        load("pg339_multi_shape_vocabulary_v1.json"),
        load("pg339_a800_multi_shape_representation_e6_v1.json"),
    )
    assert projected["record_count"] == 24
    assert projected["train_count"] == 9
    assert projected["shape_holdout_count"] == 15
    assert projected["a800_target_tokens_read"] is False
    assert projected["a800_shape_holdout_entropy_min"] > 0
    assert projected["promotion"]["vulnerability_claim_allowed"] is False
    assert "axis_entropy" in projected
    assert "context_tokens" not in projected


def test_pg339_snapshot_exposes_shape_holdout_without_promotion():
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg339_multi_shape_information_preserving_diagnostic"]
    assert model["shape_holdout_count"] == 15
    assert model["training_allowed"] is False
    metric = next(item for item in snapshot["capability"]["metrics"] if item["id"] == "pg339-multi-shape-information-preserving")
    assert metric["status"] == "blocked"


def test_pg340_balanced_axis_projection_is_bounded_and_fail_closed():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg340_balanced_axis_projection(
        load("pg340_balanced_axis_representation_dataset_v1.json"),
        load("pg340_balanced_axis_representation_audit_v1.json"),
        load("pg340_balanced_axis_vocabulary_v1.json"),
        load("pg340_a800_balanced_axis_representation_e1_v1.json"),
    )
    assert projected["record_count"] == 21
    assert projected["train_count"] == 15
    assert projected["shape_holdout_count"] == 6
    assert projected["train_implementation_count"] == 2
    assert projected["holdout_implementation_count"] == 1
    assert projected["a800_target_tokens_read"] is False
    assert projected["a800_entropy_drop_max"] == 0.0
    assert projected["training_allowed"] is False
    encoded = json.dumps(projected, ensure_ascii=False)
    for forbidden in ('"records"', '"context_tokens"', '"target_tokens"', '"payload"', '"response_body"', '"oracle"'):
        assert forbidden not in encoded


def test_pg340_snapshot_exposes_balanced_axis_candidate_without_promotion():
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg340_balanced_axis_representation_diagnostic"]
    assert model["record_count"] == 21
    assert model["train_implementation_count"] == 2
    assert model["training_allowed"] is False
    metric = next(item for item in snapshot["capability"]["metrics"] if item["id"] == "pg340-balanced-axis-representation")
    assert metric["status"] == "blocked"


def test_pg341_target_conditioned_projection_is_bounded_and_keeps_full_axis_blocked():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg341_target_conditioned_projection(
        load("pg341_target_conditioned_process_full_axis_dataset_v1.json"),
        load("pg341_target_conditioned_audit_v1.json"),
        load("pg341_target_conditioned_vocabulary_v1.json"),
        [load("pg341_a800_target_conditioned_smoke_e4_v1.json")],
    )
    assert projected["coarse_process_count"] == 183
    assert projected["full_axis_count"] == 27
    assert projected["coarse_diagnostic_training_allowed"] is True
    assert projected["full_axis_target_training_allowed"] is False
    assert projected["a800_target_tokens_read"] is True
    assert projected["promotion"]["training_allowed"] is False
    encoded = json.dumps(projected, ensure_ascii=False).casefold()
    for forbidden in ('"context_tokens"', '"target_tokens"', '"response_body"', '"payload"', '"oracle"', '"evaluator"', '"records"'):
        assert forbidden not in encoded


def test_pg341_snapshot_exposes_two_view_diagnostic_without_promotion():
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg341_target_conditioned_two_view_diagnostic"]
    assert model["coarse_process_count"] == 183
    assert model["full_axis_target_training_allowed"] is False
    assert model["training_allowed"] is False
    metric = next(item for item in snapshot["capability"]["metrics"] if item["id"] == "pg341-target-conditioned-two-view")
    assert metric["status"] == "blocked"


def test_pg342_full_axis_failure_repair_projection_is_bounded_and_diagnostic_only():
    def load(name):
        return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))

    projected = _pg342_full_axis_failure_repair_projection(
        load("pg342_full_axis_failure_repair_dataset_v1.json"),
        load("pg342_full_axis_failure_repair_audit_v1.json"),
        load("pg342_full_axis_failure_repair_vocabulary_v1.json"),
        load("pg342_a800_full_axis_representation_smoke_v2.json"),
        load("pg342_webgoat_failure_repair_report_v1.json"),
    )
    assert projected["status"] == "completed_diagnostic_only"
    assert projected["record_count"] == 15
    assert projected["train_count"] == 6
    assert projected["implementation_holdout_count"] == 9
    assert projected["get_count"] == 1
    assert projected["post_count"] == 1
    assert projected["a800_target_tokens_read"] is False
    assert projected["a800_entropy_gate_passed"] is True
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())
    encoded = json.dumps(projected, ensure_ascii=False).casefold()
    for forbidden in ('"context_tokens"', '"target_tokens"', '"response_body"', '"payload"', '"oracle"', '"evaluator"', '"records"'):
        assert forbidden not in encoded


def test_pg342_snapshot_exposes_failure_repair_diagnostic_without_promotion():
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg342_full_axis_failure_repair_diagnostic"]
    assert model["record_count"] == 15
    assert model["a800_implementation_holdout_rows"] == 9
    assert model["a800_target_tokens_read"] is False
    assert model["training_allowed"] is False
    metric = next(item for item in snapshot["capability"]["metrics"] if item["id"] == "pg342-full-axis-failure-repair")
    assert metric["status"] == "blocked"


def test_pg331_typed_capacity_projection_is_read_only_and_bounded():
    capacity = json.loads((ROOT / "research" / "pg331_pikachu_typed_model_capacity_audit_v1.json").read_text(encoding="utf-8-sig"))
    projected = _pg331_typed_capacity_projection(capacity)
    assert projected["status"] == "blocked"
    assert projected["context_min"] == 3165
    assert projected["context_max"] == 3284
    assert projected["required_context_window"] == 4145
    assert projected["model_vocabulary_size"] == 1066
    assert projected["variant_max_length"]["pg322_legacy"] == 72
    assert projected["variant_max_length"]["pg331_minimum"] == 4145
    assert projected["truncation_risk"] is True
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())

    model = build_research_ops_snapshot()["capability"]["model"]["pg331_information_preservation"]
    assert model["typed_capacity"]["status"] == "blocked"
    assert model["typed_capacity"]["required_context_window"] == 4145


def test_pg331_typed_capacity_missing_artifact_is_pending():
    projected = _pg331_typed_capacity_projection({})
    assert projected["status"] == "pending"
    assert projected["context_min"] == 0
    assert projected["context_max"] == 0
    assert projected["variant_max_length"] == {}
    assert projected["truncation_risk"] is False
    assert projected["training_allowed"] is False
    assert all(value is False for value in projected["promotion"].values())


def test_pg331_train_holdout_diagnostic_v2_is_bounded_and_fail_closed():
    names = {
        "dataset": "pg331_train_holdout_diagnostic_v2.json",
        "source_audit": "pg331_train_holdout_diagnostic_source_audit_v2.json",
        "vocabulary": "pg331_train_holdout_diagnostic_vocab_v2.json",
        "information": "pg331_train_holdout_diagnostic_information_v2.json",
        "capacity": "pg331_train_holdout_diagnostic_capacity_v2.json",
        "plan": "pg331_train_holdout_diagnostic_plan_v2.json",
    }
    documents = {key: json.loads((ROOT / "research" / value).read_text(encoding="utf-8-sig")) for key, value in names.items()}
    projected = _pg331_train_holdout_diagnostic_v2_projection(
        documents["dataset"], documents["dataset"], documents["source_audit"], documents["vocabulary"], documents["information"], documents["capacity"], documents["plan"]
    )
    assert projected["status"] == "diagnostic_blocked"
    assert projected["counts"]["dataset_records"] == 27
    assert projected["counts"]["plan_eligible_train_rows"] == 0
    assert projected["counts"]["plan_holdout_rows"] == 18
    assert projected["axis_entropy"]["request_transport"]["bits"] == 1.974938
    assert projected["capacity"]["required_context_window"] == 4610
    assert all(value is False for value in projected["promotion"].values())
    encoded = json.dumps(projected, ensure_ascii=False)
    assert "context_tokens" not in encoded and "target_tokens" not in encoded and '"records"' not in encoded
    assert build_research_ops_snapshot()["capability"]["model"]["pg331_information_preservation"]["train_holdout_diagnostic_v2"]["training_allowed"] is False


def test_pg331_train_holdout_diagnostic_v2_missing_artifacts_stays_pending():
    projected = _pg331_train_holdout_diagnostic_v2_projection({}, {}, {}, {}, {}, {}, {})
    assert projected["status"] == "pending"
    assert len(projected["missing_artifacts"]) == 7
    assert projected["axis_entropy"] == {}
    assert all(value is False for value in projected["promotion"].values())
