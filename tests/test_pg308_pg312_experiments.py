from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from app.pg313_probe_variant import bind_probe_variant_plan, probe_target_for_context


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg308_multisource_audit_and_permutation_lane() -> None:
    dataset = _load("pg308_multisource_slot_dataset_v1.json")
    audit = _load("pg308_multisource_slot_dataset_audit_v1.json")
    assert audit["status"] == "passed"
    assert dataset["counts"]["source_count"] >= 5
    assert dataset["counts"]["pg269_train"] > 0
    assert dataset["counts"]["pg266_holdout"] > 0
    assert dataset["counts"]["pg268_hard_holdout"] > 0
    assert dataset["counts"]["slot_permutation_hard_negative"] > 0
    assert all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in dataset["records"])


def test_pg309_keeps_complete_and_missing_pairs_explicit() -> None:
    dataset = _load("pg309_balanced_counterfactual_dataset_v1.json")
    audit = _load("pg309_balanced_counterfactual_dataset_audit_v1.json")
    assert audit["status"] == "passed"
    assert dataset["counts"]["generated_complete"] > 0
    assert dataset["counts"]["generated_missing"] > 0
    assert dataset["counts"]["generated_repair"] > 0
    assert dataset["counts"]["generated_mismatch"] > 0
    assert dataset["contract"]["paired_missing_complete"] is True


def test_pg310_and_pg311_show_safety_without_claiming_gate_pass() -> None:
    pg310 = _load("pg310_optimization_ablation_report_v1_local_morning.json")
    wide = pg310["variants"]["wide_zero_dropout"]
    assert wide["metrics"]["hard_bound_false_allow"]["max"] == 0
    assert wide["metrics"]["holdout_bound_slot_exact"]["min"] >= 0.9
    assert wide["metrics"]["holdout_missing_question_recall"]["min"] < 0.9

    pg311 = _load("pg311_wide_question_anchor_report_v1_local_morning.json")
    assert pg311["hypothesis_gate"]["status"] == "blocked"
    assert pg311["metrics"]["holdout_bound_slot_exact"]["min"] == 1.0
    assert pg311["metrics"]["holdout_bound_false_allow"]["max"] == 0
    assert pg311["metrics"]["holdout_missing_question_recall"]["min"] < 0.9


def test_pg312_wide_checkpoint_really_sent_and_was_typed_confirmed_locally() -> None:
    report = _load("pg312_live_wide_checkpoint_replay_report_v1.json")
    assert report["status"] == "completed_real_local_docker_evaluator"
    assert report["model"]["symbolic_checkpoint"] is True
    assert report["model"]["wire_generation"] == "source_grounded_binding_after_guard"
    assert report["counts"]["model_candidate_send_count"] == 4
    assert report["counts"]["model_confirmed_effect_count"] == 4
    assert report["counts"]["false_positive_count"] == 0
    assert report["checks"]["real_docker_contacted"] is True
    assert report["checks"]["loopback_only"] is True
    assert report["checks"]["typed_evidence_hash_per_route"] is True
    assert report["checks"]["raw_payload_in_model_context"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False


def test_pg313_keeps_probe_variants_abstract_and_blocks_worst_seed_promotion() -> None:
    dataset = _load("pg313_probe_variant_dataset_v1.json")
    audit = _load("pg313_probe_variant_dataset_audit_v1.json")
    report = _load("pg313_probe_variant_moe_training_report_v1_local_morning.json")
    assert audit["status"] == "passed"
    assert dataset["counts"]["total"] == 351
    assert dataset["counts"]["generated_variant_rows"] == 36
    assert dataset["counts"]["candidate_variant_rows"] == 12
    assert dataset["counts"]["reference_variant_rows"] == 12
    assert dataset["counts"]["negative_variant_rows"] == 12
    assert all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in dataset["records"])
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["metrics"]["holdout_missing_question_recall"]["min"] == 0.88
    assert report["metrics"]["holdout_bound_base_slot_exact"]["min"] < 0.9
    assert report["metrics"]["holdout_variant_exact"]["min"] < 0.9
    assert report["metrics"]["holdout_unnecessary_question"]["max"] == 0.0
    assert report["metrics"]["holdout_bound_false_allow"]["max"] == 0
    assert report["metrics"]["hard_bound_false_allow"]["max"] > 0
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["payload_catalog_promotion_allowed"] is False

    context = [
        "surface_method=GET",
        "surface_field_role=query_param",
        "surface_encoding=url_percent",
        "history_action=reference_request",
        "typed_available=1",
        "feedback_state=baseline_observed",
        "replay_ready=1",
        "evidence_present=1",
        "negative_control=1",
        "fresh_reset=1",
    ]
    symbolic = probe_target_for_context(context)
    bound = bind_probe_variant_plan(symbolic, context)
    assert any(token == "probe_variant_ref=reference_canary" for token in symbolic)
    assert any(token == "encoding_chain_ref=surface_encoding" for token in symbolic)
    assert any(token == "probe_variant=reference_canary" for token in (bound or []))
    assert any(token == "encoding_chain=url_percent" for token in (bound or []))


def test_pg314_replays_model_selected_variants_on_independent_get_post_image() -> None:
    report = _load("pg314_independent_variant_replay_report_v1.json")
    dataset = _load("pg314_independent_variant_training_dataset_v1.json")
    catalog = _load("pg314_independent_variant_human_catalog_v1.json")
    assert report["status"] == "completed_real_local_docker_independent_variant_replay"
    assert report["runtime"]["network"] == "none"
    assert report["counts"]["route_count"] == 2
    assert report["counts"]["get_count"] == 1
    assert report["counts"]["post_count"] == 1
    assert report["counts"]["model_variant_exact_count"] == 6
    assert report["counts"]["model_variant_send_count"] == 6
    assert report["counts"]["model_typed_effect_count"] == 2
    assert report["counts"]["negative_lane_violation_count"] == 0
    assert report["preflight_identifiability"]["question_recall"] == 1.0
    assert report["preflight_identifiability"]["unsafe_allow"] == 0
    assert report["checks"]["fresh_reset_per_route"] is True
    assert report["checks"]["zero_volume_per_route"] is True
    assert report["checks"]["source_attestation_per_route"] is True
    assert report["checks"]["typed_evidence_hash_per_route"] is True
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["hypothesis_gate"]["checks"]["variant_selection_exact"] is True
    assert report["hypothesis_gate"]["checks"]["typed_effect_on_all_routes"] is True
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in dataset["records"])
    assert catalog["raw_payloads_human_review_only"] is True
    assert catalog["raw_response_bodies_stored"] is False
    assert all(row["oracle"]["typed_effect_confirmed"] for row in catalog["entries"] if row["oracle"]["candidate_positive"])


def test_pg315_exposes_worst_seed_variant_and_failure_repair_regressions() -> None:
    report = _load("pg315_worst_seed_replay_report_v1.json")
    dataset = _load("pg315_worst_seed_training_dataset_v1.json")
    assert report["status"] == "completed_real_local_docker_all_seed_replay"
    assert report["counts"]["seed_count"] == 3
    assert report["counts"]["route_count"] == 6
    assert report["counts"]["get_count"] == 3
    assert report["counts"]["post_count"] == 3
    assert report["counts"]["model_variant_role_count"] == 18
    assert report["counts"]["model_variant_exact_count"] == 14
    assert report["counts"]["model_typed_effect_count"] == 4
    assert report["counts"]["negative_lane_violation_count"] == 2
    assert report["counts"]["repair_row_count"] == 6
    assert report["counts"]["repair_abstain_correct_count"] == 0
    assert report["worst_seed_metrics"]["question_recall_min"] == 1.0
    assert report["worst_seed_metrics"]["variant_exact_min"] < 0.9
    assert report["worst_seed_metrics"]["repair_abstain_min"] == 0.0
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["hypothesis_gate"]["checks"]["variant_exact_worst_seed"] is False
    assert report["hypothesis_gate"]["checks"]["repair_abstain_worst_seed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in dataset["records"])


def test_pg316_repair_anchor_improves_repair_without_hiding_question_gap() -> None:
    dataset = _load("pg316_failure_repair_dataset_v1.json")
    audit = _load("pg316_failure_repair_dataset_audit_v1.json")
    report = _load("pg316_failure_repair_moe_training_report_v1_local_morning.json")
    live = _load("pg316_live_independent_variant_replay_report_v1.json")
    assert audit["status"] == "passed"
    assert dataset["counts"]["total"] == 479
    assert dataset["counts"]["repair_rows"] == 128
    assert dataset["counts"]["repair_target_next_action"]["repair_abstract_plan"] == 128
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["metrics"]["holdout_repair_exact"]["min"] == 1.0
    assert report["metrics"]["holdout_variant_exact"]["min"] == 1.0
    assert report["metrics"]["hard_bound_false_allow"]["max"] == 0
    assert report["metrics"]["holdout_missing_question_recall"]["min"] == 0.88
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert live["status"] == "completed_real_local_docker_pg316_live_replay"
    assert live["counts"]["variant_exact_count"] == 6
    assert live["counts"]["model_typed_effect_count"] == 2
    assert live["failure_repair"]["correct"] == 2
    assert live["failure_repair"]["rate"] == 1.0
    assert live["counts"]["negative_lane_violation_count"] == 0
    assert live["promotion"]["training_allowed"] is False


def test_pg317_multi_missing_question_anchor_and_live_gate() -> None:
    dataset = _load("pg317_question_anchor_dataset_v1.json")
    audit = _load("pg317_question_anchor_dataset_audit_v1.json")
    report = _load("pg317_question_anchor_moe_training_report_v1_local_morning.json")
    live = _load("pg317_live_independent_variant_replay_report_v1.json")
    assert audit["status"] == "passed"
    assert dataset["counts"]["anchor_rows"] == 352
    assert dataset["counts"]["ask_rows"] == 330
    assert dataset["counts"]["complete_rows"] == 22
    assert dataset["contract"]["multi_missing_observation_pairs"] is True
    assert all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in dataset["records"])
    assert report["hypothesis_gate"]["status"] == "blocked"
    assert report["metrics"]["holdout_anchor_question_exact"]["min"] >= 0.95
    assert report["metrics"]["holdout_anchor_safe_allow_max"]["max"] == 0
    assert report["metrics"]["holdout_anchor_unnecessary_question"]["max"] == 0
    assert report["metrics"]["holdout_variant_exact"]["min"] >= 0.9
    assert report["metrics"]["holdout_repair_exact"]["min"] >= 0.9
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert live["status"] == "completed_real_local_docker_pg317_live_replay"
    assert live["counts"]["get_count"] == 1
    assert live["counts"]["post_count"] == 1
    assert live["counts"]["variant_exact_count"] == 6
    assert live["counts"]["model_typed_effect_count"] == 2
    assert live["counts"]["negative_lane_violation_count"] == 0
    assert live["preflight_identifiability"]["question_recall"] == 1.0
    assert live["failure_repair"]["rate"] == 1.0
    assert live["checks"]["fresh_reset_per_route"] is True
    assert live["checks"]["typed_evidence_hash_per_route"] is True
    assert live["promotion"]["training_allowed"] is False


def test_pg318_family_holdout_replay_keeps_missing_observation_gate_closed() -> None:
    report = _load("pg318_family_holdout_replay_report_v1.json")
    catalog = _load("pg318_family_holdout_human_catalog_v1.json")
    trace = _load("pg318_family_holdout_trace_v1.json")
    protocol = _load("pg318_family_holdout_protocol_v1.json")
    assert report["status"] == "completed_real_local_docker_pg318_family_holdout"
    assert report["counts"] == {
        **report["counts"],
        "seed_count": 3,
        "route_count": 18,
        "get_count": 15,
        "post_count": 3,
        "sql_route_count": 9,
        "xss_route_count": 9,
        "variant_role_count": 54,
        "variant_exact_count": 54,
        "model_send_count": 54,
        "typed_effect_count": 18,
        "negative_lane_violation_count": 0,
        "failure_repair_correct_count": 18,
        "failure_repair_count": 18,
        "multi_missing_question_rows": 270,
        "multi_missing_unsafe_allow": 0,
    }
    assert report["worst_seed_metrics"]["multi_missing_question_recall_min"] == 1.0
    assert report["worst_seed_metrics"]["typed_effect_route_rate_min"] == 1.0
    assert report["checks"]["docker_network_none"] is True
    assert report["checks"]["raw_payload_in_model_context"] is False
    assert report["checks"]["raw_response_bodies_stored"] is False
    assert report["hypothesis_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert len(catalog["entries"]) == 18
    assert all(row["oracle"]["typed_effect_confirmed"] for row in catalog["entries"])
    assert all(row["oracle"]["evidence_sha256"] for row in catalog["entries"])
    assert all(not row["model"]["entries"][0]["context_tokens"] or all("<" not in token for token in row["model"]["entries"][0]["context_tokens"]) for row in catalog["entries"])
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert protocol["scope"]["network"] == "none"
    assert protocol["required_gates"]["docker_network_none"] is True


def test_pg321_role_conditioned_replay_keeps_typed_and_missing_gates_closed() -> None:
    report = _load("pg321_family_holdout_replay_report_v1.json")
    catalog = _load("pg321_family_holdout_human_catalog_v1.json")
    trace = _load("pg321_family_holdout_trace_v1.json")
    protocol = _load("pg321_family_holdout_protocol_v1.json")
    assert report["status"] == "completed_real_local_docker_pg321_family_holdout"
    assert report["counts"]["seed_count"] == 3
    assert report["counts"]["route_count"] == 18
    assert report["counts"]["get_count"] == 15
    assert report["counts"]["post_count"] == 3
    assert report["counts"]["variant_role_count"] == 54
    assert report["counts"]["variant_exact_count"] == 54
    assert report["counts"]["typed_effect_count"] == 18
    assert report["counts"]["negative_lane_violation_count"] == 0
    assert report["counts"]["multi_missing_question_rows"] == 270
    assert report["counts"]["multi_missing_unsafe_allow"] == 0
    assert report["worst_seed_metrics"]["variant_exact_min"] == 1.0
    assert report["worst_seed_metrics"]["typed_effect_route_rate_min"] == 1.0
    assert report["worst_seed_metrics"]["multi_missing_question_recall_min"] == 1.0
    assert report["checks"]["docker_network_none"] is True
    assert report["checks"]["raw_payload_in_model_context"] is False
    assert report["checks"]["raw_response_bodies_stored"] is False
    assert report["hypothesis_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert len(catalog["entries"]) == 18
    assert all(row["oracle"]["typed_effect_confirmed"] for row in catalog["entries"])
    assert all(row["oracle"]["evidence_sha256"] for row in catalog["entries"])
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert protocol["scope"]["network"] == "none"
    assert protocol["required_gates"]["docker_network_none"] is True


def test_pg323_decoy_ask_anchor_vulnerableapp_replay_keeps_all_hard_gates_closed() -> None:
    report = _load("pg323_vulnerableapp_role_replay_report_v1.json")
    catalog = _load("pg323_vulnerableapp_role_catalog_v1.json")
    trace = _load("pg323_vulnerableapp_role_trace_v1.json")
    protocol = _load("pg323_vulnerableapp_role_protocol_v1.json")
    assert report["status"] == "completed_real_local_docker_pg323_vulnerableapp_role_replay"
    assert report["counts"] == {
        "seed_count": 3,
        "route_count": 18,
        "get_count": 12,
        "post_count": 6,
        "positive_route_count": 6,
        "positive_typed_effect_count": 6,
        "variant_role_count": 54,
        "variant_exact_count": 54,
        "model_send_count": 36,
        "negative_lane_violation_count": 0,
        "failure_repair_correct_count": 18,
        "failure_repair_count": 18,
        "multi_missing_question_rows": 270,
        "multi_missing_unsafe_allow": 0,
    }
    assert report["worst_seed_metrics"]["variant_exact_min"] == 1.0
    assert report["worst_seed_metrics"]["positive_typed_effect_route_rate_min"] == 1.0
    assert report["worst_seed_metrics"]["multi_missing_question_recall_min"] == 1.0
    assert report["hypothesis_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert len(catalog["entries"]) == 18
    assert all(row["oracle"]["typed_effect_confirmed"] for row in catalog["entries"] if row["oracle"]["candidate_positive"])
    assert all(row["oracle"]["evidence_sha256"] for row in catalog["entries"])
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert protocol["scope"]["network"] == "none"
    assert protocol["required_gates"]["docker_network_none"] is True


def test_pg324_runner_uses_fresh_typed_state_delta_and_forbids_domain_writes() -> None:
    source = (ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py").read_text(encoding="utf-8")
    assert "def _challenge_state" in source
    assert "challenge_state_baseline_solved" in source
    assert "challenge_state_delta" in source
    assert "\"domain_data_write\": False" in source
    assert "\"allowed_evaluator_transition\"" in source
    assert "dom_script_execution" in source
    assert "GenericBeliefState" in source
    assert "belief_trace_complete" in source
    assert "\"belief_update\": True" in source
    assert "_failure_transition_for_row" in source
    assert "failure_action_changed_all" in source
    assert "\"failure_action_changed\": True" in source


def test_pg324_typed_state_delta_truth_table() -> None:
    path = ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py"
    spec = importlib.util.spec_from_file_location("pg324_state_delta_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    delta = module._challenge_state_delta
    assert delta(baseline_available=True, baseline_solved=False, current_available=True, current_solved=True) is True
    assert delta(baseline_available=True, baseline_solved=True, current_available=True, current_solved=True) is False
    assert delta(baseline_available=False, baseline_solved=False, current_available=True, current_solved=True) is False
    assert delta(baseline_available=True, baseline_solved=False, current_available=False, current_solved=True) is False
    assert delta(baseline_available=True, baseline_solved=False, current_available=True, current_solved=False) is False


def test_pg324_safety_mode_override_is_explicitly_attested() -> None:
    path = ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py"
    spec = importlib.util.spec_from_file_location("pg324_safety_mode_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SAFETY_MODE_CONFIG == '{"challenges":{"safetyMode":"disabled"}}'
    assert len(module.SAFETY_MODE_CONFIG_SHA256) == 64
    source = path.read_text(encoding="utf-8")
    assert "safety_mode_override_all" in source
    assert "--env" in source and "NODE_CONFIG=" in source


def test_pg324_failure_transition_contract_is_abstract_and_changed() -> None:
    path = ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py"
    spec = importlib.util.spec_from_file_location("pg324_failure_transition_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.EVAL.PG314.context_tokens(
        "GET", history_action="candidate_request", failure_class="none", typed_available="1",
        replay_ready="1", evidence_present="1", negative_control="1", fresh_reset="1",
    )
    row = {
        "model": {
            "entries": [{
                "role": "candidate_request",
                "sent": True,
                "context_tokens": context,
                "proposal": {"guarded_tokens": [
                    "[TARGET_BOS]", "question=none", "next_action=assemble_abstract_plan",
                    "repair_action=none", "transport=GET", "field_role=query_param",
                    "encoding=url_percent", "canary=runtime", "oracle=typed",
                    "stop_condition=typed_effect_or_abstain", "safe_to_send=1",
                    "probe_variant=source_attested_candidate", "encoding_chain=url_percent", "[TARGET_EOS]",
                ]},
            }],
            "failure_prediction": {"guarded_tokens": [
                "[TARGET_BOS]", "question=none", "next_action=repair_abstract_plan",
                "repair_action=retry_bounded_variant", "transport=GET", "field_role=query_param",
                "encoding=url_percent", "canary=runtime", "oracle=typed",
                "stop_condition=repair_feedback_or_abstain", "safe_to_send=0",
                "probe_variant=none", "encoding_chain=none", "[TARGET_EOS]",
            ]},
        }
    }
    transition = module._failure_transition_for_row(row)
    assert transition["repair_transition_valid"] is True
    assert transition["action_changed"] is True
    assert transition["previous_action"] != transition["next_action"]


def test_pg324_failure_transition_does_not_penalize_safe_abstain() -> None:
    path = ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py"
    spec = importlib.util.spec_from_file_location("pg324_failure_abstain_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        "model": {
            "entries": [{"role": "candidate_request", "sent": False, "proposal": {"guarded_tokens": ["next_action=request_observation"]}}],
            "failure_prediction": {"guarded_tokens": ["next_action=request_observation"]},
        }
    }
    transition = module._failure_transition_for_row(row)
    assert transition["repair_transition_required"] is False
    assert transition["repair_transition_valid"] is True
    assert transition["action_changed"] is None


def test_pg324_model_context_allowlist_rejects_evaluator_metadata() -> None:
    path = ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py"
    spec = importlib.util.spec_from_file_location("pg324_context_firewall_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.EVAL.PG314.context_tokens(
        "GET", history_action="candidate_request", failure_class="none", typed_available="1",
        replay_ready="1", evidence_present="1", negative_control="1", fresh_reset="1",
    )
    humans = [{"model": {"entries": [{"context_tokens": context}], "failure_context": context}}]
    abstracts = [{"context_tokens": context}]
    assert module._model_context_firewall(humans, abstracts) is True
    leaked = list(context)
    leaked.insert(-1, "family=xss")
    assert module._model_context_firewall(
        [{"model": {"entries": [{"context_tokens": leaked}], "failure_context": context}}],
        abstracts,
    ) is False


def test_pg324_artifact_audit_checks_actual_context_tokens() -> None:
    path = ROOT / "scripts" / "audit_pg324_source_heldout_report.py"
    spec = importlib.util.spec_from_file_location("pg324_artifact_context_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    safe = ["[BOS]", "surface_method=GET", "history_action=candidate_request", "[EOS]"]
    catalog = {"entries": [{"model": {"entries": [{"context_tokens": safe}], "failure_context": safe}}]}
    trace = {"episodes": [{"context_tokens": safe}]}
    assert module._context_firewall(catalog, trace) is True
    leaked = [*safe[:-1], "family=xss", safe[-1]]
    catalog["entries"][0]["model"]["entries"][0]["context_tokens"] = leaked
    assert module._context_firewall(catalog, trace) is False
