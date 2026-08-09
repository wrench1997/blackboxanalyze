import json
from pathlib import Path


def test_cross_lab_registry_keeps_unattested_sources_out_of_training():
    report = json.loads(Path("research/pg_pk_24_cross_lab_registry_v1.json").read_text(encoding="utf-8"))
    assert report["read_only"] is True
    assert report["raw_probe_strings_stored"] is False
    assert report["evaluator_labels_stored"] is False
    eligible = [item["target_id"] for item in report["targets"] if item["training_eligible"]]
    assert eligible == ["pikachu", "pg241_pikachu_payload_acceptance", "pg241_payload_capacity_training", "pg242_pikachu_xss_dom_acceptance", "pg244_failure_repair_trajectory", "pg246_vulnerableapp_independent_dom_holdout", "pg247_vulnerableapp_capacity_training", "dual_channel_replay_fixture", "pg34_independent_fixture", "pg35_independent_fixture", "pg36_independent_maze_fixture", "pg37_counterfactual_fixture", "pg40_semantic_router_fixture", "pg42_independent_semantic_fixture", "pg48_compositional_preprobe_fixture", "pg50_stability_matrix_fixture", "pg115_small_rule_ir_train_fixture", "pg116_multisource_trace_training", "pg118_transition_delta_slot_training", "pg119_metadata_transition_slot_training", "pg121_shape_sanitized_rule_ir_training", "pg123_authorization_slot_training", "pg124_failure_conditioned_policy", "pg126_failure_only_policy", "pg127_key_feature_assembly_policy", "pg162_dataset_capacity_sweep", "pg163_large_typed_mix", "pg164_xxl_capacity", "pg165_surface_attestation", "pg166_surface_replay_adaptation", "pg167_multiseed_surface_ood", "pg168_discriminative_slot_augmentation", "pg169_replay_slot_ratio_ablation", "pg170_cross_generator_ood", "pg171_cross_generator_multiseed", "pg172_third_generator_capacity", "pg173_matched_budget_capacity", "pg174_full_holdout_routing", "pg175_joint_routing_loss_search", "pg176_routed_multiseed_new_ood", "pg177_data_capacity_sweep"]
    assert report["evaluation_only_target_count"] == 116
    pg237 = next(item for item in report["targets"] if item["target_id"] == "pg237_pikachu_result_fixture_replay")
    assert pg237["training_eligible"] is False
    assert pg237["methods"] == ["GET", "POST"]
    assert pg237["safety"]["loopback_only"] is True
    assert pg237["safety"]["raw_body_stored"] is False
    pg238 = next(item for item in report["targets"] if item["target_id"] == "pg238_pikachu_surface_replay")
    assert pg238["training_eligible"] is False
    assert pg238["family_set"] == ["xss", "url_redirect"]
    assert pg238["safety"]["loopback_only"] is True
    assert pg238["safety"]["raw_body_stored"] is False
    pg239 = next(item for item in report["targets"] if item["target_id"] == "pg239_alt_pikachu_get_post_replay")
    assert pg239["training_eligible"] is False
    assert pg239["methods"] == ["GET", "POST"]
    assert pg239["safety"]["loopback_only"] is True
    assert pg239["safety"]["environment_failure_not_model_label"] is True
    pg55 = next(item for item in report["targets"] if item["target_id"] == "pg55_invariant_rule_ir_candidate")
    assert pg55["training_eligible"] is False
    assert pg55["feature_transfer_gate"] == "blocked"
    assert pg55["holdout_unknown_misname_count_density_gated"] == 0
    assert pg55["holdout_abstain_rate_density_gated"] == 1.0
    pg56 = next(item for item in report["targets"] if item["target_id"] == "pg56_causal_trace_pretraining")
    assert pg56["training_eligible"] is False
    assert pg56["unknown_family_naming_attempts"] == 0
    assert pg56["promotion_status"] == "pretraining_baseline_family_capability_unproven"
    pg57 = next(item for item in report["targets"] if item["target_id"] == "pg57_rule_ir_oracle_posttraining")
    assert pg57["training_eligible"] is False
    assert pg57["holdout_raw_unknown_confirmed_attempts"] == 12
    assert pg57["holdout_calibrated_unknown_confirmed_attempts"] == 0
    assert pg57["holdout_calibrated_abstain_rate"] == 1.0
    pg58 = next(item for item in report["targets"] if item["target_id"] == "pg58_effect_family_decoupling")
    assert pg58["training_eligible"] is False
    assert pg58["holdout_calibrated_unknown_misname_count"] == 0
    assert pg58["holdout_calibrated_abstain_rate"] == 1.0
    pg59 = next(item for item in report["targets"] if item["target_id"] == "pg59_oracle_semantic_router")
    assert pg59["training_eligible"] is False
    assert pg59["holdout_known_family_recall"] == 1.0
    assert pg59["holdout_unknown_misname_count"] == 0
    pg60 = next(item for item in report["targets"] if item["target_id"] == "pg60_pre_oracle_probe_policy_audit")
    assert pg60["training_eligible"] is False
    assert pg60["confirmation_action_entropy"] == 0.0
    assert pg60["promotion_status"] == "blocked_fixed_action_order"
    pg61 = next(item for item in report["targets"] if item["target_id"] == "pg61_target_zone_counterfactual")
    assert pg61["training_eligible"] is False
    assert pg61["holdout_selected_action_entropy"] >= 0.5
    assert pg61["holdout_negative_false_accept_count"] == 0
    assert pg61["holdout_unknown_strict_abstain"] is True
    assert pg61["promotion_status"] == "hard_gate_passed_diagnostic_only_no_promotion"
    pg62 = next(item for item in report["targets"] if item["target_id"] == "pg62_target_zone_feature_funnel")
    assert pg62["training_eligible"] is False
    assert pg62["accepted_features"] == ["channel_hint"]
    assert pg62["baseline_negative_false_accept_count"] == 0
    assert pg62["promotion_status"] == "feature_funnel_passed_diagnostic_only_no_promotion"
    pg63 = next(item for item in report["targets"] if item["target_id"] == "pg63_independent_target_zone")
    assert pg63["training_eligible"] is False
    assert pg63["canonicalized_target_success_rate"] == 1.0
    assert pg63["canonicalized_negative_false_accept_count"] == 0
    assert pg63["raw_shift_fail_closed"] is True
    assert pg63["promotion_status"] == "independent_canonical_gate_passed_raw_shift_fail_closed_no_promotion"
    pg64 = next(item for item in report["targets"] if item["target_id"] == "pg64_multistep_belief_regret")
    assert pg64["training_eligible"] is False
    assert pg64["active_target_recall"] == 1.0
    assert pg64["active_negative_false_accept_count"] == 0
    assert pg64["active_mean_counterfactual_regret"] == 0.0
    assert pg64["promotion_status"] == "multistep_belief_hard_gate_passed_diagnostic_only_no_promotion"
    pg65 = next(item for item in report["targets"] if item["target_id"] == "pg65_trajectory_policy_head")
    assert pg65["training_eligible"] is False
    assert pg65["safety_gate_status"] == "passed"
    assert pg65["capability_gate_status"] == "blocked"
    assert pg65["dev_policy_accuracy"] < 0.8
    assert pg65["promotion_status"] == "safety_gate_passed_capability_gate_blocked"
    pg66 = next(item for item in report["targets"] if item["target_id"] == "pg66_utility_ranking_head")
    assert pg66["training_eligible"] is False
    assert pg66["dev_decisive_ranking_accuracy"] == 1.0
    assert pg66["pg64_holdout_decisive_ranking_accuracy"] == 1.0
    assert pg66["dev_mean_utility_regret"] == 0.0
    assert pg66["capability_gate_status"] == "passed"
    assert pg66["promotion_status"] == "utility_ranking_gates_passed_fixture_only_no_promotion"
    pg67 = next(item for item in report["targets"] if item["target_id"] == "pg67_independent_rule_ir_oracle_noise")
    assert pg67["training_eligible"] is False
    assert pg67["known_family_recall"] >= 0.8
    assert pg67["unknown_misname_count"] == 0
    assert pg67["negative_false_accept_count"] == 0
    assert pg67["promotion_status"] == "independent_rule_ir_oracle_noise_gate_passed_fixture_only_no_promotion"
    pg68 = next(item for item in report["targets"] if item["target_id"] == "pg68_real_local_docker_typed_oracle_adapter")
    assert pg68["training_eligible"] is False
    assert pg68["typed_positive_count"] == 7
    assert pg68["matched_negative_control_count"] == 7
    assert pg68["fresh_reset_per_action"] is False
    assert pg68["family_holdout_candidate_count"] == 0
    assert pg68["promotion_status"] == "blocked_fresh_reset_scope_no_family_holdout"
    pg69 = next(item for item in report["targets"] if item["target_id"] == "pg69_per_action_reset_unseen_family")
    assert pg69["training_eligible"] is False
    assert pg69["typed_positive_count"] == 12
    assert pg69["fresh_reset_per_action"] is True
    assert pg69["family_holdout_candidate_count"] == 1
    assert pg69["unknown_misname_count"] == 0
    assert pg69["promotion_status"] == "hard_gate_passed_evaluation_only_no_promotion"
    pg70 = next(item for item in report["targets"] if item["target_id"] == "pg70_trace_abstention_head")
    assert pg70["training_eligible"] is False
    assert pg70["unknown_misname_count"] == 0
    assert pg70["dev_confirm_recall"] == 0.0
    assert pg70["capability_gate_status"] == "blocked"
    assert pg70["promotion_status"] == "candidate_checkpoint_evaluation_only_capability_blocked"
    pg71 = next(item for item in report["targets"] if item["target_id"] == "pg71_trace_feature_drift_and_head_v2")
    assert pg71["training_eligible"] is False
    assert pg71["legacy_duplicate_label_conflict_count"] == 3
    assert pg71["pair_observable_shape_delta_count"] == 4
    assert pg71["unknown_misname_count"] == 0
    assert pg71["unknown_strict_abstain"] is True
    assert pg71["dev_confirm_recall"] == 0.0
    assert pg71["capability_gate_status"] == "blocked"
    assert pg71["promotion_status"] == "candidate_v2_evaluation_only_known_holdout_recall_blocked"
    pg72 = next(item for item in report["targets"] if item["target_id"] == "pg72_independent_seed_fresh_docker_matrix")
    assert pg72["training_eligible"] is False
    assert pg72["independent_seed_count"] == 3
    assert pg72["typed_positive_count"] == 21
    assert pg72["fresh_reset_per_action"] is True
    assert pg72["frozen_head_confirm_recall"] == 0.0
    assert pg72["frozen_head_false_accept_count"] == 0
    assert pg72["capability_gate_status"] == "blocked"
    assert pg72["promotion_status"] == "collection_gate_passed_frozen_head_all_abstain_capability_blocked"
    pg73 = next(item for item in report["targets"] if item["target_id"] == "pg73_causal_triplet_coverage_audit")
    assert pg73["training_eligible"] is False
    assert pg73["known_step_count"] == 25
    assert pg73["typed_negative_probe_count"] == 0
    assert pg73["all_current_reject_rows_are_synthetic_zero"] is True
    assert pg73["capability_gate_status"] == "blocked"
    assert pg73["promotion_status"] == "blocked_missing_causal_triplet_negative_probe"
    pg74 = next(item for item in report["targets"] if item["target_id"] == "pg74_causal_triplet_collector")
    assert pg74["training_eligible"] is False
    assert pg74["independent_seed_count"] == 3
    assert pg74["triplet_case_count"] == 21
    assert pg74["typed_positive_count"] == 21
    assert pg74["typed_negative_oracle_count"] == 42
    assert pg74["collection_gate_status"] == "passed"
    assert pg74["promotion_status"] == "triplet_collection_passed_evaluation_only_no_model_promotion"
    pg75 = next(item for item in report["targets"] if item["target_id"] == "pg75_triplet_context_delta_ablation")
    assert pg75["training_eligible"] is False
    assert pg75["known_dev_confirm_recall"] == 1.0
    assert pg75["known_dev_false_accept_count"] == 0
    assert pg75["legacy_unknown_misname_count"] == 0
    assert pg75["legacy_unknown_strict_abstain"] is True
    assert pg75["independent_triplet_unknown_holdout"] is True
    assert pg75["independent_triplet_unknown_misname_count"] == 0
    assert pg75["capability_gate_status"] == "passed"
    assert pg75["promotion_status"] == "candidate_capability_gate_passed_evaluation_only_no_promotion"
    pg76 = next(item for item in report["targets"] if item["target_id"] == "pg76_independent_unknown_triplet")
    assert pg76["triplet_case_count"] == 12
    assert pg76["typed_positive_count"] == 12
    assert pg76["typed_negative_oracle_count"] == 24
    assert pg76["model_unknown_misname_count"] == 0
    assert pg76["model_unknown_strict_abstain"] is True
    assert pg76["capability_gate_status"] == "passed"
    pg77 = next(item for item in report["targets"] if item["target_id"] == "pg77_real_triplet_transformer")
    assert pg77["train_example_count"] == 28
    assert pg77["dev_example_count"] == 14
    assert pg77["unknown_misname_count"] == 0
    assert pg77["external_known_confirm_recall"] == 0.428571
    assert pg77["capability_gate_status"] == "blocked"
    assert pg77["training_eligible"] is False
    pg78 = next(item for item in report["targets"] if item["target_id"] == "pg78_multisource_triplet_holdout")
    assert pg78["case_count"] == 270
    assert pg78["source_count"] == 5
    assert pg78["family_count"] == 9
    assert pg78["known_confirm_recall"] == 0.0
    assert pg78["screen_missing_count"] == 162
    assert pg78["capability_gate_status"] == "blocked"
    pg82 = next(item for item in report["targets"] if item["target_id"] == "pg82_canonical_triplet_collector")
    assert pg82["training_eligible"] is False
    assert pg82["negative_probe_positive_requested_count"] == 0
    assert pg82["capability_gate_status"] == "passed"
    pg83 = next(item for item in report["targets"] if item["target_id"] == "pg83_cross_seed_geometry_holdout_transformer")
    assert pg83["training_eligible"] is False
    assert pg83["dev_seeds"] == [7911]
    assert pg83["capability_gate_status"] == "passed"
    pg84 = next(item for item in report["targets"] if item["target_id"] == "pg84_cross_dataset_frozen_replay")
    assert pg84["training_eligible"] is False
    assert pg84["capability_gate_status"] == "blocked"
    pg85 = next(item for item in report["targets"] if item["target_id"] == "pg85_multisurface_composite_transformer")
    assert pg85["training_eligible"] is False
    assert pg85["capability_gate_status"] == "blocked"
    pg86 = next(item for item in report["targets"] if item["target_id"] == "pg86_surface_signal_composite_transformer")
    assert pg86["training_eligible"] is False
    assert pg86["cross_dataset_holdout_confirm_recall"] >= 0.8
    assert pg86["capability_gate_status"] == "passed"
    pg87 = next(item for item in report["targets"] if item["target_id"] == "pg87_codex_promotion_review")
    assert pg87["training_eligible"] is False
    assert pg87["controlled_offline_training_scale_allowed"] is True
    assert pg87["long_term_memory_promotion_allowed"] is False
    assert pg87["production_detector_claim_allowed"] is False
    pg88 = next(item for item in report["targets"] if item["target_id"] == "pg88_independent_html_dom_matrix")
    assert pg88["training_eligible"] is False
    assert pg88["triplet_case_count"] == 28
    assert pg88["capability_gate_status"] == "passed"
    pg89 = next(item for item in report["targets"] if item["target_id"] == "pg89_pg86_frozen_html_dom_replay")
    assert pg89["training_eligible"] is False
    assert pg89["confirm_recall"] >= 0.8
    assert pg89["false_accept_count"] == 0
    assert pg89["capability_gate_status"] == "passed"
    pg90 = next(item for item in report["targets"] if item["target_id"] == "pg90_cross_seed_codex_review")
    assert pg90["training_eligible"] is False
    assert pg90["formal_model_promotion_allowed"] is False
    assert pg90["long_term_memory_promotion_allowed"] is False
    assert pg90["capability_gate_status"] == "passed_for_controlled_offline_scale"
    pg91c = next(item for item in report["targets"] if item["target_id"] == "pg91_pg35_independent_collector")
    assert pg91c["training_eligible"] is False
    assert pg91c["target_instance_count"] == 648
    assert pg91c["capability_gate_status"] == "passed"
    pg91r = next(item for item in report["targets"] if item["target_id"] == "pg91_pg86_frozen_pg35_replay")
    assert pg91r["training_eligible"] is False
    assert pg91r["confirm_recall"] == 1.0
    assert pg91r["false_accept_count"] == 0
    assert pg91r["capability_gate_status"] == "passed"
    pg92 = next(item for item in report["targets"] if item["target_id"] == "pg92_blind_pg34_frozen_replay")
    assert pg92["training_eligible"] is False
    assert pg92["confirm_recall"] == 0.0
    assert pg92["false_accept_count"] == 0
    assert pg92["capability_gate_status"] == "blocked"
    pg96 = next(item for item in report["targets"] if item["target_id"] == "pg96_auto_goal_label_design")
    assert pg96["training_eligible"] is False
    assert pg96["proposal_input_oracle_visible"] is False
    assert pg96["proposal_input_family_visible"] is False
    assert pg96["seed_holdout_confirm_recall"] == 1.0
    assert pg96["seed_holdout_false_accept_count"] == 0
    assert pg96["layout_holdout_confirm_recall"] == 1.0
    assert pg96["layout_holdout_false_accept_count"] == 0
    assert pg96["unknown_family_strict_abstain"] is False
    assert pg96["capability_gate_status"] == "blocked"
    pg97 = next(item for item in report["targets"] if item["target_id"] == "pg97_neural_auto_goal_label_decoder")
    assert pg97["training_eligible"] is False
    assert pg97["self_supervised_visible_training"] is True
    assert pg97["proposal_input_oracle_visible"] is False
    assert pg97["proposal_input_family_visible"] is False
    assert pg97["seed_holdout_confirm_recall"] == 1.0
    assert pg97["layout_holdout_confirm_recall"] == 1.0
    assert pg97["tokenless_signal_ablation_confirm_recall"] == 0.0
    assert pg97["unknown_family_strict_abstain"] is False
    assert pg97["capability_gate_status"] == "blocked"
    pg98 = next(item for item in report["targets"] if item["target_id"] == "pg98_cross_implementation_neural_proposal")
    assert pg98["training_eligible"] is False
    assert pg98["cross_implementation_eval_source_excluded_from_training"] is True
    assert pg98["canonical_delta_projection_schema"] == "canonical-delta-projection-v1"
    assert pg98["cross_implementation_confirm_recall"] == 1.0
    assert pg98["false_accept_count"] == 0
    assert pg98["raw_vocab_failure_preserved"] is True
    assert pg98["raw_vocab_failure_recall"] == 0.666667
    assert pg98["unknown_family_strict_abstain"] is False
    assert pg98["capability_gate_status"] == "blocked"
    pg99 = next(item for item in report["targets"] if item["target_id"] == "pg99_surface_novelty_audit")
    assert pg99["training_eligible"] is False
    assert pg99["training_excludes_pg42"] is True
    assert pg99["positive_novel_surface_abstain_rate"] == 1.0
    assert pg99["all_rows_abstain"] is True
    assert pg99["known_unknown_overlap_rate"] == 1.0
    assert pg99["equivalence_class_conflict_count"] == 6
    assert pg99["impossibility_witness"] is True
    assert pg99["capability_gate_status"] == "blocked"
    pg100 = next(item for item in report["targets"] if item["target_id"] == "pg100_independent_semantic_sink_replay")
    assert pg100["training_eligible"] is False
    assert pg100["typed_positive_count"] == 7
    assert pg100["methods"] == {"GET": 6, "POST": 1}
    assert pg100["old_pg52_verdicts_discarded_before_revalidation"] is True
    assert pg100["model_input_excludes_evaluator_labels"] is True
    assert pg100["pg99_known_unknown_equivalence_witness_preserved"] is True
    assert pg100["unknown_family_strict_abstain"] is False
    assert pg100["capability_gate_status"] == "blocked"
    pg101 = next(item for item in report["targets"] if item["target_id"] == "pg101_active_probe_signature")
    assert pg101["training_eligible"] is False
    assert pg101["probe_bank_size"] == 9
    assert pg101["evaluation_episode_count"] == 522
    assert pg101["known_confirm_recall"] == 1.0
    assert pg101["false_accept_count"] == 0
    assert pg101["unknown_family_strict_abstain"] is True
    assert pg101["known_unknown_signature_overlap_count"] == 0
    assert pg101["pg99_static_overlap_count"] == 6
    assert pg101["capability_gate_status"] == "passed"
    assert pg101["third_implementation_confirm_recall"] == 1.0
    assert pg101["third_implementation_false_accept_count"] == 0
    assert pg101["order_permutation_invariant"] is True
    pg102 = next(item for item in report["targets"] if item["target_id"] == "pg102_neural_active_probe_decoder")
    assert pg102["training_eligible"] is False
    assert pg102["guarded_pg42_known_confirm_recall"] == 0.833333
    assert pg102["guarded_pg42_false_accept_count"] == 0
    assert pg102["guarded_pg42_unknown_family_strict_abstain"] is True
    assert pg102["guarded_pg35_known_confirm_recall"] == 0.875
    assert pg102["guarded_pg35_false_accept_count"] == 0
    assert pg102["raw_failure_visible"] is True
    assert pg102["raw_pg42_unknown_misname_count"] == 36
    assert pg102["raw_pg42_false_accept_count"] == 36
    assert pg102["order_permutation_invariant"] is True
    assert pg102["capability_gate_status"] == "passed_guarded_diagnostic"
    pg103 = next(item for item in report["targets"] if item["target_id"] == "pg103_auto_goal_label_active_probe")
    assert pg103["training_eligible"] is False
    assert pg103["guarded_known_confirm_recall"] == 1.0
    assert pg103["guarded_known_label_consistency"] == 1.0
    assert pg103["guarded_false_accept_count"] == 0
    assert pg103["guarded_pg42_unknown_family_strict_abstain"] is True
    assert pg103["guarded_pg76_unknown_family_strict_abstain"] is True
    assert pg103["guarded_repeat_goal_completion_rate"] == 1.0
    assert pg103["guarded_repeat_label_consistency_rate"] == 1.0
    assert pg103["raw_failure_visible"] is True
    assert pg103["raw_unknown_family_misname_count"] == 48
    assert pg103["generic_family_names_generated"] is False
    assert pg103["capability_gate_status"] == "passed_generic_goal_label_diagnostic"
    pg104 = next(item for item in report["targets"] if item["target_id"] == "pg104_probe_binding_ablation")
    assert pg104["training_eligible"] is False
    assert pg104["guarded_known_confirm_recall"] == 1.0
    assert pg104["guarded_false_accept_count"] == 0
    assert pg104["guarded_pg69_unknown_family_strict_abstain"] is False
    assert pg104["guarded_pg69_observable_unknown_strict_abstain"] is True
    assert pg104["pg69_unobservable_positive_count"] == 2
    assert pg104["surface_sign_ablation_agreement"] == 1.0
    assert pg104["binding_permutation_guarded_abstain_rate"] == 1.0
    assert pg104["compositional_rule_ir"]["copy_paste_order_invariant"] is True
    assert pg104["compositional_rule_ir"]["candidate_promotion_eligible_count"] == 0
    assert pg104["capability_gate_status"] == "blocked"
    pg105 = next(item for item in report["targets"] if item["target_id"] == "pg105_observable_projection")
    assert pg105["training_eligible"] is False
    assert pg105["guarded_known_confirm_recall"] == 1.0
    assert pg105["guarded_false_accept_count"] == 0
    assert pg105["guarded_pg69_unknown_family_strict_abstain"] is True
    assert pg105["pg69_opaque_positive_count"] == 2
    assert pg105["negative_anomaly_count"] == 0
    assert pg105["composition_candidate_promotion_eligible_count"] == 0
    assert pg105["capability_gate_status"] == "passed_observable_projection_diagnostic"
    pg106 = next(item for item in report["targets"] if item["target_id"] == "pg106_decoy_projection_holdout")
    assert pg106["training_eligible"] is False
    assert pg106["guarded_known_confirm_recall"] == 1.0
    assert pg106["guarded_false_accept_count"] == 0
    assert pg106["guarded_pg106_unknown_family_strict_abstain"] is True
    assert pg106["decoy_false_confirm_count"] == 0
    assert pg106["decoy_abstain_count"] == 4
    assert pg106["composition_candidate_promotion_eligible_count"] == 0
    assert pg106["capability_gate_status"] == "passed_cross_implementation_decoy_diagnostic"
    pg107 = next(item for item in report["targets"] if item["target_id"] == "pg107_multistep_generic_belief")
    assert pg107["training_eligible"] is False
    assert pg107["multi_step_episode_rate"] == 1.0
    assert pg107["typed_oracle_called_count"] == 0
    assert pg107["confirmed_positive_count"] == 0
    assert pg107["family_names_in_posterior"] is False
    assert pg107["decoy_never_confirm"] is True
    assert pg107["capability_gate_status"] == "passed_generic_multistep_belief_diagnostic"
    pg108 = next(item for item in report["targets"] if item["target_id"] == "pg108_belief_stress")
    assert pg108["training_eligible"] is False
    assert pg108["episode_count"] == 289
    assert pg108["scenario_count"] == 7
    assert pg108["order_invariant"] is True
    assert pg108["seed_invariant"] is True
    assert pg108["duplicate_posterior_unchanged_rate"] == 1.0
    assert pg108["conflicting_posterior_unchanged_rate"] == 1.0
    assert pg108["budget_one_fail_closed_rate"] == 1.0
    assert pg108["posterior_family_free"] is True
    assert pg108["capability_gate_status"] == "passed_belief_order_seed_stress"
    pg109 = next(item for item in report["targets"] if item["target_id"] == "pg109_fragment_composition")
    assert pg109["training_eligible"] is False
    assert pg109["episode_count"] == 289
    assert pg109["known_effect_expected_pair_count"] == 216
    assert pg109["known_effect_assembled_pair_count"] == 216
    assert pg109["known_effect_assembly_recall"] == 1.0
    assert pg109["unknown_or_decoy_abstain_rate"] == 1.0
    assert pg109["cross_sample_valid_rate"] == 1.0
    assert pg109["negative_case_abstain_rate"] == 1.0
    assert pg109["family_labels_in_fragments"] is False
    assert pg109["capability_gate_status"] == "passed_fragment_composition_diagnostic"
    pg110 = next(item for item in report["targets"] if item["target_id"] == "pg110_capacity_pressure_cycle")
    assert pg110["training_eligible"] is False
    assert pg110["scenario_count"] == 8
    assert pg110["growth_action_count"] == 2
    assert pg110["merge_ablate_action_count"] == 1
    assert pg110["ablation_pass"] is True
    assert pg110["ablation_rollback"] is True
    assert pg110["controller_is_policy_only"] is True
    assert pg110["architecture_transfer_mode"] == "bsp_v3_structure_contract_only"
    assert pg110["previous_checkpoint_reuse_forbidden"] is True
    assert pg110["fresh_checkpoint_required"] is True
    assert pg110["mandarin_foundation_training_isolated"] is True
    assert pg110["neural_weight_mutation_performed"] is False
    assert pg110["capability_gate_status"] == "passed_capacity_pressure_diagnostic"
    pg111 = next(item for item in report["targets"] if item["target_id"] == "pg111_bsp_v3_structure_adapter")
    assert pg111["training_eligible"] is False
    assert pg111["architecture_transfer_mode"] == "bsp_v3_structure_contract_only"
    assert pg111["previous_checkpoint_reuse_forbidden"] is True
    assert pg111["fresh_checkpoint_required"] is True
    assert pg111["dog_project_imported"] is False
    assert pg111["dog_weights_loaded"] is False
    assert pg111["mandarin_foundation_training_isolated"] is True
    assert pg111["python_reference_core"] is True
    assert pg111["python_core_split_forward_invariance"] is True
    assert pg111["python_core_merge_roundtrip_invariance"] is True
    assert pg111["python_core_page_mass_conservation"] is True
    assert pg111["python_core_fixed_budget_invariant"] is True
    assert pg111["old_checkpoint_reference_rejection"] is True
    assert pg111["tamper_commitment_rejection"] is True
    assert pg111["numerical_forward_parity_claimed"] is False
    assert pg111["capability_gate_status"] == "passed_bsp_v3_structure_adapter_smoke"
    pg112 = next(item for item in report["targets"] if item["target_id"] == "pg112_python_bsp_local_replay")
    assert pg112["training_eligible"] is False
    assert pg112["python_reference_core"] is True
    assert pg112["target_instance_count"] == 3
    assert pg112["surface_slot_count"] == 4
    assert pg112["episode_count"] == 12
    assert pg112["step_count"] == 48
    assert pg112["get_step_count"] == 24
    assert pg112["post_step_count"] == 24
    assert pg112["fresh_reset_per_step"] is True
    assert pg112["matched_negative_control_required"] is True
    assert pg112["evidence_sha256_required"] is True
    assert pg112["withheld_typed_oracle_must_abstain"] is True
    assert pg112["model_input_oracle_blind"] is True
    assert pg112["model_input_family_free"] is True
    assert pg112["bsp_parameter_unchanged"] is True
    assert pg112["bsp_mass_conserved"] is True
    assert pg112["cross_implementation_claim_allowed"] is False
    assert pg112["capability_gate_status"] == "passed_pg112_python_bsp_local_replay"
    pg113 = next(item for item in report["targets"] if item["target_id"] == "pg113_cross_implementation_replay")
    assert pg113["training_eligible"] is False
    assert pg113["implementation_count"] == 2
    assert pg113["reference_implementation"] == "app.main"
    assert pg113["independent_implementation"] == "app.pg113_independent_target"
    assert pg113["target_instance_count"] == 3
    assert pg113["surface_slot_count"] == 4
    assert pg113["episode_count"] == 12
    assert pg113["step_count"] == 48
    assert pg113["get_step_count"] == 24
    assert pg113["post_step_count"] == 24
    assert pg113["fresh_reset_per_action"] is True
    assert pg113["matched_negative_control_required"] is True
    assert pg113["target_evidence_sha256_required"] is True
    assert pg113["bridge_evidence_sha256_required"] is True
    assert pg113["withheld_typed_oracle_must_abstain"] is True
    assert pg113["model_input_oracle_blind"] is True
    assert pg113["model_input_family_free"] is True
    assert pg113["cross_implementation_replay_claim_allowed"] is True
    assert pg113["trained_model_capability_claim_allowed"] is False
    assert pg113["capability_gate_status"] == "passed_pg113_cross_implementation_replay"
    pg114 = next(item for item in report["targets"] if item["target_id"] == "pg114_family_holdout_decoy_replay")
    assert pg114["training_eligible"] is False
    assert pg114["implementation_count"] == 3
    assert pg114["heldout_semantic"] == "security_policy_transition"
    assert pg114["decoy_semantic"] == "shape_only_change"
    assert pg114["target_instance_count"] == 3
    assert pg114["surface_slot_count"] == 4
    assert pg114["episode_count"] == 12
    assert pg114["step_count"] == 48
    assert pg114["get_step_count"] == 24
    assert pg114["post_step_count"] == 24
    assert pg114["fresh_reset_per_action"] is True
    assert pg114["matched_negative_control_required"] is True
    assert pg114["target_evidence_sha256_required"] is True
    assert pg114["bridge_evidence_sha256_required"] is True
    assert pg114["family_holdout_confirm_recall"] == 1.0
    assert pg114["decoy_false_accept_count"] == 0
    assert pg114["withheld_oracle_abstain_rate"] == 1.0
    assert pg114["all_abstain_not_success"] is True
    assert pg114["model_input_oracle_blind"] is True
    assert pg114["model_input_family_free"] is True
    assert pg114["trained_model_capability_claim_allowed"] is False
    assert pg114["capability_gate_status"] == "passed_pg114_family_holdout_replay"
    pg115 = next(item for item in report["targets"] if item["target_id"] == "pg115_small_rule_ir_train_fixture")
    assert pg115["training_eligible"] is True
    assert pg115["sample_count"] == 800
    assert pg115["train_seed_count"] == 2
    assert pg115["dev_seed_count"] == 2
    assert pg115["training_promotion_allowed"] is False
    assert pg115["memory_promotion_allowed"] is False
    pg116 = next(item for item in report["targets"] if item["target_id"] == "pg116_multisource_trace_training")
    assert pg116["training_eligible"] is True
    assert pg116["source_count"] == 2
    assert pg116["target_instance_count"] == 12
    assert pg116["step_count"] == 192
    assert pg116["get_step_count"] == 96
    assert pg116["post_step_count"] == 96
    assert pg116["family_holdout_confirm_recall"] == 1.0
    assert pg116["decoy_false_accept_count"] == 0
    assert pg116["withheld_oracle_abstain_rate"] == 1.0
    assert pg116["training_artifact_promotion_allowed"] is False
    assert pg116["memory_promotion_allowed"] is False
    pg117 = next(item for item in report["targets"] if item["target_id"] == "pg117_double_implementation_encoding_holdout")
    assert pg117["training_eligible"] is False
    assert pg117["encoding_chain"] == ["url_percent", "html_entity"]
    assert pg117["implementation_holdout_from_pg116"] is True
    assert pg117["encoding_holdout_from_pg116"] is True
    assert pg117["frozen_route_positive_recall"] == 0.0
    assert pg117["frozen_decoy_false_accept_count"] == 0
    assert pg117["frozen_blind_oracle_abstain_rate"] == 1.0
    assert pg117["capability_gate_status"] == "blocked_double_encoding_positive_recall"
    assert pg117["memory_promotion_allowed"] is False
    pg118 = next(item for item in report["targets"] if item["target_id"] == "pg118_transition_delta_slot_training")
    assert pg118["training_eligible"] is True
    assert pg118["encoding_chain"] == ["html_entity", "url_percent"]
    assert pg118["target_instance_count"] == 6
    assert pg118["step_count"] == 96
    assert pg118["get_step_count"] == 48
    assert pg118["post_step_count"] == 48
    assert pg118["pg114_confirm_recall"] == 1.0
    assert pg118["pg117_route_positive_recall"] == 1.0
    assert pg118["pg117_decoy_false_accept_count"] == 0
    assert pg118["pg117_unknown_abstain_rate"] == 1.0
    assert pg118["capability_gate_status"] == "passed_pg118_transition_slot_double_encoding_holdout"
    assert pg118["training_artifact_promotion_allowed"] is False
    assert pg118["memory_promotion_allowed"] is False
    pg119 = next(item for item in report["targets"] if item["target_id"] == "pg119_metadata_transition_slot_training")
    assert pg119["training_eligible"] is True
    assert pg119["encoding_chain"] == ["unicode_escape", "html_entity", "url_percent"]
    assert pg119["target_instance_count"] == 6
    assert pg119["holdout_target_instance_count"] == 3
    assert pg119["step_count"] == 96
    assert pg119["get_step_count"] == 48
    assert pg119["post_step_count"] == 48
    assert pg119["pg119_positive_recall"] == 1.0
    assert pg119["pg119_decoy_false_accept_count"] == 0
    assert pg119["pg119_unknown_abstain_rate"] == 1.0
    assert pg119["pg117_cross_seed_positive_recall_variance"] == 0.0
    assert pg119["pg119_cross_seed_positive_recall_variance"] == 0.0
    assert pg119["pg119_slot_ablation_positive_recall"] == 0.0
    assert pg119["capability_gate_status"] == "passed_pg119_metadata_slot_seed_holdout_and_ablation"
    assert pg119["training_artifact_promotion_allowed"] is False
    assert pg119["memory_promotion_allowed"] is False
    pg120 = next(item for item in report["targets"] if item["target_id"] == "pg120_cross_implementation_metadata_holdout")
    assert pg120["training_eligible"] is False
    assert pg120["target_instance_count"] == 9
    assert pg120["step_count"] == 144
    assert pg120["positive_recall"] == 1.0
    assert pg120["decoy_false_accept_count"] == 0
    assert pg120["unknown_abstain_rate"] == 0.0
    assert pg120["capability_gate_status"] == "blocked_unknown_abstain_cross_implementation"
    assert pg120["memory_promotion_allowed"] is False
    pg121 = next(item for item in report["targets"] if item["target_id"] == "pg121_shape_sanitized_rule_ir_training")
    assert pg121["training_eligible"] is True
    assert pg121["shape_hash_slots_zeroed"] is True
    assert pg121["capacity_unchanged"] is True
    assert pg121["pg120_positive_recall"] == 1.0
    assert pg121["pg120_decoy_false_accept_count"] == 0
    assert pg121["pg120_unknown_abstain_rate"] == 1.0
    assert pg121["capability_gate_status"] == "passed_pg121_shape_sanitized_cross_impl_holdout"
    assert pg121["training_artifact_promotion_allowed"] is False
    assert pg121["memory_promotion_allowed"] is False
    pg122 = next(item for item in report["targets"] if item["target_id"] == "pg122_failure_guided_authorization_holdout")
    assert pg122["training_eligible"] is False
    assert pg122["target_instance_count"] == 9
    assert pg122["step_count"] == 144
    assert pg122["positive_recall"] == 0.0
    assert pg122["decoy_false_accept_count"] == 0
    assert pg122["unknown_abstain_rate"] == 0.444444
    assert pg122["failure_signatures"] is True
    assert pg122["capability_gate_status"] == "blocked_authorization_representation_gap"
    assert pg122["memory_promotion_allowed"] is False
    pg124 = next(item for item in report["targets"] if item["target_id"] == "pg124_failure_conditioned_policy")
    assert pg124["training_eligible"] is True
    assert pg124["step_count"] == 144
    assert pg124["full_policy_holdout_accuracy"] == 1.0
    assert pg124["full_model_failure_slots_zeroed_accuracy"] == 0.527778
    assert pg124["fresh_no_failure_baseline_accuracy"] == 0.854167
    assert pg124["failure_slot_behavior_changed"] is True
    assert pg124["failure_authority_fields_masked"] is True
    assert pg124["memory_promotion_allowed"] is False
    pg125 = next(item for item in report["targets"] if item["target_id"] == "pg125_scope_logic_failure_policy_ood")
    assert pg125["training_eligible"] is False
    assert pg125["step_count"] == 144
    assert pg125["full_policy_accuracy"] == 0.9375
    assert pg125["scope_surface_accuracy"] == 0.75
    assert pg125["capability_gate_status"] == "blocked_scope_surface_generalization"
    assert pg125["memory_promotion_allowed"] is False
    pg126 = next(item for item in report["targets"] if item["target_id"] == "pg126_failure_only_policy")
    assert pg126["training_eligible"] is True
    assert pg126["step_count"] == 144
    assert pg126["feature_dim"] == 17
    assert pg126["pg125_full_accuracy"] == 1.0
    assert pg126["pg125_scope_accuracy"] == 1.0
    assert pg126["pg125_zeroed_accuracy"] == 0.1875
    assert pg126["capability_gate_status"] == "passed_pg126_cross_family_failure_only_policy"
    assert pg126["memory_promotion_allowed"] is False
    pg127 = next(item for item in report["targets"] if item["target_id"] == "pg127_key_feature_assembly_policy")
    assert pg127["training_eligible"] is True
    assert pg127["full_holdout_accuracy"] == 1.0
    assert pg127["budget_accuracy"] == 1.0
    assert pg127["uniform_weight_ablation_accuracy"] == 0.583333
    assert pg127["capability_gate_status"] == "passed_pg127_failure_assembly_weight_and_budget_holdout"
    assert pg127["memory_promotion_allowed"] is False
    pg128 = next(item for item in report["targets"] if item["target_id"] == "pg128_trajectory_token_policy")
    assert pg128["training_eligible"] is False
    assert pg128["token_weight_ablation_changed_top1"] is False
    assert pg128["capability_gate_status"] == "blocked_trajectory_token_weight_not_necessary_on_current_task"
    pg129 = next(item for item in report["targets"] if item["target_id"] == "pg129_atomic_token_policy")
    assert pg129["training_eligible"] is False
    assert pg129["atomic_token_feature_dim"] == 16
    assert pg129["token_weight_ablation_changed_top1"] is False
    assert pg129["capability_gate_status"] == "blocked_atomic_token_weight_not_necessary_on_current_task"
    pg130 = next(item for item in report["targets"] if item["target_id"] == "pg130_layered_token_ir_smoke")
    assert pg130["training_eligible"] is False
    assert pg130["get_step_count"] == 36
    assert pg130["post_step_count"] == 36
    assert pg130["raw_source_saved"] is False
    assert pg130["raw_javascript_saved"] is False
    assert pg130["capability_gate_status"] == "passed_layered_token_ir_engineering_contract_only"
    pg131 = next(item for item in report["targets"] if item["target_id"] == "pg131_layered_ir_policy")
    assert pg131["training_eligible"] is False
    assert pg131["feature_dim"] == 53
    assert pg131["pg127_full_accuracy"] == 1.0
    assert pg131["pg125_family_accuracy"] == 1.0
    assert pg131["pg122_family_accuracy"] == 1.0
    assert pg131["uniform_ir_ablation_changed_top1"] is False
    assert pg131["capability_gate_status"] == "blocked_layered_ir_uniform_ablation_not_necessary_on_current_task"
    pg132 = next(item for item in report["targets"] if item["target_id"] == "pg132_open_source_ir_policy")
    assert pg132["training_eligible"] is False
    assert pg132["tokenizer_backend"] == "huggingface-tokenizers-wordlevel"
    assert pg132["pretrained_claim"] is False
    assert pg132["pg127_full_accuracy"] == 1.0
    assert pg132["pg125_family_accuracy"] == 1.0
    assert pg132["pg122_family_accuracy"] == 1.0
    assert pg132["uniform_token_ablation_changed_top1"] is False
    assert pg132["failure_scalar_ablation_changed_top1"] is False
    assert pg132["token_embedding_ablation_changed_top1"] is True
    assert pg132["history_order_counterfactual_changed_top1"] is False
    assert pg132["capability_gate_status"] == "blocked_failure_scalar_and_history_order_not_necessary_on_current_task"
    pg133 = next(item for item in report["targets"] if item["target_id"] == "pg133_history_latch_layered_token_policy")
    assert pg133["training_eligible"] is False
    assert pg133["pg133_full_accuracy"] == 1.0
    assert pg133["pg133_current_only_accuracy"] == 0.625
    assert pg133["pg133_counterfactual_prediction_separation"] == 1.0
    assert pg133["pg133_blind_final_abstain"] == 1.0
    assert pg133["token_ids_zeroed_accuracy"] == 0.25
    assert pg133["memory_promotion_allowed"] is False
    pg134 = next(item for item in report["targets"] if item["target_id"] == "pg134_independent_token_hash_gru")
    assert pg134["training_eligible"] is False
    assert pg134["independent_architecture"] is True
    assert pg134["pg133_full_accuracy"] == 1.0
    assert pg134["pg133_counterfactual_prediction_separation"] == 1.0
    assert pg134["all_unknown_steps_abstain"] is True
    assert pg134["safety_compliance_floor"] is True
    assert pg134["memory_promotion_allowed"] is False
    pg135 = next(item for item in report["targets"] if item["target_id"] == "pg135_balanced_policy")
    assert pg135["training_eligible"] is False
    assert pg135["exact_get_post_balance"] is True
    assert pg135["aggregate_holdout_get_count"] == 132
    assert pg135["aggregate_holdout_post_count"] == 132
    assert pg135["pg135_counterfactual_prediction_separation"] == 1.0
    assert pg135["pg122_family_raw_safety"] == 0.9375
    assert pg135["pg122_family_guarded_safety"] == 1.0
    assert pg135["pg122_guard_override_count"] == 3
    assert pg135["all_unknown_steps_abstain"] is True
    assert pg135["memory_promotion_allowed"] is False
    pg136 = next(item for item in report["targets"] if item["target_id"] == "pg136_causal_token_lm")
    assert pg136["training_eligible"] is False
    assert pg136["causal_lm_better_than_uniform"] is True
    assert pg136["action_labels_in_pretrain_sequences"] is False
    assert pg136["causal_order_reverse_accuracy"] == 0.5625
    assert pg136["token_identity_erased_accuracy"] == 0.5625
    assert pg136["pg122_family_raw_safety"] == 1.0
    assert pg136["unknown_all_steps_abstain"] is True
    assert pg136["memory_promotion_allowed"] is False
    pg146 = next(item for item in report["targets"] if item["target_id"] == "pg146_public_lab_replay")
    assert pg146["row_count"] == 6
    assert pg146["get_count"] == 3
    assert pg146["post_count"] == 3
    assert pg146["typed_oracle_count"] == 0
    assert pg146["training_eligible"] is False
    pg147 = next(item for item in report["targets"] if item["target_id"] == "pg147_model_capacity_sweep")
    assert pg147["model_variant_count"] == 4
    assert pg147["largest_parameter_count"] == 57064106
    assert pg147["training_completed"] is True
    assert pg147["training_eligible"] is False
    pg148 = next(item for item in report["targets"] if item["target_id"] == "pg148_large_model_posttraining")
    assert pg148["model_variant_count"] == 5
    assert pg148["largest_parameter_count"] == 57071025
    assert pg148["training_completed"] is True
    assert pg148["joint_xl_catastrophic_forgetting"] is True
    assert pg148["training_eligible"] is False
    pg149 = next(item for item in report["targets"] if item["target_id"] == "pg149_causal_action_alignment")
    assert pg149["generated_count"] == 8000
    assert pg149["model_variant_count"] == 3
    assert pg149["best_synthetic_holdout_accuracy"] == 0.865
    assert pg149["best_real_pg136_holdout_accuracy"] == 0.20454545
    assert pg149["training_eligible"] is False
    pg150 = next(item for item in report["targets"] if item["target_id"] == "pg150_real_synthetic_mix")
    assert pg150["model_variant_count"] == 3
    assert pg150["real_25_percent_real_holdout_accuracy"] == 0.86363636
    assert pg150["synthetic_only_real_holdout_accuracy"] == 0.22727273
    assert pg150["training_completed"] is True
    assert pg150["training_eligible"] is False
    pg137 = next(item for item in report["targets"] if item["target_id"] == "pg137_transfer_strategies")
    assert pg137["training_eligible"] is False
    assert pg137["selected_strategy"] == "scratch"
    assert pg137["action_gain_two_ood_sets"] is False
    assert pg137["cross_seed_stable"] is False
    assert pg137["scratch_seed_pg135_safety_min"] == 0.90625
    assert pg137["unknown_abstain_seed_min"] == 0.0
    assert pg137["memory_promotion_allowed"] is False
    pg138 = next(item for item in report["targets"] if item["target_id"] == "pg138_decoupled_loio")
    assert pg138["training_eligible"] is False
    assert pg138["loio_safety_floor"] is False
    assert pg138["loio_unknown_abstain"] is True
    assert pg138["contract_mask_is_deterministic_not_model"] is True
    assert pg138["holdout_pg127_contract_override_rate"] == 0.25
    assert pg138["action_gain_in_both_loio_folds"] is False
    assert pg138["memory_promotion_allowed"] is False
    pg139 = next(item for item in report["targets"] if item["target_id"] == "pg139_value_head_loio")
    assert pg139["training_eligible"] is False
    assert pg139["value_head_raw_safety_floor"] is False
    assert pg139["parser_ood_safety_floor"] is False
    assert pg139["deterministic_guard_is_not_value_head"] is True
    assert pg139["information_completeness_audit"]["hard_gate_passed"] is False
    assert pg139["information_completeness_audit"]["capability_training_allowed"] is False
    assert pg139["information_completeness_audit"]["representation_pretrain_allowed"] is True
    assert pg139["memory_promotion_allowed"] is False
    pg140 = next(item for item in report["targets"] if item["target_id"] == "pg140_information_complete_catalog")
    assert pg140["training_eligible"] is False
    assert pg140["catalog_get_count"] == 420
    assert pg140["catalog_post_count"] == 420
    assert pg140["explicit_unknown_projection"] is True
    assert pg140["row_provenance_manifest"] is True
    assert pg140["row_level_evidence_index"] is True
    assert pg140["memory_promotion_allowed"] is False
    pg141 = next(item for item in report["targets"] if item["target_id"] == "pg141_complete_candidate_training")
    assert pg141["training_eligible"] is False
    assert pg141["action_gain_both_loio_folds"] is False
    assert pg141["raw_safety_floor"] is False
    assert pg141["unknown_abstain_floor"] is False
    assert pg141["memory_promotion_allowed"] is False
    pg142 = next(item for item in report["targets"] if item["target_id"] == "pg142_safety_aware_candidate")
    assert pg142["training_eligible"] is False
    assert pg142["raw_safety_floor"] is False
    assert pg142["unknown_abstain_floor"] is True
    assert pg142["memory_promotion_allowed"] is False
    pg143 = next(item for item in report["targets"] if item["target_id"] == "pg143_oracle_availability_abstention")
    assert pg143["training_eligible"] is False
    assert pg143["availability_unknown_recall_floor"] is True
    assert pg143["availability_known_false_abstain_floor"] is True
    assert pg143["value_safety_floor"] is False
    assert pg143["parser_ood_value_safety_floor"] is False
    assert pg143["memory_promotion_allowed"] is False
    pg144 = next(item for item in report["targets"] if item["target_id"] == "pg144_surface_counterfactual_catalog")
    assert pg144["training_eligible"] is False
    assert pg144["representation_pretrain_allowed"] is False
    assert pg144["representation_diagnostic_only"] is True
    assert pg144["capability_train_allowed"] is False
    assert pg144["memory_promotion_allowed"] is False
    pg145 = next(item for item in report["targets"] if item["target_id"] == "pg145_local_multisurface_vulnerability_catalog")
    assert pg145["training_eligible"] is False
    assert pg145["target_instance_count"] == 150
    assert pg145["get_count"] == 300
    assert pg145["post_count"] == 300
    assert pg145["positive_count"] == 300
    assert pg145["matched_negative_count"] == 300
    assert pg145["unknown_oracle_count"] == 300
    assert pg145["waf_training_scope"] == "local_mock_only"
    assert pg145["external_bypass_payloads"] is False
    assert pg145["memory_promotion_allowed"] is False
    text = json.dumps(report, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text
