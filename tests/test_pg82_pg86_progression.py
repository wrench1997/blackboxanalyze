import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg82_collector_uses_real_benign_negative_and_geometry_v2():
    report = _read("pg82_canonical_triplet_collector_report_v1.json")
    trace = _read("pg82_canonical_triplet_collector_trace_v1.json")
    assert report["hard_gate"]["status"] == "passed"
    assert report["metrics"]["triplet_case_count"] == 270
    assert report["metrics"]["negative_probe_positive_requested_count"] == 0
    assert report["metrics"]["get_post_counts"] == {"GET": 135, "POST": 135}
    assert trace["steps"][0]["response_projection"]["projection_schema"] == "canonical_effect_projection_v2"
    assert "effect_surface" in trace["steps"][0]["response_projection"]
    assert "effect_geometry" in trace["steps"][0]["response_projection"]
    assert trace["raw_response_bodies_stored"] is False
    assert trace["long_term_memory_write"] is False


def test_pg82_and_pg83_capability_gates_pass_without_promotion():
    pg82 = _read("pg82_effect_geometry_source_holdout_transformer_report_v1.json")
    pg83 = _read("pg83_cross_seed_geometry_holdout_transformer_report_v1.json")
    for report in (pg82, pg83):
        assert report["source"]["device"] == "cuda"
        assert report["capability_gate"]["status"] == "passed"
        assert report["metrics"]["source_holdout"]["confirm_recall"] >= 0.80
        assert report["metrics"]["source_holdout"]["false_accept_count"] == 0
        assert report["metrics"]["unknown_family_holdout"]["strict_abstain"] is True
        assert report["promotion"]["training_allowed"] is False
        assert report["promotion"]["memory_promotion_allowed"] is False
    assert pg83["source"]["seed_holdout"]["dev"] == [7911]


def test_pg84_and_pg85_keep_cross_dataset_failure_visible():
    pg84 = _read("pg84_cross_dataset_frozen_replay_report_v1.json")
    pg85 = _read("pg85_multisurface_composite_transformer_report_v1.json")
    assert pg84["metrics"]["typed_negative_count"] == 21
    assert pg84["metrics"]["typed_neutral_count"] == 21
    assert pg84["metrics"]["confirm_recall"] == 0.0
    assert pg84["metrics"]["unknown_token_count"] == 1221
    assert pg84["hard_gate"]["status"] == "blocked"
    assert pg85["metrics"]["cross_dataset_holdout"]["confirm_recall"] == 0.285714
    assert pg85["capability_gate"]["status"] == "blocked"
    assert pg85["promotion"]["training_allowed"] is False


def test_pg86_surface_signal_composite_passes_all_current_capability_gates():
    report = _read("pg86_surface_signal_composite_transformer_report_v1.json")
    assert report["source"]["device"] == "cuda"
    assert report["capability_gate"]["status"] == "passed"
    assert report["metrics"]["dev_holdout"]["confirm_recall"] >= 0.80
    assert report["metrics"]["source_holdout"]["confirm_recall"] >= 0.80
    assert report["metrics"]["cross_dataset_holdout"]["confirm_recall"] >= 0.80
    assert report["metrics"]["dev_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["source_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["cross_dataset_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_family_holdout"]["strict_abstain"] is True
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg86_dataset_is_evaluation_only_and_raw_free():
    dataset = _read("pg86_surface_signal_composite_trace_dataset_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert dataset["raw_probe_strings_stored"] is False
    assert dataset["raw_response_bodies_stored"] is False
    assert dataset["long_term_memory_write"] is False


def test_pg87_codex_review_allows_only_controlled_offline_scale():
    review = _read("pg87_promotion_review_report_v1.json")
    assert review["status"] == "passed_for_controlled_offline_scale"
    assert review["regression"]["passed_count"] >= 416
    assert review["decision"]["controlled_offline_training_scale_allowed"] is True
    assert review["decision"]["long_term_memory_promotion_allowed"] is False
    assert review["decision"]["production_web_vulnerability_detector_claim_allowed"] is False
    assert review["checks"]["failed_cross_dataset_replay_preserved"] is True


def test_pg88_independent_html_dom_matrix_passes_collection_only():
    report = _read("pg88_independent_html_dom_matrix_report_v1.json")
    trace = _read("pg88_independent_html_dom_matrix_trace_v1.json")
    assert report["hard_gate"]["status"] == "passed"
    assert report["metrics"]["triplet_case_count"] == 28
    assert report["metrics"]["unique_target_instance_count"] == 28
    assert report["metrics"]["get_post_covered"] == {"GET": 24, "POST": 4}
    assert report["source"]["independent_seed_set"] == [88101, 88107, 88111, 88117]
    assert trace["accepted_episode_count"] == 4
    assert trace["raw_response_bodies_stored"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg89_frozen_pg86_replay_is_cross_seed_capable_without_promotion():
    report = _read("pg89_pg86_frozen_html_dom_replay_report_v1.json")
    assert report["capability_gate"]["status"] == "passed"
    assert report["metrics"]["confirm_recall"] >= 0.80
    assert report["metrics"]["seed_min_confirm_recall"] >= 0.75
    assert report["metrics"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_token_count"] == 0
    assert report["metrics"]["abstain_count"] > 0
    assert report["source"]["training"] is False
    assert report["source"]["memory_write"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg90_cross_seed_review_keeps_formal_promotion_blocked():
    review = _read("pg90_cross_seed_codex_review_report_v1.json")
    assert review["status"] == "passed_for_controlled_offline_scale"
    assert review["regression"]["passed_count"] >= 420
    assert review["decision"]["controlled_offline_training_scale_allowed"] is True
    assert review["decision"]["formal_model_promotion_allowed"] is False
    assert review["decision"]["long_term_memory_promotion_allowed"] is False
    assert review["decision"]["production_web_vulnerability_detector_claim_allowed"] is False
    assert review["checks"]["failed_cross_dataset_controls_preserved"] is True


def test_pg91_independent_pg35_collector_passes_only_after_reset_identity_fix():
    report = _read("pg91_pg35_independent_collector_report_v1.json")
    catalog = _read("pg91_pg35_independent_fixture_catalog_v1.json")
    assert report["hard_gate"]["status"] == "passed"
    assert report["metrics"]["sample_count"] == 648
    assert report["metrics"]["target_instance_count"] == 648
    assert report["metrics"]["typed_positive_count"] == 288
    assert report["metrics"]["typed_negative_count"] == 360
    assert report["metrics"]["methods"] == {"GET": 324, "POST": 324}
    assert report["metrics"]["encodings"] == {"identity": 324, "url_percent": 324}
    assert catalog["training_eligible"] is False
    assert catalog["raw_response_bodies_stored"] is False


def test_pg91_frozen_replay_passes_with_adapter_failure_preserved():
    report = _read("pg91_pg86_frozen_pg35_replay_report_v1.json")
    assert report["capability_gate"]["status"] == "passed"
    assert report["metrics"]["confirm_recall"] == 1.0
    assert report["metrics"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_token_count"] == 0
    assert report["metrics"]["family_min_confirm_recall"] == 1.0
    assert report["ablation"]["raw_pg84_recursive_adapter"]["confirm_recall"] == 0.0
    assert report["ablation"]["raw_pg84_recursive_adapter"]["unknown_token_count"] == 2880
    assert report["ablation"]["minimal_transport_shape_adapter"]["confirm_recall"] == 0.0
    assert report["source"]["post_hoc_schema_alignment"] is True
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg92_blind_pg34_preserves_same_shape_boundary_failure():
    report = _read("pg92_blind_pg34_frozen_replay_report_v1.json")
    assert report["capability_gate"]["status"] == "blocked"
    assert report["metrics"]["confirm_recall"] == 0.0
    assert report["metrics"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_token_count"] == 0
    assert report["capability_gate"]["checks"]["fresh_source_targets"] is True
    assert report["capability_gate"]["checks"]["not_all_abstain"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg93_surface_projection_repair_passes_without_promotion():
    report = _read("pg93_effect_surface_pg34_replay_report_v1.json")
    trace = _read("pg93_effect_surface_pg34_replay_trace_v1.json")
    assert report["capability_gate"]["status"] == "passed"
    assert report["metrics"]["confirm_recall"] == 1.0
    assert report["metrics"]["seed_min_confirm_recall"] == 1.0
    assert report["metrics"]["family_min_confirm_recall"] == 1.0
    assert report["metrics"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_token_count"] == 0
    assert report["source"]["semantic_surface_signal"] == "bounded_true_numeric_type_counts_and_key_hash_buckets"
    assert trace["evaluation_only"] is True
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg94_third_implementation_collection_is_complete_but_evaluation_only():
    catalog = _read("pg94_pg36_surface_catalog_v1.json")
    trace = _read("pg94_pg36_surface_trace_v1.json")
    assert catalog["sample_count"] == 960
    assert catalog["typed_positive_count"] == 96
    assert catalog["negative_control_count"] == 864
    assert catalog["target_instance_count"] == 960
    assert catalog["source_count"] == 2
    assert catalog["projection_schema"] == "canonical_effect_projection_v3_surface_signal"
    assert catalog["projection_repair_post_hoc"] is False
    assert all("effect_surface" in step["response_projection"] and "effect_geometry" in step["response_projection"] for step in trace["steps"])
    assert trace["evaluation_only"] is True
    assert catalog["training_eligible"] is False


def test_pg94_strict_third_implementation_keeps_unknown_all_abstain_boundary():
    report = _read("pg94_pg86_frozen_pg36_replay_report_v1.json")
    assert report["capability_gate"]["status"] == "blocked"
    assert report["metrics"]["confirm_recall"] == 0.0
    assert report["metrics"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_token_count"] == 720
    assert report["metrics"]["abstain_count"] == 192
    assert report["source"]["post_hoc_schema_alignment"] is False
    assert report["source"]["target_specific_shape_delta_fallback"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg95_candidate_and_held_out_replay_remain_quarantined():
    candidate = _read("pg95_invariant_surface_transformer_report_v1.json")
    replay = _read("pg95_pg94_frozen_replay_report_v1.json")
    assert candidate["capability_gate"]["status"] == "blocked"
    assert candidate["metrics"]["dev_holdout"]["confirm_recall"] == 0.945455
    assert candidate["metrics"]["cross_dataset_holdout"]["confirm_recall"] == 0.571429
    assert candidate["metrics"]["unknown_family_holdout"]["strict_abstain"] is True
    assert candidate["promotion"]["training_allowed"] is False
    assert replay["capability_gate"]["status"] == "passed"
    assert replay["metrics"]["confirm_recall"] == 1.0
    assert replay["metrics"]["seed_min_confirm_recall"] == 1.0
    assert replay["metrics"]["family_min_confirm_recall"] == 1.0
    assert replay["metrics"]["false_accept_count"] == 0
    assert replay["metrics"]["unknown_token_count"] == 0
    assert replay["source"]["phase_aligned_controls"] is True
    assert replay["promotion"]["training_allowed"] is False


def test_pg96_automatic_goal_label_proposal_is_oracle_blind_and_blocked_correctly():
    proposal = _read("pg96_auto_goal_label_proposal_v1.json")
    report = _read("pg96_auto_goal_label_report_v1.json")
    dataset = _read("pg96_auto_goal_label_visible_dataset_v1.json")
    assert proposal["schema_version"] == "auto-goal-label-proposal-v1"
    assert proposal["proposal_inputs"]["oracle_visible"] is False
    assert proposal["proposal_inputs"]["family_visible"] is False
    assert proposal["goal"]["budget"]["requires_fresh_reset"] is True
    assert len(proposal["labels"]) == 3
    assert report["status"] == "blocked"
    assert report["metrics"]["seed_holdout"]["confirm_recall"] == 1.0
    assert report["metrics"]["seed_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["seed_holdout"]["unknown_family_strict_abstain"] is False
    assert report["metrics"]["layout_holdout"]["confirm_recall"] == 1.0
    assert report["metrics"]["layout_holdout"]["false_accept_count"] == 0
    assert report["goal_metrics"]["seed_holdout"]["repeat_goal_positive_completion_rate"] == 1.0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert dataset["model_input_contract"]["oracle_is_label_not_feature"] is True
    assert dataset["long_term_memory_write"] is False


def test_pg97_neural_goal_label_decoder_keeps_signal_ablation_and_unknown_gate_visible():
    proposal = _read("pg97_neural_goal_label_proposal_v1.json")
    report = _read("pg97_neural_goal_label_report_v1.json")
    protocol = _read("pg97_neural_goal_label_protocol_v1.json")
    dataset = _read("pg97_neural_goal_label_visible_dataset_v1.json")
    assert proposal["schema_version"] == "neural-auto-goal-label-decoder-v1"
    assert proposal["proposal_inputs"]["oracle_visible"] is False
    assert proposal["proposal_inputs"]["family_visible"] is False
    assert proposal["model"]["architecture"] == "token_presence_autoencoder_plus_two_means_kmeans"
    assert report["status"] == "blocked"
    assert report["metrics"]["seed_holdout"]["confirm_recall"] == 1.0
    assert report["metrics"]["seed_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["layout_holdout"]["confirm_recall"] == 1.0
    assert report["metrics"]["tokenless_signal_ablation_seed_holdout"]["confirm_recall"] == 0.0
    assert report["metrics"]["tokenless_signal_ablation_seed_holdout"]["abstain_count"] == 160
    assert report["metrics"]["seed_holdout"]["unknown_family_strict_abstain"] is False
    assert report["capability_gate"]["checks"]["tokenless_signal_ablation_degrades"] is True
    assert protocol["model_contract"]["typed_oracle_before_decoder_forbidden"] is True
    assert protocol["ablation_contract"]["tokenless_signal_ablation_required"] is True
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert dataset["long_term_memory_write"] is False


def test_pg98_cross_implementation_replay_preserves_raw_vocab_failure_and_unknown_gate():
    report = _read("pg98_cross_implementation_neural_report_v1.json")
    protocol = _read("pg98_cross_implementation_neural_protocol_v1.json")
    dataset = _read("pg98_cross_implementation_visible_dataset_v1.json")
    failure = _read("pg98_cross_implementation_raw_vocab_failure_v1.json")
    assert report["status"] == "blocked"
    assert report["source"]["cross_implementation_eval_source_excluded_from_training"] is True
    assert report["source"]["canonical_delta_projection_schema"] == "canonical-delta-projection-v1"
    metrics = report["metrics"]["pg42_cross_implementation"]
    assert metrics["count"] == 1440
    assert metrics["typed_positive_count"] == 324
    assert metrics["typed_negative_count"] == 1116
    assert metrics["confirm_recall"] == 1.0
    assert metrics["known_family_confirm_recall"] == 1.0
    assert metrics["false_accept_count"] == 0
    assert metrics["implementation_confirm_recall"] == {"cobalt": 1.0, "quartz": 1.0}
    assert metrics["unknown_family_strict_abstain"] is False
    checks = report["capability_gate"]["checks"]
    assert checks["fresh_reset_per_pair"] is True
    assert checks["negative_control_matched"] is True
    assert checks["evidence_hashes_valid"] is True
    assert checks["cross_implementation_recall_min"] is True
    assert checks["unknown_family_strict_abstain"] is False
    assert protocol["canonical_delta_projection"]["field_names_discarded"] is True
    assert protocol["evaluation_contract"]["implementations"] == ["cobalt", "quartz"]
    assert dataset["training_excludes_pg42"] is True
    assert dataset["model_input_contract"]["oracle_is_label_not_feature"] is True
    assert failure["preserved_failure"] is True
    assert failure["cross_implementation_confirm_recall"] == 0.666667


def test_pg99_surface_novelty_audit_proves_bounded_projection_overlap():
    report = _read("pg99_surface_novelty_report_v1.json")
    protocol = _read("pg99_surface_novelty_protocol_v1.json")
    dataset = _read("pg99_surface_novelty_visible_dataset_v1.json")
    trace = _read("pg99_surface_novelty_trace_v1.json")
    assert report["status"] == "blocked"
    novelty = report["metrics"]["pg42_novelty"]
    overlap = report["metrics"]["pg42_known_unknown_overlap"]
    assert novelty["support_size"] == 24
    assert novelty["count"] == 1440
    assert novelty["all_rows_abstain"] is True
    assert novelty["positive_novel_surface_abstain_rate"] == 1.0
    assert overlap["unknown_overlap_rate"] == 1.0
    assert overlap["equivalence_class_conflict_count"] == 6
    assert overlap["impossibility_witness"] is True
    assert report["capability_gate"]["checks"]["fresh_reset_per_pair"] is True
    assert report["capability_gate"]["checks"]["negative_control_matched"] is True
    assert report["capability_gate"]["checks"]["evidence_hashes_valid"] is True
    assert protocol["discriminator"]["novel_action"] == "abstain"
    assert protocol["impossibility_check"]["requires_known_unknown_fingerprint_overlap_audit"] is True
    assert dataset["training_excludes_pg42"] is True
    assert dataset["model_input_contract"]["family_label_in_features"] is False
    assert dataset["long_term_memory_write"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
