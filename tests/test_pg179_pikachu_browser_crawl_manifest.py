import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_manifest() -> dict:
    return json.loads(
        (ROOT / "research" / "pg179_pikachu_browser_crawl_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_pg179_persists_real_get_and_post_request_shapes() -> None:
    manifest = _load_manifest()
    stats = manifest["stats"]
    assert stats["persisted_unique_route_count"] >= 60
    assert stats["get_query_surface_count"] > 0
    assert stats["get_form_surface_count"] > 0
    assert stats["post_form_surface_count"] > 0
    assert manifest["crawler"]["persisted_once"] is True
    assert manifest["crawler"]["re_crawl_trigger"]

    rows = manifest["request_response_rows"]
    assert rows
    assert all(row["request_schema"]["values"].startswith("not stored") for row in rows)
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["vulnerability_claim_allowed"] is False for row in rows)
    assert any(
        row["method"] == "GET"
        and row["source"] == "anchor"
        and row["request_schema"]["query_params"]
        for row in rows
    )
    assert any(
        row["method"] == "GET"
        and row["source"] == "form"
        and row["request_schema"]["query_params"]
        for row in rows
    )
    assert any(
        row["method"] == "POST" and row["request_schema"]["form_params"] for row in rows
    )


def test_pg179_response_projection_keeps_redirect_and_evidence_slots() -> None:
    manifest = _load_manifest()
    pages = manifest["page_summaries"]
    assert len(pages) == 63
    assert all("response_projection" in page for page in pages)
    observed = [page["response_projection"] for page in pages if page["response_projection"].get("status_chain")]
    assert len(observed) == 63
    assert all(projection["status_chain"] for projection in observed)
    assert all("redirect_hop_count" in projection for projection in observed)
    assert all("evidence_sha256" in projection for projection in observed)
    assert manifest["request_response_contract"]["redirect_recording"]
    assert manifest["request_response_contract"]["missing_field_policy"].startswith("missing is incomplete")


def test_pg179_rule_blocks_incomplete_rows_from_training_and_memory() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg179_browser_crawl_parameter_grounding_policy"]
    assert rule["browser_dom_crawl_required_before_catalog"] is True
    assert rule["unknown_form_submission_forbidden"] is True
    assert rule["missing_field_status"] == "incomplete"
    assert rule["training_and_memory_promotion_on_incomplete"] is False
    assert rule["parameterized_replay_required"] is True


def test_pg179b_requires_failure_guided_next_action_not_visual_similarity() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg179b_iterative_get_post_probe_learning_policy"]
    assert rule["adaptive_probe_required"] is True
    assert rule["failure_must_change_next_action_or_abstain_reason"] is True
    assert rule["candidate_signal_is_not_positive"] is True
    assert rule["static_similarity_is_not_vulnerability_evidence"] is True
    assert rule["positive_requires_typed_effect_and_replay"] is True
    assert rule["fresh_reset_per_episode"] is True
    assert rule["training_gate"]["require_failure_guided_action_gain"] is True
    assert rule["ai_request_loop"]["model_emits_action_manifest"] is True
    assert rule["ai_request_loop"]["runner_validates_manifest_before_send"] is True
    assert rule["ai_request_loop"]["runner_sends_only_allowlisted_safe_canary"] is True
    assert rule["ai_request_loop"]["response_projection_and_failure_token_return_to_model"] is True
    assert rule["ai_request_loop"]["invalid_or_unsafe_manifest_action"] == "abstain"
    assert rule["ai_request_loop"]["parameter_names_must_match_browser_crawl_manifest"] is True
    assert rule["ai_request_loop"]["unobserved_method_or_parameter_action"] == "incomplete"


def test_pg181_requires_model_manifest_validation_before_local_send() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg181_model_manifest_request_loop_policy"]
    assert rule["model_output_vocabulary"] == ["baseline", "matched_control", "safe_candidate", "abstain"]
    assert rule["model_must_not_emit_raw_payload"] is True
    assert rule["model_must_not_invent_method_or_parameter"] is True
    assert rule["browser_manifest_is_parameter_authority"] is True
    assert rule["manifest_validator_before_network_send"] is True
    assert rule["single_channel_unknown_oracle_action"] == "abstain"
    assert rule["typed_positive_required_before_vulnerability_label"] is True


def test_pg182_keeps_cross_app_unknown_oracle_fail_closed() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg182_cross_app_manifest_replay_policy"]
    assert rule["training_source_is_separate_from_target"] is True
    assert rule["parameter_authority"] == "allow-listed Juice Shop shadow manifest"
    assert rule["required_parameter"] == "q"
    assert rule["manifest_validator_before_send"] is True
    assert rule["family_specific_typed_oracle_required_for_positive"] is True
    assert rule["unknown_oracle_action"] == "abstain"
    assert rule["training_and_memory_promotion"] is False


def test_pg183_requires_frozen_checkpoint_for_independent_source_replay() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg183_independent_implementation_replay_policy"]
    assert rule["frozen_checkpoint_required"] is True
    assert rule["independent_source_required"] is True
    assert rule["fresh_server_per_surface"] is True
    assert rule["parameter_authority"] == "independent fixture observed message field"
    assert rule["typed_positive_required_before_vulnerability_label"] is True
    assert rule["unknown_oracle_action"] == "abstain"
    assert rule["weight_update_during_evaluation"] is False
    assert rule["memory_promotion_during_evaluation"] is False


def test_pg184_requires_ai_payload_manifest_gate_before_send() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg184_ai_payload_manifest_generation_gate"]
    assert rule["model_output"] == "abstract_probe_role_only"
    assert rule["raw_exploit_string_generation"] is False
    assert rule["send_gate"]["validate_before_network_send"] is True
    assert rule["send_gate"]["validation_failure_action"] == "abstain"
    assert rule["positive_gate"]["typed_family_oracle"] is True
    assert rule["positive_gate"]["matched_negative_control"] is True
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False


def test_pg185_separates_typed_dom_surface_from_vulnerability_positive() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg185_pikachu_read_only_dom_surface_replay"]
    assert rule["model_participates_in_request_loop"] is True
    assert rule["parameter_authority"] == "browser crawl manifest observed GET query fields"
    assert rule["probe_kind"] == "inert_dom_markup"
    assert rule["typed_dom_effect_is_not_xss_positive"] is True
    assert rule["raw_probe_and_response_persistence_forbidden"] is True
    assert rule["training_and_memory_promotion"] is False


def test_pg186_requires_capacity_seed_and_encoding_matrix_without_target_training() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg186_pikachu_dom_capacity_encoding_replay"]
    assert len(rule["frozen_checkpoints"]) == 3
    assert rule["encoding_holdout"] == ["identity", "html_entity", "html_entity_depth2"]
    assert rule["fresh_restart_per_episode"] is True
    assert rule["typed_dom_effect_is_not_vulnerability"] is True
    assert rule["training_on_target_trace"] is False


def test_pg187_requires_route_and_encoding_double_holdout() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg187_pikachu_cross_route_double_holdout"]
    assert rule["route_holdout"] == ["/vul/xss/xss_01.php", "/vul/xss/xss_04.php"]
    assert rule["encoding_holdout"] == ["identity", "html_entity_depth2"]
    assert rule["model_input_route_excluded"] is True
    assert rule["model_input_family_excluded"] is True
    assert rule["frozen_checkpoint_evaluation"] is True
    assert rule["typed_dom_effect_is_not_vulnerability"] is True


def test_pg188_large_model_candidate_requires_replay_and_abstain_gates() -> None:
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    rule = rules["pg188_xxl_replay_action_training"]
    assert rule["body_parameter_target"] == 101380329
    assert rule["abstract_action_rows"] == 696
    assert rule["lm_replay_rows"] == 4096
    assert rule["target_trace_training_forbidden"] is True
    assert rule["forgetting_gate"]["catastrophic_forgetting_blocks_selection"] is True
    assert rule["capability_gate"]["unknown_abstain_rate_min"] == 0.95
