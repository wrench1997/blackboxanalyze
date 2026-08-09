import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_pg113_cross_implementation_gate_passes_without_model_promotion():
    report = _load("pg113_cross_implementation_replay_report_v1.json")
    assert report["status"] == "passed_pg113_cross_implementation_replay"
    assert report["scope"]["implementation_count"] == 2
    assert report["scope"]["cross_implementation_replay_claim_allowed"] is True
    assert report["scope"]["trained_model_claim_allowed"] is False
    assert report["scope"]["external_network"] is False
    assert report["capability_gate"]["claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False

    assert report["metrics"] == {
        "target_instance_count": 3,
        "surface_count": 4,
        "episode_count": 12,
        "step_count": 48,
        "get_step_count": 24,
        "post_step_count": 24,
        "fresh_reset_count": 48,
        "evidence_hash_valid_count": 48,
        "typed_oracle_called_count": 36,
        "oracle_withheld_step_count": 12,
        "confirmed_positive_count": 9,
        "confirmed_negative_count": 24,
        "candidate_count": 12,
        "abstain_count": 3,
        "known_positive_pair_count": 9,
        "unknown_oracle_abstain_rate": 1.0,
        "bsp_parameter_unchanged_rate": 1.0,
    }
    comparison = report["cross_implementation"]
    assert comparison["reference_known_positive_pairs"] == comparison["independent_known_positive_pairs"] == 9
    assert comparison["reference_withheld_abstain_rate"] == comparison["independent_withheld_abstain_rate"] == 1.0


def test_pg113_visible_inputs_are_anonymous_and_trace_has_fresh_pairing():
    dataset = _load("pg113_cross_implementation_replay_visible_dataset_v1.json")
    trace = _load("pg113_cross_implementation_replay_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert dataset["model_input_family_free"] is True
    assert dataset["typed_oracle_labels_outside_model_input"] is True
    assert len(dataset["rows"]) == 48
    assert trace["execution_mode"] == "fresh_loopback_uvicorn_process"
    assert trace["training_eligible"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert len(trace["episodes"]) == 12
    assert len(trace["evidence_records"]) == 48

    for row in dataset["rows"]:
        model_input = row["model_input"]
        assert "oracle_projection" not in model_input
        assert "positive_authority" not in model_input
        assert "family" not in json.dumps(model_input, ensure_ascii=False).casefold()
        assert row["training_eligible"] is False

    for episode in trace["episodes"]:
        assert episode["episode_report"]["status"] == "accepted_evaluation"
        assert episode["negative_control_pair_clear"] is True
        assert [step["action_manifest"]["method"] for step in episode["steps"]] == ["GET", "GET", "POST", "POST"]
        epochs = [step["fresh_reset"]["reset_epoch"] for step in episode["steps"]]
        assert len(set(epochs)) == 4
        for step in episode["steps"]:
            assert step["fresh_reset"]["fresh_target"] is True
            assert step["fresh_reset"]["completed"] is True
            assert step["fresh_reset"]["evaluator_state_hidden"] is True
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
        if episode["oracle_available"]:
            assert episode["final_decision"] == "confirmed_positive"
            assert episode["candidate_pair_positive"] is True
        else:
            assert episode["final_decision"] == "abstain"
            assert all(step["decision"] != "confirmed_positive" for step in episode["steps"])

    for record in trace["evidence_records"]:
        declared = record["evidence_hash"]
        body = dict(record)
        body.pop("evidence_hash")
        assert declared == _sha256_json(body)
        assert record["target_evidence_sha256"] == record["oracle_projection"]["source_evidence_sha256"]
        assert record["safety"]["external_network"] is False
        assert record["safety"]["state_mutated"] is False

    text = json.dumps({"dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("<script", "union select", "sleep(", "javascript:", "raw_probe_value", "raw_response_body_value"):
        assert forbidden not in text


def test_pg113_source_hashes_match_independent_target_bridge_and_runner():
    report = _load("pg113_cross_implementation_replay_report_v1.json")
    for key, relative_path in {
        "target": "app/pg113_independent_target.py",
        "bridge": "app/pg113_cross_impl_replay.py",
        "bsp_core": "app/bsp_v3_research_core.py",
        "runner": "scripts/run_pg113_cross_implementation_replay.py",
        "reference_app": "app/main.py",
    }.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"]["source_hashes"][key]


def test_pg113_improvement_rule_keeps_cross_impl_replay_out_of_training_memory():
    rules = _load("improvement_rules.json")
    policy = rules["pg113_cross_implementation_replay_policy"]
    assert policy["implementation_count"] == 2
    assert policy["reference_implementation"] == "app.main local ASGI maze"
    assert policy["independent_implementation"] == "app.pg113_independent_target fresh uvicorn process"
    assert policy["transport"] == "fresh_loopback_uvicorn_process"
    assert policy["external_network"] is False
    assert policy["docker_claimed"] is False
    assert policy["target_instance_count"] == 3
    assert policy["surface_slot_count"] == 4
    assert policy["episode_count"] == 12
    assert policy["step_count"] == 48
    assert policy["get_step_count"] == 24
    assert policy["post_step_count"] == 24
    assert policy["fresh_reset_per_action"] is True
    assert policy["matched_negative_control_required"] is True
    assert policy["target_evidence_sha256_required"] is True
    assert policy["bridge_evidence_sha256_required"] is True
    assert policy["withheld_typed_oracle_must_abstain"] is True
    assert policy["cross_implementation_replay_claim_allowed"] is True
    assert policy["trained_model_capability_claim_allowed"] is False
    assert policy["training_eligible"] is False
    assert policy["memory_promotion_allowed"] is False
    assert policy["capability_gate_status"] == "passed_pg113_cross_implementation_replay"
