import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_pg112_report_passes_bounded_local_replay_gate_without_promotion():
    report = _load("pg112_python_bsp_local_replay_report_v1.json")
    assert report["status"] == "passed_pg112_python_bsp_local_replay"
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["scope"]["transport"] == "in_process_asgi_loopback"
    assert report["scope"]["cross_implementation_claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["safety"]["external_network"] is False
    assert report["safety"]["bsp_weights_updated"] is False

    metrics = report["metrics"]
    assert metrics == {
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
        "candidate_count": 9,
        "abstain_count": 6,
        "known_positive_pair_count": 9,
        "unknown_oracle_abstain_rate": 1.0,
        "multi_target_consistency_rate": 1.0,
        "bsp_parameter_unchanged_rate": 1.0,
    }


def test_pg112_visible_dataset_is_oracle_blind_and_trace_is_bounded():
    dataset = _load("pg112_python_bsp_local_replay_visible_dataset_v1.json")
    trace = _load("pg112_python_bsp_local_replay_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert dataset["model_input_family_free"] is True
    assert dataset["typed_oracle_labels_outside_model_input"] is True
    assert len(dataset["rows"]) == 48
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert len(trace["steps"]) == len(trace["evidence_records"]) == 48

    for row in dataset["rows"]:
        model_input = row["model_input"]
        assert "oracle_projection" not in model_input
        assert "positive_authority" not in model_input
        assert "family" not in json.dumps(model_input, ensure_ascii=False).casefold()
        assert row["training_eligible"] is False
        assert row["memory_promotion_allowed"] is False

    for record in trace["evidence_records"]:
        declared = record["evidence_hash"]
        without_hash = dict(record)
        without_hash.pop("evidence_hash")
        assert declared == _sha256_json(without_hash)
        assert record["safety"]["external_network"] is False
        assert record["safety"]["state_mutated"] is False

    text = json.dumps({"dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("<span data-sift-marker", "operator_like", "boundary_candidate", "raw_probe_value", "raw_response_body"):
        assert forbidden not in text


def test_pg112_steps_enforce_fresh_get_post_pairing_and_withheld_abstention():
    trace = _load("pg112_python_bsp_local_replay_trace_v1.json")
    episodes = trace["episodes"]
    assert len(episodes) == 12
    assert len({episode["target_instance_id"] for episode in episodes}) == 3
    assert len({episode["surface_slot"] for episode in episodes}) == 4

    for episode in episodes:
        assert episode["episode_report"]["status"] == "accepted_evaluation"
        assert episode["episode_report"]["training_candidate"] is False
        assert episode["episode_report"]["memory_promotion_allowed"] is False
        assert episode["negative_control_pair_clear"] is True
        methods = [step["action_manifest"]["method"] for step in episode["steps"]]
        assert methods == ["GET", "GET", "POST", "POST"]
        for step in episode["steps"]:
            assert step["fresh_reset"]["fresh_target"] is True
            assert step["fresh_reset"]["completed"] is True
            assert step["fresh_reset"]["evaluator_state_hidden"] is True
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
        candidate_steps = [
            step
            for step in episode["steps"]
            if step["oracle_projection"].get("candidate_signal") and step["decision"] != "confirmed_negative"
        ]
        for step in candidate_steps:
            assert step["oracle_projection"]["negative_control_pair_id"]

        if episode["oracle_available"]:
            assert episode["final_decision"] == "confirmed_positive"
            assert episode["candidate_pair_positive"] is True
        else:
            assert episode["final_decision"] == "abstain"
            assert all(step["decision"] != "confirmed_positive" for step in episode["steps"])


def test_pg112_source_hashes_match_current_python_implementation():
    report = _load("pg112_python_bsp_local_replay_report_v1.json")
    for key, relative_path in {
        "bridge": "app/pg112_replay_bridge.py",
        "bsp_core": "app/bsp_v3_research_core.py",
        "main": "app/main.py",
        "runner": "scripts/run_pg112_python_bsp_local_replay.py",
    }.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"]["source_hashes"][key]


def test_pg112_improvement_rule_keeps_replay_evaluation_only():
    rules = _load("improvement_rules.json")
    policy = rules["pg112_python_bsp_local_replay_policy"]
    assert policy["transport"] == "in_process_asgi_loopback"
    assert policy["external_network"] is False
    assert policy["target_instance_count"] == 3
    assert policy["surface_slot_count"] == 4
    assert policy["episode_count"] == 12
    assert policy["step_count"] == 48
    assert policy["get_step_count"] == 24
    assert policy["post_step_count"] == 24
    assert policy["fresh_reset_per_step"] is True
    assert policy["matched_negative_control_required"] is True
    assert policy["evidence_sha256_required"] is True
    assert policy["withheld_typed_oracle_must_abstain"] is True
    assert policy["model_input_oracle_blind"] is True
    assert policy["model_input_family_free"] is True
    assert policy["training_eligible"] is False
    assert policy["memory_promotion_allowed"] is False
    assert policy["cross_implementation_claim_allowed"] is False
    assert policy["capability_gate_status"] == "passed_pg112_python_bsp_local_replay"
