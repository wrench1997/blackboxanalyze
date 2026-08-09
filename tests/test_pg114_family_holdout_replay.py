import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_pg114_family_holdout_and_decoy_gate_passes_without_training():
    report = _load("pg114_family_holdout_replay_report_v1.json")
    assert report["status"] == "passed_pg114_family_holdout_replay"
    assert report["scope"]["implementation_count"] == 3
    assert report["scope"]["family_holdout_claim_allowed"] is True
    assert report["scope"]["trained_model_claim_allowed"] is False
    assert report["capability_gate"]["claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["metrics"]["family_holdout_confirm_recall"] == 1.0
    assert report["metrics"]["decoy_false_accept_count"] == 0
    assert report["metrics"]["withheld_oracle_abstain_rate"] == 1.0
    assert report["metrics"]["confirmed_positive_count"] == 3
    assert report["metrics"]["confirmed_negative_count"] == 33
    assert report["metrics"]["candidate_count"] == 9
    assert report["metrics"]["abstain_count"] == 3


def test_pg114_trace_has_real_pairing_and_no_raw_oracle_or_probe_values():
    dataset = _load("pg114_family_holdout_replay_visible_dataset_v1.json")
    trace = _load("pg114_family_holdout_replay_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert dataset["model_input_family_free"] is True
    assert dataset["typed_oracle_labels_outside_model_input"] is True
    assert len(dataset["rows"]) == 48
    assert len(trace["episodes"]) == 12
    assert len(trace["steps"]) == len(trace["evidence_records"]) == 48
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
        assert len({step["fresh_reset"]["reset_epoch"] for step in episode["steps"]}) == 4
        if episode["surface_kind"] == "policy":
            assert episode["final_decision"] == "confirmed_positive"
        elif episode["surface_kind"] in {"decoy", "neutral"}:
            assert episode["final_decision"] == "confirmed_negative"
        else:
            assert episode["final_decision"] == "abstain"
        for step in episode["steps"]:
            assert step["fresh_reset"]["fresh_target"] is True
            assert step["fresh_reset"]["completed"] is True
            assert step["fresh_reset"]["evaluator_state_hidden"] is True
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
    for record in trace["evidence_records"]:
        body = dict(record)
        declared = body.pop("evidence_hash")
        assert declared == _sha256_json(body)
        assert record["target_evidence_sha256"] == record["oracle_projection"]["source_evidence_sha256"]
    text = json.dumps({"dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("<script", "union select", "sleep(", "javascript:", "raw_probe_value", "raw_response_body_value"):
        assert forbidden not in text


def test_pg114_source_hashes_match_current_files():
    report = _load("pg114_family_holdout_replay_report_v1.json")
    for key, relative_path in {
        "target": "app/pg114_family_holdout_target.py",
        "bridge": "app/pg114_ood_replay.py",
        "bsp_core": "app/bsp_v3_research_core.py",
        "runner": "scripts/run_pg114_family_holdout_replay.py",
    }.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"]["source_hashes"][key]


def test_pg114_improvement_rule_allows_evaluation_but_not_promotion():
    rules = _load("improvement_rules.json")
    policy = rules["pg114_family_holdout_decoy_replay_policy"]
    assert policy["heldout_semantic"] == "security_policy_transition"
    assert policy["decoy_semantic"] == "shape_only_change"
    assert policy["transport"] == "fresh_loopback_uvicorn_process"
    assert policy["external_network"] is False
    assert policy["target_instance_count"] == 3
    assert policy["surface_slot_count"] == 4
    assert policy["episode_count"] == 12
    assert policy["step_count"] == 48
    assert policy["get_step_count"] == 24
    assert policy["post_step_count"] == 24
    assert policy["fresh_reset_per_action"] is True
    assert policy["family_holdout_confirm_recall"] == 1.0
    assert policy["decoy_false_accept_count"] == 0
    assert policy["withheld_oracle_abstain_rate"] == 1.0
    assert policy["training_eligible"] is False
    assert policy["memory_promotion_allowed"] is False
    assert policy["capability_gate_status"] == "passed_pg114_family_holdout_replay"
