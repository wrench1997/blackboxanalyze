import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_file(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_pg120_preserves_failure_and_does_not_promote():
    report = _load("pg120_cross_impl_holdout_report_v1.json")
    assert report["status"] == "completed_pg120_cross_implementation_metadata_holdout"
    assert report["scope"]["weights_frozen"] is True
    assert report["blind_pg120"]["metadata_positive_recall"] == 1.0
    assert report["blind_pg120"]["decoy_false_accept_count"] == 0
    assert report["blind_pg120"]["blind_oracle_abstain_rate"] == 0.0
    assert report["checks"]["pg120_unknown_abstain_nonzero"] is False
    assert report["checks"]["all_strengths_unknown_abstain_nonzero"] is False
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg120_trace_is_cross_impl_get_post_and_evidence_bound():
    report = _load("pg120_cross_impl_holdout_report_v1.json")
    collection = report["collection"]
    assert collection["target_implementation"] == "pg120-eta-independent-target"
    assert collection["target_instance_count"] == 9
    assert collection["seed_count"] == 3
    assert collection["episode_count"] == 36
    assert collection["step_count"] == 144
    assert collection["get_step_count"] == 72
    assert collection["post_step_count"] == 72
    assert collection["decoy_strengths"] == [0, 1, 2]
    assert collection["encoding_chain"] == ["url_percent", "unicode_escape", "html_entity"]
    assert collection["fresh_reset_per_step"] is True
    assert collection["evidence_hash_valid"] is True

    trace = _load("pg120_cross_impl_holdout_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    episodes = [episode for target in trace["sources"] for episode in target["episodes"]]
    assert len(episodes) == 36
    for episode in episodes:
        assert episode["episode_report"]["status"] == "accepted_evaluation"
        assert episode["negative_control_pair_clear"] is True
        assert [step["action_manifest"]["method"] for step in episode["steps"]] == ["GET", "GET", "POST", "POST"]
        assert len({step["fresh_reset"]["reset_epoch"] for step in episode["steps"]}) == 4
        step_hashes = {step["evidence_sha256"] for step in episode["steps"]}
        assert episode["rule_ir_slot_binding"]["evidence_sha256"] in step_hashes
        for step in episode["steps"]:
            model_text = json.dumps(step["model_input"], ensure_ascii=False).casefold()
            for forbidden in ("oracle", "positive_authority", "family", "target_instance_id", "probe_ref", "probe_sha256"):
                assert forbidden not in model_text
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
        for record in episode["evidence_records"]:
            body = dict(record)
            declared = body.pop("evidence_hash")
            assert declared == _sha256_json(body)


def test_pg120_protocol_rules_and_sources_are_persisted():
    protocol = _load("pg120_cross_implementation_metadata_holdout_protocol_v1.json")
    assert protocol["model_contract"]["weights_frozen"] is True
    assert protocol["target_contract"]["route_schema_is_distinct_from_pg119"] is True
    assert protocol["action_contract"]["fresh_reset_per_action"] is True
    assert protocol["promotion"]["memory_promotion_allowed"] is False
    rules = _load("improvement_rules.json")
    policy = rules["pg120_cross_implementation_metadata_holdout_policy"]
    assert policy["training_eligible"] is False
    assert policy["unknown_abstain_rate"] == 0.0
    assert policy["capability_gate_status"] == "blocked_unknown_abstain_cross_implementation"
    report = _load("pg120_cross_impl_holdout_report_v1.json")
    for key, relative_path in {
        "eta_target": "app/pg120_eta_metadata_target.py",
        "eta_bridge": "app/pg120_cross_impl_replay.py",
        "runner": "scripts/run_pg120_cross_impl_holdout.py",
        "frozen_checkpoint": "artifacts/pg119-metadata-rule-ir-decoder-v1/model.pt",
        "pg119_report": "research/pg119_metadata_training_report_v1.json",
    }.items():
        assert _sha256_file(relative_path) == report["source"][key]
