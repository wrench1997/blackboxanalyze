import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_pg117_double_holdout_exposes_frozen_positive_recall_gap_without_false_accepts():
    report = _load("pg117_double_holdout_report_v1.json")
    assert report["status"] == "completed_pg117_double_holdout_evaluation"
    assert report["scope"]["real_vulnerability_scanner_claim_allowed"] is False
    collection = report["collection"]
    assert collection["target_instance_count"] == 3
    assert collection["episode_count"] == 12
    assert collection["step_count"] == 48
    assert collection["get_step_count"] == 24
    assert collection["post_step_count"] == 24
    assert collection["encoding_chain"] == ["url_percent", "html_entity"]
    evaluation = report["evaluation"]
    assert evaluation["route_positive_recall"] == 0.0
    assert evaluation["decoy_false_accept_count"] == 0
    assert evaluation["blind_oracle_abstain_rate"] == 1.0
    assert evaluation["step_metrics"]["accuracy"] == 0.895833
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["checks"]["positive_recall_nonzero"] is False
    assert report["checks"]["all_abstain_not_success"] is False


def test_pg117_trace_is_fresh_paired_and_model_input_blind():
    trace = _load("pg117_double_holdout_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert trace["implementation_holdout_from_pg116"] is True
    assert trace["encoding_holdout_from_pg116"] is True
    episodes = [episode for target in trace["sources"] for episode in target["episodes"]]
    assert len(episodes) == 12
    for episode in episodes:
        assert episode["episode_report"]["status"] == "accepted_evaluation"
        assert episode["negative_control_pair_clear"] is True
        assert [step["action_manifest"]["method"] for step in episode["steps"]] == ["GET", "GET", "POST", "POST"]
        assert len({step["fresh_reset"]["reset_epoch"] for step in episode["steps"]}) == 4
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


def test_pg117_protocol_report_and_source_hashes_are_persisted():
    protocol = _load("pg117_double_holdout_protocol_v1.json")
    assert protocol["model_contract"]["weights_frozen"] is True
    assert protocol["model_contract"]["training_rows_added"] is False
    assert protocol["action_contract"]["fresh_reset_per_action"] is True
    assert protocol["promotion"]["memory_promotion_allowed"] is False
    report = _load("pg117_double_holdout_report_v1.json")
    for key, relative_path in {
        "target": "app/pg117_gamma_target.py",
        "bridge": "app/pg117_double_holdout_replay.py",
        "runner": "scripts/run_pg117_double_holdout.py",
        "pg116_report": "research/pg116_multisource_trace_training_report_v1.json",
    }.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"][key]
    rules = _load("improvement_rules.json")
    policy = rules["pg117_double_implementation_encoding_holdout_policy"]
    assert policy["training_eligible"] is False
    assert policy["frozen_route_positive_recall"] == 0.0
    assert policy["frozen_decoy_false_accept_count"] == 0
    assert policy["frozen_blind_oracle_abstain_rate"] == 1.0
    assert policy["memory_promotion_allowed"] is False
