import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg107_multistep_belief_gate_is_family_free_and_fail_closed():
    report = _load("pg107_multistep_generic_belief_report_v1.json")
    assert report["status"] == "passed_generic_multistep_belief_diagnostic"
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    metrics = report["metrics"]
    assert metrics["episode_count"] == 289
    assert metrics["multi_step_episode_rate"] == 1.0
    assert metrics["typed_oracle_called_count"] == 0
    assert metrics["confirmed_positive_count"] == 0
    assert metrics["decoy_group_count"] == 4
    assert metrics["abstain_after_anomaly_count"] == 5
    assert metrics["posterior_state_vocabulary"] == ["effect", "input_only", "no_effect", "unknown"]
    controller = report["controller"]
    assert controller["family_names_in_posterior"] is False
    assert controller["confirm_requires_typed_oracle"] is True


def test_pg107_trace_has_two_methods_fresh_evidence_and_no_raw_oracle_data():
    report = _load("pg107_multistep_generic_belief_report_v1.json")
    dataset = _load("pg107_multistep_generic_belief_visible_dataset_v1.json")
    trace = _load("pg107_multistep_generic_belief_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert trace["evaluator_labels_in_trace"] is False
    assert len(dataset["episodes"]) == 289
    assert len(trace["steps"]) == 578
    assert all(set(episode["methods"]) == {"GET", "POST"} for episode in dataset["episodes"])
    assert all(episode["typed_oracle_called"] is False and episode["confirmed_positive"] is False for episode in dataset["episodes"])
    assert all(step["typed_oracle_called"] is False and step["confirmed_positive"] is False for step in trace["steps"])
    assert all(step["negative_control_matched"] for step in trace["steps"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", step["evidence_sha256"]) for step in trace["steps"])
    assert len({step["evidence_sha256"] for step in trace["steps"]}) == len(trace["steps"])
    assert dataset["long_term_memory_write"] is False
    assert trace["long_term_memory_write"] is False
    text = json.dumps({"report": report, "dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("xss", "sql_injection", "workflow_invariant", "union select", "<script", "onerror"):
        assert forbidden not in text
    for name, digest in report["source"]["source_hashes"].items():
        path = {
            "train_dataset": "research/pg101_active_probe_signature_visible_dataset_v1.json",
            "pg105_dataset": "research/pg105_observable_projection_visible_dataset_v1.json",
            "pg105_trace": "research/pg105_observable_projection_trace_v1.json",
            "pg106_dataset": "research/pg106_decoy_projection_holdout_visible_dataset_v1.json",
            "pg106_trace": "research/pg106_decoy_projection_holdout_trace_v1.json",
            "inducer": "app/active_goal_label_inducer.py",
            "belief": "app/generic_belief_state.py",
            "runner": "scripts/run_pg107_multistep_generic_belief.py",
        }[name]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
