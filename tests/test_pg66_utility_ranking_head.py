import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg66_ranking_head_removes_non_decisive_tie_noise():
    report = _read("pg66_utility_ranking_report_v1.json")
    dev = report["dev_metrics"]
    holdout = report["pg64_holdout_metrics"]
    assert report["status"] == "diagnostic_only"
    assert report["training"]["oracle_in_input"] is False
    assert report["training"]["family_in_input"] is False
    assert report["training"]["target_in_input"] is False
    assert report["training"]["label_type"] == "continuous_pre_oracle_utility_plus_pairwise_ranking"
    assert dev["ranking_accuracy"] < 0.8
    assert dev["decisive_ranking_accuracy"] == 1.0
    assert holdout["decisive_ranking_accuracy"] == 1.0
    assert dev["mean_utility_regret"] == 0.0
    assert holdout["mean_utility_regret"] == 0.0
    assert report["capability_gate"]["status"] == "passed"
    assert report["safety_gate"]["status"] == "passed"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg66_independent_noise_safety_and_bounded_trace():
    report = _read("pg66_utility_ranking_report_v1.json")
    independent = report["independent_noise_holdout"]
    trace = _read("pg66_utility_ranking_trace_v1.json")
    assert independent["target_recall"] == 1.0
    assert independent["negative_false_accept_count"] == 0
    assert independent["unknown_strict_abstain"] is True
    assert independent["multi_step_episode_rate"] >= 0.5
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["model_retrained_on_independent"] is False
    for episode in trace["independent_noise_holdout"]["episodes"]:
        for step in episode["steps"]:
            assert step["reset"]["fresh_target"] is True
            assert step["oracle_after_action"]["evaluator_state_hidden"] is True
            assert step["raw_probe_stored"] is False
            assert step["raw_response_stored"] is False
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
            assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_hash"])
    text = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "union select" not in text
    assert "onerror" not in text


def test_pg66_protocol_preregisters_tie_margin_and_separate_gates():
    protocol = _read("pg66_utility_ranking_protocol_v1.json")
    assert protocol["authorized_scope"]["target_host"] == "127.0.0.1"
    assert protocol["authorized_scope"]["external_network"] is False
    assert protocol["label_contract"]["selected_action_hard_label_not_used"] is True
    assert protocol["label_contract"]["tie_margin"] == 0.03
    assert protocol["required_gates"]["dev_decisive_ranking_accuracy_min"] == 0.8
    assert protocol["required_gates"]["maximum_mean_utility_regret"] == 0.02
    assert protocol["run_result"]["safety_status"] == "passed"
    assert protocol["run_result"]["capability_status"] == "passed"
