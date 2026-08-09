import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg75_known_and_independent_unknown_gates_pass_but_promotion_stays_blocked():
    report = _read("pg75_triplet_context_delta_ablation_report_v4.json")
    assert report["status"] == "candidate_training_completed"
    assert report["source"]["model_retrained_on_unknown_family"] is False
    assert report["source"]["family_in_features"] is False
    assert report["source"]["oracle_in_features"] is False
    assert report["dataset"]["train_seeds"] == [74101, 74102]
    assert report["dataset"]["dev_seeds"] == [74103]
    assert report["dataset"]["neutral_surface_context_features"] is True
    assert report["metrics"]["dev_holdout"]["confirm_recall"] == 1.0
    assert report["metrics"]["dev_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["legacy_unknown_family_holdout"]["misname_count"] == 0
    assert report["metrics"]["legacy_unknown_family_holdout"]["strict_abstain"] is True
    assert report["metrics"]["independent_unknown_triplet_holdout"]["misname_count"] == 0
    assert report["metrics"]["independent_unknown_triplet_holdout"]["strict_abstain"] is True
    assert report["capability_gate"]["checks"]["independent_triplet_unknown_misname_zero"] is True
    assert report["capability_gate"]["checks"]["independent_triplet_unknown_strict_abstain"] is True
    assert report["capability_gate"]["checks"]["independent_source_attested"] is True
    assert report["capability_gate"]["status"] == "passed"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg75_candidate_trace_is_raw_free_and_evaluation_only():
    trace = _read("pg75_triplet_context_delta_ablation_trace_v4.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["model_retrained_on_unknown_family"] is False
    assert trace["family_in_features"] is False
    assert trace["oracle_in_features"] is False
    assert trace["negative_oracle_in_features"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
