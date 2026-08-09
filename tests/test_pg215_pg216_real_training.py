import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg215_real_trace_dataset_has_seed_and_route_holdouts() -> None:
    report = _load("research/pg215_pikachu_real_trace_dataset_report_v1.json")
    dataset = _load("research/pg215_pikachu_real_trace_dataset_v1.json")
    assert report["status"] == "completed_real_cross_seed_trace_collection"
    assert report["counts"] == {
        "episode_count": 28,
        "new_episode_count": 14,
        "step_row_count": 112,
        "train_row_count": 48,
        "holdout_row_count": 64,
        "get_row_count": 80,
        "post_row_count": 32,
        "clean_reset_row_count": 112,
        "restart_row_count": 0,
    }
    assert dataset["training_contract"]["only_bounded_response_features"] is True
    assert len({tuple(row["tokens"]) for row in dataset["tokens"]}) >= 18
    assert set(dataset["train_seeds"]).isdisjoint(dataset["holdout_seeds"])
    assert all(row["database_clean_reset_verified"] for row in dataset["tokens"])
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["tokens"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["tokens"])


def test_pg216_runs_both_large_capacity_variants_on_real_holdout() -> None:
    report = _load("research/pg216_real_trace_capacity_training_report_v1.json")
    assert report["status"] == "completed_real_trace_capacity_sweep"
    assert report["device"] == "cuda"
    assert report["data"]["base_train_rows"] == 600
    assert report["data"]["pg215_train_rows"] == 48
    assert len(report["variants"]) == 4
    assert {row["variant"] for row in report["variants"]} == {"160m", "200m"}
    assert {row["seed"] for row in report["variants"]} == {17701, 17702}
    assert all(row["parameter_count"] in {160089065, 197537513} for row in report["variants"])
    assert report["capacity_200m_gain_repeated"] is False
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["database_write"] is False
    assert report["safety"]["raw_payloads_in_model"] is False
    assert report["safety"]["raw_responses_in_model"] is False
