import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg81_source_holdout_isolated_and_blocked():
    report = _read("pg81_source_holdout_transformer_report_v1.json")
    assert report["source"]["device"] == "cuda"
    assert report["dataset"]["train"] == 216
    assert report["dataset"]["dev"] == 108
    assert report["dataset"]["source_holdout"] == 216
    assert report["dataset"]["unknown_family_holdout"] == 12
    assert report["source"]["family_in_tokens"] is False
    assert report["source"]["source_in_tokens"] is False
    assert report["metrics"]["dev_holdout"]["confirm_recall"] == 0.0
    assert report["metrics"]["source_holdout"]["confirm_recall"] == 0.0
    assert report["metrics"]["unknown_family_holdout"]["strict_abstain"] is True
    assert report["capability_gate"]["status"] == "blocked"
    assert report["promotion"]["training_allowed"] is False


def test_pg81_dataset_split_has_no_raw_persistence():
    dataset = _read("pg81_source_holdout_trace_dataset_v1.json")
    assert dataset["training_eligible"] is False
    assert dataset["evaluation_only"] is True
    assert dataset["model_input_contract"]["family_name_in_tokens"] is False
    assert dataset["model_input_contract"]["source_id_in_tokens"] is False
    assert dataset["model_input_contract"]["typed_oracle_before_target_marker"] is False
    assert dataset["raw_probe_strings_stored"] is False
    assert dataset["raw_response_bodies_stored"] is False
    assert dataset["long_term_memory_write"] is False
