import json
from pathlib import Path


def _read(name):
    return json.loads(Path("research", name).read_text(encoding="utf-8"))


def test_pg56_dataset_is_abstract_and_split_by_source():
    dataset = _read("pg56_causal_trace_dataset_v1.json")
    report = _read("pg56_causal_trace_dataset_report_v1.json")
    assert len(dataset["rows"]) == 630
    assert dataset["split_counts"] == {"train": 322, "dev": 188, "holdout": 120}
    assert report["token_vocabulary_size"] == 104
    contract = dataset["model_input_contract"]
    assert contract["family_name_in_tokens"] is False
    assert contract["source_id_in_tokens"] is False
    assert contract["implementation_in_tokens"] is False
    assert contract["raw_probe_in_tokens"] is False
    assert contract["raw_response_body_in_tokens"] is False
    assert contract["typed_oracle_before_target_marker"] is False
    for row in dataset["rows"]:
        assert row["raw_probe_stored"] is False
        assert row["raw_response_stored"] is False
        assert "BOS" in row["tokens"]
        assert "EOS" in row["tokens"]


def test_pg56_pretraining_is_not_promoted_as_family_capability():
    protocol = _read("pg56_causal_trace_pretraining_protocol_v1.json")
    report = _read("pg56_causal_trace_pretraining_report_v1.json")
    holdout = report["metrics"]["holdout"]
    assert report["device"] == "cuda"
    assert report["split_counts"] == {"train": 322, "dev": 188, "holdout": 120}
    assert holdout["targets"]["next_action"]["accuracy"] == 1.0
    assert holdout["targets"]["oracle_outcome"]["accuracy"] == 1.0
    assert holdout["targets"]["unknown_family_naming_attempts"] == 0
    assert holdout["targets"]["unknown_family_strict_abstain"] is True
    assert report["training_promotion_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["formal_family_capability_claim_allowed"] is False
    assert protocol["run_result"]["status"] == "pretraining_baseline_complete_family_capability_unproven"
