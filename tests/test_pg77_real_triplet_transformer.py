import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg77_real_triplet_transformer_keeps_external_schema_failure_visible():
    report = _read("pg77_real_triplet_transformer_report_v1.json")
    assert report["source"]["device"] == "cuda"
    assert report["dataset"]["train"] == 28
    assert report["dataset"]["dev"] == 14
    assert report["dataset"]["unknown_family_holdout"] == 12
    assert report["dataset"]["external_known_schema_diagnostic"] == 21
    assert report["metrics"]["dev_holdout"]["confirm_recall"] == 1.0
    assert report["metrics"]["dev_holdout"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_family_holdout"]["strict_abstain"] is True
    assert report["metrics"]["unknown_family_holdout"]["misname_count"] == 0
    assert report["metrics"]["external_known_schema_diagnostic"]["confirm_recall"] == 0.428571
    assert report["capability_gate"]["status"] == "blocked"
    assert report["capability_gate"]["checks"]["external_known_recall_min"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg77_dataset_has_oracle_after_target_and_no_raw_fields():
    dataset = _read("pg77_real_triplet_trace_dataset_v1.json")
    contract = dataset["model_input_contract"]
    assert contract["family_name_in_tokens"] is False
    assert contract["source_id_in_tokens"] is False
    assert contract["implementation_in_tokens"] is False
    assert contract["route_words_in_tokens"] is False
    assert contract["typed_oracle_before_target_marker"] is False
    assert dataset["raw_probe_strings_stored"] is False
    assert dataset["raw_response_bodies_stored"] is False
    assert dataset["long_term_memory_write"] is False
    for row in dataset["rows"]:
        assert row["raw_probe_stored"] is False
        assert row["raw_response_stored"] is False
        assert row["tokens"][0] == "BOS"
        assert "ORACLE_TARGET" in row["tokens"]
        assert row["tokens"].index("ORACLE_TARGET") == row["oracle_index"]


def test_pg77_unknown_abstain_is_calibrated_not_forced():
    protocol = _read("pg77_real_triplet_transformer_protocol_v1.json")
    assert protocol["required_gates"]["unknown_strict_abstain"] is True
    assert protocol["required_gates"]["external_known_recall_min"] == 0.80
    assert protocol["required_gates"]["fresh_multi_implementation_replay_required"] is True
    source = (ROOT / "scripts" / "run_pg77_real_triplet_transformer.py").read_text(encoding="utf-8")
    assert 'decision = "abstain" if unknown' not in source
