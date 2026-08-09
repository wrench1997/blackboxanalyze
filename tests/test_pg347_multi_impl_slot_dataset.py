import json
from pathlib import Path

from scripts.build_pg347_multi_impl_slot_dataset import build_dataset, build_vocabulary


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg347_merges_full_slot_rows_without_cross_split_leak():
    pg338 = _load("pg338_information_preserving_process_token_v1.json")
    pg345 = _load("pg345_decision_boundary_role_bound_dataset_v1.json")
    dataset = build_dataset(pg338=pg338, pg345=pg345, pg338_file_sha256="a" * 64, pg345_file_sha256="b" * 64)
    assert dataset["counts"]["rows"] == 39
    assert dataset["counts"]["train_rows"] == 27
    assert dataset["counts"]["implementation_holdout_rows"] == 12
    assert dataset["counts"]["implementation_split_leaks"] == 0
    assert dataset["counts"]["training_eligible_rows"] == 0
    assert all(row["role_step_binding"]["source_attested"] is True for row in dataset["records"])


def test_pg347_normalization_keeps_context_abstract_and_sidecars_off_context():
    dataset = build_dataset(pg338=_load("pg338_information_preserving_process_token_v1.json"), pg345=_load("pg345_decision_boundary_role_bound_dataset_v1.json"), pg338_file_sha256="a" * 64, pg345_file_sha256="b" * 64)
    for row in dataset["records"]:
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert row["oracle_answer_in_context"] is False
        assert not any(token.startswith(("payload=", "response_body=", "raw_", "oracle=", "evaluator=", "route=")) for token in row["context_tokens"])


def test_pg347_vocabulary_is_append_only_diagnostic():
    dataset = build_dataset(pg338=_load("pg338_information_preserving_process_token_v1.json"), pg345=_load("pg345_decision_boundary_role_bound_dataset_v1.json"), pg338_file_sha256="a" * 64, pg345_file_sha256="b" * 64)
    vocab = build_vocabulary(dataset)
    assert vocab["append_only"] is True
    assert vocab["forbidden_tokens"] == []
    assert vocab["context_vocabulary_size"] > 100
    assert vocab["target_vocabulary_size"] >= 10
    assert all("payload=" not in token and "response_body=" not in token for token in vocab["context_tokens"])
