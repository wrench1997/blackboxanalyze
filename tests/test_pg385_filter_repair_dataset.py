from __future__ import annotations

import json

from scripts.build_pg385_filter_repair_dataset import build_dataset


def test_dataset_has_cross_implementation_filter_repair_matrix() -> None:
    dataset = build_dataset()
    assert dataset["status"] == "abstract_adversarial_candidate_only"
    assert dataset["counts"] == {
        "records": 160,
        "train": 80,
        "implementation_holdout": 80,
        "implementations": 2,
        "seeds": 2,
        "methods": 2,
        "scenarios": 5,
        "roles": 4,
    }
    assert len({row["record_id"] for row in dataset["records"]}) == 160
    assert {row["split"] for row in dataset["records"]} == {"train", "implementation_holdout"}
    assert {row["method"] for row in dataset["records"]} == {"GET", "POST"}


def test_dataset_is_abstract_and_negative_rows_abstain() -> None:
    dataset = build_dataset()
    serialized = json.dumps(dataset, ensure_ascii=False, sort_keys=True).casefold()
    for marker in ("http://", "https://", "wire=", "response_body=", "raw_value=", "<script", "document.cookie"):
        assert marker not in serialized
    assert all(row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True} for row in dataset["records"])
    assert all(row["training_eligible"] is False for row in dataset["records"])
    negative = [row for row in dataset["records"] if row["role"] == "negative"]
    assert negative
    assert all("safe_to_send=0" in row["target_tokens"] for row in negative)
    assert any("next_action=abstain" in row["target_tokens"] for row in negative if row["scenario_id"] != "missing_filter_observation")
    assert any("next_action=ask" in row["target_tokens"] for row in negative if row["scenario_id"] == "missing_filter_observation")
    assert dataset["safety"]["training_eligible"] == 0
    assert dataset["vocabulary"]["scope"] == "declared_abstract_ontology"
