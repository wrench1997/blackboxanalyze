"""Contract tests for the PG-382 factorized abstract matrix."""

from __future__ import annotations

from scripts.build_pg382_factorized_adversarial_dataset import build_dataset


def test_factorized_matrix_has_disjoint_source_hashes_and_shared_abstract_surface() -> None:
    dataset = build_dataset()
    assert dataset["status"] == "abstract_adversarial_candidate_only"
    assert dataset["counts"] == {
        "records": 6336,
        "train": 3168,
        "implementation_holdout": 3168,
        "implementations": 2,
        "surface_templates": 12,
        "methods": 2,
        "roles": 4,
        "unique_record_ids": 6336,
        "source_hashes": 24,
        "training_eligible": 0,
    }
    assert dataset["split_contract"]["source_hashes_disjoint"] is True
    assert dataset["split_contract"]["abstract_matrix_shared_across_implementations"] is True
    assert dataset["audit"]["cross_split_source_hash_overlap"] == 0
    assert dataset["audit"]["abstract_matrix_cross_split_overlap_expected"] is True


def test_factorized_rows_keep_raw_and_promotion_firewalls_closed() -> None:
    dataset = build_dataset()
    for row in dataset["records"][::257]:
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert row["oracle_answer_in_context"] is False
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
        assert all(value is False for value in row["promotion"].values())
    assert dataset["safety"]["raw_payload_in_context"] is False
    assert dataset["promotion"]["capability_training_allowed"] is False
