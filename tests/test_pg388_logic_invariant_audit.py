from __future__ import annotations

from scripts.audit_pg388_logic_invariant_dataset import audit_dataset


def test_pg388_dataset_audit_passes_candidate_contract() -> None:
    report = audit_dataset("research/pg388_logic_invariant_dataset_v1.json")
    assert report["status"] == "passed_candidate_audit"
    assert report["counts"]["invalid_hash_rows"] == 0
    assert report["counts"]["unsafe_rows"] == 0
    assert report["context_firewall"]["marker_hits"] == []
    assert report["information_preservation"]["missing_axes"] == []
    assert report["training_eligible"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
