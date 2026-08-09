from __future__ import annotations

from copy import deepcopy

from scripts.audit_pg388_logic_canary_trajectory_dataset import audit_dataset
from scripts.build_pg388_logic_canary_trajectory_dataset import build_dataset


def test_canary_trajectory_builder_has_disjoint_matrix_and_abstract_targets() -> None:
    artifact = build_dataset()
    assert artifact["counts"] == {"records": 510, "train": 255, "implementation_holdout": 255, "cases": 17, "implementations": 2, "seeds": 3, "phases": 5, "roles": 4}
    assert len(artifact["rows"]) == 510
    assert all(row["training_eligible"] is False for row in artifact["rows"])
    assert all("vulnerable_effect" not in " ".join(row["context_tokens"]) for row in artifact["rows"])
    assert all("effect_shape=" in " ".join(row["target_tokens"]) for row in artifact["rows"])


def test_canary_trajectory_audit_passes_and_keeps_promotion_closed() -> None:
    report = audit_dataset(build_dataset())
    assert report["status"] == "passed_candidate_trajectory_audit"
    assert report["invalid_rows"] == 0
    assert report["context_target_leaks"] == 0
    assert report["cross_split_context_overlap"] == 0
    assert report["cross_split_context_target_overlap"] == 0
    assert report["promotion"]["training_allowed"] is False


def test_canary_trajectory_audit_rejects_row_hash_tampering() -> None:
    artifact = build_dataset()
    changed = deepcopy(artifact)
    changed["rows"][0]["target_tokens"] = ["effect_shape=changed"]
    report = audit_dataset(changed)
    assert report["status"] == "blocked_canary_trajectory_audit"
    assert report["invalid_rows"] >= 1
