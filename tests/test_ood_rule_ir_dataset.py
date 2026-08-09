import json

from app.ood_rule_ir_dataset import generate_manifest


def test_pg31_manifest_is_reproducible_and_evaluation_only():
    first = generate_manifest(seeds=(11, 22, 33), samples_per_role=2)
    second = generate_manifest(seeds=(11, 22, 33), samples_per_role=2)
    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["training_eligible"] is False
    assert first["training_artifact_generated"] is False
    assert first["model_evaluation_completed"] is False
    assert len(first["samples"]) == 5 * 3 * 2


def test_pg31_manifest_has_independent_roles_sources_seeds_and_typed_oracles():
    manifest = generate_manifest(seeds=(101, 202, 303), samples_per_role=1)
    rows = manifest["samples"]
    assert set(row["role"] for row in rows) == {
        "train", "dev", "family_holdout", "ood_source", "negative_control"
    }
    assert len({row["dataset_id"] for row in rows}) == 5
    assert len({row["source_hash"] for row in rows}) == 5
    assert len({row["sampling_seed"] for row in rows}) == 3
    assert len({row["target_instance_id"] for row in rows}) == 15
    assert all(row["typed_oracle"]["authority"] == "in_repo_fixture_contract" for row in rows)
    assert all(row["safety"]["training_eligible"] is False for row in rows)
    assert all(len(row["evidence_hash"]) == 64 for row in rows)
    # Negative controls cannot become positive because of a response shape.
    assert all(
        not row["typed_oracle"]["positive"]
        for row in rows
        if row["role"] == "negative_control"
    )


def test_pg31_manifest_contains_no_raw_probe_or_payload_strings():
    manifest = generate_manifest(seeds=(1, 2, 3), samples_per_role=1)
    serialized = json.dumps(manifest, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "union select" not in serialized
    assert "body_preview" not in serialized
    assert '"probe":' not in serialized
    assert manifest["source"]["raw_payloads_present"] is False
    for row in manifest["samples"]:
        assert set(row["rule_ir"]) >= {"grammar_version", "family_slot", "surface_slot", "transport_slot"}
        assert "probe" not in row["rule_ir"]


def test_pg31_dataset_test_skeleton_is_explicitly_pending():
    manifest = generate_manifest(seeds=(7, 8, 9), samples_per_role=1)
    assert len(manifest["dataset_tests"]) == 15
    assert all(item["metrics_status"] == "pending_model_run" for item in manifest["dataset_tests"])
    assert all(item["metrics"]["typed_recall"] == 0.0 for item in manifest["dataset_tests"])
    assert all(item["sample_id"] and item["target_instance_ids"] for item in manifest["dataset_tests"])
