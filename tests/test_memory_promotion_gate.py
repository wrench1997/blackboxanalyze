from app.memory_promotion_gate import assess_memory_promotion
from app.payload_grounding import SourceGroundedMemory


def _evaluations(*, false_positive=False, datasets=("a", "b", "c")):
    rows = []
    index = 0
    for dataset in datasets:
        for seed in (11, 17):
            index += 1
            rows.append({
                "dataset_id": dataset,
                "sampling_seed": seed,
                "target_instance_id": f"{dataset}-target-{seed}",
                "rule_key": "dom:encoded_dom_markup",
                "accepted": True,
                "oracle_revalidated": True,
                "false_positive": bool(false_positive and dataset == "c"),
                "evidence_hash": f"evidence-{index:032d}",
                "source_hash": (dataset * 64)[:64],
                "local_only": True,
            })
    return rows


def test_long_term_memory_requires_multiple_datasets_and_sampling_seeds():
    result = assess_memory_promotion("dom:encoded_dom_markup", _evaluations())
    assert result["promote"] is True
    assert result["status"] == "promote"
    assert result["summary"]["distinct_dataset_count"] == 3
    assert result["summary"]["distinct_source_hash_count"] == 3
    assert result["summary"]["distinct_target_instance_count"] == 6

    insufficient = assess_memory_promotion("dom:encoded_dom_markup", _evaluations(datasets=("a",)))
    assert insufficient["promote"] is False
    assert "insufficient_distinct_datasets" in insufficient["reasons"]
    assert "insufficient_target_instances" in insufficient["reasons"]


def test_any_cross_dataset_false_positive_quarantines_memory():
    result = assess_memory_promotion("dom:encoded_dom_markup", _evaluations(false_positive=True))
    assert result["promote"] is False
    assert "c:false_positive_rate_exceeded" in result["reasons"]


def test_positive_without_oracle_revalidation_is_quarantined():
    rows = _evaluations()
    rows[0].pop("oracle_revalidated")
    result = assess_memory_promotion("dom:encoded_dom_markup", rows)
    assert result["promote"] is False
    assert "row_0:missing_oracle_revalidation" in result["reasons"]


def test_source_grounded_memory_exposes_promotion_policy_without_changing_episode_support():
    memory = SourceGroundedMemory(seed=3)
    summary = memory.summary()
    assert summary["long_term_promotion_requires_cross_dataset_replay"] is True
    assert summary["promotion_policy"]["min_distinct_datasets"] == 3
    assert summary["promotion_policy"]["min_distinct_source_hashes"] == 3
    assert memory.audit_promotion("rule", _evaluations()) ["promote"] is False


def test_dataset_labels_from_one_source_cannot_promote():
    rows = _evaluations()
    for row in rows:
        row["source_hash"] = "same-source" * 8
    result = assess_memory_promotion("dom:encoded_dom_markup", rows)
    assert result["promote"] is False
    assert "insufficient_distinct_source_hashes" in result["reasons"]


def test_missing_source_hash_is_fail_closed():
    rows = _evaluations()
    for row in rows:
        row.pop("source_hash")
    result = assess_memory_promotion("dom:encoded_dom_markup", rows)
    assert result["promote"] is False
    assert "row_0:missing_source_hash" in result["reasons"]


def test_repeated_evidence_across_seeds_is_not_independent():
    rows = _evaluations()
    for row in rows:
        row["evidence_hash"] = "same-evidence" * 8
    result = assess_memory_promotion("dom:encoded_dom_markup", rows)
    assert result["promote"] is False
    assert any("insufficient_independent_seed_evidence" in reason for reason in result["reasons"])


def test_three_sources_with_distinct_seed_manifests_promote():
    result = assess_memory_promotion("dom:encoded_dom_markup", _evaluations())
    assert result["promote"] is True
    assert all(not details["reasons"] for details in result["per_dataset"].values())
