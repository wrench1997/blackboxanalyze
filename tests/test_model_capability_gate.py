import hashlib

from app.model_capability_gate import evaluate_model_capability


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metrics(*, recall: float, precision: float, fpr: float, abstain: float, ece: float, queries: float) -> dict:
    return {
        "typed_recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "abstain_precision": abstain,
        "ece": ece,
        "median_queries": queries,
    }


def _evidence() -> dict:
    roles = [
        ("train", "xss"),
        ("dev", "sqli"),
        ("family_holdout", "command_injection"),
        ("ood_source", "state_machine"),
        ("negative_control", "ordinary_response"),
    ]
    rows = []
    for role, family in roles:
        for seed in (1, 2, 3):
            baseline = _metrics(recall=.70, precision=.98, fpr=.01, abstain=.99, ece=.04, queries=5)
            candidate = _metrics(
                recall=.75 if role in {"family_holdout", "ood_source"} else .70,
                precision=.99, fpr=.01, abstain=.995, ece=.03, queries=4,
            )
            rows.append({
                "sample_id": f"sample-{role}-{seed}",
                "dataset_id": f"dataset-{role}-{seed}",
                "source_id": f"source-id-{role}",
                "source_hash": _hash(f"source-{role}"),
                "target_instance_id": f"target-{role}-{seed}",
                "target_instance_ids": [f"target-{role}-{seed}"],
                "family_set": [family],
                "sampling_seed": seed,
                "role": role,
                "evidence_hash": _hash(f"evidence-{role}-{seed}"),
                "dataset_manifest_sha256": _hash(f"dataset-manifest-{role}-{seed}"),
                "split_manifest_sha256": _hash(f"split-manifest-{role}-{seed}"),
                "probe_sha256": _hash(f"probe-{role}-{seed}"),
                "oracle_contract_sha256": _hash(f"oracle-{role}-{seed}"),
                "checkpoint_sha256": _hash(f"checkpoint-{role}-{seed}"),
                "sample_count": 10,
                "unique_sample_count": 10,
                "denominator": 10,
                "positive_count": 5,
                "negative_count": 5,
                "abstain_count": 0,
                "metrics_status": "completed",
                "metrics": candidate,
                "baseline_metrics": baseline,
                "candidate_metrics": candidate,
            })
    return {
        "claim_id": "candidate-rule-ir-v2",
        "dataset_tests": rows,
        "baseline_metrics": _metrics(recall=.70, precision=.98, fpr=.01, abstain=.99, ece=.04, queries=5),
        "candidate_metrics": _metrics(recall=.75, precision=.99, fpr=.01, abstain=.995, ece=.03, queries=4),
        "baseline_worst_case_metrics": _metrics(recall=.65, precision=.97, fpr=.02, abstain=.98, ece=.06, queries=7),
        "candidate_worst_case_metrics": _metrics(recall=.71, precision=.97, fpr=.02, abstain=.98, ece=.06, queries=7),
        "false_positive_count": 0,
        "unit_tests_passed": True,
        "oracle_validated": True,
        "data_lineage_complete": True,
        "authorized_sources_attested": True,
        "raw_data_retained": False,
    }


def test_model_capability_requires_independent_dataset_evidence_and_baseline_gain():
    report = evaluate_model_capability(_evidence())
    assert report["status"] == "pass"
    assert report["claim_allowed"] is True
    assert report["training_allowed"] is True
    assert report["memory_promotion_allowed"] is True
    assert report["summary"]["distinct_dataset_count"] == 15


def test_unit_tests_alone_cannot_claim_model_improvement():
    evidence = _evidence()
    evidence["dataset_tests"] = []
    report = evaluate_model_capability(evidence)
    assert report["status"] == "blocked"
    assert report["claim_allowed"] is False
    assert report["training_allowed"] is False
    assert "no_dataset_tests" in report["reasons"]


def test_empty_capability_evidence_returns_structured_block_instead_of_claim():
    report = evaluate_model_capability({})
    assert report["status"] == "blocked"
    assert report["training_allowed"] is False
    assert "baseline_metrics_missing_or_invalid" in report["reasons"]


def test_holdout_gain_failure_is_no_proven_gain():
    evidence = _evidence()
    evidence["candidate_metrics"]["typed_recall"] = .705
    report = evaluate_model_capability(evidence)
    assert report["status"] == "no_proven_gain"
    assert report["claim_allowed"] is False
    assert "holdout_recall_gain_below_threshold" in report["reasons"]


def test_train_eval_source_overlap_is_blocked():
    evidence = _evidence()
    holdout = next(row for row in evidence["dataset_tests"] if row["role"] == "family_holdout")
    holdout["source_hash"] = evidence["dataset_tests"][0]["source_hash"]
    report = evaluate_model_capability(evidence)
    assert report["status"] == "blocked"
    assert "train_eval_source_overlap" in report["reasons"]
