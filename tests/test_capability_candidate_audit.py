import hashlib
import json

from app.capability_candidate_audit import audit_capability_reports


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metrics(recall: float, *, precision: float = .98, fpr: float = .01, abstain: float = .99, ece: float = .04) -> dict:
    return {
        "typed_recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "abstain_precision": abstain,
        "ece": ece,
        "median_queries": 4,
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
            baseline = _metrics(.70)
            candidate = _metrics(.75 if role in {"family_holdout", "ood_source"} else .70, precision=.99, abstain=.995, ece=.03)
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
        "claim_id": "audit-candidate",
        "dataset_tests": rows,
        "baseline_metrics": _metrics(.70),
        "candidate_metrics": _metrics(.75, precision=.99, abstain=.995, ece=.03),
        "baseline_worst_case_metrics": _metrics(.65, precision=.97, fpr=.02, abstain=.98, ece=.06),
        "candidate_worst_case_metrics": _metrics(.71, precision=.97, fpr=.02, abstain=.98, ece=.06),
        "false_positive_count": 0,
        "unit_tests_passed": True,
        "oracle_validated": True,
        "data_lineage_complete": True,
        "authorized_sources_attested": True,
        "raw_data_retained": False,
    }


def test_existing_style_report_without_explicit_dataset_evidence_is_blocked(tmp_path):
    report_path = tmp_path / "frontend-report.json"
    report_path.write_text(json.dumps({
        "schema_version": "some-evaluation-v1",
        "status": "completed",
        "results": {"holdout": {"accuracy": .9}},
        "acceptance": {"accepted": True},
    }), encoding="utf-8")
    result = audit_capability_reports([report_path])
    assert result["status"] == "blocked"
    assert result["claim_allowed"] is False
    assert result["training_allowed"] is False
    assert "no_explicit_independent_dataset_evidence" in result["reasons"]
    assert result["actions"]["trainer_invoked"] is False


def test_nested_pg30_evidence_is_evaluated_without_side_effects(tmp_path):
    report_path = tmp_path / "candidate.json"
    report_path.write_text(json.dumps({
        "schema_version": "candidate-report-v1",
        "status": "completed",
        "capability_evidence": _evidence(),
    }), encoding="utf-8")
    result = audit_capability_reports([report_path])
    assert result["status"] == "pass"
    assert result["claim_allowed"] is True
    assert result["actions"] == {
        "trainer_invoked": False,
        "checkpoint_written": False,
        "training_dataset_generated": False,
        "memory_write_attempted": False,
    }
    assert result["evaluated_gate"]["summary"]["distinct_dataset_count"] == 15


def test_explicit_but_incomplete_dataset_evidence_is_blocked(tmp_path):
    evidence = _evidence()
    evidence["dataset_tests"] = []
    report_path = tmp_path / "incomplete.json"
    report_path.write_text(json.dumps({"capability_evidence": evidence}), encoding="utf-8")
    result = audit_capability_reports([report_path])
    assert result["status"] == "blocked"
    assert "no_dataset_tests" in result["reasons"]
    assert result["evaluated_gate"] is not None


def test_candidate_metrics_without_required_gain_are_not_proven(tmp_path):
    evidence = _evidence()
    evidence["candidate_metrics"]["typed_recall"] = .705
    report_path = tmp_path / "no-gain.json"
    report_path.write_text(json.dumps(evidence), encoding="utf-8")
    result = audit_capability_reports([report_path])
    assert result["status"] == "no_proven_gain"
    assert result["claim_allowed"] is False
    assert "holdout_recall_gain_below_threshold" in result["reasons"]


def test_multiple_capability_reports_are_not_merged(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"capability_evidence": _evidence()}), encoding="utf-8")
    second.write_text(json.dumps({"capability_evidence": _evidence()}), encoding="utf-8")
    result = audit_capability_reports([first, second])
    assert result["status"] == "blocked"
    assert "multiple_capability_evidence_reports_not_merged" in result["reasons"]


def test_unreadable_report_is_blocked(tmp_path):
    report_path = tmp_path / "bad.json"
    report_path.write_text("not-json", encoding="utf-8")
    result = audit_capability_reports([report_path])
    assert result["status"] == "blocked"
    assert any("report_unreadable" in reason for reason in result["reasons"])
