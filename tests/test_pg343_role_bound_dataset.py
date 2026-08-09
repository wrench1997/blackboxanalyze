from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg343_full_axis_target_conditioned import audit_sources
from scripts.build_pg343_role_bound_dataset import build


ROOT = Path(__file__).resolve().parents[1]


def test_pg343_role_bound_builder_adds_source_attested_tokens_without_target_inference() -> None:
    dataset = build()
    assert dataset["counts"]["accepted_rows"] > 0
    assert dataset["isolation"]["role_step_not_inferred_from_target"] is True
    assert all("belief_probe_role=" in " ".join(row["context_tokens"]) for row in dataset["records"])
    assert all("belief_process_step=" in " ".join(row["context_tokens"]) for row in dataset["records"])
    assert all(row["training_eligible"] is False for row in dataset["records"])
    assert all(value is False for value in dataset["promotion"].values())
    rendered = json.dumps(dataset, ensure_ascii=False).casefold()
    for forbidden in ("payload=", "response_body=", "oracle=", "evaluator=", "route_literal="):
        assert forbidden not in rendered


def test_pg343_role_bound_audit_has_no_same_context_target_conflict(tmp_path: Path) -> None:
    dataset = build()
    # Audit accepts a dataset path, so write an in-memory temporary source
    # shape through the test's existing temp directory is unnecessary; the
    # builder itself already exposes role-bound rows.  Reuse the audit helper
    # with a small JSON fixture written by pytest's tmp_path.
    # (No raw values are asserted or printed.)
    report_path = tmp_path / "pg343_role_bound_full_axis_target_conditioned_dataset_v1.json"
    report_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    try:
        report = audit_sources((report_path,))
        assert report["counts"]["ambiguous_contexts"] == 0
        assert "context_target_ambiguity_requires_role_step_token" not in report["failures"]
        assert report["failures"] == []
        assert report["status"] == "diagnostic_passed_not_training_eligible"
        assert report["promotion"]["training_allowed"] is False
    finally:
        report_path.unlink(missing_ok=True)


def test_pg343_builder_accepts_all_explicit_preflight_collector_attestations() -> None:
    dataset = build(
        (
            ROOT / "research/pg333_three_impl_get_post_diagnostic_source_rows_v1.json",
            ROOT / "research/pg337_dvwa_failure_repair_source_rows_v1.json",
            ROOT / "research/pg343_webgoat_role_step_binding_source_rows_v1.json",
        )
    )
    # The extra PG-333 rows are source-attested preflight observations.  They
    # must not be silently discarded merely because their collector id is a
    # versioned typed-method/typed-stored-post contract.
    assert dataset["counts"]["input_rows"] == 60
    assert dataset["counts"]["rejected_rows"] == 0
    assert dataset["counts"]["accepted_rows"] > 21
    assert dataset["counts"]["accepted_training_rows"] == 0
    assert all(row["training_eligible"] is False for row in dataset["records"])
