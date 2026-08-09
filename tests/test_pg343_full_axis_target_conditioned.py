from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg343_full_axis_target_conditioned import audit_sources


ROOT = Path(__file__).resolve().parents[1]


def test_pg343_finds_role_ambiguity_and_refuses_promotion() -> None:
    report = audit_sources()
    assert report["status"] == "blocked_role_step_context_missing"
    assert report["counts"]["ambiguous_contexts"] >= 1
    assert "context_target_ambiguity_requires_role_step_token" in report["failures"]
    assert report["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def test_pg343_has_target_coverage_but_keeps_context_firewall_and_opaque_hashes() -> None:
    report = audit_sources()
    assert report["counts"]["unique_context_target_rows"] > 0
    assert report["target_counts"]["train:positive_probe"] > 0
    assert report["target_counts"]["train:repair"] > 0
    assert report["target_counts"]["train:negative_abstain"] > 0
    assert report["target_counts"]["implementation_holdout:positive_probe"] > 0
    assert report["target_counts"]["implementation_holdout:repair"] > 0
    assert report["target_counts"]["implementation_holdout:negative_abstain"] > 0
    rendered = json.dumps(report, ensure_ascii=False).casefold()
    for forbidden in ("payload=", "response_body=", "oracle=", "evaluator=", "route_literal="):
        assert forbidden not in rendered


def test_pg343_report_is_bounded_and_does_not_emit_context_or_target_sequences() -> None:
    report = audit_sources()
    rendered = json.dumps(report, ensure_ascii=False)
    assert "context_tokens" not in rendered
    assert "target_tokens" not in rendered
    assert all(len(item["context_sha256"]) == 64 for item in report["ambiguity_hashes"])
    assert all(len(value) == 64 for item in report["ambiguity_hashes"] for value in item["target_sha256s"])
