from __future__ import annotations

import json
from pathlib import Path

from app.failure_guided_scheduler import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg122_failure_guided_holdout_is_preserved_and_not_promoted() -> None:
    report = _load("pg122_failure_guided_authorization_holdout_report_v1.json")
    assert report["status"] == "completed_pg122_frozen_model_holdout"
    assert report["collection"]["target_instances"] == 9
    assert report["collection"]["steps"] == 144
    assert report["collection"]["get_steps"] == report["collection"]["post_steps"] == 72
    assert report["evaluation"]["authorization_positive_recall"] == 0.0
    assert report["evaluation"]["decoy_false_accept_count"] == 0
    assert report["evaluation"]["blind_oracle_abstain_rate"] == 0.444444
    assert report["checks"]["failure_signature_present_on_all_steps"] is True
    assert report["checks"]["memory_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg122_trace_contains_bounded_failure_signatures_only() -> None:
    trace = _load("pg122_failure_guided_authorization_holdout_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["failure_signatures_visible"] is True
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    for target in trace["targets"]:
        for episode in target["episodes"]:
            for step in episode["steps"]:
                signature = step["failure_signature"]
                assert signature["schema_version"] == SCHEMA_VERSION
                assert signature["raw_probe_retained"] is False
                assert signature["raw_response_retained"] is False
                assert signature["memory_promotion_allowed"] is False
