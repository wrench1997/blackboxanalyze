from __future__ import annotations

import json
from pathlib import Path

from app.pg331_source_row import validate_pg331_source_row
from scripts.build_pg388_logic_surface_source_rows import build


def test_build_is_bounded_strict_and_diagnostic_only(tmp_path: Path) -> None:
    output = tmp_path / "surface.json"
    report = build(output)

    assert output.exists()
    assert report["status"] == "completed_inprocess_diagnostic_only"
    assert report["counts"] == {
        "source_rows": 144,
        "strict_valid": 136,
        "incomplete_rows": 8,
        "typed": 144,
        "fresh_resets": 144,
        "train": 72,
        "implementation_holdout": 72,
        "negative_violations": 0,
        "training_eligible": 0,
    }
    assert report["raw_marker_hits"] == 0
    assert all(value is False for value in report["promotion"].values())
    first_bytes = output.read_bytes()
    build(output)
    assert output.read_bytes() == first_bytes

    rows = report["rows"]
    assert sum(item["strict_valid"] for item in rows) == 136
    assert sum(not item["strict_valid"] for item in rows) == 8
    assert sum(validate_pg331_source_row(item["source_row"])["valid"] for item in rows) == 136
    assert {item["source_row"]["split"] for item in rows} == {"train", "implementation_holdout"}
    assert {item["source_row"]["source_meta"]["implementation"] for item in rows} == {
        "pg388_logic_surface_c",
        "pg388_logic_surface_d",
    }


def test_js_is_semantic_projection_and_axes_are_not_collapsed(tmp_path: Path) -> None:
    report = build(tmp_path / "surface.json")
    rows = report["rows"]

    contexts = {tuple(item["source_row"]["context_tokens"]) for item in rows}
    targets = {tuple(item["source_row"]["target_tokens"]) for item in rows}
    assert len(contexts) >= 12
    assert len(targets) >= 4

    for item in rows:
        row = item["source_row"]
        overlay = row.get("javascript_context_overlay")
        if item["strict_valid"]:
            assert overlay["source_text_stored"] is False
            assert overlay["javascript_context"]["persistent_state"] is False
            assert overlay["javascript_context"]["dynamic_code"] is False
        else:
            assert overlay is None
            assert row["target_projection"]["question"] == "ask_typed"
            assert row["axis_presence"]["document_presence"] == "not_observed"
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
        assert row["training_eligible"] is False
        for key in ("training_eligible", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
            assert row["promotion"][key] is False
        assert row["promotion"]["cross_source_review_required"] is True
        serialized = json.dumps(row, ensure_ascii=False).casefold()
        assert "http://" not in serialized
        assert "https://" not in serialized
        assert "payload=" not in serialized
        assert "wire=" not in serialized
        assert "response_body=" not in serialized


def test_field_manifest_keeps_observed_and_absent_distinct(tmp_path: Path) -> None:
    report = build(tmp_path / "surface.json")
    rows = report["rows"]
    for item in rows:
        manifest = item["source_row"]["field_capture_manifest"]
        statuses = {status for section in manifest.values() for status in section.values()}
        assert "observed" in statuses
        assert "absent" in statuses
        if item["strict_valid"]:
            assert "unknown" not in statuses
            assert "not_observed" not in statuses
        else:
            assert "not_observed" in statuses
