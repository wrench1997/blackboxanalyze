from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg146_real_loopback_replay_has_balanced_channels_and_unknown_oracle() -> None:
    report = json.loads(Path("research/pg146_public_lab_replay_report_v1.json").read_text(encoding="utf-8"))
    counts = report["counts"]
    assert counts["target_count"] == 3
    assert counts["row_count"] == 6
    assert counts["get_count"] == counts["post_count"] == 3
    assert counts["fresh_reset_count"] == 6
    assert counts["unknown_oracle_count"] == 6
    assert counts["typed_oracle_count"] == 0
    # A transport failure is an environment failure.  It must make the hard
    # gate red instead of being silently converted into a model negative.
    assert report["hard_gates_passed"] is (report["counts"]["ready_count"] == 6)
    assert report["hard_checks"]["all_target_surfaces_ready"] is (report["counts"]["ready_count"] == 6)
    assert report["training_eligible"] is False
    assert report["model_input_contract"]["raw_response_body_in_model"] is False


def test_pg146_catalog_keeps_projection_hashes_without_raw_bodies() -> None:
    catalog = json.loads(Path("research/pg146_public_lab_replay_catalog_v1.json").read_text(encoding="utf-8"))
    assert catalog["raw_request_bodies_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert len(catalog["rows"]) == 6
    for row in catalog["rows"]:
        assert len(row["evidence_hash"]) == 64
        assert len(row["response"]["projection"]["body_sha256"]) == 64
        assert row["oracle"]["availability"] == "unknown_oracle"
