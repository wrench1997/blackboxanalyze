from __future__ import annotations

import json
from pathlib import Path

from app.pg331_source_row import validate_pg331_source_row
from scripts.collect_pg348_dynamic_typed_rows import collect
from app.pg348_dynamic_runtime import load_registry


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_typed_rows_have_role_triplet_failure_repair_and_clean_context() -> None:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    dataset, sidecars, report = collect(registry, operator_reviewed=True, max_records=2)
    assert report["status"] == "completed_typed_operator_reviewed_candidate"
    assert dataset["counts"]["records"] == 10
    assert dataset["counts"]["training_eligible_rows"] == 10
    assert dataset["counts"]["failure_rows"] == 2
    assert sidecars["counts"]["confirmed_positive"] == 2
    for row in dataset["records"]:
        result = validate_pg331_source_row(row, require_training_eligible=True)
        assert result["valid"], result["failures"]
        assert row["context_firewall"]["forbidden_token_count"] == 0
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert row["oracle_answer_in_context"] is False
        assert row["target_projection"]["payload_shape_ref"].endswith("_marker")
        assert any(str(token).startswith("payload_shape_ref=") for token in row["target_tokens"])
    failure_rows = [row for row in dataset["records"] if "question=ask_failure" in row["target_tokens"]]
    assert len(failure_rows) == 2
    assert all("failure_action_not_changed" not in row["failures"] for row in failure_rows)


def test_dynamic_typed_sidecar_never_promotes_payload_or_vulnerability() -> None:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    dataset, sidecars, _ = collect(registry, operator_reviewed=True, max_records=1)
    assert dataset["promotion"]["training_allowed"] is False
    assert dataset["promotion"]["payload_catalog_promotion_allowed"] is False
    assert dataset["promotion"]["vulnerability_claim_allowed"] is False
    sidecar = sidecars["sidecars"][0]["sidecar"]
    assert sidecar["confirmed_positive"] is True
    assert sidecar["raw_payload_stored"] is False
    assert sidecar["raw_response_stored"] is False
    assert sidecar["oracle_answer_in_context"] is False
