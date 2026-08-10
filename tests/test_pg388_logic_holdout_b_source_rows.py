from __future__ import annotations

from app.pg331_source_row import validate_pg331_source_row
from scripts.run_pg388_logic_holdout_b_source_rows import collect


def test_holdout_b_materializes_full_candidate_only_matrix_in_process() -> None:
    report, rows, sidecars = collect()
    assert report["status"] == "completed_logic_rule_ir_source_rows_candidate_only"
    assert report["implementation_id"] == "pg388-logic-lab-backend-b"
    assert report["counts"] == {"expected": 140, "source_rows": 140, "strict_valid": 140, "typed": 140, "fresh_resets": 140, "failure_repair": 56, "negative_violations": 0}
    assert len(rows) == len(sidecars) == 140
    assert report["execution"]["local_fixture_in_process"] is True
    assert report["execution"]["docker_started"] is False
    assert report["source_contract"]["image_attested"] is False
    assert report["training_eligible"] == 0
    assert validate_pg331_source_row(rows[0]["source_row"])["valid"] is True


def test_holdout_b_rows_keep_raw_and_promotion_boundaries_closed() -> None:
    report, rows, sidecars = collect()
    assert all(row["training_eligible"] is False for row in rows)
    assert all(item["operator_reviewed"] is False for item in sidecars)
    assert report["execution"]["external_network"] is False
    assert report["execution"]["wire_created"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
