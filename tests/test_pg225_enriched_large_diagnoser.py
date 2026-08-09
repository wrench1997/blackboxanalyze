import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg225_adds_real_pg224_rows_without_turning_weak_signals_positive() -> None:
    report = json.loads((ROOT / "research" / "pg225_enriched_large_diagnoser_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg225_enriched_large_diagnoser_dataset_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_real_surface_enriched_large_diagnoser"
    assert report["row_counts"]["pg224_real_rows"] == 88
    assert report["pg224_real_label_counts"]["oracle_unavailable"] > 0
    assert report["selected"]["holdout"]["guarded_positive_false_accept_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert dataset["contract"]["real_pg224_rows_are_projection_only"] is True
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["rows"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["rows"])

