import json
from pathlib import Path

from app.cross_lab_safe_catalog import validate_sample


ROOT = Path(__file__).resolve().parents[1]


def test_pg25d_catalog_is_bounded_and_evaluation_only():
    registry = json.loads(
        (ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (ROOT / "research" / "pg_pk_25d_vulnerableapp_catalog_v1.json").read_text(encoding="utf-8")
    )
    assert catalog["training_eligible"] is False
    assert catalog["safety"]["raw_body_stored"] is False
    assert catalog["safety"]["attack_string_stored"] is False
    assert len(catalog["samples"]) == 3
    assert all(row["decision"]["evidence_status"] == "confirmed_negative" for row in catalog["samples"])
    assert all(row["decision"]["training_action"] == "abstain" for row in catalog["samples"])

    target_id = catalog["source"]["target_id"]
    assert not any(item["target_id"] == target_id and item.get("training_eligible") for item in registry["targets"])
    for row in catalog["samples"]:
        validate_sample(row, catalog["source"])

    raw = (ROOT / "research" / "pg_pk_25d_vulnerableapp_catalog_v1.json").read_text(encoding="utf-8")
    for forbidden in ("PG25_CANARY", '"raw_body":', '"response_body":', '"request_body":'):
        assert forbidden not in raw
