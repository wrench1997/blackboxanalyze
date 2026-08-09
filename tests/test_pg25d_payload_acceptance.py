import json
from pathlib import Path

from app.pg25d_acceptance_gate import evaluate_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_pg25d_preflight_catalog_does_not_claim_payload_success():
    catalog = json.loads(
        (ROOT / "research" / "pg_pk_25d_vulnerableapp_catalog_v1.json").read_text(encoding="utf-8")
    )
    report = evaluate_catalog(catalog)
    assert report["status"] == "preflight_only_or_abstain"
    assert report["confirmed_positive_count"] == 0
    assert report["accepted_positive_count"] == 0
    assert report["confirmed_negative_count"] == 3
    assert report["training_eligible"] is False
    assert report["weak_signals_are_not_success"] is True
    assert report["raw_bodies_retained"] is False
