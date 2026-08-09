import json
from pathlib import Path

from app.hardcore_experiment_gate import evaluate_hardcore_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_clickjacking_head_catalog_is_preflight_not_a_hardcore_pass():
    catalog = json.loads(
        (ROOT / "research" / "pg_pk_25d_clickjacking_catalog_v1.json").read_text(encoding="utf-8")
    )
    report = evaluate_hardcore_catalog(catalog, family="clickjacking", required_methods={"HEAD", "GET"})
    assert report["status"] == "preflight_only"
    assert report["training_allowed"] is False
    assert "insufficient_target_instances" in report["reasons"]
    assert "insufficient_sampling_seeds" in report["reasons"]


def test_payload_family_requires_real_get_and_post():
    catalog = json.loads(
        (ROOT / "research" / "pg_pk_25d_vulnerableapp_catalog_v1.json").read_text(encoding="utf-8")
    )
    report = evaluate_hardcore_catalog(catalog, family="xss")
    assert report["status"] == "preflight_only"
    assert report["required_methods"] == ["GET", "POST"]
    assert "missing_required_http_method" in report["reasons"]
    assert report["memory_promotion_allowed"] is False
