import json
from pathlib import Path

from app.pg208_parameter_catalog import build_parameter_catalog


ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / "research" / "pg179_pikachu_browser_crawl_manifest_v1.json").read_text(encoding="utf-8-sig"))


def test_pg208_catalog_preserves_all_crawl_surfaces_and_real_fields() -> None:
    catalog = build_parameter_catalog(_load())
    assert catalog["source_request_surface_count"] == 112
    assert catalog["unique_route_entry_count"] == 112
    assert catalog["active_replay_eligible_count"] == 15
    assert catalog["excluded_count"] == 97
    assert any(item["method"] == "GET" and item["fields"] == ["message", "submit"] for item in catalog["entries"])
    assert any(item["method"] == "POST" and item["fields"] == ["id", "submit"] for item in catalog["entries"])
    assert any(item["typed_oracle"] == "unknown_sql_backend" for item in catalog["entries"])


def test_pg208_catalog_blocks_secrets_and_stateful_routes() -> None:
    catalog = build_parameter_catalog(_load())
    blocked = {item["path"] for item in catalog["excluded_entries"]}
    assert "/vul/xss/xss_stored.php" in blocked
    assert "/vul/xss/xssblind/xss_blind.php" in blocked
    assert all(item["training_eligible_before_pg208"] is False for item in catalog["entries"])
    assert catalog["raw_request_values_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False

