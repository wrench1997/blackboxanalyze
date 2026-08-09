import json
from pathlib import Path

from app.pg224_surface_projector import build_runtime_values, route_policy, wire_placeholder


ROOT = Path(__file__).resolve().parents[1]


def test_pg224_policy_blocks_unsafe_or_stateful_surfaces_but_allows_sql_post() -> None:
    assert route_policy("/vul/rce/rce_eval.php", "POST", ["txt", "submit"])["send_allowed"] is False
    assert route_policy("/vul/xss/xss_stored.php", "POST", ["message", "submit"])["send_allowed"] is False
    assert route_policy("/vul/sqli/sqli_id.php", "POST", ["id", "submit"])["send_allowed"] is True
    values = build_runtime_values(path="/vul/sqli/sqli_id.php", method="POST", fields=["id", "submit"], marker="pg224-test", probe_kind="sql_channel_class")
    assert values["submit"] == "submit"
    assert values["id"].startswith("1")


def test_pg224_wire_shape_is_human_readable_without_retaining_runtime_value() -> None:
    wire = wire_placeholder(path="/vul/sqli/sqli_id.php", method="POST", fields=["id", "submit"], probe_kind="sql_channel_class")
    assert wire.startswith("POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php")
    assert "<RUNTIME_SQL_SHAPE>" in wire


def test_pg224_report_has_all_crawled_surfaces_and_no_unverified_positive() -> None:
    report = json.loads((ROOT / "research" / "pg224_pikachu_parameter_surface_collection_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg224_pikachu_parameter_surface_dataset_v1.json").read_text(encoding="utf-8-sig"))
    counts = report["counts"]
    assert report["status"] == "completed_crawl_grounded_parameter_surface_projection"
    assert counts["route_inventory_count"] == 44
    assert counts["surface_observation_count"] == 88
    assert counts["safe_send_count"] == 30
    assert counts["get_candidate_send_count"] == 26
    assert counts["post_candidate_send_count"] == 4
    assert counts["preflight_only_count"] == 58
    assert counts["typed_effect_count"] == 0
    assert counts["false_positive_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["rows"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["rows"])
