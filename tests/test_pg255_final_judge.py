import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg255_ai_really_participated_in_get_post_send_path() -> None:
    report = _load("research/pg255_pikachu_fixed_sql_pg254_replay_report_v1.json")
    counts = report["counts"]
    assert counts["fresh_container_count"] == 14
    assert counts["get_episode_count"] == 10
    assert counts["post_episode_count"] == 4
    assert counts["ai_candidate_send_count"] == 14
    assert counts["reference_send_count"] == 14
    assert counts["database_health_gate_count"] == 14


def test_pg255_final_judge_separates_typed_effect_from_probe_send() -> None:
    report = _load("research/pg255_pikachu_fixed_sql_pg254_replay_report_v1.json")
    counts = report["counts"]
    assert counts["typed_effect_confirmed_count"] == 8
    assert counts["confirmed_positive_count"] == 8
    assert counts["false_positive_count"] == 0
    assert counts["ai_candidate_send_count"] > counts["confirmed_positive_count"]
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False


def test_pg255_persists_only_bounded_evidence_and_keeps_unconfirmed_families() -> None:
    report = _load("research/pg255_pikachu_fixed_sql_pg254_replay_report_v1.json")
    assert report["counts"]["ephemeral_wire_count"] == 28
    assert report["safety"]["raw_wires_stdout_only"] is True
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    by_route = {(row["method"], row["path"]): row for row in report["episodes"]}
    assert by_route[("GET", "/vul/sqli/sqli_blind_b.php")]["confirmed_positive"] is False
    assert by_route[("GET", "/vul/sqli/sqli_blind_t.php")]["confirmed_positive"] is False
    assert by_route[("POST", "/vul/sqli/sqli_widebyte.php")]["confirmed_positive"] is False


def test_pg255_rule_is_registered_as_local_only_final_judge() -> None:
    rules = _load("research/improvement_rules.json")
    rule = rules["pg255_pikachu_fixed_sql_pg254_replay"]
    assert rule["final_judge"].startswith("passed_for_local_typed_effect_only")
    assert rule["training_eligible"] is False
    assert rule["vulnerability_claim_allowed"] is False
    assert rule["safety"]["raw_wires_persisted"] is False
