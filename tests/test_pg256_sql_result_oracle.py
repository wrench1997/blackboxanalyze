from types import SimpleNamespace
import json
from pathlib import Path

from app.pg256_sql_result_oracle import evaluate_widebyte_effect, project_widebyte_response


ROUTE = {"path": "/vul/sqli/sqli_widebyte.php", "method": "POST"}
RESET = {
    "fresh_target": True,
    "completed": True,
    "container_recreated": True,
    "container_restart_used": False,
    "volume_mount_count": 0,
    "database_health_gate": "mysqli_root_pikachu_ok",
    "state_change_allowed": False,
    "external_network": False,
}


def _response(body: str, status: int = 200):
    return SimpleNamespace(text=body, content=body.encode(), status_code=status)


def _projection(rows: int, label: str):
    return project_widebyte_response(_response("your uid:" * rows), label=label)


def test_projection_caps_rows_and_discards_body():
    result = project_widebyte_response(_response("your uid:" * 40), label="candidate")
    projection = result["response_projection"]
    assert projection["row_count_capped"] == 16
    assert result["raw_response_retained"] is False
    assert "body" not in result


def test_widebyte_effect_requires_candidate_reference_negative_and_fresh_reset():
    result = evaluate_widebyte_effect(
        route=ROUTE,
        baseline=_projection(1, "baseline"),
        candidate=_projection(7, "candidate"),
        reference=_projection(7, "reference"),
        negative=_projection(0, "negative"),
        reset=RESET,
        source_hash="a" * 64,
        candidate_class="widebyte_escape_boundary",
        reference_class="widebyte_escape_boundary",
    )
    assert result["confirmed_positive"] is True
    assert result["reasons"] == []
    assert len(result["evidence_hash"]) == 64


def test_widebyte_effect_abstains_on_mismatched_candidate_or_negative():
    result = evaluate_widebyte_effect(
        route=ROUTE,
        baseline=_projection(1, "baseline"),
        candidate=_projection(1, "candidate"),
        reference=_projection(7, "reference"),
        negative=_projection(1, "negative"),
        reset=RESET,
        source_hash="b" * 64,
        candidate_class="syntax_boundary",
        reference_class="widebyte_escape_boundary",
    )
    assert result["confirmed_positive"] is False
    assert "candidate_row_count_not_above_baseline" in result["reasons"]
    assert "negative_control_returned_rows" in result["reasons"]
    assert "candidate_class_not_widebyte_escape_boundary" in result["reasons"]


def test_pg256_report_and_rule_are_current_and_promotion_blocked():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "research" / "pg256_pikachu_widebyte_oracle_report_v1.json").read_text(encoding="utf-8"))
    rules = json.loads((root / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    policy = rules["pg256_pikachu_widebyte_oracle"]
    assert report["counts"]["confirmed_positive_count"] == 3
    assert report["counts"]["false_positive_count"] == 0
    assert report["promotion"]["training_eligible"] is False
    assert policy["confirmed_positive_count"] == report["counts"]["confirmed_positive_count"]
    assert policy["vulnerability_claim_allowed"] is False
