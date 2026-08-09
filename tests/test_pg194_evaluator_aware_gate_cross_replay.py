import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg194_evaluator_aware_gate_cross_replay_report_v1.json"
TRACE = ROOT / "research" / "pg194_evaluator_aware_gate_cross_replay_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg194_evaluator_aware_gate_cross_replay_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg194_gate_is_calibrated_on_complete_evidence_table() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)

    assert report["status"] == "completed_cross_seed_container_evaluator_aware_replay"
    assert report["model"]["variant"] == "xxl"
    assert report["model"]["parameter_count"] > 100_000_000
    assert report["model"]["online_weight_update"] is False

    training = report["gate_training"]
    assert training["train_rows"] == 46
    assert training["holdout_rows"] == 32
    assert training["holdout"]["accuracy"] == 1.0
    assert training["holdout"]["allow_candidate_recall"] == 1.0
    assert training["holdout"]["unsafe_allow_count"] == 0
    assert protocol["evidence_keys"] == [
        "typed_available",
        "negative_control",
        "fresh_reset",
        "evidence_hash",
        "effect_present",
    ]
    assert protocol["gate_holdout_rows"] == 32


def test_pg194_dom_replay_requires_evaluator_handoff_on_three_fresh_containers() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)

    dom_runs = report["dom_runs"]
    assert len(dom_runs) == 3
    assert len({row["seed"] for row in dom_runs}) == 3
    assert all(row["fresh_container"] for row in dom_runs)
    assert all(row["typed_oracle_available"] for row in dom_runs)
    assert all(row["typed_surface_effect"] for row in dom_runs)
    assert all(row["confirmed_positive"] is False for row in dom_runs)
    assert all(row["vulnerability_claim_allowed"] is False for row in dom_runs)
    assert all(
        any(step["controller_decision"] == "send_evaluator_aware_dom_candidate" for step in row["steps"])
        for row in dom_runs
    )
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False


def test_pg194_sql_variants_are_typed_effects_not_pikachu_vulnerability_claims() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)

    sql_runs = report["sql_runs"]
    assert len(sql_runs) == 3
    assert {row["variant"] for row in sql_runs} == {"alpha", "beta", "gamma"}
    assert all(row["fresh_target"] for row in sql_runs)
    assert all(row["typed_positive_count"] == 1 for row in sql_runs)
    assert all(row["vulnerability_claim_allowed"] is False for row in sql_runs)
    assert protocol["typed_oracle_required"] is True
    assert protocol["raw_payload_and_response_excluded"] is True
    assert protocol["training_promotion_allowed"] is False
    assert protocol["memory_promotion_allowed"] is False


def test_pg194_keeps_raw_material_quarantined_and_rule_frozen() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    rules = _load(ROOT / "research" / "improvement_rules.json")

    assert report["counts"] == {
        "dom_container_count": 3,
        "dom_effect_count": 3,
        "dom_false_positive_count": 0,
        "sql_variant_count": 3,
        "sql_typed_positive_count": 3,
        "sql_false_positive_count": 0,
    }
    for section in (report["promotion"], report["safety"]):
        assert section["raw_payload_strings_stored"] is False
        assert section["raw_response_bodies_stored"] is False
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False

    rule = rules["pg194_evaluator_aware_gate_cross_replay"]
    assert rule["gate_holdout"] == {
        "accuracy": 1.0,
        "allow_candidate_recall": 1.0,
        "unsafe_allow_count": 0,
    }
    assert rule["dom_cross_seed_runs"] == 3
    assert rule["sql_cross_variant_runs"] == 3
    assert rule["false_positive_count"] == 0
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
    assert rule["vulnerability_claim_allowed"] is False
