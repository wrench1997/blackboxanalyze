import json
from pathlib import Path

from app.pg223_large_problem_diagnoser import LargeProblemDiagnoserAdapter


ROOT = Path(__file__).resolve().parents[1]


def test_pg223_adapter_is_large_context_plus_small_trainable_head() -> None:
    model = LargeProblemDiagnoserAdapter(d_model=1024, hidden_dim=64)
    assert sum(parameter.numel() for parameter in model.parameters()) > 70_000
    assert model.diagnosis_head.out_features >= 9


def test_pg223_report_records_frozen_xxl_and_keeps_positive_gate_closed() -> None:
    report = json.loads((ROOT / "research" / "pg223_large_problem_diagnoser_report_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_frozen_xxl_problem_diagnoser_capacity_sweep"
    assert report["frozen_parameter_count"] > 100_000_000
    assert report["selected"]["holdout"]["guarded_positive_false_accept_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["payload_generation"] is False
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False

