import json
from pathlib import Path

from app.pg180_process_action_model import ALLOWED_ACTIONS, abstract_step_tokens


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg180_process_action_model_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg180_process_action_model_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg180_process_action_model_protocol_v1.json"
SOURCE_TRACE_PATH = ROOT / "research" / "pg179b_pikachu_iterative_trace_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg180_capacity_sweep_is_cross_surface_and_process_only() -> None:
    report = _load(REPORT_PATH)
    trace = _load(TRACE_PATH)
    assert report["status"] == "completed_process_policy_diagnostic"
    assert report["dataset"]["row_count"] == 35
    assert report["dataset"]["episode_count"] == 7
    assert set(report["capacity"]) == {"small", "medium", "moe_large"}
    assert report["dataset"]["excluded_from_model_input"]
    assert all(item["all_safe_action_allowlisted"] for item in report["capacity"].values())
    assert all(item["all_input_contracts_clean"] for item in report["capacity"].values())
    assert all(item["parameter_count"] > 0 for item in report["capacity"].values())
    assert all(item["safe_action_not_allowlisted_count"] == 0 for run in report["runs"] for item in [run["test"]])
    assert all(item["test"]["abstain_recall"] >= 0.0 for item in report["runs"])
    assert report["selection"]["selected_variant"] is None
    assert report["selection"]["promotion_allowed"] is False
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert len(trace["runs"]) == 18


def test_pg180_model_artifacts_do_not_retain_raw_probe_or_response_values() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_PATH, TRACE_PATH, PROTOCOL_PATH))
    lowered = serialized.casefold()
    for forbidden in ("<script", "union select", "sleep(", "benchmark(", "javascript:", "onerror", "pg179b-canary-a1", "pg179b-control-a1", "pg179b-baseline-a1"):
        assert forbidden not in lowered
    report = _load(REPORT_PATH)
    assert report["safety"]["raw_probe_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert report["safety"]["oracle_in_model_input"] is False
    assert report["safety"]["family_in_model_input"] is False
    assert report["safety"]["vulnerability_labels_in_model_input"] is False
    for run in report["runs"]:
        assert run["raw_input_retained"] is False
        assert run["oracle_in_input"] is False
        assert run["family_in_input"] is False
        assert run["route_in_input"] is False
        assert run["vulnerability_label_in_input"] is False


def test_pg180_abstract_projection_excludes_route_and_family() -> None:
    trace = _load(SOURCE_TRACE_PATH)
    step = trace["steps"][0]
    tokens = abstract_step_tokens(step)
    joined = " ".join(tokens).casefold()
    assert "path::" not in joined
    assert "route::" not in joined
    assert "family::" not in joined
    assert "raw" not in joined
    assert "body" not in joined
    assert "oracle_projection" not in joined
    assert all(token.startswith(("method::", "placement::", "encoding::", "failure::", "gate::", "candidate::", "typed_available::", "authority::", "status::", "shape::", "status_chain_len::", "round::", "budget::", "methods_seen::", "field_count::", "belief_top::", "history_len::")) for token in tokens)
    assert set(ALLOWED_ACTIONS) == {"repeat_matched_negative_pair", "abstain_unknown_oracle", "probe_candidate_other_method"}


def test_pg180_protocol_requires_leave_surface_out_and_no_promotion() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["required_gates"]["leave_surface_out"] is True
    assert protocol["required_gates"]["family_and_route_excluded_from_input"] is True
    assert protocol["required_gates"]["raw_probe_and_response_excluded"] is True
    assert protocol["required_gates"]["single_channel_probe_must_be_blocked"] is True
    assert protocol["required_gates"]["promotion_allowed"] is False
    assert protocol["required_gates"]["memory_promotion_allowed"] is False
