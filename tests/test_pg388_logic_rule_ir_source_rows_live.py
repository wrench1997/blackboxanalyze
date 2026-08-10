from __future__ import annotations

import json

import pytest

from app.pg331_source_row import validate_pg331_source_row
from scripts import run_pg388_logic_rule_ir_source_rows_live as live


def _page_request(_base: str, _path: str = "/pg388", *, timeout: float = 5.0) -> str:
    _ = timeout
    return "<!doctype html><html lang='en'><head><title>PG388</title><meta charset='utf-8'><script>const x=1;</script></head><body><main data-transport-method='GET' data-parameter-role='display_text' data-encoding-chain='identity'>Logic</main></body></html>"


def _request(_base: str, path: str, *, method: str = "GET", payload=None, timeout: float = 5.0):
    _ = method, timeout
    if path == "/health":
        return {"status": "ok"}
    if path == "/api/manifest":
        return {"status": "dynamic_fixture_only_unbound", "implementation_id": "pg388-local-display", "case_count": 28}
    if path == "/api/reset":
        return {"status": "fresh_reset", "state_clean": True}
    case_ref = payload["case_ref"]
    role = payload["role"]
    phase = payload["phase"]
    violation = role in {"candidate", "replay"} and phase != "baseline" and case_ref in {"nonce_replay", "purchase_price_binding", "execution_order"}
    return {
        "status": "typed_local_canary_result",
        "case_ref": case_ref,
        "role": role,
        "phase": phase,
        "state_before": "zero_effect",
        "state_after": "one_or_more_effects" if violation else "zero_effect",
        "state_delta": "abstract_delta" if violation else "zero",
        "effect_shape": "abstract_shape",
        "action_shape": "abstract_action",
        "invariant_holds": not violation,
        "vulnerable_effect": violation,
        "typed_observation": True,
        "negative_control_clean": role == "negative",
        "safe_to_send": False,
        "target_contacted": False,
        "external_network": False,
        "persistent_storage": False,
        "fresh_reset_required": True,
    }


@pytest.fixture(scope="module")
def matrix():
    return live.run(environ={"PG388_LOCAL_EVAL": "1"}, request=_request, page_request=_page_request)


def test_logic_rule_ir_live_is_explicit_and_row_bound(matrix):
    report, rows, sidecars = matrix
    assert report["status"] == "completed_logic_rule_ir_source_rows_candidate_only"
    assert report["counts"] == {"expected": 140, "source_rows": 140, "strict_valid": 140, "typed": 140, "fresh_resets": 140, "failure_repair": 6, "negative_violations": 0}
    assert report["source_contract"]["row_bound_typed_evidence"] is True
    assert report["source_contract"]["operator_reviewed"] is False
    assert report["training_eligible"] == 0
    assert len(rows) == len(sidecars) == 140
    assert all(row["strict_valid"] for row in rows)
    for index in (0, 1, 27, 28, 139):
        assert validate_pg331_source_row(rows[index]["source_row"])["valid"] is True
    assert all(len(row["logic_rule_ir_target_tokens"]) == 13 for row in rows)


def test_logic_rule_ir_live_projection_has_no_raw_or_evaluator_answer(matrix):
    report, rows, sidecars = matrix
    # The report's local base_url is an explicit loopback execution field;
    # the model-facing rows/sidecars must remain literal-free.
    serialized = json.dumps({"rows": rows, "sidecars": sidecars}, ensure_ascii=False).casefold()
    for marker in ("http://", "https://", "payload=", "wire=", "response_body=", "oracle_answer=", "evaluator_answer="):
        assert marker not in serialized
    assert all(row["source_row"]["target_projection"]["safe_to_send"] is False for row in rows)
    assert all(item["operator_reviewed"] is False for item in sidecars)


def test_logic_rule_ir_live_is_blocked_without_flag():
    report, rows, sidecars = live.run(environ={})
    assert report["status"] == "planning_only_live_blocked"
    assert rows == []
    assert sidecars == []
