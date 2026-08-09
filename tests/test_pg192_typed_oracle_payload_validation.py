import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg192_typed_oracle_payload_validation_report_v1.json"
TRACE = ROOT / "research" / "pg192_typed_oracle_payload_validation_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg192_typed_oracle_payload_validation_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg192_confirms_only_the_controlled_loopback_redirect_effect() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_typed_redirect_and_unknown_oracle_replay"
    assert report["counts"]["typed_positive_count"] == 1
    assert report["counts"]["confirmed_positive_count"] == 1
    assert report["counts"]["vulnerability_claim_allowed_count"] == 1
    redirect = next(row for row in report["runs"] if row["surface"] == "pg192_urlredirect")
    assert redirect["typed_oracle_available"] is True
    assert redirect["confirmed_positive"] is True
    assert any(step["controller_decision"] == "send_typed_oracle_active_candidate" for step in redirect["steps"])
    positive = next(step for step in redirect["steps"] if step.get("confirmed_positive"))
    assert positive["typed_oracle"]["positive"] is True
    assert positive["typed_oracle"]["positive_authority"] is True
    assert positive["typed_oracle"]["confirmed_effect"] == "redirect_origin"
    assert protocol["negative_control_required"] is True
    assert protocol["evidence_hash_required"] is True


def test_pg192_keeps_dom_and_sql_without_typed_evaluator_in_abstain() -> None:
    report = _load(REPORT)
    for surface in ("dom_unknown", "sql_unknown"):
        run = next(row for row in report["runs"] if row["surface"] == surface)
        assert run["typed_oracle_available"] is False
        assert run["confirmed_positive"] is False
        assert all(step["controller_decision"] == "abstain" for step in run["steps"])
        assert all(step["vulnerability_claim_allowed"] is False for step in run["steps"])


def test_pg192_positive_manifest_is_hash_bound_and_non_executing() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    redirect = next(row for row in report["runs"] if row["surface"] == "pg192_urlredirect")
    positive = next(step for step in redirect["steps"] if step.get("confirmed_positive"))
    manifest = positive["action_manifest"]
    assert manifest["probe_ref"] == "pg192-controlled-redirect-pg192_urlredirect"
    assert manifest["payload_sha256"]
    assert manifest["manifest_sha256"]
    assert manifest["safety"]["does_not_execute"] is True
    assert manifest["safety"]["no_external_network"] is True
    assert positive["evidence"]["target_instance_hash"] == redirect["target_instance_hash"]
    assert positive["long_term_memory_write"] is False
    assert trace["training_eligible"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False


def test_pg192_rule_requires_typed_positive_negative_control_and_fresh_reset() -> None:
    rules = _load(ROOT / "research" / "improvement_rules.json")
    rule = rules["pg192_typed_oracle_payload_validation"]
    assert rule["controlled_destination"] == "127.0.0.1:8767 only"
    assert rule["confirmed_positive_gate"] == ["typed_oracle_positive", "matched_negative_control", "fresh_container", "evidence_hash"]
    assert rule["model_abstain_visible_during_typed_handoff"] is True
    assert rule["raw_payload_strings_stored"] is False
    assert rule["external_network"] is False
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
