import json
from pathlib import Path

import pytest

from app.pg179b_iterative_probe import action_manifest, surface_oracle, validate_marker


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "research" / "pg179b_pikachu_iterative_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg179b_pikachu_iterative_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg179b_pikachu_iterative_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg179b_pikachu_iterative_probe_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg179b_artifacts_are_real_get_post_failure_guided_traces() -> None:
    trace = _load(TRACE_PATH)
    report = _load(REPORT_PATH)
    catalog = _load(CATALOG_PATH)

    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["episode_count"] == 7
    assert trace["accepted_evaluation_episode_count"] == 7
    assert len(trace["steps"]) == 35
    assert {step["action_manifest"]["method"] for step in trace["steps"]} == {"GET", "POST"}
    assert sum(step["action_manifest"]["method"] == "GET" for step in trace["steps"]) == 11
    assert sum(step["action_manifest"]["method"] == "POST" for step in trace["steps"]) == 24
    assert all(step["decision"] == "abstain" for step in trace["steps"])
    assert all(step["online_weight_update"] is False for step in trace["steps"])
    assert all(step["long_term_memory_write"] is False for step in trace["steps"])
    assert all(step["failure_signature"]["next_action"] == step["next_action"] for step in trace["steps"])
    assert all(step["fresh_reset"]["fresh_target"] and step["fresh_reset"]["completed"] for step in trace["steps"])
    assert all("status_chain" in step["response_projection"] for step in trace["steps"])
    assert all("redirect_chain" in step["response_projection"] for step in trace["steps"])
    assert all(step["response_projection"].get("status_chain_sha256") for step in trace["steps"])
    assert trace["adaptive_branch_counts"]["probe_candidate_other_method"] == 0
    assert trace["adaptive_branch_counts"]["repeat_matched_negative_pair"] == 6
    assert trace["adaptive_branch_counts"]["abstain_unknown_oracle"] == 1
    assert trace["parameterized_channels"] == {"GET": 1, "POST": 6}
    assert trace["dual_channel_episode_count"] == 6
    assert trace["invented_parameter_names"] is False
    assert report["trace"]["failure_guided_branch_observed"] is True
    assert report["oracle"]["positive_count"] == 0
    assert report["oracle"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert catalog["training_eligible"] is False
    assert catalog["raw_probe_strings_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert catalog["channel_grounding"]["invented_parameter_names"] is False
    assert len(catalog["samples"]) == 35
    assert all(sample["decision"]["training_action"] == "abstain" for sample in catalog["samples"])
    assert any(row["parameterized_method"] == "GET" and row["get_fields"] == ["url"] for row in catalog["route_rows"])
    assert all(row["post_fields"] for row in catalog["route_rows"] if row["parameterized_method"] == "POST")


def test_pg179b_artifacts_do_not_contain_exploit_syntax_or_canary_values() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (CATALOG_PATH, TRACE_PATH, REPORT_PATH))
    lowered = serialized.casefold()
    for forbidden in ("<script", "union select", "sleep(", "benchmark(", "javascript:", "<img", "onerror"):
        assert forbidden not in lowered
    for raw_marker in ("pg179b-canary-a1", "pg179b-control-a1", "pg179b-baseline-a1", "pg179b-control-b1", "pg179b-canary-b1"):
        assert raw_marker not in serialized
    for step in _load(TRACE_PATH)["steps"]:
        action = step["action_manifest"]
        assert "probe_ref" in action and "probe_sha256" in action
        assert "marker" not in action
        assert "raw_probe" not in step
        assert "raw_body" not in step


def test_pg179b_protocol_keeps_unknown_oracle_as_abstain() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["scope"]["network"] == "127.0.0.1_only"
    assert protocol["scope"]["fresh_container_per_episode"] is True
    assert protocol["episode_contract"]["failure_signature_required"] is True
    assert protocol["episode_contract"]["belief_update_required"] is True
    assert protocol["episode_contract"]["adaptive_next_action_required"] is True
    assert protocol["grounding"]["invented_parameter_names_forbidden"] is True
    assert protocol["episode_contract"]["channel_contract"]["dual_channel_claim_requires_both_observed"] is True
    assert protocol["oracle_contract"]["positive_requires_typed_effect"] is True
    assert protocol["oracle_contract"]["unknown_or_surface_signal_action"] == "abstain"
    assert protocol["promotion_gate"]["training_allowed"] is False
    assert protocol["promotion_gate"]["long_term_memory_allowed"] is False


def test_pg179b_surface_signal_can_never_be_typed_positive() -> None:
    with pytest.raises(ValueError):
        validate_marker("<script>alert(1)</script>")
    with pytest.raises(ValueError):
        validate_marker("union select")
    oracle = surface_oracle(
        family="xss",
        method="GET",
        signal={"candidate_signal": True, "marker_reflected": True},
        oracle_contract_sha256="a" * 64,
    )
    assert oracle["candidate_signal"] is True
    assert oracle["positive"] is False
    assert oracle["positive_authority"] is False
    assert oracle["confirmed_effect"] == "none"


def test_pg179b_action_manifest_is_bound_and_validated_before_send() -> None:
    manifest = action_manifest(
        path="/vul/xss/xss_stored.php",
        surface="xss_stored_post",
        family="xss",
        method="POST",
        field_names=["message", "submit"],
        probe_role="candidate",
        marker="pg179b-test-canary",
    )
    assert manifest["method"] == "POST"
    assert manifest["form_field_names"] == ["message", "submit"]
    assert manifest["manifest_sha256"]
    assert manifest["payload_sha256"]
    assert "marker" not in manifest

    with pytest.raises(ValueError):
        action_manifest(
            path="/vul/xss/xss_stored.php",
            surface="xss_stored_post",
            family="xss",
            method="POST",
            field_names=["message"],
            probe_role="candidate",
            marker="<script>",
        )
