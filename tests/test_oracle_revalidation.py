from app.maze_engine import sha256_json
from app.oracle_revalidation import revalidate_positive_pair


def _row(variant: str, *, positive: bool = True, candidate: str = "xss") -> dict:
    source_hash = "a" * 64
    projection = {
        "sink_kind": "html_attribute" if positive else "none",
        "marker_in_attribute": positive,
        "marker_in_script_source": False,
    }
    evidence = {
        "reset": {"fixture_source_sha256": source_hash, "fresh_target": True},
        "local_http_loopback": True,
        "network_access": False,
        "oracle_projection": projection,
    }
    evidence["evidence_hash"] = sha256_json(evidence)
    return {
        "candidate_family": candidate,
        "semantic": {"expected_oracle": "fixture_inert_attribute_oracle_v1"},
        "evidence": evidence,
        "oracle_projection": projection,
        "rule_ir_result": positive,
        "pair": {
            "pair_id": "pair-01",
            "variant": variant,
            "surface_role": "reflected_attribute",
        },
    }


def test_revalidation_accepts_only_pinned_pair_with_evidence():
    result = revalidate_positive_pair(
        [_row("plain"), _row("url_percent")],
        expected_family="xss",
        oracle_name="fixture_inert_attribute_oracle_v1",
        authorized_source_hash="a" * 64,
        required_sink_kind="html_attribute",
    )
    assert result["accepted"] is True
    assert result["reasons"] == []


def test_revalidation_fails_closed_for_negative_control_or_model_disagreement():
    result = revalidate_positive_pair(
        [_row("plain", positive=False), _row("url_percent", candidate="injection")],
        expected_family="xss",
        oracle_name="fixture_inert_attribute_oracle_v1",
        authorized_source_hash="a" * 64,
        required_sink_kind="html_attribute",
    )
    assert result["accepted"] is False
    assert "model_family_disagreement" in result["reasons"]
    assert "positive_oracle_not_satisfied" in result["reasons"]


def test_revalidation_rejects_projection_tampering_after_evidence_hash():
    rows = [_row("plain"), _row("url_percent")]
    rows[0]["oracle_projection"] = dict(rows[0]["oracle_projection"], marker_in_attribute=False)
    result = revalidate_positive_pair(
        rows,
        expected_family="xss",
        oracle_name="fixture_inert_attribute_oracle_v1",
        authorized_source_hash="a" * 64,
        required_sink_kind="html_attribute",
    )
    assert result["accepted"] is False
    assert "oracle_projection_not_bound_to_evidence" in result["reasons"]
