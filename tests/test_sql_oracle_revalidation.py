from app.maze_engine import sha256_json
from app.sql_oracle_revalidation import revalidate_sql_pair


def _row(variant: str, *, positive: bool = True, candidate: str = "injection") -> dict:
    source_hash = "b" * 64
    projection = {"controlled_differential": positive, "interpreter_boundary": positive, "modality": "bounded_timing", "database_touched": False, "real_sleep_performed": False, "external_network": False}
    evidence = {"reset": {"fixture_source_sha256": source_hash, "fresh_target": True}, "oracle_projection": projection, "database_touched": False, "real_sleep_performed": False, "external_network": False}
    evidence["evidence_hash"] = sha256_json(evidence)
    return {
        "candidate_family": candidate,
        "semantic": {"expected_oracle": "synthetic_sql_ast_differential_v1"},
        "oracle_projection": projection,
        "rule_ir_result": positive,
        "evidence": evidence,
        "pair": {"pair_id": "sql-pair-01", "variant": variant},
    }


def test_sql_revalidation_requires_safe_differential_pair():
    result = revalidate_sql_pair(
        [_row("plain"), _row("url_percent")],
        authorized_source_hash="b" * 64,
        oracle_name="synthetic_sql_ast_differential_v1",
    )
    assert result["accepted"] is True
    assert result["modalities"] == ["bounded_timing"]


def test_sql_revalidation_fails_closed_for_plain_or_model_mismatch():
    result = revalidate_sql_pair(
        [_row("plain", positive=False), _row("url_percent", candidate="xss")],
        authorized_source_hash="b" * 64,
        oracle_name="synthetic_sql_ast_differential_v1",
    )
    assert result["accepted"] is False
    assert "model_family_disagreement" in result["reasons"]
    assert "controlled_differential_missing" in result["reasons"]


def test_sql_revalidation_rejects_projection_tampering_after_evidence_hash():
    rows = [_row("plain"), _row("url_percent")]
    rows[0]["oracle_projection"] = dict(rows[0]["oracle_projection"], controlled_differential=False)
    result = revalidate_sql_pair(
        rows,
        authorized_source_hash="b" * 64,
        oracle_name="synthetic_sql_ast_differential_v1",
    )
    assert result["accepted"] is False
    assert "oracle_projection_not_bound_to_evidence" in result["reasons"]
