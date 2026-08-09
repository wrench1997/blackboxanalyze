"""PG-PK-22: offline integrity checks for evaluator oracle projections."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.logic_access_oracle import revalidate_logic_access_pair  # noqa: E402
from app.maze_engine import sha256_json  # noqa: E402
from app.oracle_revalidation import revalidate_positive_pair  # noqa: E402
from app.sql_oracle_revalidation import revalidate_sql_pair  # noqa: E402


PROTOCOL_PATH = ROOT / "research" / "pg_pk_22_oracle_projection_binding_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_22_oracle_projection_binding_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_22_oracle_projection_binding_v1.md"


def _evidence(source_hash: str, projection: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "reset": {"fixture_source_sha256": source_hash, "fresh_target": True, "state_change_allowed": False},
        "oracle_projection": copy.deepcopy(projection),
        "local_http_loopback": True,
        "database_touched": False,
        "real_sleep_performed": False,
        "external_network": False,
    }
    evidence["evidence_hash"] = sha256_json(evidence)
    return evidence


def _xss_rows() -> list[dict[str, Any]]:
    rows = []
    for variant in ("plain", "url_percent"):
        projection = {"sink_kind": "html_attribute", "marker_in_attribute": True, "marker_in_script_source": False}
        rows.append({
            "candidate_family": "xss",
            "semantic": {"expected_oracle": "fixture_inert_attribute_oracle_v1"},
            "evidence": _evidence("a" * 64, projection),
            "oracle_projection": projection,
            "rule_ir_result": True,
            "pair": {"pair_id": "pg22-xss", "variant": variant, "surface_role": "reflected_attribute"},
        })
    return rows


def _sql_rows() -> list[dict[str, Any]]:
    rows = []
    for variant in ("plain", "url_percent"):
        projection = {"controlled_differential": True, "interpreter_boundary": True, "modality": "bounded_timing", "database_touched": False, "real_sleep_performed": False, "external_network": False}
        rows.append({
            "candidate_family": "injection",
            "semantic": {"expected_oracle": "synthetic_sql_ast_differential_v1"},
            "evidence": _evidence("b" * 64, projection),
            "oracle_projection": projection,
            "rule_ir_result": True,
            "pair": {"pair_id": "pg22-sql", "variant": variant},
        })
    return rows


def _logic_rows() -> list[dict[str, Any]]:
    rows = []
    for variant in ("plain", "url_percent"):
        projection = {
            "positive": True,
            "oracle_signal": "authorization_boundary_divergence",
            "typed": {"protected_resource": True, "non_admin_subject": True, "quota_nonzero": True, "unexpected_grant": True},
        }
        rows.append({
            "candidate_family": "access_control",
            "semantic": {"expected_oracle": "synthetic_authorization_boundary_v1"},
            "evidence": _evidence("c" * 64, projection),
            "oracle_projection": projection,
            "rule_ir_result": True,
            "safety": {"state_mutated": False, "credentials_stored": False},
            "pair": {"pair_id": "pg22-logic", "variant": variant},
        })
    return rows


def _run_case(case_id: str, rows: list[dict[str, Any]], validator: Callable[[list[dict[str, Any]]], dict[str, Any]], expected: bool, tamper: bool) -> dict[str, Any]:
    candidate = copy.deepcopy(rows)
    if tamper:
        candidate[0]["oracle_projection"] = dict(candidate[0]["oracle_projection"], tampered=True)
    result = validator(candidate)
    return {
        "case_id": case_id,
        "tampered": tamper,
        "expected_accepted": expected,
        "observed_accepted": bool(result.get("accepted")),
        "passed": bool(result.get("accepted")) is expected,
        "reasons": result.get("reasons", []),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    xss = _xss_rows()
    sql = _sql_rows()
    logic = _logic_rows()
    cases = [
        _run_case("xss_untampered", xss, lambda rows: revalidate_positive_pair(rows, expected_family="xss", oracle_name="fixture_inert_attribute_oracle_v1", authorized_source_hash="a" * 64, required_sink_kind="html_attribute"), True, False),
        _run_case("xss_tampered", xss, lambda rows: revalidate_positive_pair(rows, expected_family="xss", oracle_name="fixture_inert_attribute_oracle_v1", authorized_source_hash="a" * 64, required_sink_kind="html_attribute"), False, True),
        _run_case("sql_untampered", sql, lambda rows: revalidate_sql_pair(rows, authorized_source_hash="b" * 64, oracle_name="synthetic_sql_ast_differential_v1"), True, False),
        _run_case("sql_tampered", sql, lambda rows: revalidate_sql_pair(rows, authorized_source_hash="b" * 64, oracle_name="synthetic_sql_ast_differential_v1"), False, True),
        _run_case("logic_untampered", logic, lambda rows: revalidate_logic_access_pair(rows, authorized_source_hash="c" * 64, expected_family="access_control", oracle_name="synthetic_authorization_boundary_v1", expected_signal="authorization_boundary_divergence"), True, False),
        _run_case("logic_tampered", logic, lambda rows: revalidate_logic_access_pair(rows, authorized_source_hash="c" * 64, expected_family="access_control", oracle_name="synthetic_authorization_boundary_v1", expected_signal="authorization_boundary_divergence"), False, True),
    ]
    result = {
        "schema_version": "sift-pg-pk-22-oracle-projection-binding-report-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "pass" if all(case["passed"] for case in cases) else "fail",
        "case_count": len(cases),
        "passed_case_count": sum(int(case["passed"]) for case in cases),
        "cases": cases,
        "local_only": True,
        "model_or_checkpoint_modified": False,
        "payload_generation_modified": False,
    }
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-22 oracle projection binding 实验\n\n"
        f"状态：`{result['status']}`；通过：{result['passed_case_count']}/{result['case_count']}。\n\n"
        "XSS、SQL、logic/access 三类未篡改 pair 均保持 accepted；只修改顶层 oracle projection 而不改变已哈希 evidence 的 pair 均被拒绝。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "status": result["status"],
        "passed": f"{result['passed_case_count']}/{result['case_count']}",
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
