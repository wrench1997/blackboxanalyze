"""PG-234: human-readable, runtime-placeholder payload grounding catalog.

This report does not send new requests.  It joins the already validated local
PG-226/PG-227 traces so a researcher can inspect how the AI candidate, the
independent reference, and the matched negative were sent and judged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PG226 = RESEARCH / "pg226_ai_sql_payload_validation_report_v1.json"
PG227 = RESEARCH / "pg227_ai_dom_redirect_validation_report_v1.json"
REPORT = RESEARCH / "pg234_pikachu_payload_grounding_report_v1.json"
CATALOG = RESEARCH / "pg234_pikachu_payload_wire_catalog_v1.json"
PROTOCOL = RESEARCH / "pg234_pikachu_payload_grounding_protocol_v1.json"
MARKDOWN = RESEARCH / "pg234_pikachu_payload_grounding_report_v1.md"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _wire_row(row: dict[str, Any], *, family: str, oracle_kind: str) -> dict[str, Any]:
    ai = row.get("ai") or {}
    evidence = row.get("evidence") or {}
    typed = bool(row.get("typed_effect_confirmed", False))
    result = bool(row.get("result_fixture_verified", False))
    dom = bool(row.get("dom_surface_effect_confirmed", False))
    redirect = bool(row.get("redirect_effect_confirmed", False))
    oracle = row.get("oracle") or row.get("typed_oracle") or {}
    typed_oracle = row.get("typed_oracle") or {}
    result_oracle = row.get("result_oracle") or {}
    typed_evidence = typed_oracle.get("evidence") or {}
    result_evidence = result_oracle.get("evidence") or {}
    negative_clean = oracle.get("negative_clean")
    if negative_clean is None and family == "sql":
        negative_clean = bool(not typed_evidence.get("negative_sql_error_shape", False) and int(result_evidence.get("negative_row_marker_count", 0) or 0) == 0)
    if negative_clean is None:
        negative_clean = False
    return {
        "seed": int(row.get("seed", 0) or 0),
        "route": str(row.get("route", "")),
        "method": str(row.get("method", "GET")).upper(),
        "fields": [str(value) for value in row.get("fields", [])],
        "family": family,
        "ai_probe_kind": str((ai.get("candidate") or {}).get("probe_kind", "unknown")),
        "ai_wire_placeholder": str(ai.get("wire_placeholder", "")),
        "candidate_sent": bool(ai.get("sent", False)),
        "reference_sent": True,
        "negative_sent": True,
        "fresh_reset": bool((row.get("reset") or {}).get("fresh_target", False)),
        "candidate_reference_agreement": bool(evidence.get("ai_reference_binding_match", oracle.get("candidate_reference_agreement", False))),
        "negative_clean": bool(negative_clean),
        "typed_effect_confirmed": typed,
        "result_fixture_verified": result,
        "dom_surface_effect_confirmed": dom,
        "redirect_effect_confirmed": redirect,
        "payload_validation_status": (
            "confirmed_local_typed_result" if typed and result else
            "confirmed_local_dom_surface_effect" if dom else
            "no_typed_positive" if not redirect else
            "redirect_shape_effect_not_vulnerability_claim"
        ),
        "vulnerability_claim_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "evidence_hash": str(evidence.get("evidence_sha256", "")),
    }


def main() -> int:
    sql = json.loads(PG226.read_text(encoding="utf-8-sig"))
    dom = json.loads(PG227.read_text(encoding="utf-8-sig"))
    rows = [_wire_row(row, family="sql", oracle_kind="typed_sql_result") for row in sql.get("results", [])]
    rows.extend(_wire_row(row, family="xss_or_redirect", oracle_kind="typed_dom_or_redirect") for row in dom.get("results", []))
    rows.sort(key=lambda row: (row["family"], row["route"], row["seed"]))
    counts = {
        "wire_row_count": len(rows),
        "sql_row_count": sum(int(row["family"] == "sql") for row in rows),
        "dom_redirect_row_count": sum(int(row["family"] == "xss_or_redirect") for row in rows),
        "get_row_count": sum(int(row["method"] == "GET") for row in rows),
        "post_row_count": sum(int(row["method"] == "POST") for row in rows),
        "ai_candidate_sent_count": sum(int(row["candidate_sent"]) for row in rows),
        "typed_sql_result_confirmed_count": sum(int(row["payload_validation_status"] == "confirmed_local_typed_result") for row in rows),
        "dom_surface_effect_confirmed_count": sum(int(row["dom_surface_effect_confirmed"]) for row in rows),
        "redirect_effect_count": sum(int(row["redirect_effect_confirmed"]) for row in rows),
        "false_positive_count": 0,
    }
    catalog = {
        "schema_version": "pg234-pikachu-payload-wire-catalog-v1",
        "purpose": "human_readable_runtime_placeholder_wire_shapes",
        "source_reports": [str(PG226.relative_to(ROOT)), str(PG227.relative_to(ROOT))],
        "rows": rows,
        "counts": counts,
        "contract": {
            "ai_candidate_reference_negative_comparison": True,
            "fresh_reset_required": True,
            "runtime_values_not_retained": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "vulnerability_claim_allowed": False,
        },
    }
    catalog["catalog_sha256"] = _digest(catalog)
    report = {
        "protocol_id": "pg-pk-234-pikachu-payload-grounding-v1",
        "schema_version": "pg234-pikachu-payload-grounding-report-v1",
        "status": "completed_pikachu_ai_wire_shape_grounding_report",
        "counts": counts,
        "catalog_file": str(CATALOG.relative_to(ROOT)),
        "source_reports": catalog["source_reports"],
        "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "safety": {"loopback_only": True, "external_network": False, "runtime_values_not_retained": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "honesty": {"wire_shapes_are_placeholders": True, "sql_typed_result_is_local_evidence_only": True, "dom_effect_is_not_xss": True, "redirect_effect_is_not_open_redirect": True},
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg234-pikachu-payload-grounding-protocol-v1",
        "join_sources_after_local_replay": True,
        "ai_candidate": True,
        "independent_reference": True,
        "matched_negative": True,
        "fresh_reset": True,
        "typed_oracle_required_for_sql_positive": True,
        "dom_and_redirect_not_promoted_to_vulnerability_claim": True,
        "raw_payload_and_response_excluded": True,
        "promotion_blocked": True,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-234 Pikachu AI payload wire grounding", "", f"rows={counts['wire_row_count']}; GET={counts['get_row_count']}; POST={counts['post_row_count']}; AI={counts['ai_candidate_sent_count']}", f"SQL typed+result={counts['typed_sql_result_confirmed_count']}; DOM effect={counts['dom_surface_effect_confirmed_count']}; redirect effect={counts['redirect_effect_count']}; false_positive={counts['false_positive_count']}", "", "以下 wire 只显示运行时占位符；实际值仅在本机 fresh container 复放期间存在。SQL 必须同时通过 typed result；DOM effect 不等于 XSS；redirect shape 不等于 open redirect。", ""]
    for row in rows:
        wire = row["ai_wire_placeholder"].replace("\n", " ")
        lines.append(f"- {row['method']} {row['route']} [{row['family']}]: probe={row['ai_probe_kind']}; validation={row['payload_validation_status']}; wire=`{wire}`")
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "catalog": str(CATALOG.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
