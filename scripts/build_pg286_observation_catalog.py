"""Build PG-286 from historical bounded response/DOM projections.

The catalog is a data-quality gate, not a synthetic success set.  It keeps
family/target labels outside ``context_tokens`` and marks rows incomplete when
the relevant typed observation modality (DOM, SQL AST, redirect, or logic
transition) is absent.  Incomplete rows are retained for collection planning
but are never training-eligible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg286_observation_tokens import build_observation_tokens, digest


RESEARCH = ROOT / "research"
SQL_REPORT = RESEARCH / "pg217_pikachu_typed_sql_oracle_report_v1.json"
DOM_REPORT = RESEARCH / "pg238_pikachu_surface_replay_report_v1.json"
CATALOG = RESEARCH / "pg286_observation_token_catalog_v1.json"
HARD = RESEARCH / "pg286_observation_token_hard_negative_v1.json"
AUDIT = RESEARCH / "pg286_observation_token_catalog_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _projection(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sql_records() -> list[dict[str, Any]]:
    report = _load(SQL_REPORT)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(list(report.get("results") or [])):
        ai = _projection(item.get("ai"))
        negative = _projection(item.get("negative"))
        reference = _projection(item.get("reference"))
        candidate_projection = _projection(ai.get("response_projection"))
        negative_projection = _projection((negative.get("response") or {}).get("response_projection"))
        reference_projection = _projection((reference.get("response") or {}).get("response_projection"))
        obs = build_observation_tokens(
            method=str(item.get("method", "GET")),
            fields=list(item.get("fields") or []),
            baseline=negative_projection,
            candidate=candidate_projection,
            negative=negative_projection,
            sql_response=candidate_projection,
            sql_ast=None,
        )
        target = {
            "next_action": "ask_typed",
            "method": str(item.get("method", "GET")).upper(),
            "probe_class": "sql",
            "channel": "query" if str(item.get("method", "GET")).upper() == "GET" else "form",
            "encoding": "unknown",
            "wire_kind": "query_param" if str(item.get("method", "GET")).upper() == "GET" else "form_field",
            "safe_to_send": False,
            "oracle_required": True,
        }
        rows.append({
            "record_id": f"pg286:sql:{index:03d}",
            "source": "pg217_pikachu_typed_sql_oracle_report_v1",
            "source_row": index,
            "family": "sql",
            "route_shape": "php_sqli_route",
            "method": target["method"],
            "fields_observed": list(item.get("fields") or []),
            "context_tokens": obs["context_tokens"],
            "evidence_status": obs["evidence_status"],
            "missing_modalities": obs["missing_modalities"],
            "target": target,
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_label_in_context": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "evidence_hash": str(item.get("typed_oracle", {}).get("evidence_hash", "")),
            "source_projection_hashes": {
                "baseline": str(negative_projection.get("projection_sha256", "")),
                "candidate": str(candidate_projection.get("projection_sha256", "")),
                "reference": str(reference_projection.get("projection_sha256", "")),
            },
        })
    return rows


def _dom_records() -> list[dict[str, Any]]:
    report = _load(DOM_REPORT)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(list(report.get("results") or [])):
        ai = _projection(item.get("ai"))
        candidate = _projection(item.get("candidate"))
        baseline = _projection(item.get("baseline"))
        negative = _projection(item.get("negative"))
        oracle = _projection(item.get("oracle"))
        dom = _projection(oracle.get("browser_dom"))
        obs = build_observation_tokens(
            method=str(item.get("method", "GET")),
            fields=list(item.get("fields") or []),
            baseline=_projection(baseline.get("response_projection")),
            candidate=_projection(candidate.get("response_projection")),
            negative=_projection(negative.get("response_projection")),
            dom=dom,
        )
        route = str(item.get("route", ""))
        family = "redirect" if "/urlredirect/" in route else "xss"
        probe_class = "redirect" if family == "redirect" else "xss"
        target = {
            "next_action": "ask_typed",
            "method": str(item.get("method", "GET")).upper(),
            "probe_class": probe_class,
            "channel": "query" if str(item.get("method", "GET")).upper() == "GET" else "form",
            "encoding": "plain",
            "wire_kind": "query_param" if str(item.get("method", "GET")).upper() == "GET" else "form_field",
            "safe_to_send": False,
            "oracle_required": True,
        }
        rows.append({
            "record_id": f"pg286:dom:{index:03d}",
            "source": "pg238_pikachu_surface_replay_report_v1",
            "source_row": index,
            "family": family,
            "route_shape": "php_dom_or_redirect_route",
            "method": target["method"],
            "fields_observed": list(item.get("fields") or []),
            "context_tokens": obs["context_tokens"],
            "evidence_status": obs["evidence_status"],
            "missing_modalities": obs["missing_modalities"],
            "target": target,
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_label_in_context": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "evidence_hash": str(item.get("evidence", {}).get("evidence_sha256", "")),
            "source_projection_hashes": {
                "baseline": str(_projection(baseline.get("response_projection")).get("projection_sha256", "")),
                "candidate": str(_projection(candidate.get("response_projection")).get("projection_sha256", "")),
                "negative": str(_projection(negative.get("response_projection")).get("projection_sha256", "")),
            },
        })
    return rows


def _hard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        tokens = [token for token in row["context_tokens"] if not token.startswith("candidate_")]
        target = dict(row["target"])
        target.update({"next_action": "abstain", "probe_class": "other", "channel": "unknown", "encoding": "unknown", "wire_kind": "none", "safe_to_send": False})
        result.append({
            "record_id": f"pg286:hard:{index:03d}",
            "source": "pg286_matched_projection_decoy",
            "source_row": row["source_row"],
            "family": "hidden",
            "route_shape": "decoy",
            "method": row["method"],
            "fields_observed": [],
            "context_tokens": tokens + ["evidence_status=incomplete", "[CTX_END]"],
            "evidence_status": "incomplete",
            "missing_modalities": ["matched_negative_oracle"],
            "target": target,
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_label_in_context": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "evidence_hash": row["evidence_hash"],
        })
    return result


def main() -> None:
    rows = [*_sql_records(), *_dom_records()]
    hard = _hard(rows)
    counts = {
        "total": len(rows),
        "sql": sum(row["family"] == "sql" for row in rows),
        "xss": sum(row["family"] == "xss" for row in rows),
        "redirect": sum(row["family"] == "redirect" for row in rows),
        "complete": sum(row["evidence_status"] == "complete" for row in rows),
        "incomplete": sum(row["evidence_status"] == "incomplete" for row in rows),
        "hard_negative": len(hard),
    }
    catalog = {
        "schema_version": "pg286-observation-token-catalog-v1",
        "purpose": "historical bounded response/DOM evidence tokenization and completeness audit",
        "sources": {
            "sql": "research/pg217_pikachu_typed_sql_oracle_report_v1.json",
            "dom_redirect": "research/pg238_pikachu_surface_replay_report_v1.json",
        },
        "records": rows,
        "hard_negative_records": hard,
        "counts": counts,
        "training_contract": {
            "family_hidden_in_context": True,
            "oracle_label_in_context": False,
            "raw_payload_out_of_context": True,
            "raw_response_out_of_context": True,
            "incomplete_training_eligible": False,
            "real_sql_ast_required": True,
            "fresh_get_post_replay_required": True,
            "remote_a800_required_for_training": True,
        },
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = digest({key: value for key, value in catalog.items() if key != "catalog_sha256"})
    hard_dataset = {
        "schema_version": "pg286-observation-token-hard-negative-v1",
        "records": hard,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "catalog_sha256": digest(hard),
    }
    all_rows = [*rows, *hard]
    forbidden = ("family=", "oracle=", "typed_effect", "positive", "payload=", "<script", "javascript:", "union select", "drop table")
    checks = {
        "source_rows_present": len(rows) == 28,
        "counts_match": counts["total"] == len(rows),
        "context_family_hidden": all(not any(token.startswith("family=") for token in row["context_tokens"]) and "ir_family_agnostic=1" in row["context_tokens"] for row in all_rows),
        "context_no_labels_or_literal": all(not row["oracle_label_in_context"] and not any(any(bad.casefold() in token.casefold() for bad in forbidden) for token in row["context_tokens"]) for row in all_rows),
        "raw_material_excluded": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in all_rows),
        "incomplete_not_training": all(not row["training_eligible"] for row in rows if row["evidence_status"] == "incomplete"),
        "hard_negative_abstain": all(row["target"]["next_action"] == "abstain" and row["target"]["safe_to_send"] is False and not row["training_eligible"] for row in hard),
        "missing_ast_visible": all("sql_ast_available=0" in row["context_tokens"] for row in rows if row["family"] == "sql"),
    }
    audit = {
        "schema_version": "pg286-observation-token-catalog-audit-v1",
        "catalog": str(CATALOG.relative_to(ROOT).as_posix()),
        "catalog_sha256": catalog["catalog_sha256"],
        "counts": counts,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "blocked",
        "training_eligible_rows": 0,
        "interpretation": "历史回放可提供 bounded response/DOM token，但 SQL 记录缺少 AST 差分；全部记录保持观察/收集 lane，不进入训练 gold。",
    }
    audit["audit_sha256"] = digest(audit)
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HARD.write_text(json.dumps(hard_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": counts, "catalog_sha256": catalog["catalog_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
