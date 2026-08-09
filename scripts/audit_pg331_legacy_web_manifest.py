"""Audit historical browser-manifest coverage against the PG-331 ontology.

This is a read-only diagnostic.  It summarizes what an old DOM-only crawl
actually observed (and what it explicitly did not observe) without copying
route literals, labels, response bodies, or source into a model dataset.
The report is intentionally never training-eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "research" / "pg179_pikachu_browser_crawl_manifest_v1.json"
OUTPUT = ROOT / "research" / "pg331_legacy_web_manifest_audit_v1.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _status(value: str) -> str:
    text = str(value or "").casefold()
    if text in {"observed", "complete", "present"}:
        return "observed"
    if text in {"partial", "baseline_only", "incomplete_missing_request_parameter_context"}:
        return "partial"
    return "not_observed"


def audit_manifest(document: Mapping[str, Any], *, input_path: str) -> dict[str, Any]:
    pages = document.get("page_summaries") if isinstance(document.get("page_summaries"), list) else []
    routes = document.get("route_catalog") if isinstance(document.get("route_catalog"), list) else []
    rows = document.get("request_response_rows") if isinstance(document.get("request_response_rows"), list) else []
    scripts = document.get("script_catalog") if isinstance(document.get("script_catalog"), list) else []
    stats = document.get("stats") if isinstance(document.get("stats"), Mapping) else {}

    page_loaded = sum(1 for page in pages if isinstance(page, Mapping) and page.get("loaded") is True)
    page_document = sum(1 for page in pages if isinstance(page, Mapping) and any(key in page for key in ("title", "link_observation_count", "form_count", "script_count")))
    page_navigation = sum(1 for page in pages if isinstance(page, Mapping) and "link_observation_count" in page)
    page_response = sum(1 for page in pages if isinstance(page, Mapping) and isinstance(page.get("response_projection"), Mapping))
    page_redirect = sum(1 for page in pages if isinstance(page, Mapping) and isinstance((page.get("response_projection") or {}).get("status_chain"), list))

    methods: Counter[str] = Counter()
    route_quality: Counter[str] = Counter()
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        methods.update(str(method).upper() for method in (route.get("methods_observed") or []) if method)
        route_quality[str(route.get("quality_status") or "unknown")] += 1

    baseline_response = sum(
        1
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("response_schema"), Mapping)
        and row["response_schema"].get("parameterized_response_observed") is False
    )
    parameterized_response = sum(
        1
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("response_schema"), Mapping)
        and row["response_schema"].get("parameterized_response_observed") is True
    )

    axes = {
        "document_structure": {"status": "observed" if page_document else "not_observed", "page_count": len(pages), "loaded_count": page_loaded},
        "navigation": {"status": "observed" if page_navigation else "not_observed", "page_count_with_link_counts": page_navigation, "route_count": len(routes)},
        "request_transport": {"status": "partial" if rows else "not_observed", "surface_count": len(rows), "method_counts": dict(sorted(methods.items())), "post_response_observed": False},
        "response_transport": {"status": "partial" if page_response else "not_observed", "baseline_projection_count": page_response, "redirect_chain_count": page_redirect, "parameterized_response_observed_count": parameterized_response, "baseline_only_count": baseline_response},
        "javascript_surface": {"status": "partial" if scripts else "not_observed", "script_catalog_count": len(scripts), "ast_or_sink_observed": False},
        "failure_feedback": {"status": "not_observed", "reason": "crawl_recorded_no_failure_or_repair_transition"},
        "belief_and_replay": {"status": "not_observed", "reason": "no_belief_or_typed_evaluator_sidecar"},
    }
    report: dict[str, Any] = {
        "protocol_id": "pg-pk-331-legacy-web-manifest-audit-v1",
        "schema_version": "pg331-legacy-web-manifest-audit-v1",
        "status": "diagnostic_only_blocked",
        "input": {"path": input_path, "manifest_sha256": str(document.get("manifest_sha256") or _sha256_json(document)), "raw_values_read": True, "raw_values_persisted": False},
        "coverage": {"page_count": len(pages), "route_count": len(routes), "request_response_row_count": len(rows), "script_catalog_count": len(scripts), "declared_stats": {key: stats[key] for key in ("get_query_surface_count", "get_form_surface_count", "post_form_surface_count", "incomplete_surface_count") if key in stats}},
        "axes": axes,
        "route_quality_counts": dict(sorted(route_quality.items())),
        "missing_observations": ["parameterized_get_response", "parameterized_post_response", "failure_feedback", "belief_and_replay", "typed_evaluator", "fresh_reset_attestation"],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "历史 DOM-only 清单可用于词表覆盖诊断；baseline GET 之外的字段必须保持 partial/not_observed，不能用默认值补齐，也不能进入训练或长期记忆。",
    }
    report["audit_sha256"] = _sha256_json(report)
    return report


def run(*, input_path: Path = INPUT, output_path: Path = OUTPUT) -> dict[str, Any]:
    document = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, Mapping):
        raise ValueError("legacy manifest must be a JSON object")
    report = audit_manifest(document, input_path=str(input_path.relative_to(ROOT)))
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="audit historical PG-179 manifest without creating training data")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run()
    print(json.dumps(report if args.json else {"status": report["status"], "coverage": report["coverage"], "missing_observations": report["missing_observations"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
