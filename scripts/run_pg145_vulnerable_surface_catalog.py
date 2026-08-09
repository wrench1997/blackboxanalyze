"""Build PG-145's local multi-surface vulnerability-family catalog.

The runner is deliberately offline: it generates evaluator-side records for
allow-listed loopback URLs and never sends a request.  The resulting rows are
safe training/evaluation metadata, not an operational scanner or payload set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg145_vulnerable_surface_catalog import (
    BASE_URL,
    DATASET_SCHEMA,
    FAMILIES,
    SCHEMA_VERSION,
    STYLE_REFERENCES,
    STYLES,
    build_catalog,
    sha256_json,
)


RESEARCH = ROOT / "research"
CATALOG = RESEARCH / "pg145_local_multisurface_vulnerability_catalog_v1.json"
DATASET = RESEARCH / "pg145_local_multisurface_model_dataset_v1.json"
PROTOCOL = RESEARCH / "pg145_local_multisurface_protocol_v1.json"
PROPOSAL = RESEARCH / "pg145_local_multisurface_proposal_v1.json"
TRACE = RESEARCH / "pg145_local_multisurface_trace_v1.json"
REPORT = RESEARCH / "pg145_local_multisurface_report_v1.json"
STYLE_MANIFEST = RESEARCH / "pg145_style_reference_manifest_v1.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _forbidden_raw(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return any(marker in text for marker in ("<script", "onerror", "union select", "file:///", "system(", "payload="))


def main() -> None:
    catalog_rows, model_rows, summary = build_catalog()
    raw_free = not _forbidden_raw(model_rows)
    style_manifest = {
        "schema_version": "pg145-style-reference-manifest-v1",
        "collection_policy": "public_design_metadata_only_no_raw_crawl",
        "raw_html_stored": False,
        "raw_javascript_stored": False,
        "probe_sent": False,
        "sources": STYLE_REFERENCES,
    }
    style_manifest["manifest_sha256"] = sha256_json(style_manifest)
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_pg145_local_multisurface_vulnerability_catalog",
        "scope": {
            "base_url": BASE_URL,
            "loopback_only": True,
            "external_network": False,
            "families": list(FAMILIES),
            "styles": list(STYLES),
        },
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_source_retained": False,
        "raw_probe_response_retained": False,
        "raw_payload_retained": False,
        "style_reference_manifest_sha256": style_manifest["manifest_sha256"],
        "rows": catalog_rows,
        "summary": summary,
    }
    catalog["catalog_sha256"] = sha256_json(catalog)
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "status": "representation_and_evaluator_review_only",
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "rows": model_rows,
        "model_input_contract": {
            "raw_source_included": False,
            "raw_marker_included": False,
            "raw_payload_included": False,
            "target_identity_included": False,
            "source_hash_included": False,
            "oracle_authority_included": False,
            "family_label_included": False,
            "waf_scope": "local_mock_only",
            "external_bypass_payloads": False,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    protocol = {
        "protocol_id": "pg-pk-145-local-multisurface-v1",
        "schema_version": "pg145-local-multisurface-protocol-v1",
        "objective": "用至少150个本地目标实例生成同一抽象漏洞族的多种网页/脚本表面，并保留 GET/POST、正负 oracle、unknown oracle 与实现留出。",
        "base_url": BASE_URL,
        "target_instance_minimum": 150,
        "families": list(FAMILIES),
        "styles": list(STYLES),
        "required_gates": {
            "target_instance_count_ge_150": summary["target_instance_count"] >= 150,
            "get_post_balance": summary["get_post_balance"],
            "positive_and_matched_negative": summary["positive_count"] == summary["matched_negative_count"],
            "fresh_reset_per_row": summary["fresh_reset_count"] == summary["row_count"],
            "typed_oracle_evidence": summary["typed_oracle_count"] > 0,
            "implementation_holdout": summary["implementation_holdout_count"] > 0,
            "raw_model_input_absent": raw_free,
            "waf_training_scope_local_mock_only": True,
            "external_bypass_payloads_forbidden": True,
        },
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "reason": "multi-surface fixtures require independent replay and model ablation before any training promotion",
        },
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    proposal = {
        "proposal_id": "pg-pk-145-local-multisurface-v1",
        "schema_version": "pg145-local-multisurface-proposal-v1",
        "status": "evaluation_only_local_fixture_collection",
        "selected_action": "replay_and_hold_out_source_styles_before_training",
        "target_instance_count": summary["target_instance_count"],
        "family_count": summary["vulnerability_family_count"],
        "style_count": summary["surface_style_count"],
        "training_eligible": False,
        "memory_promotion_allowed": False,
    }
    proposal["proposal_sha256"] = sha256_json(proposal)
    trace = {
        "schema_version": "pg145-local-multisurface-trace-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "completed_pg145_local_multisurface_vulnerability_catalog",
        "target_instance_count": summary["target_instance_count"],
        "row_count": summary["row_count"],
        "get_post_balance": summary["get_post_balance"],
        "positive_count": summary["positive_count"],
        "matched_negative_count": summary["matched_negative_count"],
        "unknown_oracle_count": summary["unknown_oracle_count"],
        "fresh_reset_per_row": summary["fresh_reset_count"] == summary["row_count"],
        "raw_model_input_absent": raw_free,
        "waf_training_scope": "local_mock_only",
        "external_bypass_payloads": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "catalog_sha256": catalog["catalog_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
    }
    trace["trace_sha256"] = sha256_json(trace)
    report = {
        "protocol_id": protocol["protocol_id"],
        "schema_version": "pg145-local-multisurface-report-v1",
        "status": "completed_pg145_local_multisurface_vulnerability_catalog",
        "hard_gates_passed": all(protocol["required_gates"].values()),
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "catalog_file": CATALOG.name,
        "dataset_file": DATASET.name,
        "style_reference_manifest_file": STYLE_MANIFEST.name,
        "summary": summary,
        "required_gates": protocol["required_gates"],
        "model_input_contract": dataset["model_input_contract"],
        "promotion": protocol["promotion"],
        "source": {
            "runner": _hash(Path(__file__)),
            "module": _hash(ROOT / "app" / "pg145_vulnerable_surface_catalog.py"),
        },
    }
    report["report_sha256"] = sha256_json(report)
    _write(CATALOG, catalog)
    _write(DATASET, dataset)
    _write(STYLE_MANIFEST, style_manifest)
    _write(PROTOCOL, protocol)
    _write(PROPOSAL, proposal)
    _write(TRACE, trace)
    _write(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "target_instance_count": summary["target_instance_count"],
        "row_count": summary["row_count"],
        "families": summary["family_counts"],
        "styles": summary["style_counts"],
        "get_post_balance": summary["get_post_balance"],
        "positive_count": summary["positive_count"],
        "matched_negative_count": summary["matched_negative_count"],
        "unknown_oracle_count": summary["unknown_oracle_count"],
        "hard_gates_passed": report["hard_gates_passed"],
        "training_eligible": report["training_eligible"],
        "report": str(REPORT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
