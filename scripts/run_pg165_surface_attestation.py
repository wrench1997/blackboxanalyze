"""PG-165: independently attest bounded effects from the real PG-51 replay.

Only safe surface observations are promoted to the next training pool.  The
result never contains a vulnerability label; `surface_effect` means that an
inert marker was reflected under a matched control, not that code executed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg165_surface_attestation import ATTESTATION_CONTRACT, ATTESTATION_CONTRACT_SHA256, attest_rows  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE_CATALOG = RESEARCH / "pg51_pikachu_docker_dual_channel_catalog_v1.json"
DATASET_PATH = RESEARCH / "pg165_surface_attested_training_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg165_surface_attestation_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg165_surface_attestation_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg165_surface_attestation_report_v1.md"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tokens(projection: dict[str, Any]) -> list[str]:
    return [
        "[BOS]", "[STEP]", "[SRC_TRANSPORT]",
        f"src.transport.method={projection.get('method', 'unknown')}",
        f"src.transport.placement={projection.get('placement', 'unknown')}",
        f"src.transport.encoding_depth={projection.get('encoding_depth', 0)}",
        "[IR]",
        f"ir.response.status_class={projection.get('status_class', 'unknown')}",
        f"ir.response.content_type_class={projection.get('content_type_class', 'unknown')}",
        f"ir.response.body_length_bucket={projection.get('body_length_bucket', 'unknown')}",
        f"ir.response.marker_reflected={str(bool(projection.get('marker_reflected'))).lower()}",
        f"ir.response.marker_count={projection.get('marker_count', 0)}",
        f"ir.response.marker_location={projection.get('marker_location', 'none')}",
        f"ir.response.state_changed={str(bool(projection.get('state_changed'))).lower()}",
        f"ir.response.status_changed={str(bool(projection.get('status_changed'))).lower()}",
        f"ir.response.location_origin_changed={str(bool(projection.get('location_origin_changed'))).lower()}",
        f"ir.response.transport_error={str(bool(projection.get('transport_error'))).lower()}",
        f"ir.shape.kind={projection.get('shape_kind', 'unknown')}",
        f"ir.shape.field_count={projection.get('shape_field_count', 0)}",
        f"ir.shape.scalar_count={projection.get('shape_scalar_count', 0)}",
        "[OBS]", "obs.oracle=safe_surface_attestation", "[EOS]",
    ]


def main() -> None:
    source = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    attested = attest_rows(source["samples"])
    rows: list[dict[str, Any]] = []
    for row in attested["rows"]:
        attestation = row["attestation"]
        projection = dict(row.get("model_projection") or {})
        row_out = {
            "row_id": row["sample_id"],
            "source_group": "pikachu_pg51_attested_surface",
            "method": projection.get("method", "unknown"),
            "model_input_tokens": _tokens(projection),
            "training_label": row["training_label"],
            "training_eligible": bool(row["training_eligible"]),
            "attestation_status": attestation.get("status", "abstain"),
            "attestation_sha256": attestation.get("attestation_sha256"),
            "evidence_sha256": (row.get("evidence") or {}).get("evidence_hash"),
            "vulnerability_label": False,
            "memory_promotion_allowed": False,
        }
        rows.append(row_out)
    eligible = [row for row in rows if row["training_eligible"]]
    dataset = {
        "schema_version": "pg165-surface-attested-training-dataset-v1",
        "purpose": "real local Docker safe surface-effect language/action projection; no vulnerability labels",
        "source_catalog": str(SOURCE_CATALOG.relative_to(ROOT)),
        "attestation_contract": ATTESTATION_CONTRACT,
        "attestation_contract_sha256": ATTESTATION_CONTRACT_SHA256,
        "rows": rows,
        "training_eligible_rows": eligible,
        "training_eligible": True,
        "training_role": "surface_effect_only",
        "vulnerability_claim_allowed": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_labels_in_model_input": False,
        "family_labels_in_model_input": False,
        "target_identity_in_model_input": False,
        "memory_promotion_allowed": False,
    }
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    protocol = {
        "protocol_id": "pg-pk-165-surface-attestation-v1",
        "schema_version": "pg165-surface-attestation-protocol-v1",
        "source_catalog": str(SOURCE_CATALOG.relative_to(ROOT)),
        "attestation_contract": ATTESTATION_CONTRACT,
        "gates": ["independent_projection_recompute", "fresh_target_reset", "evidence_sha256", "one_control_one_candidate", "matched_negative", "no_vulnerability_claim"],
        "methods": ["GET", "POST"],
        "training_policy": {"surface_effect_only": True, "vulnerability_labels": False, "memory_promotion_allowed": False},
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False},
    }
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    report = {
        "schema_version": "pg165-surface-attestation-report-v1",
        "protocol_id": "pg-pk-165-surface-attestation-v1",
        "status": "completed_pg165_surface_attestation",
        "source": {"catalog": str(SOURCE_CATALOG.relative_to(ROOT)), "row_count": len(rows), "source_sha256": source.get("manifest_sha256")},
        "attestation": {key: value for key, value in attested.items() if key not in {"rows", "checks"}},
        "checks": {"all_evidence_hashes_valid": all(check["evidence_valid"] for check in attested["checks"]), "all_attestation_hashes_valid": all(len(str(item.get("attestation_sha256", ""))) == 64 for item in attested["attestations"]), "get_count": sum(row["method"] == "GET" for row in rows), "post_count": sum(row["method"] == "POST" for row in rows), "training_eligible_row_count": len(eligible)},
        "claim": {"surface_effect_claim_allowed": True, "vulnerability_claim_allowed": False, "execution_oracle": False, "sql_ast_oracle": False, "external_redirect_oracle": False},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False},
        "safety": protocol["safety"],
        "source_hashes": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "attestor_sha256": hashlib.sha256((ROOT / "app" / "pg165_surface_attestation.py").read_bytes()).hexdigest(), "dataset_sha256": dataset["dataset_sha256"], "protocol_sha256": protocol["protocol_sha256"]},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-165 真实 Docker 安全表面 attestation",
        "",
        f"- rows: **{len(rows)}**；GET/POST: **{report['checks']['get_count']}/{report['checks']['post_count']}**",
        f"- safe reflection pairs: **{attested['confirmed_safe_effect_count']}**；safe no-effect pairs: **{attested['confirmed_safe_no_effect_count']}**",
        f"- training-eligible surface rows: **{len(eligible)}**",
        "",
        "attestation 只证明无执行 canary 的表面反射/无效果，不证明 XSS、SQL 注入、重定向或认证绕过；原始 probe/响应正文不进入数据集。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "row_count": len(rows), "get_count": report["checks"]["get_count"], "post_count": report["checks"]["post_count"], "safe_reflection_pairs": attested["confirmed_safe_effect_count"], "safe_no_effect_pairs": attested["confirmed_safe_no_effect_count"], "training_eligible_row_count": len(eligible), "vulnerability_claim_allowed": False, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
