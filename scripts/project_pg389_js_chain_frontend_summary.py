"""Build a bounded PG-389 candidate summary for the frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "research" / "pg389_js_chain_candidate_local_cuda_e8_v1.json"
DEFAULT_AUDIT = ROOT / "research" / "pg389_js_decode_filter_chain_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "frontend" / "public" / "research" / "pg389_js_chain_frontend_summary_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pg389_frontend_input_must_be_object")
    return value


def build_summary(report_path: Path = DEFAULT_REPORT, audit_path: Path = DEFAULT_AUDIT) -> dict[str, Any]:
    if not report_path.exists():
        return {"schema_version": "pg389-js-chain-frontend-summary-v1", "status": "missing", "report_file": report_path.name, "training_allowed": False, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    report = _load(report_path)
    audit = _load(audit_path) if audit_path.exists() else {}
    holdouts = [item.get("holdout", {}) for item in report.get("seeds", []) if isinstance(item, dict) and isinstance(item.get("holdout"), dict)]
    def values(key: str) -> list[float]:
        return [float(item[key]) for item in holdouts if isinstance(item.get(key), (int, float))]
    comp = values("composition_exact")
    slots = values("slot_accuracy")
    ask = values("ask_recall")
    repair = values("repair_recall")
    entropy = values("composition_entropy")
    false_allow = [int(item.get("negative_false_allow", 0) or 0) for item in holdouts]
    contract = report.get("contract") if isinstance(report.get("contract"), dict) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), dict) else {}
    promotion = report.get("promotion") if isinstance(report.get("promotion"), dict) else {}
    return {
        "schema_version": "pg389-js-chain-frontend-summary-v1",
        "status": "diagnostic_candidate_projection",
        "candidate_status": str(report.get("status", "unknown")),
        "report_file": report_path.name,
        "report_sha256": _sha(report_path),
        "audit_file": audit_path.name,
        "audit_sha256": _sha(audit_path) if audit_path.exists() else "",
        "dataset_status": str(audit.get("dataset_status", "unknown")),
        "chain_case_count": int(audit.get("counts", {}).get("expected_records", 0) or 0) if isinstance(audit.get("counts"), dict) else 0,
        "train_count": int(report.get("train_count", 0) or 0),
        "holdout_count": int(report.get("holdout_count", 0) or 0),
        "seed_count": len(holdouts),
        "vocabulary_size": int((report.get("train_context_vocabulary") or {}).get("size", 0) or 0),
        "vocabulary_scope": str((report.get("train_context_vocabulary") or {}).get("scope", "unknown")),
        "holdout": {
            "composition_exact": round(min(comp), 6) if comp else 0.0,
            "slot_accuracy": round(min(slots), 6) if slots else 0.0,
            "ask_recall": round(min(ask), 6) if ask else 0.0,
            "repair_recall": round(min(repair), 6) if repair else 0.0,
            "negative_false_allow": max(false_allow) if false_allow else 0,
            "composition_entropy": round(max(entropy), 6) if entropy else 0.0,
        },
        "contract": {
            "status": str(contract.get("status", "unknown")),
            "failure_count": len(contract.get("failures", [])) if isinstance(contract.get("failures"), list) else 0,
            "unknown_context_count": int(contract.get("unknown_context_count", 0) or 0),
            "context_overlap": int(contract.get("context_overlap", 0) or 0),
        },
        "execution": {
            "optimizer_started": bool(execution.get("optimizer_started")),
            "device": str(execution.get("device", "unknown")),
            "gpu_touched": bool(execution.get("gpu_touched")),
            "docker_started": bool(execution.get("docker_started")),
            "network_contacted": bool(execution.get("network_contacted")),
        },
        "training_eligible": int(report.get("training_eligible", 0) or 0),
        "capability_training_allowed": bool(report.get("capability_training_allowed")),
        "training_allowed": False,
        "promotion": {key: bool(promotion.get(key, False)) for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")},
        "note": "抽象 JS 解码/过滤链 candidate；不包含源码、原始 probe、响应体或 evaluator answer。",
    }


def write_summary(output: Path = DEFAULT_OUTPUT, report_path: Path = DEFAULT_REPORT, audit_path: Path = DEFAULT_AUDIT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_summary(report_path, audit_path), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = write_summary(args.output, args.report, args.audit)
    print(json.dumps({"output": str(output), "status": "diagnostic_candidate_projection"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
