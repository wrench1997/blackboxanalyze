"""Construct and review the PG-53 webpage feature funnel dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.web_feature_funnel import audit_feature_funnel, build_feature_dataset, review_feature_funnel  # noqa: E402


PROTOCOL_ID = "pg-pk-53-web-feature-funnel-v1"
SOURCE_REPORT = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg53_web_feature_funnel_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg53_web_feature_funnel_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg53_web_feature_funnel_report_v1.md"


def _markdown(report: dict) -> str:
    audit = report["audit"]
    review = report["review"]
    lines = [
        "# PG-53 网页特征漏斗审核",
        "",
        "输入是 PG-53 的安全 response projection；原始请求、响应正文、URL token、oracle 值和来源/种子标识不进入模型特征。surface_observation 只作 evaluator 诊断；无字段名的 generic_effect_geometry 才有资格进入漏斗。",
        "",
        "| stage | feature count |",
        "|---|---:|",
    ]
    for name, count in audit["stage_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "最终保留特征：" + (", ".join(f"`{name}`" for name in audit["accepted_features"]) if audit["accepted_features"] else "（无；需要修复数据或放宽可证实的工程门）"),
        "",
        f"Codex 特征审核：`{review['decision']}`；审核证据哈希：`{review['review_evidence_sha256']}`。",
        "",
        "审核通过只允许进入下一轮独立 OOD 实验；不会自动训练晋升或写入长期记忆。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    dataset = build_feature_dataset(source["rows"])
    audit = audit_feature_funnel(dataset)
    review = review_feature_funnel(audit)
    dataset["funnel_report_sha256"] = hashlib.sha256(json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    dataset["accepted_features"] = list(audit["accepted_features"])
    dataset["review_decision"] = review["decision"]
    dataset["review_evidence_sha256"] = review["review_evidence_sha256"]
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg53-web-feature-funnel-report-v1",
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "audit": audit,
        "review": review,
        "training_boundary": {
            "training_eligible": False,
            "long_term_memory_write": False,
            "formal_capability_claim": False,
            "reason": "feature_funnel_is_a_pretraining_quality_gate_not_a_capability_proof",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "stage_counts": audit["stage_counts"], "accepted_features": audit["accepted_features"], "review": review["decision"], "dataset": str(DATASET_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
