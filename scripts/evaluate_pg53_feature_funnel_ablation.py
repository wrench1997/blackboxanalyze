"""Compare all safe candidate features with the reviewed funnel subset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pg53_rule_ir_candidate import (  # noqa: E402
    _calibrate,
    _metrics,
    _raw_predictions,
    _train,
)


PROTOCOL_ID = "pg-pk-53-feature-funnel-ablation-v1"
SOURCE_REPORT = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
FUNNEL_DATASET = ROOT / "research" / "pg53_web_feature_funnel_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg53_feature_funnel_ablation_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg53_feature_funnel_ablation_report_v1.md"


def _run(feature_map: dict[str, dict], rows: list[dict], selected: list[str], *, device: torch.device) -> dict:
    train_rows = [row for row in rows if row["implementation"] == "pg35" and int(row["sampling_seed"]) in {5301, 5307}]
    dev_rows = [row for row in rows if row["implementation"] == "pg35" and int(row["sampling_seed"]) == 5311]
    holdout_rows = [row for row in rows if row["implementation"] == "pg36"]
    model, fit = _train(train_rows, feature_map, selected, device=device)
    dev_predictions = _raw_predictions(model, dev_rows, feature_map, selected, fit, device=device)
    threshold, dev_metrics = _calibrate(dev_predictions)
    holdout_predictions = _raw_predictions(model, holdout_rows, feature_map, selected, fit, device=device)
    return {
        "feature_count": len(selected),
        "features": selected,
        "fit": fit,
        "threshold": threshold,
        "dev": dev_metrics,
        "holdout": {
            "raw": _metrics(holdout_predictions, threshold=0.0),
            "calibrated": _metrics(holdout_predictions, threshold=threshold),
        },
    }


def main() -> int:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    funnel = json.loads(FUNNEL_DATASET.read_text(encoding="utf-8"))
    rows = list(source["rows"])
    feature_map = {str(row["sample_id"]): row for row in funnel["rows"]}
    all_features = list(funnel["model_feature_names"])
    accepted = list(funnel["accepted_features"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_safe = _run(feature_map, rows, all_features, device=device)
    reviewed = _run(feature_map, rows, accepted, device=device)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg53-feature-funnel-ablation-report-v1",
        "split": {"train": "pg35 seeds 5301/5307", "dev": "pg35 seed 5311", "holdout": "pg36 all seeds"},
        "device": str(device),
        "all_safe_candidate": all_safe,
        "reviewed_funnel": reviewed,
        "delta_reviewed_minus_all_safe": {
            "typed_recall": round(reviewed["holdout"]["calibrated"]["typed_recall"] - all_safe["holdout"]["calibrated"]["typed_recall"], 6),
            "false_accept_count": reviewed["holdout"]["calibrated"]["false_accept_count"] - all_safe["holdout"]["calibrated"]["false_accept_count"],
            "abstain_rate": round(reviewed["holdout"]["calibrated"]["abstain_rate"] - all_safe["holdout"]["calibrated"]["abstain_rate"], 6),
        },
        "funnel_review_evidence_sha256": funnel["review_evidence_sha256"],
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_claim_allowed": False,
            "reason": "ablation_is_diagnostic_and_does_not_replace_multi_source_multi_seed_capability_gate",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_metric = all_safe["holdout"]["calibrated"]
    reviewed_metric = reviewed["holdout"]["calibrated"]
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-53 特征漏斗反事实消融",
        "",
        "固定 PG-35 训练、PG-36 实现留出，比较全部安全候选与 Codex 审核后的漏斗子集。",
        "",
        "| feature set | count | typed recall | false accept | abstain |",
        "|---|---:|---:|---:|---:|",
        f"| all safe candidates | {all_safe['feature_count']} | {all_metric['typed_recall']:.3f} | {all_metric['false_accept_count']} | {all_metric['abstain_rate']:.3f} |",
        f"| reviewed funnel | {reviewed['feature_count']} | {reviewed_metric['typed_recall']:.3f} | {reviewed_metric['false_accept_count']} | {reviewed_metric['abstain_rate']:.3f} |",
        "",
        "这是特征选择诊断，不是能力晋升；两条路径都必须经过独立实现、多种子和负对照门禁。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "device": str(device), "all_safe": all_metric, "reviewed": reviewed_metric, "delta": report["delta_reviewed_minus_all_safe"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
