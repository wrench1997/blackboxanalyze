"""Evaluate the reviewed geometry features with a mixed-source seed holdout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_pg53_rule_ir_candidate import _calibrate, _metrics, _raw_predictions, _train  # noqa: E402


SOURCE_REPORT = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
FUNNEL_DATASET = ROOT / "research" / "pg53_web_feature_funnel_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg53_mixed_source_seed_holdout_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg53_mixed_source_seed_holdout_report_v1.md"
PROTOCOL_ID = "pg-pk-53-mixed-source-seed-holdout-v1"
TRAIN_SEEDS = {5301, 5307}
DEV_SEED = 5311


def main() -> int:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    funnel = json.loads(FUNNEL_DATASET.read_text(encoding="utf-8"))
    rows = list(source["rows"])
    selected = list(funnel["accepted_features"])
    feature_map = {str(row["sample_id"]): row for row in funnel["rows"]}
    train_rows = [row for row in rows if int(row["sampling_seed"]) in TRAIN_SEEDS]
    dev_rows = [row for row in rows if row["implementation"] == "pg35" and int(row["sampling_seed"]) == DEV_SEED]
    holdout_rows = [row for row in rows if row["implementation"] == "pg36" and int(row["sampling_seed"]) == DEV_SEED]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fit = _train(train_rows, feature_map, selected, device=device)
    dev_predictions = _raw_predictions(model, dev_rows, feature_map, selected, fit, device=device)
    threshold, dev_metrics = _calibrate(dev_predictions)
    holdout_predictions = _raw_predictions(model, holdout_rows, feature_map, selected, fit, device=device)
    raw = _metrics(holdout_predictions, threshold=0.0)
    calibrated = _metrics(holdout_predictions, threshold=threshold)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg53-mixed-source-seed-holdout-report-v1",
        "split": {
            "train": "PG-35 + PG-36, seeds 5301/5307",
            "dev": "PG-35, seed 5311",
            "holdout": "PG-36, seed 5311",
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "holdout_rows": len(holdout_rows),
        },
        "device": str(device),
        "selected_features": selected,
        "feature_funnel_review_evidence_sha256": funnel["review_evidence_sha256"],
        "training_fit": fit,
        "threshold": threshold,
        "dev": dev_metrics,
        "holdout": {"raw": raw, "calibrated": calibrated, "predictions": holdout_predictions},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_claim_allowed": False,
            "status": "diagnostic_only",
            "reason": "mixed_source_seed_holdout_is_one_candidate_and_not_a_full_capability_matrix",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-53 混合来源、未见种子留出",
        "",
        "训练同时使用 PG-35/PG-36 的两个种子；校准只看 PG-35 第三个种子；盲测为 PG-36 第三个种子。",
        "",
        f"设备：`{device}`；特征数：`{len(selected)}`。",
        f"PG-36 未见种子 calibrated typed recall：`{calibrated['typed_recall']:.3f}`；precision：`{calibrated['precision']:.3f}`；false accept：`{calibrated['false_accept_count']}`；abstain：`{calibrated['abstain_rate']:.3f}`。",
        "",
        "结果仍不允许训练晋升或长期记忆；必须继续加入独立来源、族外表面和更多种子复现。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "device": str(device), "dev": dev_metrics, "holdout": calibrated, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
