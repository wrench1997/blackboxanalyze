"""Build the PG-55 training-split feature funnel.

PG-55 is a second-generation candidate experiment.  It uses PG-53 replay rows
and only the ledger/envelope PG-42 rows from seeds 401/409 as the training
source.  PG-42 framed responses, seed 419, and template_injection remain
outside this funnel and are used only for the later holdout.  The funnel is
reviewed by the Codex primary reviewer, but never grants promotion by itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg53_rule_ir_candidate import PG53_MODEL_FAMILIES  # noqa: E402
from app.web_feature_funnel import audit_feature_funnel, build_feature_dataset, review_feature_funnel  # noqa: E402


PROTOCOL_ID = "pg-pk-55-invariant-feature-funnel-v1"
REPORT_PATH = ROOT / "research" / "pg55_invariant_feature_funnel_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg55_invariant_feature_funnel_dataset_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg55_invariant_feature_funnel_report_v1.md"
PG53_REPORT_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
PG54_TRACE_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_trace_v1.json"
KNOWN_FAMILIES = set(PG53_MODEL_FAMILIES)


def _rows() -> list[dict[str, Any]]:
    pg53 = json.loads(PG53_REPORT_PATH.read_text(encoding="utf-8"))["rows"]
    pg54 = json.loads(PG54_TRACE_PATH.read_text(encoding="utf-8"))["rows"]
    train_pg53 = [row for row in pg53 if int(row["sampling_seed"]) in {5301, 5307}]
    train_pg54 = [
        row
        for row in pg54
        if row["variant"] in {"ledger", "envelope"}
        and int(row["sampling_seed"]) in {401, 409}
        and row["family"] in KNOWN_FAMILIES
    ]
    return train_pg53 + train_pg54


def main() -> int:
    rows = _rows()
    dataset = build_feature_dataset(rows)
    dataset["dataset_id"] = "pg55-invariant-feature-funnel"
    dataset["training_eligible"] = False
    dataset["evaluation_only"] = True
    dataset["split_policy"] = {
        "train": "PG-53 seeds 5301/5307 + PG-42 ledger/envelope seeds 401/409",
        "excluded_from_funnel": "PG-42 framed, seed 419, template_injection",
        "holdout": "PG-42 framed all seeds",
    }
    audit = audit_feature_funnel(dataset)
    review = review_feature_funnel(audit, review_scope="PG-55 training-source safe visible response projections")
    dataset["accepted_features"] = audit["accepted_features"]
    dataset["funnel_report_sha256"] = hashlib.sha256(json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dataset["review_decision"] = review["decision"]
    dataset["review_evidence_sha256"] = review["review_evidence_sha256"]
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg55-invariant-feature-funnel-report-v1",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "source_rows": len(rows),
        "audit": audit,
        "review": review,
        "training_boundary": {
            "training_eligible": False,
            "long_term_memory_write": False,
            "formal_capability_claim": False,
            "reason": "feature_review_is_required_for_candidate_training_but_is_not_a_capability_gate",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PG-55 不变性特征漏斗",
        "",
        f"训练侧行数：`{len(rows)}`；来源数：`{audit['source_count']}`；族数：`{audit['family_count']}`。",
        "",
        "| stage | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in audit["stage_counts"].items())
    lines.extend([
        "",
        "保留特征：" + ", ".join(f"`{name}`" for name in audit["accepted_features"]),
        f"Codex 审核：`{review['decision']}`；审核证据：`{review['review_evidence_sha256']}`。",
        "PG-42 framed、seed 419 和 template_injection 不进入本漏斗，留作盲测。",
        "",
        "训练晋升/长期记忆：`False/False`。",
    ])
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "rows": len(rows),
        "stage_counts": audit["stage_counts"],
        "accepted_features": audit["accepted_features"],
        "review": review["decision"],
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
