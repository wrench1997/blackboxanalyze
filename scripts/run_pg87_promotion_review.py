"""PG-87: Codex review of the controlled-scaling gate.

This is a read-only promotion audit.  It does not alter a checkpoint or
write long-term memory.  A pass permits larger *offline, evaluation-gated*
experiments only; production claims and memory promotion remain explicitly
false until a later human-approved review with more independent targets.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg87_promotion_review_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg87_promotion_review_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg87_promotion_review_report_v1.md"
PROTOCOL_ID = "pg-pk-87-codex-promotion-review-v1"


def _read(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _run_regression() -> dict[str, Any]:
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(r"(\d+) passed", output)
    return {"return_code": int(completed.returncode), "passed_count": int(match.group(1)) if match else 0, "warning_count": output.count("warning")}


def run() -> dict[str, Any]:
    collector = _read("pg82_canonical_triplet_collector_report_v1.json")
    pg82 = _read("pg82_effect_geometry_source_holdout_transformer_report_v1.json")
    pg83 = _read("pg83_cross_seed_geometry_holdout_transformer_report_v1.json")
    pg84 = _read("pg84_cross_dataset_frozen_replay_report_v1.json")
    pg86 = _read("pg86_surface_signal_composite_transformer_report_v1.json")
    regression = _run_regression()
    checks = {
        "collection_hard_gate": collector["hard_gate"]["status"] == "passed",
        "real_benign_negative_probe": collector["metrics"]["negative_probe_positive_requested_count"] == 0,
        "get_post_balanced": collector["metrics"]["get_post_counts"] == {"GET": 135, "POST": 135},
        "pg82_source_holdout": pg82["capability_gate"]["status"] == "passed" and pg82["metrics"]["source_holdout"]["confirm_recall"] >= 0.80,
        "pg83_cross_seed": pg83["capability_gate"]["status"] == "passed" and pg83["source"]["seed_holdout"]["dev"] == [7911],
        "pg86_cross_dataset": pg86["capability_gate"]["status"] == "passed" and pg86["metrics"]["cross_dataset_holdout"]["confirm_recall"] >= 0.80 and pg86["metrics"]["cross_dataset_holdout"]["typed_positive_count"] >= 7,
        "all_false_accept_zero": all(report["metrics"].get("source_holdout", {}).get("false_accept_count", 0) == 0 and report["metrics"].get("cross_dataset_holdout", {}).get("false_accept_count", 0) == 0 for report in (pg86,)),
        "unknown_strict_abstain": pg82["metrics"]["unknown_family_holdout"]["strict_abstain"] and pg83["metrics"]["unknown_family_holdout"]["strict_abstain"] and pg86["metrics"]["unknown_family_holdout"]["strict_abstain"],
        "failed_cross_dataset_replay_preserved": pg84["hard_gate"]["status"] == "blocked" and pg84["metrics"]["confirm_recall"] == 0.0 and pg84["metrics"]["false_accept_count"] == 0,
        "raw_persistence_forbidden": all(report.get("promotion", {}).get("memory_promotion_allowed") is False for report in (pg82, pg83, pg86)),
        "full_regression": regression["return_code"] == 0 and regression["passed_count"] >= 416,
    }
    passed = all(checks.values())
    review = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg87-promotion-review-report-v1",
        "reviewer": "Codex / root",
        "review_scope": "authorized loopback/Docker causal trace learning only",
        "status": "passed_for_controlled_offline_scale" if passed else "blocked",
        "checks": checks,
        "regression": regression,
        "evidence": {
            "collector": "research/pg82_canonical_triplet_collector_report_v1.json",
            "source_holdout": "research/pg82_effect_geometry_source_holdout_transformer_report_v1.json",
            "cross_seed": "research/pg83_cross_seed_geometry_holdout_transformer_report_v1.json",
            "negative_cross_dataset_control": "research/pg84_cross_dataset_frozen_replay_report_v1.json",
            "surface_signal_candidate": "research/pg86_surface_signal_composite_transformer_report_v1.json",
        },
        "decision": {
            "controlled_offline_training_scale_allowed": bool(passed),
            "new_samples_must_remain_evaluation_gated": True,
            "long_term_memory_promotion_allowed": False,
            "production_web_vulnerability_detector_claim_allowed": False,
            "reason": "PG86 passes current capability gates, but memory/production require explicit human approval and a larger independent target set; PG84 failure remains a known boundary.",
        },
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg87-promotion-review-protocol-v1",
        "required_checks": list(checks),
        "memory_write": False,
        "checkpoint_mutation": False,
        "run_result": review["decision"] | {"status": review["status"]},
    }
    REPORT_PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-87 Codex promotion review\n\n" + f"status=`{review['status']}`；regression={regression['passed_count']} passed；controlled offline scale={review['decision']['controlled_offline_training_scale_allowed']}；long-term memory={review['decision']['long_term_memory_promotion_allowed']}；production claim={review['decision']['production_web_vulnerability_detector_claim_allowed']}。\n\nPG84 failed replay remains preserved as a boundary case.\n", encoding="utf-8")
    return review


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["status"], "regression_passed": result["regression"]["passed_count"], "controlled_offline_training_scale_allowed": result["decision"]["controlled_offline_training_scale_allowed"], "long_term_memory_promotion_allowed": result["decision"]["long_term_memory_promotion_allowed"]}, ensure_ascii=False, indent=2))
