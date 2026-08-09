"""PG-90: read-only cross-seed Codex review after PG-88/PG-89.

The review consolidates the new evidence without changing a checkpoint or
promoting a catalog.  It distinguishes a successful frozen cross-seed
capability result from the still-missing independent implementation and
human-approved memory/production gates.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg90_cross_seed_codex_review_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg90_cross_seed_codex_review_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg90_cross_seed_codex_review_report_v1.md"
PROTOCOL_ID = "pg-pk-90-cross-seed-codex-review-v1"


def _read(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _run_regression() -> dict[str, Any]:
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(r"(\d+) passed", output)
    return {"return_code": int(completed.returncode), "passed_count": int(match.group(1)) if match else 0, "warning_count": output.lower().count("warning")}


def run() -> dict[str, Any]:
    pg82 = _read("pg82_effect_geometry_source_holdout_transformer_report_v1.json")
    pg83 = _read("pg83_cross_seed_geometry_holdout_transformer_report_v1.json")
    pg84 = _read("pg84_cross_dataset_frozen_replay_report_v1.json")
    pg85 = _read("pg85_multisurface_composite_transformer_report_v1.json")
    pg87 = _read("pg87_promotion_review_report_v1.json")
    pg88 = _read("pg88_independent_html_dom_matrix_report_v1.json")
    pg89 = _read("pg89_pg86_frozen_html_dom_replay_report_v1.json")
    regression = _run_regression()
    checks = {
        "pg88_collection_passed": pg88["hard_gate"]["status"] == "passed" and pg88["metrics"]["triplet_case_count"] == 28,
        "pg88_four_seed_fresh_targets": pg88["metrics"]["unique_target_instance_count"] == 28 and pg88["metrics"]["independent_seed_count"] == 4,
        "pg88_get_post_and_negative_oracles": pg88["metrics"]["get_post_covered"] == {"GET": 24, "POST": 4} and pg88["metrics"]["typed_negative_oracle_count"] == 56,
        "pg89_frozen_replay_passed": pg89["capability_gate"]["status"] == "passed",
        "pg89_recall_and_seed_stability": pg89["metrics"]["confirm_recall"] >= 0.80 and pg89["metrics"]["seed_min_confirm_recall"] >= 0.75,
        "pg89_false_accept_zero": pg89["metrics"]["false_accept_count"] == 0,
        "pg89_unknown_token_zero": pg89["metrics"]["unknown_token_count"] == 0 and pg89["metrics"]["reference_unknown_token_count"] == 0,
        "pg89_correct_abstain_present": pg89["metrics"]["abstain_count"] > 0,
        "pg82_pg83_prior_gates_preserved": pg82["capability_gate"]["status"] == "passed" and pg83["capability_gate"]["status"] == "passed",
        "failed_cross_dataset_controls_preserved": pg84["hard_gate"]["status"] == "blocked" and pg84["metrics"]["confirm_recall"] == 0.0 and pg85["capability_gate"]["status"] == "blocked",
        "pg87_memory_production_blocked": pg87["decision"]["long_term_memory_promotion_allowed"] is False and pg87["decision"]["production_web_vulnerability_detector_claim_allowed"] is False,
        "raw_and_memory_free": pg88["promotion"]["memory_promotion_allowed"] is False and pg89["promotion"]["memory_promotion_allowed"] is False and pg89["source"]["memory_write"] is False,
        "full_regression": regression["return_code"] == 0 and regression["passed_count"] >= 419,
    }
    passed = all(checks.values())
    review = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg90-cross-seed-codex-review-report-v1",
        "reviewer": "Codex / root",
        "review_scope": "authorized loopback/Docker causal trace learning only",
        "status": "passed_for_controlled_offline_scale" if passed else "blocked",
        "checks": checks,
        "regression": regression,
        "evidence": {
            "independent_collection": "research/pg88_independent_html_dom_matrix_report_v1.json",
            "frozen_replay": "research/pg89_pg86_frozen_html_dom_replay_report_v1.json",
            "prior_geometry_holdout": "research/pg83_cross_seed_geometry_holdout_transformer_report_v1.json",
            "negative_cross_dataset_control": "research/pg84_cross_dataset_frozen_replay_report_v1.json",
            "blocked_composite_control": "research/pg85_multisurface_composite_transformer_report_v1.json",
        },
        "decision": {
            "controlled_offline_training_scale_allowed": bool(passed),
            "new_samples_must_remain_evaluation_gated": True,
            "formal_model_promotion_allowed": False,
            "long_term_memory_promotion_allowed": False,
            "production_web_vulnerability_detector_claim_allowed": False,
            "checkpoint_mutation_by_review": False,
            "memory_write_by_review": False,
            "reason": "PG89 passes four unseen seeds with zero false accepts and zero unknown tokens, but the collector still reuses one implementation and the prior PG84/PG85 transfer failures remain; independent implementation, larger family diversity and explicit human approval are still required.",
        },
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg90-cross-seed-codex-review-protocol-v1",
        "required_checks": list(checks),
        "memory_write": False,
        "checkpoint_mutation": False,
        "run_result": review["decision"] | {"status": review["status"]},
        "next_experiment": "PG91 independent implementation or second local vulnerable target family",
    }
    REPORT_PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-90 Cross-seed Codex review\n\n" + f"status=`{review['status']}`；regression={regression['passed_count']} passed；PG89 recall={pg89['metrics']['confirm_recall']}；seed-min={pg89['metrics']['seed_min_confirm_recall']}；false_accept={pg89['metrics']['false_accept_count']}。\n\ncontrolled offline scale={review['decision']['controlled_offline_training_scale_allowed']}；formal model promotion={review['decision']['formal_model_promotion_allowed']}；long-term memory={review['decision']['long_term_memory_promotion_allowed']}；production claim={review['decision']['production_web_vulnerability_detector_claim_allowed']}。\n", encoding="utf-8")
    return review


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["status"], "regression_passed": result["regression"]["passed_count"], "controlled_offline_training_scale_allowed": result["decision"]["controlled_offline_training_scale_allowed"], "formal_model_promotion_allowed": result["decision"]["formal_model_promotion_allowed"], "long_term_memory_promotion_allowed": result["decision"]["long_term_memory_promotion_allowed"]}, ensure_ascii=False, indent=2))
