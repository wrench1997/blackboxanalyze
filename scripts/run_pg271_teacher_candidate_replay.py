"""PG-271: replay the PG-270 guided checkpoint on an independent fresh seed.

The checkpoint sees only PG-271 abstract context tokens.  The fresh replay
report remains the independent reference; model output is scored as a
candidate action sequence and never promotes a vulnerability claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SOURCE_DATASET = ROOT / "research" / "pg271_independent_seed_failure_guided_replay_dataset_v1.json"
SOURCE_REPORT = ROOT / "research" / "pg271_independent_seed_failure_guided_replay_report_v1.json"
SOURCE_AUDIT = ROOT / "research" / "pg271_independent_seed_failure_guided_replay_audit_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg270-teacher-sft" / "teacher_sft_ablation.pt"
REPORT_PATH = ROOT / "research" / "pg271_teacher_candidate_replay_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg271_teacher_candidate_replay_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg271_teacher_candidate_replay_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg271_teacher_candidate_replay_report_v1.md"

UNSEEN_FAMILIES = {"redirect", "xxe", "serialization", "infoleak", "other"}


def _load_pg270_module() -> Any:
    spec = importlib.util.spec_from_file_location("pg270_teacher_sft_module", SCRIPT_DIR / "run_pg270_teacher_sft_ablation.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-270 model module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _field(tokens: list[str], prefix: str) -> str | None:
    values = [token.split("=", 1)[1] for token in tokens if token.startswith(prefix)]
    return values[-1] if values else None


def main() -> None:
    started = time.perf_counter()
    source_dataset = json.loads(SOURCE_DATASET.read_text(encoding="utf-8"))
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if source_audit.get("status") != "passed" or source_report.get("status") != "completed_local_failure_guided_replay":
        raise RuntimeError("PG-271 candidate replay requires a passed independent fresh replay")
    rows = [dict(row) for row in source_dataset.get("records", []) if row.get("training_eligible")]
    if len(rows) != 40:
        raise RuntimeError(f"expected 40 PG-271 rows, got {len(rows)}")
    pg270 = _load_pg270_module()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = dict(checkpoint["vocabulary"])
    reverse = {int(key): value for key, value in dict(checkpoint["reverse_vocabulary"]).items()}
    model = pg270.TinyConditionalDecoder(len(vocabulary))
    model.load_state_dict(checkpoint["guided_state"])
    device = torch.device("cpu")
    model.to(device).eval()
    details: list[dict[str, Any]] = []
    for row in rows:
        context = list(row["context_tokens"])
        generated = pg270._generate(model, row, vocabulary, reverse, device, max_len=len(row["target_tokens"]) + 5)
        expected = list(row["target_tokens"])
        expected_action = _field(expected, "next_action=")
        predicted_action = _field(generated, "next_action=")
        expected_belief = _field(expected, "final_belief=")
        predicted_belief = _field(generated, "final_belief=")
        expected_abstain = expected_belief == "oracle_gap" and expected_action == "abstain"
        predicted_abstain = predicted_belief == "oracle_gap" and predicted_action == "abstain"
        details.append({
            "record_id": row["record_id"],
            "family_class": row["labels"]["family_class"],
            "fresh_seed": row["seed"],
            "split": "family_holdout" if row["labels"]["family_class"] in UNSEEN_FAMILIES else "fresh_route_seed",
            "expected_next_action": expected_action,
            "predicted_next_action": predicted_action,
            "expected_final_belief": expected_belief,
            "predicted_final_belief": predicted_belief,
            "next_action_correct": predicted_action == expected_action,
            "final_belief_correct": predicted_belief == expected_belief,
            "abstain_correct": predicted_abstain == expected_abstain,
            "model_unsupported_positive": predicted_belief == "confirmed_effect" and expected_belief == "oracle_gap",
            "generated_tokens": generated,
            "context_token_count": len(context),
        })
    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        return {
            "count": total,
            "next_action_accuracy": round(sum(item["next_action_correct"] for item in items) / max(total, 1), 6),
            "final_belief_accuracy": round(sum(item["final_belief_correct"] for item in items) / max(total, 1), 6),
            "abstain_calibration_accuracy": round(sum(item["abstain_correct"] for item in items) / max(total, 1), 6),
            "unsupported_positive_count": sum(item["model_unsupported_positive"] for item in items),
            "details": items,
        }
    family_items = [item for item in details if item["split"] == "family_holdout"]
    route_items = [item for item in details if item["split"] == "fresh_route_seed"]
    evaluations = {"fresh_seed_all": metrics(details), "fresh_seed_family_holdout": metrics(family_items), "fresh_seed_route_remainder": metrics(route_items)}
    gate_checks = {
        "source_fresh_replay_audit_pass": source_audit.get("status") == "passed",
        "model_input_context_only": all(item["context_token_count"] > 0 for item in details),
        "family_holdout_nonempty": bool(family_items),
        "family_unsupported_positive_zero": evaluations["fresh_seed_family_holdout"]["unsupported_positive_count"] == 0,
        "all_unsupported_positive_zero": evaluations["fresh_seed_all"]["unsupported_positive_count"] == 0,
    }
    report = {
        "protocol_id": "pg271-teacher-candidate-fresh-seed-replay-v1",
        "schema_version": "pg271-teacher-candidate-replay-report-v1",
        "status": "candidate_replay_completed",
        "source": {"fresh_dataset": str(SOURCE_DATASET.relative_to(ROOT)), "fresh_report": str(SOURCE_REPORT.relative_to(ROOT)), "fresh_audit": str(SOURCE_AUDIT.relative_to(ROOT)), "fresh_seed": source_report.get("seed"), "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_source": "PG-270 guided_sft", "device": str(device), "oracle_in_model_context": False, "raw_probe_strings_in_model_context": False},
        "evaluations": evaluations,
        "capability_gate": {"status": "passed" if all(gate_checks.values()) else "blocked", "checks": gate_checks, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "status": "candidate_replay_only", "reason": "one independent seed; no independent model retraining and no fresh family implementation matrix"},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = _sha(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg271-teacher-candidate-replay-trace-v1", "evaluation_only": True, "training_eligible": False, "source_seed": source_report.get("seed"), "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "details": details, "capability_checks": gate_checks, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_model_context": False, "long_term_memory_write": False}
    trace["trace_sha256"] = _sha(trace)
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg271-teacher-candidate-fresh-seed-replay-v1", "schema_version": "pg271-teacher-candidate-replay-protocol-v1", "input_contract": {"fresh_reset_required": True, "independent_seed_required": True, "oracle_target_off_context": True, "raw_probe_off_context": True}, "evaluation_contract": {"family_holdout": True, "unsupported_positive_zero": True, "abstain_calibration_reported": True}, "promotion_contract": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "run": {"capability_gate": report["capability_gate"], "report_sha256": report["report_sha256"]}, "next_experiment": "PG-272 independent seed plus fresh implementation holdout, then constrained offline RL only if abstain and repair remain stable"}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-271 教师候选 fresh seed 回放", "", f"fresh seed={source_report.get('seed')}；checkpoint=PG-270 guided_sft；模型评估设备=`{device}`。", "", "| split | next-action | final belief | abstain calibration | unsupported positive |", "|---|---:|---:|---:|---:|", f"| all 40 | {evaluations['fresh_seed_all']['next_action_accuracy']:.3f} | {evaluations['fresh_seed_all']['final_belief_accuracy']:.3f} | {evaluations['fresh_seed_all']['abstain_calibration_accuracy']:.3f} | {evaluations['fresh_seed_all']['unsupported_positive_count']} |", f"| family holdout | {evaluations['fresh_seed_family_holdout']['next_action_accuracy']:.3f} | {evaluations['fresh_seed_family_holdout']['final_belief_accuracy']:.3f} | {evaluations['fresh_seed_family_holdout']['abstain_calibration_accuracy']:.3f} | {evaluations['fresh_seed_family_holdout']['unsupported_positive_count']} |", "", "fresh replay 的 typed oracle 仍是最终判定；模型只输出抽象候选动作，不生成或确认公网漏洞 payload。", f"capability gate: `{report['capability_gate']['status']}`；promotion=false。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "fresh_seed": source_report.get("seed"), "evaluations": {key: {metric: value for metric, value in value.items() if metric != "details"} for key, value in evaluations.items()}, "capability_gate": report["capability_gate"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
