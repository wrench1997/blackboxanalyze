"""Remote A800 training/evaluation for PG-283 multi-step feedback policy.

The runner is intentionally remote-only.  It trains on PG-283 process
supervision and evaluates route/family holdouts plus evaluation-only hard
negatives.  A passing process score is not a live payload claim: remote
Docker is probed separately and the promotion gate remains closed until a
typed target evaluator produces fresh evidence.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg283_feedback_policy_dataset_v1.json"
AUDIT = RESEARCH / "pg283_feedback_policy_dataset_audit_v1.json"
HARD = RESEARCH / "pg283_feedback_policy_hard_negative_v1.json"
REMOTE_PROBE = RESEARCH / "pg280_remote_docker_probe_v2.json"
OUT_DIR = ROOT / "artifacts" / "pg283-feedback-policy"
CHECKPOINT = OUT_DIR / "pg283_feedback_policy.pt"
REPORT = RESEARCH / "pg283_feedback_policy_report_v1.json"
TRACE = RESEARCH / "pg283_feedback_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg283_feedback_policy_protocol_v1.json"
MARKDOWN = RESEARCH / "pg283_feedback_policy_report_v1.md"

SEEDS = (28311, 28312, 28313)
RISK_VARIANTS = {"plain_sft": 1.0, "guarded_sft": 2.0, "risk_4_0": 4.0}


def _sha(value: Any) -> str:
    import hashlib

    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    keys = sorted({key for item in values for key, value in item.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {key: {"mean": round(sum(float(item[key]) for item in values) / len(values), 6), "min": round(min(float(item[key]) for item in values), 6), "max": round(max(float(item[key]) for item in values), 6)} for key in keys}


def main() -> None:
    if os.environ.get("PG283_REMOTE_RUN") != "1":
        raise RuntimeError("PG-283 training is remote-only; set PG283_REMOTE_RUN=1 on the authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.pg283_feedback_policy import build_vocab, evaluate, train_model

    data = _load(DATASET)
    audit = _load(AUDIT)
    hard_data = _load(HARD)
    remote_probe = _load(REMOTE_PROBE)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-283 dataset audit must pass before training")
    if data.get("training_contract", {}).get("remote_a800_required") is not True:
        raise RuntimeError("PG-283 dataset is not marked remote-A800-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-283 requires remote A800 GPU0, got {assignment}")
    records = list(data.get("records") or [])
    hard_rows = list(hard_data.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train"]
    route_rows = [row for row in records if row.get("split") == "route_dev"]
    family_rows = [row for row in records if row.get("split") == "family_holdout"]
    vocab = build_vocab(train_rows)
    started = time.perf_counter()
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in RISK_VARIANTS}
    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        random.seed(seed)
        for variant, risk_weight in RISK_VARIANTS.items():
            model = train_model(train_rows, vocab, device, seed, risk_weight=risk_weight, epochs=220)
            evaluation = {
                "train": evaluate(model, train_rows, vocab, device),
                "route_dev": evaluate(model, route_rows, vocab, device),
                "family_holdout": evaluate(model, family_rows, vocab, device),
                "hard_negative": evaluate(model, hard_rows, vocab, device),
            }
            per_seed[variant].append({"seed": seed, **evaluation})
            if seed == SEEDS[-1]:
                checkpoints[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    aggregated: dict[str, Any] = {}
    for variant, values in per_seed.items():
        aggregated[variant] = {section: _aggregate([value[section] for value in values]) for section in ("train", "route_dev", "family_holdout", "hard_negative")}
    variant_summary = {
        variant: {
            "risk_weight": risk,
            "route_action_safe_exact_min": float(aggregated[variant]["route_dev"].get("action_safe_exact_accuracy", {}).get("min", 0.0)),
            "family_action_safe_exact_min": float(aggregated[variant]["family_holdout"].get("action_safe_exact_accuracy", {}).get("min", 0.0)),
            "hard_negative_safe_reject_min": float(aggregated[variant]["hard_negative"].get("safe_reject_rate", {}).get("min", 0.0)),
            "hard_negative_false_allow_max": int(aggregated[variant]["hard_negative"].get("false_allow_count", {}).get("max", 0)),
            "hard_negative_guarded_false_allow_max": int(aggregated[variant]["hard_negative"].get("guarded_false_allow_count", {}).get("max", 0)),
        }
        for variant, risk in RISK_VARIANTS.items()
    }
    # Selection is intentionally conservative.  If no variant has zero raw
    # hard-negative false-allow, the gate remains blocked instead of silently
    # selecting a model and calling the failure success.
    eligible = [
        (summary["route_action_safe_exact_min"], summary["family_action_safe_exact_min"], -summary["risk_weight"], variant)
        for variant, summary in variant_summary.items()
        if summary["hard_negative_false_allow_max"] == 0
    ]
    selected_variant = max(eligible)[-1] if eligible else min(variant_summary, key=lambda variant: (variant_summary[variant]["hard_negative_false_allow_max"], -variant_summary[variant]["hard_negative_safe_reject_min"], variant_summary[variant]["risk_weight"]))
    selected_summary = variant_summary[selected_variant]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"],
        "route_process_score_recorded": selected_summary["route_action_safe_exact_min"] >= 0.0,
        "family_process_score_recorded": selected_summary["family_action_safe_exact_min"] >= 0.0,
        "hard_negative_evaluated": len(hard_rows) > 0,
        "hard_negative_false_allow_zero": selected_summary["hard_negative_false_allow_max"] == 0,
        "literal_payload_excluded": data.get("training_contract", {}).get("literal_payload_values_out_of_context") is True,
        "remote_docker_honest": remote_probe.get("status") != "available" or remote_probe.get("real_application_gold_rows", 0) == 0,
        "promotion_blocked": True,
    }
    gate_status = "passed" if all(checks.values()) else "blocked"
    report = {
        "protocol_id": "pg283-feedback-policy-v1",
        "schema_version": "pg283-feedback-policy-report-v1",
        "status": "completed_remote_pg283_feedback_policy",
        "source": {
            "dataset": str(DATASET.relative_to(ROOT).as_posix()),
            "dataset_sha256": data["dataset_sha256"],
            "dataset_audit": str(AUDIT.relative_to(ROOT).as_posix()),
            "dataset_audit_sha256": audit["audit_sha256"],
            "hard_negative_dataset": str(HARD.relative_to(ROOT).as_posix()),
            "hard_negative_sha256": hard_data["dataset_sha256"],
            "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()),
            "remote_docker_status": remote_probe.get("status", "unknown"),
            "real_application_gold_rows": int(remote_probe.get("real_application_gold_rows", 0) or 0),
            "remote_host": "112.111.7.91:60228",
            "loopback_only": True,
            "external_network": False,
            "literal_payload_in_context": False,
            "raw_response_body_in_context": False,
            "live_send": False,
        },
        "device": assignment,
        "split": {"train": len(train_rows), "route_dev": len(route_rows), "family_holdout": len(family_rows), "hard_negative": len(hard_rows), "seeds": list(SEEDS)},
        "aggregated": aggregated,
        "per_seed": per_seed,
        "risk_weight_sweep": {"variants": variant_summary, "selected_variant": selected_variant, "selection_rule": "zero hard-negative false-allow first; then highest route/family action+safe exact; ties prefer lower risk weight"},
        "hypothesis_gate": {"status": gate_status, "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"live_target_evaluator": False, "real_application_gold": False, "generated_transition_templates": True, "cross_template_generalization": False}, "reasons": ["transition rows are generated from fixed process templates", "remote Docker/evaluator unavailable", "no literal payload success or fresh target replay"], "claim_allowed": False},
        "policy_scope": {"outputs": ["next_action", "abstract_probe_class", "channel", "encoding", "safe_to_send"], "multi_step_feedback": True, "family_hidden": True, "literal_payload_generation": False, "live_send": False, "typed_oracle_required_for_confirmation": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "process-policy training only; remote Docker is unavailable and no live evaluator evidence exists"},
        "formal_conclusion": "PG-283 评估了 failure→diagnose→repair→replay 的抽象动作学习和安全门；hard-negative false-allow 若不为零就保持 blocked，不能把训练分数当成真实 payload 成功。",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = _sha(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg283-feedback-policy-checkpoint-v1", "assignment": assignment, "vocabulary": vocab, "selected_variant": selected_variant, "state": checkpoints.get(selected_variant, {})}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg283-feedback-policy-trace-v1", "report_sha256": report["report_sha256"], "source_dataset_sha256": data["dataset_sha256"], "hard_negative_sha256": hard_data["dataset_sha256"], "selected_variant": selected_variant, "gate_status": gate_status, "scientific_gate_status": "blocked", "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "live_send": False}
    trace["trace_sha256"] = _sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg283-feedback-policy-protocol-v1", "remote_a800_gpu0_only": True, "multi_step_feedback_states": ["negative_clean", "reference_clean", "typed_ready", "candidate_gap", "typed_effect", "reference_mismatch", "fresh_missing", "replay_done"], "hard_negative_evaluation_only": True, "literal_payload_generation": False, "live_send": False, "remote_docker_required_for_promotion": True, "scientific_gate": "blocked_until_fresh_target_and_non_template_holdout", "report_sha256": report["report_sha256"], "next_experiment": "PG-284: connect PG-283 guarded next-action head to an available target evaluator and measure true GET/POST replay success separately from abstention."}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-283 multi-step feedback policy", "", f"engineering_gate={gate_status}", "scientific_gate=blocked", f"selected={selected_variant}", f"route action+safe exact={selected_summary['route_action_safe_exact_min']}", f"family action+safe exact={selected_summary['family_action_safe_exact_min']}", f"hard-negative false-allow={selected_summary['hard_negative_false_allow_max']}", f"remote Docker={remote_probe.get('status', 'unknown')}", "template-derived transitions; literal payload/live send=false", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "engineering_gate": report["hypothesis_gate"], "scientific_gate": report["scientific_gate"], "selected_variant": selected_variant, "variant_summary": variant_summary, "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
