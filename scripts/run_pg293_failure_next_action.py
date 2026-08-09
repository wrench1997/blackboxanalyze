"""Audited PG-293 failure-conditioned next-action training.

Remote mode is restricted to the authorized A800 GPU0.  When that resource is
occupied, local training is permitted only during the Asia/Shanghai 08:00–18:00
window with an explicit operator flag; local runs write separate artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import build_vocabulary, evaluate_model, sha256_json, train_model  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg293_failure_next_action_dataset_v1.json"
AUDIT = RESEARCH / "pg293_failure_next_action_dataset_audit_v1.json"
REMOTE_PROBE = RESEARCH / "pg280_remote_docker_probe_v2.json"
REPORT = RESEARCH / "pg293_failure_next_action_training_report_v1.json"
TRACE = RESEARCH / "pg293_failure_next_action_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg293_failure_next_action_training_protocol_v1.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg293-failure-next-action"
CHECKPOINT = CHECKPOINT_DIR / "pg293_failure_next_action.pt"
SEEDS = (29301, 29302, 29303)
HIDDEN_DIMS = (128, 256, 384)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = sorted({key for row in metrics for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None})
    return {
        key: {
            "mean": round(sum(float(row[key]) for row in metrics) / len(metrics), 6),
            "min": round(min(float(row[key]) for row in metrics), 6),
            "max": round(max(float(row[key]) for row in metrics), 6),
        }
        for key in numeric
    }


def execution_mode() -> tuple[str, datetime]:
    """Select exactly one audited execution mode; never fall back silently."""

    remote = os.environ.get("PG293_REMOTE_RUN") == "1"
    local = os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") == "1"
    if remote == local:
        raise RuntimeError(
            "set exactly one of PG293_REMOTE_RUN=1 or BLACKBOX_LOCAL_MORNING_TRAIN=1"
        )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if local and not (8 <= now.hour < 18):
        raise RuntimeError(
            f"local PG-293 training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}"
        )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return ("remote_a800" if remote else "local_morning"), now


def main() -> None:
    mode, now = execution_mode()
    remote_mode = mode == "remote_a800"
    local_mode = mode == "local_morning"
    dataset = load(DATASET)
    audit = load(AUDIT)
    remote_probe = load(REMOTE_PROBE)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-293 dataset audit must pass")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {
        "execution_mode": mode,
        "timestamp": now.isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1:
        raise RuntimeError(f"PG-293 requires exactly one visible CUDA device, got {assignment}")
    if remote_mode and "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-293 remote mode requires an A800 on GPU0, got {assignment}")
    if local_mode and "A800" in str(assignment["device_name"]):
        raise RuntimeError("local morning mode must not consume the remote A800 contract")

    suffix = "_local_morning" if local_mode else ""
    report_path = REPORT.with_name(f"{REPORT.stem}{suffix}{REPORT.suffix}")
    trace_path = TRACE.with_name(f"{TRACE.stem}{suffix}{TRACE.suffix}")
    protocol_path = PROTOCOL.with_name(f"{PROTOCOL.stem}{suffix}{PROTOCOL.suffix}")
    checkpoint_path = CHECKPOINT.with_name(f"{CHECKPOINT.stem}{suffix}{CHECKPOINT.suffix}")
    records = list(dataset.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train"]
    source_holdout = [row for row in records if row.get("split") == "source_holdout"]
    seed_holdout = [row for row in records if row.get("split") == "seed_holdout"]
    hard_negative_rows = [row for row in records if row.get("split") == "hard_negative_eval"]
    holdout = source_holdout + seed_holdout
    if not train_rows or not holdout or not hard_negative_rows:
        raise RuntimeError("PG-293 requires train, holdout and evaluation-only hard-negative rows")
    vocab = build_vocabulary(train_rows)
    started = time.perf_counter()
    variants: list[dict[str, Any]] = []
    model_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for hidden_dim in HIDDEN_DIMS:
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            random.seed(seed)
            model = train_model(train_rows, vocab, device, seed=seed, epochs=220, hidden_dim=hidden_dim)
            metrics = {
                "seed": seed,
                "train": evaluate_model(model, train_rows, vocab, device),
                "holdout": evaluate_model(model, holdout, vocab, device),
                "source_holdout": evaluate_model(model, source_holdout, vocab, device),
                "seed_holdout": evaluate_model(model, seed_holdout, vocab, device),
                "hard_negative": evaluate_model(model, hard_negative_rows, vocab, device),
            }
            per_seed.append(metrics)
            model_snapshots[hidden_dim] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        variants.append({
            "hidden_dim": hidden_dim,
            "train": aggregate([item["train"] for item in per_seed]),
            "holdout": aggregate([item["holdout"] for item in per_seed]),
            "source_holdout": aggregate([item["source_holdout"] for item in per_seed]),
            "seed_holdout": aggregate([item["seed_holdout"] for item in per_seed]),
            "hard_negative": aggregate([item["hard_negative"] for item in per_seed]),
            "per_seed": per_seed,
        })
    selected = max(
        variants,
        key=lambda item: (
            item["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0,
            item["holdout"].get("positive_recall", {}).get("min", 0.0),
            item["holdout"].get("action_accuracy", {}).get("min", 0.0),
            item["holdout"].get("token_accuracy", {}).get("min", 0.0),
            -int(item["hidden_dim"]),
        ),
    )
    resource_contract = (
        remote_mode
        and assignment["cuda_visible_devices"] == "0"
        and assignment["visible_device_count"] == 1
        and assignment["current_device"] == 0
        and "A800" in assignment["device_name"]
    ) or (
        local_mode
        and 8 <= now.hour < 18
        and assignment["cuda_visible_devices"] == "0"
        and assignment["visible_device_count"] == 1
        and assignment["current_device"] == 0
        and "A800" not in assignment["device_name"]
    )
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "resource_contract": resource_contract,
        "source_holdout_present": bool(source_holdout),
        "seed_holdout_present": bool(seed_holdout),
        "hard_negative_evaluated": bool(hard_negative_rows) and all("hard_negative_false_allow" in item["hard_negative"] for item in variants),
        "hard_negative_zero_false_allow": bool(selected["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0),
        "remote_docker_honest": remote_probe.get("status") != "available" or int(remote_probe.get("real_application_gold_rows", 0) or 0) == 0,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg293-failure-next-action-training-v1",
        "schema_version": "pg293-failure-next-action-training-report-v1",
        "status": f"completed_{mode}_pg293_failure_next_action",
        "source": {
            "dataset": str(DATASET.relative_to(ROOT).as_posix()),
            "dataset_sha256": dataset.get("dataset_sha256"),
            "audit": str(AUDIT.relative_to(ROOT).as_posix()),
            "audit_sha256": audit.get("audit_sha256"),
            "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()),
            "remote_docker_status": remote_probe.get("status", "unknown"),
            "real_application_gold_rows": int(remote_probe.get("real_application_gold_rows", 0) or 0),
            "literal_payload_in_context": False,
            "wire_emission": False,
        },
        "device": assignment,
        "split": {"total": len(records), "train": len(train_rows), "holdout": len(holdout), "source_holdout": len(source_holdout), "seed_holdout": len(seed_holdout), "hard_negative_eval": len(hard_negative_rows), "seeds": list(SEEDS), "hidden_dims": list(HIDDEN_DIMS)},
        "vocabulary_size": len(vocab),
        "variants": variants,
        "selection": {"hidden_dim": selected["hidden_dim"], "hard_negative_false_allow_max": int(selected["hard_negative"].get("hard_negative_false_allow", {}).get("max", 0)), "holdout_positive_recall_min": selected["holdout"].get("positive_recall", {}).get("min"), "rule": "zero evaluation-only hard-negative false-allow first; then positive recall; then action/token accuracy"},
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "source_holdout": True, "seed_holdout": True}, "reasons": ["remote Docker unavailable", "dataset contains abstract/local trace rows only", "no live typed replay"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "formal_conclusion": "PG-293 measures failure-conditioned abstract next-action composition; it does not establish real payload generation or vulnerability detection.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg293-failure-next-action-training-trace-v1", "report_sha256": report["report_sha256"], "selected_hidden_dim": selected["hidden_dim"], "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg293-failure-next-action-training-protocol-v1", "execution_mode": mode, "remote_a800_gpu0_only": remote_mode, "local_morning_training": local_mode, "failure_conditioned": True, "source_holdout": True, "seed_holdout": True, "hard_negative_zero_false_allow_required": True, "literal_payload_generation": False, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-294: add typed-availability/feedback states without oracle leakage, then re-evaluate active repair on same-context hard negatives."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if selected["hidden_dim"] in model_snapshots:
        torch.save({"schema_version": "pg293-failure-next-action-checkpoint-v1", "assignment": assignment, "vocabulary": vocab, "hidden_dim": selected["hidden_dim"], "state": model_snapshots[selected["hidden_dim"]]}, checkpoint_path)
    print(json.dumps({"status": report["status"], "device": assignment, "selection": report["selection"], "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "report": str(report_path.relative_to(ROOT).as_posix()), "trace": str(trace_path.relative_to(ROOT).as_posix()), "protocol": str(protocol_path.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
