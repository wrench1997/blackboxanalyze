"""Run PG-294 on the local morning GPU under the explicit resource gate.

The experiment is deliberately offline and abstract.  It trains a small
next-action decoder on state cells, evaluates same-context opposite targets,
and writes separate local artifacts.  It never starts a target, emits a wire,
or promotes a payload catalog.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import random
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import build_vocabulary, evaluate_model, sha256_json, train_model  # noqa: E402
from app.pg294_active_repair import evaluate_question_metrics  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg294_active_repair_dataset_v1.json"
AUDIT = RESEARCH / "pg294_active_repair_dataset_audit_v1.json"
REPORT = RESEARCH / "pg294_active_repair_training_report_v1_local_morning.json"
TRACE = RESEARCH / "pg294_active_repair_training_trace_v1_local_morning.json"
PROTOCOL = RESEARCH / "pg294_active_repair_training_protocol_v1_local_morning.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg294-active-repair"
CHECKPOINT = CHECKPOINT_DIR / "pg294_active_repair_local_morning.pt"
SEEDS = (29401, 29402, 29403)
HIDDEN_DIMS = (128, 256, 384)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def verify_local_morning() -> tuple[dict[str, Any], datetime]:
    if os.environ.get("PG294_LOCAL_RUN") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-294 requires PG294_LOCAL_RUN=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-294 local training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("PG-294 local training requires one visible CUDA device")
    assignment = {
        "execution_mode": "local_morning",
        "timestamp": now.isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
    }
    if assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or assignment["current_device"] != 0:
        raise RuntimeError(f"PG-294 requires exactly one visible local GPU0, got {assignment}")
    if "A800" in assignment["device_name"]:
        raise RuntimeError("PG-294 local mode refuses the remote A800 contract")
    return assignment, now


def main() -> None:
    assignment, now = verify_local_morning()
    dataset = load(DATASET)
    audit = load(AUDIT)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-294 dataset audit must pass")
    records = list(dataset.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    source_holdout = [row for row in records if row.get("split") == "source_holdout"]
    seed_holdout = [row for row in records if row.get("split") == "seed_holdout"]
    hard_negative_rows = [row for row in records if row.get("split") == "hard_negative_eval"]
    source_missing = [row for row in source_holdout if row.get("state_id") == "missing_key"]
    seed_missing = [row for row in seed_holdout if row.get("state_id") == "missing_key"]
    all_missing = [row for row in records if row.get("state_id") == "missing_key"]
    if not train_rows or not source_holdout or not seed_holdout or not hard_negative_rows:
        raise RuntimeError("PG-294 requires train, source holdout, seed holdout and hard-negative rows")
    vocab = build_vocabulary(train_rows)
    started = time.perf_counter()
    variants: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    device = torch.device("cuda")
    for hidden_dim in HIDDEN_DIMS:
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            random.seed(seed)
            model = train_model(train_rows, vocab, device, seed=seed, epochs=180, hidden_dim=hidden_dim)
            per_seed.append({
                "seed": seed,
                "train": evaluate_model(model, train_rows, vocab, device),
                "source_holdout": evaluate_model(model, source_holdout, vocab, device),
                "seed_holdout": evaluate_model(model, seed_holdout, vocab, device),
                "hard_negative": evaluate_model(model, hard_negative_rows, vocab, device),
                "missing_question": evaluate_question_metrics(model, all_missing, vocab, device),
                "source_missing_question": evaluate_question_metrics(model, source_missing, vocab, device),
                "seed_missing_question": evaluate_question_metrics(model, seed_missing, vocab, device),
            })
            snapshots[hidden_dim] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        variants.append({
            "hidden_dim": hidden_dim,
            "train": aggregate([item["train"] for item in per_seed]),
            "source_holdout": aggregate([item["source_holdout"] for item in per_seed]),
            "seed_holdout": aggregate([item["seed_holdout"] for item in per_seed]),
            "hard_negative": aggregate([item["hard_negative"] for item in per_seed]),
            "missing_question": aggregate([item["missing_question"] for item in per_seed]),
            "source_missing_question": aggregate([item["source_missing_question"] for item in per_seed]),
            "seed_missing_question": aggregate([item["seed_missing_question"] for item in per_seed]),
            "per_seed": per_seed,
        })
    selected = max(
        variants,
        key=lambda item: (
            item["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0,
            item["seed_holdout"].get("positive_recall", {}).get("min", 0.0) or 0.0,
            item["source_holdout"].get("positive_recall", {}).get("min", 0.0) or 0.0,
            item["seed_missing_question"].get("missing_question_recall", {}).get("min", 0.0) or 0.0,
            item["seed_holdout"].get("action_accuracy", {}).get("min", 0.0) or 0.0,
            -int(item["hidden_dim"]),
        ),
    )
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "local_morning_gpu_contract": assignment["execution_mode"] == "local_morning" and 8 <= now.hour < 18 and assignment["visible_device_count"] == 1,
        "oracle_blind_contract": bool(dataset.get("contract", {}).get("context_excludes_oracle_verdict")) and all(row.get("oracle_label_in_context") is False for row in records),
        "state_cells_present": set(dataset.get("counts", {}).get("state_cells", [])) >= {"unavailable", "available_unresolved", "transport", "observable", "progress", "missing_key"},
        "missing_question_rows_present": bool(all_missing and source_missing and seed_missing),
        "hard_negative_evaluated": bool(hard_negative_rows),
        "hard_negative_zero_false_allow": selected["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg294-active-repair-training-v1",
        "schema_version": "pg294-active-repair-training-report-v1",
        "status": "completed_local_morning_pg294_active_repair",
        "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT).as_posix()), "audit_sha256": audit.get("audit_sha256"), "oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False},
        "device": assignment,
        "split": {"total": len(records), "train": len(train_rows), "source_holdout": len(source_holdout), "seed_holdout": len(seed_holdout), "hard_negative_eval": len(hard_negative_rows), "seeds": list(SEEDS), "hidden_dims": list(HIDDEN_DIMS)},
        "vocabulary_size": len(vocab),
        "variants": variants,
        "selection": {"hidden_dim": selected["hidden_dim"], "hard_negative_false_allow_max": int(selected["hard_negative"].get("hard_negative_false_allow", {}).get("max", 0)), "seed_holdout_positive_recall_min": selected["seed_holdout"].get("positive_recall", {}).get("min"), "source_holdout_positive_recall_min": selected["source_holdout"].get("positive_recall", {}).get("min"), "seed_missing_question_recall_min": selected["seed_missing_question"].get("missing_question_recall", {}).get("min"), "rule": "zero same-context hard-negative false-allow first; then missing-question recall; then seed/source positive recall"},
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "oracle_blind_projection": True}, "reasons": ["dataset is an abstract PG-293 projection", "no fresh typed replay", "no real application gold"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "formal_conclusion": "PG-294 tests active repair under oracle-blind availability/feedback states; it does not establish real payload generation or vulnerability detection.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if selected["hidden_dim"] in snapshots:
        torch.save({"schema_version": "pg294-active-repair-checkpoint-v1", "assignment": assignment, "vocabulary": vocab, "hidden_dim": selected["hidden_dim"], "state": snapshots[selected["hidden_dim"]]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg294-active-repair-training-trace-v1", "report_sha256": report["report_sha256"], "training_eligible": False, "memory_write": False, "oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg294-active-repair-training-protocol-v1", "execution_mode": "local_morning", "window": "08:00-18:00 Asia/Shanghai", "oracle_blind": True, "typed_availability_only": True, "same_context_hard_negative_required": True, "literal_payload_generation": False, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-295: add independent implementation state cells and test whether repair actions remain calibrated without evaluator-name leakage."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "selection": report["selection"], "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
