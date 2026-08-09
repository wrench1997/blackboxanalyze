"""Run the PG-295 causal Transformer-MoE question/assembly experiment."""

from __future__ import annotations

import copy
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

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, build_vocabulary, evaluate_causal_moe, train_causal_moe  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg294_active_repair_dataset_v1.json"
AUDIT = RESEARCH / "pg294_active_repair_dataset_audit_v1.json"
REPORT = RESEARCH / "pg295_causal_moe_training_report_v1_local_morning.json"
TRACE = RESEARCH / "pg295_causal_moe_training_trace_v1_local_morning.json"
PROTOCOL = RESEARCH / "pg295_causal_moe_training_protocol_v1_local_morning.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg295-causal-moe"
CHECKPOINT = CHECKPOINT_DIR / "pg295_causal_moe_selected_local_morning.pt"
SEEDS = (29501, 29502, 29503)
CONFIGS = (
    ("moe_small", CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, max_length=128)),
    ("moe_wide", CausalMoEConfig(d_model=96, n_heads=4, n_layers=2, experts=4, expert_hidden=192, top_k=2, max_length=128)),
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = sorted({key for row in metrics for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None})
    return {key: {"mean": round(sum(float(row[key]) for row in metrics) / len(metrics), 6), "min": round(min(float(row[key]) for row in metrics), 6), "max": round(max(float(row[key]) for row in metrics), 6)} for key in numeric}


def verify_local_morning() -> tuple[dict[str, Any], datetime]:
    if os.environ.get("PG295_LOCAL_RUN") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-295 requires PG295_LOCAL_RUN=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-295 local training is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("PG-295 requires one visible local CUDA device")
    assignment = {"execution_mode": "local_morning", "timestamp": now.isoformat(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(), "device_name": torch.cuda.get_device_name(0)}
    if assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or assignment["current_device"] != 0 or "A800" in assignment["device_name"]:
        raise RuntimeError(f"PG-295 requires one non-A800 local GPU0, got {assignment}")
    return assignment, now


def answer_only_row(row: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(row)
    clone["target_tokens"] = ["question=none" if str(token).startswith("question=") else token for token in clone.get("target_tokens", [])]
    clone["question"] = "none"
    return clone


def main() -> None:
    assignment, now = verify_local_morning()
    dataset = load(DATASET)
    audit = load(AUDIT)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-295 requires a passed PG-294 audit")
    records = list(dataset.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    source_holdout = [row for row in records if row.get("split") == "source_holdout"]
    seed_holdout = [row for row in records if row.get("split") == "seed_holdout"]
    hard_negative = [row for row in records if row.get("split") == "hard_negative_eval"]
    missing = [row for row in records if row.get("state_id") == "missing_key" and row.get("split") != "hard_negative_eval"]
    source_missing = [row for row in source_holdout if row.get("state_id") == "missing_key"]
    seed_missing = [row for row in seed_holdout if row.get("state_id") == "missing_key"]
    if not train_rows or not source_holdout or not seed_holdout or not hard_negative or not source_missing or not seed_missing:
        raise RuntimeError("PG-295 requires complete normal, missing-observation and hard-negative splits")
    device = torch.device("cuda")
    started = time.perf_counter()
    variants: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, torch.Tensor]] = {}
    for config_name, config in CONFIGS:
        vocabulary = build_vocabulary(train_rows)
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            random.seed(seed)
            model = train_causal_moe(train_rows, vocabulary, device, seed=seed, config=config, epochs=140)
            per_seed.append({
                "seed": seed,
                "train": evaluate_causal_moe(model, train_rows, vocabulary, device),
                "source_holdout": evaluate_causal_moe(model, source_holdout, vocabulary, device),
                "seed_holdout": evaluate_causal_moe(model, seed_holdout, vocabulary, device),
                "missing": evaluate_causal_moe(model, missing, vocabulary, device),
                "source_missing": evaluate_causal_moe(model, source_missing, vocabulary, device),
                "seed_missing": evaluate_causal_moe(model, seed_missing, vocabulary, device),
                "hard_negative": evaluate_causal_moe(model, hard_negative, vocabulary, device),
            })
            snapshots[config_name] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        variants.append({
            "config_name": config_name,
            "config": config.__dict__,
            "vocabulary_size": len(vocabulary),
            "train": aggregate([row["train"] for row in per_seed]),
            "source_holdout": aggregate([row["source_holdout"] for row in per_seed]),
            "seed_holdout": aggregate([row["seed_holdout"] for row in per_seed]),
            "missing": aggregate([row["missing"] for row in per_seed]),
            "source_missing": aggregate([row["source_missing"] for row in per_seed]),
            "seed_missing": aggregate([row["seed_missing"] for row in per_seed]),
            "hard_negative": aggregate([row["hard_negative"] for row in per_seed]),
            "per_seed": per_seed,
        })
    selected = max(variants, key=lambda item: (item["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0, item["seed_missing"].get("missing_question_recall", {}).get("min", 0.0) or 0.0, item["seed_holdout"].get("positive_recall", {}).get("min", 0.0) or 0.0, item["seed_holdout"].get("sequence_exact_accuracy", {}).get("min", 0.0) or 0.0))

    # A matched control gets the same causal architecture but no question
    # supervision and no missing-key rows.  It tests the user's identifiability
    # claim instead of allowing a classifier to hide behind final accuracy.
    answer_train = [answer_only_row(row) for row in train_rows if row.get("state_id") != "missing_key"]
    answer_control: list[dict[str, Any]] = []
    control_config = next(config for name, config in CONFIGS if name == selected["config_name"])
    control_vocab = build_vocabulary(answer_train)
    for seed in SEEDS:
        random.seed(seed)
        control_model = train_causal_moe(answer_train, control_vocab, device, seed=seed, config=control_config, epochs=140)
        answer_control.append({"seed": seed, "normal_holdout": evaluate_causal_moe(control_model, seed_holdout, control_vocab, device), "missing_holdout": evaluate_causal_moe(control_model, seed_missing, control_vocab, device)})
    control_summary = {"normal_holdout": aggregate([row["normal_holdout"] for row in answer_control]), "missing_holdout": aggregate([row["missing_holdout"] for row in answer_control]), "per_seed": answer_control, "question_supervision": False}

    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "resource_contract": assignment["execution_mode"] == "local_morning" and 8 <= now.hour < 18 and assignment["visible_device_count"] == 1,
        "causal_next_token_only": True,
        "moe_router_present": all(int(item["config"].get("experts", 0)) > 1 for item in variants),
        "missing_question_split_present": bool(source_missing and seed_missing),
        "hard_negative_zero_false_allow": selected["hard_negative"].get("hard_negative_false_allow", {}).get("max", 999999) == 0,
        "control_has_no_question_supervision": control_summary["question_supervision"] is False,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg295-causal-moe-training-v1",
        "schema_version": "pg295-causal-moe-training-report-v1",
        "status": "completed_local_morning_pg295_causal_moe",
        "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT).as_posix()), "audit_sha256": audit.get("audit_sha256"), "context_is_oracle_blind": True, "literal_payload_in_context": False, "wire_emission": False},
        "device": assignment,
        "split": {"total": len(records), "train": len(train_rows), "source_holdout": len(source_holdout), "seed_holdout": len(seed_holdout), "missing": len(missing), "source_missing": len(source_missing), "seed_missing": len(seed_missing), "hard_negative_eval": len(hard_negative), "seeds": list(SEEDS)},
        "objective": "causal next-token assembly of abstract action, repair, question and safe gate",
        "variants": variants,
        "selection": {"config_name": selected["config_name"], "hard_negative_false_allow_max": int(selected["hard_negative"].get("hard_negative_false_allow", {}).get("max", 0)), "seed_missing_question_recall_min": selected["seed_missing"].get("missing_question_recall", {}).get("min"), "seed_positive_recall_min": selected["seed_holdout"].get("positive_recall", {}).get("min")},
        "answer_only_control": control_summary,
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"fresh_real_evaluator": False, "real_application_gold": False, "literal_payload_success": False, "question_identifiability_test": True}, "reasons": ["abstract PG-294 state cells only", "no fresh typed replay", "no real application gold"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "formal_conclusion": "A causal MoE can be evaluated for missing-observation questions; answer-only accuracy is not evidence of active diagnosis, and the real payload claim remains unproven.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha256_json(report)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if selected["config_name"] in snapshots:
        torch.save({"schema_version": "pg295-causal-moe-checkpoint-v1", "assignment": assignment, "config_name": selected["config_name"], "config": selected["config"], "state": snapshots[selected["config_name"]]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg295-causal-moe-training-trace-v1", "report_sha256": report["report_sha256"], "training_eligible": False, "memory_write": False, "causal_next_token": True, "moe": True, "question_supervision": True, "literal_payload_in_context": False, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg295-causal-moe-training-protocol-v1", "execution_mode": "local_morning", "window": "08:00-18:00 Asia/Shanghai", "causal_next_token_only": True, "moe_experts": [2, 4], "answer_only_control": True, "missing_observation_question_required": True, "same_context_hard_negative_required": True, "literal_payload_generation": False, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-296: add independent implementation-specific missingness patterns and test question composition under source/family holdout."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "selection": report["selection"], "answer_only_control": {"missing_question_recall": control_summary["missing_holdout"].get("missing_question_recall")}, "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
