"""PG-276: train on v1, keep v2 as old canary, hold out v3."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from train_pg275_hypothesis_ablation import (  # noqa: E402
    OUT as _PG275_OUT,
    Policy,
    build_vocab,
    conservative_update,
    dpo_update,
    evaluate,
    fit_weighted,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg276_third_implementation_dataset_v1.json"
REPORT = ROOT / "research" / "pg276_third_implementation_report_v1.json"
TRACE = ROOT / "research" / "pg276_third_implementation_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg276_third_implementation_protocol_v1.json"
ARTIFACT = ROOT / "artifacts" / "pg276-third-implementation" / "pg276_policies.pt"


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    started = time.perf_counter()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    train = [x for x in data["records"] if x["split"] == "implementation_v1_train"]
    canary = [x for x in data["records"] if x["split"] == "implementation_v2_canary"]
    holdout = [x for x in data["records"] if x["split"] == "implementation_v3_holdout"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    vocab = build_vocab(train, "atomic")
    weighted = fit_weighted(train, vocab, "atomic", device, weighted=True, seed=27601)
    conservative = conservative_update(weighted, train, vocab, "atomic", device)
    dpo = dpo_update(weighted, train, vocab, "atomic", device)
    policies = {"weighted_sft_atomic": weighted, "conservative_offline_update": conservative, "dpo_preference_update": dpo}
    evaluations = {name: {"old_canary_v2": evaluate(model, canary, vocab, "atomic", device), "v3_holdout": evaluate(model, holdout, vocab, "atomic", device)} for name, model in policies.items()}
    final = {name: item["v3_holdout"] for name, item in evaluations.items()}
    canary = {name: item["old_canary_v2"] for name, item in evaluations.items()}
    gates = {
        "v3_positive_recall_min": all(float(item["positive_recall"]) >= 0.5 for item in final.values()),
        "v3_negative_reject_min": all(float(item["negative_reject"]) >= 0.8 for item in final.values()),
        "v3_false_positive_zero": all(int(item["false_positive_count"]) == 0 for item in final.values()),
        "old_canary_positive_recall_min": all(float(item["positive_recall"]) >= 0.5 for item in canary.values()),
        "promotion_blocked": True,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg276-third-implementation-policy-v1", "vocabulary": vocab, "policies": {name: model.state_dict() for name, model in policies.items()}, "assignment": assignment}, ARTIFACT)
    report = {"protocol_id": "pg276-third-implementation-v1", "schema_version": "pg276-third-implementation-report-v1", "status": "completed_third_implementation_replay", "source": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": data["dataset_sha256"], "device": str(device), "cuda_assignment": assignment, "external_network": False, "raw_payload_in_context": False, "oracle_in_context": False}, "split": {"train": len(train), "old_canary_v2": len(canary), "v3_holdout": len(holdout), "train_implementation": "heterogeneous_surface_v1", "old_canary_implementation": "heterogeneous_surface_v2", "holdout_implementation": "heterogeneous_surface_v3", "disjoint": True}, "evaluations": evaluations, "gates": {"status": "passed" if all(gates.values()) else "blocked", "checks": gates, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "third implementation is a research holdout; fresh seed and multi-family canary remain required"}, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg276-third-implementation-trace-v1", "evaluation_only": True, "training_eligible": False, "source_dataset_sha256": data["dataset_sha256"], "evaluations": {name: {split: {k: value for k, value in metrics.items() if k != "details"} for split, metrics in item.items()} for name, item in evaluations.items()}, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False, "memory_write": False}
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg276-third-implementation-v1", "schema_version": "pg276-third-implementation-protocol-v1", "train": "v1 only", "old_canary": "v2 only and not used for update", "holdout": "v3 only", "policy_variants": list(policies), "gates": gates, "report_sha256": report["report_sha256"], "next_experiment": "PG-277 multi-seed v3 and failure/repair trajectory canary"}
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": assignment, "gates": report["gates"], "v3": {name: {k: metrics["v3_holdout"][k] for k in ("next_action_accuracy", "belief_accuracy", "positive_recall", "negative_reject", "false_positive_count", "false_negative_count")} for name, metrics in evaluations.items()}, "old_canary": {name: {k: metrics["old_canary_v2"][k] for k in ("positive_recall", "negative_reject", "false_positive_count")} for name, metrics in evaluations.items()}, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
