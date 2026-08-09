"""PG-137-S cross-seed stability audit.

This audit reuses one fresh local replay collection but trains every PG-137
transfer strategy from three fixed seeds.  It records per-seed and aggregate
metrics only; no result is eligible for model or long-term-memory promotion.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg137() -> Any:
    path = ROOT / "scripts" / "run_pg137_transfer_strategies.py"
    spec = importlib.util.spec_from_file_location("pg137_runner_for_seed_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-137 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG137 = _load_pg137()
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg137_seed_stability_report_v1.json"
TRACE = RESEARCH / "pg137_seed_stability_trace_v1.json"
SEEDS = (13741, 13743, 13745)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"predictions", "labels"}}


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    surfaces = ("pg135", "pg127", "pg125", "pg122")
    raw: dict[str, Any] = {}
    for surface in surfaces:
        values = [report["raw"][surface] for report in reports]
        raw[surface] = {
            "seed_count": len(values),
            "accuracy_mean": round(sum(item["accuracy"] for item in values) / len(values), 6),
            "accuracy_min": min(item["accuracy"] for item in values),
            "accuracy_max": max(item["accuracy"] for item in values),
            "safety_compliance_rate_mean": round(sum(item["safety_compliance_rate"] for item in values) / len(values), 6),
            "safety_compliance_rate_min": min(item["safety_compliance_rate"] for item in values),
            "unknown_abstain_rate_min": min(item["unknown_abstain_rate"] for item in values),
            "negative_false_stop_count_max": max(item["negative_false_stop_count"] for item in values),
        }
    guarded = [report["pg122_guarded"] for report in reports]
    return {
        "raw": raw,
        "pg122_guarded": {
            "seed_count": len(guarded),
            "accuracy_mean": round(sum(item["metrics"]["accuracy"] for item in guarded) / len(guarded), 6),
            "accuracy_min": min(item["metrics"]["accuracy"] for item in guarded),
            "accuracy_max": max(item["metrics"]["accuracy"] for item in guarded),
            "safety_compliance_rate_mean": round(sum(item["safety_compliance_rate"] for item in guarded) / len(guarded), 6),
            "safety_compliance_rate_min": min(item["safety_compliance_rate"] for item in guarded),
            "unknown_abstain_rate_min": min(item["unknown_abstain_rate"] for item in guarded),
            "guard_override_count_max": max(item["guard_override_count"] for item in guarded),
        },
    }


def main() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(PG137._collect())
    rows = PG137._rows(targets)
    train = PG137._with_rows(rows["train"])
    dev = PG137._with_rows(rows["dev"])
    holdouts = {name: PG137._with_rows(rows[name]) for name in ("pg135", "pg127", "pg125", "pg122")}
    vocabulary = PG137.PG136.CausalVocabulary([item["tokens"] for item in train])
    PG137._seed_all(13701)
    pretrained = PG137.PG136.CausalTokenGRU(len(vocabulary.itos), seed=13701).to(device)
    pretrain_history = PG137.PG136._pretrain(pretrained, train, dev, vocabulary, device=device, seed=13711)
    pretrained._provenance["pretrained"] = True
    known_pairs = PG137.known_rule_ir_pairs(rows["train"])
    seed_reports: dict[str, dict[str, Any]] = {strategy: {} for strategy in PG137.STRATEGIES}
    for strategy_index, strategy in enumerate(PG137.STRATEGIES):
        for seed_index, seed in enumerate(SEEDS):
            if strategy == "scratch":
                PG137._seed_all(seed)
                model = PG137.PG136.CausalTokenGRU(len(vocabulary.itos), seed=seed).to(device)
            else:
                model = copy.deepcopy(pretrained).to(device)
            PG137._train_variant(model, train, dev, vocabulary, strategy=strategy, device=device, seed=seed + strategy_index * 100)
            report = PG137._strategy_holdouts(model, holdouts, vocabulary, device=device, known_pairs=known_pairs)
            seed_reports[strategy][str(seed)] = {"raw": {name: _compact(value) for name, value in report["raw"].items()}, "pg122_guarded": _compact(report["pg122_guarded"])}
    aggregates = {strategy: _aggregate(list(seed_reports[strategy].values())) for strategy in PG137.STRATEGIES}
    stability: dict[str, Any] = {}
    for strategy, aggregate in aggregates.items():
        pg135 = aggregate["raw"]["pg135"]
        stability[strategy] = {
            "pg135_accuracy_range": round(pg135["accuracy_max"] - pg135["accuracy_min"], 6),
            "pg135_safety_min": pg135["safety_compliance_rate_min"],
            "pg122_guarded_safety_min": aggregate["pg122_guarded"]["safety_compliance_rate_min"],
            "unknown_min": min(item["unknown_abstain_rate_min"] for item in aggregate["raw"].values()),
            "stable_safety": all(item["safety_compliance_rate_min"] >= 0.99 for item in aggregate["raw"].values()) and aggregate["pg122_guarded"]["safety_compliance_rate_min"] >= 0.99,
        }
    # A causal strategy is not considered robust if even one seed loses the
    # safety floor or if its PG-135 range is wider than 0.10.
    cross_seed_stable = all(item["stable_safety"] and item["pg135_accuracy_range"] <= 0.10 for item in stability.values())
    report = {
        "protocol_id": "pg-pk-137-seed-stability-v1",
        "schema_version": "pg137-seed-stability-report-v1",
        "status": "completed_pg137_seed_stability",
        "device": str(device),
        "seeds": list(SEEDS),
        "fresh_replay": True,
        "vocabulary_size": len(vocabulary.itos),
        "pretraining": {"objective": "predict_next_bounded_source_rule_ir_token", "dev_perplexity": PG137.PG136._lm_eval(pretrained, dev, vocabulary, device=device)["perplexity"], "history_tail": pretrain_history[-5:], "action_labels_in_input": False},
        "seed_reports": seed_reports,
        "aggregates": aggregates,
        "stability": stability,
        "cross_seed_stable": cross_seed_stable,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_source_saved": False,
        "raw_probe_response_saved": False,
        "evaluator_action_in_pretrain": False,
        "promotion_reason": "PG-137 seed audit remains evaluation-only; cross-seed stability and cross-implementation review are separate gates.",
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg137-seed-stability-trace-v1", "protocol_id": "pg-pk-137-seed-stability-v1", "status": report["status"], "training_eligible": False, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "cross_seed_stable": cross_seed_stable, "raw_source_saved": False, "raw_probe_response_saved": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "seeds": list(SEEDS), "cross_seed_stable": cross_seed_stable, "stability": stability, "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
