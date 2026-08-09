#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.research_events import emit_event  # noqa: E402
from app.synthetic_curriculum import generate_curriculum  # noqa: E402
from evaluate_url_root_cause import ADAPTED, REGRESSION_FAMILIES, TARGET, load_model, rule_diagnostics  # noqa: E402
from train_rule_memory_pilot import PromptDataset, collate, evaluate, records_to_examples  # noqa: E402


DATA_SEEDS = [20261123, 20261211]
MODEL_SEEDS = [20261001, 20261019, 20261107]


def make_examples(records: list[dict[str, Any]], seed: int, structured: bool):
    return records_to_examples(
        records,
        random.Random(seed),
        4,
        8,
        routed_semantic_features=True,
        episode_rule_features=True,
        structured_url_rule=structured,
    )


def make_loader(examples) -> DataLoader:
    return DataLoader(PromptDataset(examples, 640), batch_size=64, shuffle=False, collate_fn=collate)


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {"accuracy": result["accuracy"], "correct": result["correct"], "total": result["total"], "by_family": result["by_family"]}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {seed: load_model(ADAPTED[seed], device) for seed in MODEL_SEEDS}
    cross_product = []
    data_lineage = []
    for data_seed in DATA_SEEDS:
        records = generate_curriculum(2700, 20, data_seed)
        target_records = [record for record in records if record["family"] == TARGET]
        regression_records = [record for record in records if record["family"] in REGRESSION_FAMILIES]
        target_legacy = make_examples(target_records, data_seed, structured=False)
        target_c5 = make_examples(target_records, data_seed, structured=True)
        regression_legacy = make_examples(regression_records, data_seed, structured=False)
        regression_c5 = make_examples(regression_records, data_seed, structured=True)
        manifest_path = ROOT / f"artifacts/url-root-cause-loop-10b-seed-{data_seed}/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_lineage.append({
            "seed": data_seed,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "dataset_sha256": manifest["dataset_sha256"],
            "verified_traces": manifest["verified_traces"],
            "verified_counterexamples": manifest["verified_counterexamples"],
            "target_rule_diagnostics": rule_diagnostics(target_c5),
        })
        for model_seed, model in models.items():
            legacy_target_result = evaluate(model, make_loader(target_legacy), device, max_length=640, rule_memory_gate=True)
            c5_target_result = evaluate(
                model,
                make_loader(target_c5),
                device,
                max_length=640,
                rule_memory_gate=True,
                confidence_arbitration=True,
                abstain_on_rule_conflict=True,
            )
            neural_target_result = evaluate(model, make_loader(target_legacy), device, max_length=640, rule_memory_gate=False)
            legacy_regression_result = evaluate(model, make_loader(regression_legacy), device, max_length=640, rule_memory_gate=True)
            c5_regression_result = evaluate(
                model,
                make_loader(regression_c5),
                device,
                max_length=640,
                rule_memory_gate=True,
                confidence_arbitration=True,
                abstain_on_rule_conflict=True,
            )
            regression = {
                family: {
                    "legacy": legacy_regression_result["by_family"][family],
                    "c5": c5_regression_result["by_family"][family],
                    "delta": round(c5_regression_result["by_family"][family] - legacy_regression_result["by_family"][family], 6),
                }
                for family in sorted(REGRESSION_FAMILIES)
            }
            cross_product.append({
                "data_seed": data_seed,
                "model_seed": model_seed,
                "legacy_target": compact(legacy_target_result),
                "c5_target": compact(c5_target_result),
                "neural_target": compact(neural_target_result),
                "target_delta": round(c5_target_result["accuracy"] - legacy_target_result["accuracy"], 6),
                "regression": regression,
                "worst_regression_delta": min(item["delta"] for item in regression.values()),
            })

    target_scores = [row["c5_target"]["accuracy"] for row in cross_product]
    legacy_scores = [row["legacy_target"]["accuracy"] for row in cross_product]
    neural_scores = [row["neural_target"]["accuracy"] for row in cross_product]
    regression_deltas = [row["worst_regression_delta"] for row in cross_product]
    accepted = min(target_scores) >= 0.90 and min(regression_deltas) >= -0.02
    report = {
        "schema_version": "sift-url-root-cause-confirmation-v1",
        "experiment": "url-root-cause-loop-10b-fresh-confirmation",
        "status": "confirmed" if accepted else "failed_confirmation",
        "scope": "fresh-data, frozen-checkpoint, zero-parameter confirmation",
        "device": str(device),
        "protocol": {
            "fresh_data_seeds": DATA_SEEDS,
            "model_seeds": MODEL_SEEDS,
            "cross_product_evaluations": len(cross_product),
            "target_family_examples_seen_in_training": 0,
            "algorithm_changes_after_registration": 0,
        },
        "aggregate": {
            "legacy_target_mean": round(sum(legacy_scores) / len(legacy_scores), 6),
            "c5_target_mean": round(sum(target_scores) / len(target_scores), 6),
            "c5_target_min": min(target_scores),
            "neural_target_mean": round(sum(neural_scores) / len(neural_scores), 6),
            "mean_c5_gain": round(sum(b - a for a, b in zip(legacy_scores, target_scores)) / len(target_scores), 6),
            "worst_regression_delta": min(regression_deltas),
        },
        "acceptance": {
            "each_target_at_least_90pct": min(target_scores) >= 0.90,
            "each_regression_family_within_minus_2pp": min(regression_deltas) >= -0.02,
            "accepted": accepted,
            "claim": "zero-parameter architecture/rule-abstraction repair; not neural generalization" if accepted else "confirmation failed",
        },
        "data_lineage": data_lineage,
        "cross_product": cross_product,
        "lineage": {
            "preregistration": "research/url_root_cause_loop_10b_confirmation_preregistration.json",
            "discovery_results": "research/url_root_cause_loop_10_results.json",
        },
    }
    json_path = ROOT / "research/url_root_cause_loop_10b_confirmation.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "\n".join(
        f"| {row['data_seed']} | {row['model_seed']} | {row['legacy_target']['accuracy']:.2%} | {row['c5_target']['accuracy']:.2%} | {row['neural_target']['accuracy']:.2%} | {row['worst_regression_delta']:+.2%} |"
        for row in cross_product
    )
    markdown = f"""# URL Root-Cause Loop 10b

Status: **{'confirmed' if accepted else 'failed confirmation'}**.

C5 was frozen before generating two fresh datasets. Across all six data-seed × model-seed evaluations, URL agreement moved from a {report['aggregate']['legacy_target_mean']:.2%} legacy mean to {report['aggregate']['c5_target_mean']:.2%}; the minimum C5 score was {report['aggregate']['c5_target_min']:.2%}. The neural path remained {report['aggregate']['neural_target_mean']:.2%}, so this is an architecture/rule-abstraction repair, not neural generalization.

| Data seed | Model seed | Legacy | C5 | Neural | Worst regression |
|---:|---:|---:|---:|---:|---:|
{rows}

Worst regression across all six evaluations: {report['aggregate']['worst_regression_delta']:+.2%}. The C5 rule head uses only black-box observations, requires both label classes, chooses maximum empirical fit, and abstains when equally fitting rules disagree.
"""
    (ROOT / "research/url_root_cause_loop_10b_confirmation.md").write_text(markdown, encoding="utf-8")
    emit_event(
        actor="url-root-cause-confirmer",
        tool="generalization.url_root_cause.confirm",
        phase="fresh_seed_confirmation",
        status="complete" if accepted else "failed",
        message=f"C5 fresh confirmation {'passed' if accepted else 'failed'}: min URL {min(target_scores):.2%}, worst regression {min(regression_deltas):+.2%}",
        payload={"aggregate": report["aggregate"], "acceptance": report["acceptance"]},
        artifact=str(json_path.relative_to(ROOT)),
    )
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"], "acceptance": report["acceptance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
