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
from train_neural_url_set_head import MAX_LENGTH, REGRESSION_FAMILIES, SetPromptDataset, TinyRuleSetGPT, evaluate_set, set_collate  # noqa: E402
from train_rule_memory_pilot import (  # noqa: E402
    PromptDataset,
    TinyRuleGPT,
    collate,
    evaluate,
    records_to_counterexample_examples,
    records_to_examples,
)


DATA_SEEDS = [20261603, 20261621]
CANDIDATE_PATH = ROOT / "artifacts/neural-url-loop-11-url-meta-v2-20261529/tiny_rule_set_gpt.pt"
CANONICAL_BASELINE_PATH = ROOT / "artifacts/neural-url-loop-11-pilot-20261231/tiny_rule_gpt.pt"
FROZEN_PATH = ROOT / "artifacts/frontend-loop-05-routed-memory-20260801/tiny_rule_gpt.pt"
C5_MODEL_PATH = ROOT / "artifacts/generalization-matrix-09-stage-b-stratified-20261001/tiny_rule_gpt.pt"


def set_loader(examples) -> DataLoader:
    return DataLoader(SetPromptDataset(examples, MAX_LENGTH), batch_size=64, shuffle=False, collate_fn=set_collate)


def base_loader(examples) -> DataLoader:
    return DataLoader(PromptDataset(examples, 640), batch_size=64, shuffle=False, collate_fn=collate)


def load_base(path: Path, device: torch.device) -> TinyRuleGPT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyRuleGPT(640, 128, 4, 4).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


@torch.inference_mode()
def counterexample_top1(model: TinyRuleSetGPT, records: list[dict[str, Any]], seed: int, device: torch.device) -> dict[str, float]:
    examples = records_to_counterexample_examples(
        records,
        random.Random(seed + 971),
        8,
        False,
        routed_semantic_features=True,
        canonical_url_slots=True,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for batch in set_loader(examples):
        probabilities = torch.softmax(model(
            batch["tokens"].to(device),
            batch["lengths"].to(device),
            batch["url_set_features"].to(device),
        ), dim=-1).cpu()
        for example, probability in zip(batch["examples"], probabilities):
            intended = int(example.intended_label or 0)
            grouped.setdefault(example.record_id, []).append({
                "score": float(probability[1 - intended]),
                "counterexample": example.label != intended,
            })
    successes = 0
    random_sum = 0.0
    for rows in grouped.values():
        best = max(rows, key=lambda row: row["score"])
        successes += int(best["counterexample"])
        random_sum += sum(int(row["counterexample"]) for row in rows) / len(rows)
    total = len(grouped)
    return {"top1": round(successes / total, 6), "random_top1": round(random_sum / total, 6), "lift": round((successes - random_sum) / total, 6)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = TinyRuleSetGPT().to(device)
    candidate.load_state_dict(torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)["model_state"])
    canonical_baseline = load_base(CANONICAL_BASELINE_PATH, device)
    frozen = load_base(FROZEN_PATH, device)
    c5_model = load_base(C5_MODEL_PATH, device)
    if sum(parameter.numel() for parameter in candidate.parameters()) != 908546:
        raise RuntimeError("candidate parameter budget changed")

    per_seed = []
    for seed in DATA_SEEDS:
        records = generate_curriculum(2700, 20, seed)
        target_records = [record for record in records if record["family"] == "url_scheme_downgrade"]
        regression_records = [record for record in records if record["family"] in REGRESSION_FAMILIES]
        canonical_target = records_to_examples(target_records, random.Random(seed), 4, 8, routed_semantic_features=True, canonical_url_slots=True)
        legacy_target = records_to_examples(target_records, random.Random(seed), 4, 8, routed_semantic_features=True, episode_rule_features=True)
        c5_target = records_to_examples(target_records, random.Random(seed), 4, 8, routed_semantic_features=True, episode_rule_features=True, structured_url_rule=True)
        canonical_regression = records_to_examples(regression_records, random.Random(seed + 1), 4, 8, routed_semantic_features=True, canonical_url_slots=True)

        candidate_result = evaluate_set(candidate, set_loader(canonical_target), device)
        no_head_result = evaluate_set(candidate, set_loader(canonical_target), device, disable_set_head=True)
        canonical_baseline_result = evaluate(canonical_baseline, base_loader(canonical_target), device, max_length=640)
        frozen_result = evaluate(frozen, base_loader(legacy_target), device, max_length=640, rule_memory_gate=False)
        c5_result = evaluate(c5_model, base_loader(c5_target), device, max_length=640, rule_memory_gate=True, confidence_arbitration=True, abstain_on_rule_conflict=True)
        candidate_regression = evaluate_set(candidate, set_loader(canonical_regression), device)
        baseline_regression = evaluate(canonical_baseline, base_loader(canonical_regression), device, max_length=640)
        regression = {
            family: {
                "baseline": baseline_regression["by_family"][family],
                "candidate": candidate_regression["by_family"][family],
                "delta": round(candidate_regression["by_family"][family] - baseline_regression["by_family"][family], 6),
            }
            for family in sorted(REGRESSION_FAMILIES)
        }
        manifest_path = ROOT / f"artifacts/neural-url-loop-11-final-seed-{seed}/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        per_seed.append({
            "seed": seed,
            "dataset_sha256": manifest["dataset_sha256"],
            "verified_traces": manifest["verified_traces"],
            "candidate_neural": candidate_result,
            "same_checkpoint_without_set_head": no_head_result,
            "canonical_meta_baseline": canonical_baseline_result,
            "frozen_native_neural": frozen_result,
            "C5_rule_system": c5_result,
            "candidate_minus_frozen": round(candidate_result["accuracy"] - frozen_result["accuracy"], 6),
            "set_head_gain": round(candidate_result["accuracy"] - no_head_result["accuracy"], 6),
            "counterexample": counterexample_top1(candidate, target_records, seed, device),
            "regression": regression,
            "worst_regression_delta": min(row["delta"] for row in regression.values()),
        })

    candidate_scores = [row["candidate_neural"]["accuracy"] for row in per_seed]
    frozen_scores = [row["frozen_native_neural"]["accuracy"] for row in per_seed]
    head_gains = [row["set_head_gain"] for row in per_seed]
    frozen_gains = [row["candidate_minus_frozen"] for row in per_seed]
    regressions = [row["worst_regression_delta"] for row in per_seed]
    accepted = min(candidate_scores) >= 0.70 and min(head_gains) >= 0.10 and min(frozen_gains) >= 0.10 and min(regressions) >= -0.02
    report: dict[str, Any] = {
        "schema_version": "sift-neural-url-confirmation-v1",
        "experiment": "neural-url-loop-11-final-fresh-confirmation",
        "status": "confirmed" if accepted else "failed_confirmation",
        "scope": "synthetic complete-family holdout; no production detector claim",
        "model": {"parameters": 908546, "learned_url_set_head_parameters": 128, "target_family_examples_in_training": 0},
        "aggregate": {
            "candidate_neural_mean": round(sum(candidate_scores) / len(candidate_scores), 6),
            "candidate_neural_min": min(candidate_scores),
            "frozen_native_neural_mean": round(sum(frozen_scores) / len(frozen_scores), 6),
            "candidate_minus_frozen_min": min(frozen_gains),
            "set_head_gain_min": min(head_gains),
            "worst_regression_delta": min(regressions),
            "counterexample_top1_mean": round(sum(row["counterexample"]["top1"] for row in per_seed) / len(per_seed), 6),
            "random_top1_mean": round(sum(row["counterexample"]["random_top1"] for row in per_seed) / len(per_seed), 6),
        },
        "acceptance": {
            "candidate_each_at_least_70pct": min(candidate_scores) >= 0.70,
            "candidate_minus_frozen_each_at_least_10pp": min(frozen_gains) >= 0.10,
            "set_head_ablation_each_at_least_10pp": min(head_gains) >= 0.10,
            "old_regression_each_within_minus_2pp": min(regressions) >= -0.02,
            "confirmed": accepted,
        },
        "causal_attribution": {
            "canonical_slot": "removes unstable source-field shortcuts and makes URL component positions invariant",
            "meta_label_training": "penalizes fixed family-to-label shortcuts and rewards episode binding",
            "learned_set_head": "provides the missing query-to-labeled-context comparison inductive bias; measured by same-checkpoint ablation",
            "C5": "reported independently as a zero-parameter executable rule system and excluded from neural scores",
        },
        "per_seed": per_seed,
        "lineage": {"preregistration": "research/neural_url_loop_11_final_confirmation_preregistration.json", "pilot": "research/neural_url_loop_11_url_meta_v2_pilot.json", "previous_failed_confirmation": "research/neural_url_loop_11_confirmation_failed_v1.json"},
    }
    json_path = ROOT / "research/neural_url_loop_11_final_confirmation.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "\n".join(
        f"| {row['seed']} | {row['candidate_neural']['accuracy']:.2%} | {row['same_checkpoint_without_set_head']['accuracy']:.2%} | {row['frozen_native_neural']['accuracy']:.2%} | {row['candidate_minus_frozen']:+.2%} | {row['set_head_gain']:+.2%} | {row['worst_regression_delta']:+.2%} |"
        for row in per_seed
    )
    markdown = f"""# Neural URL Generalization Loop 11

Status: **{'confirmed' if accepted else 'failed confirmation'}**. The target family remained completely absent from training.

| Fresh seed | Candidate neural | Same checkpoint, no set head | Frozen neural | Gain vs frozen | Set-head gain | Worst old regression |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

Candidate neural mean: {report['aggregate']['candidate_neural_mean']:.2%}; frozen native neural mean: {report['aggregate']['frozen_native_neural_mean']:.2%}. Counterexample Top-1 averaged {report['aggregate']['counterexample_top1_mean']:.2%}, versus {report['aggregate']['random_top1_mean']:.2%} random.

The result is attributed to a fixed-budget learned set-comparison architecture plus canonical URL slots and episode-consistent meta-label training. C5 remained a separate executable rule path and is not included in the neural result.
"""
    (ROOT / "research/neural_url_loop_11_final_confirmation.md").write_text(markdown, encoding="utf-8")
    emit_event(actor="neural-url-confirmer", tool="generalization.neural_url.confirm", phase="fresh_family_holdout", status="complete" if accepted else "failed", message=f"Neural URL confirmation {'passed' if accepted else 'failed'}: min {min(candidate_scores):.2%}, min gain {min(frozen_gains):+.2%}", payload={"aggregate": report["aggregate"], "acceptance": report["acceptance"]}, artifact=str(json_path.relative_to(ROOT)))
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"], "acceptance": report["acceptance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
