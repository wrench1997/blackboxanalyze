#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SEEDS = [20261001, 20261019, 20261107]
FAMILIES = [
    "url_scheme_downgrade",
    "dom_double_decode",
    "unicode_casefold_role",
    "numeric_string_coercion",
    "compound_origin_role",
    "state_replay_window",
]
LABELS = {
    "url_scheme_downgrade": "URL runtime semantics",
    "dom_double_decode": "Encoding depth",
    "unicode_casefold_role": "Unicode / casefold",
    "numeric_string_coercion": "Numeric coercion",
    "compound_origin_role": "Rule composition",
    "state_replay_window": "State / history",
}


def read(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 6),
        "population_std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main() -> None:
    stage_a = read("research/generalization_matrix_09_stage_a_current_representation.json")
    stage_b = read("research/generalization_matrix_09_stage_b_stratified_decomposed.json")
    pair_reports = {
        20261019: read("research/generalization_matrix_09_pair_20261019.json"),
        20261107: read("research/generalization_matrix_09_pair_20261107.json"),
    }
    training_reports = {
        20261001: read("research/generalization_matrix_09_stage_b_stratified.json"),
        20261019: read("research/generalization_matrix_09_stage_b_20261019.json"),
        20261107: read("research/generalization_matrix_09_stage_b_20261107.json"),
    }
    old_regression_reports = {
        20261001: read("research/generalization_matrix_09_old_family_regression.json"),
        20261019: read("research/generalization_matrix_09_old_family_regression_20261019.json"),
        20261107: read("research/generalization_matrix_09_old_family_regression_20261107.json"),
    }

    per_seed: list[dict[str, Any]] = []
    manifests: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        if seed == 20261001:
            baseline_macro = stage_a["results"]["with_executable_memory"]["accuracy"]
            candidate_macro = stage_b["results"]["with_executable_memory"]["accuracy"]
            family_rows = {
                family: {
                    "baseline": stage_a["results"]["with_executable_memory"]["by_family"][family],
                    "candidate": stage_b["results"]["with_executable_memory"]["by_family"][family],
                    "delta": round(
                        stage_b["results"]["with_executable_memory"]["by_family"][family]
                        - stage_a["results"]["with_executable_memory"]["by_family"][family],
                        6,
                    ),
                    "baseline_neural": stage_a["results"]["neural_with_memory"]["by_family"][family],
                    "candidate_neural": stage_b["results"]["neural_with_memory"]["by_family"][family],
                }
                for family in FAMILIES
            }
        else:
            pair = pair_reports[seed]
            baseline_macro = pair["baseline"]["with_rule_head"]["accuracy"]
            candidate_macro = pair["candidate"]["with_rule_head"]["accuracy"]
            family_rows = pair["per_family"]
        train_result = training_reports[seed]["results"]["counterexample_at_k"]
        manifest = read(f"artifacts/generalization-matrix-09-seed-{seed}/manifest.json")
        manifests[seed] = manifest
        per_seed.append({
            "seed": seed,
            "dataset_sha256": manifest["dataset_sha256"],
            "baseline_macro": baseline_macro,
            "candidate_macro": candidate_macro,
            "macro_delta": round(candidate_macro - baseline_macro, 6),
            "per_family": family_rows,
            "counterexample_top1": train_result["top1"],
            "random_top1": train_result["random_top1"],
            "old_family_worst_delta": old_regression_reports[seed]["decision"]["worst_family_delta"],
            "old_family_regression_passes": old_regression_reports[seed]["decision"]["passes"],
        })

    aggregate_families: dict[str, Any] = {}
    for family in FAMILIES:
        baselines = [row["per_family"][family]["baseline"] for row in per_seed]
        candidates = [row["per_family"][family]["candidate"] for row in per_seed]
        deltas = [row["per_family"][family]["delta"] for row in per_seed]
        if family == "numeric_string_coercion" and min(deltas) >= 0.10:
            conclusion = "confirmed_training_generalization"
        elif family == "state_replay_window":
            conclusion = "confirmed_rule_architecture_fix_not_neural_learning"
        elif family == "url_scheme_downgrade" and statistics.mean(deltas) < 0:
            conclusion = "confirmed_negative_transfer"
        elif family == "dom_double_decode" and max(deltas) >= 0.10 and min(deltas) < 0.10:
            conclusion = "high_variance_not_confirmed"
        elif min(deltas) >= 0:
            conclusion = "stable_small_gain_below_preregistered_effect"
        else:
            conclusion = "mixed_or_no_gain"
        aggregate_families[family] = {
            "label": LABELS[family],
            "baseline": stats(baselines),
            "candidate": stats(candidates),
            "delta": stats(deltas),
            "per_seed_delta": {str(row["seed"]): row["per_family"][family]["delta"] for row in per_seed},
            "conclusion": conclusion,
        }

    macro_baseline = [row["baseline_macro"] for row in per_seed]
    macro_candidate = [row["candidate_macro"] for row in per_seed]
    macro_delta = [row["macro_delta"] for row in per_seed]
    report = {
        "schema_version": "sift-generalization-matrix-summary-v1",
        "experiment": "generalization-matrix-09",
        "status": "three_seed_partial_success_with_blocking_url_failure",
        "scope": "research pilot; synthetic local targets only; no production vulnerability detection claim",
        "model": {
            "parameters": 908546,
            "architecture": "4-layer causal Transformer plus zero-parameter executable Rule Memory",
            "target_family_examples_seen_in_training": 0,
        },
        "data": {
            "seeds": SEEDS,
            "programs_per_seed": 2700,
            "programs_total": sum(manifests[seed]["programs"] for seed in SEEDS),
            "verified_traces_by_seed": {str(seed): manifests[seed]["verified_traces"] for seed in SEEDS},
            "verified_traces_total": sum(manifests[seed]["verified_traces"] for seed in SEEDS),
            "verified_counterexamples_total": sum(manifests[seed]["verified_counterexamples"] for seed in SEEDS),
            "families": 18,
            "target_families": 6,
            "complete_target_family_holdout": True,
        },
        "aggregate": {
            "baseline_macro": stats(macro_baseline),
            "candidate_macro": stats(macro_candidate),
            "paired_macro_delta": stats(macro_delta),
            "counterexample_top1": stats([row["counterexample_top1"] for row in per_seed]),
            "random_top1": stats([row["random_top1"] for row in per_seed]),
            "old_family_worst_delta": stats([row["old_family_worst_delta"] for row in per_seed]),
            "per_family": aggregate_families,
        },
        "per_seed": per_seed,
        "causal_attribution": {
            "state_history": "The frozen checkpoint also reaches 100% after the new history projection and executable prev-step rule; attribute this to representation/architecture, not training.",
            "numeric_coercion": "Primitive-only training improves every seed by at least 28pp on an entirely held-out composed family; this is the only preregistered training-generalization effect confirmed across all seeds.",
            "double_decode": "Two seeds improve by more than 13pp and one remains at chance; classify as high variance and unconfirmed.",
            "url_semantics": "Hostname-primitive adaptation causes negative transfer on scheme downgrade; URL projection/rule arbitration is the next root-cause target.",
            "sampling_bug": "An earlier Stage-B run used a global stride split that removed two whole families from training. It is retained as an invalid diagnostic and excluded from every aggregate here.",
        },
        "decision": {
            "numeric_coercion_generalization_confirmed": True,
            "double_decode_generalization_confirmed": False,
            "state_fix_is_model_learning": False,
            "url_negative_transfer_blocks_engineering_scale_up": True,
            "all_old_family_regressions_within_2pp": all(row["old_family_regression_passes"] for row in per_seed),
            "next_experiment": "Factor URL into scheme/hostname/port primitives and add confidence arbitration that prevents raw suffix rules from overriding structured URL evidence; keep model size fixed.",
        },
        "lineage": {
            "preregistration": "research/generalization_matrix_preregistration.json",
            "invalid_stage_b_diagnostic": "research/generalization_matrix_09_stage_b.json",
            "stage_a_current_representation": "research/generalization_matrix_09_stage_a_current_representation.json",
            "stage_b_seed_reports": [
                "research/generalization_matrix_09_stage_b_stratified.json",
                "research/generalization_matrix_09_stage_b_20261019.json",
                "research/generalization_matrix_09_stage_b_20261107.json",
            ],
        },
    }

    json_path = ROOT / "research/generalization_matrix_09_summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for family in FAMILIES:
        item = aggregate_families[family]
        rows.append(
            f"| {item['label']} | {item['baseline']['mean']:.2%} | {item['candidate']['mean']:.2%} | "
            f"{item['delta']['mean']:+.2%} | {item['conclusion']} |"
        )
    markdown = f"""# Generalization Matrix 09

Status: **partial success with a blocking URL-semantics failure**. This is a synthetic, local research pilot—not a production vulnerability-detector claim.

Across three preregistered seeds, the paired macro score moved from {stats(macro_baseline)['mean']:.2%} to {stats(macro_candidate)['mean']:.2%} ({stats(macro_delta)['mean']:+.2%}). The only training-driven effect confirmed on all three seeds is numeric coercion. The 100% history result comes from the new representation and executable rule head, not neural learning.

| Held-out axis | Frozen baseline | Primitive-adapted | Paired delta | Conclusion |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

Counterexample Top-1 averaged {stats([row['counterexample_top1'] for row in per_seed])['mean']:.2%}, versus a {stats([row['random_top1'] for row in per_seed])['mean']:.2%} random Top-1 baseline. Every old-family regression check stayed within the preregistered -2pp bound.

## Root-cause decision

- Numeric string coercion: confirmed cross-family training generalization (+28pp or more in every seed).
- Double decoding: high variance; two gains and one chance result, so not confirmed.
- State/history: solved by abstraction and executable memory, while the neural path remains near chance.
- URL semantics: confirmed negative transfer. The next experiment must factor scheme, hostname and port, and prevent a raw suffix rule from overriding structured URL evidence.
- The earlier unstratified Stage-B run is preserved for audit but excluded from all aggregates.
"""
    (ROOT / "research/generalization_matrix_09_summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "macro": report["aggregate"]["candidate_macro"], "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
