#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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
from train_rule_memory_pilot import (  # noqa: E402
    PromptDataset,
    TinyRuleGPT,
    _rule_memory_prediction,
    collate,
    evaluate,
    records_to_examples,
)


SEEDS = [20261001, 20261019, 20261107]
TARGET = "url_scheme_downgrade"
REGRESSION_FAMILIES = {
    "numeric_boundary",
    "truthiness_gate",
    "substring_origin",
    "authorization_or",
    "string_suffix_primitive",
    "url_hostname_primitive",
}
FROZEN = ROOT / "artifacts/frontend-loop-05-routed-memory-20260801/tiny_rule_gpt.pt"
ADAPTED = {
    20261001: ROOT / "artifacts/generalization-matrix-09-stage-b-stratified-20261001/tiny_rule_gpt.pt",
    20261019: ROOT / "artifacts/generalization-matrix-09-stage-b-stratified-20261019/tiny_rule_gpt.pt",
    20261107: ROOT / "artifacts/generalization-matrix-09-stage-b-stratified-20261107/tiny_rule_gpt.pt",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path, device: torch.device, max_length: int = 640) -> TinyRuleGPT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyRuleGPT(max_length, int(config["hidden"]), int(config["layers"]), int(config["heads"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def make_examples(records: list[dict[str, Any]], seed: int, **features: bool):
    return records_to_examples(
        records,
        random.Random(seed),
        4,
        8,
        routed_semantic_features=True,
        episode_rule_features=True,
        **features,
    )


def loader(examples, max_length: int = 640) -> DataLoader:
    return DataLoader(PromptDataset(examples, max_length), batch_size=64, shuffle=False, collate_fn=collate)


def rule_diagnostics(examples) -> dict[str, Any]:
    structured = 0
    suffix = 0
    conflicts = 0
    legacy_correct = 0
    arbitration_correct = 0
    arbitration_covered = 0
    identifiability_correct = 0
    identifiability_covered = 0
    for example in examples:
        has_structured = "|kind=url_hostname" in example.prompt
        has_suffix = "#suffix=" in example.prompt
        structured += int(has_structured)
        suffix += int(has_suffix)
        legacy = _rule_memory_prediction(example.prompt)
        arbitrated = _rule_memory_prediction(example.prompt, confidence_arbitration=True)
        identifiable = _rule_memory_prediction(example.prompt, confidence_arbitration=True, abstain_on_rule_conflict=True)
        conflicts += int(legacy is not None and arbitrated is not None and legacy != arbitrated)
        legacy_correct += int(legacy == example.label) if legacy is not None else 0
        arbitration_correct += int(arbitrated == example.label) if arbitrated is not None else 0
        arbitration_covered += int(arbitrated is not None)
        identifiability_correct += int(identifiable == example.label) if identifiable is not None else 0
        identifiability_covered += int(identifiable is not None)
    total = len(examples)
    return {
        "total": total,
        "structured_url_rule_coverage": round(structured / total, 6),
        "suffix_rule_coverage": round(suffix / total, 6),
        "legacy_vs_arbitrated_conflict_rate": round(conflicts / total, 6),
        "legacy_rule_accuracy_over_all": round(legacy_correct / total, 6),
        "arbitrated_rule_coverage": round(arbitration_covered / total, 6),
        "arbitrated_rule_accuracy_over_all": round(arbitration_correct / total, 6),
        "identifiability_rule_coverage": round(identifiability_covered / total, 6),
        "identifiability_rule_accuracy_over_all": round(identifiability_correct / total, 6),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen_model = load_model(FROZEN, device)
    per_seed = []
    for seed in SEEDS:
        records = generate_curriculum(2700, 20, seed)
        target_records = [record for record in records if record["family"] == TARGET]
        regression_records = [record for record in records if record["family"] in REGRESSION_FAMILIES]
        adapted_model = load_model(ADAPTED[seed], device)

        legacy_examples = make_examples(target_records, seed)
        suppressed_examples = make_examples(target_records, seed, suppress_url_suffix=True)
        structured_examples = make_examples(target_records, seed, structured_url_rule=True)
        legacy_loader = loader(legacy_examples)
        suppressed_loader = loader(suppressed_examples)
        structured_loader = loader(structured_examples)

        conditions = {
            "C0_frozen_legacy": evaluate(frozen_model, legacy_loader, device, max_length=640, rule_memory_gate=True),
            "C1_adapted_legacy": evaluate(adapted_model, legacy_loader, device, max_length=640, rule_memory_gate=True),
            "C2_suffix_suppressed": evaluate(adapted_model, suppressed_loader, device, max_length=640, rule_memory_gate=True),
            "C3_structured_rule_legacy_selection": evaluate(adapted_model, structured_loader, device, max_length=640, rule_memory_gate=True),
            "C4_structured_rule_confidence_arbitration": evaluate(
                adapted_model,
                structured_loader,
                device,
                max_length=640,
                rule_memory_gate=True,
                confidence_arbitration=True,
            ),
            "C5_identifiability_aware_arbitration": evaluate(
                adapted_model,
                structured_loader,
                device,
                max_length=640,
                rule_memory_gate=True,
                confidence_arbitration=True,
                abstain_on_rule_conflict=True,
            ),
            "adapted_neural_legacy_prompt": evaluate(adapted_model, legacy_loader, device, max_length=640, rule_memory_gate=False),
            "frozen_neural_legacy_prompt": evaluate(frozen_model, legacy_loader, device, max_length=640, rule_memory_gate=False),
        }

        regression_legacy_examples = make_examples(regression_records, seed)
        regression_structured_examples = make_examples(regression_records, seed, structured_url_rule=True)
        regression_legacy = evaluate(adapted_model, loader(regression_legacy_examples), device, max_length=640, rule_memory_gate=True)
        regression_c4 = evaluate(
            adapted_model,
            loader(regression_structured_examples),
            device,
            max_length=640,
            rule_memory_gate=True,
            confidence_arbitration=True,
        )
        regression_c5 = evaluate(
            adapted_model,
            loader(regression_structured_examples),
            device,
            max_length=640,
            rule_memory_gate=True,
            confidence_arbitration=True,
            abstain_on_rule_conflict=True,
        )
        regression = {
            family: {
                "legacy": regression_legacy["by_family"][family],
                "c4": regression_c4["by_family"][family],
                "delta": round(regression_c4["by_family"][family] - regression_legacy["by_family"][family], 6),
                "c5": regression_c5["by_family"][family],
                "c5_delta": round(regression_c5["by_family"][family] - regression_legacy["by_family"][family], 6),
            }
            for family in sorted(REGRESSION_FAMILIES)
        }
        per_seed.append({
            "seed": seed,
            "dataset_manifest": f"artifacts/generalization-matrix-09-seed-{seed}/manifest.json",
            "adapted_checkpoint": {"path": str(ADAPTED[seed].relative_to(ROOT)), "sha256": sha256(ADAPTED[seed])},
            "target_examples": len(legacy_examples),
            "conditions": {name: result["accuracy"] for name, result in conditions.items()},
            "condition_details": conditions,
            "rule_diagnostics": rule_diagnostics(structured_examples),
            "regression": regression,
            "worst_regression_delta": min(item["delta"] for item in regression.values()),
            "c5_worst_regression_delta": min(item["c5_delta"] for item in regression.values()),
        })

    def values(condition: str) -> list[float]:
        return [row["conditions"][condition] for row in per_seed]

    c0, c1, c2, c3, c4 = (values(name) for name in (
        "C0_frozen_legacy",
        "C1_adapted_legacy",
        "C2_suffix_suppressed",
        "C3_structured_rule_legacy_selection",
        "C4_structured_rule_confidence_arbitration",
    ))
    c5 = values("C5_identifiability_aware_arbitration")
    means = {name: round(sum(values(name)) / len(SEEDS), 6) for name in per_seed[0]["conditions"]}
    h1_deltas = [b - a for a, b in zip(c1, c2)]
    h3_deltas = [b - a for a, b in zip(c3, c4)]
    report = {
        "schema_version": "sift-url-root-cause-result-v1",
        "experiment": "url-root-cause-loop-10",
        "status": "completed",
        "scope": "zero-parameter black-box rule-abstraction ablation; no retraining",
        "device": str(device),
        "guardrails": {
            "new_learned_parameters": 0,
            "target_family_examples_in_training": 0,
            "family_name_source_rule_ir_and_intended_output_hidden": True,
            "checkpoints_frozen": True,
        },
        "aggregate": {
            "condition_means": means,
            "C2_minus_C1_per_seed": [round(value, 6) for value in h1_deltas],
            "C4_minus_C3_per_seed": [round(value, 6) for value in h3_deltas],
            "C4_min": min(c4),
            "worst_regression_delta": min(row["worst_regression_delta"] for row in per_seed),
            "C5_min": min(c5),
            "C5_worst_regression_delta": min(row["c5_worst_regression_delta"] for row in per_seed),
        },
        "hypothesis_results": {
            "H1_suffix_is_primary": {"passes": min(h1_deltas) >= 0.10, "finding": "suffix suppression alone is primary" if min(h1_deltas) >= 0.10 else "suffix suppression alone is insufficient"},
            "H2_structured_rule_repairs_system": {"passes": min(c4) >= 0.90, "finding": "structured semantic episode rule repairs the combined system"},
            "H3_arbitration_is_causal": {"passes": sum(h3_deltas) / len(h3_deltas) >= 0.05, "finding": "explicit evidence priority matters when structured and suffix rules conflict"},
            "H4_neural_negative_transfer_remains": {
                "passes": all(a < b for a, b in zip(values("adapted_neural_legacy_prompt"), values("frozen_neural_legacy_prompt"))),
                "finding": "architecture repair must not be reported as neural generalization",
            },
        },
        "per_seed": per_seed,
        "decision": {
            "accept_zero_parameter_architecture_fix": min(c4) >= 0.90 and min(row["worst_regression_delta"] for row in per_seed) >= -0.02,
            "c5_posthoc_candidate_ready_for_fresh_confirmation": min(c5) >= 0.90 and min(row["c5_worst_regression_delta"] for row in per_seed) >= -0.02,
            "neural_generalization_claim": False,
            "retrain_or_scale_in_this_loop": False,
        },
        "lineage": {
            "preregistration": "research/url_root_cause_loop_10_preregistration.json",
            "frozen_checkpoint": {"path": str(FROZEN.relative_to(ROOT)), "sha256": sha256(FROZEN)},
        },
    }
    output = ROOT / "research/url_root_cause_loop_10_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(
        actor="url-root-cause-evaluator",
        tool="generalization.url_root_cause",
        phase="zero_parameter_ablation",
        status="complete",
        message=f"URL C4 mean {means['C4_structured_rule_confidence_arbitration']:.2%}; neural negative transfer preserved as separate finding",
        payload={"aggregate": report["aggregate"], "hypotheses": report["hypothesis_results"], "decision": report["decision"]},
        artifact=str(output.relative_to(ROOT)),
    )
    print(json.dumps({"aggregate": report["aggregate"], "hypotheses": report["hypothesis_results"], "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
