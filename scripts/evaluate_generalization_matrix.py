#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
    evaluate_counterexample_at_k,
    records_to_examples,
)


TARGET_FAMILIES = {
    "url_scheme_downgrade",
    "dom_double_decode",
    "unicode_casefold_role",
    "numeric_string_coercion",
    "compound_origin_role",
    "state_replay_window",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint evaluation across mechanism-level generalization axes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--programs", type=int, default=2700)
    parser.add_argument("--traces-per-program", type=int, default=20)
    parser.add_argument("--examples-per-program", type=int, default=4)
    parser.add_argument("--memory-items", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--experiment", default="generalization-matrix-09-checkpoint-evaluation")
    parser.add_argument("--output", type=Path, default=Path("research/generalization_matrix_09_stage_a.json"))
    args = parser.parse_args()

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    records = generate_curriculum(args.programs, args.traces_per_program, args.seed)
    target_records = [row for row in records if row["family"] in TARGET_FAMILIES]
    if {row["family"] for row in target_records} != TARGET_FAMILIES:
        raise RuntimeError("not every preregistered target family is present")
    rng = random.Random(args.seed)
    options = {"routed_semantic_features": True, "episode_rule_features": True}
    examples = records_to_examples(target_records, rng, args.examples_per_program, args.memory_items, **options)
    loader = DataLoader(PromptDataset(examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyRuleGPT(args.max_length, int(config["hidden"]), int(config["layers"]), int(config["heads"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    with_memory = evaluate(model, loader, device, max_length=args.max_length, rule_memory_gate=True)
    neural_with_memory = evaluate(model, loader, device, max_length=args.max_length, rule_memory_gate=False)
    without_memory = evaluate(model, loader, device, drop_memory=True, max_length=args.max_length, rule_memory_gate=True)
    counterexample = evaluate_counterexample_at_k(
        model, target_records, device, args.max_length, args.memory_items,
        False, False, True, True, True, args.seed, k=10, batch_size=args.batch_size,
    )
    gated_by_family: Counter[str] = Counter()
    total_by_family: Counter[str] = Counter()
    for example in examples:
        total_by_family[example.family] += 1
        if _rule_memory_prediction(example.prompt) is not None:
            gated_by_family[example.family] += 1

    axes = {row["family"]: row["generalization"]["axis"] for row in target_records}
    per_axis = {
        axes[family]: {
            "family": family,
            "behavior_agreement": with_memory["by_family"][family],
            "neural_with_memory": neural_with_memory["by_family"][family],
            "without_memory": without_memory["by_family"][family],
            "executable_gate_delta": round(with_memory["by_family"][family] - neural_with_memory["by_family"][family], 6),
            "memory_delta": round(with_memory["by_family"][family] - without_memory["by_family"][family], 6),
            "executable_memory_coverage": round(gated_by_family[family] / total_by_family[family], 6),
            "counterexample_top1": counterexample["by_family_top1"].get(family) if "by_family_top1" in counterexample else None,
        }
        for family in sorted(TARGET_FAMILIES)
    }
    diagnostic = []
    for axis, values in per_axis.items():
        score = values["behavior_agreement"]
        if score < 0.65 and axis == "state_and_history_shift":
            cause = "history representation gap: current prompt projection omits trace.history"
        elif score < 0.65 and values["executable_memory_coverage"] == 0:
            cause = "primitive/representation gap: no executable episode rule was induced"
        elif score < 0.65:
            cause = "decoder or composition gap despite available episode memory"
        else:
            cause = "no blocking gap in frozen zero-shot pilot"
        diagnostic.append({"axis": axis, "family": values["family"], "classification": cause, "score": score})

    manifest_path = PROJECT_ROOT / f"artifacts/generalization-matrix-09-seed-{args.seed}/manifest.json"
    prereg_path = PROJECT_ROOT / "research/generalization_matrix_preregistration.json"
    report: dict[str, Any] = {
        "schema_version": "sift-generalization-matrix-result-v2",
        "experiment": args.experiment,
        "status": "completed",
        "seed": args.seed,
        "device": str(device),
        "checkpoint": {"path": str(args.checkpoint), "sha256": digest(checkpoint_path), "parameters": sum(p.numel() for p in model.parameters())},
        "lineage": {
            "preregistration": "research/generalization_matrix_preregistration.json",
            "preregistration_sha256": digest(prereg_path),
            "dataset_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "dataset_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_sha256"],
        },
        "guardrails": {
            "target_family_examples_in_training": 0,
            "checkpoint_frozen_before_family_creation": True,
            "source_family_rule_ir_cwe_and_intended_output_hidden": True,
            "new_learned_parameters": 0,
        },
        "data": {"programs": args.programs, "target_programs": len(target_records), "target_examples": len(examples), "families": sorted(TARGET_FAMILIES)},
        "results": {
            "with_executable_memory": with_memory,
            "neural_with_memory": neural_with_memory,
            "without_memory": without_memory,
            "executable_gate_delta": round(with_memory["accuracy"] - neural_with_memory["accuracy"], 6),
            "memory_delta": round(with_memory["accuracy"] - without_memory["accuracy"], 6),
            "counterexample_at_10": counterexample,
            "per_axis": per_axis,
        },
        "diagnostic": diagnostic,
        "decision": "Use this map to select primitive-only stage-B adaptations; do not scale model size.",
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(
        actor="generalization-evaluator", tool="generalization.matrix.evaluate", phase="frozen_zero_shot", status="complete",
        message=f"Frozen model macro agreement {with_memory['accuracy']:.2%} across six unseen shift families",
        payload={"seed": args.seed, "macro": with_memory["accuracy"], "memory_delta": report["results"]["memory_delta"], "per_axis": per_axis, "diagnostic": diagnostic},
        artifact=str(output.relative_to(PROJECT_ROOT)),
    )
    print(json.dumps({"macro": with_memory["accuracy"], "neural_with_memory": neural_with_memory["accuracy"], "without_memory": without_memory["accuracy"], "per_axis": per_axis, "diagnostic": diagnostic}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
