#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
    collate,
    evaluate,
    records_to_examples,
)


OLD_FAMILIES = {"numeric_boundary", "truthiness_gate", "substring_origin", "authorization_or"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path, max_length: int, device: torch.device) -> tuple[TinyRuleGPT, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyRuleGPT(max_length, int(config["hidden"]), int(config["layers"]), int(config["heads"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two checkpoints on identical legacy-family prompts.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--programs", type=int, default=2700)
    parser.add_argument("--traces-per-program", type=int, default=20)
    parser.add_argument("--examples-per-program", type=int, default=4)
    parser.add_argument("--memory-items", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument("--families", default=",".join(sorted(OLD_FAMILIES)))
    parser.add_argument("--experiment", default="generalization-matrix-09-old-family-regression")
    parser.add_argument("--output", type=Path, default=Path("research/generalization_matrix_09_old_family_regression.json"))
    args = parser.parse_args()

    baseline_path = args.baseline if args.baseline.is_absolute() else PROJECT_ROOT / args.baseline
    candidate_path = args.candidate if args.candidate.is_absolute() else PROJECT_ROOT / args.candidate
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    records = generate_curriculum(args.programs, args.traces_per_program, args.seed)
    selected = [record for record in records if record["family"] in families]
    if {record["family"] for record in selected} != families:
        raise RuntimeError("not every requested family is present")
    examples = records_to_examples(
        selected,
        random.Random(args.seed),
        args.examples_per_program,
        args.memory_items,
        routed_semantic_features=True,
        episode_rule_features=True,
    )
    loader = DataLoader(PromptDataset(examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model, _ = load_model(baseline_path, args.max_length, device)
    candidate_model, _ = load_model(candidate_path, args.max_length, device)
    baseline = evaluate(baseline_model, loader, device, max_length=args.max_length, rule_memory_gate=True)
    candidate = evaluate(candidate_model, loader, device, max_length=args.max_length, rule_memory_gate=True)
    baseline_neural = evaluate(baseline_model, loader, device, max_length=args.max_length, rule_memory_gate=False)
    candidate_neural = evaluate(candidate_model, loader, device, max_length=args.max_length, rule_memory_gate=False)
    per_family = {
        family: {
            "baseline": baseline["by_family"][family],
            "candidate": candidate["by_family"][family],
            "delta": round(candidate["by_family"][family] - baseline["by_family"][family], 6),
            "baseline_neural": baseline_neural["by_family"][family],
            "candidate_neural": candidate_neural["by_family"][family],
            "neural_delta": round(candidate_neural["by_family"][family] - baseline_neural["by_family"][family], 6),
        }
        for family in sorted(families)
    }
    worst_delta = min(row["delta"] for row in per_family.values())
    report = {
        "schema_version": "sift-checkpoint-pair-evaluation-v1",
        "experiment": args.experiment,
        "status": "completed",
        "seed": args.seed,
        "data": {"programs": len(selected), "examples": len(examples), "families": sorted(families), "identical_prompts": True},
        "baseline": {"path": str(args.baseline), "sha256": digest(baseline_path), "with_rule_head": baseline, "neural_only": baseline_neural},
        "candidate": {"path": str(args.candidate), "sha256": digest(candidate_path), "with_rule_head": candidate, "neural_only": candidate_neural},
        "per_family": per_family,
        "decision": {
            "max_allowed_regression": args.max_regression,
            "worst_family_delta": round(worst_delta, 6),
            "passes": worst_delta >= -args.max_regression,
        },
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(
        actor="generalization-evaluator",
        tool="generalization.checkpoint_pair",
        phase="regression",
        status="complete" if report["decision"]["passes"] else "failed",
        message=f"Old-family worst delta {worst_delta:+.2%}",
        payload={"per_family": per_family, "decision": report["decision"]},
        artifact=str(output.relative_to(PROJECT_ROOT)),
    )
    print(json.dumps({"baseline": baseline["accuracy"], "candidate": candidate["accuracy"], "per_family": per_family, "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
