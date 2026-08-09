#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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
    evaluate_counterexample_at_k,
    records_to_examples,
)


TRAIN_FAMILIES = {
    "numeric_boundary",
    "truthiness_gate",
    "substring_origin",
    "authorization_or",
    "string_suffix_primitive",
    "markup_lexeme_primitive",
}
TEST_FAMILIES = {"postmessage_origin", "dom_sink_injection"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the executable Rule Memory gate on a frozen checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--programs", type=int, default=1800)
    parser.add_argument("--traces-per-program", type=int, default=20)
    parser.add_argument("--examples-per-program", type=int, default=4)
    parser.add_argument("--memory-items", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--feature-mode", choices=("raw", "routed"), default="routed")
    parser.add_argument("--output", type=Path, default=Path("research/frontend_loop_06_gate_20260801.json"))
    args = parser.parse_args()

    records = generate_curriculum(args.programs, args.traces_per_program, args.seed)
    train_records = [row for row in records if row["family"] in TRAIN_FAMILIES]
    iid_records = train_records[::5]
    test_records = [row for row in records if row["family"] in TEST_FAMILIES]
    rng = random.Random(args.seed)
    routed = args.feature_mode == "routed"
    options = {"routed_semantic_features": routed, "episode_rule_features": True}
    iid_examples = records_to_examples(iid_records, rng, args.examples_per_program, args.memory_items, **options)
    test_examples = records_to_examples(test_records, rng, args.examples_per_program, args.memory_items, **options)
    iid_loader = DataLoader(PromptDataset(iid_examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PromptDataset(test_examples, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyRuleGPT(args.max_length, int(config["hidden"]), int(config["layers"]), int(config["heads"])).to(device)
    model.load_state_dict(checkpoint["model_state"])

    iid_neural = evaluate(model, iid_loader, device, max_length=args.max_length)
    iid_gate = evaluate(model, iid_loader, device, max_length=args.max_length, rule_memory_gate=True)
    holdout_neural = evaluate(model, test_loader, device, max_length=args.max_length)
    holdout_gate = evaluate(model, test_loader, device, max_length=args.max_length, rule_memory_gate=True)
    holdout_no_memory = evaluate(model, test_loader, device, drop_memory=True, max_length=args.max_length, rule_memory_gate=True)
    counterexample_gate = evaluate_counterexample_at_k(
        model,
        test_records,
        device,
        args.max_length,
        args.memory_items,
        False,
        False,
        routed,
        True,
        True,
        args.seed,
        k=10,
        batch_size=args.batch_size,
    )
    report = {
        "experiment": f"frontend-{args.feature_mode}-semantics-rule-memory-gate",
        "status": "completed",
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learned_parameters_added_by_gate": 0,
        "data": {
            "programs": args.programs,
            "train_families": sorted(TRAIN_FAMILIES),
            "heldout_families": sorted(TEST_FAMILIES),
            "target_family_examples_in_training": 0,
            "feature_mode": args.feature_mode,
        },
        "results": {
            "iid_neural": iid_neural,
            "iid_gate": iid_gate,
            "holdout_neural": holdout_neural,
            "holdout_gate": holdout_gate,
            "holdout_no_memory": holdout_no_memory,
            "gate_delta": round(holdout_gate["accuracy"] - holdout_neural["accuracy"], 6),
            "memory_delta": round(holdout_gate["accuracy"] - holdout_no_memory["accuracy"], 6),
            "counterexample_at_k": counterexample_gate,
        },
        "guardrails": {
            "source_rule_ir_cwe_or_target_label_visible": False,
            "gate_requires_context_fit": 0.75,
            "same_frozen_checkpoint_for_gate_ablation": True,
        },
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(
        actor="memory-gate",
        tool="rule-memory-gate.evaluate",
        phase="family-holdout",
        status="complete",
        message=f"Rule Memory gate holdout accuracy {holdout_gate['accuracy']:.2%}",
        payload={
            "seed": args.seed,
            "holdout_gate": holdout_gate["accuracy"],
            "gate_delta": report["results"]["gate_delta"],
            "memory_delta": report["results"]["memory_delta"],
            "by_family": holdout_gate["by_family"],
        },
        artifact=str(output.relative_to(PROJECT_ROOT)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
