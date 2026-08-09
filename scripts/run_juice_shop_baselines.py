#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import (  # noqa: E402
    DockerJuiceShopManager,
    EvidenceLedger,
    JuiceShopAdapter,
    JuiceShopEpisode,
    canonical_json,
)
from app.juice_shop_baselines import (  # noqa: E402
    LOOP11_CHECKPOINT,
    file_sha256,
    load_action_bank,
    policy_builders,
    ranking_summary,
)


PROTOCOL_PATH = ROOT / "research/juice_shop_loop_12_baseline_protocol.json"
RANKINGS_PATH = ROOT / "research/juice_shop_loop_12_frozen_rankings.json"
RUNS_PATH = ROOT / "research/juice_shop_loop_12_baseline_runs.json"
CATALOG_PATH = ROOT / "research/juice_shop_loop_12_catalog_v3.json"
EVIDENCE_DIR = ROOT / "artifacts/juice-shop-loop-12/baselines"
POLICY_SEED = 20262041
ENVIRONMENT_SEEDS = {
    "random": 20262051,
    "frozen_neural_no_memory": 20262053,
    "frozen_neural_synthetic_memory": 20262057,
    "C5_executable_rule": 20262059,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_rankings() -> dict[str, Any]:
    if RANKINGS_PATH.exists():
        raise RuntimeError(f"refusing to overwrite frozen rankings: {RANKINGS_PATH}")
    protocol = read_json(PROTOCOL_PATH)
    actions = load_action_bank(PROTOCOL_PATH)
    budget = int(protocol["action_budget_per_policy"])
    rankings = {
        name: ranking_summary(builder(actions), budget)
        for name, builder in policy_builders(POLICY_SEED).items()
    }
    artifact = {
        "schema_version": "sift-juice-shop-loop-12-frozen-rankings-v1",
        "status": "frozen-before-any-baseline-action",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "checkpoint": str(LOOP11_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(LOOP11_CHECKPOINT),
        "policy_seed": POLICY_SEED,
        "rankings": rankings,
    }
    write_json(RANKINGS_PATH, artifact)
    return artifact


def run_policy(policy: str) -> dict[str, Any]:
    frozen = read_json(RANKINGS_PATH)
    if file_sha256(PROTOCOL_PATH) != frozen["protocol_sha256"]:
        raise RuntimeError("baseline protocol changed after ranking freeze")
    if file_sha256(LOOP11_CHECKPOINT) != frozen["checkpoint_sha256"]:
        raise RuntimeError("Loop 11 checkpoint changed after ranking freeze")
    if policy not in frozen["rankings"]:
        raise ValueError(f"unknown policy: {policy}")

    runs = read_json(RUNS_PATH) if RUNS_PATH.exists() else {
        "schema_version": "sift-juice-shop-loop-12-baseline-runs-v1",
        "rankings": str(RANKINGS_PATH.relative_to(ROOT)),
        "runs": {},
    }
    if policy in runs["runs"]:
        raise RuntimeError(f"refusing to overwrite completed policy: {policy}")

    catalog = read_json(CATALOG_PATH)
    selected_by_key = {row["key"]: row for row in catalog["challenges"]}
    adapter = JuiceShopAdapter()
    manager = DockerJuiceShopManager(adapter)
    environment = manager.reset(ENVIRONMENT_SEEDS[policy])
    before = adapter.evaluator_solved_state()
    evidence_path = EVIDENCE_DIR / f"{policy}.jsonl"
    if evidence_path.exists():
        raise RuntimeError(f"refusing to append to prior baseline evidence: {evidence_path}")
    ledger = EvidenceLedger(evidence_path, ROOT)

    selected = frozen["rankings"][policy]["ranked"][: frozen["rankings"][policy]["budget"]]
    transitions: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    previous = dict(before)
    with JuiceShopEpisode(adapter, ledger=ledger) as episode:
        for item in selected:
            observation = episode.act(item["action"])
            current = adapter.evaluator_solved_state()
            changed = [
                key for key, solved in current.items()
                if solved and not previous.get(key, False) and key in selected_by_key
            ]
            transition_rows = [
                {
                    "key": key,
                    "family": selected_by_key[key]["family"],
                    "split": selected_by_key[key]["split"],
                }
                for key in changed
            ]
            if transition_rows:
                transitions.append({"rank": item["rank"], "action": item["action"], "challenges": transition_rows})
            action_rows.append({
                "rank": item["rank"],
                "action": item["action"],
                "score": item["score"],
                "inferred_family": item["inferred_family"],
                "status_code": observation["observation"]["status_code"],
                "body_length": observation["observation"]["summary"]["body_length"],
                "selected_challenge_transitions": transition_rows,
            })
            previous = current

    solved_rows = [challenge for transition in transitions for challenge in transition["challenges"]]
    abstraction_attempts = [
        {
            "expected_family": challenge["family"],
            "inferred_family": action["inferred_family"],
            "exact": action["inferred_family"] == challenge["family"],
        }
        for action in action_rows
        for challenge in action["selected_challenge_transitions"]
        if action["inferred_family"] is not None
    ]
    run = {
        "policy": policy,
        "environment": environment,
        "action_budget": len(selected),
        "actions": action_rows,
        "episode_success": bool(solved_rows),
        "selected_challenges_solved": len(solved_rows),
        "first_success_probe": transitions[0]["rank"] if transitions else None,
        "transitions": transitions,
        "counterexample_top1": frozen["rankings"][policy]["counterexample_top1"],
        "negative_control_false_positive_rate": frozen["rankings"][policy]["negative_control_false_positive_rate"],
        "rule_abstraction_output_coverage": frozen["rankings"][policy]["rule_abstraction_coverage"],
        "transition_rule_abstraction_attempts": abstraction_attempts,
        "transition_rule_abstraction_accuracy": (
            round(sum(row["exact"] for row in abstraction_attempts) / len(abstraction_attempts), 6)
            if abstraction_attempts else None
        ),
        "evidence": str(evidence_path.relative_to(ROOT)),
    }
    runs["runs"][policy] = run
    write_json(RUNS_PATH, runs)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-rankings", action="store_true")
    parser.add_argument("--run-policy", choices=list(ENVIRONMENT_SEEDS))
    args = parser.parse_args()
    if args.freeze_rankings == bool(args.run_policy):
        parser.error("choose exactly one of --freeze-rankings or --run-policy")
    result = freeze_rankings() if args.freeze_rankings else run_policy(str(args.run_policy))
    print(canonical_json(result))


if __name__ == "__main__":
    main()
