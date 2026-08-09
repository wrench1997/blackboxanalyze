#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import search_rules  # noqa: E402


ACTIONS = ["verify", "wait", "inspect", "cancel"]


def history_event(action: str) -> dict[str, Any]:
    return {"input": {"action": action}, "context": {}, "state": {}, "output": False}


def build_dataset(episodes: int, delay: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    long_context_rows = []
    memory_rows = []
    for index in range(episodes):
        relevant = "verify" if index % 2 == 0 else rng.choice(["wait", "inspect", "cancel"])
        history = [history_event(relevant)]
        history.extend(history_event(rng.choice(ACTIONS[1:])) for _ in range(delay - 1))
        current = "commit" if index % 4 in {0, 1} else "wait"
        output = current == "commit" and relevant == "verify"
        common = {
            "episode_id": f"memory-{index + 1}",
            "step": delay,
            "input": {"action": current},
            "context": {},
            "output": output,
        }
        long_context_rows.append({**common, "state": {}, "history": history})
        memory_rows.append({
            **common,
            "state": {"retrieved_verified": relevant == "verify"},
            "history": [],
        })
    return long_context_rows, memory_rows


def run_condition(
    name: str,
    fields: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    history_depth: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = search_rules(fields, observations, max_depth=2, beam_width=180, history_depth=history_depth)
    elapsed = time.perf_counter() - started
    best = candidates[0] if candidates else None
    return {
        "condition": name,
        "history_depth": history_depth,
        "visible_memory_slots": history_depth if history_depth else int(any(field["path"].startswith("state.") for field in fields)),
        "best_accuracy": round(best.accuracy, 6) if best else 0.0,
        "best_rule": best.to_dict()["pretty"] if best else None,
        "candidate_count": len(candidates),
        "search_seconds": round(elapsed, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no memory, long context and compressed episodic memory.")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--delay", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.delay < 1 or args.delay > 8:
        raise SystemExit("--delay must be between 1 and 8 for this bounded experiment")

    long_rows, memory_rows = build_dataset(args.episodes, args.delay, args.seed)
    action_fields = [{"path": "input.action", "type": "enum", "domain": ["commit", "wait", "verify", "inspect", "cancel"]}]
    memory_fields = action_fields + [{"path": "state.retrieved_verified", "type": "bool", "domain": [False, True]}]
    report = {
        "experiment": "delayed-rule-memory-ablation",
        "episodes": args.episodes,
        "relevant_event_delay": args.delay,
        "conditions": [
            run_condition("no_memory", action_fields, long_rows, history_depth=0),
            run_condition("raw_long_context", action_fields, long_rows, history_depth=args.delay),
            run_condition("retrieved_episodic_memory", memory_fields, memory_rows, history_depth=0),
        ],
        "interpretation": {
            "working_memory": "Raw context can recover the delayed dependency but its token cost grows with delay.",
            "long_term_memory": "A retrieved structured memory preserves the relevant fact with constant context cost.",
            "rl_role": "Train write/retrieve/forget actions using future rule accuracy, counterexample validity and token cost as reward terms.",
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
