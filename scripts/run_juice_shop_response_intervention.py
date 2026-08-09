#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import DockerJuiceShopManager, EvidenceLedger, JuiceShopAdapter, JuiceShopEpisode  # noqa: E402
from app.response_projection import ResponseProjection  # noqa: E402


PROTOCOL = ROOT / "research/juice_shop_loop_12_response_intervention_protocol.json"
RUNS = ROOT / "research/juice_shop_loop_12_response_intervention_runs_v3.json"
CATALOG = ROOT / "research/juice_shop_loop_12_catalog_v3.json"
EVIDENCE = ROOT / "artifacts/juice-shop-loop-12/response-intervention"
SEEDS = {"response_projection": 20262061, "ablation_disabled_projection": 20262067}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=list(SEEDS))
    args = parser.parse_args()
    if RUNS.exists():
        all_runs = json.loads(RUNS.read_text(encoding="utf-8"))
    else:
        all_runs = {
            "schema_version": "sift-juice-shop-loop-12-response-intervention-runs-v1",
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "runs": {},
        }
    if args.policy in all_runs["runs"]:
        raise RuntimeError("refusing to overwrite an intervention run")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    selected_keys = {row["key"] for row in catalog["challenges"]}
    adapter = JuiceShopAdapter()
    manager = DockerJuiceShopManager(adapter)
    environment = manager.reset(SEEDS[args.policy])
    before = adapter.evaluator_solved_state()
    evidence_path = EVIDENCE / f"{args.policy}.jsonl"
    ledger = EvidenceLedger(evidence_path, ROOT)
    probes = protocol["probe_phase"]
    observations: list[dict[str, Any]] = []
    step_transitions: list[dict[str, Any]] = []
    with JuiceShopEpisode(adapter, ledger=ledger) as episode:
        for action in probes:
            before_step = adapter.evaluator_solved_state()
            observation = episode.act(action)
            projection = ResponseProjection.from_observation(observation)
            after_step = adapter.evaluator_solved_state()
            changed = [key for key, solved in after_step.items() if solved and not before_step.get(key, False)]
            observations.append({"action": action, "projection": projection.to_dict(), "observation": observation, "challenge_transitions": changed})
            step_transitions.extend({
                "request": len(observations),
                "action": action,
                "challenge": key,
                "selected_in_loop12_catalog": key in selected_keys,
            } for key in changed)
        if args.policy == "response_projection":
            chosen = max(
                observations,
                key=lambda row: (row["projection"]["score"], -probes.index(row["action"])),
            )
        else:
            chosen = observations[0]
        get_action = {"method": "GET", "path": chosen["action"]["path"]}
        before_get = adapter.evaluator_solved_state()
        get_observation = episode.act(get_action)
        after_get = adapter.evaluator_solved_state()
        get_all_transitions = [key for key, solved in after_get.items() if solved and not before_get.get(key, False)]
        get_transitions = [key for key in get_all_transitions if key in selected_keys]
    after = adapter.evaluator_solved_state()
    all_transitions = [key for key, solved in after.items() if solved and not before.get(key, False)]
    transitions = [key for key in all_transitions if key in selected_keys]
    selected_step_transitions = [row for row in step_transitions if row["selected_in_loop12_catalog"]]
    run = {
        "policy": args.policy,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "request_budget": len(probes) + 1,
        "probe_observations": observations,
        "step_transitions": step_transitions,
        "all_evaluator_transitions": all_transitions,
        "chosen_get": {"action": get_action, "projection": chosen["projection"]},
        "get_status_code": get_observation["observation"]["status_code"],
        "get_transitions": get_transitions,
        "selected_challenge_transitions": transitions,
        "episode_success": bool(transitions),
        "first_success_request": (
            min(row["request"] for row in selected_step_transitions)
            if selected_step_transitions
            else (len(probes) + 1 if get_transitions else None)
        ),
        "evidence": str(evidence_path.relative_to(ROOT)),
    }
    all_runs["runs"][args.policy] = run
    RUNS.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
