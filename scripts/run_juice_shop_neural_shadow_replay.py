#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.juice_shop_adapter import DockerJuiceShopManager, EvidenceLedger, JuiceShopAdapter, JuiceShopEpisode  # noqa: E402
from app.response_head import score_observation  # noqa: E402
from app.response_projection import ResponseProjection  # noqa: E402
from train_neural_url_set_head import TinyRuleSetGPT  # noqa: E402
from run_juice_shop_shadow_replay import cleanup_shadow, shadow_probe, start_shadow  # noqa: E402


PROTOCOL = ROOT / "research/juice_shop_loop_12_neural_shadow_protocol.json"
RUNS = ROOT / "research/juice_shop_loop_12_neural_shadow_runs.json"
CHECKPOINT = ROOT / "artifacts/neural-juice-loop-12-response-head-v2-20262097-rerun/tiny_rule_set_gpt.pt"
EVIDENCE_DIR = ROOT / "artifacts/juice-shop-loop-12/neural-shadow"
POLICIES = {"trained_response_head": (20262093, 20262095, True), "trained_response_head_ablation": (20262099, 20262101, False)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=list(POLICIES))
    args = parser.parse_args()
    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else {"schema_version": "sift-juice-shop-loop-12-neural-shadow-runs-v1", "protocol": str(PROTOCOL.relative_to(ROOT)), "runs": {}}
    if args.policy in runs["runs"]:
        raise RuntimeError("refusing to overwrite an existing neural shadow run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state"])
    model.eval()
    shadow_seed, target_seed, enabled = POLICIES[args.policy]
    try:
        shadow = start_shadow()
        actions = json.loads(PROTOCOL.read_text(encoding="utf-8"))["shadow_actions"]
        shadow_rows = []
        for path in actions:
            raw = shadow_probe(path)
            observation = {"action": {"method": "GET", "path": path}, "observation": {"status_code": raw.get("status_code", 0), "headers": raw.get("headers", {}), "summary": {"body_length": raw.get("body_length", 0), "body_shape": raw.get("body_shape")}}}
            projection = ResponseProjection.from_observation(observation)
            score = score_observation(model, projection, device=device, enabled=enabled)
            shadow_rows.append({"path": path, "projection": projection.to_dict(), "model_score": score, "raw": raw})
        chosen = max(shadow_rows, key=lambda row: (row["model_score"], -shadow_rows.index(row)))
    finally:
        cleanup_shadow()

    adapter = JuiceShopAdapter()
    environment = DockerJuiceShopManager(adapter).reset(target_seed)
    evidence_path = EVIDENCE_DIR / f"{args.policy}.jsonl"
    before = adapter.evaluator_solved_state()
    with JuiceShopEpisode(adapter, ledger=EvidenceLedger(evidence_path, ROOT)) as episode:
        action = {"method": "GET", "path": chosen["path"]}
        result = episode.act(action)
    after = adapter.evaluator_solved_state()
    catalog = json.loads((ROOT / "research/juice_shop_loop_12_catalog_v3.json").read_text(encoding="utf-8"))
    selected_keys = {row["key"] for row in catalog["challenges"]}
    transitions = [key for key, solved in after.items() if solved and not before.get(key, False)]
    selected_transitions = [key for key in transitions if key in selected_keys]
    run = {
        "policy": args.policy,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "shadow_seed": shadow_seed,
        "shadow": shadow,
        "shadow_rows": shadow_rows,
        "evaluation_environment": environment,
        "evaluation_action": action,
        "evaluation_status_code": result["observation"]["status_code"],
        "all_evaluator_transitions": transitions,
        "selected_loop12_transitions": selected_transitions,
        "episode_success": bool(selected_transitions),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "evidence": str(evidence_path.relative_to(ROOT)),
    }
    runs["runs"][args.policy] = run
    RUNS.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
