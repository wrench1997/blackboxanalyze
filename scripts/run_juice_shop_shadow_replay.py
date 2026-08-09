#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import DockerJuiceShopManager, EvidenceLedger, JuiceShopAdapter, JuiceShopEpisode  # noqa: E402
from app.response_projection import ResponseProjection  # noqa: E402


PROTOCOL = ROOT / "research/juice_shop_loop_12_shadow_replay_protocol.json"
RUNS = ROOT / "research/juice_shop_loop_12_shadow_replay_runs.json"
PROBE = ROOT / "infra/loop12/shadow_probe.mjs"
EVIDENCE_DIR = ROOT / "artifacts/juice-shop-loop-12/shadow-replay"
SHADOW_CONTAINER = "sift-loop12-juice-shadow"
SHADOW_SEEDS = {"response_projection": 20262073, "ablation_disabled_projection": 20262079}
TARGET_SEEDS = {"response_projection": 20262083, "ablation_disabled_projection": 20262089}
PINNED_IMAGE = "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe"
TARGET_NETWORK = "sift-loop12-internal"


def docker(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=timeout)


def exact_shadow_exists() -> bool:
    result = docker(["ps", "-a", "--filter", f"name=^/{SHADOW_CONTAINER}$", "--format", "{{.Names}}"])
    return result.stdout.strip() == SHADOW_CONTAINER


def cleanup_shadow() -> None:
    if exact_shadow_exists():
        docker(["rm", "-f", SHADOW_CONTAINER])


def start_shadow() -> dict[str, Any]:
    cleanup_shadow()
    command = [
        "run", "-d",
        "--name", SHADOW_CONTAINER,
        "--network", TARGET_NETWORK,
        "--add-host", "www.alchemy.com:127.0.0.1",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--memory", "1g",
        "--cpus", "2",
        "--mount", f"type=bind,source={PROBE.resolve()},target=/shadow_probe.mjs,readonly",
        PINNED_IMAGE,
    ]
    container_id = docker(command).stdout.strip()
    deadline = time.monotonic() + 120
    last_error = "not checked"
    while time.monotonic() < deadline:
        try:
            result = shadow_probe("/")
            if result.get("status_code") == 200:
                return {"container_id": container_id, "network": TARGET_NETWORK, "health_status": 200}
            last_error = json.dumps(result)
        # A just-started image may reject the connection until the app server
        # is listening.  shadow_probe wraps the Node-side error in
        # RuntimeError so the startup loop must treat that as a transient
        # health failure rather than aborting the whole replay.
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError, RuntimeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise RuntimeError(f"shadow target did not become healthy: {last_error}")


def shadow_probe(path: str) -> dict[str, Any]:
    try:
        result = docker(["exec", SHADOW_CONTAINER, "/nodejs/bin/node", "/shadow_probe.mjs", path], timeout=20)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"shadow probe failed path={path!r} rc={error.returncode} "
            f"stdout={error.stdout!r} stderr={error.stderr!r}"
        ) from error
    if not result.stdout.strip():
        raise RuntimeError(f"shadow probe returned empty output path={path!r} stderr={result.stderr!r}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"shadow probe returned invalid JSON path={path!r} stdout={result.stdout!r} stderr={result.stderr!r}") from error


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=list(SHADOW_SEEDS))
    args = parser.parse_args()
    if RUNS.exists():
        runs = json.loads(RUNS.read_text(encoding="utf-8"))
    else:
        runs = {"schema_version": "sift-juice-shop-loop-12-shadow-replay-runs-v1", "protocol": str(PROTOCOL.relative_to(ROOT)), "runs": {}}
    if args.policy in runs["runs"]:
        raise RuntimeError("refusing to overwrite an existing shadow replay run")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    probe_actions = protocol["shadow_phase"]["actions"]
    try:
        shadow = start_shadow()
        shadow_rows = []
        for action in probe_actions:
            raw = shadow_probe(action["path"])
            observation = {
                "action": action,
                "observation": {
                    "status_code": raw.get("status_code", 0),
                    "headers": raw.get("headers", {}),
                    "summary": {"body_length": raw.get("body_length", 0), "body_shape": raw.get("body_shape")},
                },
            }
            projection = ResponseProjection.from_observation(observation)
            shadow_rows.append({"action": action, "projection": projection.to_dict(), "raw": raw})
    finally:
        cleanup_shadow()

    if args.policy == "response_projection":
        chosen = max(shadow_rows, key=lambda row: (row["projection"]["score"], -shadow_rows.index(row)))
    else:
        chosen = shadow_rows[0]

    adapter = JuiceShopAdapter()
    environment = DockerJuiceShopManager(adapter).reset(TARGET_SEEDS[args.policy])
    evidence_path = EVIDENCE_DIR / f"{args.policy}.jsonl"
    before = adapter.evaluator_solved_state()
    with JuiceShopEpisode(adapter, ledger=EvidenceLedger(evidence_path, ROOT)) as episode:
        evaluation_action = {"method": "GET", "path": chosen["action"]["path"]}
        observation = episode.act(evaluation_action)
    after = adapter.evaluator_solved_state()
    catalog = json.loads((ROOT / "research/juice_shop_loop_12_catalog_v3.json").read_text(encoding="utf-8"))
    selected_keys = {row["key"] for row in catalog["challenges"]}
    all_transitions = [key for key, solved in after.items() if solved and not before.get(key, False)]
    selected_transitions = [key for key in all_transitions if key in selected_keys]
    run = {
        "policy": args.policy,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "shadow": shadow,
        "shadow_probe_count": len(shadow_rows),
        "shadow_observations": shadow_rows,
        "evaluation_environment": environment,
        "evaluation_action": evaluation_action,
        "evaluation_status_code": observation["observation"]["status_code"],
        "all_evaluator_transitions": all_transitions,
        "selected_loop12_transitions": selected_transitions,
        "episode_success": bool(selected_transitions),
        "evaluation_request_count": 1,
        "evidence": str(evidence_path.relative_to(ROOT)),
    }
    runs["runs"][args.policy] = run
    RUNS.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
