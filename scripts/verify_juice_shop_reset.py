from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.juice_shop_adapter import (  # noqa: E402
    DockerJuiceShopManager,
    JuiceShopAdapter,
    JuiceShopEpisode,
    PINNED_IMAGE,
    TARGET_CONTAINER,
    catalog_sha256,
)


def observe_environment(adapter: JuiceShopAdapter, seed: int, startup_seconds: float | None) -> dict:
    inspect_row = json.loads(subprocess.run(
        ["docker", "inspect", TARGET_CONTAINER], check=True, capture_output=True, text=True, timeout=30
    ).stdout)[0]
    catalog = adapter.evaluator_catalog()
    solved = adapter.evaluator_solved_state()
    with JuiceShopEpisode(adapter) as episode:
        observation = episode.act({"method": "GET", "path": "/rest/products/search?q=apple"})
    return {
        "environment_seed": seed,
        "container_id": inspect_row["Id"],
        "startup_seconds": startup_seconds,
        "initial_solved_count": sum(solved.values()),
        "challenge_count": len(solved),
        "catalog_sha256": catalog_sha256(catalog),
        "probe_status": observation["observation"]["status_code"],
        "probe_body_sha256": observation["observation"]["summary"]["body_sha256"],
        "probe_semantic_sha256": observation["observation"]["summary"]["semantic_body_sha256"],
        "target_networks": sorted(inspect_row["NetworkSettings"]["Networks"]),
        "published_target_ports": [key for key, value in inspect_row["NetworkSettings"]["Ports"].items() if value],
        "image": inspect_row["Image"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-seed", type=int, required=True)
    parser.add_argument("--fresh-seed", type=int, required=True)
    parser.add_argument("--current-startup-seconds", type=float)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "juice_shop_loop_12_reset_verification.json")
    args = parser.parse_args()

    adapter = JuiceShopAdapter()
    manager = DockerJuiceShopManager(adapter)
    current = observe_environment(adapter, args.current_seed, args.current_startup_seconds)
    started = time.perf_counter()
    reset = manager.reset(args.fresh_seed)
    fresh = observe_environment(adapter, args.fresh_seed, round(time.perf_counter() - started, 3))
    checks = {
        "both_start_unsolved": current["initial_solved_count"] == fresh["initial_solved_count"] == 0,
        "catalog_reproduced": current["catalog_sha256"] == fresh["catalog_sha256"],
        "probe_semantically_reproduced": current["probe_semantic_sha256"] == fresh["probe_semantic_sha256"],
        "target_has_no_published_ports": not current["published_target_ports"] and not fresh["published_target_ports"],
        "target_only_on_internal_network": current["target_networks"] == fresh["target_networks"] == ["sift-loop12-internal"],
        "image_pinned": current["image"] == fresh["image"] == PINNED_IMAGE.rsplit("@", 1)[1],
    }
    report = {
        "schema_version": "sift-juice-shop-reset-verification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "current": current,
        "fresh": fresh,
        "manager_reset": reset,
        "checks": checks,
        "reset_reproduction_rate": sum((
            checks["both_start_unsolved"], checks["catalog_reproduced"], checks["probe_semantically_reproduced"]
        )) / 3,
        "engineering_note": "Startup cost is measured separately from scientific task performance.",
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
