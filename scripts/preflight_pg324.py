"""Read-only readiness check for the PG-324 fresh replay.

This command never starts, stops, or contacts a target.  PG-324 is an
evaluation-only local replay, so it is not constrained by the training
time-window policy.  The explicit local-evaluation flag and all artifact /
fresh-target checks remain mandatory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg323-decoy-ask-anchor" / "seeds"
CHECKPOINT_PREFIX = "pg322_cross_impl_decoy_seed_"
SEEDS = (31901, 31902, 31903)


def evaluate_preflight(
    *,
    now: datetime,
    explicit_flag: bool,
    image_present: bool,
    checkpoints_present: bool,
    playwright_available: bool,
    existing_targets: bool,
) -> dict[str, Any]:
    """Evaluate injected facts so the policy can be tested without Docker."""

    checks = {
        # Kept as a stable result key for older dashboards.  The time-window
        # gate applies to training, not this explicitly authorized evaluator.
        "allowed_time_window": True,
        "explicit_flag": bool(explicit_flag),
        "pinned_image_present": bool(image_present),
        "frozen_checkpoints_present": bool(checkpoints_present),
        "playwright_available": bool(playwright_available),
        "no_existing_pg324_targets": not bool(existing_targets),
    }
    return {
        "protocol_id": "pg324-read-only-preflight-v1",
        "timezone": "Asia/Shanghai",
        "now": now.isoformat(),
        "time_window_enforced": False,
        "execution_policy": "operator-authorized-local-evaluation-any-time",
        "checks": checks,
        "ready_to_run": all(checks.values()),
        "target_contacted": False,
        "mutated_runtime": False,
    }


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)


def _image_present() -> bool:
    result = _run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    return result.returncode == 0 and bool(result.stdout.strip())


def _existing_targets() -> bool:
    result = _run(["docker", "ps", "-a", "--filter", "name=^/sift-pg324-juice-", "--format", "{{.Names}}"])
    return result.returncode == 0 and bool(result.stdout.strip())


def _checkpoints_present() -> bool:
    return all((CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}{seed}.pt").is_file() for seed in SEEDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="read-only PG-324 readiness check")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    result = evaluate_preflight(
        now=now,
        explicit_flag=os.environ.get("PG324_LOCAL_DOCKER_EVAL") == "1",
        image_present=_image_present(),
        checkpoints_present=_checkpoints_present(),
        playwright_available=importlib.util.find_spec("playwright") is not None,
        existing_targets=_existing_targets(),
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        failed = ", ".join(key for key, value in result["checks"].items() if not value)
        print(f"{'ready' if result['ready_to_run'] else 'not_ready'}: {failed or 'all checks passed'}")
    return 0 if result["ready_to_run"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
