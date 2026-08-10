"""Read-only preflight for the optional PG-388 implementation-B container.

This command never builds, starts, stops, or contacts a target.  It only
checks local source hashes, compose text, explicit environment gates, and
optionally runs ``docker compose config --quiet`` (configuration parsing
only).  A passing preflight is not image attestation or training approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_B_SOURCE_SHA256 = "102270024d7e7f40c4ca4a9435d701438228db30ef84b79eeba780873978fb8f"
EXPECTED_B_DOCKERFILE_SHA256 = "cc16b097bcca6da455d7e2e45366299b1742b6ec248eebfb77d91ace617aeef6"
SCHEMA_VERSION = "pg388-holdout-docker-preflight-v1"
_DIGEST_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_ok(value: str | None) -> bool:
    if not value:
        return False
    candidate = value.rsplit("@", 1)[-1]
    return bool(_DIGEST_RE.fullmatch(candidate))


def _compose_contract() -> tuple[bool, list[str]]:
    text = (ROOT / "docker-compose.pg388.yml").read_text(encoding="utf-8")
    failures: list[str] = []
    required = {
        'profiles: ["holdout"]': "holdout_profile",
        "Dockerfile.b": "holdout_dockerfile",
        "PG388_PYTHON_IMAGE_DIGEST_B": "holdout_digest_arg",
        "read_only: true": "read_only",
        "internal: true": "internal_network",
        '"8089"': "private_holdout_port",
    }
    for marker, name in required.items():
        if marker not in text:
            failures.append(name)
    block = text.split("  pg388-backend-b:", 1)[-1].split("  pg388-frontend:", 1)[0]
    if "ports:" in block:
        failures.append("holdout_publishes_port")
    return not failures, failures


def _run_config(env: Mapping[str, str], runner: Callable[..., Any] | None = None) -> tuple[bool, str]:
    command: Sequence[str] = ("docker", "compose", "--profile", "holdout", "-f", "docker-compose.pg388.yml", "config", "--quiet")
    if runner is not None:
        result = runner(command, cwd=str(ROOT), env=dict(env), capture_output=True, text=True, check=False)
    else:
        try:
            result = subprocess.run(command, cwd=ROOT, env=dict(env), capture_output=True, text=True, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, "docker_compose_config_unavailable"
    return result.returncode == 0, "ok" if result.returncode == 0 else "docker_compose_config_failed"


def preflight(environ: Mapping[str, str] | None = None, *, check_compose: bool = False, runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    gate = env.get("PG388_LOCAL_DOCKER_EVAL") == "1"
    source_sha = _sha(ROOT / "fixtures/pg388/logic_lab_b.py")
    dockerfile_sha = _sha(ROOT / "fixtures/pg388/Dockerfile.b")
    compose_ok, compose_failures = _compose_contract()
    checks: dict[str, Any] = {
        "explicit_local_eval_gate": gate,
        "holdout_python_digest_format": _digest_ok(env.get("PG388_PYTHON_IMAGE_DIGEST_B")),
        "display_python_digest_format": _digest_ok(env.get("PG388_PYTHON_IMAGE_DIGEST")),
        "node_base_digest_format": _digest_ok(env.get("PG388_NODE_BASE_IMAGE")),
        "source_hash_locked": source_sha == EXPECTED_B_SOURCE_SHA256,
        "dockerfile_hash_locked": dockerfile_sha == EXPECTED_B_DOCKERFILE_SHA256,
        "compose_contract": compose_ok,
        "compose_config_checked": False,
    }
    reasons: list[str] = list(compose_failures)
    if not gate:
        reasons.append("PG388_LOCAL_DOCKER_EVAL=1_required")
    for name in ("holdout_python_digest_format", "display_python_digest_format", "node_base_digest_format", "source_hash_locked", "dockerfile_hash_locked"):
        if checks[name] is not True:
            reasons.append(name)
    if check_compose and not reasons:
        config_ok, config_reason = _run_config(env, runner=runner)
        checks["compose_config_checked"] = True
        checks["compose_config"] = config_ok
        if not config_ok:
            reasons.append(config_reason)
    status = "ready_for_operator_review" if not reasons else "blocked_holdout_docker_preflight"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checks": checks,
        "reasons": reasons,
        "image_attested": False,
        "runtime_started": False,
        "docker_started": False,
        "target_contacted": False,
        "external_network": False,
        "gpu_touched": False,
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "next_action": "operator_review_immutable_images_then_run_explicit_holdout" if not reasons else "fix_preflight_reasons_before_any_runtime",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-compose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = preflight(check_compose=args.check_compose)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))


if __name__ == "__main__":  # pragma: no cover
    main()
