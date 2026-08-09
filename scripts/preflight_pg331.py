"""Read-only readiness check for the PG-331 Pikachu source collector.

The preflight never starts, stops, or contacts a target.  It only verifies
that the explicit local-evaluation flag, Asia/Shanghai collection window,
pinned image, source-row/whole-page assets, hash locks, and exact target-name
namespace are ready.  A successful result authorizes a *diagnostic* GET/POST
collection; it never authorizes model training or promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = "Asia/Shanghai"
IMAGE = "sift/pikachu-fixed@sha256:cca4288b6b701725e7a771f47ce7fcafd6cea9bd7622fa34ef2ed0b440f472c6"
TARGET_PREFIXES = ("sift-pg331-neutral-", "sift-pg324-juice-")
RULES_PATH = ROOT / "research" / "improvement_rules.json"

REQUIRED_FILES = (
    ROOT / "research" / "pg331_web_token_ontology_v1.json",
    ROOT / "research" / "pg331_web_token_vocabulary_v1.json",
    ROOT / "research" / "pg331_source_row_schema_v1.json",
    ROOT / "app" / "pg331_source_row.py",
    ROOT / "app" / "pg331_evaluator_sidecar.py",
    ROOT / "app" / "pg331_pikachu_docker_relay.py",
    ROOT / "scripts" / "run_pg331_pikachu_source_collection.py",
    ROOT / "scripts" / "run_pg331_typed_replay.py",
    ROOT / "scripts" / "run_pg331_pikachu_typed_source_rows.py",
    ROOT / "scripts" / "run_pg331_juice_shop_source_rows_live.py",
    ROOT / "scripts" / "run_pg331_a800_next_token_smoke.py",
    ROOT / "tests" / "test_pg331_juice_shop_source_rows_live.py",
)

HASH_CONTRACTS = (
    ("pg331_source_row_contract", "implementation", "implementation_sha256"),
    ("pg331_source_row_contract", "schema", "schema_sha256"),
    ("pg331_source_row_contract", "batch_ingest", "batch_ingest_sha256"),
    ("pg331_source_row_contract", "audit", "audit_script_sha256"),
    ("pg331_evaluator_sidecar_contract", "implementation", "implementation_sha256"),
    ("pg331_evaluator_sidecar_contract", "test", "test_sha256"),
    ("pg331_typed_replay_plan_contract", "implementation", "implementation_sha256"),
    ("pg331_typed_replay_plan_contract", "test", "test_sha256"),
    ("pg331_model_training_contract", "implementation", "implementation_sha256"),
    ("pg331_model_training_contract", "test", "test_sha256"),
    ("pg331_model_training_contract", "model_implementation", "model_implementation_sha256"),
    ("pg331_model_training_contract", "capacity_audit", "capacity_audit_sha256"),
    ("pg331_model_training_contract", "capacity_audit_test", "capacity_audit_test_sha256"),
    ("pg331_typed_source_row_contract", "implementation", "implementation_sha256"),
    ("pg331_typed_source_row_contract", "test", "test_sha256"),
    ("pg331_pikachu_docker_collection_contract", "implementation", "implementation_sha256"),
    ("pg331_pikachu_docker_collection_contract", "runner", "runner_sha256"),
    ("pg331_pikachu_docker_collection_contract", "test", "test_sha256"),
    ("pg331_pikachu_docker_collection_contract", "preflight", "preflight_sha256"),
    ("pg331_pikachu_docker_collection_contract", "adapter", "adapter_sha256"),
    ("pg331_pikachu_docker_collection_contract", "adapter_test", "adapter_test_sha256"),
    ("pg331_juice_shop_source_row_live_contract", "implementation", "implementation_sha256"),
    ("pg331_juice_shop_source_row_live_contract", "test", "test_sha256"),
)


def _within_window(now: datetime) -> bool:
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now.tzinfo else now.replace(tzinfo=ZoneInfo(TIMEZONE))
    return 8 <= local.hour < 18


def evaluate_preflight(
    *,
    now: datetime,
    explicit_flag: bool,
    image_present: bool,
    required_files_present: bool,
    hash_lock_valid: bool,
    existing_targets: bool,
) -> dict[str, Any]:
    """Evaluate injected facts so the gate can be tested without Docker."""

    checks = {
        "allowed_local_collection_window": _within_window(now),
        "explicit_flag": bool(explicit_flag),
        "pinned_image_present": bool(image_present),
        "whole_web_and_source_row_assets_present": bool(required_files_present),
        "code_hash_lock_valid": bool(hash_lock_valid),
        "no_existing_pg331_targets": not bool(existing_targets),
    }
    ready = all(checks.values())
    return {
        "protocol_id": "pg331-read-only-preflight-v1",
        "timezone": TIMEZONE,
        "now": now.isoformat(),
        "time_window_enforced": True,
        "execution_policy": "operator-authorized-local-diagnostic-collection-only",
        "checks": checks,
        "ready_for_diagnostic_collection": ready,
        "target_contacted": False,
        "mutated_runtime": False,
        "first_pass_expected": "incomplete_ask_without_typed_evaluator",
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)


def _image_present() -> bool:
    result = _run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    return result.returncode == 0 and bool(result.stdout.strip())


def _existing_targets() -> bool:
    for prefix in TARGET_PREFIXES:
        result = _run(["docker", "ps", "-a", "--filter", f"name=^/{prefix}", "--format", "{{.Names}}"])
        if result.returncode != 0:
            return True
        if result.stdout.strip():
            return True
    return False


def _required_files_present() -> bool:
    return all(path.is_file() for path in REQUIRED_FILES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_lock_valid() -> bool:
    try:
        rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        for contract_name, path_key, hash_key in HASH_CONTRACTS:
            contract = rules[contract_name]
            expected = str(contract[hash_key])
            path = ROOT / str(contract[path_key])
            if not path.is_file() or _sha256(path) != expected:
                return False
        collection = rules["pg331_pikachu_docker_collection_contract"]
        if collection.get("pinned_image") != IMAGE:
            return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="read-only PG-331 collection readiness check")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    now = datetime.now(ZoneInfo(TIMEZONE))
    result = evaluate_preflight(
        now=now,
        explicit_flag=os.environ.get("PG331_LOCAL_DOCKER_EVAL") == "1",
        image_present=_image_present(),
        required_files_present=_required_files_present(),
        hash_lock_valid=_hash_lock_valid(),
        existing_targets=_existing_targets(),
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        failed = ", ".join(key for key, value in result["checks"].items() if not value)
        print(f"{'ready' if result['ready_for_diagnostic_collection'] else 'not_ready'}: {failed or 'all checks passed'}")
    return 0 if result["ready_for_diagnostic_collection"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
