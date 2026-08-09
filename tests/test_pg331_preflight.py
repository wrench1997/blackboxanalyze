from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "preflight_pg331.py"
SPEC = importlib.util.spec_from_file_location("pg331_preflight_test", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _facts(**overrides):
    facts = {
        "now": datetime(2026, 8, 8, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        "explicit_flag": True,
        "image_present": True,
        "required_files_present": True,
        "hash_lock_valid": True,
        "existing_targets": False,
    }
    facts.update(overrides)
    return facts


def test_pg331_preflight_is_ready_only_for_diagnostic_collection():
    result = MODULE.evaluate_preflight(**_facts())

    assert result["ready_for_diagnostic_collection"] is True
    assert result["target_contacted"] is False
    assert result["mutated_runtime"] is False
    assert result["promotion"]["training_allowed"] is False


def test_pg331_preflight_blocks_outside_window_and_missing_flag():
    result = MODULE.evaluate_preflight(
        **_facts(
            now=datetime(2026, 8, 8, 7, 59, tzinfo=ZoneInfo("Asia/Shanghai")),
            explicit_flag=False,
        )
    )

    assert result["ready_for_diagnostic_collection"] is False
    assert result["checks"]["allowed_local_collection_window"] is False
    assert result["checks"]["explicit_flag"] is False


def test_pg331_preflight_blocks_target_reuse_and_hash_or_asset_gaps():
    result = MODULE.evaluate_preflight(
        **_facts(existing_targets=True, required_files_present=False, hash_lock_valid=False)
    )

    assert result["ready_for_diagnostic_collection"] is False
    assert result["checks"]["no_existing_pg331_targets"] is False
    assert result["checks"]["whole_web_and_source_row_assets_present"] is False
    assert result["checks"]["code_hash_lock_valid"] is False


def test_pg331_preflight_locks_evaluator_sidecar_before_collection():
    assert MODULE.ROOT / "app" / "pg331_evaluator_sidecar.py" in MODULE.REQUIRED_FILES
    assert (
        "pg331_evaluator_sidecar_contract",
        "implementation",
        "implementation_sha256",
    ) in MODULE.HASH_CONTRACTS
    assert (
        "pg331_evaluator_sidecar_contract",
        "test",
        "test_sha256",
    ) in MODULE.HASH_CONTRACTS
