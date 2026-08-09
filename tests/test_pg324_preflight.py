from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preflight_pg324.py"
_SPEC = importlib.util.spec_from_file_location("pg324_preflight_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_preflight = _MODULE.evaluate_preflight


def test_pg324_preflight_is_not_blocked_by_training_window() -> None:
    result = evaluate_preflight(
        now=datetime(2026, 8, 7, 19, 0),
        explicit_flag=True,
        image_present=True,
        checkpoints_present=True,
        playwright_available=True,
        existing_targets=False,
    )
    assert result["ready_to_run"] is True
    assert result["checks"]["allowed_time_window"] is True
    assert result["time_window_enforced"] is False
    assert result["target_contacted"] is False
    assert result["mutated_runtime"] is False


def test_pg324_preflight_passes_only_when_all_facts_are_ready() -> None:
    result = evaluate_preflight(
        now=datetime(2026, 8, 7, 9, 0),
        explicit_flag=True,
        image_present=True,
        checkpoints_present=True,
        playwright_available=True,
        existing_targets=False,
    )
    assert result["ready_to_run"] is True


def test_pg324_preflight_blocks_reuse_or_missing_flag() -> None:
    result = evaluate_preflight(
        now=datetime(2026, 8, 7, 9, 0),
        explicit_flag=False,
        image_present=True,
        checkpoints_present=True,
        playwright_available=True,
        existing_targets=True,
    )
    assert result["ready_to_run"] is False
    assert result["checks"]["explicit_flag"] is False
    assert result["checks"]["no_existing_pg324_targets"] is False
