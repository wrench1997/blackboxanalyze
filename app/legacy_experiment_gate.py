"""Quarantine pre-typed-oracle experiments from future training admission."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


LEGACY_POLICY_SCHEMA = "pg-pk-26-legacy-experiment-quarantine-v1"
LEGACY_DIAGNOSTIC_ENV = "PG25_ALLOW_LEGACY_DIAGNOSTIC"


def legacy_status(artifact: str) -> dict[str, object]:
    """Return the immutable status applied to old, non-acceptance-gated data."""

    return {
        "artifact": str(artifact),
        "status": "legacy_diagnostic_only",
        "training_eligible": False,
        "calibration_eligible": False,
        "memory_promotion": False,
        "allowed_use": ["regression", "failure_analysis", "historical_comparison"],
        "required_replacement": "typed_oracle_acceptance_catalog",
    }


def assert_legacy_training_blocked(artifacts: Iterable[Path | str]) -> None:
    """Fail closed unless a caller explicitly asks for historical diagnostics."""

    if os.environ.get(LEGACY_DIAGNOSTIC_ENV) == "1":
        return
    names = ", ".join(str(Path(item)) for item in artifacts)
    raise RuntimeError(
        "legacy experiments are diagnostic-only: missing typed payload-success acceptance; "
        f"blocked training inputs={names}. Set {LEGACY_DIAGNOSTIC_ENV}=1 only for regression diagnostics."
    )


__all__ = ["LEGACY_DIAGNOSTIC_ENV", "LEGACY_POLICY_SCHEMA", "assert_legacy_training_blocked", "legacy_status"]
