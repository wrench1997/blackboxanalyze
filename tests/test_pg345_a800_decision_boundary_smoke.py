from __future__ import annotations

from scripts import run_pg345_a800_decision_boundary_smoke as launcher


def test_pg345_launcher_uses_preregistered_seed_block() -> None:
    assert launcher.SEEDS == (34501, 34502, 34503)
    assert launcher.base.FORBIDDEN
