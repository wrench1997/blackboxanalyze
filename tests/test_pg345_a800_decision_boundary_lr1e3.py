from __future__ import annotations

from scripts import run_pg345_a800_decision_boundary_lr1e3 as launcher


def test_pg345_high_lr_launcher_uses_separate_seed_block() -> None:
    assert launcher.SEEDS == (34511, 34512, 34513)
