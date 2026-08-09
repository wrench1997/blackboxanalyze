from __future__ import annotations

from scripts import run_pg344_a800_cross_impl_smoke as launcher


def test_pg344_launcher_uses_distinct_seed_block_and_shared_gates() -> None:
    assert launcher.SEEDS == (34401, 34402, 34403)
    assert launcher.base.SCHEMA_VERSION.startswith("pg343-a800-target-conditioned")
    assert launcher.base.FORBIDDEN
