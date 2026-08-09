from __future__ import annotations

from scripts import run_pg343_a800_target_loss_ablation_v2 as ablation
from scripts import run_pg343_a800_target_conditioned_smoke as base


def test_v2_launcher_uses_distinct_seed_triplet_without_changing_base_contract() -> None:
    assert ablation.SEEDS == (34321, 34322, 34323)
    assert base.SEEDS == (34311, 34312, 34313)
    assert ablation.main.__module__ == ablation.__name__
