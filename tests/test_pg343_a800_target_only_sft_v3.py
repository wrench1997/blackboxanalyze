from __future__ import annotations

from scripts import run_pg343_a800_target_conditioned_smoke as base
from scripts import run_pg343_a800_target_only_sft_v3 as v3


def test_v3_launcher_preserves_full_context_and_uses_new_seeds() -> None:
    assert v3.SEEDS == (34331, 34332, 34333)
    assert base.SEEDS == (34311, 34312, 34313)
