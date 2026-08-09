from __future__ import annotations

from scripts import run_pg343_a800_lr1e3_target_sft_v4 as v4


def test_v4_launcher_has_distinct_seeds() -> None:
    assert v4.SEEDS == (34341, 34342, 34343)
