from pathlib import Path

from scripts import run_pg347_a800_multi_impl_slot_smoke as runner


ROOT = Path(__file__).resolve().parents[1]


def test_pg347_wrapper_uses_distinct_seeds_without_mutating_base_at_import():
    assert runner.SEEDS == (34701, 34702, 34703)
    assert runner.base.SEEDS != runner.SEEDS


def test_pg347_runner_source_does_not_enable_payload_or_network():
    text = (ROOT / "scripts" / "run_pg347_a800_multi_impl_slot_smoke.py").read_text(encoding="utf-8")
    assert "docker" not in text.casefold()
    assert "requests" not in text.casefold()
