import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("pg263_test_runner", ROOT / "scripts" / "run_pg263_pg262_augmented_masked_capacity_training.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pg263_split_contains_audited_pg262_and_fresh_even_seed_holdout():
    runner = _module()
    rows = runner._load_records()
    runner.PG260.FRESH_SOURCE_PREFIXES = ("pg259_", "pg260_", "pg262_")
    holdout = [row for row in rows if runner._is_holdout(row)]
    fresh_holdout = [row for row in holdout if runner.PG260._is_fresh_source(row)]
    pg262 = [row for row in rows if str(row.get("source", "")).startswith("pg262_")]
    pg262_holdout = [row for row in pg262 if runner._is_holdout(row)]

    assert len(rows) == 278
    assert len(pg262) == 20
    assert len(pg262_holdout) == 9
    assert len(fresh_holdout) == 32
    assert Counter(row["rule_ir_class"] for row in pg262_holdout)
    assert all(row.get("tokens") for row in rows)
