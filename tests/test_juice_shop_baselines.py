from pathlib import Path

from app.juice_shop_baselines import (
    CONFIRMED_DEVIATION_PATHS,
    FrozenLoop11Ranker,
    c5_surface_ranking,
    fixed_synthetic_url_context,
    load_action_bank,
    random_ranking,
    ranking_summary,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "research/juice_shop_loop_12_baseline_protocol.json"


def test_protocol_action_bank_is_unique_and_local_relative():
    actions = load_action_bank(PROTOCOL)
    assert len(actions) == 18
    assert all(action["method"] == "GET" for action in actions)
    assert all(action["path"].startswith("/") and "://" not in action["path"] for action in actions)


def test_random_ranking_is_seeded_and_complete():
    actions = load_action_bank(PROTOCOL)
    first = random_ranking(actions, 12)
    second = random_ranking(actions, 12)
    assert [row.action for row in first] == [row.action for row in second]
    assert {row.action["path"] for row in first} == {row["path"] for row in actions}


def test_c5_is_separate_deterministic_rule_path():
    actions = load_action_bank(PROTOCOL)
    rows = c5_surface_ranking(actions)
    assert rows[0].action["path"] == "/metrics"
    assert rows[0].inferred_family == "observability"
    assert rows[0].action["path"] in CONFIRMED_DEVIATION_PATHS
    summary = ranking_summary(rows, 6)
    assert summary["counterexample_top1"] is True
    assert summary["rule_abstraction_coverage"] == 1.0


def test_synthetic_memory_is_pre_target_and_balanced():
    rows = fixed_synthetic_url_context()
    assert len(rows) == 8
    assert {bool(row["output"]) for row in rows} == {False, True}
    assert all("127.0.0.1" not in row["input"]["endpoint"] for row in rows)


def test_frozen_neural_checkpoint_loads_and_scores_native_prompts():
    ranker = FrozenLoop11Ranker()
    no_memory, _ = ranker.score("/metrics", use_synthetic_memory=False)
    with_memory, _ = ranker.score("/metrics", use_synthetic_memory=True)
    assert 0.0 <= no_memory <= 1.0
    assert 0.0 <= with_memory <= 1.0
