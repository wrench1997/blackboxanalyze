from __future__ import annotations

from scripts.run_pg366_a800_context_identifiability_null import permute_training_targets


def _row(index: int, target: str) -> dict[str, object]:
    return {"context_tokens": [f"document_field={index}", "request_presence=observed"], "target_tokens": ["[TARGET_BOS]", f"next_action={target}", "[TARGET_EOS]"]}


def test_permutation_preserves_contexts_and_target_multiset() -> None:
    rows = [_row(1, "repair"), _row(2, "abstain"), _row(3, "replay"), _row(4, "select_probe_variant")]
    shuffled, meta = permute_training_targets(rows, seed=36601)
    assert meta["contexts_unchanged"] is True
    assert meta["target_multiset_preserved"] is True
    assert [row["target_tokens"] for row in shuffled] != [row["target_tokens"] for row in rows]


def test_permutation_is_deterministic() -> None:
    rows = [_row(index, "repair" if index % 2 else "abstain") for index in range(1, 9)]
    first, _ = permute_training_targets(rows, seed=36602)
    second, _ = permute_training_targets(rows, seed=36602)
    assert first == second
