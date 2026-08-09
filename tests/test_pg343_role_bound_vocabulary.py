from __future__ import annotations

import json

from scripts.build_pg343_role_bound_dataset import build
from scripts.build_pg343_role_bound_vocabulary import build as build_vocabulary


def test_pg343_vocabulary_is_append_only_and_keeps_holdout_as_inventory_only() -> None:
    dataset = build()
    vocab = build_vocabulary(dataset, json.loads(open("research/pg331_web_token_vocabulary_v1.json", encoding="utf-8").read()))
    assert vocab["append_only"] is True
    assert vocab["holdout_policy"]["context_inventory_only"] is True
    assert vocab["holdout_policy"]["target_labels_used_for_vocabulary"] is False
    assert vocab["forbidden_tokens"] == []
    assert "oracle=typed" in vocab["base_forbidden_tokens_removed"]
    assert "belief_probe_role=candidate" in vocab["context_tokens"]
    assert "next_action=select_probe_variant" in vocab["target_tokens"]
    assert all(value is False for value in vocab["promotion"].values())
