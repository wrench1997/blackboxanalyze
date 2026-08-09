from __future__ import annotations

from scripts.build_pg367_waf_staircase_dataset_v2 import build


def test_v2_is_compositional_and_value_covered() -> None:
    document = build()
    assert document["schema_version"].endswith("v2")
    assert document["counts"]["train_rows"] > document["counts"]["implementation_holdout_rows"]
    values = {"train": set(), "implementation_holdout": set()}
    for row in document["records"]:
        values[row["split"]].update(row["context_tokens"])
        values[row["split"]].update(row["target_tokens"])
    assert values["implementation_holdout"] <= values["train"]


def test_v2_keeps_promotion_closed() -> None:
    document = build()
    assert all(value is False for value in document["promotion"].values())
