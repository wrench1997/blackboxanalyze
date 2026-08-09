from __future__ import annotations

import json
from pathlib import Path

from scripts.build_pg348_dynamic_context_dataset import build


ROOT = Path(__file__).resolve().parents[1]


def test_pg348_dynamic_context_dataset_preserves_roles_and_dynamic_axes() -> None:
    dataset = json.loads((ROOT / "research" / "pg348_context_only_dataset_v1.json").read_text(encoding="utf-8"))
    trace = json.loads((ROOT / "research" / "pg348_dynamic_shape_trace_v1.json").read_text(encoding="utf-8"))
    output, vocab, audit = build(dataset, trace)
    assert output["counts"] == {"rows": 2080, "train_rows": 960, "implementation_holdout_rows": 1120, "training_eligible_rows": 0}
    assert len({row["dynamic_role"] for row in output["records"]}) == 4
    assert audit["counts"]["axis_token_sequence_entropy"]["response_transport"]["unique_sequences"] >= 2
    assert audit["counts"]["axis_token_sequence_entropy"]["belief_and_replay"]["unique_sequences"] >= 2
    assert vocab["forbidden_tokens"] == []
    assert all(row["training_eligible"] is False for row in output["records"])
