from pathlib import Path

import pytest

from scripts.plan_pg388_logic_composed_candidate import build_plan


def test_pg388_composed_plan_is_structured_but_blocked_before_optimizer() -> None:
    report = build_plan(dataset_path=Path("research/pg388_logic_canary_trajectory_dataset_v1.json"))
    assert report["status"] == "blocked_capability_contract"
    assert report["counts"] == {"records": 840, "train": 420, "implementation_holdout": 420, "slots": 11}
    assert report["model_design"]["slot_decoder"] == "autoregressive_causal_previous_slot_conditioned"
    assert report["gate"]["optimizer_started"] is False
    assert report["training_eligible"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert "typed_evaluator_not_attested" in report["gate"]["failures"]


def test_pg388_composed_plan_rejects_raw_context_marker(tmp_path: Path) -> None:
    import json

    data = json.loads(Path("research/pg388_logic_canary_trajectory_dataset_v1.json").read_text(encoding="utf-8"))
    data["rows"][0]["context_tokens"].append("payload=forbidden")
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="firewall"):
        build_plan(dataset_path=source)
