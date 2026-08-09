from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.pg137_transfer_strategies import STRATEGIES, strategy_manifest


def test_pg137_strategy_manifest_is_explicit() -> None:
    assert STRATEGIES == ("scratch", "frozen_body", "low_lr_full", "joint_lm_action")
    manifest = strategy_manifest()
    assert {item["name"] for item in manifest} == set(STRATEGIES)
    assert next(item for item in manifest if item["name"] == "frozen_body")["freeze_causal_body"] is True
    assert next(item for item in manifest if item["name"] == "joint_lm_action")["lm_loss_weight"] > 0.0


def test_pg137_matrix_preserves_negative_transfer_and_seed_failure() -> None:
    report = json.loads(Path("research/pg137_transfer_strategies_report_v1.json").read_text(encoding="utf-8"))
    seed = json.loads(Path("research/pg137_seed_stability_report_v1.json").read_text(encoding="utf-8"))
    assert report["hard_gates_passed"] is False
    assert report["training_eligible"] is False
    assert report["selection"]["selected_strategy"] == "scratch"
    assert report["selection"]["action_gain_two_ood_sets"] is False
    assert report["transport_balance"] == {"get_count": 132, "post_count": 132, "exact": True}
    assert seed["cross_seed_stable"] is False
    assert seed["stability"]["scratch"]["pg135_safety_min"] == 0.90625
    assert seed["stability"]["frozen_body"]["pg122_guarded_safety_min"] == 0.875
    assert seed["stability"]["low_lr_full"]["unknown_min"] < 1.0
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg137_report_hashes_are_recomputable() -> None:
    for name in ("pg137_transfer_strategies_report_v1.json", "pg137_seed_stability_report_v1.json"):
        report = json.loads(Path("research", name).read_text(encoding="utf-8"))
        declared = report.pop("report_sha256")
        actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert declared == actual
