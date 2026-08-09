from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg159_keeps_family_labels_out_of_model_inputs_and_reports_loo() -> None:
    report = _load("pg159_family_invariant_adapter_report_v1.json")
    assert report["status"] == "completed_pg159_family_invariant_adapter"
    assert report["device"] == "cuda"
    assert report["data_policy"]["family_labels_as_model_inputs"] is False
    assert report["data_policy"]["family_labels_for_adversary_only"] is True
    assert report["data_policy"]["unseen_authorization_family_training_used"] is False
    assert report["promotion"]["capability_claim_allowed"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"shared_adapter_256", "loo_steady_adapter_256", "loo_blind_adapter_512", "loo_scope_adapter_512"}
    assert variants["loo_steady_adapter_256"]["held_out_family"] == "steady"
    assert variants["loo_blind_adapter_512"]["held_out_family"] == "blind"
    assert variants["loo_scope_adapter_512"]["held_out_family"] == "scope"
    assert variants["shared_adapter_256"]["surface_lm"]["perplexity"] < 10.0
    assert variants["shared_adapter_256"]["synthetic_holdout"]["false_stop_count"] > 0
    assert variants["loo_blind_adapter_512"]["unseen_authorization_family_holdout"]["accuracy"] == 0.75
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())


def test_pg159_report_hash_is_recomputable() -> None:
    report = _load("pg159_family_invariant_adapter_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg159_protocol_declares_adversarial_and_loo_gates() -> None:
    protocol = _load("pg159_family_invariant_adapter_protocol_v1.json")
    assert protocol["family_adversary"]["family_labels_are_not_model_inputs"] is True
    assert "held-out real family removed" in protocol["leave_one_family_out"]
    assert protocol["promotion"]["long_term_memory_promotion_allowed"] is False
