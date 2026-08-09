from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg161_reveals_seed_and_source_instability() -> None:
    report = _load("pg161_adapter_seed_source_report_v1.json")
    assert report["status"] == "completed_pg161_adapter_seed_source"
    assert report["device"] == "cuda"
    assert report["data_policy"]["source_heldout_labels_in_training"] is False
    assert report["data_policy"]["unseen_authorization_family_training_used"] is False
    assert report["promotion"]["capability_claim_allowed"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"seed_16101", "seed_16102", "seed_16103", "source_heldout_pg149", "source_heldout_pg136"}
    assert [variants[name]["synthetic_holdout"]["false_stop_count"] for name in ("seed_16101", "seed_16102", "seed_16103")] == [0, 33, 106]
    assert variants["source_heldout_pg149"]["synthetic_holdout"]["false_stop_count"] == 108
    assert variants["source_heldout_pg136"]["real_pg136_holdout"]["false_stop_count"] == 0
    assert all(row["surface_lm"]["perplexity"] == 8.85841751 for row in variants.values())
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())


def test_pg161_report_hash_is_recomputable() -> None:
    report = _load("pg161_adapter_seed_source_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg161_protocol_requires_seed_and_source_gates() -> None:
    protocol = _load("pg161_adapter_seed_source_protocol_v1.json")
    assert protocol["training"]["body_frozen"] is True
    assert "both action train and balanced replay" in protocol["source_heldout"]
    assert protocol["promotion"]["long_term_memory_promotion_allowed"] is False
