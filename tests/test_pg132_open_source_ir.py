from __future__ import annotations

import json
from pathlib import Path


def test_pg132_report_records_open_source_embedding_without_false_pretrained_claim() -> None:
    report = json.loads(Path("research/pg132_open_source_ir_report_v1.json").read_text(encoding="utf-8"))
    assert report["embedding_provenance"]["tokenizer_backend"] == "huggingface-tokenizers-wordlevel"
    assert report["embedding_provenance"]["pretrained"] is False
    assert report["embedding_provenance"]["weights_source"] == "fresh_seeded_torch_embedding"
    assert report["scope"]["real_vulnerability_scanner_claim_allowed"] is False
    assert report["holdout"]["pg127_seed_holdout"]["metrics"]["accuracy"] == 1.0
    assert report["holdout"]["pg125_family_ood"]["metrics"]["accuracy"] == 1.0
    assert report["holdout"]["pg122_family_ood"]["metrics"]["accuracy"] == 1.0
    assert report["holdout"]["pg127_seed_holdout"]["blind_final_abstain_rate"] == 1.0
    assert report["holdout"]["pg125_family_ood"]["blind_final_abstain_rate"] == 1.0
    assert report["holdout"]["pg122_family_ood"]["blind_final_abstain_rate"] == 1.0
    assert report["checks"]["uniform_token_ablation_changes"] is False
    assert report["checks"]["failure_scalar_ablation_changes"] is False
    assert report["checks"]["token_embedding_ablation_changes"] is True
    assert report["holdout"]["token_ids_zeroed_ablation"]["metrics"]["accuracy"] == 0.416667
    assert report["checks"]["history_order_counterfactual_changes"] is False
    assert report["holdout"]["history_order_counterfactual"]["metrics"]["accuracy"] == 1.0
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
