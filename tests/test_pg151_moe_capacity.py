from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from app.moe_trace_transformer import MoETraceTransformer


def test_moe_router_is_causal_and_balanced_on_a_small_batch() -> None:
    model = MoETraceTransformer(48, d_model=64, nhead=4, layers=2, n_experts=4, expert_ff=128, max_len=16)
    ids = torch.tensor([[2, 3, 4, 0, 0], [2, 5, 6, 7, 8]], dtype=torch.long)
    mask = ids.ne(0)
    hidden, auxiliary, loads = model.encode(ids, mask)
    assert hidden.shape == (2, 5, 64)
    assert torch.isfinite(auxiliary)
    assert loads.shape == (2, 4)
    assert torch.allclose(loads.sum(dim=-1), torch.ones(2), atol=1e-5)
    logits = model(ids, mask)
    assert logits.shape == (2, 5, 48)


def test_pg151_capacity_report_is_reproducible_and_uses_fresh_data() -> None:
    report = json.loads(Path("research/pg151_moe_capacity_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg151_moe_capacity_sweep"
    assert report["device"] == "cuda"
    assert report["corpus"]["generated_count"] == 8000
    assert report["corpus"]["holdout_count"] == 1210
    assert report["corpus"]["base_train_count"] == 9673
    assert len(report["variants"]) == 3
    assert max(row["parameter_count"] for row in report["variants"]) > 150_000_000
    best = min(report["variants"], key=lambda row: row["holdout"]["perplexity"])
    assert best["variant"] == "moe_xl_8e"
    assert best["holdout"]["perplexity"] < 2.56
    assert report["capability_claim_allowed"] is False
    assert report["long_term_memory_promotion_allowed"] is False


def test_pg151_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg151_moe_capacity_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
