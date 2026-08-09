from __future__ import annotations

import json
from pathlib import Path

import torch

from app.pg285_payload_grounding import PayloadGroundingDecoder, build_vocabs, encode_rows, greedy_decode, render_wire_plan


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg285_dataset_audit_and_hard_negative_contract():
    data = _load("pg285_payload_grounding_dataset_v1.json")
    audit = _load("pg285_payload_grounding_dataset_audit_v1.json")
    hard = _load("pg285_payload_grounding_hard_negative_v1.json")
    assert audit["status"] == "passed"
    assert data["counts"] == {"train": 5418, "route_dev": 504, "family_holdout": 630, "hard_negative": 1512, "total": 8064}
    assert data["training_contract"]["family_hidden_in_context"] is True
    assert data["training_contract"]["literal_probe_values_out_of_context"] is True
    assert data["training_contract"]["hard_negative_training_eligible"] is False
    assert hard["training_eligible"] is False
    assert all(row["target"]["next_action"] == "abstain" and row["target"]["safe_to_send"] is False for row in hard["records"][:100])
    assert all("family_hidden=1" in row["context_tokens"] and "oracle_label_in_context=0" in row["context_tokens"] for row in data["records"][:100])


def test_pg285_decoder_emits_bounded_plan_without_literal_payload():
    data = _load("pg285_payload_grounding_dataset_v1.json")
    train = [row for row in data["records"] if row["split"] == "train"]
    rows = [row for row in data["records"] if row["split"] == "route_dev" and row["state"] == "reference_aligned"][:2]
    context_vocab, target_vocab = build_vocabs(train)
    model = PayloadGroundingDecoder(len(context_vocab), len(target_vocab), embed_dim=24, hidden_dim=32)
    contexts, lengths, _, _ = encode_rows(rows, context_vocab, target_vocab)
    with torch.inference_mode():
        tokens = greedy_decode(model, contexts, lengths, target_vocab, max_tokens=24)
    assert len(tokens) == 2
    rendered = render_wire_plan(tokens[0], path="/observed")
    assert "<RUNTIME_CANARY>" in rendered or rendered.startswith("ABSTAIN")
    lowered = rendered.casefold()
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "union select" not in lowered


def test_pg285_remote_report_selects_guarded_variant_but_blocks_scientific_claim():
    report = _load("pg285_payload_grounding_report_v1.json")
    audit = _load("pg285_payload_grounding_audit_v1.json")
    assert report["status"] == "completed_remote_pg285_payload_grounding"
    assert report["device"]["cuda_visible_devices"] == "0"
    assert report["device"]["visible_device_count"] == 1
    assert "A800" in report["device"]["device_name"]
    assert report["risk_weight_sweep"]["selected_variant"] == "guarded_sft"
    assert report["risk_weight_sweep"]["variants"]["guarded_sft"]["hard_negative_false_allow_max"] == 0
    assert report["risk_weight_sweep"]["variants"]["plain_sft"]["hard_negative_false_allow_max"] > 0
    assert report["scientific_gate"]["status"] == "blocked"
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert audit["status"] == "passed"

