from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from app.causal_forgetting import compare_causal_lm_canary
from app.pg136_causal_token_lm import BOS_TOKEN, CausalTokenGRU, CausalVocabulary, EOS_TOKEN, canonical_tokens


def test_pg136_canonical_tokens_are_bounded_and_ordered() -> None:
    steps = [
        {
            "source_token_layers": [{"modality": "transport", "tokens": [{"kind": "route", "value": "/local", "value_hash": "x"}]}],
            "ir_layer": {"tokens": [{"slot_id": "failure.kind", "value": "no_surface_delta", "weight": 2.0}]},
        }
    ]
    tokens = canonical_tokens(steps)
    assert tokens[0] == BOS_TOKEN
    assert tokens[-1] == EOS_TOKEN
    assert "src.transport.route=hash_present" in tokens
    assert "ir.failure.kind=no_surface_delta" in tokens
    assert "ir_weight=2.0" in tokens
    assert "/local" not in " ".join(tokens)
    vocab = CausalVocabulary([tokens])
    assert vocab.encode(tokens)[0] == 1


def test_pg136_report_has_separate_lm_and_action_gates() -> None:
    report = json.loads(Path("research/pg136_causal_token_lm_report_v1.json").read_text(encoding="utf-8"))
    # The new canary intentionally caught forgetting after action tuning;
    # preserving the failure is the expected result of the safety gate.
    assert report["hard_gates_passed"] is False
    assert report["training_eligible"] is False
    assert report["pretraining"]["action_labels_in_input"] is False
    assert report["pretraining"]["dev"]["perplexity"] < 2.0
    assert report["action_finetune"]["pretrained"]["holdouts"]["pg135"]["accuracy"] >= 0.90
    assert report["guarded_ood"]["pg122"]["safety_compliance_rate"] >= 0.99
    assert report["ablations"]["causal_order_reverse_pg135"]["accuracy"] < report["action_finetune"]["pretrained"]["holdouts"]["pg135"]["accuracy"]
    assert report["action_finetune"]["pretrained"]["holdouts"]["pg135"]["accuracy"] < report["action_finetune"]["scratch"]["holdouts"]["pg135"]["accuracy"]
    assert report["stability"]["deterministic_algorithms"] is True
    assert report["promotion"]["memory_promotion_allowed"] is False
    forgetting = report["catastrophic_forgetting"]
    assert forgetting["canary_labels_or_oracle_used"] is False
    assert forgetting["raw_request_response_used"] is False
    assert forgetting["catastrophic_forgetting_detected"] is True
    assert forgetting["delta"]["relative_perplexity_increase"] > forgetting["thresholds"]["max_relative_perplexity_increase"]
    assert report["checks"]["catastrophic_forgetting_not_detected"] is False


def test_pg136_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg136_causal_token_lm_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg136_forgetting_canary_detects_lm_head_drift_without_labels() -> None:
    tokens = [[BOS_TOKEN, "ir.failure.kind=stable", EOS_TOKEN]]
    vocabulary = CausalVocabulary(tokens)
    before = CausalTokenGRU(len(vocabulary.itos), seed=13601)
    after = copy.deepcopy(before)
    with torch.no_grad():
        after.lm_head.bias[0] += 20.0
    report = compare_causal_lm_canary(before, after, [{"tokens": tokens[0]}], vocabulary, device=torch.device("cpu"), thresholds={
        "max_relative_perplexity_increase": 0.0,
        "max_next_token_accuracy_drop": 0.0,
        "max_mean_logit_kl": 0.0,
    })
    assert report["canary_labels_or_oracle_used"] is False
    assert report["raw_request_response_used"] is False
    assert report["catastrophic_forgetting_detected"] is True
