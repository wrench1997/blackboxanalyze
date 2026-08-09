"""Canary-based catastrophic-forgetting checks for causal token models.

The checker deliberately evaluates only the bounded token sequence.  It does
not inspect labels, oracle authority, probes, responses, or target identity.
The same canary is scored before and after a downstream head is trained so a
drop in next-token behavior cannot be hidden by a better action score.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch import nn


DEFAULT_THRESHOLDS: dict[str, float] = {
    "max_relative_perplexity_increase": 0.20,
    "max_next_token_accuracy_drop": 0.05,
    "max_mean_logit_kl": 0.10,
}


def _padded_ids(examples: Iterable[Mapping[str, Any]], vocabulary: Any, *, device: torch.device) -> torch.Tensor:
    encoded = [vocabulary.encode(item["tokens"]) for item in examples]
    if not encoded:
        raise ValueError("catastrophic-forgetting canary is empty")
    width = max(len(sequence) for sequence in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for row_index, sequence in enumerate(encoded):
        ids[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return ids


def _score(model: nn.Module, examples: list[Mapping[str, Any]], vocabulary: Any, *, device: torch.device) -> tuple[dict[str, float], torch.Tensor]:
    ids = _padded_ids(examples, vocabulary, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:]
    input_mask = inputs.ne(0)
    model.eval()
    with torch.inference_mode():
        if hasattr(model, "next_token_logits"):
            logits = model.next_token_logits(inputs)
        else:
            # CausalTraceTransformer exposes the LM through ``forward`` and
            # accepts the bounded padding mask explicitly.
            logits = model(inputs, input_mask)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=0,
        )
    valid = targets.ne(0)
    token_count = int(valid.sum().item())
    predictions = logits.argmax(dim=-1)
    accuracy = float(((predictions == targets) & valid).sum().item() / max(token_count, 1))
    scalar_loss = float(loss.item())
    return {
        "loss": round(scalar_loss, 8),
        "perplexity": round(math.exp(min(scalar_loss, 20.0)), 8),
        "next_token_accuracy": round(accuracy, 8),
        "token_count": token_count,
    }, logits.detach()


def compare_causal_lm_canary(
    before_model: nn.Module,
    after_model: nn.Module,
    examples: Iterable[Mapping[str, Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare a frozen pre-training anchor with a downstream-tuned model."""

    canary = list(examples)
    if not canary:
        raise ValueError("catastrophic-forgetting canary is empty")
    active_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        active_thresholds.update({str(key): float(value) for key, value in thresholds.items()})
    before, before_logits = _score(before_model, canary, vocabulary, device=device)
    after, after_logits = _score(after_model, canary, vocabulary, device=device)
    valid = _padded_ids(canary, vocabulary, device=device)[:, 1:].ne(0)
    before_probs = torch.softmax(before_logits, dim=-1)
    after_log_probs = torch.log_softmax(after_logits, dim=-1)
    # KL(before || after) is a distribution-drift diagnostic, not a label.
    token_kl = torch.sum(before_probs * (torch.log(before_probs.clamp_min(1e-8)) - after_log_probs), dim=-1)
    mean_kl = float(token_kl[valid].mean().item()) if bool(valid.any()) else 0.0
    relative_ppl_increase = (after["perplexity"] - before["perplexity"]) / max(before["perplexity"], 1e-8)
    accuracy_drop = before["next_token_accuracy"] - after["next_token_accuracy"]
    exceeded = {
        "relative_perplexity_increase": relative_ppl_increase > active_thresholds["max_relative_perplexity_increase"],
        "next_token_accuracy_drop": accuracy_drop > active_thresholds["max_next_token_accuracy_drop"],
        "mean_logit_kl": mean_kl > active_thresholds["max_mean_logit_kl"],
    }
    return {
        "canary_count": len(canary),
        "before": before,
        "after": after,
        "delta": {
            "perplexity": round(after["perplexity"] - before["perplexity"], 8),
            "relative_perplexity_increase": round(relative_ppl_increase, 8),
            "next_token_accuracy_drop": round(accuracy_drop, 8),
            "mean_logit_kl": round(mean_kl, 8),
        },
        "thresholds": active_thresholds,
        "exceeded": exceeded,
        "catastrophic_forgetting_detected": bool(any(exceeded.values())),
        "canary_labels_or_oracle_used": False,
        "raw_request_response_used": False,
    }


__all__ = ["DEFAULT_THRESHOLDS", "compare_causal_lm_canary"]
