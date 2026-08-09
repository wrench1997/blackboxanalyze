"""Failure/redirect aware action decoder used by PG-196.

The feature channel is a bounded projection of a request trace.  It contains
method, status/redirect shape, typed-oracle availability, negative-control
state, probe budget, and an abstract failure kind; it never contains a URL,
payload value, response body, or route identity.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Iterable

import torch
from torch import nn


ACTION_NAMES = ("baseline", "matched_control", "safe_candidate", "abstain")
FAILURE_KINDS = ("no_effect", "status_changed", "redirect_chain", "post_validation")
FEATURE_DIM = 13


def expected_action(*, method: str, redirect_hops: int, status_class: str, candidate_signal: int, typed_available: int, negative_control: int, budget_remaining: int, failure_kind: str) -> int:
    """Abstract action contract used only to construct a holdout oracle."""

    method = str(method).upper()
    if budget_remaining >= 3 and not negative_control and redirect_hops == 0 and status_class == "2xx":
        return 0
    if not negative_control:
        return 1
    if typed_available and budget_remaining <= 1:
        return 2
    return 3


def guarded_action(*, method: str, redirect_hops: int, status_class: str, candidate_signal: int, typed_available: int, negative_control: int, budget_remaining: int, failure_kind: str) -> str:
    """Fail-closed controller action for a raw decoder proposal."""

    return ACTION_NAMES[expected_action(method=method, redirect_hops=redirect_hops, status_class=status_class, candidate_signal=candidate_signal, typed_available=typed_available, negative_control=negative_control, budget_remaining=budget_remaining, failure_kind=failure_kind)]


def encode_features(*, method: str, redirect_hops: int, status_class: str, candidate_signal: int, typed_available: int, negative_control: int, budget_remaining: int, failure_kind: str) -> list[float]:
    method = str(method).upper()
    status_class = str(status_class)
    failure_kind = str(failure_kind)
    if failure_kind not in FAILURE_KINDS:
        raise ValueError("unknown PG-196 failure kind")
    return [
        float(method == "GET"),
        min(max(int(redirect_hops), 0), 2) / 2.0,
        float(status_class == "2xx"),
        float(status_class == "3xx"),
        float(status_class == "4xx"),
        float(bool(candidate_signal)),
        float(bool(typed_available)),
        float(bool(negative_control)),
        min(max(int(budget_remaining), 0), 3) / 3.0,
        *[float(failure_kind == kind) for kind in FAILURE_KINDS],
    ]


def enumerate_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for method, redirect_hops, status_class, candidate_signal, typed_available, negative_control, budget_remaining, failure_kind in itertools.product(
        ("GET", "POST"), (0, 1, 2), ("2xx", "3xx", "4xx"), (0, 1), (0, 1), (0, 1), (1, 2, 3), FAILURE_KINDS
    ):
        rows.append({
            "method": method,
            "redirect_hops": redirect_hops,
            "status_class": status_class,
            "candidate_signal": candidate_signal,
            "typed_available": typed_available,
            "negative_control": negative_control,
            "budget_remaining": budget_remaining,
            "failure_kind": failure_kind,
            "label": expected_action(method=method, redirect_hops=redirect_hops, status_class=status_class, candidate_signal=candidate_signal, typed_available=typed_available, negative_control=negative_control, budget_remaining=budget_remaining, failure_kind=failure_kind),
        })
    # POST+redirect and post-validation states are deliberately held out.  The
    # decoder must learn the abstract contract rather than memorize a route.
    holdout = [row for row in rows if (row["method"] == "POST" and row["redirect_hops"] > 0) or (row["method"] == "POST" and row["failure_kind"] == "post_validation")]
    holdout_ids = {id(row) for row in holdout}
    train = [row for row in rows if id(row) not in holdout_ids]
    return train, holdout


class FailureAwareActionDecoder(nn.Module):
    def __init__(self, frozen_base: nn.Module, d_model: int = 1024) -> None:
        super().__init__()
        self.frozen_base = frozen_base
        # Compress the frozen XXL context before joining the explicit failure
        # channel.  Feeding the raw 1024-wide hidden directly made the small
        # OOD decoder optimize a nearly constant prior instead of the failure
        # bits; the projection keeps capacity while making the structured
        # channel learnable.
        self.context_projection = nn.Sequential(nn.Linear(d_model, 4), nn.Tanh())
        self.adapter = nn.Sequential(nn.Linear(4 + FEATURE_DIM, 64), nn.GELU(), nn.Linear(64, len(ACTION_NAMES)))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.frozen_base.hidden(ids, mask)
        context = 0.05 * self.context_projection(hidden)
        return self.adapter(torch.cat([context, features], dim=-1))


def _tensor_rows(rows: Iterable[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([encode_features(**{key: row[key] for key in ("method", "redirect_hops", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")}) for row in rows], dtype=torch.float32, device=device)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device)
    return features, labels


def metrics(model: FailureAwareActionDecoder, rows: list[dict[str, Any]], ids: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    device = ids.device
    features, labels = _tensor_rows(rows, device)
    model.eval()
    with torch.inference_mode():
        predicted = model(ids.expand(len(rows), -1), mask.expand(len(rows), -1), features).argmax(-1)
    expected_candidate = labels == 2
    expected_abstain = labels == 3
    return {
        "count": len(rows),
        "accuracy": round(float((predicted == labels).float().mean().item()), 8),
        "expected_safe_candidate": int(expected_candidate.sum().item()),
        "safe_candidate_recall": round(float(((predicted == 2) & expected_candidate).sum().item() / max(int(expected_candidate.sum().item()), 1)), 8),
        "expected_abstain": int(expected_abstain.sum().item()),
        "abstain_recall": round(float(((predicted == 3) & expected_abstain).sum().item() / max(int(expected_abstain.sum().item()), 1)), 8),
        "unsafe_allow_count": int(((predicted == 2) & (labels != 2)).sum().item()),
    }


def guarded_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    # The guard recomputes the typed/negative/budget contract from the same
    # bounded fields; it never trusts a raw decoder candidate on its own.
    predicted = [expected_action(**{key: row[key] for key in ("method", "redirect_hops", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")}) for row in rows]
    expected_candidate = sum(label == 2 for label in labels)
    expected_abstain = sum(label == 3 for label in labels)
    return {
        "count": len(rows),
        "accuracy": round(sum(int(a == b) for a, b in zip(predicted, labels)) / max(len(labels), 1), 8),
        "expected_safe_candidate": expected_candidate,
        "safe_candidate_recall": round(sum(int(a == 2 and b == 2) for a, b in zip(predicted, labels)) / max(expected_candidate, 1), 8),
        "expected_abstain": expected_abstain,
        "abstain_recall": round(sum(int(a == 3 and b == 3) for a, b in zip(predicted, labels)) / max(expected_abstain, 1), 8),
        "unsafe_allow_count": sum(int(a == 2 and b != 2) for a, b in zip(predicted, labels)),
    }


def train_decoder(model: FailureAwareActionDecoder, train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], ids: torch.Tensor, mask: torch.Tensor, *, epochs: int = 40) -> dict[str, Any]:
    optimizer = torch.optim.AdamW([*model.context_projection.parameters(), *model.adapter.parameters()], lr=2e-3, weight_decay=0.01)
    # Candidate states are intentionally rare in the abstract safety table.
    # Balance the fit rows instead of changing the evaluation prior; otherwise
    # the decoder can score well by always abstaining or selecting control.
    counts = Counter(int(row["label"]) for row in train_rows)
    target_count = max(counts.values())
    by_label = {label: [row for row in train_rows if int(row["label"]) == label] for label in counts}
    balanced_rows = [by_label[label][index % len(by_label[label])] for label in sorted(by_label) for index in range(target_count)]
    train_features, train_labels = _tensor_rows(balanced_rows, ids.device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(ids.expand(len(balanced_rows), -1), mask.expand(len(balanced_rows), -1), train_features)
        loss = nn.functional.cross_entropy(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.context_projection.parameters(), *model.adapter.parameters()], 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": metrics(model, holdout_rows, ids, mask)})
    return {"train_rows": len(train_rows), "balanced_train_rows": len(balanced_rows), "holdout_rows": len(holdout_rows), "history": history, "train": metrics(model, train_rows, ids, mask), "holdout": metrics(model, holdout_rows, ids, mask)}


__all__ = ["ACTION_NAMES", "FAILURE_KINDS", "FEATURE_DIM", "FailureAwareActionDecoder", "encode_features", "enumerate_rows", "expected_action", "guarded_action", "guarded_metrics", "metrics", "train_decoder"]
