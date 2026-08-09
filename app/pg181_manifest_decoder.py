"""PG-181 safe action-manifest decoder.

This head decides which *bounded abstract probe role* should be sent next.
The controller supplies the actual observed method/field names from the browser
manifest and the safe canary reference; the model never invents either value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .pg180_process_action_model import (
    BOS_TOKEN,
    MODEL_VARIANTS,
    abstract_step_tokens,
    build_model,
    build_vocabulary,
    collate,
    encode_examples,
    last_logits,
    parameter_count,
)


SCHEMA_VERSION = "sift-pg181-manifest-decoder-v1"
MANIFEST_ACTION_TOKENS = (
    "manifest::baseline",
    "manifest::matched_control",
    "manifest::safe_candidate",
    "manifest::abstain",
)
MANIFEST_ACTIONS = tuple(token.split("::", 1)[1] for token in MANIFEST_ACTION_TOKENS)


def pre_action_tokens(previous_step: Mapping[str, Any] | None, history: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """Project the state visible immediately before the next request."""

    if previous_step is None:
        return [BOS_TOKEN, "phase::initial", "response_state::none", "history_len::0"]
    tokens = [BOS_TOKEN, "phase::followup"]
    tokens.extend(f"history::{token}" for token in abstract_step_tokens(previous_step, history=history))
    return tokens


def manifest_target(step: Mapping[str, Any]) -> str:
    """Derive the training-only role label from a recorded action step."""

    step_id = str(step.get("step_id", ""))
    if "baseline" in step_id:
        return "manifest::baseline"
    if "candidate" in step_id:
        return "manifest::safe_candidate"
    return "manifest::matched_control"


def build_manifest_examples(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build pre-action rows without putting target/role into the input."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_step in trace.get("steps", []):
        grouped.setdefault(str(raw_step["episode_id"]), []).append(dict(raw_step))
    rows: list[dict[str, Any]] = []
    for episode_id, steps in grouped.items():
        history: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            previous = history[-1] if history else None
            context = pre_action_tokens(previous, history=history[:-1])
            rows.append({
                "episode_id_hash": __import__("hashlib").sha256(episode_id.encode("utf-8")).hexdigest(),
                "surface": episode_id.removeprefix("pg179b-pikachu-"),
                "context": context,
                "target": manifest_target(step),
                "single_channel": not bool((next((item for item in trace.get("episodes", []) if item.get("episode_id") == episode_id), {}) or {}).get("method_contract", {}).get("dual_channel")),
                "step_index": index,
            })
            history.append(step)
    if len(rows) != 35:
        raise ValueError(f"PG-181 expected 35 pre-action rows, got {len(rows)}")
    return rows


def manifest_vocabulary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    examples = [(list(row["context"]), str(row["target"])) for row in rows]
    vocabulary = build_vocabulary(examples)
    for token in MANIFEST_ACTION_TOKENS:
        vocabulary.setdefault(token, len(vocabulary))
    return vocabulary


def manifest_encode(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int]) -> list[dict[str, Any]]:
    return encode_examples([(list(row["context"]), str(row["target"])) for row in rows], vocabulary)


def restrict_manifest_action(logits: torch.Tensor, vocabulary: Mapping[str, int], *, single_channel: bool = False) -> tuple[str, float]:
    ids = [int(vocabulary[token]) for token in MANIFEST_ACTION_TOKENS]
    probabilities = torch.softmax(logits[ids], dim=0)
    confidence, index = torch.max(probabilities, dim=0)
    action = MANIFEST_ACTION_TOKENS[int(index)].split("::", 1)[1]
    # A single-channel route may still emit a safe candidate on that channel;
    # only the controller decides whether a method/field is actually observed.
    return action, float(confidence.detach().cpu())


__all__ = [
    "MANIFEST_ACTIONS",
    "MANIFEST_ACTION_TOKENS",
    "MODEL_VARIANTS",
    "SCHEMA_VERSION",
    "build_manifest_examples",
    "build_model",
    "collate",
    "last_logits",
    "manifest_encode",
    "manifest_target",
    "manifest_vocabulary",
    "parameter_count",
    "pre_action_tokens",
    "restrict_manifest_action",
]
