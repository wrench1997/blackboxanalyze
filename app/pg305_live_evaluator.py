"""PG-305 helpers for real loopback payload-plan evaluation.

PG-305 is the first bridge from the abstract causal Transformer/MoE composer
to a disposable, authorized local target.  The model still predicts only
Rule-IR slots.  A target adapter may bind a source-attested candidate after
the guarded plan passes; literal values and wire strings stay in a separate
human-review artifact and never enter the model context or training rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .pg284_evaluator_contract import evaluate_typed_replay, sha256_json
from .pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, generate_target
from .pg301_payload_assembly import (
    TARGET_KEYS,
    assembly_target_for_context,
    canonical_assembly_context,
    target_map,
)
from .pg302_symbolic_assembly import bind_symbolic_plan
from .pg303_guarded_composer import compose_guarded_plan


SCHEMA_VERSION = "pg305-live-loopback-evaluator-v1"
OBSERVATION_KEYS = (
    "typed_available",
    "feedback_state",
    "replay_ready",
    "evidence_present",
    "negative_control",
    "fresh_reset",
)
SURFACE_KEYS = ("surface_method", "surface_field_role", "surface_encoding")
MISSING_ORDER = (
    "typed_available",
    "replay_ready",
    "evidence_present",
    "feedback_state",
    "negative_control",
    "fresh_reset",
)
_CONTEXT_FORBIDDEN_MARKERS = (
    "family=",
    "route=",
    "oracle=",
    "evaluator",
    "payload",
    "response",
    "raw_body",
    "source_code",
    "sql",
    "xss",
    "<script",
    "javascript:",
)


def _validate_context_value(key: str, value: Any) -> str:
    """Reject values that could smuggle evaluator metadata into model input."""

    text = str(value)
    folded = text.casefold()
    if not text or "=" in text or any(marker in folded for marker in _CONTEXT_FORBIDDEN_MARKERS) or any(char in text for char in "\r\n"):
        raise ValueError(f"forbidden model-context value for {key}")
    return text


def surface_slots(method: str) -> dict[str, str]:
    """Map a real request method to shared abstract surface slots only."""

    normalized = str(method).upper()
    if normalized == "GET":
        return {"surface_method": "GET", "surface_field_role": "query_param", "surface_encoding": "url_percent"}
    if normalized == "POST":
        return {"surface_method": "POST", "surface_field_role": "form_field", "surface_encoding": "form_urlencoded"}
    raise ValueError("PG-305 supports only GET and POST surfaces")


def context_tokens(
    method: str,
    *,
    typed_available: str = "unknown",
    feedback_state: str = "unknown",
    replay_ready: str = "unknown",
    evidence_present: str = "unknown",
    negative_control: str = "unknown",
    fresh_reset: str = "unknown",
    history_action: str = "none",
    failure_class: str = "none",
    step_budget: str = "present",
) -> list[str]:
    """Create a model-visible context without route, family, oracle or body."""

    slots = surface_slots(method)
    values = {
        "typed_available": typed_available,
        "feedback_state": feedback_state,
        "replay_ready": replay_ready,
        "evidence_present": evidence_present,
        "negative_control": negative_control,
        "fresh_reset": fresh_reset,
        "history_action": history_action,
        "failure_class": failure_class,
        "step_budget": step_budget,
        **slots,
    }
    values = {key: _validate_context_value(key, value) for key, value in values.items()}
    return canonical_assembly_context([f"{key}={value}" for key, value in values.items()])


def missing_question_contexts(method: str) -> list[dict[str, Any]]:
    """Return paired missing-slot contexts for the identifiability audit."""

    complete = {
        key: "1" for key in OBSERVATION_KEYS
    }
    complete.update({"feedback_state": "negative_control_clear", "history_action": "none", "failure_class": "none"})
    rows: list[dict[str, Any]] = []
    for missing in MISSING_ORDER:
        values = dict(complete)
        values[missing] = "unknown"
        rows.append(
            {
                "missing_slot": missing,
                "context_tokens": context_tokens(method, **values),
                "target_tokens": assembly_target_for_context(context_tokens(method, **values)),
            }
        )
    rows.append(
        {
            "missing_slot": "none",
            "context_tokens": context_tokens(method, **complete),
            "target_tokens": assembly_target_for_context(context_tokens(method, **complete)),
        }
    )
    return rows


def load_causal_checkpoint(path: str | Path, device: torch.device) -> tuple[CausalMoELanguageModel, dict[str, int], bool]:
    """Load a frozen PG-301 or PG-302B decoder on the requested device."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in dict(checkpoint["vocabulary"]).items()}
    config = CausalMoEConfig(**dict(checkpoint["config"]))
    model = CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval()
    symbolic = "transport_ref=surface_method" in vocabulary
    return model, vocabulary, symbolic


def propose_plan(
    model: CausalMoELanguageModel,
    vocabulary: Mapping[str, int],
    device: torch.device,
    tokens: Sequence[str],
    *,
    symbolic: bool,
) -> dict[str, Any]:
    """Generate raw next-token slots, bind symbolic refs, then apply the guard."""

    raw = generate_target(model, tokens, len(TARGET_KEYS) + 2, vocabulary, device)
    bound = bind_symbolic_plan(raw, tokens) if symbolic else list(raw)
    proposal = bound or []
    guarded = compose_guarded_plan(proposal, tokens)
    return {
        "raw_tokens": list(raw),
        "bound_tokens": list(bound or []),
        "guarded_tokens": list(guarded),
        "raw_fields": target_map(raw),
        "bound_fields": target_map(bound or []),
        "guarded_fields": target_map(guarded),
        "raw_safe_to_send": target_map(raw).get("safe_to_send") == "1",
        "guarded_safe_to_send": target_map(guarded).get("safe_to_send") == "1",
    }


def abstract_projection(value: Mapping[str, Any], *, effect_marker: str = "none", backend_observed: bool = True) -> dict[str, Any]:
    """Project a live response to the PG-284 bounded evaluator shape."""

    status_class = str(value.get("status_class", "unknown"))
    if status_class not in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"}:
        status_class = "unknown"
    shape = {
        "status_class": status_class,
        "content_type": str(value.get("content_type", "unknown")),
        "body_length_bucket": _length_bucket(value.get("body_length")),
        "location_class": "loopback" if str(value.get("location", "")).startswith("http://127.0.0.1:") else "none",
        "marker_reflected": bool(value.get("marker_reflected", False)),
        "dom_executed": bool(value.get("executed", False)),
        "body_shape": str(value.get("body_shape", "unknown")),
    }
    return {
        "status_class": status_class,
        "shape_sha256": sha256_json(shape),
        "redirect_hops": int(value.get("redirect_hops", 0) or 0),
        "backend_observed": bool(backend_observed),
        "effect_marker": str(effect_marker)[:80],
    }


def typed_evidence(
    *,
    effect_type: str,
    typed_effect_confirmed: bool,
    negative_control_clean: bool,
    reference_agreement: bool,
    replay_consistent: bool,
    evaluator_id: str,
) -> dict[str, Any]:
    """Build the signed evidence projection consumed by PG-284."""

    unsigned = {
        "effect_type": effect_type,
        "typed_effect_confirmed": bool(typed_effect_confirmed),
        "negative_control_clean": bool(negative_control_clean),
        "reference_agreement": bool(reference_agreement),
        "replay_consistent": bool(replay_consistent),
        "non_destructive": True,
        "evaluator_id": str(evaluator_id),
    }
    return {**unsigned, "evidence_sha256": sha256_json(unsigned)}


def evaluator_result(
    *,
    surface: Mapping[str, Any],
    reset: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replay: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_attestation: str,
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Run the independent fail-closed typed evaluator for one live episode."""

    normalized_surface = dict(surface)
    normalized_surface.setdefault("authorization", "operator_allowlisted_remote_docker")
    normalized_surface.setdefault("source_attestation_sha256", source_attestation)
    return evaluate_typed_replay(
        surface=normalized_surface,
        reset=reset,
        reference=reference,
        negative=negative,
        candidate=candidate,
        replay=replay,
        typed_evidence=evidence,
        remote_probe={"status": "available", "loopback_only": True, "external_network": False},
        hard_negative=hard_negative,
    )


def _length_bucket(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number < 256:
        return "0-255"
    if number < 4096:
        return "256-4095"
    if number < 65536:
        return "4096-65535"
    return "65536+"


def abstract_training_record(
    *,
    record_id: str,
    method: str,
    context: Sequence[str],
    target: Sequence[str],
    split: str,
    outcome_class: str,
    typed_effect_confirmed: bool,
    evidence_hash: str,
) -> dict[str, Any]:
    """Create a promotion-blocked abstract row from a live episode."""

    row = {
        "schema_version": f"{SCHEMA_VERSION}-record",
        "record_id": str(record_id),
        "method_token": str(method).upper(),
        "context_tokens": list(context),
        "target_tokens": list(target),
        "split": str(split),
        "outcome_class": str(outcome_class),
        "typed_effect_confirmed": bool(typed_effect_confirmed),
        "evidence_sha256": str(evidence_hash),
        "oracle_target_off_input": True,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
    }
    row["record_sha256"] = sha256_json(row)
    return row


__all__ = [
    "MISSING_ORDER",
    "OBSERVATION_KEYS",
    "SCHEMA_VERSION",
    "SURFACE_KEYS",
    "abstract_projection",
    "abstract_training_record",
    "context_tokens",
    "evaluator_result",
    "load_causal_checkpoint",
    "missing_question_contexts",
    "propose_plan",
    "sha256_json",
    "surface_slots",
    "typed_evidence",
]
