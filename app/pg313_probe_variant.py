"""PG-313 abstract probe-variant and encoding-chain Rule-IR slots.

The neural decoder predicts bounded references only.  A local, source-attested
adapter may later bind ``source_attested_candidate``/``reference_canary``/
``negative_control`` to human-review catalog entries.  Literal payloads and
response bodies never enter the model-visible sequence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pg293_failure_next_action import TARGET_BOS, TARGET_EOS
from .pg301_payload_assembly import TARGET_KEYS, target_map
from .pg302_symbolic_assembly import bind_symbolic_plan, symbolic_target_for_context

SCHEMA_VERSION = "pg313-probe-variant-assembly-v1"
PROBE_VARIANT_REFS = frozenset({"none", "source_attested_candidate", "reference_canary", "negative_control"})
ENCODING_CHAIN_REFS = frozenset({"none", "surface_encoding"})
EXTRA_KEYS = ("probe_variant_ref", "encoding_chain_ref")


def _context_map(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        token = str(token)
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def _variant_for_context(values: Mapping[str, str], safe: bool) -> str:
    if not safe:
        return "none"
    action = str(values.get("history_action", "none"))
    if action in {"reference_request", "reference_probe"}:
        return "reference_canary"
    if action in {"negative_control", "negative_probe"}:
        return "negative_control"
    return "source_attested_candidate"


def probe_target_for_context(context_tokens: Sequence[str]) -> list[str]:
    """Extend PG-302's target with two bounded probe-selection references."""

    base = symbolic_target_for_context(context_tokens)
    values = target_map(base)
    safe = values.get("safe_to_send") == "1"
    context = _context_map(context_tokens)
    variant = _variant_for_context(context, safe)
    encoding = "surface_encoding" if safe else "none"
    return [
        *base[:-1],
        f"probe_variant_ref={variant}",
        f"encoding_chain_ref={encoding}",
        TARGET_EOS,
    ]


def _base_tokens(symbolic_tokens: Sequence[str]) -> list[str] | None:
    tokens = [str(token) for token in symbolic_tokens]
    expected = len(TARGET_KEYS) + len(EXTRA_KEYS) + 2
    if len(tokens) != expected or tokens[0] != TARGET_BOS or tokens[-1] != TARGET_EOS:
        return None
    return [tokens[0], *tokens[1 : 1 + len(TARGET_KEYS)], TARGET_EOS]


def bind_probe_variant_plan(symbolic_tokens: Sequence[str], context_tokens: Sequence[str]) -> list[str] | None:
    """Bind base slots plus probe references; all invalid choices fail closed."""

    tokens = [str(token) for token in symbolic_tokens]
    base = _base_tokens(tokens)
    if base is None:
        return None
    values = target_map(tokens)
    if any(key not in values for key in EXTRA_KEYS):
        return None
    variant_ref = values["probe_variant_ref"]
    encoding_ref = values["encoding_chain_ref"]
    if variant_ref not in PROBE_VARIANT_REFS or encoding_ref not in ENCODING_CHAIN_REFS:
        return None
    bound_base = bind_symbolic_plan(base, context_tokens)
    if bound_base is None:
        return None
    bound_values = target_map(bound_base)
    context = _context_map(context_tokens)
    variant = variant_ref
    encoding = "none" if encoding_ref == "none" else context.get("surface_encoding", "unknown")
    safe = bound_values.get("safe_to_send") == "1"
    if not safe:
        variant = "none"
        encoding = "none"
    elif variant_ref == "none" or encoding not in {"url_percent", "form_urlencoded", "json_string", "base64_marker", "identity"}:
        bound_values["safe_to_send"] = "0"
        variant = "none"
        encoding = "none"
    output_base = [TARGET_BOS, *[f"{key}={bound_values.get(key, 'unknown')}" for key in TARGET_KEYS]]
    return [*output_base, f"probe_variant={variant}", f"encoding_chain={encoding}", TARGET_EOS]


def audit_probe_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bad_shape: list[int] = []
    bad_refs: list[int] = []
    forbidden: list[int] = []
    for index, row in enumerate(records):
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        if _base_tokens(target) is None:
            bad_shape.append(index)
        values = target_map(target)
        if values.get("probe_variant_ref") not in PROBE_VARIANT_REFS or values.get("encoding_chain_ref") not in ENCODING_CHAIN_REFS:
            bad_refs.append(index)
        if any(key in {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss"} for key in {str(token).split("=", 1)[0] for token in context + target if "=" in str(token)}):
            forbidden.append(index)
    splits = {str(row.get("split")) for row in records}
    checks = {
        "records_present": bool(records),
        "target_shape": not bad_shape,
        "reference_values_bounded": not bad_refs,
        "forbidden_fields_absent": not forbidden,
        "train_present": "train" in splits,
        "implementation_holdout_present": "implementation_holdout" in splits,
        "hard_negative_present": "hard_negative_eval" in splits,
        "payload_strings_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in records),
    }
    return {"schema_version": f"{SCHEMA_VERSION}-audit", "checks": checks, "bad_shape_indices": bad_shape, "bad_reference_indices": bad_refs, "forbidden_indices": forbidden, "status": "passed" if all(checks.values()) else "failed"}


__all__ = ["ENCODING_CHAIN_REFS", "EXTRA_KEYS", "PROBE_VARIANT_REFS", "SCHEMA_VERSION", "audit_probe_records", "bind_probe_variant_plan", "probe_target_for_context"]
