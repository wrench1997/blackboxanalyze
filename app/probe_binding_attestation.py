"""Bounded attestation for the safe active-probe bank.

The attestation contains no runtime probe values.  It commits only to the
allow-listed probe IDs and their schema, so a decoder can distinguish a
valid adapter binding from an out-of-contract slot permutation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .active_probe_signature import PROBE_IDS, SCHEMA_VERSION


BINDING_SCHEMA_VERSION = "safe-active-probe-binding-v1"
_BINDING_PAYLOAD = {
    "schema_version": BINDING_SCHEMA_VERSION,
    "signature_schema_version": SCHEMA_VERSION,
    "probe_order": list(PROBE_IDS),
    "adapter_contract": "loopback_safe_probe_bank_only",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


CANONICAL_BINDING_SHA256 = hashlib.sha256(_canonical(_BINDING_PAYLOAD).encode("utf-8")).hexdigest()


def add_binding_attestation(signature: Mapping[str, Any], *, binding_sha256: str = CANONICAL_BINDING_SHA256) -> dict[str, Any]:
    """Attach a bounded probe-bank commitment to a signature copy."""

    value = json.loads(json.dumps(signature, ensure_ascii=False))
    value["probe_binding"] = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "signature_schema_version": SCHEMA_VERSION,
        "probe_order": list(PROBE_IDS),
        "binding_sha256": str(binding_sha256),
    }
    return value


def binding_attestation_valid(value: Mapping[str, Any], *, expected_sha256: str = CANONICAL_BINDING_SHA256) -> bool:
    binding = value.get("probe_binding")
    if not isinstance(binding, Mapping):
        return False
    return (
        str(binding.get("schema_version", "")) == BINDING_SCHEMA_VERSION
        and str(binding.get("signature_schema_version", "")) == SCHEMA_VERSION
        and list(binding.get("probe_order") or []) == list(PROBE_IDS)
        and str(binding.get("binding_sha256", "")) == str(expected_sha256)
    )


def binding_digest_for_order(order: list[str]) -> str:
    """Return a deterministic non-canonical digest for permutation ablations."""

    payload = dict(_BINDING_PAYLOAD)
    payload["probe_order"] = list(order)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


__all__ = [
    "BINDING_SCHEMA_VERSION",
    "CANONICAL_BINDING_SHA256",
    "add_binding_attestation",
    "binding_attestation_valid",
    "binding_digest_for_order",
]

