"""Typed, evaluator-only oracle for the Clickjacking response-header family."""

from __future__ import annotations

from typing import Any


FRAME_POLICY_PROTECTED = frozenset({"sameorigin", "deny", "ancestors_none"})
FRAME_POLICY_UNPROTECTED = frozenset({"none", "allowall"})


def classify_frame_policy(headers: dict[str, str]) -> str:
    """Map raw adapter headers to a bounded enum; raw values never leave the adapter."""

    x_frame = str(headers.get("x-frame-options", "")).strip().casefold()
    csp = str(headers.get("content-security-policy", "")).strip().casefold()
    if "frame-ancestors 'none'" in csp or "frame-ancestors \"none\"" in csp:
        return "ancestors_none"
    if x_frame == "deny":
        return "deny"
    if x_frame == "sameorigin":
        return "sameorigin"
    if x_frame == "allowall":
        return "allowall"
    if not x_frame and "frame-ancestors" not in csp:
        return "none"
    return "other"


def build_clickjacking_oracle(
    *,
    oracle_contract_sha256: str,
    frame_policy: str,
    expected_vulnerable: bool,
    regex_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a typed evaluator projection from a hidden expected outcome.

    ``expected_vulnerable`` is evaluator-only ground truth.  It must not be
    included in the model-visible trace; the returned projection is for the
    oracle/evidence side of the Catalog only.
    """

    if frame_policy not in {"none", "allowall", "sameorigin", "deny", "ancestors_none", "other", "unknown"}:
        raise ValueError("unknown bounded frame policy")
    observed_unprotected = frame_policy in FRAME_POLICY_UNPROTECTED
    positive = bool(expected_vulnerable and observed_unprotected)
    signals: dict[str, Any] = {
        "frame_policy": frame_policy,
        "protected_policy_observed": frame_policy in FRAME_POLICY_PROTECTED,
    }
    if regex_evidence is not None:
        signals["regex_evidence"] = dict(regex_evidence)
    return {
        "oracle_id": "pg25d-clickjacking-header-v1",
        "oracle_contract_sha256": oracle_contract_sha256,
        "family": "clickjacking",
        "modality": "header_policy",
        "candidate_signal": observed_unprotected,
        "positive": positive,
        "positive_authority": True,
        "confirmed_effect": "frame_protection" if positive else "none",
        "signals": signals,
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


__all__ = ["build_clickjacking_oracle", "classify_frame_policy"]
