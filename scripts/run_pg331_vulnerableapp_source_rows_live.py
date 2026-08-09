"""Blocked-by-default PG-331 whole-page collector contract for PG-246.

The historical PG-246 reset script provides a pinned, loopback-only,
fresh-container lifecycle, but it uses a dedicated bridge network to publish a
loopback port.  PG-331's stricter collector contract requires network-none.
This module deliberately contains no Docker/HTTP/Playwright imports and will
not start that legacy lifecycle.  It preserves the exact blocker in a static
plan and provides a pure adapter bridge for a future reviewed network-none
relay implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_vulnerableapp_adapter import capture_vulnerableapp_projection  # noqa: E402
from app.pg331_network_none_relay import RELAY_BIND_HOST, role_container_name  # noqa: E402
from scripts.plan_pg331_vulnerableapp_source_rows import ROLES, SEEDS, _BY_ID, _route_attestation  # noqa: E402


SCHEMA_VERSION = "pg331-vulnerableapp-whole-page-live-contract-v1"
IMAGE_DIGEST = "7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"
RESET_SCRIPT = ROOT / "scripts" / "reset_pg25d_vulnerableapp.ps1"
SOURCE_ROLES = ("candidate", "reference", "negative")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(seed: int, case_id: str, role: str) -> str:
    return hashlib.sha256(json.dumps({"schema": SCHEMA_VERSION, "seed": int(seed), "case_ref": _route_attestation(_BY_ID[case_id]), "role": role, "image": IMAGE_DIGEST, "fresh_reset": True}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_live_contract(*, seeds: Sequence[int] = SEEDS, case_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Describe, but never execute, the required strict collection episodes."""
    selected = tuple(str(value) for value in (case_ids if case_ids is not None else _BY_ID))
    if not selected or set(selected) != set(_BY_ID) or len(selected) != len(_BY_ID):
        raise ValueError("PG-331 VulnerableApp contract requires all six fixed PG-246 cases")
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("PG-331 VulnerableApp contract requires a seed")
    episodes = []
    for seed in normalized:
        for case_id in selected:
            route = _BY_ID[case_id]
            method = str(route["method"])
            episodes.append({
                "seed": seed,
                "case_id": case_id,
                "case_ref_sha256": _route_attestation(route),
                "method": method,
                "roles": {role: {"fresh_reset_required": True, "fresh_reset_observed": False, "fresh_reset_evidence_sha256_required": True, "target_identity_sha256": _identity(seed, case_id, role), "container_name": role_container_name(seed=seed, case_ref_sha256=_route_attestation(route), role=role), "source_row_allowed": role in SOURCE_ROLES} for role in ROLES},
                "relay_contract": {"network_none_attestation_required": True, "host_bind": RELAY_BIND_HOST, "published_port_allowed": False, "bind_or_volume_allowed": False, "legacy_bridge_reclassification_allowed": False, "role_container_names": {role: role_container_name(seed=seed, case_ref_sha256=_route_attestation(route), role=role) for role in ROLES}},
                "typed_contract": {"candidate_reference_negative_required": True, "replay_required": True, "replay_is_evaluator_only": True, "role_bound_evidence_sha256_required": True, "typed_post_available": False if method == "POST" else "unknown_until_evaluator"},
                "model_projection": {"next_action": "ask_typed", "safe_to_send": False},
                "source_row_status": "incomplete_ask",
                "training_eligible": False,
            })
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_network_contract",
        "execution": {"planning_only": True, "docker_started": False, "network_contacted": False, "image_digest": IMAGE_DIGEST, "required_network_mode": "none", "required_loopback_only": True, "required_external_network": False, "legacy_reset_network_mode": "hostonly_bridge_loopback_publish", "legacy_reset_compatible": False},
        "blockers": ["legacy_pg246_reset_uses_bridge_not_network_none", "network_none_relay_lifecycle_not_yet_reviewed", "typed_post_unavailable_requires_ask"],
        "reset_script_sha256": _sha256_file(RESET_SCRIPT),
        "episodes": episodes,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    plan["contract_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return plan


def prepare_role_capture(*, method: str, html: str | None, headers: Mapping[str, Any] | None, request_projection: Mapping[str, Any] | None, response_projection: Mapping[str, Any] | None, post_supported: bool = False) -> dict[str, Any]:
    """Pure bridge to the strict adapter; never emits a trainable source row."""
    projection = capture_vulnerableapp_projection(html=html, headers=headers, request_projection=request_projection, response_projection=response_projection, post_supported=post_supported)
    method = str(method).upper()
    if method not in {"GET", "POST"}:
        raise ValueError("method must be GET or POST")
    if request_projection is not None and str(request_projection.get("method", "")).upper() != method:
        raise ValueError("method disagrees with request projection")
    typed_available = bool(projection["typed_projection"]["typed_available"]) if method == "GET" else False
    return {"observation": projection["observation"], "field_capture_manifest": projection["field_capture_manifest"], "typed_projection": {"typed_available": typed_available, "next_action": "ask_typed", "safe_to_send": False}, "raw_payload_stored": False, "raw_response_body_stored": False, "training_eligible": False}


def run() -> None:
    raise RuntimeError("PG-331 VulnerableApp live collection is blocked: the existing PG-246 lifecycle is bridge-networked, not network-none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print PG-331 VulnerableApp blocked live-collection contract")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_live_contract(), ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
