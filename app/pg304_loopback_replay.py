"""PG-304 evaluator-only loopback replay contract.

This adapter accepts an abstract guarded Rule-IR plan plus bounded evaluator
projections.  It never constructs a request, binds a canary, stores a body, or
contacts a target.  A confirmed typed effect is still an evaluation result,
not a vulnerability or training-memory promotion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pg284_evaluator_contract import evaluate_typed_replay, sha256_json
from .pg301_payload_assembly import canonical_assembly_context, target_map
from .pg303_guarded_composer import compose_guarded_plan


SCHEMA_VERSION = "pg304-loopback-replay-v1"
METHODS = frozenset({"GET", "POST"})


def _raw_key_present(value: Any, key: str = "") -> bool:
    if key.casefold() in {"payload", "raw_payload", "request_body", "response_body", "body_text", "html", "query_value", "form_value", "credential"}:
        return True
    if isinstance(value, Mapping):
        return any(_raw_key_present(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_raw_key_present(child, key) for child in value)
    return False


def _validate_plan(plan_tokens: Sequence[str], context_tokens: Sequence[str]) -> dict[str, Any]:
    compiled = compose_guarded_plan(plan_tokens, context_tokens)
    fields = target_map(compiled)
    if fields.get("safe_to_send") != "1":
        raise ValueError("PG-304 guarded plan is not safe for evaluator replay")
    if fields.get("transport") not in METHODS or fields.get("canary") != "runtime" or fields.get("oracle") != "typed":
        raise ValueError("PG-304 plan lacks bounded transport/canary/typed-oracle slots")
    return {"tokens": list(compiled), "fields": fields}


def evaluate_loopback_batch(episodes: Sequence[Mapping[str, Any]], *, require_get_post_pair: bool = True) -> dict[str, Any]:
    if not episodes:
        raise ValueError("PG-304 replay batch cannot be empty")
    if any(_raw_key_present(episode) for episode in episodes):
        raise ValueError("PG-304 replay batch contains raw request/response material")
    results: list[dict[str, Any]] = []
    methods: set[str] = set()
    for index, episode in enumerate(episodes):
        context = [str(token) for token in episode.get("context_tokens") or []]
        plan = _validate_plan(list(episode.get("plan_tokens") or []), context)
        surface = dict(episode.get("surface") or {})
        method = str(surface.get("method", "")).upper()
        methods.add(method)
        remote = dict(episode.get("remote_probe") or {})
        if remote.get("status") == "available" and (remote.get("loopback_only") is not True or remote.get("external_network") is not False):
            raise ValueError("PG-304 available remote probe must declare loopback_only and external_network=false")
        replay = evaluate_typed_replay(
            surface=surface,
            reset=dict(episode.get("reset") or {}),
            reference=dict(episode.get("reference") or {}),
            negative=dict(episode.get("negative") or {}),
            candidate=dict(episode.get("candidate") or {}),
            replay=dict(episode.get("replay") or {}),
            typed_evidence=dict(episode.get("typed_evidence") or {}),
            remote_probe=remote,
            hard_negative=bool(episode.get("hard_negative", False)),
        )
        results.append({
            "index": index,
            "surface_id": str(surface.get("surface_id", "")),
            "method": method,
            "plan_fields": plan["fields"],
            "typed_status": replay.get("status"),
            "typed_effect_confirmed": bool(replay.get("typed_effect_confirmed")),
            "checks": dict(replay.get("checks") or {}),
            "reasons": list(replay.get("reasons") or []),
            "evidence_projection_sha256": str(replay.get("evidence_projection_sha256", "")),
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "wire_emission": False,
        })
    pair_ok = methods == METHODS if require_get_post_pair else True
    confirmed = sum(int(row["typed_effect_confirmed"]) for row in results)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_loopback_evaluator_only",
        "pair_contract": {"required": bool(require_get_post_pair), "methods": sorted(methods), "get_post_pair": pair_ok},
        "episodes": results,
        "metrics": {"episode_count": len(results), "typed_positive_count": confirmed, "blocked_count": len(results) - confirmed, "get_post_covered": {method: sum(int(row["method"] == method) for row in results) for method in sorted(METHODS)}, "training_eligible_count": 0, "memory_promotion_allowed_count": 0, "vulnerability_claim_allowed_count": 0},
        "checks": {"loopback_only": all(bool(dict(episode.get("remote_probe") or {}).get("loopback_only")) for episode in episodes), "external_network_disabled": all(dict(episode.get("remote_probe") or {}).get("external_network") is False for episode in episodes), "get_post_pair": pair_ok, "wire_emission": False, "raw_material_stored": False, "promotion_blocked": True},
        "scientific_gate": {"status": "blocked", "reasons": ["evaluator projections only", "fresh real Docker not asserted by this adapter", "no real application gold"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
    }
    result["batch_evidence_sha256"] = sha256_json(result)
    return result


__all__ = ["METHODS", "SCHEMA_VERSION", "evaluate_loopback_batch", "sha256_json"]
