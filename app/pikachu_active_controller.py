"""Budgeted active-probe controller for the local Pikachu safe track."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from .active_probe import choose_active_probe
from .belief_state import DECODER_FAMILIES, MultiStepBelief
from .pikachu_replay_collector import PikachuReplayCollector, validate_pikachu_spec


ACTIVE_CONTROLLER_SCHEMA = "sift-pikachu-active-controller-v1"
DEFAULT_MAX_REQUESTS = 12


def _normalise(values: dict[str, float]) -> dict[str, float]:
    clipped = {family: max(float(values.get(family, 0.0)), 1e-8) for family in DECODER_FAMILIES}
    total = sum(clipped.values())
    return {family: value / total for family, value in clipped.items()}


def _probe_prior(spec: dict[str, Any]) -> dict[str, float]:
    """Weak modality prior; it is not a vulnerability label."""

    probabilities = {family: 1.0 for family in DECODER_FAMILIES}
    kind = str(spec.get("probe_kind", ""))
    if kind == "sql_channel_class":
        probabilities["injection"] = 3.0
    elif kind in {"inert_dom_markup", "encoded_dom_markup"}:
        probabilities["xss"] = 3.0
    return _normalise(probabilities)


def _projection_likelihood(record: dict[str, Any]) -> dict[str, float]:
    """Map bounded observable signals to a softened belief likelihood."""

    projection = record["oracle_projection"]
    probabilities = {family: 1.0 for family in DECODER_FAMILIES}
    # Family-specific differential evidence must take precedence over a
    # generic reflection bit.  Otherwise a SQL error/timeout response that
    # happens to echo a marker can collapse the posterior toward xss.
    if projection.get("controlled_differential") and projection.get("interpreter_boundary"):
        probabilities["injection"] = 6.0
    elif projection.get("sink_kind") in {"html_attribute", "script_source"} or projection.get("marker_in_attribute") or projection.get("marker_in_script_source"):
        probabilities["xss"] = 6.0
    elif projection.get("marker_in_html_text") or projection.get("marker_in_json_value") or projection.get("marker_in_header"):
        probabilities["xss"] = 1.4
    elif projection.get("sql_error_shape") or projection.get("body_length_delta_abs", 0) >= 256:
        probabilities["injection"] = 6.0
    elif projection.get("external_redirect"):
        probabilities["url_redirect"] = 6.0
    return _normalise(probabilities)


def _surface_key(spec: dict[str, Any]) -> str:
    pair = dict(spec.get("pair") or {})
    return f"{pair.get('pair_id', '')}::{pair.get('surface_role', spec.get('path', ''))}"


def _fuse_shared_route(projection: dict[str, float], shared_route: dict[str, Any] | None) -> dict[str, float]:
    """Use the shared router only as a weak prior; OOD/abstain stays neutral."""

    if not isinstance(shared_route, dict) or bool(shared_route.get("abstained", True)) or bool(shared_route.get("ood", False)):
        return projection
    prior = dict(shared_route.get("belief_prior") or {})
    if not prior:
        return projection
    return _normalise({
        family: 0.75 * float(projection.get(family, 0.0)) + 0.25 * float(prior.get(family, 0.0))
        for family in DECODER_FAMILIES
    })


class PikachuActiveController:
    """Screen surfaces first, then refine only suspicious surfaces."""

    def __init__(self, collector: PikachuReplayCollector | None = None, *, max_requests: int = DEFAULT_MAX_REQUESTS, shared_router: Any | None = None) -> None:
        self.collector = collector or PikachuReplayCollector()
        self.max_requests = max(1, min(int(max_requests), 32))
        self.belief = MultiStepBelief()
        self.shared_router = shared_router

    async def run(self, raw_specs: Iterable[dict[str, Any]]) -> dict[str, Any]:
        specs = [validate_pikachu_spec(dict(spec)) for spec in raw_specs]
        by_surface: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            by_surface.setdefault(_surface_key(spec), []).append(spec)
        for group in by_surface.values():
            group.sort(key=lambda spec: (0 if (spec.get("pair") or {}).get("variant") == "plain" else 1, str((spec.get("pair") or {}).get("variant", ""))))

        screening = [group[0] for group in sorted(by_surface.values(), key=lambda group: _surface_key(group[0]))]
        screening = screening[: self.max_requests]
        records: list[dict[str, Any]] = []
        selection_trace: list[dict[str, Any]] = []
        shared_routes: dict[str, dict[str, Any]] = {}
        screened_keys: set[str] = set()
        for spec in screening:
            record = await self.collector.collect(spec)
            records.append(record)
            key = _surface_key(spec)
            screened_keys.add(key)
            shared_route = self.shared_router.inspect(record) if self.shared_router is not None else None
            if shared_route is not None:
                shared_routes[key] = shared_route
            step = self.belief.observe(spec["path"], _fuse_shared_route(_projection_likelihood(record), shared_route), evidence_hash=record["evidence"]["evidence_hash"])
            selection_trace.append({
                "stage": "screen",
                "surface_key": key,
                "variant": (spec.get("pair") or {}).get("variant", "plain"),
                "probe_kind": spec["probe_kind"],
                "belief_step": step,
                "evidence_hash": record["evidence"]["evidence_hash"],
                "shared_router": shared_route,
            })

        refinements: list[dict[str, Any]] = []
        for record, spec in zip(records, screening):
            if not bool(record.get("rule_ir_result")):
                continue
            key = _surface_key(spec)
            for candidate in by_surface.get(key, [])[1:]:
                row = copy.deepcopy(candidate)
                shared_route = shared_routes.get(key)
                prior = dict(shared_route.get("belief_prior") or {}) if isinstance(shared_route, dict) and not bool(shared_route.get("abstained", True)) and not bool(shared_route.get("ood", False)) else _probe_prior(candidate)
                row["surface_discriminator"] = {"probabilities": prior}
                row["rule_ir_decoder"] = {"probabilities": prior, "confidence": max(prior.values())}
                row["shared_router"] = shared_route
                row["model_score"] = float(record["oracle_projection"].get("body_length_delta_abs", 0) > 0)
                refinements.append(row)

        while refinements and len(records) < self.max_requests:
            chosen = self.belief.choose_next_probe(refinements)
            # The generic active score is a tie-breaker; it sees only priors
            # attached to the candidate, never family/evaluator labels.
            active_choice = choose_active_probe([chosen])
            selected_key = _surface_key(active_choice)
            selected_variant = (active_choice.get("pair") or {}).get("variant", "")
            record = await self.collector.collect(active_choice)
            records.append(record)
            shared_route = self.shared_router.inspect(record) if self.shared_router is not None else None
            step = self.belief.observe(active_choice["path"], _fuse_shared_route(_projection_likelihood(record), shared_route), evidence_hash=record["evidence"]["evidence_hash"])
            selection_trace.append({
                "stage": "refine",
                "surface_key": selected_key,
                "variant": selected_variant,
                "probe_kind": active_choice["probe_kind"],
                "belief_probe_score": chosen.get("belief_probe_score"),
                "belief_information_gain": chosen.get("belief_information_gain"),
                "belief_step": step,
                "evidence_hash": record["evidence"]["evidence_hash"],
                "shared_router": shared_route,
            })
            refinements = [
                row for row in refinements
                if not (_surface_key(row) == selected_key and (row.get("pair") or {}).get("variant") == selected_variant)
            ]

        safety = {
            "schema_version": ACTIVE_CONTROLLER_SCHEMA,
            "max_requests": self.max_requests,
            "request_count": len(records),
            "records": records,
            "selection_trace": selection_trace,
            "belief": self.belief.snapshot(),
            "safety": {
                "loopback_only": True,
                "read_only_get_only": True,
                "fresh_target": bool(getattr(self.collector, "fresh_target", False)),
                "target_instance_id": str(getattr(self.collector, "target_instance_id", "unattested")),
                "external_network": False,
                "script_execution": False,
                "database_write": False,
                "raw_body_stored": False,
            },
        }
        if self.shared_router is not None:
            safety["safety"]["shared_router_diagnostic_only"] = True
        return safety


__all__ = ["ACTIVE_CONTROLLER_SCHEMA", "PikachuActiveController"]
