"""Maze-style state tracking and goal predicates for local security episodes.

The agent only sees HTTP/browser observations.  Evaluator state is deliberately
not used by the graph or the policy.  A transition can therefore be a maze
candidate without being a confirmed challenge solve.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlsplit


AUTH_DENIED_STATUS_CODES = frozenset({401, 403})
REDIRECT_STATUS_CODES = frozenset(range(300, 400))

# These are observable exit predicates supplied by a local adapter or a
# browser/instrumentation layer.  They are deliberately not exploit strings:
# each predicate describes the semantic boundary that the lab is testing.
RULE_EXIT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "authentication": ("protected_resource_transition", "session_signal"),
    "access_control": ("protected_resource_transition",),
    "input_validation": ("controlled_differential", "invariant_violation"),
    "injection": ("controlled_differential", "interpreter_boundary"),
    "xss": ("browser_sink_observed", "dom_change"),
    "url_redirect": ("location_origin_changed",),
    "observability": ("sensitive_surface_visible",),
    "information_exposure": ("sensitive_surface_visible",),
    "misconfiguration": ("unsafe_surface_reachable",),
    "logic": ("invariant_violation", "state_replay"),
}

RULE_EXIT_CHANNEL_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "injection": {
        "ast_shape": ("controlled_differential", "interpreter_boundary"),
        "syntax_error": ("controlled_differential", "syntax_error_observed"),
        "blind_response": ("controlled_differential", "blind_boolean_differential"),
        "bounded_timing": ("controlled_differential", "timing_differential", "timeout_observed"),
        "local_side_channel": ("controlled_differential", "local_callback_observed"),
    },
}


def _payload(observation: dict[str, Any] | None) -> dict[str, Any]:
    """Return the response observation from either adapter or raw shape."""

    if not isinstance(observation, dict):
        return {}
    nested = observation.get("observation")
    return dict(nested) if isinstance(nested, dict) else dict(observation)


def _summary(observation: dict[str, Any] | None) -> dict[str, Any]:
    summary = _payload(observation).get("summary")
    return dict(summary) if isinstance(summary, dict) else {}


def _headers(observation: dict[str, Any] | None) -> dict[str, str]:
    headers = _payload(observation).get("headers")
    if not isinstance(headers, dict):
        return {}
    return {str(key).casefold(): str(value) for key, value in headers.items()}


def status_code(observation: dict[str, Any] | None) -> int:
    try:
        return int(_payload(observation).get("status_code", 0))
    except (TypeError, ValueError):
        return 0


def status_class(code: int) -> str:
    if 100 <= code <= 599:
        return f"{code // 100}xx"
    return "other"


def action_resource(action_or_observation: dict[str, Any] | None) -> str:
    """Normalize a route for same-resource before/after comparisons.

    Query values are intentionally omitted.  The maze can still retain query
    *keys* in action identities, while an auth goal compares the protected
    resource rather than treating every payload as a different room.
    """

    if not isinstance(action_or_observation, dict):
        return ""
    action = action_or_observation.get("action")
    if not isinstance(action, dict):
        action = action_or_observation
    method = str(action.get("method", "GET")).upper()
    path = str(action.get("path", ""))
    parsed = urlsplit(path)
    normalized = parsed.path.rstrip("/") or "/"
    return f"{method} {normalized}"


def action_identity(action: dict[str, Any] | None) -> dict[str, Any]:
    """Create a non-secret identity for an action edge."""

    action = dict(action or {})
    path = str(action.get("path", ""))
    parsed = urlsplit(path)
    return {
        "method": str(action.get("method", "GET")).upper(),
        "path": parsed.path.rstrip("/") or "/",
        "query_keys": sorted({str(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}),
        "json_keys": sorted(str(key) for key in dict(action.get("json") or {}).keys()),
        "form_keys": sorted(str(key) for key in dict(action.get("form") or {}).keys()),
    }


def _length_bucket(value: Any) -> str:
    try:
        length = max(0, int(value))
    except (TypeError, ValueError):
        return "unknown"
    if length == 0:
        return "0"
    if length < 256:
        return "1-255"
    if length < 4096:
        return "256-4095"
    if length < 65536:
        return "4096-65535"
    return "65536+"


def _location_projection(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    parsed = urlsplit(str(value))
    # Keep only a local path and query-key shape.  No external destination is
    # followed and query values are not persisted in maze state.
    query_keys = sorted({part.split("=", 1)[0] for part in parsed.query.split("&") if part})
    path = parsed.path.rstrip("/") or "/"
    return path + ("?" + "&".join(query_keys) if query_keys else "")


def state_projection(observation: dict[str, Any] | None) -> dict[str, Any]:
    """Project a response into a compact, challenge-label-free maze state."""

    payload = _payload(observation)
    summary = _summary(observation)
    headers = _headers(observation)
    location = headers.get("location")
    return {
        "status_code": status_code(observation),
        "status_class": status_class(status_code(observation)),
        "content_type": headers.get("content-type", "").split(";", 1)[0].strip().casefold(),
        "location": _location_projection(location),
        "has_www_authenticate": bool(headers.get("www-authenticate")),
        "json_shape": summary.get("json_shape"),
        "semantic_body_sha256": str(summary.get("semantic_body_sha256", "")),
        "body_length_bucket": _length_bucket(summary.get("body_length")),
        "cookie_jar_changed": bool(summary.get("cookie_jar_changed", False)),
        "transport_error": bool(summary.get("transport_error", False)),
        # The action is useful for resource identity but not for the response
        # node itself.  This keeps node fingerprints stable across replays.
        "resource": action_resource(payload.get("action") or observation),
    }


def state_fingerprint(observation: dict[str, Any] | None) -> str:
    projection = state_projection(observation)
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def edge_fingerprint(before: dict[str, Any] | None, action: dict[str, Any] | None, after: dict[str, Any] | None) -> str:
    projection = {
        "from": state_fingerprint(before),
        "action": action_identity(action),
        "to": state_fingerprint(after),
    }
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_denied(observation: dict[str, Any] | None) -> bool:
    return status_code(observation) in AUTH_DENIED_STATUS_CODES


def _is_accepted(observation: dict[str, Any] | None) -> bool:
    return 200 <= status_code(observation) < 300


def _extract_after(transition: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(transition, dict):
        return {}
    after = transition.get("after")
    if isinstance(after, dict):
        return after
    return transition


def _transition_parts(transition: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(transition, dict):
        return None, None, {}
    if isinstance(transition.get("after"), dict):
        before = transition.get("before") if isinstance(transition.get("before"), dict) else None
        action = transition.get("action") if isinstance(transition.get("action"), dict) else None
        return before, action, dict(transition["after"])
    return None, None, dict(transition)


def transition_signal(
    before: dict[str, Any] | None,
    action: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize a visible edge without assigning a hidden challenge label."""

    before_projection = state_projection(before)
    after_projection = state_projection(after)
    return {
        "resource_before": action_resource(before),
        "resource_after": action_resource(after),
        "action": action_identity(action),
        "status_before": status_code(before),
        "status_after": status_code(after),
        "denied_to_accepted": _is_denied(before) and _is_accepted(after),
        "status_changed": status_code(before) != status_code(after),
        "semantic_body_changed": before_projection["semantic_body_sha256"] != after_projection["semantic_body_sha256"],
        "cookie_jar_changed": bool(_summary(after).get("cookie_jar_changed", False)),
        "credential_state_changed": bool(_summary(after).get("credential_state_changed", False)),
        "redirect": status_code(after) in REDIRECT_STATUS_CODES,
        "from_state": state_fingerprint(before),
        "to_state": state_fingerprint(after),
    }


def _has_session_signal(auth_transition: dict[str, Any] | None) -> bool:
    if not auth_transition:
        return False
    after = _extract_after(auth_transition)
    summary = _summary(after)
    return bool(
        summary.get("cookie_jar_changed", False)
        or summary.get("credential_state_changed", False)
        or auth_transition.get("cookie_jar_changed", False)
        or auth_transition.get("credential_state_changed", False)
    )


def assess_authentication_exit(
    protected_before: dict[str, Any] | None,
    protected_after: dict[str, Any] | None,
    *,
    auth_transition: dict[str, Any] | None = None,
    protected_rechecks: Iterable[dict[str, Any]] | None = None,
    evaluator_confirmed: bool = False,
) -> dict[str, Any]:
    """Classify whether a maze path reached an authentication exit.

    A public 2xx response is never enough.  The resource must first have a
    denied baseline, the follow-up must be the same resource, and a second
    accepted recheck is required for an observable success.  The only way to
    reach ``evaluator_confirmed`` is an explicit evaluator-side transition
    supplied by the harness after a fresh reset; it is not inferred here.
    """

    baseline_resource = action_resource(protected_before)
    after_resource = action_resource(protected_after)
    rechecks = [dict(row) for row in (protected_rechecks or [])]
    reasons: list[str] = []
    same_resource = bool(baseline_resource) and baseline_resource == after_resource
    baseline_denied = _is_denied(protected_before)
    followup_accepted = bool(protected_after) and _is_accepted(protected_after)
    rechecks_accepted = all(_is_accepted(row) for row in rechecks)
    session_signal = _has_session_signal(auth_transition)

    if not same_resource:
        reasons.append("protected_resource_changed")
    if not baseline_denied:
        reasons.append("baseline_was_not_denied")
    if not followup_accepted:
        reasons.append("protected_followup_not_2xx")
    if rechecks and not rechecks_accepted:
        reasons.append("protected_recheck_not_2xx")
    if not session_signal:
        reasons.append("no_visible_session_state_change")

    valid_candidate = same_resource and baseline_denied and followup_accepted
    stable_observable = valid_candidate and session_signal and bool(rechecks) and rechecks_accepted
    if stable_observable and evaluator_confirmed:
        status = "evaluator_confirmed"
        reasons.append("fresh_evaluator_transition_supplied")
    elif stable_observable:
        status = "observable_success"
        reasons.append("same_protected_resource_accepted_on_recheck")
    elif valid_candidate:
        status = "candidate"
        reasons.append("single_protected_resource_acceptance_needs_recheck")
    else:
        status = "not_goal"

    auth_before, auth_action, auth_after = _transition_parts(auth_transition)
    return {
        "goal": "authentication_exit",
        "status": status,
        "observable": status in {"observable_success", "evaluator_confirmed"},
        "evaluator_confirmed": status == "evaluator_confirmed",
        "resource": baseline_resource or after_resource,
        "baseline_status": status_code(protected_before),
        "followup_status": status_code(protected_after),
        "recheck_count": len(rechecks),
        "session_signal": session_signal,
        "auth_transition": transition_signal(auth_before, auth_action, auth_after) if auth_transition else None,
        "reasons": reasons,
    }


def assess_rule_exit(
    family: str,
    *,
    visible_evidence: dict[str, Any] | None = None,
    rechecks: Iterable[dict[str, Any]] | None = None,
    evaluator_confirmed: bool = False,
) -> dict[str, Any]:
    """Apply one common three-level exit protocol to any rule family.

    ``visible_evidence`` is produced by an instrumented local adapter.  For
    example, an XSS adapter must report a browser sink and DOM delta; an HTTP
    response that merely reflects text is only a candidate.  SQL/injection
    families require a controlled differential and an interpreter-boundary
    signal, while stateful business logic requires a replayed invariant.
    ``evaluator_confirmed`` is an external harness result and is never inferred
    from these observations.
    """

    family = str(family).casefold()
    evidence = dict(visible_evidence or {})
    modality = str(evidence.get("modality", ""))
    requirements = RULE_EXIT_CHANNEL_REQUIREMENTS.get(family, {}).get(
        modality,
        RULE_EXIT_REQUIREMENTS.get(family, ("predicate_satisfied",)),
    )
    missing = [name for name in requirements if not bool(evidence.get(name, False))]
    candidate_signal = bool(evidence.get("candidate_signal", False)) or any(
        bool(evidence.get(name, False)) for name in requirements
    )
    rows = [dict(row) for row in (rechecks or [])]
    stable = bool(rows) and all(
        all(bool(row.get(name, False)) for name in requirements)
        for row in rows
    )
    reasons: list[str] = []
    if not candidate_signal:
        reasons.append("no_visible_exit_signal")
    if missing:
        reasons.append("missing_required_evidence:" + ",".join(missing))
    if requirements and rows and not stable:
        reasons.append("recheck_did_not_reproduce_exit")
    if evidence.get("dead_end"):
        reasons.append("adapter_marked_dead_end")

    if evidence.get("dead_end") or not candidate_signal:
        status = "dead_end" if evidence.get("dead_end") else "not_goal"
    elif missing or not stable:
        status = "candidate"
    elif evaluator_confirmed:
        status = "evaluator_confirmed"
        reasons.append("fresh_evaluator_transition_supplied")
    else:
        status = "observable_success"
        reasons.append("family_exit_reproduced_on_recheck")

    return {
        "goal": "rule_exit",
        "family": family,
        "status": status,
        "observable": status == "observable_success",
        "evaluator_confirmed": status == "evaluator_confirmed",
        "required_evidence": list(requirements),
        "missing_evidence": missing,
        "recheck_count": len(rows),
        "reproduced": stable,
        "evidence": evidence,
        "reasons": reasons,
    }


@dataclass
class MazeNode:
    fingerprint: str
    projection: dict[str, Any]
    visits: int = 0
    outgoing_edges: list[str] = field(default_factory=list)
    dead_end: bool = False
    dead_end_reasons: list[str] = field(default_factory=list)


@dataclass
class MazeEdge:
    fingerprint: str
    from_node: str
    to_node: str
    action: dict[str, Any]
    kind: str
    visits: int = 1
    goal_status: str | None = None


class MazeFrontier:
    """Priority queue for unexplored exits from visible maze nodes."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], dict[str, Any]] = {}
        self._consumed: set[tuple[str, str]] = set()
        self._serial = 0

    @staticmethod
    def _key(node_fingerprint: str, action: dict[str, Any]) -> tuple[str, str]:
        identity = json.dumps(action_identity(action), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return node_fingerprint, identity

    def add(self, node_fingerprint: str, action: dict[str, Any], *, priority: float = 0.0, reason: str = "novel") -> bool:
        key = self._key(node_fingerprint, action)
        if key in self._pending or key in self._consumed:
            return False
        self._pending[key] = {
            "node": node_fingerprint,
            "action": action_identity(action),
            "priority": float(priority),
            "reason": str(reason),
            "serial": self._serial,
        }
        self._serial += 1
        return True

    def pop(self) -> dict[str, Any] | None:
        if not self._pending:
            return None
        key, item = max(self._pending.items(), key=lambda pair: (pair[1]["priority"], -pair[1]["serial"]))
        del self._pending[key]
        self._consumed.add(key)
        return dict(item)

    def __len__(self) -> int:
        return len(self._pending)

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": sorted(self._pending.values(), key=lambda item: (-item["priority"], item["serial"])),
            "consumed_count": len(self._consumed),
        }


class MazeGraph:
    """Append-visible state graph with explicit loop and dead-end markers."""

    def __init__(self) -> None:
        self.nodes: dict[str, MazeNode] = {}
        self.edges: dict[str, MazeEdge] = {}
        self.path: list[str] = []
        self.frontier = MazeFrontier()

    def ensure_node(self, observation: dict[str, Any] | None) -> MazeNode:
        fingerprint = state_fingerprint(observation)
        node = self.nodes.get(fingerprint)
        if node is None:
            node = MazeNode(fingerprint=fingerprint, projection=state_projection(observation))
            self.nodes[fingerprint] = node
        return node

    def record_transition(
        self,
        before: dict[str, Any] | None,
        action: dict[str, Any],
        after: dict[str, Any] | None,
        *,
        goal_status: str | None = None,
    ) -> dict[str, Any]:
        from_node = self.ensure_node(before)
        to_node = self.ensure_node(after)
        from_node.visits += 1
        to_node.visits += 1
        if not self.path:
            self.path.append(from_node.fingerprint)
        if to_node.fingerprint == from_node.fingerprint:
            kind = "self_loop"
        elif to_node.fingerprint in self.path:
            kind = "loop"
        elif to_node.visits > 1:
            kind = "revisit"
        else:
            kind = "forward"
        fingerprint = edge_fingerprint(before, action, after)
        edge = self.edges.get(fingerprint)
        if edge is None:
            edge = MazeEdge(
                fingerprint=fingerprint,
                from_node=from_node.fingerprint,
                to_node=to_node.fingerprint,
                action=action_identity(action),
                kind=kind,
                goal_status=goal_status,
            )
            self.edges[fingerprint] = edge
            if fingerprint not in from_node.outgoing_edges:
                from_node.outgoing_edges.append(fingerprint)
        else:
            edge.visits += 1
            if goal_status is not None:
                edge.goal_status = goal_status
        self.path.append(to_node.fingerprint)
        return {
            "edge": fingerprint,
            "from": from_node.fingerprint,
            "to": to_node.fingerprint,
            "kind": kind,
            "goal_status": goal_status,
            "path_depth": len(self.path),
        }

    def mark_dead_end(self, node_or_observation: str | dict[str, Any], reason: str) -> str:
        fingerprint = node_or_observation if isinstance(node_or_observation, str) else self.ensure_node(node_or_observation).fingerprint
        node = self.nodes[fingerprint]
        node.dead_end = True
        if reason not in node.dead_end_reasons:
            node.dead_end_reasons.append(str(reason))
        return fingerprint

    def enqueue(self, observation: dict[str, Any] | None, action: dict[str, Any], *, priority: float = 0.0, reason: str = "novel") -> bool:
        return self.frontier.add(state_fingerprint(observation), action, priority=priority, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "path": list(self.path),
            "nodes": [
                {
                    "fingerprint": node.fingerprint,
                    "projection": node.projection,
                    "visits": node.visits,
                    "outgoing_edges": list(node.outgoing_edges),
                    "dead_end": node.dead_end,
                    "dead_end_reasons": list(node.dead_end_reasons),
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "fingerprint": edge.fingerprint,
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "action": edge.action,
                    "kind": edge.kind,
                    "visits": edge.visits,
                    "goal_status": edge.goal_status,
                }
                for edge in self.edges.values()
            ],
            "frontier": self.frontier.snapshot(),
        }
