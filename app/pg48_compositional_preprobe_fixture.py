"""PG-48 method-sensitive compositional loopback fixture.

The fixture is independent of PG-37/42.  Each surface has a language-neutral
semantic slot and a required channel slot.  The positive typed observation is
emitted only on the required GET or POST confirm action.  Nothing executes or
mutates state; responses are safe JSON projections for local research.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


PG48_SCHEMA = "sift-pg48-compositional-preprobe-fixture-v1"
IMPLEMENTATIONS = ("ember", "frost")
VARIANTS = ("plain", "wrapped")
PHASES = ("screen", "confirm", "error", "timeout")
LAYOUTS = {
    "ember": {"prefix": "/observe/ember", "surface_key": "surface_slot", "probe_key": "probe_slot", "variant_key": "view_slot", "phase_key": "phase_slot", "post_content_type": "application/json"},
    "frost": {"prefix": "/edge/frost/check", "surface_key": "resource_slot", "probe_key": "signal_slot", "variant_key": "envelope_slot", "phase_key": "stage_slot", "post_content_type": "application/x-www-form-urlencoded"},
}
SURFACE_SPECS: dict[str, dict[str, Any]] = {
    "node-01": {"family": "xss", "semantic": "markup-context", "channel": "query-channel", "probe": "markup-observation", "effect": "dom_structure", "modality": "typed_dom_effect"},
    "node-02": {"family": "injection", "semantic": "operator-context", "channel": "form-channel", "probe": "operator-observation", "effect": "interpreter_boundary", "modality": "typed_ast_difference"},
    "node-03": {"family": "authentication", "semantic": "auth-boundary", "channel": "query-channel", "probe": "auth-observation", "effect": "authentication_boundary", "modality": "typed_auth_boundary"},
    "node-04": {"family": "access_control", "semantic": "subject-boundary", "channel": "form-channel", "probe": "subject-observation", "effect": "authorization_boundary", "modality": "typed_authorization_boundary"},
    "node-05": {"family": "logic", "semantic": "state-boundary", "channel": "query-channel", "probe": "state-observation", "effect": "business_invariant", "modality": "typed_logic_invariant"},
    "node-06": {"family": "url_redirect", "semantic": "destination-context", "channel": "form-channel", "probe": "destination-observation", "effect": "redirect_origin", "modality": "typed_redirect_projection"},
    "node-07": {"family": "template_injection", "semantic": "template-boundary", "channel": "form-channel", "probe": "template-observation", "effect": "interpreter_boundary", "modality": "typed_template_boundary"},
    "node-08": {"family": "ordinary_response", "semantic": "ordinary-surface", "channel": "query-channel", "probe": "ordinary-observation", "effect": "none", "modality": "negative_control"},
}


def _decode(value: Any) -> str:
    return unquote(str(value))[:96]


def _values(handler: BaseHTTPRequestHandler, layout: dict[str, str]) -> dict[str, str]:
    if handler.command == "GET":
        parsed = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        return {key: _decode(parsed.get(layout[key], [""])[0]) for key in ("surface_key", "probe_key", "variant_key", "phase_key")}
    try:
        length = min(max(int(handler.headers.get("content-length", "0")), 0), 2048)
    except (TypeError, ValueError):
        length = 0
    raw = handler.rfile.read(length)
    content_type = str(handler.headers.get("content-type", "")).split(";", 1)[0].casefold()
    if content_type == "application/json":
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
        values = parsed if isinstance(parsed, dict) else {}
    else:
        parsed = parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
        values = {key: item[0] for key, item in parsed.items() if item}
    return {key: _decode(values.get(layout[key], "")) for key in ("surface_key", "probe_key", "variant_key", "phase_key")}


def _typed_fields(family: str, positive: bool) -> dict[str, Any]:
    base: dict[str, Any] = {"candidate_signal": bool(positive), "bounded_observation": bool(positive), "state_mutated": False, "database_touched": False, "credentials_accessed": False, "external_network": False, "script_execution": False}
    fields = {
        "xss": {"dom_structure_delta": bool(positive), "marker_hits": int(positive)},
        "injection": {"ast_shape_delta": bool(positive), "operator_boundary": bool(positive), "ast_node_delta": int(positive)},
        "authentication": {"authentication_boundary": bool(positive), "authenticated": False},
        "access_control": {"authorization_boundary": bool(positive), "cross_subject_access": bool(positive)},
        "logic": {"business_invariant_boundary": bool(positive), "history_changed": False},
        "url_redirect": {"redirect_candidate": bool(positive), "same_origin": True, "external_redirect": False},
        "template_injection": {"template_context_delta": bool(positive), "render_executed": False},
        "ordinary_response": {"ordinary_response": True, "candidate_signal": False},
    }
    base.update(fields[family]); return base


def _render(implementation: str, surface: str, variant: str, phase: str, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]; typed = _typed_fields(spec["family"], positive)
    if variant == "plain":
        payload: dict[str, Any] = {"schema": PG48_SCHEMA, "impl": implementation, "phase": phase, "typed": typed, "accepted": True}
    else:
        payload = {"envelope": {"schema": PG48_SCHEMA, "impl": implementation, "phase": phase}, "payload": {"typed": typed, "accepted": True}, "meta": {"bounded": True, "view": variant}}
    if positive:
        payload["effect"] = {"confirmed": True, "scope": spec["effect"]}; payload["proof"] = {"read_only": True, "bounded": True}
    payload["ambiguous"] = phase == "screen"; return payload


class PG48Handler(BaseHTTPRequestHandler):
    server_version = "PG48Compositional/1.0"; protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve(self) -> None:
        implementation = str(getattr(self.server, "pg48_implementation", "")); layout = LAYOUTS.get(implementation); path = urlsplit(self.path).path
        surface = path.rsplit("/", 1)[-1] if layout and path.startswith(f"{layout['prefix']}/") else ""
        if layout is None or surface not in SURFACE_SPECS:
            self._write(404, {"ok": False, "bounded_error": "unknown_surface"}); return
        values = _values(self, layout); spec = SURFACE_SPECS[surface]; valid_variant = values["variant_key"] in VARIANTS; method_channel = "query-channel" if self.command == "GET" else "form-channel"
        candidate = values["surface_key"] == surface and values["probe_key"] == spec["probe"] and valid_variant
        positive = candidate and values["phase_key"] == "confirm" and method_channel == spec["channel"] and spec["family"] != "ordinary_response"
        phase = values["phase_key"]
        if phase == "error": self._write(422, {"ok": False, "fault": {"class": "bounded_syntax"}, "state_mutated": False, "external_network": False}); return
        if phase == "timeout": self._write(429, {"ok": False, "fault": {"class": "bounded_timeout"}, "state_mutated": False, "external_network": False}); return
        self._write(200, _render(implementation, surface, values["variant_key"] if valid_variant else "plain", phase if phase in {"screen", "confirm"} else "unknown", positive))

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"); self.send_response(int(status)); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(encoded))); self.send_header("x-pg48-surface", "bounded"); self.send_header("connection", "close"); self.end_headers(); self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()


class PG48Server(ThreadingHTTPServer):
    def __init__(self, port: int, implementation: str) -> None:
        if implementation not in LAYOUTS: raise ValueError("unknown PG-48 implementation")
        super().__init__(("127.0.0.1", int(port)), PG48Handler); self.pg48_implementation = implementation


def make_pg48_server(port: int = 31980, implementation: str = "ember") -> PG48Server:
    return PG48Server(int(port), implementation)


__all__ = ["IMPLEMENTATIONS", "LAYOUTS", "PHASES", "PG48_SCHEMA", "SURFACE_SPECS", "VARIANTS", "make_pg48_server"]
