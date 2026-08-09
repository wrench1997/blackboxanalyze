"""PG-50 independent multi-implementation stability-matrix fixture.

The fixture exposes only bounded JSON projections.  It models ten semantic
surface families, three independent HTTP layouts, three response variants and
method-sensitive confirm channels.  No probe executes, writes state, touches a
database, accesses credentials or leaves loopback.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


PG50_SCHEMA = "sift-pg50-stability-matrix-fixture-v1"
IMPLEMENTATIONS = ("ember", "frost", "quartz")
VARIANTS = ("plain", "wrapped", "framed")
PHASES = ("screen", "confirm", "error", "timeout")
LAYOUTS: dict[str, dict[str, str]] = {
    "ember": {
        "prefix": "/matrix/ember/check",
        "surface_key": "node_ref",
        "probe_key": "marker_ref",
        "variant_key": "view_ref",
        "phase_key": "state_ref",
        "post_content_type": "application/json",
    },
    "frost": {
        "prefix": "/lane/frost/inspect",
        "surface_key": "resource_ref",
        "probe_key": "signal_ref",
        "variant_key": "envelope_ref",
        "phase_key": "phase_ref",
        "post_content_type": "application/x-www-form-urlencoded",
    },
    "quartz": {
        "prefix": "/qz/quartz/verify",
        "surface_key": "object_ref",
        "probe_key": "canary_ref",
        "variant_key": "frame_ref",
        "phase_key": "transition_ref",
        "post_content_type": "application/json",
    },
}

SURFACE_SPECS: dict[str, dict[str, Any]] = {
    "surface-01": {"family": "xss", "semantic": "markup-context", "channel": "query-channel", "probe": "markup-observation", "effect": "dom_structure", "modality": "typed_dom_effect"},
    "surface-02": {"family": "injection", "semantic": "operator-context", "channel": "form-channel", "probe": "operator-observation", "effect": "interpreter_boundary", "modality": "typed_ast_difference"},
    "surface-03": {"family": "authentication", "semantic": "auth-boundary", "channel": "query-channel", "probe": "auth-observation", "effect": "authentication_boundary", "modality": "typed_auth_boundary"},
    "surface-04": {"family": "access_control", "semantic": "subject-boundary", "channel": "form-channel", "probe": "subject-observation", "effect": "authorization_boundary", "modality": "typed_authorization_boundary"},
    "surface-05": {"family": "logic", "semantic": "state-boundary", "channel": "query-channel", "probe": "state-observation", "effect": "business_invariant", "modality": "typed_logic_invariant"},
    "surface-06": {"family": "url_redirect", "semantic": "destination-context", "channel": "form-channel", "probe": "destination-observation", "effect": "redirect_origin", "modality": "typed_redirect_projection"},
    "surface-07": {"family": "input_validation", "semantic": "validation-boundary", "channel": "query-channel", "probe": "validation-observation", "effect": "validation_boundary", "modality": "typed_validation_boundary"},
    "surface-08": {"family": "command_injection", "semantic": "command-boundary", "channel": "form-channel", "probe": "command-observation", "effect": "command_canary", "modality": "typed_command_boundary"},
    "surface-09": {"family": "template_injection", "semantic": "template-boundary", "channel": "form-channel", "probe": "template-observation", "effect": "interpreter_boundary", "modality": "typed_template_boundary"},
    "surface-10": {"family": "ordinary_response", "semantic": "ordinary-surface", "channel": "query-channel", "probe": "ordinary-observation", "effect": "none", "modality": "negative_control"},
}


def _text(value: Any) -> str:
    return str(value)[:96]


def _request_values(handler: BaseHTTPRequestHandler, layout: dict[str, str]) -> dict[str, str]:
    if handler.command == "GET":
        parsed = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        return {name: _text(parsed.get(layout[name], [""])[0]) for name in ("surface_key", "probe_key", "variant_key", "phase_key")}
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
    return {name: _text(values.get(layout[name], "")) for name in ("surface_key", "probe_key", "variant_key", "phase_key")}


def _typed_fields(family: str, positive: bool) -> dict[str, Any]:
    base: dict[str, Any] = {
        "candidate_signal": bool(positive),
        "bounded_observation": bool(positive),
        "state_mutated": False,
        "database_touched": False,
        "credentials_accessed": False,
        "external_network": False,
        "script_execution": False,
    }
    base.update({
        "xss": {"dom_structure_delta": bool(positive), "marker_hits": int(positive)},
        "injection": {"ast_shape_delta": bool(positive), "operator_boundary": bool(positive), "ast_node_delta": int(positive)},
        "authentication": {"authentication_boundary": bool(positive), "authenticated": False},
        "access_control": {"authorization_boundary": bool(positive), "cross_subject_access": bool(positive)},
        "logic": {"business_invariant_boundary": bool(positive), "history_changed": False},
        "url_redirect": {"redirect_candidate": bool(positive), "same_origin": True, "external_redirect": False},
        "input_validation": {"validation_boundary": bool(positive), "rejected": bool(positive)},
        "command_injection": {"command_boundary": bool(positive), "command_canary": bool(positive), "executed": False},
        "template_injection": {"template_context_delta": bool(positive), "render_executed": False},
        "ordinary_response": {"ordinary_response": True, "candidate_signal": False},
    }[family])
    return base


def _render(implementation: str, surface: str, variant: str, phase: str, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    typed = _typed_fields(spec["family"], positive)
    if variant == "plain":
        result: dict[str, Any] = {"schema": PG50_SCHEMA, "implementation": implementation, "phase": phase, "typed": typed, "bounded": True}
    elif variant == "wrapped":
        result = {"envelope": {"schema": PG50_SCHEMA, "implementation": implementation, "phase": phase}, "payload": {"typed": typed, "bounded": True}}
    else:
        result = {"frame": {"schema": PG50_SCHEMA, "implementation": implementation, "phase": phase, "bounded": True}, "observation": typed}
    if positive:
        result["effect"] = {"confirmed": True, "scope": spec["effect"]}
        result["proof"] = {"read_only": True, "bounded": True}
    result["ambiguous"] = phase == "screen"
    return result


class PG50Handler(BaseHTTPRequestHandler):
    server_version = "PG50StabilityMatrix/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve(self) -> None:
        implementation = str(getattr(self.server, "pg50_implementation", ""))
        layout = LAYOUTS.get(implementation)
        path = urlsplit(self.path).path
        surface = path.rsplit("/", 1)[-1] if layout and path.startswith(f"{layout['prefix']}/") else ""
        if layout is None or surface not in SURFACE_SPECS:
            self._write(404, {"ok": False, "bounded_error": "unknown_surface"})
            return
        values = _request_values(self, layout)
        spec = SURFACE_SPECS[surface]
        valid_variant = values["variant_key"] in VARIANTS
        method_channel = "query-channel" if self.command == "GET" else "form-channel"
        candidate = values["surface_key"] == surface and values["probe_key"] == spec["probe"] and valid_variant
        positive = candidate and values["phase_key"] == "confirm" and method_channel == spec["channel"] and spec["family"] != "ordinary_response"
        phase = values["phase_key"]
        if phase == "error":
            self._write(422, {"ok": False, "fault": {"class": "bounded_syntax"}, "state_mutated": False, "external_network": False})
            return
        if phase == "timeout":
            self._write(429, {"ok": False, "fault": {"class": "bounded_timeout"}, "state_mutated": False, "external_network": False})
            return
        self._write(200, _render(implementation, surface, values["variant_key"] if valid_variant else "plain", phase if phase in {"screen", "confirm"} else "unknown", positive))

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("x-pg50-surface", "bounded")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()


class PG50Server(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, port: int, implementation: str) -> None:
        if implementation not in LAYOUTS:
            raise ValueError("unknown PG-50 implementation")
        super().__init__(("127.0.0.1", int(port)), PG50Handler)
        self.pg50_implementation = implementation


def make_pg50_server(port: int = 32080, implementation: str = "ember") -> PG50Server:
    return PG50Server(int(port), implementation)


__all__ = ["IMPLEMENTATIONS", "LAYOUTS", "PHASES", "PG50_SCHEMA", "SURFACE_SPECS", "VARIANTS", "make_pg50_server"]
