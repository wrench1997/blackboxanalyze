"""PG-42 independent loopback fixture.

This target intentionally does not import or reuse the PG-37/40 HTTP
implementation.  It exposes the same safe abstract probe contract through
different route/field names and three different response envelopes.  A
positive response only reports a bounded typed observation; no script,
command, SQL statement, redirect, credential flow, or persistent mutation is
performed.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


PG42_SCHEMA = "sift-pg42-independent-semantic-fixture-v1"
IMPLEMENTATIONS = ("cobalt", "quartz")
VARIANTS = ("ledger", "envelope", "framed")
PHASES = ("screen", "confirm", "error", "timeout")

LAYOUTS: dict[str, dict[str, str]] = {
    "cobalt": {
        "prefix": "/v2/cobalt/check",
        "surface_key": "resource_id",
        "probe_key": "probe_class",
        "variant_key": "render_mode",
        "phase_key": "checkpoint",
        "post_content_type": "application/json",
    },
    "quartz": {
        "prefix": "/gateway/quartz/evaluate",
        "surface_key": "subject_ref",
        "probe_key": "signal_type",
        "variant_key": "projection_mode",
        "phase_key": "stage",
        "post_content_type": "application/x-www-form-urlencoded",
    },
}

SURFACE_SPECS: dict[str, dict[str, str]] = {
    "node-01": {"family": "xss", "semantic": "markup-context", "probe": "markup-observation", "effect": "dom_structure_delta", "modality": "typed_dom_effect"},
    "node-02": {"family": "injection", "semantic": "operator-context", "probe": "operator-observation", "effect": "ast_shape_delta", "modality": "typed_ast_difference"},
    "node-03": {"family": "authentication", "semantic": "auth-boundary", "probe": "auth-observation", "effect": "auth_boundary_delta", "modality": "typed_auth_boundary"},
    "node-04": {"family": "access_control", "semantic": "subject-boundary", "probe": "subject-observation", "effect": "subject_boundary_delta", "modality": "typed_authorization_boundary"},
    "node-05": {"family": "logic", "semantic": "state-boundary", "probe": "state-observation", "effect": "invariant_delta", "modality": "typed_logic_invariant"},
    "node-06": {"family": "url_redirect", "semantic": "destination-context", "probe": "destination-observation", "effect": "origin_projection_delta", "modality": "typed_redirect_projection"},
    "node-07": {"family": "input_validation", "semantic": "scalar-boundary", "probe": "scalar-observation", "effect": "validation_delta", "modality": "typed_validation_boundary"},
    "node-08": {"family": "command_injection", "semantic": "local-callback", "probe": "canary-observation", "effect": "local_canary_delta", "modality": "typed_local_canary"},
    "node-09": {"family": "template_injection", "semantic": "template-boundary", "probe": "template-observation", "effect": "template_context_delta", "modality": "typed_template_boundary"},
    "node-10": {"family": "ordinary_response", "semantic": "ordinary-surface", "probe": "ordinary-observation", "effect": "none", "modality": "negative_control"},
}


def _decode(value: Any) -> str:
    return unquote(str(value))[:96]


def _read_values(handler: BaseHTTPRequestHandler, layout: dict[str, str]) -> dict[str, str]:
    if handler.command == "GET":
        values = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        return {key: _decode(values.get(layout[key], [""])[0]) for key in ("surface_key", "probe_key", "variant_key", "phase_key")}
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
    common: dict[str, Any] = {
        "candidate_signal": bool(positive),
        "bounded_observation": bool(positive),
        "state_mutated": False,
        "database_touched": False,
        "credentials_accessed": False,
        "external_network": False,
        "script_execution": False,
    }
    fields = {
        "xss": {"dom_structure_delta": bool(positive), "marker_hits": int(positive)},
        "injection": {"ast_shape_delta": bool(positive), "operator_boundary": bool(positive), "ast_node_delta": int(positive)},
        "authentication": {"auth_boundary_delta": bool(positive), "authenticated": False},
        "access_control": {"subject_boundary_delta": bool(positive), "cross_subject_access": bool(positive)},
        "logic": {"invariant_delta": bool(positive), "history_changed": False},
        "url_redirect": {"origin_projection_delta": bool(positive), "same_origin": True, "external_redirect": False},
        "input_validation": {"validation_delta": bool(positive), "rejected": bool(positive)},
        "command_injection": {"local_canary_delta": bool(positive), "command_executed": False, "execution_blocked": True},
        "template_injection": {"template_context_delta": bool(positive), "render_executed": False},
        "ordinary_response": {"ordinary_response": True, "candidate_signal": False},
    }
    common.update(fields[family])
    return common


def _render(implementation: str, surface: str, variant: str, phase: str, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    base = _typed_fields(spec["family"], positive)
    if variant == "ledger":
        payload: dict[str, Any] = {
            "schema": PG42_SCHEMA,
            "ledger": {"implementation": implementation, "stage": phase, "accepted": True},
            "observation": {"signal": bool(positive), "kind": "bounded"},
            "typed": base,
        }
    elif variant == "envelope":
        payload = {
            "envelope": {"schema": PG42_SCHEMA, "implementation": implementation, "stage": phase},
            "payload": {"accepted": True, "observation": base},
            "meta": {"mode": variant, "bounded": True},
        }
    elif variant == "framed":
        payload = {
            "frame": [{"implementation": implementation, "stage": phase}, {"accepted": True}],
            "summary": {"signal": bool(positive), "kind": "bounded"},
            "audit": {"schema": PG42_SCHEMA, "mode": variant},
        }
    else:
        raise ValueError("unknown PG-42 surface variant")
    if positive:
        # The extra top-level fields create an observable but bounded delta;
        # they are not executable payloads and are not persisted as raw text.
        payload["effect"] = {"confirmed": True, "scope": spec["effect"]}
        payload["proof"] = {"bounded": True, "read_only": True}
    payload["ambiguous"] = phase == "screen"
    return payload


class PG42Handler(BaseHTTPRequestHandler):
    server_version = "PG42Independent/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve(self) -> None:
        implementation = str(getattr(self.server, "pg42_implementation", ""))
        layout = LAYOUTS.get(implementation)
        path = urlsplit(self.path).path
        surface = path.rsplit("/", 1)[-1] if layout and path.startswith(f"{layout['prefix']}/") else ""
        if layout is None or surface not in SURFACE_SPECS:
            self._write(404, {"ok": False, "bounded_error": "unknown_surface"})
            return
        values = _read_values(self, layout)
        spec = SURFACE_SPECS[surface]
        valid_variant = values["variant_key"] in VARIANTS
        candidate_signal = values["surface_key"] == surface and values["probe_key"] == spec["probe"] and valid_variant
        positive = candidate_signal and values["phase_key"] == "confirm" and spec["family"] != "ordinary_response"
        phase = values["phase_key"]
        if phase == "error":
            self._write(422, {"ok": False, "fault": {"class": "bounded_syntax", "recoverable": True}, "state_mutated": False, "external_network": False})
            return
        if phase == "timeout":
            self._write(429, {"ok": False, "fault": {"class": "bounded_timeout", "retryable": False}, "state_mutated": False, "external_network": False})
            return
        variant = values["variant_key"] if valid_variant else "ledger"
        self._write(200, _render(implementation, surface, variant, phase if phase in {"screen", "confirm"} else "unknown", positive))

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("x-pg42-surface", "bounded")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()


class PG42Server(ThreadingHTTPServer):
    def __init__(self, port: int, implementation: str) -> None:
        if implementation not in LAYOUTS:
            raise ValueError("unknown PG-42 implementation")
        super().__init__(("127.0.0.1", int(port)), PG42Handler)
        self.pg42_implementation = implementation


def make_pg42_server(port: int = 31960, implementation: str = "cobalt") -> PG42Server:
    return PG42Server(int(port), implementation)


__all__ = ["IMPLEMENTATIONS", "LAYOUTS", "PHASES", "PG42_SCHEMA", "SURFACE_SPECS", "VARIANTS", "make_pg42_server"]
