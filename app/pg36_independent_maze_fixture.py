"""PG-36 independent maze fixture with delayed, ambiguous safe effects.

This is a separate ``http.server`` implementation from PG-33/34/35.  It
accepts only a bounded route slot, probe class and phase.  ``screen`` is an
ambiguous observation, ``confirm`` may produce a typed effect, and ``error`` /
``timeout`` are deterministic negative controls (no real sleep, subprocess,
network or state mutation).  The response is projection-friendly JSON and
never echoes the submitted identifier.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


PG36_SCHEMA = "sift-pg36-independent-maze-fixture-v1"
LAYOUTS = {
    "north": {
        "prefix": "/maze/north",
        "slot_key": "branch",
        "probe_key": "signal",
        "phase_key": "phase",
        "post_content_type": "application/json",
    },
    "south": {
        "prefix": "/lab/south",
        "slot_key": "channel",
        "probe_key": "class_id",
        "phase_key": "step",
        "post_content_type": "application/x-www-form-urlencoded",
    },
}

SURFACE_SPECS: dict[str, dict[str, Any]] = {
    "surface-01": {"family": "xss", "positive": "markup_candidate", "effect": "dom_structure", "modality": "typed_dom_effect", "probe_kind": "inert_dom_markup"},
    "surface-02": {"family": "injection", "positive": "operator_like", "effect": "interpreter_boundary", "modality": "typed_ast_difference", "probe_kind": "abstract_channel_class"},
    "surface-03": {"family": "authentication", "positive": "auth_boundary_candidate", "effect": "authentication_boundary", "modality": "typed_auth_boundary", "probe_kind": "abstract_channel_class"},
    "surface-04": {"family": "access_control", "positive": "id_reference", "effect": "authorization_boundary", "modality": "typed_authorization_boundary", "probe_kind": "abstract_channel_class"},
    "surface-05": {"family": "logic", "positive": "invariant_boundary", "effect": "business_invariant", "modality": "typed_logic_invariant", "probe_kind": "abstract_channel_class"},
    "surface-06": {"family": "url_redirect", "positive": "relative_redirect", "effect": "redirect_origin", "modality": "typed_redirect_projection", "probe_kind": "abstract_channel_class"},
    "surface-07": {"family": "input_validation", "positive": "boundary_value", "effect": "validation_boundary", "modality": "typed_validation_boundary", "probe_kind": "abstract_channel_class"},
    "surface-08": {"family": "command_injection", "positive": "local_canary", "effect": "command_canary", "modality": "typed_local_canary", "probe_kind": "http_canary"},
    "surface-09": {"family": "ordinary_response", "positive": "never_positive", "effect": "none", "modality": "negative_control", "probe_kind": "abstract_channel_class"},
    "surface-10": {"family": "unknown_surface", "positive": "never_positive", "effect": "none", "modality": "negative_control", "probe_kind": "abstract_channel_class"},
}
PHASES = ("screen", "confirm", "error", "timeout")


def _decode(value: Any) -> str:
    return unquote(str(value))[:96]


def _params(handler: BaseHTTPRequestHandler, layout: dict[str, Any]) -> tuple[str, str, str]:
    if handler.command == "GET":
        values = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        return (
            _decode(values.get(layout["slot_key"], [""])[0]),
            _decode(values.get(layout["probe_key"], [""])[0]),
            _decode(values.get(layout["phase_key"], [""])[0]),
        )
    try:
        length = min(max(int(handler.headers.get("content-length", "0")), 0), 2048)
    except (TypeError, ValueError):
        length = 0
    raw = handler.rfile.read(length)
    content_type = str(handler.headers.get("content-type", "")).split(";", 1)[0].casefold()
    if content_type == "application/json":
        try:
            value = json.loads(raw.decode("utf-8", errors="replace"))
        except (TypeError, ValueError, json.JSONDecodeError):
            value = {}
        values = value if isinstance(value, dict) else {}
    else:
        parsed = parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
        values = {key: item[0] for key, item in parsed.items() if item}
    return (
        _decode(values.get(layout["slot_key"], "")),
        _decode(values.get(layout["probe_key"], "")),
        _decode(values.get(layout["phase_key"], "")),
    )


def _effect(surface: str, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    family = spec["family"]
    common: dict[str, Any] = {
        "candidate_signal": bool(positive),
        "state_mutated": False,
        "database_touched": False,
        "credentials_accessed": False,
        "external_network": False,
        "script_execution": False,
    }
    if family == "xss":
        common.update({"dom_change": bool(positive), "marker_hits": int(positive)})
    elif family == "injection":
        common.update({"ast_shape_diff": bool(positive), "interpreter_boundary": bool(positive), "ast_node_delta": int(positive)})
    elif family == "authentication":
        common.update({"authentication_boundary": bool(positive), "authenticated": False, "state_class": "class_a"})
    elif family == "access_control":
        common.update({"authorization_boundary": bool(positive), "cross_subject_access": bool(positive), "subject_delta": int(positive)})
    elif family == "logic":
        common.update({"business_invariant_boundary": bool(positive), "history_changed": False, "counter_delta": int(positive)})
    elif family == "url_redirect":
        common.update({"redirect_candidate": bool(positive), "same_origin": True, "external_redirect": False})
    elif family == "input_validation":
        common.update({"validation_boundary": bool(positive), "rejected": bool(positive), "rejected_field_count": int(positive)})
    elif family == "command_injection":
        common.update({"local_canary_observed": bool(positive), "command_executed": False, "execution_blocked": True})
    else:
        common.update({"ordinary_response": True, "candidate_signal": False})
    return common


class PG36Handler(BaseHTTPRequestHandler):
    server_version = "PG36IndependentMaze/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve(self) -> None:
        implementation = str(getattr(self.server, "pg36_implementation", ""))
        layout = LAYOUTS.get(implementation)
        path = urlsplit(self.path).path
        surface = ""
        if layout and path.startswith(f"{layout['prefix']}/"):
            surface = path.rsplit("/", 1)[-1]
        if layout is None or surface not in SURFACE_SPECS:
            self._write(404, {"ok": False, "candidate_signal": False, "surface": "unknown"})
            return
        slot, probe, phase = _params(self, layout)
        spec = SURFACE_SPECS[surface]
        candidate = slot == surface and probe == spec["positive"]
        positive = candidate and phase == "confirm" and spec["family"] not in {"ordinary_response", "unknown_surface"}
        if phase == "error":
            self._write(400, {"ok": False, "error_class": "bounded_syntax", "candidate_signal": False, "state_mutated": False, "external_network": False})
            return
        if phase == "timeout":
            # Deterministic timeout-like negative; no real sleeping or socket
            # stall is performed, so collection remains bounded and local.
            self._write(504, {"ok": False, "timeout_class": "bounded_timeout", "candidate_signal": False, "state_mutated": False, "external_network": False})
            return
        payload: dict[str, Any] = {
            "fixture": PG36_SCHEMA,
            "implementation": implementation,
            "surface_slot": surface,
            "phase": phase if phase in {"screen", "confirm"} else "unknown",
            "ambiguous": phase == "screen",
            **_effect(surface, positive),
        }
        if positive:
            payload["bounded_response_delta"] = {"changed": True, "stage": "second_stage"}
            payload["typed_effect_ready"] = True
        self._write(200, payload)

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()


class PG36Server(ThreadingHTTPServer):
    def __init__(self, port: int, implementation: str) -> None:
        if implementation not in LAYOUTS:
            raise ValueError("unknown PG-36 implementation")
        super().__init__(("127.0.0.1", int(port)), PG36Handler)
        self.pg36_implementation = implementation


def make_pg36_server(port: int = 31936, implementation: str = "north") -> PG36Server:
    return PG36Server(int(port), implementation)


__all__ = ["LAYOUTS", "PG36_SCHEMA", "PHASES", "PG36Server", "SURFACE_SPECS", "make_pg36_server"]
