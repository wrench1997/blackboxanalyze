"""PG-35's independent, read-only GET/POST fixture.

The fixture is intentionally unlike the FastAPI maze and the PG-34 fixture:
it uses a small ``http.server`` implementation, three route/field layouts,
and a bounded abstract probe vocabulary.  It never evaluates markup, SQL,
commands, redirects, credentials, or arbitrary input.  ``identity`` and
``url_percent`` are transport encodings of the same inert identifier so the
collector can test encoding invariance without retaining an attack string.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


PG35_SCHEMA = "sift-pg35-independent-http-fixture-v1"
PG35_VARIANTS = {
    "alpha": {
        "prefix": "/pg35/a",
        "slot_key": "surface",
        "probe_key": "probe_class",
        "post_content_type": "application/x-www-form-urlencoded",
    },
    "beta": {
        "prefix": "/observe/b",
        "slot_key": "channel",
        "probe_key": "signal_class",
        "post_content_type": "application/json",
    },
    "gamma": {
        "prefix": "/check/c",
        "slot_key": "route_slot",
        "probe_key": "candidate_class",
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
}


def _decode_value(value: Any) -> str:
    # ``parse_qs`` has already decoded one form/query layer.  JSON values may
    # still carry a percent-encoded inert identifier, so one explicit decode
    # makes both transports semantically equivalent without accepting code.
    return unquote(str(value))[:96]


def _bounded_params(handler: BaseHTTPRequestHandler, variant: str) -> tuple[str, str]:
    layout = PG35_VARIANTS[variant]
    if handler.command == "GET":
        values = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        slot = _decode_value(values.get(layout["slot_key"], [""])[0])
        probe = _decode_value(values.get(layout["probe_key"], [""])[0])
        return slot, probe
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
        values = {key: value[0] for key, value in parsed.items() if value}
    slot = _decode_value(values.get(layout["slot_key"], ""))
    probe = _decode_value(values.get(layout["probe_key"], ""))
    return slot, probe


def _typed_effect(surface_id: str, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface_id]
    family = spec["family"]
    common: dict[str, Any] = {
        "surface_slot": surface_id,
        "candidate_signal": bool(positive),
        "state_mutated": False,
        "database_touched": False,
        "credentials_accessed": False,
        "external_network": False,
        "script_execution": False,
    }
    if family == "xss":
        common.update({"dom_change": bool(positive), "marker_hits": int(positive), "network_access": False})
    elif family == "injection":
        common.update({"ast_shape_diff": bool(positive), "interpreter_boundary": bool(positive), "ast_node_delta": int(positive), "branch_delta": int(positive)})
    elif family == "authentication":
        common.update({"authentication_boundary": bool(positive), "authenticated": False, "state_class": "class_a"})
    elif family == "access_control":
        common.update({"authorization_boundary": bool(positive), "cross_subject_access": bool(positive), "subject_delta": int(positive), "subject_vector": [int(positive)]})
    elif family == "logic":
        common.update({"business_invariant_boundary": bool(positive), "history_changed": False, "history_depth_delta": int(positive), "counter_delta": int(positive), "history_vector": [int(positive), 0]})
    elif family == "url_redirect":
        common.update({"redirect_candidate": bool(positive), "same_origin": True, "external_redirect": False, "origin_class": "same_origin"})
    elif family == "input_validation":
        common.update({"validation_boundary": bool(positive), "rejected": bool(positive), "rejected_field_count": int(positive), "boundary_class": "class_b"})
    elif family == "command_injection":
        common.update({"local_canary_observed": bool(positive), "command_executed": False, "execution_blocked": True, "canary_count": int(positive)})
    else:
        common.update({"ordinary_response": True, "candidate_signal": False})
    return common


class PG35Handler(BaseHTTPRequestHandler):
    server_version = "PG35IndependentFixture/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Never log request values or response content.
        return

    def _serve(self) -> None:
        path = urlsplit(self.path).path
        variant = str(getattr(self.server, "pg35_variant", ""))
        layout = PG35_VARIANTS.get(variant)
        surface_id = ""
        if layout and path.startswith(f"{layout['prefix']}/"):
            surface_id = path.rsplit("/", 1)[-1]
        if not layout or surface_id not in SURFACE_SPECS:
            self._write(404, {"ok": False, "surface": "unknown", "candidate_signal": False})
            return
        slot, probe = _bounded_params(self, variant)
        spec = SURFACE_SPECS[surface_id]
        positive = probe == spec["positive"] and slot == surface_id
        payload = {
            "fixture": PG35_SCHEMA,
            "variant": variant,
            "surface_slot": surface_id,
            "candidate_signal": bool(positive),
            **_typed_effect(surface_id, positive),
        }
        # A generic, bounded response differential is the visible observation
        # available before typed-oracle interpretation.  It deliberately does
        # not name the family or expose evaluator state; it only lets a model
        # learn that candidate and control projections are not identical.
        if positive:
            payload["bounded_response_delta"] = {"changed": True}
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


class PG35Server(ThreadingHTTPServer):
    def __init__(self, port: int, variant: str) -> None:
        if variant not in PG35_VARIANTS:
            raise ValueError("unknown PG-35 fixture variant")
        super().__init__(("127.0.0.1", int(port)), PG35Handler)
        self.pg35_variant = variant


def make_pg35_server(port: int = 31935, variant: str = "alpha") -> PG35Server:
    return PG35Server(int(port), variant)


__all__ = ["PG35_SCHEMA", "PG35_VARIANTS", "PG35Server", "SURFACE_SPECS", "make_pg35_server"]
