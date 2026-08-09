"""PG-37 local counterfactual fixture.

Each safe abstract probe is replayed through several response surfaces.  The
typed effect is held constant within a vulnerability family while the
projection shape changes (compact, nested, or headerized).  This lets the
experiment measure whether a Rule IR learner uses an invariant or memorizes a
surface.  The fixture is loopback-only, read-only, and never executes a
script, command, database write, redirect, or credential flow.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


PG37_SCHEMA = "sift-pg37-counterfactual-fixture-v1"
LAYOUTS = {
    "atlas": {
        "prefix": "/counterfactual/atlas",
        "slot_key": "branch",
        "probe_key": "signal",
        "variant_key": "surface",
        "phase_key": "phase",
        "post_content_type": "application/json",
    },
    "orbit": {
        "prefix": "/lab/orbit/cf",
        "slot_key": "channel",
        "probe_key": "class_id",
        "variant_key": "view",
        "phase_key": "step",
        "post_content_type": "application/x-www-form-urlencoded",
    },
}

VARIANTS = ("compact", "nested", "headerized")
PHASES = ("screen", "confirm", "error", "timeout")

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


def _decode(value: Any) -> str:
    return unquote(str(value))[:96]


def _params(handler: BaseHTTPRequestHandler, layout: dict[str, Any]) -> tuple[str, str, str, str]:
    if handler.command == "GET":
        values = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        return tuple(_decode(values.get(layout[key], [""])[0]) for key in ("slot_key", "probe_key", "variant_key", "phase_key"))  # type: ignore[return-value]
    try:
        length = min(max(int(handler.headers.get("content-length", "0")), 0), 2048)
    except (TypeError, ValueError):
        length = 0
    raw = handler.rfile.read(length)
    content_type = str(handler.headers.get("content-type", "")).split(";", 1)[0].casefold()
    if content_type == "application/json":
        try:
            values = json.loads(raw.decode("utf-8", errors="replace"))
        except (TypeError, ValueError, json.JSONDecodeError):
            values = {}
        values = values if isinstance(values, dict) else {}
    else:
        parsed = parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
        values = {key: item[0] for key, item in parsed.items() if item}
    return tuple(_decode(values.get(layout[key], "")) for key in ("slot_key", "probe_key", "variant_key", "phase_key"))  # type: ignore[return-value]


def _effect(family: str, positive: bool) -> dict[str, Any]:
    common: dict[str, Any] = {"candidate_signal": bool(positive), "state_mutated": False, "database_touched": False, "credentials_accessed": False, "external_network": False, "script_execution": False}
    fields = {
        "xss": {"dom_change": bool(positive), "marker_hits": int(positive)},
        "injection": {"ast_shape_diff": bool(positive), "interpreter_boundary": bool(positive), "ast_node_delta": int(positive)},
        "authentication": {"authentication_boundary": bool(positive), "authenticated": False, "state_class": "class_a"},
        "access_control": {"authorization_boundary": bool(positive), "cross_subject_access": bool(positive), "subject_delta": int(positive)},
        "logic": {"business_invariant_boundary": bool(positive), "history_changed": False, "counter_delta": int(positive)},
        "url_redirect": {"redirect_candidate": bool(positive), "same_origin": True, "external_redirect": False},
        "input_validation": {"validation_boundary": bool(positive), "rejected": bool(positive), "rejected_field_count": int(positive)},
        "command_injection": {"local_canary_observed": bool(positive), "command_executed": False, "execution_blocked": True},
        "ordinary_response": {"ordinary_response": True, "candidate_signal": False},
        "unknown_surface": {"ordinary_response": True, "candidate_signal": False},
    }
    common.update(fields[family])
    return common


def _surface_payload(implementation: str, surface: str, variant: str, phase: str, positive: bool) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    base = _effect(spec["family"], positive)
    # These are intentionally generic shapes.  The family-specific typed
    # effect remains in the evaluator projection, not in the learner input.
    if variant == "compact":
        payload: dict[str, Any] = {"fixture": PG37_SCHEMA, "impl": implementation, "view": variant, "stage": phase, **base}
    elif variant == "nested":
        payload = {"meta": {"fixture": PG37_SCHEMA, "impl": implementation, "view": variant}, "data": {"stage": phase, **base}}
    elif variant == "headerized":
        payload = {"fixture": PG37_SCHEMA, "header_profile": "bounded", "impl": implementation, "view": variant, "stage": phase, "vector": [int(positive), 0, 1], **base}
    else:
        raise ValueError("unknown PG-37 surface variant")
    if positive:
        payload["bounded_response_delta"] = {"changed": True, "stage": "second_stage", "variant": variant}
        payload["typed_effect_ready"] = True
    return payload


class PG37Handler(BaseHTTPRequestHandler):
    server_version = "PG37Counterfactual/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve(self) -> None:
        implementation = str(getattr(self.server, "pg37_implementation", ""))
        layout = LAYOUTS.get(implementation)
        path = urlsplit(self.path).path
        surface = path.rsplit("/", 1)[-1] if layout and path.startswith(f"{layout['prefix']}/") else ""
        if layout is None or surface not in SURFACE_SPECS:
            self._write(404, {"ok": False, "candidate_signal": False, "surface": "unknown"})
            return
        slot, probe, variant, phase = _params(self, layout)
        spec = SURFACE_SPECS[surface]
        candidate = slot == surface and variant in VARIANTS and probe == spec["positive"]
        positive = candidate and phase == "confirm" and spec["family"] not in {"ordinary_response", "unknown_surface"}
        if phase == "error":
            self._write(400, {"ok": False, "error_class": "bounded_syntax", "candidate_signal": False, "state_mutated": False, "external_network": False})
            return
        if phase == "timeout":
            self._write(504, {"ok": False, "timeout_class": "bounded_timeout", "candidate_signal": False, "state_mutated": False, "external_network": False})
            return
        payload = _surface_payload(implementation, surface, variant if variant in VARIANTS else "compact", phase if phase in {"screen", "confirm"} else "unknown", positive)
        payload["ambiguous"] = phase == "screen"
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


class PG37Server(ThreadingHTTPServer):
    def __init__(self, port: int, implementation: str) -> None:
        if implementation not in LAYOUTS:
            raise ValueError("unknown PG-37 implementation")
        super().__init__(("127.0.0.1", int(port)), PG37Handler)
        self.pg37_implementation = implementation


def make_pg37_server(port: int = 31937, implementation: str = "atlas") -> PG37Server:
    return PG37Server(int(port), implementation)


__all__ = ["LAYOUTS", "PHASES", "PG37_SCHEMA", "SURFACE_SPECS", "VARIANTS", "make_pg37_server"]
