"""Loopback-only dynamic runtime for PG-348 synthetic fixture pages.

This is an evaluator harness, not a vulnerable public server.  It produces
dynamic GET/POST/redirect/state *shapes* from the frozen registry, keeps state
in memory, never echoes raw input, and never writes a database or filesystem.
Only the bounded projection may be used for model context.
"""

from __future__ import annotations

import hashlib
import html
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SCHEMA_VERSION = "pg348-dynamic-runtime-v1"
PROBE_VARIANTS = frozenset({"baseline_observe", "candidate_surface", "reference_surface", "negative_control", "unsupported_variant"})


def _bucket(value: int) -> str:
    return "zero" if value <= 0 else "one" if value == 1 else "two" if value == 2 else "few" if value <= 5 else "many"


@dataclass
class _State:
    reset_id: str
    events: list[str] = field(default_factory=list)


class DynamicFixtureApplication:
    def __init__(self, registry: Mapping[str, Any]) -> None:
        self.registry = registry
        self.records = {str(row.get("challenge_id")): dict(row) for row in list(registry.get("records") or [])}
        workspace_root = registry.get("_workspace_root")
        self.workspace_root = Path(str(workspace_root)).resolve() if workspace_root else None
        self._states: dict[str, _State] = {}
        self._lock = threading.Lock()

    def _render_body(self, record: Mapping[str, Any], *, method: str, present: bool, state_events: int, effect_confirmed: bool, failure_class: str) -> str:
        """Render the evaluator-only dynamic page in memory.

        The frozen HTML fixture is used only as a browser surface.  It never
        enters a trace/model row; the bounded adapter consumes it and drops
        the bytes.  A runtime marker makes the state transition observable
        without reflecting submitted values.
        """

        base = ""
        if self.workspace_root is not None:
            local_path = str(record.get("local_path", ""))
            candidate = (self.workspace_root / local_path).resolve()
            try:
                candidate.relative_to(self.workspace_root)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.is_file():
                base = candidate.read_text(encoding="utf-8", errors="replace")
        if not base:
            base = "<!doctype html><html lang=\"en\"><head><title>dynamic fixture</title></head><body><main></main></body></html>"
        marker = (
            "<section data-runtime=\"dynamic\" "
            f"data-method=\"{html.escape(method)}\" data-input=\"{'present' if present else 'absent'}\" "
            f"data-events=\"{_bucket(state_events)}\" data-effect=\"{'observed' if effect_confirmed else 'none'}\" "
            f"data-failure=\"{html.escape(failure_class)}\"><output>shape-observed</output></section>"
        )
        inline_script = "<script>document.documentElement.dataset.runtimeState='observed';</script>"
        lower = base.casefold()
        insertion = marker + inline_script
        if "</body>" in lower:
            index = lower.rfind("</body>")
            return base[:index] + insertion + base[index:]
        return base + insertion

    def reset(self, challenge_id: str) -> dict[str, Any]:
        if challenge_id not in self.records:
            raise KeyError("unknown_fixture")
        with self._lock:
            serial = len(self._states) + 1
            reset_id = hashlib.sha256(f"{challenge_id}:{serial}".encode("utf-8")).hexdigest()
            self._states[challenge_id] = _State(reset_id=reset_id)
        return {"schema_version": SCHEMA_VERSION, "challenge_id": challenge_id, "fresh_reset": True, "reset_id": reset_id, "state_clean": True, "external_network": False, "persistent_storage": False}

    def _state(self, challenge_id: str) -> _State:
        with self._lock:
            state = self._states.get(challenge_id)
        if state is None:
            self.reset(challenge_id)
            with self._lock:
                state = self._states[challenge_id]
        return state

    def handle(
        self,
        method: str,
        challenge_id: str,
        query: Mapping[str, list[str]] | None = None,
        form: Mapping[str, list[str]] | None = None,
        *,
        probe_variant: str = "baseline_observe",
    ) -> dict[str, Any]:
        if challenge_id not in self.records:
            raise KeyError("unknown_fixture")
        method = str(method).upper()
        if method not in {"GET", "POST"}:
            raise ValueError("method_not_allowlisted")
        probe_variant = str(probe_variant).casefold()
        if probe_variant not in PROBE_VARIANTS:
            raise ValueError("probe_variant_not_allowlisted")
        state = self._state(challenge_id)
        record = self.records[challenge_id]
        values = query if method == "GET" else form
        present = bool(values)
        if method == "POST":
            with self._lock:
                state.events.append("post_observed")
        effect_confirmed = probe_variant in {"candidate_surface", "reference_surface"}
        unsupported = probe_variant == "unsupported_variant"
        if effect_confirmed:
            with self._lock:
                state.events.append("typed_effect")
        response_shape = str(record.get("response_shape", "html_document:200"))
        redirect_shape = str(record.get("redirect_shape", "none"))
        is_redirect = redirect_shape not in {"none", "absent", "unknown", "not_observed"} or ":302" in response_shape
        status = 400 if unsupported else 302 if is_redirect else 200
        content_type = "application/json" if "json" in response_shape else "text/html"
        failure_class = "blocked_variant" if unsupported else "none"
        state_delta = "disposable_evaluator_state" if effect_confirmed else "event_count_changed" if method == "POST" else "none"
        # Dynamic response intentionally reports only bounded categories to
        # the collector; the rendered body is evaluator-side and never
        # serialized into model context.
        body = self._render_body(record, method=method, present=present, state_events=len(state.events), effect_confirmed=effect_confirmed, failure_class=failure_class)
        return {
            "schema_version": SCHEMA_VERSION,
            "challenge_id": challenge_id,
            "method": method,
            "status": status,
            "content_type": content_type,
            "body": body,
            "body_length": len(body),
            "redirect_shape": redirect_shape if is_redirect else "none",
            "input_presence": "present" if present else "absent",
            "state_delta": state_delta,
            "state_event_count": len(state.events),
            "probe_variant": probe_variant,
            "typed_effect_confirmed": effect_confirmed,
            "effect_class": "logic_transition" if effect_confirmed else "none",
            "failure_class": failure_class,
            "failure_stage": "probe_validation" if unsupported else "none",
            "error_shape": "blocked_variant" if unsupported else "empty",
            "repair_delta_axis": "probe_variant" if unsupported else "none",
            "reset_id": state.reset_id,
            "persistent_storage": False,
            "external_network": False,
        }


class _Handler(BaseHTTPRequestHandler):
    server_version = "PG348Loopback/1"

    def log_message(self, *_args: Any) -> None:
        return

    @property
    def application(self) -> DynamicFixtureApplication:
        return self.server.application  # type: ignore[attr-defined]

    def _send(self, result: Mapping[str, Any]) -> None:
        status = int(result.get("status", 200))
        body = str(result.get("body", "")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", str(result.get("content_type", "text/html")))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-PG348-Dynamic", "1")
        if status == 302:
            self.send_header("Location", "/pg348/dynamic/final")
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method: str, payload: Mapping[str, list[str]]) -> None:
        parts = [part for part in urlsplit(self.path).path.split("/") if part]
        if len(parts) < 3 or parts[0] != "pg348" or parts[1] != "dynamic":
            self.send_error(404)
            return
        probe_variant = self.headers.get("X-PG348-Probe-Variant", "baseline_observe")
        try:
            result = self.application.handle(method, parts[2], payload, payload, probe_variant=probe_variant)
        except (KeyError, ValueError):
            self.send_error(404)
            return
        self._send(result)

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET", parse_qs(urlsplit(self.path).query, keep_blank_values=True))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(min(length, 8192)).decode("utf-8", errors="replace")
        self._route("POST", parse_qs(raw, keep_blank_values=True))


def start_server(application: DynamicFixtureApplication, *, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("PG-348 dynamic runtime is loopback-only")
    server = ThreadingHTTPServer((host, int(port)), _Handler)
    server.application = application  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="pg348-loopback", daemon=True)
    thread.start()
    return server, thread


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        # Runtime-only provenance; collectors must not serialize this key.
        payload["_workspace_root"] = str(path.resolve().parents[2])
    return payload


__all__ = ["DynamicFixtureApplication", "LOOPBACK_HOSTS", "PROBE_VARIANTS", "SCHEMA_VERSION", "load_registry", "start_server"]
