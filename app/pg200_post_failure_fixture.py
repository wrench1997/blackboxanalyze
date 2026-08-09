"""Unseen, local-only POST failure fixture for PG-200.

It emits bounded status/redirect shapes that are not present in PG-196's
training rows.  No credentials, state mutation, script execution, or external
redirect is allowed.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .maze_engine import sha256_json


PG200_POST_SCHEMA = "pg200-post-failure-fixture-v1"
PG200_POST_MODES = frozenset({"validation", "method_mismatch", "server_error", "redirect_loop"})
PG200_POST_PORTS = (8850, 8851, 8852)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _response(mode: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    if mode == "validation":
        return 422, {"ok": False, "failure": "validation_shape", "field_count": 1}, {}
    if mode == "method_mismatch":
        return 405, {"ok": False, "failure": "method_shape", "allow_class": "GET"}, {"Allow": "GET"}
    if mode == "server_error":
        return 500, {"ok": False, "failure": "bounded_server_shape", "retryable": False}, {}
    return 302, {"ok": False, "failure": "same_origin_redirect_shape"}, {"Location": "/post?mode=validation"}


class _PostFailureHandler(BaseHTTPRequestHandler):
    server_version = "sift-post-failure-v1"

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        length = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
        parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"), keep_blank_values=True)
        mode = str(parse_qs(parsed.query, keep_blank_values=True).get("mode", ["validation"])[0])
        mode = mode if mode in PG200_POST_MODES else "validation"
        status, body, headers = _response(mode)
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(405)
        self.send_header("Allow", "POST")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class PostFailureServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if int(address[1]) not in PG200_POST_PORTS:
            raise ValueError("PG-200 POST failure port is not allow-listed")
        super().__init__(address, _PostFailureHandler)
        self.fixture_variant = str(variant)


def make_post_failure_server(*, port: int = 8850, variant: str = "unseen") -> PostFailureServer:
    return PostFailureServer(("127.0.0.1", int(port)), variant)


def collect_post_failure(*, target: str, port: int, mode: str, sample_id: str) -> dict[str, Any]:
    mode = str(mode)
    if target.rstrip("/") != f"http://127.0.0.1:{port}" or port not in PG200_POST_PORTS or mode not in PG200_POST_MODES:
        raise ValueError("PG-200 POST failure target or mode is not allow-listed")
    payload = build_detection_payload(
        target=target,
        method="POST",
        path="/post",
        marker=f"pg200-{sample_id}",
        probe_kind="http_canary",
        probe="pg200-canary",
        form={"probe": "pg200-canary"},
        expected={"mode": mode},
    )
    with httpx.Client(base_url=target, timeout=5.0, follow_redirects=False) as client:
        response = client.post(f"/post?mode={mode}", data={"probe": "pg200-canary"})
    body = response.content
    projection = {
        "status_code": int(response.status_code),
        "status_class": f"{response.status_code // 100}xx",
        "body_length_bucket": "0" if not body else "1-255",
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "location_present": bool(response.headers.get("location")),
        "allow_present": bool(response.headers.get("allow")),
    }
    failure = {
        "schema_version": "pg200-unseen-post-failure-v1",
        "method": "POST",
        "mode": mode,
        "status_class": projection["status_class"],
        "redirect_present": projection["location_present"],
        "candidate_signal": False,
        "typed_available": False,
        "positive_authority": False,
    }
    envelope = {"schema": PG200_POST_SCHEMA, "target": target, "method": "POST", "mode": mode, "projection": projection, "failure": failure, "fresh_target": True, "external_network": False}
    return {
        "schema_version": PG200_POST_SCHEMA,
        "sample_id": sample_id,
        "target": target,
        "method": "POST",
        "mode": mode,
        "payload_sha256": payload["payload_sha256"],
        "response_projection": projection,
        "failure_signature": failure,
        "evidence_hash": sha256_json(envelope),
        "fresh_target": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "database_touched": False,
    }


__all__ = ["PG200_POST_MODES", "PG200_POST_PORTS", "collect_post_failure", "make_post_failure_server"]
