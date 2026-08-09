"""PG-146 real loopback replay for pinned public vulnerable labs.

This collector is intentionally a surface/transport baseline.  It reads the
HTTP body in memory to derive bounded HTML/response-shape features, then keeps
only projections and evidence hashes.  No raw request body, response body, or
operational exploit string is serialized.  Since this phase has no typed
vulnerability oracle, every row remains ``unknown_oracle`` and is evaluation
only.
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pg146-public-lab-replay-v1"
RUN_ID = "pg146"
MAX_BODY_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15
READINESS_TIMEOUT_SECONDS = 90


@dataclass(frozen=True)
class PublicLabTarget:
    target_id: str
    lab: str
    container: str
    image_digest: str
    base_url: str
    get_path: str
    post_path: str
    post_content_type: str
    post_body: bytes

    @property
    def get_url(self) -> str:
        return self.base_url.rstrip("/") + self.get_path

    @property
    def post_url(self) -> str:
        return self.base_url.rstrip("/") + self.post_path


TARGETS: tuple[PublicLabTarget, ...] = (
    PublicLabTarget(
        target_id="pg146_webgoat_login_surface",
        lab="webgoat",
        container="sift-pg146-webgoat",
        image_digest="webgoat/webgoat@sha256:3101bd9e7bcfe122d7ef91e690ef3720de36cc4e86b3d06763a1ddf2e2751a4b",
        base_url="http://127.0.0.1:31450",
        get_path="/WebGoat/login",
        post_path="/WebGoat/login",
        post_content_type="application/x-www-form-urlencoded",
        post_body=b"",
    ),
    PublicLabTarget(
        target_id="pg146_dvwa_login_surface",
        lab="dvwa",
        container="sift-pg146-dvwa",
        image_digest="vulnerables/web-dvwa@sha256:dae203fe11646a86937bf04db0079adef295f426da68a92b40e3b181f337daa7",
        base_url="http://127.0.0.1:31452",
        get_path="/login.php",
        post_path="/login.php",
        post_content_type="application/x-www-form-urlencoded",
        post_body=b"",
    ),
    PublicLabTarget(
        target_id="pg146_juice_shop_login_surface",
        lab="juice_shop",
        container="sift-pg146-juice",
        image_digest="bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe",
        base_url="http://127.0.0.1:31453",
        get_path="/",
        post_path="/rest/user/login",
        post_content_type="application/json",
        post_body=b'{"email":"canary@example.invalid","password":"canary"}',
    ),
)


class _HTMLShapeParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tag_counts: dict[str, int] = {}
        self.attribute_count = 0
        self.text_length = 0
        self.title_text = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = str(tag).casefold()
        self.tag_counts[normalized] = self.tag_counts.get(normalized, 0) + 1
        self.attribute_count += len(attrs)
        self._in_title = normalized == "title"

    def handle_endtag(self, tag: str) -> None:
        if str(tag).casefold() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        self.text_length += len(data)
        if self._in_title:
            self.title_text += data


def _bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    if length <= 65535:
        return "4096-65535"
    return "65536+"


def _status_class(status: int | None) -> str:
    return f"{status // 100}xx" if status is not None and status > 0 else "transport_error"


def _origin(location: str | None) -> str:
    if not location:
        return "none"
    parsed = urllib.parse.urlparse(location)
    if not parsed.netloc:
        return "relative"
    return "loopback" if parsed.hostname in {"127.0.0.1", "localhost", "::1"} else "external_or_unknown"


def _body_projection(body: bytes, content_type: str) -> dict[str, Any]:
    limited = body[:MAX_BODY_BYTES]
    projection: dict[str, Any] = {
        "body_length": len(body),
        "body_length_bucket": _bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": hashlib.sha256(re.sub(rb"\s+", b" ", limited)).hexdigest(),
        "content_type_class": content_type.split(";", 1)[0].casefold() if content_type else "unknown",
        "html_shape": {
            "tag_count": 0,
            "form_count": 0,
            "input_count": 0,
            "script_count": 0,
            "link_count": 0,
            "attribute_count": 0,
            "text_length_bucket": "0",
            "title_present": False,
        },
    }
    if "html" not in projection["content_type_class"]:
        return projection
    parser = _HTMLShapeParser()
    try:
        parser.feed(limited.decode("utf-8", errors="replace"))
    except Exception:
        projection["html_parse_error"] = True
        return projection
    projection["html_shape"] = {
        "tag_count": int(sum(parser.tag_counts.values())),
        "form_count": int(parser.tag_counts.get("form", 0)),
        "input_count": int(parser.tag_counts.get("input", 0)),
        "script_count": int(parser.tag_counts.get("script", 0)),
        "link_count": int(parser.tag_counts.get("link", 0)),
        "attribute_count": int(parser.attribute_count),
        "text_length_bucket": _bucket(parser.text_length),
        "title_present": bool(parser.title_text.strip()),
    }
    return projection


def _docker_label_value(container: str, label: str) -> str:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{index .Config.Labels \"" + label + "\"}}", container],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip()


def assert_target_scope(target: PublicLabTarget) -> None:
    parsed = urllib.parse.urlparse(target.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("PG-146 accepts loopback HTTP targets only")
    if _docker_label_value(target.container, "com.openai.sift.run_id") != RUN_ID:
        raise ValueError(f"container {target.container} is not labeled for {RUN_ID}")
    if _docker_label_value(target.container, "com.openai.sift.target") != target.lab.replace("_", "-") and target.lab != "juice_shop":
        # Juice Shop label is intentionally kept as the public lab name.
        if _docker_label_value(target.container, "com.openai.sift.target") != "juice-shop":
            raise ValueError(f"container {target.container} has an unexpected target label")


def restart_target(target: PublicLabTarget) -> None:
    assert_target_scope(target)
    subprocess.run(["docker", "restart", target.container], check=True, capture_output=True, text=True, timeout=60)


def _request(method: str, url: str, *, content_type: str | None = None, body: bytes = b"") -> dict[str, Any]:
    request = urllib.request.Request(url, data=body if method == "POST" else None, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_body = response.read(MAX_BODY_BYTES + 1)
            truncated = len(response_body) > MAX_BODY_BYTES
            if truncated:
                response_body = response_body[:MAX_BODY_BYTES]
            headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
            projection = _body_projection(response_body, headers.get("content-type", ""))
            return {
                "status_code": int(response.status),
                "status_class": _status_class(int(response.status)),
                "headers_present": sorted(set(headers) & {"content-type", "location", "set-cookie", "allow"}),
                "location_origin": _origin(headers.get("location")),
                "truncated": truncated,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "transport_error": False,
                "projection": projection,
            }
    except urllib.error.HTTPError as exc:
        response_body = exc.read(MAX_BODY_BYTES + 1)
        truncated = len(response_body) > MAX_BODY_BYTES
        if truncated:
            response_body = response_body[:MAX_BODY_BYTES]
        headers = {str(key).casefold(): str(value) for key, value in exc.headers.items()} if exc.headers else {}
        return {
            "status_code": int(exc.code),
            "status_class": _status_class(int(exc.code)),
            "headers_present": sorted(set(headers) & {"content-type", "location", "set-cookie", "allow"}),
            "location_origin": _origin(headers.get("location")),
            "truncated": truncated,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "transport_error": False,
            "projection": _body_projection(response_body, headers.get("content-type", "")),
        }
    except Exception as exc:
        return {
            "status_code": None,
            "status_class": "transport_error",
            "headers_present": [],
            "location_origin": "none",
            "truncated": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "transport_error": True,
            "transport_error_type": type(exc).__name__,
            "projection": _body_projection(b"", ""),
        }


def readiness_url(target: PublicLabTarget) -> str:
    return target.get_url


def wait_ready(target: PublicLabTarget, *, timeout_seconds: int = READINESS_TIMEOUT_SECONDS) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        last = _request("GET", readiness_url(target))
        if not last["transport_error"] and last.get("status_code") in {200, 301, 302, 303, 307, 308, 401, 403}:
            return {"ready": True, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3), "probe": last}
        time.sleep(1.0)
    return {"ready": False, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3), "probe": last}


def _failure_signature(response: Mapping[str, Any]) -> dict[str, Any]:
    projection = dict(response.get("projection") or {})
    shape = dict(projection.get("html_shape") or {})
    return {
        "status_class": response.get("status_class", "unknown"),
        "transport_error": bool(response.get("transport_error", False)),
        "location_origin": response.get("location_origin", "none"),
        "content_type_class": projection.get("content_type_class", "unknown"),
        "body_length_bucket": projection.get("body_length_bucket", "0"),
        "html_tag_count": shape.get("tag_count", 0),
        "form_count": shape.get("form_count", 0),
        "input_count": shape.get("input_count", 0),
        "script_count": shape.get("script_count", 0),
    }


def collect_target(target: PublicLabTarget) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actions = (("GET", target.get_path, target.get_url, None, b""), ("POST", target.post_path, target.post_url, target.post_content_type, target.post_body))
    for method, path, url, content_type, body in actions:
        restart_target(target)
        readiness = wait_ready(target)
        response = _request(method, url, content_type=content_type, body=body)
        request_digest = hashlib.sha256(body).hexdigest()
        evidence = {
            "request_body_sha256": request_digest,
            "response_body_sha256": response["projection"]["body_sha256"],
            "container_image_digest": target.image_digest,
        }
        evidence_hash = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        rows.append(
            {
                "row_id": f"{target.target_id}::{method}",
                "target_id": target.target_id,
                "lab": target.lab,
                "method": method,
                "route_class": "login_surface",
                "fresh_reset": True,
                "readiness": readiness,
                "response": response,
                "failure_signature": _failure_signature(response),
                "oracle": {
                    "availability": "unknown_oracle",
                    "typed_effect": "not_observed",
                    "confirmed_positive": False,
                    "matched_negative": False,
                },
                "evidence_hash": evidence_hash,
                "model_projection": {
                    "tokens": [
                        "[BOS]",
                        "[STEP]",
                        f"transport.method={method}",
                        "transport.channel=loopback",
                        "surface.route_class=login_surface",
                        f"obs.status_class={response['status_class']}",
                        f"obs.content_type={response['projection']['content_type_class']}",
                        f"obs.body_length={response['projection']['body_length_bucket']}",
                        f"obs.failure.transport={str(bool(response['transport_error'])).lower()}",
                        "obs.oracle=unknown_oracle",
                        "[EOS]",
                    ],
                    "raw_request_body_in_model": False,
                    "raw_response_body_in_model": False,
                    "target_identity_in_model": False,
                    "oracle_authority_in_model": False,
                },
            }
        )
    return rows


__all__ = ["PublicLabTarget", "SCHEMA_VERSION", "TARGETS", "collect_target", "readiness_url", "wait_ready"]
