from __future__ import annotations

import json
from urllib.request import Request, urlopen

from app.pg367_waf_runtime import start_runtime


def test_loopback_runtime_serves_html_and_get_post_projection() -> None:
    server, thread, origin = start_runtime()
    try:
        page = urlopen(f"{origin}/pg367/page/allow_baseline", timeout=2).read().decode("utf-8")
        assert "data-waf-policy" in page
        headers = {"X-PG367-Role": "candidate", "X-PG367-Syntax": "marker", "X-PG367-Encoding": "identity"}
        get_body = json.loads(urlopen(Request(f"{origin}/pg367/waf/allow_baseline?q=runtime-canary%27", headers=headers), timeout=2).read())
        assert get_body["projection"]["typed_effect_confirmed"] is True
        post = Request(f"{origin}/pg367/waf/allow_baseline", data=b"q=runtime-canary%27", method="POST", headers=headers)
        post_body = json.loads(urlopen(post, timeout=2).read())
        assert post_body["projection"]["method"] == "POST"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_filter_projection_has_failure_without_raw_response() -> None:
    server, thread, origin = start_runtime()
    try:
        headers = {"X-PG367-Role": "candidate", "X-PG367-Syntax": "marker", "X-PG367-Encoding": "identity"}
        body = json.loads(urlopen(Request(f"{origin}/pg367/waf/delimiter_normalizer?q=runtime-canary%27", headers=headers), timeout=2).read())
        projection = body["projection"]
        assert projection["failure_signature"] == "encoded_delimiter"
        assert projection["raw_payload_stored"] is False
        assert projection["raw_response_stored"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
