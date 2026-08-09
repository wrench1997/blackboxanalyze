from __future__ import annotations

from scripts.run_pg332_dvwa_typed_stored_post_source_rows import (
    _page_probe,
    _post_probe,
    _probe_value,
    _state_delta,
)
from scripts.run_pg332_dvwa_source_rows_live import _route_request


class _FakeTarget:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes]] = []

    def request(self, method: str, path: str, *, body: bytes = b"", content_type: str = "application/x-www-form-urlencoded"):
        self.calls.append((method, path, body))
        return {"status": 200, "headers": {"content-type": "text/html; charset=UTF-8"}, "body": b"<html><body>fixture</body></html>"}


def test_state_delta_is_typed_only_for_raw_marker_and_not_negative() -> None:
    candidate = _probe_value(33410, "candidate")
    baseline = _page_probe(candidate)(b"<html></html>", 200, {"content-type": "text/html"})
    after = _page_probe(candidate)((b"<html>" + candidate.encode() + b"</html>"), 200, {"content-type": "text/html"})
    posted = _post_probe(b"", 302, {"location": "/vulnerabilities/xss_s/"})
    state = _state_delta(role="candidate", marker=candidate, baseline=baseline, posted=posted, after=after)
    assert state["typed_effect"] is True
    assert state["disposable_state_delta"] is True
    negative = _probe_value(33410, "negative")
    negative_after = _page_probe(negative)(b"<html>plain-33410</html>", 200, {"content-type": "text/html"})
    negative_state = _state_delta(role="negative", marker=negative, baseline=baseline, posted=posted, after=negative_after)
    assert negative_state["typed_effect"] is False
    assert negative_state["negative_control_clean"] is True


def test_route_request_marks_post_as_supported_without_serializing_wire() -> None:
    target = _FakeTarget()
    route = {"path": "/vulnerabilities/xss_s/", "field": "txtName", "method": "POST"}
    capture, shape, effect = _route_request(
        target,
        route,
        probe_values={"txtName": "abstract-marker", "mtxMessage": "abstract-marker", "btnSign": "Sign Guestbook"},
        post_supported=True,
        effect_probe=_post_probe,
    )
    assert target.calls[0][0] == "POST"
    assert capture["typed_projection"]["post_supported"] is True
    assert capture["typed_projection"]["post_unavailable"] is False
    assert shape["content_type_class"] == "html"
    assert effect["status_ok"] is True
    assert "abstract-marker" not in capture.__repr__()
