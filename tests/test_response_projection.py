from __future__ import annotations

from app.response_projection import ResponseProjection


def observation(path: str, content_type: str, content_length: str | None, body_length: int = 0):
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = content_length
    return {"action": {"method": "HEAD", "path": path}, "observation": {"status_code": 200, "headers": headers, "summary": {"body_length": body_length}}}


def test_projection_detects_generic_text_surface_without_route_name():
    projection = ResponseProjection.from_observation(observation("/arbitrary", "text/plain; version=0.0.4", None, 14000))
    assert projection.generic_text_surface is True
    assert projection.inferred_family() == "observability"
    assert projection.score() == 1.0
    assert projection.body_shape == "prometheus"


def test_projection_detects_html_listing_by_size_not_path():
    projection = ResponseProjection.from_observation(observation("/arbitrary", "text/html", "11322"))
    assert projection.generic_listing_surface is True
    assert projection.inferred_family() == "information_exposure"


def test_projection_rejects_spa_html_shell():
    projection = ResponseProjection.from_observation(observation("/arbitrary", "text/html", "9903"))
    assert projection.generic_listing_surface is False
    assert projection.score() == 0.0


def test_projection_does_not_call_security_text_an_operational_metrics_surface():
    projection = ResponseProjection.from_observation(observation("/arbitrary", "text/plain", "475"))
    assert projection.generic_text_surface is False
    assert projection.score() == 0.0
    assert projection.body_shape == "unknown"


def test_projection_uses_body_shape_slots_for_debug_and_spa():
    debug = ResponseProjection.from_observation({"observation": {"status_code": 200, "headers": {"content-type": "application/json"}, "summary": {"body_length": 300, "body_preview": '{"debug":true,"error":"diagnostic"}'}}})
    spa = ResponseProjection.from_observation(observation("/arbitrary", "text/html", "9903"))
    assert debug.body_shape == "diagnostic"
    assert spa.body_shape == "spa_shell"
    assert debug.feature_vector()[2] == 1.0
    assert spa.feature_vector()[5] == 1.0
