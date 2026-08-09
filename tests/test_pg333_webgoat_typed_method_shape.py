from __future__ import annotations

from scripts.run_pg333_webgoat_typed_get_post_source_rows import (
    _abstract_projection,
    _action_method,
    _role_target,
    _typed_effect,
)


def test_webgoat_method_shape_oracle_is_not_a_vulnerability_oracle():
    get_action = {"status": 200, "status_class": "2xx", "content_type_class": "text/html", "location_class": "none"}
    post_action = {"status": 302, "status_class": "3xx", "content_type_class": "unknown", "location_class": "loopback"}
    assert _typed_effect(expected_method="GET", action_method="GET", action=get_action, body=b"<html>" + (b"page" * 30) + b"</html>") is True
    assert _typed_effect(expected_method="GET", action_method="POST", action=post_action, body=b"") is False
    assert _typed_effect(expected_method="POST", action_method="POST", action=post_action, body=b"") is True
    assert _typed_effect(expected_method="POST", action_method="GET", action=get_action, body=b"<html>" + (b"page" * 30) + b"</html>") is False
    projection, effect_class = _abstract_projection(action=post_action, body=b"", typed=True, expected_method="POST")
    assert effect_class == "redirect_hop"
    assert projection["database_touched"] is False
    assert projection["external_network_blocked"] is True


def test_webgoat_target_abstains_when_manifest_is_incomplete():
    assert _action_method("GET", "negative") == "POST"
    assert _action_method("POST", "negative") == "GET"
    target = _role_target(expected_method="POST", action_method="POST", role="candidate", complete=False)
    assert target["question"] == "ask_typed"
    assert target["next_action"] == "ask_typed"
    assert target["safe_to_send"] is False
