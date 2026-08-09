import pytest

from app.detection_payload import build_detection_payload, validate_detection_payload


def test_detection_payload_is_local_inert_and_non_executing():
    payload = build_detection_payload(
        path="/api/products",
        marker="sift-probe-01",
        probe="<span data-sift-marker=\"sift-probe-01\">sift-probe-01</span>",
        probe_kind="inert_dom_markup",
        expected={"status_class": "2xx", "body_shape": "json"},
    )
    assert payload["schema_version"] == "sift-detection-payload-v1"
    assert payload["method"] == "GET"
    assert payload["probe_kind"] == "inert_dom_markup"
    assert "sift-probe-01" in payload["probe"]
    assert payload["safety"]["does_not_execute"] is True
    assert payload["safety"]["no_external_network"] is True
    assert len(payload["payload_sha256"]) == 64


def test_detection_payload_rejects_scope_credentials_and_destructive_markers():
    with pytest.raises(ValueError):
        build_detection_payload(path="/api/products", target="https://example.test")
    with pytest.raises(ValueError):
        build_detection_payload(path="/api/products", headers={"authorization": "Bearer x"})
    with pytest.raises(ValueError):
        validate_detection_payload({"path": "/api/products", "marker": "<script>"})
    with pytest.raises(ValueError):
        build_detection_payload(path="/api/products", method="POST")
    with pytest.raises(ValueError):
        build_detection_payload(path="/api/products", probe="UNION SELECT", probe_kind="sql_channel_class")
    with pytest.raises(ValueError):
        build_detection_payload(path="/api/products", probe="raw", probe_kind="sql_channel_class")


def test_safe_post_requires_and_preserves_non_credential_form_fields():
    payload = build_detection_payload(
        target="http://127.0.0.1:8766",
        method="POST",
        path="/vul/xss/xss_stored.php",
        form={"message": "sift-safe-canary", "submit": "submit"},
    )
    assert payload["method"] == "POST"
    assert payload["form"]["message"] == "sift-safe-canary"
    with pytest.raises(ValueError):
        build_detection_payload(
            target="http://127.0.0.1:8766",
            method="POST",
            path="/vul/xss/xss_stored.php",
            form={"password": "not-allowed"},
        )
