import pytest

from app.authorized_target_session import _validate_target


def test_loopback_docker_published_ports_are_authorized_without_static_port_entry():
    assert _validate_target("http://127.0.0.1:49173/") == "http://127.0.0.1:49173/"
    assert _validate_target("http://localhost:49201/challenge") == "http://localhost:49201/challenge"


def test_private_docker_address_still_requires_explicit_operator_allowlist(monkeypatch):
    monkeypatch.delenv("SIFT_AUTHORIZED_DOCKER_TARGETS", raising=False)
    with pytest.raises(ValueError, match="SIFT_AUTHORIZED_DOCKER_TARGETS"):
        _validate_target("http://172.17.0.2:8080/")


def test_explicit_private_docker_address_is_authorized(monkeypatch):
    monkeypatch.setenv("SIFT_AUTHORIZED_DOCKER_TARGETS", "http://172.17.0.2:8080")
    assert _validate_target("http://172.17.0.2:8080/") == "http://172.17.0.2:8080/"
