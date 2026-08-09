from __future__ import annotations

import json

import pytest

import app.pg332_dvwa_docker_relay as relay


def test_attest_uses_fixed_image_network_none_and_no_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "mounts": "[]",
        "ports": "{}",
        "image": relay.IMAGE,
        "network": "none",
    }

    def fake_docker(*args: str, **kwargs: object) -> str:
        if "{{json .Mounts}}" in args:
            return values["mounts"]
        if "{{json .NetworkSettings.Ports}}" in args:
            return values["ports"]
        if "{{.Config.Image}}" in args:
            return values["image"]
        if "{{.HostConfig.NetworkMode}}" in args:
            return values["network"]
        raise AssertionError(args)

    monkeypatch.setattr(relay, "_docker", fake_docker)
    result = relay._attest("pg332-dvwa-test", "container-id")
    assert result["fresh_reset"] is True
    assert result["network_mode"] == "none"
    assert result["external_network"] is False
    assert result["published_port_count"] == 0
    assert result["bind_or_volume_mount_count"] == 0
    assert len(result["target_instance_digest"]) == 64


def test_attest_rejects_mount_or_published_port(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter((json.dumps([{"Type": "volume"}]), json.dumps({"80/tcp": [{"HostPort": "1234"}]}), relay.IMAGE, "none"))
    monkeypatch.setattr(relay, "_docker", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="attestation"):
        relay._attest("pg332-dvwa-test", "container-id")


class _FakeBridge:
    def __init__(self, *, authenticated: bool = True) -> None:
        self.calls: list[tuple[str, str, bytes]] = []
        self.authenticated = authenticated

    def request(self, method: str, path: str, *, body: bytes = b"", headers: object = None) -> dict[str, object]:
        self.calls.append((method, path, body))
        if method == "GET" and path == "/setup.php":
            return {"status": 200, "headers": {}, "body": b'<input name="create_db" value="Create / Reset Database"><input name="user_token" value="abcdef">'}
        if method == "POST" and path == "/setup.php":
            return {"status": 302, "headers": {"location": "/setup.php"}, "body": b""}
        if method == "GET" and path == "/login.php":
            return {"status": 200, "headers": {}, "body": b'<input name="user_token" value="abcdef">'}
        if method == "POST" and path == "/login.php":
            return {"status": 302, "headers": {"location": "/"}, "body": b""}
        if method == "GET" and path == "/":
            body = b'<a href="logout.php">Logout</a>' if self.authenticated else b'<input name="username"><input name="password">'
            return {"status": 200, "headers": {}, "body": body}
        raise AssertionError((method, path))


def test_fresh_database_setup_and_authenticated_health_gate_are_evaluator_only() -> None:
    target = object.__new__(relay.DisposableDvwa)
    bridge = _FakeBridge()
    target.bridge = bridge
    target._initialize_database()
    target._login(b'<input name="user_token" value="abcdef">')
    target._assert_authenticated()
    assert [call[:2] for call in bridge.calls] == [
        ("GET", "/setup.php"),
        ("POST", "/setup.php"),
        ("POST", "/login.php"),
        ("GET", "/"),
    ]
    assert b"vulnerables" not in b"".join(call[2] for call in bridge.calls)


def test_authentication_gate_rejects_login_form() -> None:
    target = object.__new__(relay.DisposableDvwa)
    target.bridge = _FakeBridge(authenticated=False)
    with pytest.raises(RuntimeError, match="authentication health"):
        target._assert_authenticated()
