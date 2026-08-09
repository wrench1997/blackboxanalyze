import json
import subprocess

import pytest

import app.authorized_remote_docker as remote
from app.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_remote_probe_uses_fixed_read_only_commands(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), dict(kwargs)))
        if args[-3:] == ["command", "-v", "docker"]:
            return subprocess.CompletedProcess(args, 0, stdout="/usr/bin/docker\n", stderr="")
        if args[-3:] == ["docker", "version", "--format={{json .Server}}"]:
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"Version": "27.0"}), stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="sift-target\n", stderr="")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    result = remote.probe_authorized_remote_docker(remote.RemoteDockerConfig())
    assert result["status"] == "available"
    assert result["docker_server"] is True
    assert result["running_containers"] == ["sift-target"]
    assert result["scope"]["mutating_docker_commands_allowed"] is False
    assert len(calls) == 3
    assert all(not ("docker" in call[0] and "run" in call[0]) for call in calls)
    assert all(call[1].get("shell", False) is not True for call in calls)


def test_remote_probe_unavailable_is_not_real_gold(monkeypatch):
    def fake_run(args, **kwargs):
        if args[-3:] == ["command", "-v", "docker"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 127, stdout="", stderr="docker: not found")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    result = remote.probe_authorized_remote_docker(remote.RemoteDockerConfig())
    assert result["status"] == "unavailable"
    assert result["docker_server"] is False
    assert result["running_containers"] == []
    assert result["real_application_gold_rows"] == 0
    assert result["training_or_replay_started"] is False


def test_container_name_requires_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("SIFT_REMOTE_DOCKER_TARGET_CONTAINERS", "sift-target")
    assert remote.validate_container_name("sift-target") == "sift-target"
    with pytest.raises(ValueError, match="allowlist"):
        remote.validate_container_name("other-target")
    with pytest.raises(ValueError, match="shell"):
        remote.validate_container_name("target;docker ps")


def test_forward_command_targets_only_private_container_via_loopback():
    args = remote._ssh_forward_args(remote.RemoteDockerConfig(), local_port=43123, remote_ip="172.18.0.4", remote_port=8080)
    assert args[-1] == "jirongtech@112.111.7.91"
    assert "-N" in args and "-T" in args and "ExitOnForwardFailure=yes" in args
    assert "127.0.0.1:43123:172.18.0.4:8080" in args
    assert args.index("-N") < args.index("jirongtech@112.111.7.91")


def test_remote_api_requires_operator_confirmation_before_network_call(monkeypatch):
    called = False

    def fail_probe():
        nonlocal called
        called = True
        raise AssertionError("probe must not run without confirmation")

    monkeypatch.setattr("app.main.probe_authorized_remote_docker", fail_probe)
    response = client.post("/api/maze/remote-docker/probe", json={"operator_confirmed": False})
    assert response.status_code == 400
    assert called is False


def test_remote_probe_api_surfaces_unavailable_without_gold(monkeypatch):
    monkeypatch.setattr(
        "app.main.probe_authorized_remote_docker",
        lambda: {
            "status": "unavailable",
            "docker_binary": False,
            "docker_server": False,
            "running_containers": [],
            "evidence_sha256": "a" * 64,
        },
    )
    response = client.post("/api/maze/remote-docker/probe", json={"operator_confirmed": True})
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


def test_remote_scope_is_non_mutating_and_fixed_host():
    response = client.get("/api/maze/remote-docker/scope")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"]["remote_host"] == "112.111.7.91:60228"
    assert body["scope"]["mutating_docker_commands_allowed"] is False
    assert body["typed_oracle_required"] is True
