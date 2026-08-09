"""PG-280 adapter for an explicitly authorized remote Docker host.

The adapter is intentionally narrow.  It only performs read-only Docker
availability/inspection commands on the fixed research host and can open a
loopback SSH tunnel to a private container *after* an operator has supplied an
allowlisted container name and port.  It never creates, restarts, removes, or
executes a container and it never sends a vulnerability payload.  The tunnel
is only a transport for the inert GET/POST observer in
``authorized_target_session``.

This module makes the remote integration real without turning an unavailable
Docker daemon into synthetic evidence.  A missing daemon is a first-class
``unavailable`` result; no training or memory promotion is possible until a
target-specific evaluator supplies fresh-reset, negative-control and typed
effect evidence.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlsplit


SCHEMA_VERSION = "pg280-authorized-remote-docker-adapter-v1"
REMOTE_HOST = "112.111.7.91"
REMOTE_SSH_PORT = 60228
DEFAULT_REMOTE_USER = "jirongtech"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 15
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
ALLOWED_REMOTE_COMMANDS = (
    ("command", "-v", "docker"),
    ("docker", "version", "--format={{json .Server}}"),
    ("docker", "ps", "--format={{.Names}}"),
)
CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$")
PRIVATE_TARGET_PORT_MIN = 1
PRIVATE_TARGET_PORT_MAX = 65535


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class RemoteDockerConfig:
    """Connection parameters for the one operator-authorized host."""

    host: str = REMOTE_HOST
    ssh_port: int = REMOTE_SSH_PORT
    user: str = DEFAULT_REMOTE_USER
    connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls) -> "RemoteDockerConfig":
        # The host and port are deliberately not environment-overridable.  A
        # future host must be added by a code/config review, not pasted into a
        # request or inherited from an untrusted process environment.
        user = os.environ.get("SIFT_REMOTE_DOCKER_USER", DEFAULT_REMOTE_USER).strip()
        if not user or not re.fullmatch(r"[a-zA-Z0-9_.@-]{1,64}", user):
            raise ValueError("SIFT_REMOTE_DOCKER_USER is invalid")
        return cls(user=user)

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    def scope(self) -> dict[str, Any]:
        return {
            "remote_host": f"{self.host}:{self.ssh_port}",
            "ssh_target": self.ssh_target,
            "authorization": "operator_allowlisted_remote_docker",
            "command_allowlist": [list(command) for command in ALLOWED_REMOTE_COMMANDS],
            "mutating_docker_commands_allowed": False,
            "container_creation_allowed": False,
            "container_restart_allowed": False,
            "container_removal_allowed": False,
            "external_network": False,
            "target_private_address_required": True,
            "raw_response_body_stored": False,
            "training_or_replay_started": False,
        }


def validate_container_name(name: str) -> str:
    if not isinstance(name, str) or not CONTAINER_NAME_RE.fullmatch(name):
        raise ValueError("container_name must be an exact Docker name, not a shell expression")
    allowed = {
        item.strip()
        for item in os.environ.get("SIFT_REMOTE_DOCKER_TARGET_CONTAINERS", "").split(",")
        if item.strip()
    }
    if name not in allowed:
        raise ValueError("container_name is not in the explicit target allowlist SIFT_REMOTE_DOCKER_TARGET_CONTAINERS")
    return name


def validate_container_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not PRIVATE_TARGET_PORT_MIN <= port <= PRIVATE_TARGET_PORT_MAX:
        raise ValueError("container_port must be an integer between 1 and 65535")
    return port


def validate_origin_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or len(path) > 2048:
        raise ValueError("path must be an origin-relative path")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment or ".." in parsed.path.split("/"):
        raise ValueError("path must not contain an origin, fragment, or parent traversal")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        if key.casefold() != "sift_probe":
            raise ValueError("remote target path may only carry sift_probe")
    return path


def _safe_error_class(stderr: str, returncode: int) -> str:
    lower = stderr.casefold()
    if returncode == 255 or "timed out" in lower or "could not resolve" in lower:
        return "ssh_unreachable"
    if "command not found" in lower or "not found" in lower:
        return "command_unavailable"
    if "permission denied" in lower:
        return "permission_denied"
    if returncode:
        return f"remote_exit_{returncode}"
    return "none"


def _ssh_base(config: RemoteDockerConfig) -> list[str]:
    if shutil.which("ssh") is None:
        raise RuntimeError("ssh executable is unavailable")
    return [
        "ssh",
        "-p",
        str(config.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.connect_timeout_seconds}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=1",
        config.ssh_target,
    ]


def _run_remote(config: RemoteDockerConfig, command: tuple[str, ...]) -> dict[str, Any]:
    if command not in ALLOWED_REMOTE_COMMANDS:
        raise ValueError("remote command is not in the PG-280 read-only allowlist")
    args = _ssh_base(config) + list(command)
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.command_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr_class": "command_timeout"}
    except OSError as error:
        return {"returncode": 127, "stdout": "", "stderr_class": type(error).__name__}
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr_class": _safe_error_class(completed.stderr, completed.returncode),
    }


def probe_authorized_remote_docker(config: RemoteDockerConfig | None = None) -> dict[str, Any]:
    """Probe the fixed host without changing Docker or starting a target."""

    config = config or RemoteDockerConfig.from_environment()
    scope = config.scope()
    try:
        docker_binary = _run_remote(config, ALLOWED_REMOTE_COMMANDS[0])
        docker_version = _run_remote(config, ALLOWED_REMOTE_COMMANDS[1])
        docker_ps = _run_remote(config, ALLOWED_REMOTE_COMMANDS[2])
    except RuntimeError as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "scope": scope,
            "status": "ssh_unavailable",
            "ssh_reachable": False,
            "docker_binary": False,
            "docker_server": False,
            "running_containers": [],
            "error_class": type(error).__name__,
            "real_application_gold_rows": 0,
            "training_or_replay_started": False,
            "interpretation": "SSH executable is unavailable; no remote action was attempted.",
        }
        result["evidence_sha256"] = sha256(result)
        return result

    docker_found = bool(docker_binary.get("stdout")) and docker_binary.get("returncode") == 0
    server_json = None
    if docker_version.get("returncode") == 0 and docker_version.get("stdout"):
        try:
            server_json = json.loads(docker_version["stdout"])
        except json.JSONDecodeError:
            server_json = None
    server_available = isinstance(server_json, dict) and bool(server_json.get("Version"))
    containers = []
    if docker_ps.get("returncode") == 0:
        containers = [line for line in docker_ps.get("stdout", "").splitlines() if CONTAINER_NAME_RE.fullmatch(line)]
    ssh_reachable = any(item.get("stderr_class") != "ssh_unreachable" for item in (docker_binary, docker_version, docker_ps))
    if not ssh_reachable:
        status = "unreachable"
    elif docker_found and server_available:
        status = "available"
    else:
        status = "unavailable"
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "status": status,
        "ssh_reachable": ssh_reachable,
        "docker_binary": docker_found,
        "docker_server": server_available,
        "docker_server_version": server_json.get("Version") if isinstance(server_json, dict) else None,
        "running_containers": containers,
        "command_results": {
            "docker_binary": {"returncode": docker_binary.get("returncode"), "stderr_class": docker_binary.get("stderr_class")},
            "docker_version": {"returncode": docker_version.get("returncode"), "stderr_class": docker_version.get("stderr_class")},
            "docker_ps": {"returncode": docker_ps.get("returncode"), "stderr_class": docker_ps.get("stderr_class")},
        },
        "real_application_gold_rows": 0,
        "training_or_replay_started": False,
        "interpretation": (
            "远程 Docker 可用，但尚未绑定目标专属 evaluator；只能进入待验收状态。"
            if status == "available"
            else "远程主机未提供可用 Docker daemon；没有启动容器，也没有生成真实应用 gold。"
        ),
    }
    result["evidence_sha256"] = sha256(result)
    return result


def _inspect_container(config: RemoteDockerConfig, container_name: str) -> dict[str, Any]:
    """Inspect is kept separate and is intentionally not part of the probe list."""

    validate_container_name(container_name)
    command = ("docker", "inspect", "--type=container", "--format={{json .State}}", container_name)
    # Do not let a caller smuggle this command through _run_remote; the
    # inspect shape is constructed only from the validated exact name.
    args = _ssh_base(config) + list(command)
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=config.command_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("remote docker inspect timed out") from error
    if completed.returncode != 0:
        raise RuntimeError(f"remote docker inspect failed: {_safe_error_class(completed.stderr, completed.returncode)}")
    try:
        state = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise RuntimeError("remote docker inspect returned invalid state") from error
    if not isinstance(state, dict):
        raise RuntimeError("remote docker inspect state is not an object")
    return state


def _remote_container_ip(config: RemoteDockerConfig, container_name: str) -> str:
    validate_container_name(container_name)
    command = ("docker", "inspect", "--type=container", f"--format={{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}", container_name)
    args = _ssh_base(config) + list(command)
    completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=config.command_timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(f"remote container address unavailable: {_safe_error_class(completed.stderr, completed.returncode)}")
    address = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as error:
        raise RuntimeError("remote container did not expose a valid private IP") from error
    if ip.is_loopback or ip.is_unspecified or ip.is_global:
        raise RuntimeError("remote container IP is not private/internal")
    return str(ip)


def _ssh_forward_args(config: RemoteDockerConfig, *, local_port: int, remote_ip: str, remote_port: int) -> list[str]:
    """Build an OpenSSH forward command with options before the destination."""

    base = _ssh_base(config)
    destination = base.pop()
    return base + [
        "-N",
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{local_port}:{remote_ip}:{remote_port}",
        destination,
    ]


@dataclass
class RemoteTunnel:
    process: subprocess.Popen[str]
    local_port: int
    remote_ip: str
    remote_port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)

    def __enter__(self) -> "RemoteTunnel":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def open_authorized_tunnel(
    config: RemoteDockerConfig,
    *,
    container_name: str,
    container_port: int,
) -> RemoteTunnel:
    """Open a local-only SSH forward to a running private container."""

    container_name = validate_container_name(container_name)
    container_port = validate_container_port(container_port)
    state = _inspect_container(config, container_name)
    if not bool(state.get("Running")):
        raise RuntimeError("remote target container is not running")
    remote_ip = _remote_container_ip(config, container_name)
    local_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        local_socket.bind(("127.0.0.1", 0))
        local_port = int(local_socket.getsockname()[1])
    finally:
        local_socket.close()
    args = _ssh_forward_args(config, local_port=local_port, remote_ip=remote_ip, remote_port=container_port)
    process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = process.stderr.read(240) if process.stderr else ""
            raise RuntimeError(f"remote tunnel failed: {_safe_error_class(detail, process.returncode or 1)}")
        time.sleep(0.05)
    return RemoteTunnel(process=process, local_port=local_port, remote_ip=remote_ip, remote_port=container_port)


async def analyze_authorized_remote_target(
    *,
    container_name: str,
    container_port: int,
    path: str,
    allow_safe_post: bool,
    config: RemoteDockerConfig | None = None,
) -> dict[str, Any]:
    """Observe a running private container through a short-lived SSH tunnel.

    The underlying observer sends only an inert canary.  The result is still a
    candidate/abstain projection because a generic remote container has no
    typed evaluator, fresh-reset proof, or matched negative oracle attached.
    """

    config = config or RemoteDockerConfig.from_environment()
    container_name = validate_container_name(container_name)
    container_port = validate_container_port(container_port)
    path = validate_origin_relative_path(path)
    probe = probe_authorized_remote_docker(config)
    if probe.get("status") != "available":
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "remote": {"scope": config.scope(), "container_name": container_name, "container_port": container_port, "path": path},
            "probe": probe,
            "observation": None,
            "promotion": {"training_sample": False, "long_term_memory": False, "real_application_gold": False, "reason": "remote Docker is unavailable"},
        }
        blocked["evidence_sha256"] = sha256(blocked)
        return blocked
    from .authorized_target_session import analyze_authorized_target

    with open_authorized_tunnel(config, container_name=container_name, container_port=container_port) as tunnel:
        observation = await analyze_authorized_target(target_url=f"{tunnel.base_url}{path}", allow_safe_post=allow_safe_post)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "observed",
        "remote": {"scope": config.scope(), "container_name": container_name, "container_port": container_port, "path": path, "tunnel": {"remote_ip": tunnel.remote_ip, "remote_port": tunnel.remote_port}},
        "probe": probe,
        "observation": observation,
        "promotion": {"training_sample": False, "long_term_memory": False, "real_application_gold": False, "reason": "generic observer has no target-specific typed evaluator or fresh-reset proof"},
    }
    result["evidence_sha256"] = sha256(result)
    return result


def iter_remote_scope() -> Iterator[str]:
    """Yield stable non-secret scope fields for an operator-facing API."""

    config = RemoteDockerConfig.from_environment()
    yield f"ssh://{config.host}:{config.ssh_port}"
    yield f"user={config.user}"
    yield "docker mutations=disabled"
    yield "container allowlist=SIFT_REMOTE_DOCKER_TARGET_CONTAINERS"
    yield "typed oracle required"
