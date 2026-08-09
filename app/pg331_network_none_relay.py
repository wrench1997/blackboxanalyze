"""Pure contract builder for a PG-331 network-none local relay.

No function here launches Docker, opens a socket, or forwards a request.  It
only produces/validates bounded configuration and Docker inspection facts for
a future reviewed relay.  It intentionally cannot reinterpret the legacy
PG-246 bridge-network lifecycle as network-none.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re
from typing import Any


IMAGE = "sasanlabs/owasp-vulnerableapp@sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"
CONTAINER_NAME = "pg331-vulnerableapp-network-none"
RELAY_BIND_HOST = "127.0.0.1"
RELAY_BIND_PORT = 19091
TARGET_URL = "http://127.0.0.1:9090/VulnerableApp"
ALLOWED_METHODS = frozenset({"GET", "POST"})
ROLES = frozenset({"candidate", "reference", "negative", "replay"})
ROLE_CONTAINER_RE = re.compile(r"^pg331-vapp-nn-[0-9]{5}-[a-f0-9]{16}-(candidate|reference|negative|replay)$")


def role_container_name(*, seed: int, case_ref_sha256: str, role: str) -> str:
    """Derive an exact, non-literal fresh target name for one evaluator role."""
    if not isinstance(seed, int) or not 0 <= seed <= 99999:
        raise ValueError("seed must fit the fixed container-name contract")
    if not isinstance(case_ref_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", case_ref_sha256):
        raise ValueError("case reference must be a SHA-256 digest")
    if role not in ROLES:
        raise ValueError("role is not allowlisted")
    return f"pg331-vapp-nn-{seed:05d}-{case_ref_sha256[:16]}-{role}"


def _validate_container_name(container_name: str) -> str:
    name = str(container_name)
    if name == CONTAINER_NAME or ROLE_CONTAINER_RE.fullmatch(name):
        return name
    raise ValueError("container name must be the fixed name or a hash-bound allowlisted role name")


def build_container_command(*, container_name: str = CONTAINER_NAME) -> tuple[str, ...]:
    """Return the exact non-executing Docker command for the fixed image."""
    container_name = _validate_container_name(container_name)
    return (
        "docker", "run", "-d", "--name", container_name, "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "1g", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        "--tmpfs", "/run:rw,noexec,nosuid,size=16m", "--tmpfs", "/app/resources/static/upload:rw,noexec,nosuid,size=64m",
        "--tmpfs", "/contentDispositionUpload:rw,noexec,nosuid,size=64m", "--restart", "no", IMAGE,
    )


def build_relay_contract(*, bind_host: str = RELAY_BIND_HOST, bind_port: int = RELAY_BIND_PORT, target_url: str = TARGET_URL, container_name: str = CONTAINER_NAME) -> dict[str, Any]:
    """Build a fixed host-relay configuration without creating it."""
    if bind_host != RELAY_BIND_HOST or not isinstance(bind_port, int) or not 1024 <= bind_port <= 65535:
        raise ValueError("relay must bind only an explicit loopback port")
    if target_url != TARGET_URL:
        raise ValueError("relay target must be the fixed in-container VulnerableApp URL")
    container_name = _validate_container_name(container_name)
    return {
        "container_name": container_name,
        "image": IMAGE,
        "container_network_mode": "none",
        "published_ports": False,
        "bind_or_volume_mounts": False,
        "host_relay": {"bind_host": bind_host, "bind_port": bind_port, "loopback_only": True},
        "forwarding": {"transport": "container_exec_loopback", "target_url": target_url, "allowed_methods": ["GET", "POST"], "raw_response_off_context": True},
        "legacy_bridge_lifecycle_compatible": False,
        "execution": {"docker_started": False, "network_contacted": False},
    }


def attest_container_inspection(value: Mapping[str, Any], *, expected_name: str = CONTAINER_NAME) -> dict[str, Any]:
    """Fail closed unless a Docker-inspect projection proves the exact contract."""
    if not isinstance(value, Mapping):
        return {"valid": False, "failures": ["inspection_not_mapping"]}
    config = value.get("Config") if isinstance(value.get("Config"), Mapping) else {}
    host = value.get("HostConfig") if isinstance(value.get("HostConfig"), Mapping) else {}
    network = value.get("NetworkSettings") if isinstance(value.get("NetworkSettings"), Mapping) else {}
    mounts = value.get("Mounts")
    name = str(value.get("Name", "")).lstrip("/")
    ports = network.get("Ports")
    failures: list[str] = []
    try:
        _validate_container_name(expected_name)
    except ValueError:
        failures.append("expected_container_name")
    if name != expected_name: failures.append("container_name")
    if str(config.get("Image", "")) != IMAGE: failures.append("image")
    if str(host.get("NetworkMode", "")) != "none": failures.append("network_mode")
    if ports not in ({}, None): failures.append("published_ports")
    if not isinstance(mounts, list) or mounts: failures.append("bind_or_volume_mounts")
    if host.get("ReadonlyRootfs") is not True: failures.append("read_only")
    return {"valid": not failures, "failures": failures, "container_name": expected_name, "image": IMAGE, "network_mode": "none"}


def response_projection(*, method: str, status: int | None, headers: Mapping[str, Any] | None, body_length: int) -> dict[str, Any]:
    """Reduce a relay result to safe response shape facts; body is never accepted."""
    normalized = str(method).upper()
    if normalized not in ALLOWED_METHODS: raise ValueError("only GET/POST are allowlisted")
    if status is not None and (not isinstance(status, int) or isinstance(status, bool) or not 100 <= status < 600): raise ValueError("invalid HTTP status")
    if not isinstance(body_length, int) or isinstance(body_length, bool) or not 0 <= body_length <= 2 * 1024 * 1024: raise ValueError("invalid body length")
    values = headers or {}
    if not isinstance(values, Mapping): raise ValueError("headers must be a mapping")
    content_type = next((str(v).casefold() for k, v in values.items() if str(k).casefold() == "content-type"), "")
    return {"method": normalized, "status_class": f"{status // 100}xx" if status is not None else "transport", "content_type_class": "html" if "html" in content_type else "json" if "json" in content_type else "text" if content_type else "absent", "body_length_bucket": "zero" if body_length == 0 else "short" if body_length <= 64 else "medium" if body_length <= 1024 else "long", "redirect_hop_count": 1 if status is not None and 300 <= status < 400 else 0, "raw_response_body_stored": False}


__all__ = ["IMAGE", "CONTAINER_NAME", "TARGET_URL", "attest_container_inspection", "build_container_command", "build_relay_contract", "response_projection", "role_container_name"]
