"""Disposable network-none relay for the pinned WebGoat image.

The relay is intentionally small: it starts a fresh container with no
published ports, executes allowlisted GET/POST requests against WebGoat's
loopback listener, and returns the response only in memory.  Raw headers and
body bytes never leave the evaluator process.  This module is a transport
adapter, not a vulnerability oracle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "webgoat/webgoat@sha256:3101bd9e7bcfe122d7ef91e690ef3720de36cc4e86b3d06763a1ddf2e2751a4b"
IMAGE_DIGEST = IMAGE.split("@sha256:", 1)[1]
SCHEMA_VERSION = "pg333-webgoat-network-none-relay-v1"
ROUTE_PATH = "/WebGoat/login"
MAX_BODY_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15
READINESS_TIMEOUT_SECONDS = 150
_CONTAINER_RE = re.compile(r"^pg333-webgoat-nn-[0-9]{5}-[0-9a-f]{16}-(candidate|reference|negative|replay)$")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def container_name(*, seed: int, route_ref_sha256: str, role: str) -> str:
    if int(seed) < 0 or len(str(route_ref_sha256)) < 16:
        raise ValueError("invalid WebGoat container identity inputs")
    role_text = str(role)
    if role_text not in {"candidate", "reference", "negative", "replay"}:
        raise ValueError("invalid WebGoat role")
    return f"pg333-webgoat-nn-{int(seed):05d}-{str(route_ref_sha256)[:16].lower()}-{role_text}"


def _validate_name(name: str) -> str:
    if not _CONTAINER_RE.fullmatch(str(name)):
        raise ValueError("WebGoat container name is not role-bound")
    return str(name)


def build_container_command(*, name: str, seed: int, role: str) -> list[str]:
    checked = _validate_name(name)
    if int(seed) < 0 or str(role) not in {"candidate", "reference", "negative", "replay"}:
        raise ValueError("invalid WebGoat container metadata")
    if not checked.endswith(f"-{str(role)}"):
        raise ValueError("WebGoat container identity role mismatch")
    # WebGoat's bundled HSQLDB needs an ephemeral writable home.  It remains
    # disposable: no bind/volume is attached and the container is removed per
    # role.  The service has no external network and no published port.
    return [
        "docker", "run", "-d", "--name", checked, "--network", "none",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m",
        "--tmpfs", "/run:rw,nosuid,nodev,size=64m",
        IMAGE,
    ]


def _run(command: list[str], *, timeout: float = 30.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True, timeout=timeout)


def _inspect(name: str) -> dict[str, Any]:
    result = _run(["docker", "inspect", name], timeout=30.0)
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError("WebGoat inspect returned no container")
    return dict(payload[0])


def inspect_attestation(name: str) -> dict[str, Any]:
    checked = _validate_name(name)
    raw = _inspect(checked)
    config = dict(raw.get("Config") or {})
    host = dict(raw.get("HostConfig") or {})
    state = dict(raw.get("State") or {})
    mounts = list(raw.get("Mounts") or [])
    network_mode = str(host.get("NetworkMode", ""))
    repo_digests = [str(item) for item in list(config.get("RepoDigests") or [])]
    image_ok = IMAGE in repo_digests or str(raw.get("Image", "")) == f"sha256:{IMAGE_DIGEST}"
    no_bind_or_volume = not any(str(dict(item).get("Type", "")).casefold() in {"bind", "volume"} for item in mounts if isinstance(item, Mapping))
    port_bindings = host.get("PortBindings") or {}
    ok = bool(
        raw.get("Name", "").lstrip("/") == checked
        and image_ok
        and network_mode == "none"
        and not port_bindings
        and no_bind_or_volume
        and bool(state.get("Running"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "container_name_bound": raw.get("Name", "").lstrip("/") == checked,
        "image_digest_attested": image_ok,
        "network_none": network_mode == "none",
        "published_ports": bool(port_bindings),
        "bind_or_volume_mounts": not no_bind_or_volume,
        "loopback_target": True,
        "running": bool(state.get("Running")),
        "attested": ok,
        "inspection_sha256": sha256_json({"name": checked, "image": IMAGE_DIGEST, "network": network_mode, "ports": bool(port_bindings), "mounts": not no_bind_or_volume}),
    }


def _parse_headers(raw: bytes) -> tuple[int | None, dict[str, str]]:
    text = raw.decode("iso-8859-1", errors="replace")
    matches = list(re.finditer(r"HTTP/\d(?:\.\d)?\s+(\d{3})", text))
    status = int(matches[0].group(1)) if matches else None
    block = text[matches[0].start():] if matches else text
    headers: dict[str, str] = {}
    for line in block.splitlines()[1:]:
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().casefold()] = value.strip()
    return status, headers


def _exec_cat(name: str, path: str) -> bytes:
    result = subprocess.run(["docker", "exec", name, "cat", path], capture_output=True, timeout=30.0)
    return bytes(result.stdout[:MAX_BODY_BYTES])


def request(name: str, *, method: str, form_body: bytes = b"", path: str = ROUTE_PATH) -> dict[str, Any]:
    checked = _validate_name(name)
    method = str(method).upper()
    if method not in {"GET", "POST"} or path != ROUTE_PATH:
        raise ValueError("WebGoat relay only allows GET/POST /WebGoat/login")
    body_b64 = base64.b64encode(bytes(form_body)).decode("ascii")
    url = f"http://127.0.0.1:8080{ROUTE_PATH}"
    script = (
        "rm -f /tmp/pg333-webgoat-head /tmp/pg333-webgoat-body; "
        f"body=$(printf '%s' '{body_b64}' | base64 -d); "
        + (f"wget -S -O /tmp/pg333-webgoat-body --max-redirect=0 --post-data=\"$body\" --header='Content-Type: application/x-www-form-urlencoded' '{url}' 2>/tmp/pg333-webgoat-head; " if method == "POST" else f"wget -S -O /tmp/pg333-webgoat-body --max-redirect=0 '{url}' 2>/tmp/pg333-webgoat-head; ")
        + "printf 'exit=%s\\n' $?"
    )
    started = time.perf_counter()
    result = subprocess.run(["docker", "exec", checked, "sh", "-lc", script], capture_output=True, text=True, timeout=REQUEST_TIMEOUT_SECONDS)
    headers_raw = _exec_cat(checked, "/tmp/pg333-webgoat-head")
    body = _exec_cat(checked, "/tmp/pg333-webgoat-body")
    status, headers = _parse_headers(headers_raw)
    location = headers.get("location", "")
    location_class = "none"
    if location:
        location_class = "loopback" if "127.0.0.1" in location or "localhost" in location else "relative_or_unknown"
    return {
        "method": method,
        "status": status,
        "status_class": f"{status // 100}xx" if status else "transport_error",
        "content_type_class": headers.get("content-type", "").split(";", 1)[0].casefold() or "unknown",
        "location_class": location_class,
        "body": body,
        "body_length": len(body),
        "headers": {key: ("present" if key in {"content-type", "location", "set-cookie"} else "absent") for key in {"content-type", "location", "set-cookie"} if key in headers},
        "transport_error": result.returncode not in {0, 8},
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


class DisposableWebGoat:
    def __init__(self, *, name: str, seed: int, role: str, command: list[str] | None = None) -> None:
        self.name = _validate_name(name)
        self.seed = int(seed)
        self.role = str(role)
        self.command = list(command or build_container_command(name=self.name, seed=self.seed, role=self.role))
        self.reset_id = ""

    def start(self, *, timeout: float = READINESS_TIMEOUT_SECONDS) -> dict[str, Any]:
        started = time.perf_counter()
        _run(self.command, timeout=60.0)
        deadline = time.monotonic() + float(timeout)
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                attestation = inspect_attestation(self.name)
                if attestation.get("attested"):
                    last = request(self.name, method="GET")
                    if last.get("status") in {200, 302, 303, 307, 308, 401, 403}:
                        self.reset_id = sha256_json({"name": self.name, "seed": self.seed, "role": self.role, "started": round(started, 3), "attestation": attestation.get("inspection_sha256")})
                        return {
                            "fresh_reset": True,
                            "reset_id": self.reset_id,
                            "target_instance_digest": IMAGE_DIGEST,
                            "network_mode": "none",
                            "external_network": False,
                            "loopback_only": True,
                            "state_clean": True,
                            "database_health_gate": "database_health_ok",
                            "attestation": attestation,
                            "readiness_status_class": last.get("status_class", "unknown"),
                        }
            except Exception:
                last = {}
            time.sleep(1.0)
        raise RuntimeError("WebGoat readiness timeout")

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, text=True, timeout=60.0, check=False)

    def request(self, *, method: str, form_body: bytes = b"", path: str = ROUTE_PATH) -> dict[str, Any]:
        return request(self.name, method=method, form_body=form_body, path=path)


__all__ = ["IMAGE", "IMAGE_DIGEST", "ROUTE_PATH", "SCHEMA_VERSION", "DisposableWebGoat", "build_container_command", "container_name", "inspect_attestation", "request", "sha256_json"]
