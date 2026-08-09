"""Bounded, operator-driven HTTP observation for Docker-local targets.

This module is deliberately a *surface observer*, not a vulnerability scanner.
It sends a baseline GET, an inert canary GET, and (only when the operator opts
in) an inert form POST to an explicitly allowlisted Docker target.  It stores
response projections and hashes, never raw response bodies or credentials, and
cannot claim a typed vulnerability effect without a target-specific oracle.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import secrets
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

SCHEMA_VERSION = "sift-authorized-docker-target-session-v1"
PAYLOAD_SCHEMA = "sift-detection-payload-v1"
MAX_BODY_BYTES = 128 * 1024
MAX_URL_LENGTH = 2048
SAFE_QUERY_NAMES = frozenset({"sift_probe"})
SENSITIVE_QUERY_NAMES = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "session", "auth", "authorization"})
DEFAULT_AUTHORIZED_ORIGINS = frozenset(
    {
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3100",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8766",
        "http://127.0.0.1:19090",
        "http://localhost:3000",
        "http://localhost:3100",
        "http://localhost:8080",
        "http://pikachu:80",
        "http://juice-shop:3000",
        "http://vulnerableapp:80",
        "http://target:8080",
    }
)
LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        body = value
    else:
        body = _canonical(value).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _origin(raw: str) -> str:
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("目标必须使用 http 或 https")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("目标不得包含凭据或 fragment")
    if not parsed.hostname:
        raise ValueError("目标缺少主机名")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise ValueError("目标端口无效") from error
    return f"{parsed.scheme.lower()}://{parsed.hostname.casefold()}:{port}"


def _configured_origins() -> frozenset[str]:
    configured = os.environ.get("SIFT_AUTHORIZED_DOCKER_TARGETS", "")
    values: set[str] = set(DEFAULT_AUTHORIZED_ORIGINS)
    for item in configured.split(","):
        item = item.strip()
        if item:
            values.add(_origin(item))
    return frozenset(values)


def _is_loopback_origin(raw: str) -> bool:
    """Allow any loopback port while keeping the observer off private networks.

    Docker Desktop commonly publishes a disposable challenge on a new host
    port.  Requiring every such port to be copied into a static list created a
    false permission failure.  Loopback is still a local operator boundary;
    Docker bridge/private addresses remain opt-in through
    ``SIFT_AUTHORIZED_DOCKER_TARGETS``.
    """

    hostname = (urlsplit(raw).hostname or "").casefold()
    if hostname in LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_target(raw: str) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > MAX_URL_LENGTH:
        raise ValueError("目标 URL 为空或超出长度限制")
    parsed = urlsplit(raw)
    target_origin = _origin(raw)
    if target_origin not in _configured_origins() and not _is_loopback_origin(raw):
        raise ValueError(
            "目标不在 Docker 授权清单内；loopback 任意端口可用，"
            "Docker bridge/private 地址请设置 SIFT_AUTHORIZED_DOCKER_TARGETS，"
            "不要用任意公网 URL"
        )
    if ".." in parsed.path.split("/"):
        raise ValueError("目标路径不得穿越父目录")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    for key, _ in query:
        normalized = key.casefold()
        if normalized in SENSITIVE_QUERY_NAMES:
            raise ValueError("目标查询参数不得包含凭据或会话字段")
        if normalized not in SAFE_QUERY_NAMES:
            raise ValueError("目标 URL 只能携带 sift_probe 查询参数")
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, parsed.query, ""))


def _with_probe(url: str, marker: str) -> str:
    parsed = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key.casefold() != "sift_probe"]
    query.append(("sift_probe", marker))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", urlencode(query), ""))


async def _bounded_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    marker: str,
    form: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"accept": "text/html,application/json;q=0.9,*/*;q=0.1", "x-sift-probe": marker}
    try:
        async with client.stream(method, url, headers=headers, data=form or None) as response:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                if total < MAX_BODY_BYTES:
                    keep = chunk[: MAX_BODY_BYTES - total]
                    chunks.append(keep)
                total += len(chunk)
                if total >= MAX_BODY_BYTES:
                    break
            body = b"".join(chunks)
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
            location = response.headers.get("location", "")
            return {
                "ok": True,
                "status_code": response.status_code,
                "status_class": f"{response.status_code // 100}xx",
                "content_type": content_type,
                "location_present": bool(location),
                "allow_present": "allow" in response.headers,
                "body_bytes_observed": total,
                "body_truncated": total > len(body),
                "body_sha256": _sha256(body),
                "marker_reflected": marker.encode("utf-8") in body,
            }
    except (httpx.HTTPError, OSError) as error:
        return {"ok": False, "error_class": type(error).__name__, "error": str(error)[:240]}


def _shape(projection: dict[str, Any]) -> tuple[Any, ...]:
    return (
        projection.get("status_class"),
        projection.get("content_type"),
        projection.get("location_present"),
        projection.get("allow_present"),
        min(int(projection.get("body_bytes_observed", 0)), MAX_BODY_BYTES),
        projection.get("marker_reflected"),
    )


def _diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    fields = ("status_class", "content_type", "location_present", "allow_present", "marker_reflected")
    return {
        "shape_changed": _shape(baseline) != _shape(candidate),
        "changed_fields": [field for field in fields if baseline.get(field) != candidate.get(field)],
        "baseline_ok": bool(baseline.get("ok")),
        "candidate_ok": bool(candidate.get("ok")),
    }


def _safe_manifest(*, target: str, method: str, path: str, marker: str, form: dict[str, str] | None, expected: dict[str, Any]) -> dict[str, Any]:
    """Build the shared manifest shape while retaining the Docker origin scope.

    ``app.detection_payload`` intentionally accepts loopback-only origins for
    generic callers.  This route has a second, explicit Docker allowlist, so
    the manifest is built here and bound to that allowlist instead of widening
    the generic payload validator for every API consumer.
    """

    body: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA,
        "target": target,
        "method": method,
        "path": path,
        "headers": {"accept": "text/html,application/json;q=0.9,*/*;q=0.1", "x-sift-probe": marker},
        "marker": marker,
        "probe_kind": "http_canary",
        "probe": marker,
        "expected": json.loads(_canonical(expected)),
        "safety": {
            "authorized_scope": "docker_local_allowlist",
            "non_destructive": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
            "no_data_exfiltration": True,
            "no_external_network": True,
            "does_not_execute": False,
        },
    }
    if method == "POST":
        body["form"] = dict(form or {})
    body["payload_sha256"] = _sha256(body)
    return body


def _event(
    *,
    node: str,
    phase: str,
    status: str,
    action: str,
    observation: str,
    detail: str,
    belief: dict[str, float],
    rule: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "node": node,
        "phase": phase,
        "status": status,
        "action": action,
        "observation": observation,
        "detail": detail,
        "belief": belief,
        "rule": rule,
        "evidence": evidence,
    }


async def analyze_authorized_target(*, target_url: str, allow_safe_post: bool) -> dict[str, Any]:
    """Run a bounded observation loop against one explicitly authorized target."""

    target = _validate_target(target_url)
    marker = f"sift-canary-{secrets.token_hex(5)}"
    target_path = urlsplit(target).path or "/"
    target_query = urlsplit(target).query
    manifest_path = target_path + (f"?{target_query}" if target_query else "")
    timeout = httpx.Timeout(5.0, connect=3.0, read=5.0, write=3.0, pool=3.0)
    limits = httpx.Limits(max_connections=2, max_keepalive_connections=1)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False) as client:
        baseline = await _bounded_request(client, "GET", target, marker=marker)
        candidate_url = _with_probe(target, marker)
        candidate_get = await _bounded_request(client, "GET", candidate_url, marker=marker)
        post_projection: dict[str, Any] | None = None
        if allow_safe_post:
            post_projection = await _bounded_request(client, "POST", target, marker=marker, form={"sift_probe": marker})
        replay = await _bounded_request(client, "GET", candidate_url, marker=marker)

    get_diff = _diff(baseline, candidate_get)
    replay_diff = _diff(candidate_get, replay)
    shape_changed = bool(get_diff["shape_changed"])
    evidence_body = {
        "schema_version": SCHEMA_VERSION,
        "target_origin": _origin(target),
        "marker": marker,
        "baseline": baseline,
        "candidate_get": candidate_get,
        "post": post_projection,
        "replay": replay,
        "get_diff": get_diff,
        "replay_diff": replay_diff,
    }
    evidence_hash = f"sha256: {_sha256(evidence_body)}"
    candidate_status = "candidate" if shape_changed else "abstain"
    candidate_belief = {"effect": 0.06 if shape_changed else 0.01, "inputOnly": 0.58 if shape_changed else 0.12, "none": 0.19 if shape_changed else 0.62, "unknown": 0.17 if shape_changed else 0.25}
    events = [
        _event(
            node="start",
            phase="RESET",
            status="ready",
            action="RESET · operator session",
            observation="authorized Docker origin accepted",
            detail="目标通过显式 Docker 授权清单；本轮不写入训练集或长期记忆。",
            belief={"effect": 0.0, "inputOnly": 0.0, "none": 0.0, "unknown": 1.0},
            rule="scope := explicit_docker_allowlist(target_origin)",
            evidence=evidence_hash,
        ),
        _event(
            node="scan",
            phase="PROBE",
            status="probe",
            action="GET · baseline",
            observation=f"{baseline.get('status_class', 'transport error')} / {baseline.get('content_type', 'unknown')}",
            detail="只记录响应投影：状态类、内容类型、体积桶、重定向存在性和哈希，不保存正文。",
            belief={"effect": 0.0, "inputOnly": 0.0, "none": 0.52, "unknown": 0.48},
            rule="baseline := project(GET(target))",
            evidence=evidence_hash,
        ),
        _event(
            node="junction",
            phase="BELIEF",
            status=candidate_status,
            action="GET · inert canary",
            observation="shape delta" if shape_changed else "no stable shape delta",
            detail="形状变化只能进入 candidate；没有目标侧 typed oracle 时不得升级为 confirmed_positive。",
            belief=candidate_belief,
            rule="candidate := diff(GET_control, GET_canary)",
            evidence=evidence_hash,
        ),
    ]
    if allow_safe_post:
        events.append(
            _event(
                node="candidate",
                phase="PROBE",
                status="candidate" if post_projection and post_projection.get("ok") else "abstain",
                action="POST · inert form canary",
                observation=f"{post_projection.get('status_class', 'transport error') if post_projection else 'not sent'}",
                detail="POST 只发送 sift_probe canary；是否存在业务影响仍需目标侧 evaluator-only oracle。",
                belief={"effect": 0.08 if post_projection and post_projection.get("ok") else 0.01, "inputOnly": 0.44, "none": 0.18, "unknown": 0.3},
                rule="channel_pair := compare(GET_canary, POST_canary)",
                evidence=evidence_hash,
            )
        )
    events.extend(
        [
            _event(
                node="replay",
                phase="REPLAY",
                status="negative" if not replay_diff["shape_changed"] else "candidate",
                action="GET · replay canary",
                observation="replay stable" if not replay_diff["shape_changed"] else "replay drift",
                detail="复放检查形状是否稳定；通用 URL 无法自行证明 fresh reset，因此不会解锁出口。",
                belief={"effect": 0.1 if shape_changed else 0.01, "inputOnly": 0.48, "none": 0.13, "unknown": 0.29},
                rule="replay_stable := diff(candidate, replay) == ∅",
                evidence=evidence_hash,
            ),
            _event(
                node="oracle",
                phase="ORACLE",
                status="abstain",
                action="typed oracle · waiting for target evaluator",
                observation="typed effect unavailable",
                detail="没有目标专属的预期效果、阴性对照和 fresh reset 证据，最终状态必须 abstain。",
                belief={"effect": 0.02, "inputOnly": 0.2, "none": 0.08, "unknown": 0.7},
                rule="confirmed_positive := typed_effect AND negative_control AND fresh_reset AND evidence_hash",
                evidence=evidence_hash,
            ),
        ]
    )
    manifests = [_safe_manifest(
        target=_origin(target),
        method="GET",
        path=manifest_path,
        marker=marker,
        form=None,
        expected={"shape_changed": bool(shape_changed), "typed_effect": "unknown"},
    )]
    if allow_safe_post:
        manifests.append(_safe_manifest(
            target=_origin(target),
            method="POST",
            path=manifest_path,
            form={"sift_probe": marker},
            marker=marker,
            expected={"typed_effect": "unknown"},
        ))
    result = {
        "schema_version": SCHEMA_VERSION,
        "target": {"origin": _origin(target), "path": target_path, "authorized_scope": "docker_local_allowlist"},
        "marker": marker,
        "request_count": 4 if allow_safe_post else 3,
        "maze_events": events,
        "candidate_status": candidate_status,
        "exit_unlocked": False,
        "typed_oracle": {"status": "unavailable", "confirmed_positive": False},
        "response_projection": {"baseline": baseline, "candidate_get": candidate_get, "post": post_projection, "replay": replay, "get_diff": get_diff, "replay_diff": replay_diff},
        "safe_probe_manifests": manifests,
        "evidence_sha256": evidence_hash,
        "promotion": {"training_sample": False, "long_term_memory": False, "reason": "generic URL lacks typed evaluator-only oracle"},
        "safety": {"authorized_local_only": True, "docker_allowlist_only": True, "follow_redirects": False, "raw_body_stored": False, "credentials_stored": False, "destructive_payloads": False, "script_execution": False, "database_writes": False},
    }
    return result


def iter_authorized_origins() -> Iterable[str]:
    return sorted(_configured_origins())
