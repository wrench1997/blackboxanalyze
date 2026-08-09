"""PG-379 dynamic whole-page source-row collector.

PG-379 is the first live lane for the two implementation-disjoint dynamic
implementations described by :mod:`scripts.plan_pg379_source_collection`.
The route matrix contains abstract shape coordinates only; an implementation
runtime is injected by the operator and is never allowed to receive a raw
route literal from this module.  The collector starts a *fresh* runtime for
every implementation/seed/route/role episode, obtains a bounded browser/HTTP
whole-page observation, and immediately reduces it through the reviewed
PG-377 adapter.  HTML and response bytes are held only in the episode's
process memory and are not returned, written, or put in an evaluator sidecar.

The default lane is fail-closed.  Live collection requires the explicit
``PG379_LOCAL_DOCKER_EVAL=1`` flag, two independently attested immutable image
digests, and an injected runtime factory.  A missing build/attestation,
mutable image tag, unsafe network/mount setting, or failed image probe is
reported as ``blocked_preflight`` before the first runtime is constructed.
This module never creates training rows, starts a GPU job, or promotes a
source/evaluator artifact.

``runtime_factory`` is intentionally an injection point rather than a generic
Docker launcher.  The two dynamic implementations have not been bound to a
shared route adapter; accepting an unreviewed default launcher here would
turn the abstract planning route classes into guessed wire paths.  Contract
tests use a deterministic fake runtime implementing ``start``, ``request``
and ``stop``.  An operator can bind a reviewed browser/HTTP runtime without
changing the adapter or this lifecycle contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_pg331_evaluator_sidecar, sha256_json  # noqa: E402
from app.pg387_ctf_frontend_projection import project_js_source  # noqa: E402
from app.pg377_webgoat_source_row_adapter import (  # noqa: E402
    FIELD_COUNT,
    capture_pg377_webgoat_source_row,
    validate_pg377_webgoat_source_row,
)
from scripts.plan_pg379_source_collection import (  # noqa: E402
    ROLES,
    ROUTE_SHAPES,
    SEEDS,
    SLOTS,
    SOURCE_ROLES,
    build_pg379_source_collection_plan,
    validate_pg379_source_collection_plan,
)
from scripts.run_pg377_webgoat_source_rows_live import (  # noqa: E402
    _abstract_headers,
    _belief,
    _failure_projection_for_role,
    _request_projection,
    _response_projection,
    _role_input,
)


SCHEMA_VERSION = "pg379-dynamic-source-rows-live-v1"
OPERATOR_FLAG = "PG379_LOCAL_DOCKER_EVAL"
MAX_PAGE_BYTES = 2 * 1024 * 1024
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

ALL_ROLES = tuple(ROLES)
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}

# These names are rejected from serialized reports/sidecars.  ``html`` and
# ``body`` are accepted by the in-memory runtime boundary, then dropped before
# any adapter result is returned.
RAW_KEYS = frozenset(
    {
        "url",
        "uri",
        "path",
        "payload",
        "raw_payload",
        "request_body",
        "request_value",
        "query_value",
        "form_value",
        "response_body",
        "raw_response",
        "body",
        "body_text",
        "html",
        "markup",
        "source_code",
        "wire",
        "oracle_answer",
        "evaluator_answer",
    }
)

_ABSTRACT_MARKERS = ("http://", "https://", "<script", "union select")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bare_digest(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text if HEX_DIGEST_RE.fullmatch(text) else None


def _image_digest(value: Any) -> str | None:
    bare = _bare_digest(value)
    return f"sha256:{bare}" if bare else None


def _route_ref(route: Mapping[str, Any]) -> str:
    value = str(route.get("route_ref_sha256", "")).casefold()
    if HEX_DIGEST_RE.fullmatch(value):
        return value
    # A custom test/runtime route may omit the precomputed reference.  Derive
    # only from abstract shape fields; no path or route literal is accepted.
    return _json_digest(
        {
            "schema": SCHEMA_VERSION,
            "route_class": str(route.get("route_class", "unknown")),
            "method": str(route.get("method", "unknown")).upper(),
            "parameter_role": str(route.get("parameter_role", "unknown")),
            "encoding_chain": str(route.get("encoding_chain", "unknown")),
            "response_shape": str(route.get("response_shape", "unknown")),
            "script_surface": str(route.get("script_surface", "unknown")),
        }
    )


def _plan_seed_values(plan: Mapping[str, Any]) -> tuple[int, ...]:
    """Read seeds from either a future top-level field or planned rows."""

    direct = plan.get("seeds")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes, bytearray)):
        values = tuple(dict.fromkeys(int(seed) for seed in direct))
        if values:
            return values
    found: set[int] = set()
    collections = plan.get("planned_collections")
    if isinstance(collections, Mapping):
        for records in collections.values():
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
                for record in records:
                    if isinstance(record, Mapping) and record.get("seed") is not None:
                        try:
                            found.add(int(record["seed"]))
                        except (TypeError, ValueError):
                            pass
    return tuple(sorted(found))


def _find_raw(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in RAW_KEYS:
                found.append(f"{path}.{key_text}")
            found.extend(_find_raw(child, f"{path}.{key_text}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_find_raw(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        folded = value.casefold()
        if any(marker in folded for marker in _ABSTRACT_MARKERS):
            found.append(path)
    return found


def _scrub(value: Any, *, name: str = "artifact") -> None:
    found = _find_raw(value)
    if found:
        raise ValueError(f"PG-379 raw material in {name}: {', '.join(found[:4])}")


def _call_with_supported_kwargs(function: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Call an injected runtime function without guessing positional wire args."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**dict(kwargs))
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**dict(kwargs))
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
        and parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return function(**accepted)


def _factory_runtime(
    runtime_factory: Callable[..., Any],
    *,
    implementation_id: str,
    lane: str,
    seed: int,
    route: Mapping[str, Any],
    role: str,
    image_digest: str,
    attestation: Mapping[str, Any],
) -> Any:
    return _call_with_supported_kwargs(
        runtime_factory,
        {
            "implementation_id": implementation_id,
            "implementation": implementation_id,
            "lane": lane,
            "seed": int(seed),
            "route": dict(route),
            "route_ref_sha256": _route_ref(route),
            "role": role,
            "image_digest": image_digest,
            "attestation": dict(attestation),
            "network_mode": "none",
            "loopback_only": True,
            "projection": "whole_page",
        },
    )


def _normalize_attestation(
    *,
    lane: str,
    implementation_id: str,
    value: Mapping[str, Any] | None,
    expected_digest: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    if not isinstance(value, Mapping):
        return None, [f"{lane}:attestation_missing"]
    digest = _image_digest(value.get("image_digest"))
    if digest is None:
        failures.append(f"{lane}:image_digest_invalid_or_mutable")
    if expected_digest is not None and digest != expected_digest:
        failures.append(f"{lane}:image_digest_mismatch")
    if str(value.get("implementation_id", implementation_id)) != implementation_id:
        failures.append(f"{lane}:implementation_id_mismatch")
    bound = value.get("bound") is True
    built = value.get("image_built", value.get("built", False)) is True
    attested = value.get("image_attested", value.get("attested", False)) is True
    status = str(value.get("attestation_status", "")).casefold()
    explicit_attestation_flag = "image_attested" in value or "attested" in value
    if not bound:
        failures.append(f"{lane}:bound_required")
    if not built:
        failures.append(f"{lane}:image_build_unattested")
    attestation_ok = attested or (not explicit_attestation_flag and status in {"passed", "attested", "operator_reviewed"})
    if not attestation_ok:
        failures.append(f"{lane}:image_attestation_required")
    hash_fields: dict[str, str] = {}
    for field in ("runtime_module_sha256", "process_boundary_sha256", "source_digest"):
        digest_value = _bare_digest(value.get(field))
        if digest_value is None:
            failures.append(f"{lane}:{field}_invalid")
        else:
            hash_fields[field] = digest_value
    authorization_id = str(value.get("authorization_id", ""))
    if not authorization_id or len(authorization_id) > 128 or "\n" in authorization_id or "\r" in authorization_id:
        failures.append(f"{lane}:authorization_id_required")
    network_mode = str(value.get("network_mode", "")).casefold()
    if network_mode not in {"none", "loopback"}:
        failures.append(f"{lane}:network_mode_must_be_none_or_loopback")
    if value.get("external_network") is not False:
        failures.append(f"{lane}:external_network_must_be_false")
    if value.get("loopback_only") is not True:
        failures.append(f"{lane}:loopback_only_required")
    if value.get("bind_or_volume_mounts_allowed", value.get("bind_or_volume_mounts", True)) is not False:
        failures.append(f"{lane}:bind_or_volume_mounts_forbidden")
    if value.get("published_ports", True) is not False:
        failures.append(f"{lane}:published_ports_forbidden")
    if value.get("fresh_reset_contract") is not True:
        failures.append(f"{lane}:fresh_reset_contract_required")
    if value.get("independent_source_review") is not True:
        failures.append(f"{lane}:independent_source_review_required")
    if value.get("side_effects_enabled", False) is not False:
        failures.append(f"{lane}:side_effects_must_be_disabled")
    normalized = {
        "implementation_id": implementation_id,
        "lane": lane,
        "bound": bound,
        "attestation_status": status or "operator_reviewed",
        "image_digest": digest or "",
        "image_built": built,
        "image_attested": attestation_ok,
        **hash_fields,
        "authorization_id": authorization_id,
        "network_mode": network_mode,
        "external_network": False,
        "loopback_only": True,
        "bind_or_volume_mounts_allowed": False,
        "published_ports": False,
        "fresh_reset_contract": True,
        "independent_source_review": True,
        "side_effects_enabled": False,
    }
    return normalized, sorted(set(failures))


def _preflight(
    *,
    plan: Mapping[str, Any],
    attestations: Mapping[str, Any] | None,
    image_digest: str | Mapping[str, str] | None,
    runtime_factory: Callable[..., Any] | None,
    image_probe: Callable[..., Any] | None,
    environment: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    env = dict(environment) if environment is not None else dict(os.environ)
    failures: list[str] = []
    if env.get(OPERATOR_FLAG) != "1":
        failures.append(f"{OPERATOR_FLAG}=1_required")
    if runtime_factory is None:
        failures.append("runtime_factory_required")
    requirements = dict(plan.get("new_implementation_requirements") or {})
    normalized: dict[str, dict[str, Any]] = {}
    expected_map: dict[str, str | None] = {}
    if isinstance(image_digest, Mapping):
        expected_map = {str(key): _image_digest(value) for key, value in image_digest.items()}
        failures.extend(f"image_digest:{key}:invalid" for key, value in expected_map.items() if value is None)
    elif image_digest is not None:
        expected = _image_digest(image_digest)
        if expected is None:
            failures.append("image_digest:invalid_or_mutable")
        expected_map = {"train": expected, "holdout": expected}
    for lane in ("train", "holdout"):
        configured = dict(requirements.get(lane) or {})
        implementation_id = str(configured.get("implementation_id", f"pg379_{lane}_unbound"))
        attestation_value = dict(attestations.get(lane) or {}) if isinstance(attestations, Mapping) else None
        normalized_attestation, attestation_failures = _normalize_attestation(
            lane=lane,
            implementation_id=implementation_id,
            value=attestation_value,
            expected_digest=expected_map.get(lane),
        )
        failures.extend(attestation_failures)
        if normalized_attestation is not None:
            normalized[lane] = normalized_attestation
    train_id = str(requirements.get("train", {}).get("implementation_id", "")) if isinstance(requirements.get("train"), Mapping) else ""
    holdout_id = str(requirements.get("holdout", {}).get("implementation_id", "")) if isinstance(requirements.get("holdout"), Mapping) else ""
    if train_id and holdout_id and train_id == holdout_id:
        failures.append("implementation_ids_must_differ")
    if train_id and holdout_id and normalized.get("train") and normalized.get("holdout"):
        train_att = normalized["train"]
        holdout_att = normalized["holdout"]
        independent = any(
            train_att.get(field) != holdout_att.get(field)
            for field in ("image_digest", "runtime_module_sha256", "process_boundary_sha256", "source_digest")
        )
        if not independent:
            failures.append("train_holdout_attestations_not_independent")
    if image_probe is not None and not failures:
        for lane, attestation in normalized.items():
            try:
                probe_result = _call_with_supported_kwargs(
                    image_probe,
                    {
                        "lane": lane,
                        "implementation_id": attestation["implementation_id"],
                        "image_digest": attestation["image_digest"],
                        "attestation": dict(attestation),
                    },
                )
                probe_ok = bool(probe_result if isinstance(probe_result, bool) else dict(probe_result).get("attested", False) if isinstance(probe_result, Mapping) else False)
                if not probe_ok:
                    failures.append(f"{lane}:image_probe_failed")
            except Exception:
                failures.append(f"{lane}:image_probe_error")
    runtime_preflight = getattr(runtime_factory, "preflight", None) if runtime_factory is not None else None
    if callable(runtime_preflight) and not failures:
        for lane, attestation in normalized.items():
            try:
                result = _call_with_supported_kwargs(
                    runtime_preflight,
                    {
                        "lane": lane,
                        "implementation_id": attestation["implementation_id"],
                        "image_digest": attestation["image_digest"],
                        "attestation": dict(attestation),
                    },
                )
                if result is not True and not (isinstance(result, Mapping) and result.get("ready") is True):
                    failures.append(f"{lane}:runtime_factory_preflight_failed")
            except Exception:
                failures.append(f"{lane}:runtime_factory_preflight_error")
    gate = {
        "requested": True,
        "operator_flag": OPERATOR_FLAG,
        "operator_flag_present": env.get(OPERATOR_FLAG) == "1",
        "ready": not failures,
        "status": "passed" if not failures else "blocked_preflight",
        "blocked_reasons": sorted(set(failures)),
        "runtime_factory_bound": runtime_factory is not None,
        "image_probe_bound": image_probe is not None,
        "target_start_allowed": not failures,
    }
    return gate, normalized


def _normalize_reset(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = {
        "fresh_reset",
        "reset_id",
        "target_instance_digest",
        "network_mode",
        "external_network",
        "loopback_only",
        "state_clean",
        "volume_mount_count",
        "bind_or_volume_mount_count",
        "container_restart_used",
        "database_health_gate",
    }
    result = {str(key): value[key] for key in value if str(key) in allowed}
    if "volume_mount_count" not in result and "bind_or_volume_mount_count" in result:
        result["volume_mount_count"] = result.pop("bind_or_volume_mount_count")
    return result


def _reset_safe(reset: Mapping[str, Any] | None) -> bool:
    if not isinstance(reset, Mapping):
        return False
    target = _bare_digest(reset.get("target_instance_digest"))
    try:
        volume = int(reset.get("volume_mount_count", -1))
    except (TypeError, ValueError):
        return False
    return bool(
        reset.get("fresh_reset") is True
        and target is not None
        and reset.get("network_mode") in {"none", "loopback"}
        and reset.get("external_network") is False
        and reset.get("loopback_only") is True
        and reset.get("state_clean") is True
        and volume == 0
        and reset.get("container_restart_used") is False
    )


def _normalize_response(value: Any) -> tuple[dict[str, Any], bytes, bool]:
    """Reduce one runtime response; raw body is returned only to the caller."""

    if isinstance(value, (bytes, bytearray)):
        body = bytes(value)
        metadata: Mapping[str, Any] = {}
    elif isinstance(value, str):
        body = value.encode("utf-8", errors="replace")
        metadata = {}
    elif isinstance(value, Mapping):
        metadata = value
        raw_body = value.get("body", value.get("html", b""))
        if isinstance(raw_body, str):
            body = raw_body.encode("utf-8", errors="replace")
        elif isinstance(raw_body, (bytes, bytearray)):
            body = bytes(raw_body)
        elif raw_body in (None, ""):
            body = b""
        else:
            raise ValueError("runtime response body must be bytes/text")
    else:
        raise ValueError("runtime response must be a mapping, bytes, or text")
    if len(body) > MAX_PAGE_BYTES:
        raise ValueError("runtime page exceeds bounded in-memory limit")
    status = int(metadata.get("status", 0) or 0)
    status_class = str(metadata.get("status_class", ""))
    if not status_class:
        status_class = f"{status // 100}xx" if 100 <= status < 600 else "transport_error"
    content_type = str(metadata.get("content_type_class", metadata.get("content_type", "text/html" if body else "unknown")))
    action = {
        "method": str(metadata.get("method", "GET")).upper(),
        "status": status,
        "status_class": status_class,
        "content_type_class": content_type,
        "location_class": str(metadata.get("location_class", "loopback" if status in {301, 302, 303, 307, 308} else "none")),
    }
    typed_value = metadata.get("typed_effect_confirmed", metadata.get("typed_effect", metadata.get("typed", False)))
    typed = typed_value is True
    return action, body, typed


def _runtime_request(runtime: Any, *, method: str, route: Mapping[str, Any], role: str, phase: str) -> tuple[dict[str, Any], bytes, bool]:
    function = getattr(runtime, "request", None)
    if not callable(function):
        function = getattr(runtime, "observe", None)
    if not callable(function):
        raise ValueError("runtime must expose request(...) or observe(...)")
    value = _call_with_supported_kwargs(
        function,
        {
            "method": str(method).upper(),
            "route": dict(route),
            "route_ref_sha256": _route_ref(route),
            "role": role,
            "phase": phase,
            "surface": "whole_page",
            "projection": "whole_page",
            "form_body": b"" if str(method).upper() == "POST" else None,
            "body": b"" if str(method).upper() == "POST" else None,
        },
    )
    return _normalize_response(value)


def _docker_response_json(body: bytes) -> Mapping[str, Any]:
    """Parse an evaluator-only health/route projection from a container."""

    if not body:
        return {}
    try:
        value = json.loads(body.decode("utf-8", errors="replace"))
    except (TypeError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


_INLINE_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_SCRIPT_TAG_RE = re.compile(r"<script\b([^>]*)>", re.IGNORECASE)


def _project_page_javascript(html: str) -> dict[str, Any]:
    """Project inline page JS to abstract tokens while discarding source text."""

    if not isinstance(html, str) or not html.strip():
        return {
            "script_count": 0,
            "source_text_stored": False,
            "js_semantic_tokens": ["js_script_count=zero", "js_source=absent"],
            "javascript_context": {"source_kind": "absent", "sink_context": "sink_not_observed"},
            "next_action": "ask",
            "safe_to_send": False,
            "ask_reason": "javascript_not_observed",
        }
    tags = list(_SCRIPT_TAG_RE.finditer(html))
    inline = [match.group(2) for match in _INLINE_SCRIPT_RE.finditer(html) if match.group(2).strip()]
    external = any(re.search(r"\bsrc\s*=", match.group(1), re.IGNORECASE) for match in tags)
    if not tags:
        return {
            "script_count": 0,
            "source_text_stored": False,
            "js_semantic_tokens": ["js_script_count=zero", "js_source=absent"],
            "javascript_context": {"source_kind": "absent", "sink_context": "sink_not_observed"},
            "next_action": "ask",
            "safe_to_send": False,
            "ask_reason": "javascript_not_observed",
        }
    if external and not inline:
        return {
            "script_count": len(tags),
            "source_text_stored": False,
            "js_semantic_tokens": ["js_script_count=present", "js_external_script=present", "js_source=not_observed"],
            "javascript_context": {"source_kind": "external_script", "sink_context": "sink_not_observed", "external_or_dynamic_loader": True},
            "next_action": "ask",
            "safe_to_send": False,
            "ask_reason": "external_script_not_projected",
        }
    bounded = "\n".join(inline)
    if len(bounded.encode("utf-8")) > 64 * 1024:
        return {
            "script_count": len(tags),
            "source_text_stored": False,
            "js_semantic_tokens": ["js_script_count=present", "js_source=not_observed"],
            "javascript_context": {"source_kind": "bounded_script_over_limit", "sink_context": "sink_not_observed"},
            "next_action": "ask",
            "safe_to_send": False,
            "ask_reason": "javascript_source_limit",
        }
    projection = project_js_source(bounded, local_fixture=True)
    projection["script_count"] = len(tags)
    projection["source_text_stored"] = False
    return projection


def _typed_from_bounded_projection(
    *,
    action: Mapping[str, Any],
    body: bytes,
    route: Mapping[str, Any],
    role: str,
) -> bool:
    """Recognize only the fixture's bounded response-shape evidence.

    Implementation B renders HTML for several route classes.  The evaluator
    may inspect the in-memory DOM-shape marker and input-class attribute, but
    it must never persist or expose the markup.  JSON projections continue to
    use their abstract ``response_shape`` field.  Negative roles are always
    non-typed regardless of what a fixture returns.
    """

    if role == "negative" or int(action.get("status", 0) or 0) >= 400:
        return False
    expected_shape = str(route.get("response_shape", ""))
    status = int(action.get("status", 0) or 0)
    if expected_shape == "redirect_shape":
        return 300 <= status < 400 and str(action.get("location_class", "none")) == "loopback"
    projected = _docker_response_json(body)
    if projected:
        return bool(
            projected.get("typed_shape_delta") is True
            or projected.get("response_shape") == expected_shape
            or projected.get("state_delta") is True
        )
    if str(action.get("content_type_class", "")).casefold() not in {"text/html", "application/xhtml+xml"}:
        return False
    text = body.decode("utf-8", errors="replace")
    shape_match = re.search(r'data-pg379-b-shape="([a-z_]+)"', text)
    input_match = re.search(r'data-input-class="([a-z_]+)"', text)
    return bool(
        shape_match
        and input_match
        and shape_match.group(1) == expected_shape
        and input_match.group(1) == "safe_canary"
    )


def _docker_exec_http(
    *,
    name: str,
    language: str,
    port: int,
    method: str,
    path: str,
    body: bytes = b"",
    content_type: str = "application/json",
) -> tuple[dict[str, Any], bytes]:
    """Issue one loopback request from inside a network-none container.

    A network-none container has no host port to publish.  The reviewed
    fixtures ship Python or Node standard libraries, so the evaluator uses a
    one-shot in-container HTTP client selected by the attested runtime
    language.  The response is decoded in memory and never serialized by this
    module.
    """

    method = str(method).upper()
    encoded_body = base64.b64encode(bytes(body)).decode("ascii")
    content_type = str(content_type or "application/json")
    if language == "python":
        script = (
            "import base64,json,sys,urllib.error,urllib.request\n"
            "m,p,b,port,ct=sys.argv[1],sys.argv[2],base64.b64decode(sys.argv[3]),int(sys.argv[4]),sys.argv[5]\n"
            "u='http://127.0.0.1:'+str(port)+p\n"
            "r=urllib.request.Request(u,data=(b if m=='POST' else None),method=m,headers={'Content-Type':ct})\n"
            "class NoRedirect(urllib.request.HTTPRedirectHandler):\n"
            "    def redirect_request(self,*args,**kwargs): return None\n"
            "opener=urllib.request.build_opener(NoRedirect)\n"
            "o=None; s=0; h={}; x=b''\n"
            "try:\n"
            "    with opener.open(r,timeout=12) as z:\n"
            "        s=int(z.status)\n"
            "        h={str(k).lower():str(v) for k,v in z.headers.items()}\n"
            "        x=z.read(2097152)\n"
            "except urllib.error.HTTPError as e:\n"
            "    s=int(e.code)\n"
            "    h={str(k).lower():str(v) for k,v in e.headers.items()}\n"
            "    x=e.read(2097152)\n"
            "except Exception as e:\n"
            "    o=type(e).__name__\n"
            "    s=0\n"
            "    x=b''\n"
            "print(json.dumps({'status':s,'headers':h,'body_b64':base64.b64encode(x).decode('ascii'),'error_class':o},separators=(',',':')))"
        )
        executable = "python"
    elif language == "node":
        script = (
            "const http=require('http');const [m,p,b64,port,ct]=process.argv.slice(1);"
            "const b=Buffer.from(b64,'base64');const q=http.request({host:'127.0.0.1',port:Number(port),path:p,method:m,headers:{'Content-Type':ct,'Content-Length':b.length}},r=>{"
            "const a=[];r.on('data',x=>{if(Buffer.concat(a).length<2097152)a.push(x)});r.on('end',()=>process.stdout.write(JSON.stringify({status:r.statusCode||0,headers:r.headers||{},body_b64:Buffer.concat(a).subarray(0,2097152).toString('base64')})))});"
            "q.on('error',e=>process.stdout.write(JSON.stringify({status:0,headers:{},body_b64:'',error_class:String(e.code||e.name||'error')})));"
            "if(m==='POST')q.write(b);q.end();"
        )
        executable = "node"
    else:
        raise ValueError("Docker runtime language must be python or node")
    eval_flag = "-c" if language == "python" else "-e"
    completed = subprocess.run(
        ["docker", "exec", name, executable, eval_flag, script, method, path, encoded_body, str(int(port)), content_type],
        check=False,
        capture_output=True,
        timeout=20.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("loopback_http_exec_failed")
    try:
        envelope = json.loads(bytes(completed.stdout).decode("utf-8", errors="replace"))
    except (TypeError, ValueError, UnicodeError) as error:
        raise RuntimeError("loopback_http_projection_invalid") from error
    if not isinstance(envelope, Mapping):
        raise RuntimeError("loopback_http_projection_invalid")
    try:
        response_body = base64.b64decode(str(envelope.get("body_b64", "")), validate=True)[:MAX_PAGE_BYTES]
    except (ValueError, TypeError):
        response_body = b""
    headers = envelope.get("headers") if isinstance(envelope.get("headers"), Mapping) else {}
    status = int(envelope.get("status", 0) or 0)
    location = str(headers.get("location", ""))
    action = {
        "method": method,
        "status": status,
        "status_class": f"{status // 100}xx" if 100 <= status < 600 else "transport_error",
        "content_type_class": str(headers.get("content-type", "unknown")).split(";", 1)[0].casefold() or "unknown",
        "location_class": "loopback" if location.startswith(("/", "http://127.0.0.1", "http://localhost")) else "none",
        "headers": {str(key).casefold(): "present" for key in headers},
    }
    return action, response_body


class _DockerWholePageRuntime:
    """Disposable runtime bound to an operator-reviewed fixture manifest."""

    def __init__(
        self,
        *,
        image_ref: str,
        image_digest: str,
        runtime_language: str,
        port: int,
        route_map: Mapping[str, Mapping[str, Any]],
        implementation_id: str,
        lane: str,
        seed: int,
        role: str,
        route: Mapping[str, Any],
    ) -> None:
        digest = _image_digest(image_digest)
        if digest is None or "@" not in str(image_ref) or not str(image_ref).endswith(f"@{digest}"):
            raise ValueError("Docker runtime requires an immutable image_ref matching the attested digest")
        if str(runtime_language) not in {"python", "node"}:
            raise ValueError("Docker runtime language is not attested")
        if int(port) < 1 or int(port) > 65535:
            raise ValueError("Docker runtime port is invalid")
        self.image_ref = str(image_ref)
        self.image_digest = digest
        self.runtime_language = str(runtime_language)
        self.port = int(port)
        self.route_map = {str(key): dict(value) for key, value in route_map.items() if isinstance(value, Mapping)}
        self.implementation_id = implementation_id
        self.lane = lane
        self.seed = int(seed)
        self.role = role
        self.route = dict(route)
        self.name = f"pg379-nn-{lane}-{self.seed}-{_route_ref(route)[:12]}-{role}"
        self.started = False

    def _exec(self, method: str, path: str, body: bytes = b"") -> tuple[dict[str, Any], bytes]:
        return _docker_exec_http(
            name=self.name,
            language=self.runtime_language,
            port=self.port,
            method=method,
            path=path,
            body=body,
            content_type="application/json" if body.lstrip().startswith(b"{") else "application/x-www-form-urlencoded",
        )

    def start(self) -> dict[str, Any]:
        if os.environ.get(OPERATOR_FLAG) != "1":
            raise RuntimeError(f"{OPERATOR_FLAG}=1_required_before_docker_start")
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.name,
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=64m",
                self.image_ref,
            ],
            check=True,
            capture_output=True,
            timeout=60.0,
        )
        self.started = True
        deadline = time.monotonic() + 30.0
        health: Mapping[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                action, body = self._exec("GET", "/__health")
                health = _docker_response_json(body)
                if int(action.get("status", 0) or 0) == 200 and health:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        if not health:
            raise RuntimeError("docker_fixture_health_timeout")
        try:
            _, reset_body = self._exec("POST", "/__reset")
            reset = dict(_docker_response_json(reset_body))
        except Exception:
            reset = {}
        reported_digest = _bare_digest(reset.get("target_instance_digest") or reset.get("instance_digest") or health.get("target_instance_digest") or health.get("instance_digest"))
        # Bind the evaluator reset to this disposable container identity even
        # when a stateless fixture reports its immutable source digest for
        # every process.  The resulting digest is an abstract attestation,
        # not a raw container name or response value.
        target_digest = _json_digest(
            {
                "implementation_id": self.implementation_id,
                "container_identity": self.name,
                "reported_digest": reported_digest or "unknown",
                "reset": reset,
            }
        )
        return {
            "fresh_reset": True,
            "reset_id": _json_digest({"name": self.name, "health": dict(health), "reset": reset}),
            "target_instance_digest": target_digest,
            "network_mode": "none",
            "external_network": False,
            "loopback_only": True,
            "state_clean": reset.get("state_clean", health.get("state_clean", True)) is True,
            "volume_mount_count": 0,
            "container_restart_used": False,
        }

    def _route_request(self, method: str) -> tuple[bytes, str]:
        route_class = str(self.route.get("route_class", ""))
        metadata = dict(self.route_map.get(route_class) or {})
        path = str(metadata.get("path", ""))
        if not path.startswith("/") or "http://" in path.casefold() or "https://" in path.casefold():
            raise ValueError("Docker route manifest path is missing or external")
        parameter = str(metadata.get("parameter", ""))
        if not parameter:
            role = str(self.route.get("parameter_role", ""))
            if self.runtime_language == "node":
                # Implementation B's manifest uses the semantic role itself
                # as the reviewed field name.
                parameter = role or "value"
            else:
                # Implementation A exposes only bounded semantic roles in its
                # manifest; bind those roles to the fixture's reviewed field
                # name inside the evaluator, never in model context or rows.
                parameter = {
                    "query_text": "q",
                    "fragment_identifier": "fragment_identifier",
                    "json_value": "value",
                    "view_mode": "mode",
                    "form_field": "value",
                    "attribute_value": "value",
                    "structured_value": "value",
                    "record_cursor": "record",
                }.get(role, "value")
        # Each fixture has a source-attested safe-canary namespace.  This
        # value exists only inside the evaluator request and is never written
        # to a row, model context, or report.
        prefix = "PG379B_CANARY" if self.runtime_language == "node" else "PG379_CANARY"
        value = f"{prefix}_{self.seed}_{self.role}"
        input_source = str(metadata.get("input_source", "query"))
        if "<value>" in path:
            path = path.replace("<value>", quote(value, safe=""))
        if str(method).upper() == "GET":
            if "<value>" not in path and input_source in {"query", "fragment"}:
                path = f"{path}{'&' if '?' in path else '?'}{urlencode({parameter: value})}"
            return b"", path
        if input_source == "json" or "json" in str(self.route.get("encoding_chain", "")).casefold():
            return json.dumps({parameter: value}, separators=(",", ":")).encode("utf-8"), path
        return urlencode({parameter: value}).encode("utf-8"), path

    def request(self, *, method: str, **_: Any) -> dict[str, Any]:
        body, path = self._route_request(method)
        action, response_body = self._exec(str(method).upper(), path, body)
        typed = _typed_from_bounded_projection(
            action=action,
            body=response_body,
            route=self.route,
            role=self.role,
        )
        action["typed_effect"] = typed
        return {**action, "body": response_body}

    def stop(self) -> None:
        if self.started:
            subprocess.run(["docker", "rm", "-f", self.name], check=False, capture_output=True, timeout=60.0)
            self.started = False


class _DockerRuntimeFactory:
    is_docker_runtime = True

    def __init__(self, specs: Mapping[str, Any]) -> None:
        self.specs: dict[str, dict[str, Any]] = {}
        for lane in ("train", "holdout"):
            spec = dict(specs.get(lane) or {}) if isinstance(specs, Mapping) else {}
            route_values = spec.get("routes")
            route_map: dict[str, Mapping[str, Any]] = {}
            if isinstance(route_values, Mapping):
                route_map = {str(key): value for key, value in route_values.items() if isinstance(value, Mapping)}
            elif isinstance(route_values, Sequence) and not isinstance(route_values, (str, bytes, bytearray)):
                route_map = {str(value.get("route_class")): value for value in route_values if isinstance(value, Mapping) and value.get("route_class")}
            if not spec.get("image_ref") or not route_map:
                raise ValueError(f"Docker runtime spec for {lane} requires image_ref and route map")
            image_ref = str(spec["image_ref"])
            image_ref_digest = image_ref.rsplit("@", 1)[-1] if "@" in image_ref else ""
            if _image_digest(image_ref_digest) is None:
                raise ValueError(f"Docker runtime spec for {lane} requires immutable @sha256 image_ref")
            language = str(spec.get("runtime_language", ""))
            if language not in {"python", "node"}:
                raise ValueError(f"Docker runtime spec for {lane} has unsupported runtime language")
            port = int(spec.get("port", 0))
            if port < 1 or port > 65535:
                raise ValueError(f"Docker runtime spec for {lane} has invalid port")
            for route_class, route_value in route_map.items():
                route_path = str(route_value.get("path", ""))
                if not route_path.startswith("/") or "http://" in route_path.casefold() or "https://" in route_path.casefold():
                    raise ValueError(f"Docker runtime spec route {route_class} is not a loopback path")
            self.specs[lane] = {
                "image_ref": image_ref,
                "runtime_language": language,
                "port": port,
                "routes": route_map,
            }

    def __call__(self, *, implementation_id: str, lane: str, seed: int, role: str, route: Mapping[str, Any], image_digest: str, **_: Any) -> _DockerWholePageRuntime:
        spec = self.specs.get(str(lane))
        if spec is None:
            raise ValueError("Docker runtime lane is not bound")
        return _DockerWholePageRuntime(
            image_ref=spec["image_ref"],
            image_digest=image_digest,
            runtime_language=spec["runtime_language"],
            port=spec["port"],
            route_map=spec["routes"],
            implementation_id=implementation_id,
            lane=lane,
            seed=int(seed),
            role=role,
            route=route,
        )

    def preflight(self, *, lane: str, image_digest: str, **_: Any) -> bool:
        spec = self.specs.get(str(lane))
        if spec is None:
            return False
        return spec["image_ref"].endswith(f"@{_image_digest(image_digest) or ''}")

    def image_probe(self, *, lane: str, image_digest: str, **_: Any) -> bool:
        """Read-only local Docker image inspection; never pulls or starts."""

        spec = self.specs.get(str(lane))
        digest = _image_digest(image_digest)
        if spec is None or digest is None or not spec["image_ref"].endswith(f"@{digest}"):
            return False
        try:
            completed = subprocess.run(
                ["docker", "image", "inspect", spec["image_ref"]],
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
        except Exception:
            return False
        if completed.returncode != 0:
            return False
        output = f"{completed.stdout}\n{completed.stderr}".casefold()
        return digest.casefold() in output


def build_pg379_docker_runtime_factory(specs: Mapping[str, Any]) -> Callable[..., Any]:
    """Bind a reviewed Docker image/route manifest to the live collector.

    ``specs`` is evaluator-side configuration (image refs include immutable
    ``@sha256:`` digests and route paths); it is not copied to rows, context,
    reports, or sidecars.  The caller must still provide matching operator
    attestations and ``PG379_LOCAL_DOCKER_EVAL=1``.
    """

    return _DockerRuntimeFactory(specs)


def _action_method(expected: str, role: str) -> str:
    expected = str(expected).upper()
    if role == "negative":
        return "POST" if expected == "GET" else "GET"
    return expected


def _incomplete_capture(*, implementation_id: str, lane: str, seed: int, role: str, route: Mapping[str, Any], reason: str) -> dict[str, Any]:
    evidence = _json_digest(
        {
            "schema": SCHEMA_VERSION,
            "implementation_id": implementation_id,
            "lane": lane,
            "seed": int(seed),
            "route_ref_sha256": _route_ref(route),
            "role": role,
            "reason": reason,
        }
    )
    return {
        "seed": int(seed),
        "role": role,
        "route_ref_sha256": _route_ref(route),
        "expected_method": str(route.get("method", "unknown")).upper(),
        "action_method": "unknown",
        "reset": None,
        "html": None,
        "headers": None,
        "request_projection": None,
        "response_projection": None,
        "failure_projection": None,
        "belief_projection": None,
        "role_input": {
            "sent": False,
            "available": False,
            "executed": False,
            "typed_effect_confirmed": False,
            "effect_class": "unknown",
            "projection": {},
            "evidence_sha256": evidence,
            "non_destructive": True,
        },
        "typed": False,
        "evidence_sha256": evidence,
        "capture_failure": reason,
        "implementation_id": implementation_id,
        "lane": lane,
        "runtime_started": False,
    }


def _capture_episode(
    *,
    runtime_factory: Callable[..., Any],
    implementation_id: str,
    lane: str,
    seed: int,
    role: str,
    route: Mapping[str, Any],
    image_digest: str,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    runtime: Any = None
    capture_result: dict[str, Any] | None = None
    runtime_started = False
    try:
        runtime = _factory_runtime(
            runtime_factory,
            implementation_id=implementation_id,
            lane=lane,
            seed=seed,
            route=route,
            role=role,
            image_digest=image_digest,
            attestation=attestation,
        )
        if runtime is None:
            raise ValueError("runtime_factory returned None")
        start = getattr(runtime, "start", None)
        if not callable(start):
            raise ValueError("runtime must expose start()")
        reset = _normalize_reset(_call_with_supported_kwargs(start, {}))
        if not _reset_safe(reset):
            raise ValueError("unsafe_or_incomplete_fresh_reset")
        runtime_started = True
        expected = str(route.get("method", "unknown")).upper()
        action_method = _action_method(expected, role)
        baseline_action, baseline_body, baseline_typed = _runtime_request(runtime, method="GET", route=route, role=role, phase="baseline")
        if action_method == "GET":
            action, body, typed = baseline_action, baseline_body, baseline_typed
        else:
            action, body, typed = _runtime_request(runtime, method=action_method, route=route, role=role, phase="candidate")
        # A negative control deliberately uses the wrong transport method.  A
        # runtime cannot turn that control into a typed positive by accident.
        if role == "negative":
            typed = False
        html_body = body if action_method == "GET" else baseline_body
        javascript_context = _project_page_javascript(html_body.decode("utf-8", errors="replace"))
        request = _request_projection(
            method=action_method,
            body_length=40 if action_method == "POST" else 0,
            html_body=html_body,
        )
        response = _response_projection(action=action, body=body, typed=typed, role=role)
        failure = _failure_projection_for_role(expected_method=expected, action_method=action_method, typed=typed, role=role)
        belief = _belief(
            expected_method=expected,
            action_method=action_method,
            typed=typed,
            role=role,
            csrf_class=request.get("csrf_presence_class", "unknown"),
            cookie_class=request.get("cookie_presence_class", "unknown"),
        )
        evidence = _json_digest(
            {
                "schema": SCHEMA_VERSION,
                "implementation_id": implementation_id,
                "lane": lane,
                "seed": int(seed),
                "route_ref_sha256": _route_ref(route),
                "role": role,
                "reset_id": str(reset.get("reset_id", "")),
                "expected_method": expected,
                "action_method": action_method,
                "typed": bool(typed),
                "status_class": str(action.get("status_class", "transport_error")),
                "content_type_class": str(action.get("content_type_class", "unknown")),
                "body_length": len(body),
            }
        )
        capture_result = {
            "seed": int(seed),
            "role": role,
            "route_ref_sha256": _route_ref(route),
            "expected_method": expected,
            "action_method": action_method,
            "reset": reset,
            "html": html_body.decode("utf-8", errors="replace"),
            "javascript_context_projection": javascript_context,
            "headers": _abstract_headers(action),
            "request_projection": request,
            "response_projection": response,
            "failure_projection": failure,
            "belief_projection": belief,
            "role_input": _role_input(
                role=role,
                expected_method=expected,
                action_method=action_method,
                action=action,
                body=body,
                typed=typed,
                evidence=evidence,
            ),
            "typed": bool(typed),
            "evidence_sha256": evidence,
            "implementation_id": implementation_id,
            "lane": lane,
            "runtime_started": True,
        }
        return capture_result
    except Exception as error:
        capture_result = _incomplete_capture(
            implementation_id=implementation_id,
            lane=lane,
            seed=seed,
            role=role,
            route=route,
            reason=f"runtime_{type(error).__name__}",
        )
        capture_result["runtime_started"] = runtime_started
        return capture_result
    finally:
        if runtime is not None:
            stop = getattr(runtime, "stop", None)
            if callable(stop):
                try:
                    _call_with_supported_kwargs(stop, {})
                except Exception:
                    # Preserve teardown failures in the in-memory episode so
                    # the lifecycle gate cannot report a clean replay when a
                    # target may have remained alive.
                    if capture_result is not None:
                        capture_result["capture_failure"] = "runtime_teardown_error"


def _incomplete_sidecar(*, record_id: str, replay: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        key: False
        for key in (
            "candidate_present",
            "reference_present",
            "negative_present",
            "candidate_available",
            "reference_available",
            "negative_available",
            "typed_effect",
            "negative_control_clean",
            "reference_agreement",
            "replay_consistent",
            "fresh_reset",
            "evidence_hashes",
            "non_destructive",
        )
    }
    unsigned = {
        "schema_version": "pg331-evaluator-sidecar-v1",
        "record_id": record_id,
        "evaluator_id": "pg379-dynamic-whole-page-evaluator-v1",
        "checks": checks,
        "reasons": ["incomplete_episode"],
        "typed_effect_confirmed": False,
        "effect_class": "unknown",
        "negative_control_clean": False,
        "reference_agreement": False,
        "replay_consistent": False,
        "hard_negative": False,
    }
    digest = sha256_json(unsigned)
    return {
        **unsigned,
        "evidence_sha256": digest,
        "evidence_hash": digest,
        "evidence_hash_valid": True,
        "confirmed_positive": False,
        "raw_payload_stored": False,
        "raw_response_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "replay": {
            "present": True,
            "typed_effect_confirmed": bool(replay.get("typed")),
            "evidence_sha256": str(replay.get("evidence_sha256", "")),
            "fresh_reset": False,
        },
    }


def _build_sidecar(
    *,
    record_id: str,
    captures: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = dict(captures.get("candidate") or {})
    reference = dict(captures.get("reference") or {})
    negative = dict(captures.get("negative") or {})
    replay = dict(captures.get("replay") or {})
    reset = candidate.get("reset")
    role_inputs = {
        role: dict(dict(captures.get(role) or {}).get("role_input") or {})
        for role in SOURCE_ROLES
    }
    if not all(_reset_safe(captures.get(role, {}).get("reset")) for role in SOURCE_ROLES) or not _reset_safe(reset):
        return _incomplete_sidecar(record_id=record_id, replay=replay)
    candidate_typed = bool(candidate.get("typed", candidate.get("typed_effect_confirmed", False)))
    reference_typed = bool(reference.get("typed", reference.get("typed_effect_confirmed", False)))
    negative_typed = bool(negative.get("typed", negative.get("typed_effect_confirmed", False)))
    replay_typed = bool(replay.get("typed", replay.get("typed_effect_confirmed", False)))
    try:
        sidecar = build_pg331_evaluator_sidecar(
            record_id=record_id,
            reset=dict(reset),
            candidate=role_inputs["candidate"],
            reference=role_inputs["reference"],
            negative=role_inputs["negative"],
            replay_consistent=bool(candidate_typed and replay_typed),
            reference_agreement=bool(candidate_typed and reference_typed),
            negative_control_clean=not negative_typed,
            evaluator_id="pg379-dynamic-whole-page-evaluator-v1",
            hard_negative=False,
        )
    except Exception:
        return _incomplete_sidecar(record_id=record_id, replay=replay)
    # A source-row collection sidecar is never a vulnerability claim, even if
    # all evaluator checks happen to agree.  Keep the bounded evidence hash but
    # close all promotion/claim paths explicitly.
    sidecar["confirmed_positive"] = False
    sidecar["training_eligible"] = False
    sidecar["memory_promotion_allowed"] = False
    sidecar["payload_catalog_promotion_allowed"] = False
    sidecar["vulnerability_claim_allowed"] = False
    sidecar["replay"] = {
        "present": True,
        "typed_effect_confirmed": replay_typed,
        "evidence_sha256": str(replay.get("evidence_sha256", "")),
        "fresh_reset": _reset_safe(replay.get("reset")),
    }
    return sidecar


def _target_slots(*, route: Mapping[str, Any], sidecar: Mapping[str, Any], role: str = "candidate") -> dict[str, Any]:
    method = str(route.get("method", "unknown")).upper()
    checks = dict(sidecar.get("checks") or {})
    encoding = str(route.get("encoding_chain", "unknown")).casefold().replace("-", "_").replace(" ", "_")
    if encoding not in {
        "identity",
        "url_percent",
        "fragment",
        "json_string",
        "query_parameter",
        "form_urlencoded",
        "json_object_then_utf8",
        "form_urlencoded_then_url_percent",
        "query_parameter_then_url_percent",
    }:
        encoding = "unknown"
    slots = {
        "question": "none" if checks.get("typed_effect") else "ask_typed",
        "ask_reason": "typed_observation_available" if checks.get("typed_effect") else "missing_typed_observation",
        "next_action": "assemble_rule_ir" if checks.get("typed_effect") else "ask_typed",
        "repair_action": "none" if checks.get("typed_effect") else "observe",
        "transport_ref": "post_surface" if method == "POST" else "get_surface" if method == "GET" else "unknown",
        "field_role_ref": "parameter_role",
        "encoding_ref": encoding,
        "syntax_category_ref": "structured_value",
        "probe_variant_ref": {
            "candidate": "source_attested_candidate",
            "reference": "reference",
            "negative": "negative_control",
            "replay": "unknown",
        }.get(role, "unknown"),
        "safe_to_send": False,
        "payload_shape_ref": "state_transition_marker" if method == "POST" else "html_text_marker",
        "oracle_ref": "typed_effect" if checks.get("typed_effect") else "unknown",
        "negative_control_presence_ref": "matched_triplet" if checks.get("negative_control_clean") else "unknown",
    }
    if set(slots) != set(SLOTS):
        raise AssertionError("PG-379 target slot inventory drift")
    return slots


def _materialize_row(
    *,
    capture: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    route: Mapping[str, Any],
    lane: str,
    implementation_id: str,
    image_digest: str,
    authorization_id: str,
    seed: int,
    role: str,
) -> dict[str, Any]:
    reset = capture.get("reset") if isinstance(capture.get("reset"), Mapping) else None
    source_meta: dict[str, Any] | None = None
    record_id = f"pg379-{lane}-{seed}-{_route_ref(route)[:16]}-{role}"
    if reset is not None and sidecar:
        source_meta = {
            "source_id": "pg379-dynamic-local",
            "implementation": implementation_id,
            "collector_id": SCHEMA_VERSION,
            "authorization_id": str(authorization_id),
            "image_digest": _bare_digest(image_digest),
            "source_digest": _json_digest(
                {
                    "implementation_id": implementation_id,
                    "lane": lane,
                    "seed": int(seed),
                    "route_ref_sha256": _route_ref(route),
                    "role": role,
                    "evidence_sha256": str(capture.get("evidence_sha256", "")),
                }
            ),
        }
    try:
        row = capture_pg377_webgoat_source_row(
            html=str(capture.get("html")) if isinstance(capture.get("html"), str) else None,
            headers=dict(capture.get("headers") or {}) if isinstance(capture.get("headers"), Mapping) else None,
            request_projection=dict(capture.get("request_projection") or {}) if isinstance(capture.get("request_projection"), Mapping) else None,
            response_projection=dict(capture.get("response_projection") or {}) if isinstance(capture.get("response_projection"), Mapping) else None,
            role=role,
            reset=dict(reset) if isinstance(reset, Mapping) else None,
            evaluator_sidecar=dict(sidecar),
            failure_projection=dict(capture.get("failure_projection") or {}) if isinstance(capture.get("failure_projection"), Mapping) else None,
            belief_projection=dict(capture.get("belief_projection") or {}) if isinstance(capture.get("belief_projection"), Mapping) else None,
            javascript_context_projection=dict(capture.get("javascript_context_projection") or {}) if isinstance(capture.get("javascript_context_projection"), Mapping) else None,
            post_supported=True,
            source_meta=source_meta,
            record_id=record_id if source_meta is not None else None,
            split="train" if lane == "train" else "implementation_holdout",
            # This collector has not passed the independent source audit.  It
            # therefore cannot create a training-eligible row.
            operator_reviewed=False,
            hard_negative=role == "negative",
        )
    except Exception as error:
        # Keep an abstract ASK-safe wrapper for a malformed/incomplete page;
        # never guess a source row or elevate a runtime error.
        row = capture_pg377_webgoat_source_row(
            html=None,
            headers=None,
            request_projection=None,
            response_projection=None,
            role=role,
            reset=None,
            evaluator_sidecar=dict(sidecar),
            failure_projection=None,
            belief_projection=None,
            post_supported=True,
        )
        row["capture_materialization_failure"] = type(error).__name__
    validation = validate_pg377_webgoat_source_row(row)
    row["adapter_validation"] = {
        "valid": bool(validation.get("valid")),
        "failures": list(validation.get("failures") or []),
    }
    # ``adapter_validation`` is collector bookkeeping, not part of the
    # adapter's canonical projection.  If it is retained on the in-memory
    # wrapper, refresh the wrapper hash after adding it; otherwise every
    # otherwise-valid row fails its own record_sha256 check on the next
    # validation pass.
    if isinstance(row.get("record_sha256"), str):
        row["record_sha256"] = sha256_json({key: value for key, value in row.items() if key != "record_sha256"})
    validation_after = validate_pg377_webgoat_source_row(row)
    row["adapter_validation"] = {
        "valid": bool(validation_after.get("valid")),
        "failures": list(validation_after.get("failures") or []),
    }
    row["training_eligible"] = False
    return row


def _empty_report(*, plan: Mapping[str, Any], gate: Mapping[str, Any], status: str, mode: str) -> dict[str, Any]:
    seeds = list(_plan_seed_values(plan))
    routes = list(plan.get("route_shape_matrix") or [])
    implementations = dict(plan.get("new_implementation_requirements") or {})
    impl_count = len(implementations)
    route_count = len(routes)
    episode_count = impl_count * len(seeds) * route_count * len(ALL_ROLES)
    source_count = impl_count * len(seeds) * route_count * len(SOURCE_ROLES)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": mode,
        "plan_sha256": str(plan.get("plan_sha256", "")),
        "live_gate": dict(gate),
        "objective": {
            "whole_page_projection": True,
            "pg377_adapter": True,
            "target_slot_count": len(SLOTS),
            "training_rows_created": False,
            "raw_payload_response_context": False,
        },
        "counts": {
            "implementation_count": impl_count,
            "seed_count": len(seeds),
            "route_count": route_count,
            "role_episode_expected": episode_count,
            "role_episode_observed": 0,
            "runtime_started_count": 0,
            "source_row_expected": source_count,
            "source_row_count": 0,
            "valid_source_row_count": 0,
            "training_eligible_count": 0,
            "target_sidecar_count": 0,
            "typed_role_count": 0,
            "negative_violation_count": 0,
            "failure_observed_count": 0,
            "failure_action_changed_count": 0,
            "fresh_reset_failure_count": 0,
            "capture_failure_count": 0,
        },
        "execution": {
            "target_contacted": False,
            "docker_started": False,
            "network_contacted": False,
            "external_network_contacted": False,
            "gpu_touched": False,
            "training_started": False,
            "rows_written": False,
            "raw_response_persisted": False,
            "sidecars_written": False,
        },
        "hard_gate": {
            "image_attestation": False,
            "fresh_reset_per_role": False,
            "network_none_loopback_only": False,
            "candidate_reference_negative_replay": False,
            "target_slots_13": False,
            "pg377_adapter": False,
            "context_firewall": False,
            "negative_zero_violation": False,
            "training_promotion": False,
        },
        "failures": list(gate.get("blocked_reasons") or []),
        "route_summaries": [],
        "promotion": dict(PROMOTION),
        "interpretation": "PG-379 dynamic source collection is blocked or candidate-only; no training rows or vulnerability claim are created.",
    }


def collect_pg379_dynamic_source_rows_live(
    *,
    seeds: Sequence[int] | None = None,
    route_classes: Sequence[str] | None = None,
    plan: Mapping[str, Any] | None = None,
    live: bool = False,
    attestations: Mapping[str, Any] | None = None,
    image_digest: str | Mapping[str, str] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    image_probe: Callable[..., Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect PG-379 rows in memory under an explicit fail-closed gate.

    ``runtime_factory`` receives abstract route metadata and must return a
    fresh object for one role episode.  It may expose a browser-backed
    ``observe`` method or HTTP ``request`` method; both are reduced to the
    same bounded whole-page projection.  No dataset writer is provided by
    this API on purpose.
    """

    if isinstance(plan, Mapping):
        selected_plan = dict(plan)
        selected_seeds = tuple(int(seed) for seed in (seeds if seeds is not None else _plan_seed_values(selected_plan)))
    else:
        selected_seeds = tuple(int(seed) for seed in (seeds if seeds is not None else SEEDS))
        selected_plan = build_pg379_source_collection_plan(seeds=selected_seeds)
    validation = validate_pg379_source_collection_plan(selected_plan)
    if validation.get("status") != "passed":
        raise ValueError(f"PG-379 source plan is invalid: {validation.get('failures')}")
    if route_classes is not None:
        requested_routes = tuple(str(value) for value in route_classes)
        route_map = {str(route.get("route_class")): route for route in selected_plan.get("route_shape_matrix", []) if isinstance(route, Mapping)}
        missing = sorted(set(requested_routes) - set(route_map))
        if missing:
            raise ValueError(f"PG-379 route class is not in the reviewed plan: {missing}")
        selected_plan = dict(selected_plan)
        selected_plan["route_shape_matrix"] = [route_map[value] for value in requested_routes]
        selected_plan["route_selection"] = list(requested_routes)
    if not live:
        gate = {
            "requested": False,
            "operator_flag": OPERATOR_FLAG,
            "operator_flag_present": (environment or os.environ).get(OPERATOR_FLAG) == "1",
            "ready": False,
            "status": "not_requested",
            "blocked_reasons": ["planning_only_mode"],
            "runtime_factory_bound": False,
            "image_probe_bound": False,
            "target_start_allowed": False,
        }
        report = _empty_report(plan=selected_plan, gate=gate, status="planning_only_live_blocked", mode="planning_only")
        report["plan_validation"] = validation
        report["report_sha256"] = _json_digest(report)
        _scrub(report)
        return {"report": report, "rows": [], "sidecars": []}
    effective_image_probe = image_probe
    if effective_image_probe is None and runtime_factory is not None:
        candidate_probe = getattr(runtime_factory, "image_probe", None)
        if callable(candidate_probe):
            effective_image_probe = candidate_probe
    gate, normalized_attestations = _preflight(
        plan=selected_plan,
        attestations=attestations,
        image_digest=image_digest,
        runtime_factory=runtime_factory,
        image_probe=effective_image_probe,
        environment=environment,
    )
    if not gate["ready"]:
        report = _empty_report(plan=selected_plan, gate=gate, status="blocked_preflight", mode="live_blocked")
        report["plan_validation"] = validation
        report["report_sha256"] = _json_digest(report)
        _scrub(report)
        return {"report": report, "rows": [], "sidecars": []}

    started_at = time.monotonic()
    requirements = dict(selected_plan.get("new_implementation_requirements") or {})
    routes = list(selected_plan.get("route_shape_matrix") or [])
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    role_episode_observed = 0
    runtime_started_count = 0
    valid_source_rows = 0
    typed_role_count = 0
    negative_violations = 0
    failure_observed = 0
    failure_changed = 0
    fresh_reset_failures = 0
    adapter_failure_counts: dict[str, int] = {}
    materialization_failure_counts: dict[str, int] = {}
    for lane in ("train", "holdout"):
        implementation_id = str(requirements[lane]["implementation_id"])
        attestation = normalized_attestations[lane]
        digest = str(attestation["image_digest"])
        for seed in selected_seeds:
            for route in routes:
                route_ref = _route_ref(route)
                captures: dict[str, dict[str, Any]] = {}
                for role in ALL_ROLES:
                    capture = _capture_episode(
                        runtime_factory=runtime_factory,  # type: ignore[arg-type]
                        implementation_id=implementation_id,
                        lane=lane,
                        seed=int(seed),
                        role=role,
                        route=route,
                        image_digest=digest,
                        attestation=attestation,
                    )
                    captures[role] = capture
                    role_episode_observed += 1
                    runtime_started_count += int(bool(capture.get("runtime_started")))
                    if capture.get("capture_failure"):
                        failures.append(f"{lane}:{seed}:{route_ref}:{role}:{capture['capture_failure']}")
                        if str(capture.get("capture_failure")).endswith("unsafe_or_incomplete_fresh_reset"):
                            fresh_reset_failures += 1
                    typed_role_count += int(bool(capture.get("typed")))
                    failure_projection = capture.get("failure_projection")
                    if isinstance(failure_projection, Mapping):
                        failure_class = str(failure_projection.get("failure_class", "none"))
                        previous = str(failure_projection.get("previous_action", ""))
                        next_action = str(failure_projection.get("next_action", ""))
                        if failure_class not in {"", "none", "unknown"}:
                            failure_observed += 1
                            if previous and next_action and previous != next_action:
                                failure_changed += 1
                    if role == "negative" and capture.get("typed"):
                        negative_violations += 1
                record_id = f"pg379-{lane}-{seed}-{route_ref[:16]}"
                sidecar = _build_sidecar(record_id=record_id, captures=captures)
                target = _target_slots(route=route, sidecar=sidecar, role="candidate")
                sidecars.append(
                    {
                        "implementation_id": implementation_id,
                        "lane": lane,
                        "seed": int(seed),
                        "route_ref_sha256": route_ref,
                        "target_slots": target,
                        "target_slot_count": len(target),
                        "roles": {
                            role: {
                                "typed_effect_confirmed": bool(captures[role].get("typed")),
                                "evidence_sha256": str(captures[role].get("evidence_sha256", "")),
                                "fresh_reset": _reset_safe(captures[role].get("reset")),
                            }
                            for role in ALL_ROLES
                        },
                        "evaluator_sidecar": sidecar,
                        "context_firewall": {
                            "sidecar_off_context": True,
                            "oracle_answer_in_context": False,
                            "raw_payload_response_in_context": False,
                        },
                    }
                )
                for role in SOURCE_ROLES:
                    row = _materialize_row(
                        capture=captures[role],
                        sidecar=sidecar,
                        route=route,
                        lane=lane,
                        implementation_id=implementation_id,
                        image_digest=digest,
                        authorization_id=str(normalized_attestations[lane].get("authorization_id", "")),
                        seed=int(seed),
                        role=role,
                    )
                    rows.append(row)
                    valid_source_rows += int(bool(row.get("adapter_validation", {}).get("valid")))
                    adapter_validation = row.get("adapter_validation") if isinstance(row.get("adapter_validation"), Mapping) else {}
                    for failure in adapter_validation.get("failures") or []:
                        key = str(failure)
                        adapter_failure_counts[key] = adapter_failure_counts.get(key, 0) + 1
                    materialization_failure = row.get("capture_materialization_failure")
                    if materialization_failure:
                        key = str(materialization_failure)
                        materialization_failure_counts[key] = materialization_failure_counts.get(key, 0) + 1
                route_summaries.append(
                    {
                        "implementation_id": implementation_id,
                        "lane": lane,
                        "seed": int(seed),
                        "route_ref_sha256": route_ref,
                        "method": str(route.get("method", "unknown")).upper(),
                        "typed": {role: bool(captures[role].get("typed")) for role in ALL_ROLES},
                        "fresh_reset": {role: _reset_safe(captures[role].get("reset")) for role in ALL_ROLES},
                    }
                )

    all_resets = all(
        bool(entry["roles"][role]["fresh_reset"])
        for entry in sidecars
        for role in ALL_ROLES
    )
    all_slots = all(int(entry.get("target_slot_count", 0)) == len(SLOTS) and set(entry.get("target_slots") or {}) == set(SLOTS) for entry in sidecars)
    adapter_ok = all(bool(row.get("adapter_validation", {}).get("valid")) for row in rows)
    context_ok = all(
        dict(row.get("context_firewall") or {}).get("forbidden_token_count") == 0
        and dict(row.get("context_firewall") or {}).get("sidecars_off_context") is True
        for row in rows
    ) and all(bool(dict(entry.get("context_firewall") or {}).get("sidecar_off_context")) for entry in sidecars)
    network_ok = all(
        bool(entry["roles"][role]["fresh_reset"])
        for entry in sidecars
        for role in ALL_ROLES
    ) and all(
        str(normalized_attestations[lane].get("network_mode")) in {"none", "loopback"}
        and normalized_attestations[lane].get("external_network") is False
        and normalized_attestations[lane].get("loopback_only") is True
        for lane in ("train", "holdout")
    )
    complete_roles = role_episode_observed == len(selected_seeds) * len(routes) * len(ALL_ROLES) * 2
    complete_source = len(rows) == len(selected_seeds) * len(routes) * len(SOURCE_ROLES) * 2
    clean_lifecycle = complete_roles and all_resets and not failures
    status = "completed_source_row_candidate_only" if not failures else "completed_incomplete_source_rows"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": "live_injected_runtime",
        "plan_sha256": str(selected_plan.get("plan_sha256", "")),
        "route_selection": [
            str(route.get("route_class"))
            for route in routes
            if isinstance(route, Mapping) and route.get("route_class") is not None
        ],
        "plan_validation": validation,
        "live_gate": gate,
        "attestations": normalized_attestations,
        "objective": {
            "whole_page_projection": True,
            "pg377_adapter": True,
            "target_slot_count": len(SLOTS),
            "training_rows_created": False,
            "raw_payload_response_context": False,
        },
        "counts": {
            "implementation_count": 2,
            "seed_count": len(selected_seeds),
            "route_count": len(routes),
            "role_episode_expected": len(selected_seeds) * len(routes) * len(ALL_ROLES) * 2,
            "role_episode_observed": role_episode_observed,
            "runtime_started_count": runtime_started_count,
            "source_row_expected": len(selected_seeds) * len(routes) * len(SOURCE_ROLES) * 2,
            "source_row_count": len(rows),
            "valid_source_row_count": valid_source_rows,
            "training_eligible_count": 0,
            "target_sidecar_count": len(sidecars),
            "typed_role_count": typed_role_count,
            "negative_violation_count": negative_violations,
            "failure_observed_count": failure_observed,
            "failure_action_changed_count": failure_changed,
            "fresh_reset_failure_count": fresh_reset_failures,
            "capture_failure_count": len(failures),
            "adapter_failure_counts": dict(sorted(adapter_failure_counts.items())),
            "materialization_failure_counts": dict(sorted(materialization_failure_counts.items())),
        },
        "hard_gate": {
            "image_attestation": bool(gate.get("ready")),
            "fresh_reset_per_role": clean_lifecycle,
            "network_none_loopback_only": network_ok,
            "candidate_reference_negative_replay": complete_roles and complete_source and not failures,
            "target_slots_13": all_slots and len(sidecars) == len(selected_seeds) * len(routes) * 2,
            "pg377_adapter": complete_source and adapter_ok,
            "context_firewall": context_ok,
            "negative_zero_violation": negative_violations == 0,
            "training_promotion": False,
        },
        "failures": sorted(set(failures))[:128],
        "route_summaries": route_summaries,
        "execution": {
            # An injected fake runtime exercises the lifecycle without
            # contacting Docker.  A reviewed Docker factory can opt in by
            # declaring ``is_docker_runtime = True``; this keeps test traces
            # honest and prevents a fake run from becoming live evidence.
            "target_contacted": bool(runtime_started_count > 0 and getattr(runtime_factory, "is_docker_runtime", False)),
            "docker_started": bool(runtime_started_count > 0 and getattr(runtime_factory, "is_docker_runtime", False)),
            "network_contacted": bool(runtime_started_count > 0 and getattr(runtime_factory, "is_docker_runtime", False)),
            "external_network_contacted": False,
            "runtime_episodes_started": role_episode_observed,
            "gpu_touched": False,
            "training_started": False,
            "rows_written": False,
            "raw_response_persisted": False,
            "sidecars_written": False,
        },
        "promotion": dict(PROMOTION),
        "interpretation": "Fresh dynamic whole-page rows are candidate/evaluator evidence only; no training row, payload, or vulnerability claim is created.",
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    _scrub(report)
    _scrub(rows, name="rows")
    _scrub(sidecars, name="sidecars")
    report["report_sha256"] = _json_digest(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any], *, output: Path, sidecar_output: Path | None = None) -> dict[str, str]:
    """Write only the bounded report/evaluator sidecar; never a dataset."""

    report = dict(result.get("report") or {})
    _scrub(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths = {"report": str(output)}
    if sidecar_output is not None:
        sidecar_output.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "status": "evaluator_sidecar_only",
            "sidecars": list(result.get("sidecars") or []),
            "promotion": dict(PROMOTION),
            "training_rows_written": False,
        }
        _scrub(document, name="sidecars")
        document["sidecars_sha256"] = _json_digest(document)
        sidecar_output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths["sidecars"] = str(sidecar_output)
    return paths


def _load_attestations(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("--attestation must contain an object")
    return value


def _load_docker_specs(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("--docker-spec must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="request explicitly authorised live collection")
    parser.add_argument("--attestation", type=Path, help="operator-reviewed implementation attestation JSON")
    parser.add_argument("--docker-spec", type=Path, help="operator-reviewed Docker image/route manifest JSON")
    parser.add_argument("--image-digest", help="fixed immutable sha256:<64-hex> digest expected by both lanes")
    parser.add_argument("--output", type=Path, help="optional report output; no dataset is written")
    parser.add_argument("--sidecar-output", type=Path, help="optional evaluator-only sidecar output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        attestations = _load_attestations(args.attestation)
        docker_specs = _load_docker_specs(args.docker_spec)
        runtime_factory = build_pg379_docker_runtime_factory(docker_specs) if docker_specs is not None else None
        result = collect_pg379_dynamic_source_rows_live(
            live=bool(args.live),
            attestations=attestations,
            image_digest=args.image_digest,
            runtime_factory=runtime_factory,
        )
        paths = write_artifacts(result, output=args.output, sidecar_output=args.sidecar_output) if args.output else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"pg379_live_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    report = result["report"]
    summary = {
        "status": report.get("status"),
        "counts": report.get("counts", {}),
        "hard_gate": report.get("hard_gate", {}),
        "report_sha256": report.get("report_sha256", ""),
        "artifacts": paths,
    }
    print(json.dumps({"summary": summary, "report": report} if args.json else summary, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if report.get("status") in {"planning_only_live_blocked", "blocked_preflight", "completed_source_row_candidate_only"} else 2


# Short aliases mirror the planning runner's public API and make the
# lifecycle collector straightforward to call from research-ops contracts.
collect_pg379_source_rows_live = collect_pg379_dynamic_source_rows_live
collect_pg379_dynamic_source_rows = collect_pg379_dynamic_source_rows_live


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_ROLES",
    "FIELD_COUNT",
    "OPERATOR_FLAG",
    "PROMOTION",
    "ROLES",
    "ROUTE_SHAPES",
    "SCHEMA_VERSION",
    "SEEDS",
    "SLOTS",
    "SOURCE_ROLES",
    "collect_pg379_dynamic_source_rows_live",
    "collect_pg379_source_rows_live",
    "collect_pg379_dynamic_source_rows",
    "build_pg379_docker_runtime_factory",
    "write_artifacts",
    "_capture_episode",
    "_normalize_attestation",
]
