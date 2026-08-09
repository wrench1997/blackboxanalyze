"""PG-331 fresh Juice Shop source-row collector (diagnostic only).

This is the live companion to ``run_pg331_juice_shop_source_rows_plan.py``.
It reuses only the already-audited PG-324 disposable-container/loopback
transport and browser challenge-state adapter.  The collector itself never
serializes a URL, probe, response body or oracle answer: response bytes are
parsed in memory by the PG-331 loopback adapter and evaluator projections are
passed through the typed sidecar.

The command is fail-closed behind ``PG331_LOCAL_DOCKER_EVAL=1`` and the local
08:00--18:00 Asia/Shanghai window.  It writes diagnostic source rows only;
operator review is deliberately disabled and every promotion flag is false.
POST lanes that the fixed Juice Shop route contract does not support are still
sent as neutral transport observations, but their typed evaluator is marked
unavailable and the target is forced to ASK.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode, urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_pg331_evaluator_record, sha256_json  # noqa: E402
from app.pg331_loopback_adapter import _PageParser, _field_capture_manifest, capture_loopback  # noqa: E402
from app.pg331_source_row import collect_pg331_source_row  # noqa: E402


SCHEMA_VERSION = "pg331-juice-shop-source-row-live-v1"
TIMEZONE = "Asia/Shanghai"
IMAGE = "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe"
SAFETY_MODE_CONFIG = '{"challenges":{"safetyMode":"disabled"}}'
SAFETY_MODE_CONFIG_SHA256 = hashlib.sha256(SAFETY_MODE_CONFIG.encode("utf-8")).hexdigest()
SEEDS = (33111, 33112, 33113)
ROLES = ("candidate", "reference", "negative")
PAGE_CAPTURE_AXES = ("document_structure", "navigation", "javascript_surface")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

ROUTES: tuple[dict[str, Any], ...] = (
    {
        "id": "juice-track-order-xss-get",
        "method": "GET",
        "path": "/rest/track-order/{id}",
        "value_field": "id",
        "family": "xss",
        "surface": "dom_track",
        "expected_lane": "positive",
        "post_supported": False,
    },
    {
        "id": "juice-products-search-get",
        "method": "GET",
        "path": "/rest/products/search",
        "value_field": "q",
        "family": "xss",
        "surface": "json_response",
        "expected_lane": "negative",
        "post_supported": False,
    },
    {
        "id": "juice-login-post-unsupported",
        "method": "POST",
        "path": "/rest/user/login",
        "value_field": "email",
        "family": "authentication",
        "surface": "json_response",
        "expected_lane": "unsupported_post",
        "post_supported": False,
    },
)
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _load_pg324() -> Any:
    """Load PG-324 utilities lazily so static tests never import Docker/torch."""

    path = ROOT / "scripts" / "run_pg324_juice_shop_source_heldout.py"
    spec = importlib.util.spec_from_file_location("pg324_live_transport", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load audited PG-324 transport")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_window() -> None:
    if os.environ.get("PG331_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-331 Juice Shop live collection requires PG331_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo(TIMEZONE))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-331 local collection is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")


def _length_bucket(value: int) -> str:
    return "zero" if value <= 0 else "one" if value == 1 else "two" if value == 2 else "few" if value <= 5 else "many"


def _status_class(value: Any) -> str:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return f"{status // 100}xx" if 100 <= status < 600 else "unknown"


def _browser_page_failure(reason: str, blocked_external_count: int) -> dict[str, Any]:
    """Return a fail-closed browser projection without retaining page data."""

    # ``None`` is intentional: the tokenizer emits ``not_observed`` for all
    # page axes, and the strict source-row collector consequently turns the
    # target into ASK.  Keep the environment reason outside ``observation`` so
    # it cannot become a model token or a route/evaluator side channel.
    return {
        "ok": False,
        "observation": {axis: None for axis in PAGE_CAPTURE_AXES},
        "environment_failure_class": str(reason),
        "blocked_external_count": int(blocked_external_count),
        "raw_html_stored": False,
    }


def _capture_browser_page_projection(browser: Any, origin: str, *, timeout_ms: int = 10_000) -> dict[str, Any]:
    """Capture a rendered SPA page as bounded structural projections.

    The loopback relay is the only permitted browser origin.  Every other
    request is aborted before navigation, and ``page.content()`` is consumed
    only in memory by the existing ``_PageParser``.  The returned mapping
    contains no markup, URL, payload, response body, or evaluator answer.
    Browser/DOM failures are data-quality failures, not reasons to guess: the
    helper returns missing page axes and an explicit environment failure for
    ``_merge_observation``/the source-row ASK gate to carry forward.
    """

    blocked_external = [0]
    context: Any | None = None
    page: Any | None = None
    markup: str | None = None
    try:
        parsed_origin = urlsplit(str(origin))
        host = (parsed_origin.hostname or "").casefold()
        if parsed_origin.scheme.casefold() not in {"http", "https"} or host not in LOOPBACK_HOSTS:
            return _browser_page_failure("invalid_loopback_origin", 0)
        if parsed_origin.port is None or parsed_origin.username or parsed_origin.password:
            return _browser_page_failure("invalid_loopback_origin", 0)
        origin_scheme = parsed_origin.scheme.casefold()
        origin_host = host
        origin_port = int(parsed_origin.port)

        context = browser.new_context(java_script_enabled=True, service_workers="block")
        page = context.new_page()

        def _request_handler(request_route: Any) -> None:
            local = False
            try:
                request = request_route.request
                candidate = urlsplit(str(getattr(request, "url", "")))
                local = (
                    candidate.scheme.casefold() == origin_scheme
                    and (candidate.hostname or "").casefold() == origin_host
                    and candidate.port == origin_port
                )
            except Exception:
                local = False
            if local:
                request_route.continue_()
                return
            blocked_external[0] += 1
            # Do not retain or report the rejected URL; only the bounded count
            # is useful to the evaluator-side diagnostic.
            request_route.abort()

        page.route("**/*", _request_handler)
        # Juice Shop's document shell is routed through the hash SPA entry;
        # the fragment never leaves the loopback relay and avoids the root API
        # response being mistaken for the whole-page observation.
        page.goto(f"{origin.rstrip('/')}/#/", wait_until="domcontentloaded", timeout=int(timeout_ms))
        # Angular may render its shell after the initial DOMContentLoaded event.
        # This bounded wait is deliberately optional for fake/static tests and
        # never follows external resources because the route handler aborts
        # them above.
        wait_for_timeout = getattr(page, "wait_for_timeout", None)
        if callable(wait_for_timeout):
            wait_for_timeout(min(750, max(0, int(timeout_ms) // 4)))
        content = getattr(page, "content", None)
        if not callable(content):
            return _browser_page_failure("browser_content_unavailable", blocked_external[0])
        markup = content()
        if not isinstance(markup, str) or not markup.strip():
            return _browser_page_failure("empty_rendered_document", blocked_external[0])

        parser = _PageParser(parsed_origin)
        parser.feed(markup)
        parser.close()
        page_observation = parser.observation_projection(
            request_method="GET",
            request_data=None,
            query_data=None,
            response={
                "status_class": "2xx",
                "status_shape": "numeric",
                "content_type_class": "html",
                "connection_outcome": "complete",
                "body_length": len(markup),
                "body_shape": "html",
                "redirect_hop_count": 0,
                "redirect_location_class": "none",
                "redirect_chain_shape": "empty",
                "path": parsed_origin.path or "/",
                "query_key_count": 0,
                "query_key_shapes": [],
            },
        )
        # Keep only the three page axes.  Request/response/failure/belief data
        # is supplied by the loopback transport and evaluator path below.
        projection = {axis: page_observation.get(axis) for axis in PAGE_CAPTURE_AXES}
        markup = None
        return {
            "ok": True,
            "observation": projection,
            "environment_failure_class": "none",
            "blocked_external_count": int(blocked_external[0]),
            "raw_html_stored": False,
        }
    except Exception:
        # Keep a stable abstract category.  The exception type/message can
        # contain a URL, a page fragment, or a browser diagnostic that must
        # stay off-row.
        return _browser_page_failure("browser_capture_error", blocked_external[0])
    finally:
        markup = None
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if context is not None:
            try:
                context.close()
            except Exception:
                pass


def _response_evaluator(details: dict[str, Any]):
    """Return only bounded response-shape facts while retaining bytes in memory."""

    def callback(body: bytes, headers: Mapping[str, Any], status: int | None) -> dict[str, Any]:
        content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].casefold()
        body_lower = body.lower()
        body_shape = "html" if b"<html" in body_lower or b"<body" in body_lower else "json" if "json" in content_type else "text" if body else "empty"
        details.update({"status": int(status or 0), "status_class": _status_class(status), "content_type": content_type or "unknown", "body_length": len(body), "body_shape": body_shape, "body_sha256": hashlib.sha256(body).hexdigest()})
        return {
            "status_class": _status_class(status),
            "content_type_class": content_type if content_type else "unknown",
            "body_shape": body_shape,
            "body_length_bucket": _length_bucket(len(body)),
            "effect_marker": "absent",
            "effect_shape": "transport_shape",
            "connection_outcome": "complete" if status is not None else "transport_error",
            "non_destructive": True,
            "database_touched": False,
        }

    return callback


def _route_request(route: Mapping[str, Any]) -> tuple[str, dict[str, str] | None]:
    """Build a neutral evaluator request; values never leave this process's memory."""

    marker = "PG331-NEUTRAL"
    method = str(route["method"]).upper()
    path = str(route["path"])
    if "{id}" in path:
        path = path.replace("{id}", quote(marker, safe=""))
    elif str(route["id"]) == "juice-products-search-get":
        path += "?" + urlencode({"q": marker})
    if method == "POST":
        return path, {"email": "", "password": ""}
    return path, None


def _merge_observation(
    page: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    role: str,
    typed_available: bool,
    page_failure: str | None = None,
) -> dict[str, Any]:
    observation = {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in page.items()}
    for axis in ("request_transport", "response_transport", "failure_feedback"):
        if isinstance(request.get(axis), Mapping):
            observation[axis] = dict(request[axis])
    if page_failure:
        failure = dict(observation.get("failure_feedback") or {})
        # Keep the environment failure in the abstract failure axis.  The
        # source-row collector will turn the missing page axes into ASK and
        # never allow a probe to be labelled safe after this branch.
        failure.update(
            {
                "failure_class": "environment_failure",
                "failure_stage": "environment",
                "error_shape": "environment",
                "blocked_reason_class": "browser_capture",
                "environment_failure_class": str(page_failure),
                "previous_action": f"{role}_request",
                "next_action": "ask",
                "repair_delta_axis": "document_structure",
                "repair_outcome": "ask",
                "timeout_ms": 0,
            }
        )
        observation["failure_feedback"] = failure
    observation["belief_and_replay"] = {
        "observation_presence": "present",
        "observation_delta_axis": "document_structure" if page_failure else "response_transport",
        "belief_prior_bucket": "low",
        "belief_posterior_bucket": "mid" if typed_available else "unknown",
        "belief_delta_axis": "document_structure" if page_failure else "response_transport",
        "history_action": f"{role}_request",
        "typed_available": "present" if typed_available else "absent",
        "evidence_present": "present" if typed_available else "unknown",
        "negative_control": "present",
        "fresh_reset": "present",
        "replay_ready": "unknown" if page_failure else ("present" if typed_available else "unknown"),
        "reference_present": "present",
        "candidate_present": "present",
        "step_budget": "present",
        "evidence_hash_present": "present" if typed_available else "unknown",
        "history_length": 3,
        "probe_count": 1,
    }
    return observation


def _strict_reset(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce PG-324's reset attestation to the PG-331 source-row schema."""

    return {
        "reset_id": str(raw.get("reset_id", "")),
        "fresh_reset": bool(raw.get("fresh_target", raw.get("completed", False))),
        "target_instance_digest": str(raw.get("container_id_sha256", "")),
        "network_mode": str(raw.get("network_mode", "none")),
        # Preserve the attested boolean; ``False`` means external access is
        # disabled.  Do not invert it while translating PG-324 aliases.
        "external_network": bool(raw.get("external_network", True)),
        "loopback_only": raw.get("relay_loopback_only") is True,
        "state_clean": raw.get("domain_data_write_allowed") is False,
        "database_health_gate": str(raw.get("database_health_gate", "unknown")),
    }


def _oracle_projection(oracle: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "challenge_state_available",
        "challenge_state_baseline_available",
        "challenge_state_baseline_solved",
        "challenge_state_delta",
        "challenge_solved",
        "sink_present",
        "dom_script_execution",
        "script_execution",
        "network_request_count",
        "external_network_blocked",
        "navigation_allowed",
        "database_touched",
        "error_class",
    }
    return {key: oracle[key] for key in allowed if key in oracle}


def _role_target(route: Mapping[str, Any], role: str, *, environment_failure: str | None = None) -> dict[str, Any]:
    if environment_failure:
        return {
            "question": "ask_failure",
            "next_action": "ask",
            "repair_action": "observe",
            "transport_ref": "unknown",
            "field_role_ref": "unknown",
            "encoding_ref": "unknown",
            "probe_variant_ref": "none",
            "safe_to_send": False,
        }
    if str(route["method"]).upper() == "POST" and not bool(route.get("post_supported")):
        return {"question": "ask_typed", "next_action": "ask_typed", "repair_action": "observe", "transport_ref": "unknown", "field_role_ref": "unknown", "encoding_ref": "unknown", "probe_variant_ref": "none", "safe_to_send": False}
    return {"question": "none", "next_action": "send_probe", "repair_action": "none", "transport_ref": "request_method", "field_role_ref": "parameter_role", "encoding_ref": "encoding_chain", "probe_variant_ref": "source_attested_candidate" if role == "candidate" else "reference" if role == "reference" else "negative_control", "safe_to_send": True}


def _source_meta(seed: int, route: Mapping[str, Any], role: str, evidence_hash: str) -> dict[str, Any]:
    route_attestation = sha256_json({"id": str(route["id"]), "method": str(route["method"]), "image": IMAGE})
    return {"source_id": "pg331-juice-shop-fixed-local", "implementation": "bkimminich-juice-shop", "family_id": str(route["family"]), "surface_id": str(route["surface"]), "collector_id": "pg331-juice-shop-source-row-live-v1", "authorization_id": "operator-authorized-local-docker-loopback", "image_digest": IMAGE.split("@sha256:", 1)[1], "source_digest": sha256_json({"seed": int(seed), "route_attestation_sha256": route_attestation, "role": role, "evidence_sha256": evidence_hash})}


def _collect_role(pg324: Any, browser: Any, *, seed: int, route: Mapping[str, Any], index: int, role: str) -> dict[str, Any]:
    # PG-324's audited `_start`/`_stop` pair owns this exact name template;
    # keeping it identical is required for guaranteed cleanup and target
    # reuse rejection.
    name = f"sift-pg324-juice-{seed}-{index}"
    raw_reset: Mapping[str, Any] | None = None
    try:
        _, relay_port, _, raw_reset = pg324._start(seed, index)
        base = f"http://127.0.0.1:{relay_port}"
        # The relay root may be an API/empty response even when the browser
        # renders a complete SPA shell.  Capture that rendered shell in-memory
        # with Playwright and feed only bounded DOM/navigation/JS projections
        # into the row observation.
        page_capture = capture_loopback(base + "/", method="GET", timeout=20.0)
        browser_page = _capture_browser_page_projection(browser, base)
        path, form = _route_request(route)
        details: dict[str, Any] = {}
        request_capture = capture_loopback(base + path, method=str(route["method"]), form_data=form, timeout=20.0, evaluator=_response_evaluator(details))
        typed_available = str(route["method"]).upper() == "GET"
        oracle: dict[str, Any]
        if typed_available:
            marker = f"pg331-{seed}-{route['id']}-{role}_request"
            oracle = dict(pg324._safe_browser_oracle(browser, "", route, marker))
        else:
            oracle = {"available": False, "executed": False, "typed_effect_confirmed": False, "error_class": "typed_unavailable", "evidence_sha256": sha256_json({"seed": seed, "route": route["id"], "role": role, "typed": False})}
        browser_observation = browser_page.get("observation") if isinstance(browser_page.get("observation"), Mapping) else {axis: None for axis in PAGE_CAPTURE_AXES}
        browser_failure = str(browser_page.get("environment_failure_class") or "") if not bool(browser_page.get("ok")) else None
        observation = _merge_observation(browser_observation, request_capture["observation"], role=role, typed_available=typed_available, page_failure=browser_failure)
        evidence_hash = str(oracle.get("evidence_sha256") or sha256_json({"details": details, "role": role}))
        details["browser_page_capture"] = {
            "ok": bool(browser_page.get("ok")),
            "blocked_external_count": int(browser_page.get("blocked_external_count", 0) or 0),
            "environment_failure_class": str(browser_page.get("environment_failure_class", "none")),
            "raw_html_stored": False,
        }
        return {"role": role, "name": name, "reset": _strict_reset(raw_reset), "observation": observation, "manifest": _field_capture_manifest(observation), "details": details, "oracle": oracle, "evidence_sha256": evidence_hash, "target": _role_target(route, role, environment_failure=browser_failure), "target_contacted": bool(browser_page.get("ok") and page_capture.get("target_contacted") and request_capture.get("target_contacted"))}
    finally:
        pg324._stop(name)


def _run_route(pg324: Any, browser: Any, *, seed: int, route: Mapping[str, Any], index: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for offset, role in enumerate(ROLES):
        roles[role] = _collect_role(pg324, browser, seed=seed, route=route, index=index * 10 + offset, role=role)
    candidate = roles["candidate"]
    reference = roles["reference"]
    negative = roles["negative"]
    replay = _collect_role(pg324, browser, seed=seed, route=route, index=index * 10 + 3, role="candidate")
    typed_available = str(route["method"]).upper() == "GET"
    candidate_typed = bool(candidate["oracle"].get("typed_effect_confirmed")) if typed_available else False
    reference_typed = bool(reference["oracle"].get("typed_effect_confirmed")) if typed_available else False
    negative_typed = bool(negative["oracle"].get("typed_effect_confirmed")) if typed_available else False
    replay_typed = bool(replay["oracle"].get("typed_effect_confirmed")) if typed_available else False
    negative_clean = not negative_typed
    replay_consistent = replay_typed == candidate_typed
    sidecar_record = build_pg331_evaluator_record(
        record_id=f"pg331js:{seed}:{route['id']}",
        reset=candidate["reset"],
        candidate={"sent": True, "available": typed_available, "executed": bool(candidate["oracle"].get("executed")), "typed_effect_confirmed": candidate_typed, "effect_class": "dom_effect" if candidate_typed else "none", "projection": _oracle_projection(candidate["oracle"]), "evidence_sha256": candidate["evidence_sha256"]},
        reference={"sent": True, "available": typed_available, "executed": bool(reference["oracle"].get("executed")), "typed_effect_confirmed": reference_typed, "effect_class": "dom_effect" if reference_typed else "none", "projection": _oracle_projection(reference["oracle"]), "evidence_sha256": reference["evidence_sha256"]},
        negative={"sent": True, "available": typed_available, "executed": bool(negative["oracle"].get("executed")), "typed_effect_confirmed": False, "effect_class": "none", "projection": _oracle_projection(negative["oracle"]), "evidence_sha256": negative["evidence_sha256"]},
        replay_consistent=replay_consistent,
        reference_agreement=bool(candidate_typed and reference_typed),
        negative_control_clean=negative_clean,
        evaluator_id="pg331-juice-shop-dom-state-v1",
    )
    sidecar = sidecar_record["evaluator_sidecar"]
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        item = roles[role]
        evaluator = {"typed_available": typed_available, "negative_control": True, "reference_present": True, "candidate_present": True, "fresh_reset": True, "evidence_hash": str(sidecar.get("evidence_sha256", "")), "confirmed_positive": bool(sidecar.get("confirmed_positive")) if role == "candidate" else False, "effect_class": "dom_effect" if bool(item["oracle"].get("typed_effect_confirmed")) else "none", "evaluator_version": "pg331-juice-shop-dom-state-v1"}
        row = collect_pg331_source_row(record_id=f"pg331js:{seed}:{route['id']}:{role}", observation=item["observation"], source_meta=_source_meta(seed, route, role, str(item["evidence_sha256"])), reset=item["reset"], evaluator=evaluator, field_capture_manifest=item["manifest"], target_projection=item["target"], split="implementation_holdout", operator_reviewed=False, hard_negative=role == "negative")
        rows.append(row)
    report = {"seed": seed, "route_id": str(route["id"]), "method": str(route["method"]), "typed_available": typed_available, "candidate_typed": candidate_typed, "reference_typed": reference_typed, "negative_clean": negative_clean, "replay_consistent": replay_consistent, "target_contacted": all(bool(item.get("target_contacted")) for item in [*roles.values(), replay]), "evidence_sha256": str(sidecar.get("evidence_sha256", "")), "confirmed_positive": bool(sidecar.get("confirmed_positive"))}
    return rows, report, sidecar_record


def run(*, seeds: Sequence[int] = SEEDS, route_ids: Sequence[str] | None = None, report_path: Path | None = None, dataset_path: Path | None = None, evaluator_path: Path | None = None) -> dict[str, Any]:
    _require_window()
    pg324 = _load_pg324()
    if getattr(pg324.EVAL, "sync_playwright", None) is None:
        raise RuntimeError("PG-331 Juice Shop live collector requires Playwright")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    selected_routes = tuple(ROUTES if route_ids is None else (route for route in ROUTES if str(route["id"]) in {str(value) for value in route_ids}))
    requested_route_ids = {str(value) for value in route_ids} if route_ids is not None else {str(route["id"]) for route in ROUTES}
    if not selected_routes or {str(route["id"]) for route in selected_routes} != requested_route_ids:
        raise ValueError("PG-331 Juice Shop route selection contains an unknown or empty route")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    route_reports: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    playwright = pg324.EVAL.sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    try:
        for seed in normalized_seeds:
            for index, route in enumerate(selected_routes):
                try:
                    route_rows, route_report, sidecar = _run_route(pg324, browser, seed=seed, route=route, index=seed * 100 + index)
                    rows.extend(route_rows)
                    route_reports.append(route_report)
                    sidecars.append(sidecar)
                except Exception as error:
                    errors.append({"seed": seed, "route_id": str(route["id"]), "error_class": type(error).__name__, "error_message": str(error)[:160]})
    finally:
        browser.close()
        playwright.stop()
        for name in list(getattr(pg324, "_RELAYS", {})):
            try:
                pg324._stop(name)
            except Exception:
                pass
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "completed_diagnostic_only" if route_reports and not errors else "incomplete", "runtime": {"image": IMAGE, "safety_mode_override_sha256": SAFETY_MODE_CONFIG_SHA256, "network": "none", "loopback_only": True, "external_network": False, "fresh_container_per_role": True, "seed_count": len(normalized_seeds), "route_count": len(selected_routes), "elapsed_seconds": round(time.monotonic() - started, 3)}, "counts": {"seed_count": len(normalized_seeds), "route_count": len(route_reports), "row_count": len(rows), "get_count": sum(int(item.get("method") == "GET") for item in route_reports), "post_count": sum(int(item.get("method") == "POST") for item in route_reports), "typed_positive_routes": sum(int(item.get("confirmed_positive")) for item in route_reports), "training_eligible": 0, "errors": len(errors)}, "route_reports": route_reports, "errors": errors, "operator_reviewed": False, "promotion": PROMOTION, "interpretation": "Juice Shop is an independent implementation diagnostic; raw evaluator material stays off-context and rows cannot train until cross-implementation information/fresh-holdout gates pass."}
    report["report_sha256"] = sha256_json(report)
    dataset: dict[str, Any] = {"schema_version": "pg331-source-row-collection-v1", "collector": "scripts/run_pg331_juice_shop_source_rows_live.py", "records": rows, "counts": {"input": len(rows), "accepted": len(rows), "incomplete": len(rows), "rejected": len(errors), "training_eligible": 0}, "source": {"image": IMAGE, "network": "none", "loopback_only": True, "external_network": False}, "promotion": PROMOTION}
    dataset["dataset_sha256"] = sha256_json(dataset)
    evaluator_artifact: dict[str, Any] = {"schema_version": "pg331-juice-shop-evaluator-sidecars-v1", "sidecars": sidecars, "raw_payload_stored": False, "raw_response_bodies_stored": False, "promotion": PROMOTION}
    evaluator_artifact["artifact_sha256"] = sha256_json(evaluator_artifact)
    report_out = report_path or ROOT / "research" / "pg331_juice_shop_source_rows_report_v1.json"
    dataset_out = dataset_path or ROOT / "research" / "pg331_juice_shop_source_rows_v1.json"
    evaluator_out = evaluator_path or ROOT / "research" / "pg331_juice_shop_evaluator_sidecars_v1.json"
    for path, value in ((report_out, report), (dataset_out, dataset), (evaluator_out, evaluator_artifact)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": report, "dataset": dataset, "evaluator": evaluator_artifact}


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-331 fresh Juice Shop source rows; diagnostic only")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", dest="seeds", action="append", type=int, help="seed to run; repeat for multiple seeds")
    parser.add_argument("--route-id", dest="route_ids", action="append", help="allowlisted route to run; repeat for a bounded diagnostic subset")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--evaluator", type=Path)
    args = parser.parse_args()
    result = run(seeds=tuple(args.seeds) if args.seeds else SEEDS, route_ids=tuple(args.route_ids) if args.route_ids else None, report_path=args.report, dataset_path=args.dataset, evaluator_path=args.evaluator)
    print(json.dumps(result if args.json else {"status": result["report"]["status"], "counts": result["report"]["counts"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IMAGE",
    "ROUTES",
    "ROLES",
    "SEEDS",
    "run",
    "_capture_browser_page_projection",
    "_merge_observation",
]
