"""PG-246: independent VulnerableApp DOM holdout for the black-box loop.

This runner deliberately does not reuse Pikachu routes or its payload binder.
The frozen field-token model only chooses an abstract ``dom_markup`` channel;
an independent VulnerableApp adapter binds that choice to a short-lived,
loopback-only DOM-marker probe.  Every case is recreated twice: once for the
initial observation and once for a fresh replay.  Positive claims require the
AI marker, an independent reference marker, clean controls, and a typed DOM
oracle.  A POST-only 405 is recorded as an abstention/environment observation,
never as a vulnerability.

Raw wires are printed for a human to inspect and are never persisted.  Reports
contain only hashes, bounded projections, lineage, and Rule-IR/process tokens.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit

import torch
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg198_payload_grounding import generate_grounded_candidates  # noqa: E402
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet  # noqa: E402
from app.pg230_next_token_quality_funnel import digest, prepare_record, quality_lane  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG208 = _load_script("run_pg208_pikachu_typed_payload_loop.py")

RESEARCH = ROOT / "research"
RESET_SCRIPT = ROOT / "scripts" / "reset_pg25d_vulnerableapp.ps1"
REGISTRY = RESEARCH / "pg_pk_24_cross_lab_registry_v1.json"
REPORT = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_report_v1.json"
DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
TRACE = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_trace_v1.json"
PROTOCOL = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_protocol_v1.json"
MARKDOWN = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_report_v1.md"

BASE_ORIGIN = "http://127.0.0.1:19090"
BASE_URI = f"{BASE_ORIGIN}/VulnerableApp"
IMAGE_DIGEST = "sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406"
SOURCE_ID = "pg246_vulnerableapp_source_independent"
SOURCE_COMMIT = "sasanlabs-vulnerableapp-2.1.44"
SEEDS = (24601, 24602, 24603)
GENERATOR_ID = "pg246-vulnerableapp-dom-surface-generator-v1"
MODEL_ARTIFACT = ROOT / "artifacts" / "pg206-body-capacity-v1" / "xxl_field_token_adapter.pt"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _case_specs() -> list[dict[str, Any]]:
    # LEVEL_1/LEVEL_4 and LEVEL_1/LEVEL_6 are route/implementation holdouts
    # from Pikachu.  src is the actual observed parameter in the VulnerableApp
    # class; probe is the actual observed parameter in its HTML-tag class.
    return [
        {
            "case_id": "vapp_html_level1_get",
            "path": "/VulnerableApp/XSSWithHtmlTagInjection/LEVEL_1",
            "field": "probe",
            "method": "GET",
            "surface_role": "xss_html_tag_surface",
            "expected": "positive",
            "style": "html",
        },
        {
            "case_id": "vapp_html_level4_secure_get",
            "path": "/VulnerableApp/XSSWithHtmlTagInjection/LEVEL_4",
            "field": "probe",
            "method": "GET",
            "surface_role": "xss_html_tag_surface",
            "expected": "negative",
            "style": "html",
        },
        {
            "case_id": "vapp_img_level1_get",
            "path": "/VulnerableApp/XSSInImgTagAttribute/LEVEL_1",
            "field": "src",
            "method": "GET",
            "surface_role": "xss_img_attribute_surface",
            "expected": "positive",
            "style": "img",
        },
        {
            "case_id": "vapp_img_level6_secure_get",
            "path": "/VulnerableApp/XSSInImgTagAttribute/LEVEL_6",
            "field": "src",
            "method": "GET",
            "surface_role": "xss_img_attribute_surface",
            "expected": "negative",
            "style": "img",
        },
        {
            "case_id": "vapp_html_level1_post_405",
            "path": "/VulnerableApp/XSSWithHtmlTagInjection/LEVEL_1",
            "field": "probe",
            "method": "POST",
            "surface_role": "xss_html_tag_surface",
            "expected": "unsupported_post",
            "style": "html",
        },
        {
            "case_id": "vapp_img_level1_post_405",
            "path": "/VulnerableApp/XSSInImgTagAttribute/LEVEL_1",
            "field": "src",
            "method": "POST",
            "surface_role": "xss_img_attribute_surface",
            "expected": "unsupported_post",
            "style": "img",
        },
    ]


def _source() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "target_id": "owasp_vulnerableapp_2_1_44",
        "app_family": "owasp_vulnerableapp",
        "target_implementation": "sasanlabs-vulnerableapp-java-spring",
        "source_commit": SOURCE_COMMIT,
        "license": "local-container",
        "authorization": "workspace_local_only",
        "loopback_origin": BASE_ORIGIN,
        "container_image_digest": IMAGE_DIGEST,
        "reset_script_sha256": _hash_text(RESET_SCRIPT.read_text(encoding="utf-8")),
        "independent_from_pikachu": True,
        "external_network": False,
    }


def _docker_inspect(template: str) -> str:
    return subprocess.check_output(
        ["docker", "inspect", "pg25-vulnerableapp", "--format", template],
        cwd=ROOT,
        text=True,
        timeout=30,
    ).strip()


def _fresh_reset(*, seed: int, run_index: int) -> dict[str, Any]:
    # The reset script owns exactly one named loopback container.  It recreates
    # the writable layer; its tmpfs mounts are allowed, but bind/volume state
    # is not.
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RESET_SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    container_id = _docker_inspect("{{.Id}}")
    image_ref = _docker_inspect("{{.Config.Image}}")
    mounts = json.loads(_docker_inspect("{{json .Mounts}}") or "[]")
    mount_types = sorted({str(item.get("Type", "unknown")) for item in mounts})
    forbidden_mounts = [item for item in mounts if str(item.get("Type", "")).casefold() in {"bind", "volume"}]
    if image_ref != f"sasanlabs/owasp-vulnerableapp@{IMAGE_DIGEST}":
        raise RuntimeError("PG-246 image digest attestation mismatch")
    if forbidden_mounts:
        raise RuntimeError("PG-246 refuses bind/volume state on a fresh target")
    return {
        "reset_id": f"pg246-reset-{seed}-{run_index}-{container_id[:12]}",
        "target_instance_hash": _hash_text(container_id),
        "state_epoch": f"pg246-{seed}-{run_index}-{container_id[:12]}",
        "fresh_target": True,
        "completed": True,
        "container_recreated": True,
        "container_restart_used": False,
        "image_digest": IMAGE_DIGEST,
        "mount_types": mount_types,
        "bind_or_volume_mount_count": len(forbidden_mounts),
        "state_change_allowed": False,
        "evaluator_state_hidden": True,
        "external_network": False,
    }


def _new_page(browser: Browser) -> tuple[BrowserContext, Page, list[str], list[str]]:
    context = browser.new_context()
    page = context.new_page()
    requests: list[str] = []
    blocked_external: list[str] = []
    origin_parts = urlsplit(BASE_ORIGIN)

    def _guard(route: Any) -> None:
        url = str(route.request.url)
        parts = urlsplit(url)
        if parts.scheme in {"http", "https"} and (parts.hostname, parts.port) != (origin_parts.hostname, origin_parts.port):
            blocked_external.append(url)
            route.abort()
        else:
            route.continue_()

    page.route("**/*", _guard)
    page.on("request", lambda request: requests.append(str(request.url)))
    page.on("dialog", lambda dialog: dialog.dismiss())
    return context, page, requests, blocked_external


def _payload(case: Mapping[str, Any], marker: str, *, reference: bool) -> str:
    # These are local DOM-marker probes, not cookie reads, navigation, or
    # exfiltration.  The alternate syntax is deliberately independent from
    # the first candidate so agreement is not string memorization.
    if str(case["style"]) == "html":
        if reference:
            return f'<img src=x onerror="document.body.dataset.pg246=\'{marker}\'">'
        return f'<svg/onload="document.body.dataset.pg246=\'{marker}\'">'
    if reference:
        return f'y onerror="document.body.dataset.pg246=\'{marker}\'"'
    return f'x onerror="document.body.dataset.pg246=\'{marker}\'"'


def _wire(case: Mapping[str, Any], value: str | None) -> str:
    method = str(case["method"]).upper()
    field = str(case["field"])
    if method == "GET":
        query = "" if value is None else f"?{urlencode({field: value})}"
        return f"GET {BASE_ORIGIN}{case['path']}{query}"
    form = {} if value is None else {field: value}
    return f"POST {BASE_ORIGIN}{case['path']}\nContent-Type: application/x-www-form-urlencoded\n\n{urlencode(form)}"


def _run_probe(browser: Browser, case: Mapping[str, Any], value: str | None, expected_marker: str | None) -> dict[str, Any]:
    context, page, requests, blocked_external = _new_page(browser)
    status = 0
    body_bytes = b""
    transport_error = False
    try:
        method = str(case["method"]).upper()
        field = str(case["field"])
        if method == "GET":
            query = "" if value is None else f"?{urlencode({field: value})}"
            response = page.goto(f"{BASE_ORIGIN}{case['path']}{query}", wait_until="domcontentloaded", timeout=15000)
            status = int(response.status if response else 200)
            page.wait_for_timeout(350)
            body_bytes = page.content().encode("utf-8")
        else:
            # Establish the loopback origin before fetch; evaluating from
            # about:blank would make a valid local POST look like a CORS or
            # transport failure.
            page.goto(f"{BASE_URI}/", wait_until="domcontentloaded", timeout=15000)
            body = "" if value is None else urlencode({field: value})
            result = page.evaluate(
                """
                async ({url, body}) => {
                  try {
                    const response = await fetch(url, {
                      method: 'POST',
                      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                      body
                    });
                    const text = await response.text();
                    return {status: response.status, text};
                  } catch (error) {
                    return {status: 0, text: ''};
                  }
                }
                """,
                {"url": f"{BASE_ORIGIN}{case['path']}", "body": body},
            )
            status = int(result.get("status", 0))
            body_bytes = str(result.get("text", "")).encode("utf-8", errors="replace")
            page.wait_for_timeout(100)
    except Exception:
        transport_error = True
    marker = str(page.evaluate("document.body.dataset.pg246 || ''")) if not transport_error else ""
    return {
        "status_class": "transport_error" if transport_error else (f"{status // 100}xx" if 100 <= status <= 599 else "unknown"),
        "status_code": status,
        "body_length_bucket": "0" if len(body_bytes) == 0 else "1-255" if len(body_bytes) <= 255 else "256-4095" if len(body_bytes) <= 4095 else "4096+",
        "body_sha256": _hash_text(body_bytes.decode("utf-8", errors="replace")),
        "marker_observed": bool(expected_marker and marker == expected_marker),
        "marker_sha256": _hash_text(marker) if marker else None,
        "script_execution": bool(expected_marker and marker == expected_marker),
        "request_count": len(requests),
        "external_request_blocked": len(blocked_external),
        "external_network": False,
        "transport_error": transport_error,
    }


def _model_plan(model: Any, vocabulary: Mapping[str, int], device: torch.device, case: Mapping[str, Any], control: Mapping[str, Any], marker: str) -> dict[str, Any]:
    candidates = generate_grounded_candidates(
        family="xss",
        target=BASE_ORIGIN,
        path=str(case["path"]),
        method=str(case["method"]),
        fields=[str(case["field"])],
        marker=marker,
    )
    route = {
        "path": str(case["path"]),
        "method": str(case["method"]),
        "fields": [str(case["field"])],
        "surface": str(case["surface_role"]),
        "typed_available": bool(str(case["method"]).upper() == "GET"),
    }
    packet = build_field_token_packet(
        candidates[0],
        route=route,
        response_projection=control,
        typed_available=bool(route["typed_available"]),
        redirect_hops=0,
    )
    validation = validate_field_token_packet(
        packet,
        candidate=candidates[0],
        route=route,
        response_projection=control,
        typed_available=bool(route["typed_available"]),
        redirect_hops=0,
    )
    if not validation["valid"]:
        return {"action": "abstain", "effective_action": "abstain", "reason": validation["reason"], "validation": validation, "candidate_count": len(candidates)}
    decision = PG208._model_decision(model, vocabulary, device, packet=packet, route=route, projection=control)
    return {
        "action": str(decision.get("action", "abstain")),
        "effective_action": str(decision.get("effective_action", "abstain")),
        "encoding": str(decision.get("encoding", "unknown")),
        "failure": str(decision.get("failure", "unknown")),
        "candidate_count": len(candidates),
        "validation": validation,
        "binding_sha256": str(packet.get("binding_sha256", "")),
    }


def _reset_projection(reset: Mapping[str, Any], case: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "reset_id": reset["reset_id"],
        "target_instance_hash": reset["target_instance_hash"],
        "state_epoch": reset["state_epoch"],
        "fresh_target": bool(reset["fresh_target"]),
        "method": str(case["method"]),
        "route_template_hash": _hash_text(str(case["path"])),
        "bind_or_volume_mount_count": int(reset["bind_or_volume_mount_count"]),
    }


def _record(
    *,
    case: Mapping[str, Any],
    seed: int,
    reset: Mapping[str, Any],
    evidence_hash: str,
    lane_hint: str,
    candidate_sent: bool,
    oracle_available: bool,
    typed_effect: bool,
    reference_agreement: bool,
    negative_clean: bool,
    failure_signature: str,
    failure_stage: str,
    model_environment: str,
    next_step: str,
    parent_record_id: str | None = None,
    repair_delta: Mapping[str, Any] | None = None,
    replay_match: bool = False,
    negative_control_confirmed: bool = False,
) -> dict[str, Any]:
    row = {
        "source": SOURCE_ID,
        "seed": int(seed),
        "surface_role": str(case["surface_role"]),
        "method": str(case["method"]),
        "field_count": 1,
        "status_class": "2xx" if str(case["method"]) == "GET" else "4xx",
        "history_len": 0 if parent_record_id is None else 1,
        "candidate_sent": bool(candidate_sent),
        "oracle_available": bool(oracle_available),
        "typed_effect_confirmed": bool(typed_effect),
        "typed_effect_observed": bool(typed_effect),
        "candidate_reference_agreement": bool(reference_agreement),
        "negative_clean": bool(negative_clean),
        "negative_control_confirmed": bool(negative_control_confirmed),
        "fresh_reset_ok": bool(reset["fresh_target"]),
        "reset_completed": bool(reset["completed"]),
        "reset_not_attempted": False,
        "failure_signature": str(failure_signature),
        "failure_stage": str(failure_stage),
        "failure_is_model_or_environment": str(model_environment),
        "next_step": str(next_step),
        "previous_feedback": "result_verified" if typed_effect else "failure_adjusted" if failure_signature else "none",
        "candidate_result_present": bool(typed_effect),
        "model_claimed_positive": bool(candidate_sent and typed_effect),
        "model_abstained": not bool(candidate_sent),
        "abstention_required": not bool(oracle_available) or not bool(candidate_sent),
        "evidence_hash": evidence_hash,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "payload_grounded_eligible": bool(typed_effect),
        "parent_record_id": parent_record_id,
        "repair_delta_projection": dict(repair_delta or {}),
        "repair_outcome": "fresh_replay_match" if replay_match else "pending_replay",
        "repair_replay_count": 1 if replay_match else 0,
        "candidate_reference_replay_match": bool(replay_match),
        "lane_hint": str(lane_hint),
    }
    prepared = prepare_feedback_record(row)
    prepared.update({
        "failure_stage": str(failure_stage),
        "failure_is_model_or_environment": str(model_environment),
        "parent_record_id": parent_record_id,
        "repair_delta_projection": dict(repair_delta or {}),
        "repair_outcome": "fresh_replay_match" if replay_match else "pending_replay",
        "repair_replay_count": 1 if replay_match else 0,
        "candidate_reference_replay_match": bool(replay_match),
        "source_implementation": "owasp-vulnerableapp-java-spring",
        "generator_id": GENERATOR_ID,
        "route_template_hash": _hash_text(str(case["path"])),
        "implementation_holdout": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    })
    # The lane is computed independently from the caller's expectation.  Keep
    # lane_hint only as an audit comparison so a label cannot be hand-written.
    computed_lane, reasons = quality_lane(row)
    prepared["lane"] = computed_lane
    prepared["lane_index"] = int({"gold": 0, "hard_negative": 1, "silver": 2, "quarantine": 3, "reject": 4}[computed_lane])
    prepared["quality_reasons"] = reasons
    prepared["record_id"] = f"pg246:{seed}:{case['case_id']}:{evidence_hash[:12]}"
    prepared["record_hash"] = _hash(prepared)
    return prepared


def _episode(model: Any, vocabulary: Mapping[str, int], device: torch.device, browser: Browser, case: Mapping[str, Any], *, seed: int, run_index: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    reset = _fresh_reset(seed=seed, run_index=run_index)
    marker_base = f"pg246-{seed}-{case['case_id'].replace('_', '-')[:18]}"
    baseline = _run_probe(browser, case, None, None)
    safe_value = f"pg246-safe-{seed}"
    negative = _run_probe(browser, case, safe_value, None)
    control = {
        "status_class": str(baseline["status_class"]),
        "marker": {"reflected": False},
        "redirect_hop_count": 0,
        "projection_sha256": _hash({"baseline": baseline, "negative": negative}),
    }
    model_plan = _model_plan(model, vocabulary, device, case, control, f"{marker_base}-model")
    ai_marker = f"{marker_base}-ai"
    reference_marker = f"{marker_base}-ref"
    ai_value = _payload(case, ai_marker, reference=False)
    reference_value = _payload(case, reference_marker, reference=True)
    ai = {"sent": False}
    reference = {"sent": False}
    wires = [_wire(case, None), _wire(case, safe_value)]
    if str(case["method"]).upper() == "GET" and model_plan.get("effective_action") == "safe_candidate":
        ai_result = _run_probe(browser, case, ai_value, ai_marker)
        ai = {"sent": True, "response": ai_result, "payload_sha256": _hash_text(ai_value), "generator_id": GENERATOR_ID}
        wires.append(_wire(case, ai_value))
    elif str(case["method"]).upper() == "GET":
        ai = {"sent": False, "abstain_reason": "model_or_binding_abstain"}
    reference_result = _run_probe(browser, case, reference_value, reference_marker) if str(case["method"]).upper() == "GET" else _run_probe(browser, case, safe_value, None)
    reference = {"sent": str(case["method"]).upper() == "GET", "response": reference_result, "payload_sha256": _hash_text(reference_value) if str(case["method"]).upper() == "GET" else _hash_text(safe_value)}
    if str(case["method"]).upper() == "GET":
        wires.append(_wire(case, reference_value))
    ai_effect = bool((ai.get("response") or {}).get("marker_observed"))
    ref_effect = bool((reference.get("response") or {}).get("marker_observed"))
    control_clean = not bool(baseline.get("script_execution") or negative.get("script_execution"))
    external_blocked = int(
        baseline.get("external_request_blocked", 0)
        + negative.get("external_request_blocked", 0)
        + (ai.get("response") or {}).get("external_request_blocked", 0)
        + reference_result.get("external_request_blocked", 0)
    )
    transport = bool(
        baseline.get("transport_error")
        or negative.get("transport_error")
        or (ai.get("response") or {}).get("transport_error")
        or reference_result.get("transport_error")
    )
    expected = str(case["expected"])
    if expected == "positive":
        confirmed = bool(ai_effect and ref_effect and control_clean and not transport and external_blocked == 0)
        if confirmed:
            failure_signature, failure_stage, attribution, next_step = "typed_effect", "oracle", "none", "abstain"
        elif transport or external_blocked:
            failure_signature, failure_stage, attribution, next_step = "environment_transport_failure", "transport", "environment", "inspect_environment"
        elif ref_effect and not ai_effect:
            failure_signature, failure_stage, attribution, next_step = "model_abstain_or_no_effect", "candidate_selection", "model", "retry_candidate"
        else:
            failure_signature, failure_stage, attribution, next_step = "candidate_no_effect", "typed_oracle", "model", "recheck_oracle"
    elif expected == "negative":
        confirmed = False
        if ai_effect or ref_effect:
            failure_signature, failure_stage, attribution, next_step = "negative_control_effect", "typed_oracle", "model", "gate_correction"
        elif transport or external_blocked:
            failure_signature, failure_stage, attribution, next_step = "environment_transport_failure", "transport", "environment", "inspect_environment"
        else:
            failure_signature, failure_stage, attribution, next_step = "counterfactual_candidate_no_effect", "typed_oracle", "none", "abstain"
    else:
        confirmed = False
        post_statuses = {int(baseline.get("status_code", 0)), int(negative.get("status_code", 0)), int(reference_result.get("status_code", 0))}
        if 405 in post_statuses:
            failure_signature, failure_stage, attribution, next_step = "unsupported_post_405", "transport_or_adapter", "environment", "abstain"
        elif transport:
            failure_signature, failure_stage, attribution, next_step = "environment_transport_failure", "transport", "environment", "inspect_environment"
        else:
            failure_signature, failure_stage, attribution, next_step = "oracle_unavailable_post", "oracle", "environment", "abstain"
    evidence = {
        "schema_version": "pg246-vulnerableapp-dom-evidence-v1",
        "source_id": SOURCE_ID,
        "case_id": str(case["case_id"]),
        "method": str(case["method"]),
        "surface_role": str(case["surface_role"]),
        "reset": _reset_projection(reset, case, phase="initial"),
        "baseline": baseline,
        "negative": negative,
        "ai_marker_observed": ai_effect,
        "reference_marker_observed": ref_effect,
        "confirmed_positive": confirmed,
        "negative_control_clean": control_clean,
        "external_request_blocked": external_blocked,
        "transport_error": transport,
        "model_decision": {key: model_plan.get(key) for key in ("action", "effective_action", "encoding", "failure", "binding_sha256")},
        "generator_id": GENERATOR_ID,
        "evidence_scope": "loopback_dom_marker_only",
    }
    evidence_hash = _hash(evidence)
    counterfactual = _record(
        case=case,
        seed=seed,
        reset=reset,
        evidence_hash=_hash({"parent": evidence_hash, "role": "counterfactual", "negative": negative}),
        lane_hint="hard_negative",
        candidate_sent=True,
        oracle_available=True,
        typed_effect=False,
        reference_agreement=True,
        negative_clean=control_clean,
        failure_signature="counterfactual_candidate_no_effect",
        failure_stage="typed_oracle",
        model_environment="none",
        next_step="retry_candidate" if expected == "positive" else "abstain",
        repair_delta={"from": "inert_marker", "to": "independent_dom_event"},
        negative_control_confirmed=True,
    )
    main_record = _record(
        case=case,
        seed=seed,
        reset=reset,
        evidence_hash=evidence_hash,
        lane_hint="gold" if confirmed else "hard_negative",
        candidate_sent=bool(ai.get("sent")),
        oracle_available=bool(str(case["method"]).upper() == "GET"),
        typed_effect=confirmed,
        reference_agreement=bool(ai_effect == ref_effect),
        negative_clean=control_clean,
        failure_signature=failure_signature,
        failure_stage=failure_stage,
        model_environment=attribution,
        next_step=next_step,
        parent_record_id=counterfactual["record_id"],
        repair_delta={"from": "counterfactual_candidate_no_effect", "to": "typed_effect" if confirmed else failure_signature},
        negative_control_confirmed=bool(expected != "positive"),
    )
    episode = {
        "schema_version": "pg246-vulnerableapp-dom-episode-v1",
        "source_id": SOURCE_ID,
        "case_id": str(case["case_id"]),
        "route_template_hash": _hash_text(str(case["path"])),
        "surface_role": str(case["surface_role"]),
        "method": str(case["method"]),
        "field_names": [str(case["field"])],
        "expected": expected,
        "seed": int(seed),
        "reset": _reset_projection(reset, case, phase="initial"),
        "baseline": baseline,
        "negative": negative,
        "model": model_plan,
        "ai": ai,
        "reference": reference,
        "typed_oracle": {
            "oracle_id": "pg246-controlled-browser-dom-marker-v1",
            "oracle_available": bool(str(case["method"]).upper() == "GET"),
            "confirmed_positive": confirmed,
            "negative_control_clean": control_clean,
            "evidence_hash": evidence_hash,
            "reasons": [] if confirmed else [failure_signature],
        },
        "failure_signature": failure_signature,
        "failure_stage": failure_stage,
        "failure_is_model_or_environment": attribution,
        "counterfactual_record_id": counterfactual["record_id"],
        "record_id": main_record["record_id"],
        "external_network": False,
        "external_request_blocked": external_blocked,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "vulnerability_claim_allowed": False,
    }
    # Fresh replay uses the reference channel only.  It is a second target
    # instance and cannot share a DOM marker or state with the initial run.
    replay_reset = _fresh_reset(seed=seed, run_index=run_index + 1000)
    replay_value = _payload(case, f"{marker_base}-replay", reference=True) if str(case["method"]).upper() == "GET" else safe_value
    replay = _run_probe(browser, case, replay_value, f"{marker_base}-replay" if str(case["method"]).upper() == "GET" else None)
    replay_effect = bool(replay.get("marker_observed"))
    replay_expected_match = bool((expected == "positive" and replay_effect) or (expected != "positive" and not replay_effect))
    replay_evidence_hash = _hash({"parent": evidence_hash, "reset": _reset_projection(replay_reset, case, phase="replay"), "response": replay, "expected_match": replay_expected_match})
    replay_record = _record(
        case=case,
        seed=seed,
        reset=replay_reset,
        evidence_hash=replay_evidence_hash,
        lane_hint="gold" if (confirmed and replay_expected_match) else "hard_negative",
        candidate_sent=bool(str(case["method"]).upper() == "GET"),
        oracle_available=bool(str(case["method"]).upper() == "GET"),
        typed_effect=bool(expected == "positive" and replay_effect),
        reference_agreement=True,
        negative_clean=bool(control_clean),
        failure_signature="typed_effect" if expected == "positive" and replay_effect else "counterfactual_candidate_no_effect" if expected != "positive" and not replay_effect else "replay_mismatch",
        failure_stage="replay",
        model_environment="none" if replay_expected_match else "environment",
        next_step="abstain" if replay_expected_match else "inspect_environment",
        parent_record_id=main_record["record_id"],
        repair_delta={"from": "initial_transition", "to": "fresh_replay"},
        replay_match=replay_expected_match,
        negative_control_confirmed=bool(expected != "positive"),
    )
    episode["replay"] = {
        "reset": _reset_projection(replay_reset, case, phase="replay"),
        "response": replay,
        "marker_observed": replay_effect,
        "expected_match": replay_expected_match,
        "evidence_hash": replay_evidence_hash,
        "record_id": replay_record["record_id"],
    }
    wires.append(_wire(case, replay_value))
    return episode, [counterfactual, main_record, replay_record], wires


def main() -> int:
    source = _source()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary = PG208._load_model(device)
    model.eval()
    model_hash = _hash_text(MODEL_ARTIFACT.read_bytes().hex()) if MODEL_ARTIFACT.exists() else ""
    cases = _case_specs()
    episodes: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    wire_count = 0
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                for seed in SEEDS:
                    for case_index, case in enumerate(cases):
                        print(f"\nPG246 seed={seed} case={case['case_id']} method={case['method']}", flush=True)
                        episode, episode_records, wires = _episode(
                            model,
                            vocabulary,
                            device,
                            browser,
                            case,
                            seed=seed,
                            run_index=(seed - SEEDS[0]) * len(cases) + case_index,
                        )
                        for wire in wires:
                            print("--- WIRE (display-only; not persisted) ---\n" + wire, flush=True)
                        print(
                            json.dumps(
                                {
                                    "case_id": episode["case_id"],
                                    "method": episode["method"],
                                    "model_action": episode["model"].get("effective_action"),
                                    "ai_sent": episode["ai"].get("sent", False),
                                    "ai_marker": episode["ai"].get("response", {}).get("marker_observed", False),
                                    "reference_marker": episode["reference"].get("response", {}).get("marker_observed", False),
                                    "confirmed_positive": episode["typed_oracle"]["confirmed_positive"],
                                    "failure_signature": episode["failure_signature"],
                                    "replay_match": episode["replay"]["expected_match"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        episodes.append(episode)
                        records.extend(episode_records)
                        wire_count += len(wires)
            finally:
                browser.close()
    finally:
        # Leave no named target running after an evaluation-only run.  This is
        # an exact, owned container cleanup, not a workspace-wide prune.
        subprocess.run(["docker", "stop", "--time", "5", "pg25-vulnerableapp"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        subprocess.run(["docker", "rm", "pg25-vulnerableapp"], cwd=ROOT, capture_output=True, text=True, timeout=30)

    counts = {
        "seed_count": len(SEEDS),
        "case_count": len(cases),
        "initial_episode_count": len(episodes),
        "replay_count": sum(int("replay" in row) for row in episodes),
        "fresh_container_count": sum(1 + int("replay" in row) for row in episodes),
        "get_count": sum(int(row["method"] == "GET") for row in episodes),
        "post_count": sum(int(row["method"] == "POST") for row in episodes),
        "model_safe_candidate_count": sum(int(row["model"].get("effective_action") == "safe_candidate") for row in episodes),
        "ai_send_count": sum(int(row["ai"].get("sent", False)) for row in episodes),
        "reference_send_count": sum(int(row["reference"].get("sent", False)) for row in episodes),
        "confirmed_positive_count": sum(int(row["typed_oracle"]["confirmed_positive"]) for row in episodes),
        "negative_control_clean_count": sum(int(row["typed_oracle"]["negative_control_clean"]) for row in episodes),
        "model_missed_positive_count": sum(int(row["failure_signature"] == "model_abstain_or_no_effect") for row in episodes),
        "model_false_positive_count": sum(int(row["failure_signature"] == "negative_control_effect") for row in episodes),
        "environment_failure_count": sum(int(row["failure_is_model_or_environment"] == "environment") for row in episodes),
        "post_405_abstain_count": sum(int(row["failure_signature"] == "unsupported_post_405") for row in episodes),
        "replay_match_count": sum(int(row["replay"]["expected_match"]) for row in episodes),
        "external_network_count": sum(int(bool(row.get("external_network"))) for row in episodes),
        "external_request_blocked_count": sum(int(row.get("external_request_blocked", 0) > 0) for row in episodes),
        "wire_display_count": wire_count,
        "dataset_record_count": len(records),
        "gold_record_count": sum(int(row.get("lane") == "gold") for row in records),
        "hard_negative_record_count": sum(int(row.get("lane") == "hard_negative") for row in records),
        "quarantine_record_count": sum(int(row.get("lane") == "quarantine") for row in records),
    }
    route_holdout = {
        "positive_cases": ["vapp_html_level1_get", "vapp_img_level1_get"],
        "secure_negative_cases": ["vapp_html_level4_secure_get", "vapp_img_level6_secure_get"],
        "positive_recall": round(
            sum(int(row["typed_oracle"]["confirmed_positive"]) for row in episodes if row["expected"] == "positive")
            / max(sum(int(row["expected"] == "positive") for row in episodes), 1),
            8,
        ),
        "secure_false_accept_count": sum(int(row["failure_signature"] == "negative_control_effect") for row in episodes if row["expected"] == "negative"),
        "post_abstain_recall": round(
            sum(int(row["failure_signature"] == "unsupported_post_405") for row in episodes if row["expected"] == "unsupported_post")
            / max(sum(int(row["expected"] == "unsupported_post") for row in episodes), 1),
            8,
        ),
        "implementation_holdout_from_pikachu": True,
    }
    report = {
        "protocol_id": "pg-pk-246-vulnerableapp-independent-dom-holdout-v1",
        "schema_version": "pg246-vulnerableapp-independent-dom-holdout-report-v1",
        "status": "completed_independent_implementation_route_holdout",
        "device": str(device),
        "model": {
            "variant": "frozen_xxl_field_token_decoder",
            "artifact": str(MODEL_ARTIFACT.relative_to(ROOT)) if MODEL_ARTIFACT.exists() else None,
            "artifact_sha256": model_hash,
            "online_weight_update": False,
            "ai_selects_abstract_channel": True,
        },
        "source": source,
        "generator": {"id": GENERATOR_ID, "independent_from_pikachu_binder": True},
        "parameter_discovery": {
            "source": "VulnerableApp scanner metadata plus local class-level RequestParam annotation audit",
            "route_parameter_fields": {case["case_id"]: [str(case["field"])] for case in cases},
            "raw_source_persisted": False,
            "route_parameter_hashes": {case["case_id"]: _hash_text(str(case["field"])) for case in cases},
        },
        "seeds": list(SEEDS),
        "cases": [{key: value for key, value in case.items() if key not in {"path"}} | {"route_template_hash": _hash_text(str(case["path"]))} for case in cases],
        "counts": counts,
        "route_holdout": route_holdout,
        "episodes": episodes,
        "promotion": {
            "training_eligible": bool(counts["gold_record_count"] > 0 and counts["hard_negative_record_count"] > 0),
            "training_artifact_generated": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "honesty": {
            "typed_dom_marker_only": True,
            "independent_reference_required": True,
            "counterfactual_and_fresh_replay_present": True,
            "general_web_capability_not_established": True,
            "payload_strings_are_stdout_only": True,
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "script_execution": True,
            "script_execution_dom_marker_only": True,
            "database_write": False,
            "credentials_accessed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
    }
    report["report_sha256"] = _hash(report)
    dataset = {
        "schema_version": "pg246-vulnerableapp-independent-dom-holdout-dataset-v1",
        "source_report": str(REPORT.relative_to(ROOT)),
        "source_id": SOURCE_ID,
        "records": records,
        "counts": {"records": len(records), "gold": counts["gold_record_count"], "hard_negative": counts["hard_negative_record_count"], "quarantine": counts["quarantine_record_count"]},
        "split": {"implementation_holdout": True, "route_holdout": True, "seeds": list(SEEDS), "train_promotion_blocked_until_capacity_replay": True},
        "contract": {"typed_dom_oracle": True, "ai_participates_in_send": True, "independent_generator": True, "matched_negative": True, "fresh_reset_replay": True, "failure_attribution": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    dataset["dataset_sha256"] = _hash(dataset)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg246-vulnerableapp-independent-dom-holdout-protocol-v1",
        "target_image_digest": IMAGE_DIGEST,
        "source_implementation": "owasp-vulnerableapp-java-spring",
        "independent_from_pikachu": True,
        "independent_generator": GENERATOR_ID,
        "ai_selects_abstract_dom_channel": True,
        "typed_oracle": "controlled_browser_body_dataset_pg246_marker",
        "required_cases": [case["case_id"] for case in cases],
        "required_seeds": list(SEEDS),
        "fresh_reset_before_initial_and_replay": True,
        "matched_negative_and_counterfactual_required": True,
        "get_post_context_required": True,
        "unsupported_post_is_abstain_not_positive": True,
        "model_environment_attribution_required": True,
        "route_and_implementation_holdout_required": True,
        "canary_replay_required": True,
        "external_network_forbidden": True,
        "database_write_forbidden": True,
        "raw_payload_and_response_excluded": True,
        "promotion_blocked_until_capacity_and_forgetting_canary": True,
    }
    protocol["protocol_sha256"] = _hash(protocol)
    trace = {"schema_version": "pg246-vulnerableapp-independent-dom-holdout-trace-v1", "episodes": episodes, "records": records, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": bool(report["promotion"]["training_eligible"]), "memory_promotion_allowed": False}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text(
        "\n".join(
            [
                "# PG-246 VulnerableApp independent DOM holdout",
                "",
                f"device={device}; seeds={len(SEEDS)}; initial={counts['initial_episode_count']}; fresh={counts['fresh_container_count']}; GET={counts['get_count']}; POST={counts['post_count']}",
                f"AI sends={counts['ai_send_count']}; typed positives={counts['confirmed_positive_count']}; secure false accepts={counts['model_false_positive_count']}; POST-405 abstain={counts['post_405_abstain_count']}; replay matches={counts['replay_match_count']}",
                f"route positive recall={route_holdout['positive_recall']}; secure false accepts={route_holdout['secure_false_accept_count']}; POST abstain recall={route_holdout['post_abstain_recall']}",
                "",
                "VulnerableApp 是独立于 Pikachu 的 Java/Spring 实现；模型只选择抽象 DOM 通道，运行时绑定器使用临时 DOM marker。wire 只 stdout 显示，持久化为投影、哈希、失败归因、修复链和 fresh replay。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "device": str(device), "counts": counts, "route_holdout": route_holdout, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
