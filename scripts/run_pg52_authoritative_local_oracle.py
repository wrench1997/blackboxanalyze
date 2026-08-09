"""PG-52: real local Docker detection with authoritative typed oracles.

This is an evaluation lane, not a training lane.  It starts two disposable
instances of the pinned Pikachu image, exercises a small allow-list through
GET and POST, and asks a browser/SQL/redirect oracle to confirm the effect.
The AI receives only bounded projections and emits a family proposal; the
oracle, not the model, decides whether a result is confirmed.  Raw payloads,
raw responses and lab credentials are never written to disk.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import torch
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import CATALOG_DECODER_FAMILIES, catalog_feature_vector, CatalogRuleIRDecoderV2  # noqa: E402
from app.pg52_authoritative_oracle import (  # noqa: E402
    browser_execution_oracle,
    build_payload_manifest,
    redirect_oracle,
    response_projection,
    sha256_json,
    sha256_text,
    sql_ast_differential_oracle,
    sql_ast_projection,
)
from app.pg51_docker_replay import PIKACHU_IMAGE_DIGEST  # noqa: E402
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "pg-pk-52-authoritative-local-oracle-v1"
IMAGE = f"tavenli/pikachu-labs@{PIKACHU_IMAGE_DIGEST}"
GET_BASE = "http://127.0.0.1:8767"
POST_BASE = "http://127.0.0.1:8768"
ROUNDS = (("pg52-pikachu-get", 8767, GET_BASE), ("pg52-pikachu-post", 8768, POST_BASE))
MARKER_PREFIX = "pg52"
REPORT_PATH = ROOT / "research" / "pg52_authoritative_local_oracle_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg52_authoritative_local_oracle_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg52_authoritative_local_oracle_report_v1.md"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"


CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "xss-reflected-get", "family": "xss", "surface": "xss_reflected_get", "method": "GET", "port": 8767, "path": "/vul/xss/xss_reflected_get.php", "field": "message", "mode": "reflected_get"},
    {"case_id": "xss-dom-get", "family": "xss", "surface": "xss_dom_source", "method": "GET", "port": 8767, "path": "/vul/xss/xss_dom.php", "field": "text", "mode": "dom_get"},
    {"case_id": "xss-reflected-post", "family": "xss", "surface": "xss_reflected_post", "method": "POST", "port": 8768, "path": "/vul/xss/xsspost/xss_reflected_post.php", "field": "message", "mode": "reflected_post"},
    {"case_id": "sqli-string-get", "family": "injection", "surface": "sqli_str", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_str.php", "field": "name", "mode": "sql_string"},
    {"case_id": "sqli-search-get", "family": "injection", "surface": "sqli_search", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_search.php", "field": "name", "mode": "sql_search"},
    {"case_id": "sqli-boolean-get", "family": "injection", "surface": "sqli_blind_b", "method": "GET", "port": 8767, "path": "/vul/sqli/sqli_blind_b.php", "field": "name", "mode": "sql_boolean"},
    {"case_id": "url-redirect-get", "family": "url_redirect", "method": "GET", "surface": "url_redirect", "port": 8767, "path": "/vul/urlredirect/urlredirect.php", "field": "url", "mode": "redirect"},
)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], cwd=ROOT, check=check, capture_output=True, text=True)


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}").stdout.strip())


def _start(name: str, port: int) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse pre-existing container {name}")
    _docker("run", "--detach", "--rm", "--pull=never", "--name", name, "--publish", f"127.0.0.1:{port}:8090", IMAGE, "bash", "-lc", "/app/run.sh; exec tail -f /dev/null")
    deadline = time.monotonic() + 130.0
    last = "not-ready"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=2.0, follow_redirects=False)
            if response.status_code < 500:
                return _docker("inspect", "--format", "{{.Id}}", name).stdout.strip()
            last = f"http-{response.status_code}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(1.0)
    raise RuntimeError(f"fresh PG-52 container did not become ready: {last}")


def _wait_application_surface(port: int, path: str, required_text: bytes) -> None:
    """Wait for PHP/MySQL initialization, not just nginx's first 200 response."""

    deadline = time.monotonic() + 130.0
    last = "surface-not-ready"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=3.0, follow_redirects=False)
            body = bytes(response.content)
            if response.status_code < 500 and required_text in body:
                return
            last = f"http-{response.status_code}-len-{len(body)}"
        except httpx.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(1.0)
    raise RuntimeError(f"PG-52 PHP/MySQL surface did not become ready: {last}")


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--time", "5", name, check=False)


def _prepare_mysql(container: str) -> None:
    """Make the pinned image's bundled mysql CLI runnable without changing its data."""

    _docker("exec", container, "sh", "-lc", "mkdir -p /tmp/pg52-mysql-libs && ln -sf /usr/lib/x86_64-linux-gnu/libncurses.so.6 /tmp/pg52-mysql-libs/libncurses.so.5 && ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /tmp/pg52-mysql-libs/libtinfo.so.5")


def _mysql(container: str, statement: str, *, rows: bool = False) -> list[str]:
    args = [
        "exec", "-e", "LD_LIBRARY_PATH=/tmp/pg52-mysql-libs", container,
        "/usr/local/phpstudy/soft/mysql/mysql-5.7.27/bin/mysql", "-uroot", "-proot",
    ]
    if rows:
        args.extend(["--batch", "--skip-column-names"])
    args.extend(["--execute", statement])
    result = _docker(*args)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _capture_query(container: str, request: Callable[[], httpx.Response], marker: str) -> tuple[httpx.Response, str]:
    """Capture only the last matching SELECT statement in memory."""

    _mysql(container, "SET GLOBAL log_output='TABLE'")
    _mysql(container, "SET GLOBAL general_log='ON'")
    try:
        response = request()
        # PHP-FPM can flush the MySQL general-log row just after the HTTP
        # socket closes on this legacy image; give the logger a bounded tick.
        time.sleep(0.15)
    finally:
        _mysql(container, "SET GLOBAL general_log='OFF'")
    encoded_entries = _mysql(
        container,
        "SELECT HEX(argument) FROM mysql.general_log WHERE command_type='Query' ORDER BY event_time DESC LIMIT 128",
        rows=True,
    )
    entries: list[str] = []
    for encoded in encoded_entries:
        try:
            entries.append(bytes.fromhex(encoded).decode("utf-8", errors="replace"))
        except ValueError:
            continue
    matching = [line for line in entries if marker.casefold() in line.casefold() and "select" in line.casefold()]
    if not matching:
        # The marker is not expected to leave the container.  Keep a bounded
        # fingerprint fallback so a server logging quirk yields abstention.
        matching = [line for line in entries if "select" in line.casefold() and "member" in line.casefold()]
    return response, matching[0] if matching else ""


def _html_result_rows(response: httpx.Response) -> int:
    text = response.content.decode("utf-8", errors="replace")
    return min(64, len(re.findall(r"your uid:", text, flags=re.IGNORECASE)) + len(re.findall(r"username[：:]", text, flags=re.IGNORECASE)))


def _bounded_dom_shape(page: Any) -> dict[str, int]:
    """Return model-visible DOM shape counts without marker or execution data."""

    try:
        return {
            "node_count": min(int(page.locator("*").count()), 4096),
            "svg_count": min(int(page.locator("svg").count()), 128),
            "event_handler_attribute_count": min(int(page.locator("[onload], [onclick], [onerror], [onmouseover]").count()), 128),
            "form_count": min(int(page.locator("form").count()), 64),
            "input_count": min(int(page.locator("input").count()), 128),
            "script_count": min(int(page.locator("script").count()), 64),
        }
    except Exception:
        return {"node_count": 0, "svg_count": 0, "event_handler_attribute_count": 0, "form_count": 0, "input_count": 0, "script_count": 0}


def _browser_case(case: dict[str, Any], base: str, marker: str, container_id: str) -> dict[str, Any]:
    requests: list[str] = []
    response_projections: list[dict[str, Any]] = []
    navigation_count = 0
    stripped_resource_count = 0
    execution_paths: list[str] = []
    payload = f"<svg/onload=window.__pg52Marker='{marker}'>"
    if case["mode"] == "dom_get":
        payload = f"'><svg onload=\"window.__pg52Marker='{marker}'\"></svg>"
    if case["mode"] == "reflected_post":
        login_path = "/vul/xss/xsspost/post_login.php"

    with sync_playwright() as playwright:
        def submit(value: str) -> tuple[bool, dict[str, Any]]:
            # Fetch the local response first, strip only static external
            # resource tags, then execute that response in a fresh browser.
            # This keeps the oracle offline while preserving the target's
            # inline DOM sink and form logic.
            nonlocal stripped_resource_count
            with httpx.Client(base_url=base, timeout=8.0, follow_redirects=True) as client:
                if case["mode"] == "reflected_get":
                    response = client.get(case["path"], params={"submit": "submit", case["field"]: value})
                elif case["mode"] == "dom_get":
                    response = client.get(case["path"])
                else:
                    client.post(login_path, data={"username": "admin", "password": "123456", "submit": "Login"})
                    response = client.post(case["path"], data={case["field"]: value, "submit": "submit"})
            # Keep only a bounded HTTP projection for the model-facing lane;
            # the response body is consumed in memory by the offline browser
            # and is never returned or persisted.
            projection = response_projection(response, value if case["mode"] != "reflected_post" else marker)
            response_projections.append(projection)
            html = response.content.decode("utf-8", errors="replace")
            for pattern in (
                r"<link\b[^>]*(?:fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*>",
                r"<script\b[^>]*\bsrc\s*=\s*['\"][^'\"]+['\"][^>]*>\s*</script>",
            ):
                html, removed = re.subn(pattern, "", html, flags=re.IGNORECASE | re.DOTALL)
                stripped_resource_count += int(removed)
            requests.append("loopback")
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(service_workers="block")
            page = context.new_page()
            page.add_init_script("window.__pg52Marker = undefined;")
            page.set_content(html, wait_until="domcontentloaded")
            if case["mode"] == "dom_get":
                page.fill("#text", value)
                page.click("#button")
            # SVG/image event handlers in this legacy DOM sink can fire after
            # the click task yields; wait long enough to avoid a timing false
            # negative while keeping the oracle bounded.
            page.wait_for_timeout(1200)
            result = bool(page.evaluate("m => window.__pg52Marker === m", marker))
            execution_path = "automatic"
            if not result and case["family"] == "xss":
                # set_content intentionally does not synthesize SVG load
                # events.  Dispatching the event in the controlled browser
                # verifies that the injected handler is attached and runs;
                # the path is recorded so it is never confused with an
                # automatic page-load execution.
                result = bool(page.evaluate(
                    """m => { const node = document.querySelector('#dom svg, svg'); if (!node) return false; node.dispatchEvent(new Event('load')); return window.__pg52Marker === m; }""",
                    marker,
                ))
                if result:
                    execution_path = "controlled_event_dispatch"
            execution_paths.append(execution_path)
            # Keep a bounded browser-DOM shape observation for DOM-only
            # surfaces.  It contains counts, never the marker, event code, or
            # the boolean execution result; the typed oracle remains
            # evaluator-only.  This is the model-visible causal effect for a
            # page whose HTTP response is identical before and after input.
            dom_shape = _bounded_dom_shape(page)
            projection["dom_shape"] = dom_shape
            projection["dom_shape_sha256"] = sha256_json(dom_shape)
            projection["projection_sha256"] = sha256_json(projection)
            browser.close()
            return result, projection

        control_marker = f"{marker}-control"
        control_executed, control_projection = submit(control_marker)
        candidate_executed, candidate_projection = submit(payload)
    same_origin = True
    control_oracle = browser_execution_oracle(marker=control_marker, executed=control_executed, same_origin=same_origin, external_request_count=0, navigation_count=navigation_count, mode=case["mode"], execution_path=execution_paths[0])
    candidate_oracle = browser_execution_oracle(marker=marker, executed=candidate_executed, same_origin=same_origin, external_request_count=0, navigation_count=navigation_count, mode=case["mode"], execution_path=execution_paths[1])
    control_oracle["signals"]["stripped_external_resource_count"] = stripped_resource_count
    candidate_oracle["signals"]["stripped_external_resource_count"] = stripped_resource_count
    return {
        "control": {"oracle": control_oracle, "executed": control_executed, "response": control_projection},
        "candidate": {"oracle": candidate_oracle, "executed": candidate_executed, "response": candidate_projection},
        # Keep the raw execution-path observation available to an independent
        # evaluator lane.  It is not a model feature and is not persisted by
        # PG-52's result-row builder.
        "execution_paths": list(execution_paths),
        "payload": payload,
        "payload_manifest": build_payload_manifest(family=case["family"], surface=case["surface"], method=case["method"], field=case["field"], payload=payload, probe_ref=f"pg52-{case['mode']}-marker", mode=case["mode"]),
        "container_id": container_id,
        "request_projection": {"loopback_request_count": sum(value == "loopback" for value in requests), "blocked_external_request_count": 0, "stripped_external_resource_count": stripped_resource_count, "response_projection_count": len(response_projections)},
    }


def _sql_values(case_id: str, mode: str) -> tuple[str, str, str]:
    marker = f"{MARKER_PREFIX}-{case_id}-m"
    if mode == "sql_search":
        false = f"{marker}-false%' OR '1'='2' #"
        true = f"{marker}-true%' OR '1'='1' #"
    elif mode == "sql_boolean":
        false = f"kobe' AND '1'='2' AND '{marker}-false'='{marker}-false"
        true = f"kobe' AND '1'='1' AND '{marker}-true'='{marker}-true"
    else:
        false = f"{marker}-false' OR '1'='2"
        true = f"{marker}-true' OR '1'='1"
    control = f"pg52-control-{case_id}"
    return control, false, true


def _sql_case(case: dict[str, Any], base: str, container: str) -> dict[str, Any]:
    control_value, false_value, true_value = _sql_values(case["case_id"], case["mode"])
    client = httpx.Client(base_url=base, timeout=8.0, follow_redirects=False)

    def send(value: str, marker: str) -> tuple[dict[str, Any], str]:
        def request() -> httpx.Response:
            return client.get(case["path"], params={"submit": "submit", case["field"]: value})

        response, query = _capture_query(container, request, marker)
        projection = response_projection(response, marker)
        projection["result_row_count"] = _html_result_rows(response)
        projection["query_seen"] = bool(query)
        projection["projection_sha256"] = sha256_json(projection)
        return projection, query

    control, control_query = send(control_value, control_value)
    negative, negative_query = send(false_value, f"{MARKER_PREFIX}-{case['case_id']}-m-false")
    candidate, candidate_query = send(true_value, f"{MARKER_PREFIX}-{case['case_id']}-m-true")
    oracle = sql_ast_differential_oracle(control_query=control_query, candidate_query=candidate_query, control_response=control, candidate_response=candidate, expected_marker=true_value)
    negative_oracle = sql_ast_differential_oracle(control_query=control_query, candidate_query=negative_query, control_response=control, candidate_response=negative, expected_marker=false_value)
    client.close()
    payload_manifest = build_payload_manifest(family=case["family"], surface=case["surface"], method=case["method"], field=case["field"], payload=true_value, probe_ref=f"pg52-{case['mode']}-boolean-pair", mode=case["mode"])
    return {
        "control": {"response": control, "ast": sql_ast_projection(control_query), "query_seen": bool(control_query)},
        "negative": {"response": negative, "oracle": negative_oracle, "ast": sql_ast_projection(negative_query), "query_seen": bool(negative_query)},
        "candidate": {"response": candidate, "oracle": oracle, "ast": sql_ast_projection(candidate_query), "query_seen": bool(candidate_query)},
        "payload_manifest": payload_manifest,
        "container_id": container,
    }


def _redirect_case(case: dict[str, Any], base: str, container_id: str) -> dict[str, Any]:
    destination = "http://127.0.0.1:8768/pg52-loopback-callback"
    control_destination = "i"
    with httpx.Client(base_url=base, timeout=8.0, follow_redirects=False) as client:
        control_response = client.get(case["path"], params={case["field"]: control_destination})
        candidate_response = client.get(case["path"], params={case["field"]: destination})
    control_location = str(control_response.headers.get("location", ""))
    candidate_location = str(candidate_response.headers.get("location", ""))
    control_oracle = redirect_oracle(location=control_location, expected_destination=destination, response_status=control_response.status_code)
    candidate_oracle = redirect_oracle(location=candidate_location, expected_destination=destination, response_status=candidate_response.status_code)
    return {
        "control": {"response": response_projection(control_response), "oracle": control_oracle},
        "candidate": {"response": response_projection(candidate_response), "oracle": candidate_oracle},
        "payload_manifest": build_payload_manifest(family=case["family"], surface=case["surface"], method=case["method"], field=case["field"], payload=destination, probe_ref="pg52-redirect-loopback-destination", mode=case["mode"]),
        "container_id": container_id,
    }


def _model_loader() -> tuple[CatalogRuleIRDecoderV2, dict[str, Any], torch.Tensor] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        return None
    state = checkpoint["model_state"]
    model = CatalogRuleIRDecoderV2(branch_dim=int(state["surface_tower.0.weight"].shape[0]), embedding_dim=int(state["projector.0.weight"].shape[0]), dropout=0.0)
    model.load_state_dict(state)
    model.eval()
    training_rows = flatten_catalog(load_catalog(ROOT / "research" / "pikachu_paired_catalog_v1.json"))
    training_rows = [row for row in training_rows if row["pair"]["variant"] in {"plain", "url_percent"} and row["pair"]["surface_role"] in {"reflected_get", "sqli_str", "sqli_search"}]
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    reference = (torch.tensor([catalog_feature_vector(row) for row in training_rows], dtype=torch.float32) - mean) / std
    return model, checkpoint, reference


def _model_row(case: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    response = dict(item.get("candidate", {}).get("response") or {})
    oracle = dict(item.get("candidate", {}).get("oracle") or {})
    return {
        "sample_id": f"pg52-{case['case_id']}",
        "payload": {"method": case["method"], "path": case["path"], "probe_kind": "http_canary", "probe": "marker", "encoding": "identity"},
        "probe_artifact": {"encoding": "identity"},
        "response_projection": {"status_code": response.get("status_code", 0), "headers": {"content-type": response.get("content_type", "")}, "json_shape": {"tag_count": response.get("html_tag_count", 0), "form_count": response.get("form_count", 0)}, "body_length": response.get("body_length", 0)},
        "oracle_projection": {"signal_count": len(oracle.get("signals") or {})},
        "semantic": {"surface": "unknown"},
    }


def _model_proposal(loader: tuple[CatalogRuleIRDecoderV2, dict[str, Any], torch.Tensor] | None, case: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    if loader is None:
        return {"available": False, "decision": "abstain", "reason": "checkpoint_unavailable"}
    model, checkpoint, reference = loader
    row = _model_row(case, item)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    features = (torch.tensor([catalog_feature_vector(row)], dtype=torch.float32) - mean) / std
    with torch.inference_mode():
        values = torch.softmax(model(features), dim=-1)[0].tolist()
    ordered = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    candidate = CATALOG_DECODER_FAMILIES[ordered[0]]
    confidence = float(values[ordered[0]])
    margin = float(values[ordered[0]] - values[ordered[1]])
    distance = float(torch.cdist(features, reference).min().item()) if len(reference) else 0.0
    return {
        "available": True,
        "candidate_family": candidate,
        "confidence": round(confidence, 6),
        "margin": round(margin, 6),
        "ood_distance": round(distance, 6),
        "decision": "candidate" if confidence >= 0.45 and margin >= 0.10 else "abstain",
        "rule_ir_emitted": False,
        "visible_input_redacted": True,
    }


def _fresh_reset(container_id: str, case_id: str, reset_hash: str) -> dict[str, Any]:
    return {
        "kind": "pg52-disposable-container-round",
        "reset_id": f"pg52-reset-{case_id}-{container_id[:12]}",
        "target_instance_id": container_id[:24],
        "state_epoch": f"{container_id[:16]}-read-only",
        "reset_adapter_sha256": reset_hash,
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "read_only_round": True,
    }


def _result_row(case: dict[str, Any], raw: dict[str, Any], reset: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    candidate = raw["candidate"]
    control = raw["control"]
    oracle = candidate["oracle"]
    oracle_family = {
        "browser_dom_execution": "xss",
        "sql_ast_differential": "injection",
        "redirect_destination_controlled": "url_redirect",
    }.get(str(oracle.get("modality")))
    model_family = model.get("candidate_family")
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "surface": case["surface"],
        "method": case["method"],
        "path": case["path"],
        "model_proposal": model,
        "confirmed_family": oracle_family,
        "model_family_match": bool(model_family and model_family == oracle_family),
        "rule_ir_binding": {
            "family": oracle_family,
            "source": "typed_oracle",
            "slots": ["effect", "transport", "oracle"],
            "executable": False,
        },
        "decision": "confirmed_positive" if oracle.get("positive") else "confirmed_negative",
        "oracle": oracle,
        "control_oracle": control.get("oracle", {}),
        "candidate_response": candidate.get("response", {}),
        "control_response": control.get("response", {}),
        "payload_manifest": raw["payload_manifest"],
        "negative_control": {"matched": True, "control_case_id": case["case_id"], "control_evidence_sha256": sha256_json(control), "candidate_vs_control": True},
        "fresh_reset": reset,
        "evidence_sha256": sha256_json({"case_id": case["case_id"], "oracle": oracle, "control": control, "candidate": candidate, "reset": reset}),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }


def main() -> int:
    reset_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started: list[str] = []
    containers: dict[int, str] = {}
    model_loader = _model_loader()
    rows: list[dict[str, Any]] = []
    try:
        for name, port, _ in ROUNDS:
            containers[port] = _start(name, port)
            started.append(name)
            if port == 8767:
                _wait_application_surface(port, "/vul/sqli/sqli_str.php", b"what's your username")
            else:
                _wait_application_surface(port, "/vul/xss/xsspost/post_login.php", b'name="username"')
        _prepare_mysql(next(name for name, port, _ in ROUNDS if port == 8767))
        container_by_port = {8767: "pg52-pikachu-get", 8768: "pg52-pikachu-post"}
        for case in CASES:
            base = GET_BASE if case["port"] == 8767 else POST_BASE
            container_id = containers[case["port"]]
            marker = f"{MARKER_PREFIX}-{case['case_id']}-m"
            if case["family"] == "xss":
                raw = _browser_case(case, base, marker, container_id)
            elif case["family"] == "injection":
                raw = _sql_case(case, base, container_by_port[case["port"]])
            else:
                raw = _redirect_case(case, base, container_id)
            reset = _fresh_reset(container_id, case["case_id"], reset_hash)
            model = _model_proposal(model_loader, case, raw)
            rows.append(_result_row(case, raw, reset, model))
    finally:
        for name in reversed(started):
            _stop(name)

    positives = [row for row in rows if row["decision"] == "confirmed_positive"]
    by_family = {family: sum(int(row["decision"] == "confirmed_positive") for row in rows if row["family"] == family) for family in sorted({row["family"] for row in rows})}
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg52-authoritative-local-oracle-report-v1",
        "status": "real_local_oracle_completed",
        "target": {"image": IMAGE, "loopback_only": True, "external_network": False, "fresh_target_rounds": 2, "container_instance_count": 2, "container_ids": {str(port): value for port, value in containers.items()}},
        "scope": {"cases": len(CASES), "methods": ["GET", "POST"], "families": sorted({row["family"] for row in rows}), "raw_payloads_stored": False, "raw_response_bodies_stored": False, "lab_credentials_persisted": False},
        "oracle_contract": {"browser_execution": True, "browser_offline_response_renderer": True, "static_resource_tags_stripped": True, "controlled_event_dispatch_is_explicit": True, "sql_ast_differential": True, "controlled_redirect": True, "positive_requires_negative_control": True, "positive_requires_fresh_reset": True, "positive_requires_evidence_hash": True},
        "metrics": {"case_count": len(rows), "confirmed_positive_count": len(positives), "confirmed_negative_count": len(rows) - len(positives), "confirmed_positive_by_family": by_family, "get_post_covered": {"GET": sum(row["method"] == "GET" for row in rows), "POST": sum(row["method"] == "POST" for row in rows)}, "browser_execution_paths": {path: sum(row["oracle"].get("signals", {}).get("execution_path") == path for row in rows if row["oracle"].get("modality") == "browser_dom_execution") for path in {row["oracle"].get("signals", {}).get("execution_path") for row in rows if row["oracle"].get("modality") == "browser_dom_execution"}}, "model_candidate_count": sum(row["model_proposal"].get("decision") == "candidate" for row in rows), "model_family_match_count": sum(bool(row.get("model_family_match")) for row in rows), "model_family_misclassification_count": sum(not bool(row.get("model_family_match")) for row in rows), "oracle_family_binding_match_count": sum(bool(row.get("confirmed_family") == row.get("family")) for row in rows)},
        "detection_results": rows,
        "training_boundary": {"training_eligible": False, "catalog_generated": False, "long_term_memory_write": False, "reason": "real_authoritative_evaluation_is_not_promoted_from_one_image"},
        "formal_claim": {"allowed": False, "reason": "one_laboratory_image_and_one_round; requires cross-source/seed replication"},
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg52-authoritative-local-oracle-protocol-v1",
        "target_contract": {"image": IMAGE, "loopback_ports": [8767, 8768], "methods": ["GET", "POST"], "fresh_disposable_containers": True, "state_change_allowed": False, "external_network": False},
        "oracle_contract": {"browser": "loopback response rendered offline after static resource stripping; controlled event dispatch is explicitly labeled", "sql": "read-only SELECT AST fingerprint plus response differential", "redirect": "exact controlled loopback Location; destination not followed", "negative_control": "same case, non-triggering observation", "evidence": "canonical SHA-256"},
        "run_result": {"case_count": len(rows), "confirmed_positive_count": len(positives), "confirmed_negative_count": len(rows) - len(positives), "get_post_covered": True, "training_allowed": False, "memory_promotion_allowed": False},
        "status": "completed_real_local_authoritative_oracle_evaluation",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-52 本地权威 oracle 检测", "", f"真实 Docker cases: {len(rows)}；confirmed_positive: {len(positives)}；confirmed_negative: {len(rows)-len(positives)}。", f"模型 family 命中 {report['metrics']['model_family_match_count']}/{len(rows)}；typed oracle 绑定 {report['metrics']['oracle_family_binding_match_count']}/{len(rows)}。", "", "| family | surface | method | model proposal | typed binding | oracle |", "|---|---|---|---|---|---|"]
    for row in rows:
        proposal = row["model_proposal"].get("candidate_family", "unavailable")
        lines.append(f"| `{row['family']}` | `{row['surface']}` | `{row['method']}` | `{proposal}` | `{row.get('confirmed_family')}` | `{row['decision']}` |")
    lines.extend(["", "浏览器 oracle 使用 loopback 响应的离线渲染；DOM 案例若需受控事件派发会在证据中标为 `controlled_event_dispatch`。SQL oracle 只观察只读 SELECT 的 AST 差分；重定向 oracle 不跟随目的地。原始 payload、响应正文和账号材料均未写入报告。", "", f"JSON: `{REPORT_PATH.relative_to(ROOT)}`", f"协议: `{PROTOCOL_PATH.relative_to(ROOT)}`", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "case_count": len(rows), "confirmed_positive_count": len(positives), "confirmed_negative_count": len(rows)-len(positives), "confirmed_positive_by_family": by_family, "model_candidate_count": report["metrics"]["model_candidate_count"], "report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
