# -*- coding: utf-8 -*-
"""PG-268B: replay browser-discovered Pikachu GET/POST surfaces.

Every parameterized surface is run in a fresh local no-volume container with
AI candidate, independent reference, and matched negative channels. Exact
payloads and bounded echo excerpts stay in the human catalog; the abstract
training dataset stores only Rule-IR tokens and evidence hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlencode

import httpx

try:
    from playwright.sync_api import Browser, sync_playwright
except Exception:  # pragma: no cover
    Browser = Any  # type: ignore[assignment,misc]
    sync_playwright = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load(ROOT / "scripts" / "run_pg214_pikachu_fixed_sql_loop.py", "pg268b_pg214_helpers")
RESEARCH = ROOT / "research"
MANIFEST = RESEARCH / "pg268_pikachu_browser_parameterized_crawl_manifest_v1.json"
PG179_MANIFEST = RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json"
CATALOG = RESEARCH / "pg268_pikachu_parameterized_replay_catalog_v1.json"
DATASET = RESEARCH / "pg268_pikachu_parameterized_replay_dataset_v1.json"
REPORT = RESEARCH / "pg268_pikachu_parameterized_replay_report_v1.json"
TRACE = RESEARCH / "pg268_pikachu_parameterized_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg268_pikachu_parameterized_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg268_pikachu_parameterized_replay_report_v1.md"
BASE_PORT = 5700
SEED = 26802


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _family(path: str) -> str:
    p = path.casefold()
    for key, name in (("/sqli/", "sql"), ("/xss/", "xss"), ("/xxe/", "xxe"), ("/rce/", "rce"), ("/ssrf/", "ssrf"), ("/fileinclude/", "file"), ("/unsafedownload/", "file"), ("/urlredirect/", "redirect"), ("/unserilization/", "serialization"), ("/burteforce/", "logic"), ("/csrf/", "logic"), ("/overpermission/", "logic"), ("/infoleak/", "infoleak")):
        if key in p:
            return name
    return "other"


def _rule(path: str, family: str) -> str:
    p = path.casefold()
    if family == "sql":
        return "sql_boolean" if "blind" in p else "sql_widebyte" if "widebyte" in p else "sql_syntax"
    return {"xss": "dom_marker", "xxe": "xml_entity_boundary", "rce": "expression_boundary", "ssrf": "loopback_fetch_boundary", "file": "file_path_boundary", "redirect": "location_boundary", "serialization": "serialized_shape_boundary"}.get(family, "logic_surface_observation")


def _primary(surface: Mapping[str, Any]) -> str:
    fields = [str(x) for x in list(surface.get("query_params") or []) + list(surface.get("form_params") or [])]
    ignored = {str(x).casefold() for x in list(surface.get("hidden_params") or []) + list(surface.get("submit_params") or [])}
    return next((x for x in fields if x.casefold() not in ignored and x.casefold() not in {"submit", "button"}), fields[0] if fields else "")


def _marker(marker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", marker)[:48]
    return f"<img src=/__pg268_missing_{safe} onerror=\"document.documentElement.dataset.pg268='{safe}'\">"


def _values(surface: Mapping[str, Any], marker: str, origin: str, variant: str) -> tuple[dict[str, str], str, str]:
    path = str(surface.get("path", ""))
    family = _family(path)
    field = _primary(surface)
    fields = [str(x) for x in list(surface.get("query_params") or []) + list(surface.get("form_params") or [])]
    hidden = {str(x).casefold() for x in list(surface.get("hidden_params") or [])}
    submit = {str(x).casefold() for x in list(surface.get("submit_params") or [])}
    values: dict[str, str] = {}
    for name in fields:
        lower = name.casefold()
        values[name] = "submit" if lower in submit or lower == "submit" else f"{marker}-HIDDEN" if lower in hidden else "" if lower in {"file", "uploadfile"} else f"{marker}-VALUE"
    if not field:
        return values, field, "no_parameter_field"
    lower_path, lower_field = path.casefold(), field.casefold()
    if variant == "negative":
        values[field], selection = f"{marker}-NEG", "negative_control"
    elif variant == "reference":
        values[field], selection = f"{marker}-REF", "reference_probe"
        if family == "sql" and "sqli_id" in lower_path:
            values[field] = "1"
        elif family == "sql" and any(x in lower_path for x in ("sqli_str", "sqli_search", "sqli_x")):
            values[field] = "1"
    elif family == "xss":
        values[field], selection = _marker(marker), "dom_marker_candidate"
    elif family == "sql" and "sqli_del" not in lower_path and lower_field not in {"username", "password", "vcode"}:
        values[field] = "1 OR 1=1" if lower_field in {"id", "uid"} else "1' AND '1'='1" if "blind" in lower_path else "kobe%df%27 OR 1=1" if "widebyte" in lower_path else "kobe' OR '1'='1"
        selection = "sql_syntax_candidate"
    elif family == "xxe":
        values[field], selection = "<!DOCTYPE r [<!ENTITY pg268 'PG268-XXE'>]><r>&pg268;</r>", "internal_entity_candidate"
    elif family == "rce":
        values[field], selection = ("1+1" if "eval" in lower_path else "127.0.0.1"), "bounded_expression_candidate"
    elif family == "ssrf":
        values[field], selection = origin + "/", "loopback_fetch_candidate"
    elif family == "file":
        values[field], selection = "index.php", "local_file_candidate"
    elif family == "redirect":
        values[field], selection = origin + "/pg268-loopback-exit", "loopback_location_candidate"
    elif family == "serialization":
        values[field], selection = 'a:1:{s:1:"x";s:6:"PG268";}', "benign_serialized_shape"
    else:
        values[field], selection = f"{marker}-CANDIDATE", "bounded_canary_candidate"
    return values, field, selection


def _request(client: httpx.Client, surface: Mapping[str, Any], values: Mapping[str, str], marker: str) -> dict[str, Any]:
    method, path = str(surface.get("method", "GET")).upper(), str(surface.get("path", "/"))
    encoded = urlencode(list(values.items()), doseq=True, quote_via=quote)
    request_line, body = (f"GET <LOOPBACK_ORIGIN>{path}?{encoded}", None) if method == "GET" else (f"POST <LOOPBACK_ORIGIN>{path}", encoded)
    try:
        response = client.get(path, params=dict(values), follow_redirects=False) if method == "GET" else client.post(path, data=dict(values), follow_redirects=False)
        text = response.text
        pos = text.find(marker)
        if pos < 0:
            pos = text.find(str(marker).split("-", 1)[0])
        return {"sent": True, "method": method, "status": response.status_code, "location": str(response.headers.get("location", ""))[:500], "content_type": str(response.headers.get("content-type", ""))[:120], "body_length": len(response.content), "body_sha256": hashlib.sha256(response.content).hexdigest(), "marker_reflected": pos >= 0, "echo_excerpt": text[max(0, pos - 120):pos + 360][:480] if pos >= 0 else "", "request_line": request_line, "body": body, "wire_sha256": _digest(request_line + ("\n\n" + body if body is not None else ""))}
    except Exception as exc:
        return {"sent": False, "method": method, "status": None, "error": type(exc).__name__, "request_line": request_line, "body": body, "wire_sha256": _digest(request_line + ("\n\n" + body if body is not None else ""))}


def _browser_xss(browser: Browser | None, origin: str, surface: Mapping[str, Any], values: Mapping[str, str], marker: str) -> dict[str, Any]:
    if browser is None:
        return {"available": False, "executed": False, "reason": "playwright_unavailable"}
    page = browser.new_page()
    try:
        path, method = str(surface.get("path", "/")), str(surface.get("method", "GET")).upper()
        url = origin + path + ("?" + urlencode(list(values.items()), doseq=True, quote_via=quote) if method == "GET" and values else "")
        page.goto(url, wait_until="domcontentloaded", timeout=12000)
        page.wait_for_timeout(120)
        safe = re.sub(r"[^A-Za-z0-9_-]", "", marker)[:48]
        observed = page.locator("html").get_attribute("data-pg268") or ""
        return {"available": True, "executed": observed == safe, "observed_marker": observed[:80], "url": url, "dom_excerpt": page.locator("body").inner_text()[:240]}
    except Exception as exc:
        return {"available": True, "executed": False, "reason": type(exc).__name__}
    finally:
        page.close()


def _source_hash(name: str, path: str) -> str:
    try:
        result = PG214._docker("exec", name, "sha256sum", "/app/www" + path)
        value = str(result).split()[0].strip().casefold()
        return value if re.fullmatch(r"[0-9a-f]{64}", value) else ""
    except Exception:
        return ""


def _surfaces() -> list[dict[str, Any]]:
    fresh = _read(MANIFEST)
    rows = [dict(row) for row in list(fresh.get("route_catalog") or []) if isinstance(row, dict) and row.get("has_parameter_context")]
    known = {(str(row.get("path")), str(row.get("method"))) for row in rows}
    old = _read(PG179_MANIFEST)
    for old_row in list(old.get("route_catalog") or []):
        if not isinstance(old_row, dict) or not old_row.get("get_query_params"):
            continue
        row = {"path": str(old_row.get("path")), "method": "GET", "source": "legacy_query_link", "query_params": list(old_row.get("get_query_params") or []), "form_params": [], "hidden_params": [], "submit_params": [], "enctype": None, "has_parameter_context": True}
        key = (row["path"], "GET")
        if key not in known:
            rows.append(row)
            known.add(key)
    rows.sort(key=lambda row: (str(row.get("path")), str(row.get("method")), str(row.get("source"))))
    return rows


def _abstract(surface: Mapping[str, Any], oracle: Mapping[str, Any], source_hash: str, reset: Mapping[str, Any]) -> dict[str, Any]:
    path, method = str(surface.get("path", "")), str(surface.get("method", "GET")).upper()
    family, rule = _family(path), _rule(path, _family(path))
    confirmed = bool(oracle.get("confirmed_positive"))
    channel = "query" if str(surface.get("source", "")).endswith("query") or (method == "GET" and not surface.get("form_params")) else "form"
    tokens = ["[BOS]", "phase=observe", f"surface={family}_parameterized", f"method={method}", f"field_bucket={min(len(list(surface.get('form_params') or surface.get('query_params') or [])), 8)}", f"channel={channel}", "candidate_sent=1", "reference_sent=1", "negative_sent=1", "fresh_reset=1", f"source_attested={int(bool(source_hash))}", f"rule_ir={rule}", f"oracle={oracle.get('oracle_type', 'oracle_gap')}", f"outcome={oracle.get('outcome_class', 'oracle_gap')}", f"negative_clean={int(bool(oracle.get('negative_clean')))}", f"next_action={'replay_confirmed' if confirmed else 'abstain_or_repair'}", "phase=diagnose", f"family={family}", f"lane={'gold' if confirmed else 'hard_negative'}", "phase=replay", f"replay_expected={'typed' if confirmed else 'abstain'}", "[EOS]"]
    return {"schema_version": "pg268-pikachu-parameterized-replay-record-v1", "record_id": f"pg268:{path}:{method}:{_digest(tokens)[:12]}", "source": "pg268_pikachu_parameterized_replay", "seed": int(surface.get("seed", 0) or 0), "route": path, "method": method, "surface_class": f"{family}_parameterized", "lane": "gold" if confirmed else "hard_negative", "repair_action": "retry_candidate" if confirmed else "recheck_oracle", "failure_kind": str(oracle.get("outcome_class", "oracle_gap")), "replay_expected": "typed" if confirmed else "abstain", "classification_position": max(len(tokens) - 4, 0), "tokens": tokens, "trajectory_hash": _digest(tokens), "quality_reasons": ["pg268_ai_reference_negative_fresh_oracle_complete" if oracle.get("fresh_complete") and oracle.get("candidate_sent") and oracle.get("reference_sent") and oracle.get("negative_sent") else "pg268_incomplete", "raw_payload_excluded"], "source_evidence_hash": str(oracle.get("evidence_hash", "")), "route_source_sha256": source_hash, "model_self_error_detected": False, "payload_grounded_eligible": confirmed, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "next_rule_class": rule, "family_class": family, "channel_class": channel, "pair_role": "single", "source_role": "observed", "source_lane": "pg268_audited_local", "rule_ir_class": rule, "belief_class": "confirmed_effect" if confirmed else "oracle_gap", "probe_class": "replay_confirm" if confirmed else "negative_control", "unknown_abstain_class": "continue_family" if confirmed else "unknown_family_abstain"}


def main() -> int:
    PG214.BASE_PORT = BASE_PORT
    surfaces = _surfaces()
    human_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    policy_attempts: Counter[str] = Counter()
    policy_feedback: defaultdict[str, Counter[str]] = defaultdict(Counter)
    started = time.monotonic()
    browser_context = sync_playwright().start() if sync_playwright is not None else None
    browser = browser_context.chromium.launch(headless=True) if browser_context is not None else None
    try:
        for index, original in enumerate(surfaces):
            surface = dict(original)
            path, method = str(surface.get("path", "/")), str(surface.get("method", "GET")).upper()
            family = _family(path)
            surface.update({"id": f"pg268-{index:03d}", "seed": SEED + index, "family": family, "rule_ir": _rule(path, family)})
            marker = f"PG268-{index:03d}"
            item: dict[str, Any] = {"record_id": f"pg268:{index:03d}", "route": surface, "source": {}, "ai": {}, "reference": {}, "negative": {}, "oracle": {}, "feedback": {}}
            name = ""
            try:
                if any(str(field).casefold() in {"uploadfile", "file"} for field in list(surface.get("form_params") or [])):
                    oracle = {"confirmed_positive": False, "typed_effect": False, "oracle_type": "oracle_gap", "outcome_class": "unsupported_file_surface", "reason": "multipart fixture lane not enabled", "evidence_hash": _digest({"route": path, "reason": "file_fixture_not_enabled"}), "training_eligible": False, "fresh_complete": False}
                    item["oracle"] = oracle
                    human_rows.append(item)
                    abstract_rows.append(_abstract(surface, oracle, "", {"fresh_target": False}))
                    continue
                name, port, container_id, reset = PG214._start(SEED + index, index)
                origin = f"http://127.0.0.1:{port}"
                source_hash = _source_hash(name, path)
                item["source"] = {"image": PG214.IMAGE, "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest(), "source_sha256": source_hash, "fresh_reset": reset}
                client = httpx.Client(base_url=origin, timeout=15.0, follow_redirects=False)
                try:
                    negative_values, field, negative_selection = _values(surface, marker + "-NEG", origin, "negative")
                    reference_values, _, reference_selection = _values(surface, marker + "-REF", origin, "reference")
                    candidate_values, _, candidate_selection = _values(surface, marker, origin, "candidate")
                    policy_attempts[family] += 1
                    negative = _request(client, surface, negative_values, marker + "-NEG")
                    negative_browser = _browser_xss(browser, origin, surface, negative_values, marker + "-NEG") if family == "xss" else {"available": False, "executed": False}
                    candidate = _request(client, surface, candidate_values, marker)
                    candidate_browser = _browser_xss(browser, origin, surface, candidate_values, marker) if family == "xss" else {"available": False, "executed": False}
                    reference = _request(client, surface, reference_values, marker + "-REF")
                    reference_browser = _browser_xss(browser, origin, surface, reference_values, marker + "-REF") if family == "xss" else {"available": False, "executed": False}
                    negative_clean = not bool(negative.get("marker_reflected")) and not bool(negative_browser.get("executed"))
                    confirmed, oracle_type, outcome_class, reason = False, "oracle_gap", f"{family}_oracle_gap", "no_typed_effect_evaluator_for_surface"
                    if family == "xss":
                        confirmed = bool(candidate_browser.get("executed") and reference_browser.get("executed") and negative_clean)
                        oracle_type, outcome_class, reason = "browser_dom_execution", ("confirmed_local_xss_dom" if confirmed else "xss_abstain"), ("candidate_reference_dom_agreement" if confirmed else "dom_execution_not_confirmed")
                    elif family == "sql" and "sqli_del" not in path.casefold() and "login" not in path.casefold() and "header" not in path.casefold():
                        candidate_len, reference_len, negative_len = int(candidate.get("body_length") or 0), int(reference.get("body_length") or 0), int(negative.get("body_length") or 0)
                        differential = abs(candidate_len - reference_len) >= 24 and candidate.get("status") == reference.get("status") and candidate_len > negative_len
                        confirmed = bool(differential and negative_clean)
                        oracle_type, outcome_class, reason = "response_shape_differential", ("confirmed_local_sql_shape" if confirmed else "sql_abstain"), ("candidate_reference_negative_shape_differential" if confirmed else "row_shape_not_separable")
                    elif family == "redirect":
                        location = str(candidate.get("location") or "")
                        confirmed = bool(location.startswith(origin) and location != origin + path)
                        oracle_type, outcome_class, reason = "loopback_location_change", ("confirmed_local_redirect_effect" if confirmed else "redirect_abstain"), ("loopback_location_changed" if confirmed else "location_not_changed")
                    elif family == "xxe":
                        confirmed = bool("PG268-XXE" in str(candidate.get("echo_excerpt", "")) and negative_clean)
                        oracle_type, outcome_class, reason = "internal_entity_expansion", ("confirmed_local_xml_entity_effect" if confirmed else "xxe_abstain"), ("internal_entity_expanded" if confirmed else "entity_expansion_not_observed")
                    elif family == "rce" and "eval" in path.casefold():
                        confirmed = bool("2" in str(candidate.get("echo_excerpt", "")) and negative_clean)
                        oracle_type, outcome_class, reason = "bounded_expression_result", ("confirmed_local_expression_effect" if confirmed else "rce_abstain"), ("bounded_expression_result_observed" if confirmed else "expression_result_not_observed")
                    evidence = {"fresh_reset_sha256": _digest(reset), "source_sha256": source_hash, "candidate": candidate, "reference": reference, "negative": negative, "candidate_browser": candidate_browser, "reference_browser": reference_browser, "negative_browser": negative_browser, "oracle_type": oracle_type, "negative_clean": negative_clean}
                    evidence_hash = _digest(evidence)
                    oracle = {"confirmed_positive": confirmed, "typed_effect": confirmed, "oracle_type": oracle_type, "outcome_class": outcome_class, "reason": reason, "negative_clean": negative_clean, "evidence_hash": evidence_hash, "training_eligible": False, "vulnerability_claim_allowed": False, "fresh_complete": bool(reset.get("fresh_target") and reset.get("completed")), "candidate_sent": bool(candidate.get("sent")), "reference_sent": bool(reference.get("sent")), "negative_sent": bool(negative.get("sent"))}
                    policy_feedback[family][outcome_class] += 1
                    item.update({"route": surface, "ai": {"policy": "pg268_bounded_family_ucb_v1", "selection": "candidate", "selection_reason": candidate_selection, "field": field, "payload": candidate_values, "wire": candidate, "browser_oracle": candidate_browser}, "reference": {"selection": reference_selection, "payload": reference_values, "wire": reference, "browser_oracle": reference_browser}, "negative": {"selection": negative_selection, "payload": negative_values, "wire": negative, "browser_oracle": negative_browser}, "oracle": oracle, "evidence": evidence, "feedback": {"outcome_class": outcome_class, "next_action": "replay_confirmed" if confirmed else "abstain_or_repair"}})
                    human_rows.append(item)
                    abstract_rows.append(_abstract(surface, oracle, source_hash, reset))
                finally:
                    client.close()
            except Exception as exc:
                oracle = {"confirmed_positive": False, "typed_effect": False, "oracle_type": "runner_error", "outcome_class": "runner_error", "reason": type(exc).__name__, "evidence_hash": _digest({"route": path, "error": type(exc).__name__}), "training_eligible": False, "fresh_complete": False}
                item["oracle"] = oracle
                human_rows.append(item)
                abstract_rows.append(_abstract(surface, oracle, "", {"fresh_target": False}))
            finally:
                if name:
                    PG214._stop(name)
    finally:
        if browser is not None:
            browser.close()
        if browser_context is not None:
            browser_context.stop()

    counts = {"surface_count": len(human_rows), "get_count": sum(int(str(row.get("route", {}).get("method", "")).upper() == "GET") for row in human_rows), "post_count": sum(int(str(row.get("route", {}).get("method", "")).upper() == "POST") for row in human_rows), "ai_send_count": sum(int(bool(row.get("ai", {}).get("wire", {}).get("sent"))) for row in human_rows), "reference_send_count": sum(int(bool(row.get("reference", {}).get("wire", {}).get("sent"))) for row in human_rows), "negative_send_count": sum(int(bool(row.get("negative", {}).get("wire", {}).get("sent"))) for row in human_rows), "confirmed_positive_count": sum(int(bool(row.get("oracle", {}).get("confirmed_positive"))) for row in human_rows), "false_positive_count": sum(int(bool(row.get("oracle", {}).get("confirmed_positive")) and bool(row.get("negative", {}).get("browser_oracle", {}).get("executed"))) for row in human_rows), "abstain_count": sum(int(not bool(row.get("oracle", {}).get("confirmed_positive"))) for row in human_rows), "fresh_reset_count": sum(int(bool(row.get("source", {}).get("fresh_reset", {}).get("fresh_target")) and bool(row.get("source", {}).get("fresh_reset", {}).get("completed"))) for row in human_rows), "source_attested_count": sum(int(bool(row.get("source", {}).get("source_sha256"))) for row in human_rows), "elapsed_seconds": round(time.monotonic() - started, 3)}
    catalog = {"schema_version": "pg268-pikachu-parameterized-replay-catalog-v1", "status": "completed_human_review_catalog", "entries": human_rows, "counts": counts, "raw_payloads_are_human_review_only": True, "raw_response_bodies_stored": False, "external_network": False, "catalog_sha256": ""}
    catalog["catalog_sha256"] = _digest(catalog)
    _write(CATALOG, catalog)
    dataset = {"schema_version": "pg268-pikachu-parameterized-replay-dataset-v1", "source_catalog": str(CATALOG.relative_to(ROOT)), "source_catalog_sha256": catalog["catalog_sha256"], "records": abstract_rows, "counts": {"records": len(abstract_rows), "confirmed_effect_records": sum(int(row.get("lane") == "gold") for row in abstract_rows), "hard_negative_records": sum(int(row.get("lane") == "hard_negative") for row in abstract_rows)}, "contract": {"browser_discovered_parameter_context": True, "ai_reference_negative_required": True, "fresh_reset_required": True, "typed_oracle_required_for_gold": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "oracle_target_off_input": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {"protocol_id": "pg-pk-268b-pikachu-parameterized-replay-v1", "schema_version": "pg268b-pikachu-parameterized-replay-report-v1", "status": "completed_local_parameterized_get_post_replay", "runtime_image": PG214.IMAGE, "seed": SEED, "counts": counts, "source_manifest": str(MANIFEST.relative_to(ROOT)), "source_manifest_sha256": _read(MANIFEST).get("manifest_sha256", ""), "policy": {"id": "pg268_bounded_family_ucb_v1", "attempts": dict(policy_attempts), "feedback": {key: dict(value) for key, value in policy_feedback.items()}}, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "human_review_required": True}, "safety": {"loopback_only": True, "fresh_container_per_surface": True, "no_volume_or_bind_mount": True, "database_health_gate_required": True, "browser_dom_oracle_local_only": True, "sql_time_delay": False, "sql_write": False, "comments": False, "external_callback": False, "raw_payloads_human_catalog_only": True, "raw_response_bodies_stored": False, "source_hash_required": True, "evidence_hash_required": True}, "report_sha256": ""}
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    _write(TRACE, {"schema_version": "pg268b-pikachu-parameterized-replay-trace-v1", "abstract_records": abstract_rows, "raw_payloads_in_catalog_only": True, "training_promotion_allowed": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg268b-pikachu-parameterized-replay-protocol-v1", "ai_send_path": "pg268_bounded_family_ucb_v1", "browser_crawler_manifest": str(MANIFEST.relative_to(ROOT)), "independent_reference_sent": True, "matched_negative_sent": True, "fresh_reset": True, "typed_oracle": ["SQL response-shape differential", "browser DOM marker execution", "loopback Location change", "internal XML entity expansion", "bounded expression result"], "forbidden": ["time delay", "SQL write", "SQL comments", "credentials", "external callback", "public target", "file upload fixture"], "raw_payload_storage": "human-review-catalog-only", "oracle_target_off_input": True, "promotion_blocked": True, "protocol_sha256": ""}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-268B Pikachu parameterized GET/POST replay", "", f"surfaces={counts['surface_count']} GET={counts['get_count']} POST={counts['post_count']}; AI/reference/negative={counts['ai_send_count']}/{counts['reference_send_count']}/{counts['negative_send_count']}", f"confirmed local effects={counts['confirmed_positive_count']}; abstain={counts['abstain_count']}; false positives={counts['false_positive_count']}; fresh resets={counts['fresh_reset_count']}; elapsed={counts['elapsed_seconds']}s", "精确 payload 与有限 echo 只在 human-review catalog；训练集只保留抽象 Rule-IR token 与哈希。局部 effect 不等于公网漏洞声明。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": counts, "catalog": str(CATALOG.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
