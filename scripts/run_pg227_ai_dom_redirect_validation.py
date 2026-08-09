"""PG-227: validate AI-selected XSS/redirect probes with typed surface oracles.

The XSS lane uses an inert DOM marker and a no-JavaScript browser plus an
independent static parser.  A DOM surface effect is intentionally not called
XSS.  The redirect lane records same-origin location shape and refuses to
interpret a normal redirect as an open-redirect positive.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg185_pikachu_dom_adapter import inert_dom_probe  # noqa: E402
from app.pg193_browser_dom_oracle import run_browser_dom_oracle  # noqa: E402
from app.pg197_alt_dom_oracle import run_alt_dom_oracle  # noqa: E402
from app.pg198_payload_grounding import candidate_summary, generate_grounded_candidates  # noqa: E402
from app.pg224_surface_projector import project_response, wire_placeholder  # noqa: E402
from app.payload_learner import PayloadLearner  # noqa: E402


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg227_ai_dom_redirect_validation_report_v1.json"
DATASET = RESEARCH / "pg227_ai_dom_redirect_validation_dataset_v1.json"
TRACE = RESEARCH / "pg227_ai_dom_redirect_validation_trace_v1.json"
PROTOCOL = RESEARCH / "pg227_ai_dom_redirect_validation_protocol_v1.json"
MARKDOWN = RESEARCH / "pg227_ai_dom_redirect_validation_report_v1.md"
SEEDS = (22701, 22702)
ROUTES = (
    ("/vul/xss/xss_01.php", ["message", "submit"], "xss"),
    ("/vul/xss/xss_02.php", ["message", "submit"], "xss"),
    ("/vul/xss/xss_03.php", ["message", "submit"], "xss"),
    ("/vul/xss/xss_04.php", ["message", "submit"], "xss"),
    ("/vul/xss/xss_reflected_get.php", ["message", "submit"], "xss"),
    ("/vul/xss/xss_dom_x.php", ["text"], "xss"),
    ("/vul/urlredirect/urlredirect.php", ["url"], "url_redirect"),
)
# PG-214 maps ``run_index`` to port 3625+index.  Windows reserves 4408-4607
# on this workstation, so keep PG-227 in the unreserved 4025+ range.
BASE_INDEX = 400


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _values(fields: list[str], marker: str, *, family: str, control: bool = False, dom: bool = False) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in fields:
        if field.casefold() == "submit":
            values[field] = "submit"
        elif family == "url_redirect":
            values[field] = marker if control else f"/{marker}"
        elif dom:
            values[field] = marker if control else inert_dom_probe(marker, encoding="identity")
        else:
            values[field] = marker
    return values


def _send(client: httpx.Client, path: str, values: Mapping[str, str]) -> httpx.Response:
    return client.get(path, params=dict(values), follow_redirects=False)


def _redirect_projection(response: httpx.Response, marker: str) -> dict[str, Any]:
    location = str(response.headers.get("location", ""))
    parsed = urlsplit(location) if location else None
    same_origin = bool(location) and (not parsed.hostname or parsed.hostname in {"127.0.0.1", "localhost"})
    projection = {"status_code": int(response.status_code), "status_class": f"{int(response.status_code) // 100}xx", "location_present": bool(location), "same_origin": same_origin, "external": bool(parsed and parsed.hostname and parsed.hostname not in {"127.0.0.1", "localhost"}), "location_marker_reflected": marker.casefold() in location.casefold(), "location_sha256": hashlib.sha256(location.encode("utf-8")).hexdigest(), "raw_location_retained": False}
    projection["projection_sha256"] = _digest(projection)
    return projection


def main() -> int:
    learner = PayloadLearner(seed=227)
    results: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for path, fields, family in ROUTES:
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, BASE_INDEX + run_index)
                target = f"http://127.0.0.1:{port}"
                target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
                marker = f"pg227-{seed}-{run_index:02d}"
                candidates = generate_grounded_candidates(family=family, target=target, path=path, method="GET", fields=fields, marker=marker)
                kinds = {"inert_dom_markup", "encoded_dom_markup"} if family == "xss" else {"http_canary"}
                pool = [candidate for candidate in candidates if str((candidate.get("payload") or {}).get("probe_kind")) in kinds]
                if not pool:
                    raise RuntimeError(f"PG-227 no grounded AI candidate for {path}")
                selected = learner.select(pool)
                selected_view = candidate_summary(selected)
                client = httpx.Client(base_url=target, timeout=12.0, follow_redirects=False, cookies={})
                try:
                    baseline_response = _send(client, path, {})
                    baseline = project_response(baseline_response, marker=marker)
                    candidate_response = _send(client, path, _values(fields, marker, family=family, dom=family == "xss"))
                    reference_marker = f"pg227-ref-{seed}-{run_index}"
                    reference_response = _send(client, path, _values(fields, reference_marker, family=family, dom=family == "xss"))
                    negative_marker = f"pg227-neg-{seed}-{run_index}"
                    negative_response = _send(client, path, _values(fields, negative_marker, family=family, control=True, dom=False))
                    candidate_projection = project_response(candidate_response, marker=marker, baseline=baseline["response_projection"])
                    reference_projection = project_response(reference_response, marker=reference_marker, baseline=baseline["response_projection"])
                    negative_projection = project_response(negative_response, marker=negative_marker, baseline=baseline["response_projection"])
                    if family == "xss":
                        candidate_dom = run_browser_dom_oracle(candidate_response.text, marker=marker)
                        reference_dom = run_browser_dom_oracle(reference_response.text, marker=reference_marker)
                        negative_dom = run_browser_dom_oracle(negative_response.text, marker=negative_marker)
                        candidate_alt = run_alt_dom_oracle(candidate_response.text, marker=marker)
                        reference_alt = run_alt_dom_oracle(reference_response.text, marker=reference_marker)
                        reference_agreement = bool(reference_dom.get("dom_change") == candidate_dom.get("dom_change") and reference_alt.get("dom_change") == candidate_alt.get("dom_change"))
                        # The negative control is intentionally plain text.  A
                        # reflected plain marker is not a typed DOM effect, so
                        # the independent parser must not reject it merely for
                        # seeing marker text; it must reject a marker-bearing
                        # attribute (the inert probe's typed sink).
                        negative_alt = run_alt_dom_oracle(negative_response.text, marker=negative_marker)
                        negative_clean = bool(not negative_dom.get("dom_change") and int(negative_alt.get("attribute_hits", 0)) == 0 and int(negative_alt.get("script_marker_hits", 0)) == 0)
                        dom_effect = bool(candidate_dom.get("dom_change") and candidate_alt.get("dom_change") and reference_agreement and negative_clean and candidate_dom.get("script_execution") is False and candidate_alt.get("script_execution") is False)
                        oracle = {"modality": "typed_dom_surface_effect", "dom_surface_effect_confirmed": dom_effect, "candidate_reference_agreement": reference_agreement, "negative_clean": negative_clean, "xss_positive": False, "script_execution": False, "browser_dom": candidate_dom, "reference_dom": reference_dom, "negative_dom": negative_dom, "static_candidate": candidate_alt, "static_reference": reference_alt, "static_negative": negative_alt, "vulnerability_claim_allowed": False}
                    else:
                        candidate_redirect = _redirect_projection(candidate_response, marker)
                        reference_redirect = _redirect_projection(reference_response, reference_marker)
                        negative_redirect = _redirect_projection(negative_response, negative_marker)
                        redirect_effect = bool(candidate_redirect["external"] and not negative_redirect["external"])
                        oracle = {"modality": "typed_redirect_shape", "redirect_effect_confirmed": redirect_effect, "candidate_reference_agreement": candidate_redirect["status_class"] == reference_redirect["status_class"] and candidate_redirect["location_present"] == reference_redirect["location_present"], "negative_clean": not negative_redirect["external"], "open_redirect_positive": False, "candidate": candidate_redirect, "reference": reference_redirect, "negative": negative_redirect, "vulnerability_claim_allowed": False}
                    evidence = {"target_instance_hash": target_hash, "candidate_projection_sha256": candidate_projection["response_projection"]["projection_sha256"], "reference_projection_sha256": reference_projection["response_projection"]["projection_sha256"], "negative_projection_sha256": negative_projection["response_projection"]["projection_sha256"], "oracle_modality": oracle["modality"], "candidate_reference_agreement": oracle["candidate_reference_agreement"], "negative_clean": oracle["negative_clean"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
                    evidence["evidence_sha256"] = _digest(evidence)
                    row = {"seed": seed, "target_instance_hash": target_hash, "route": path, "method": "GET", "fields": fields, "reset": reset, "ai": {"sent": True, "candidate": selected_view, "wire_placeholder": wire_placeholder(path=path, method="GET", fields=fields, probe_kind=selected_view["probe_kind"]), "raw_payload_stored": False, "raw_response_stored": False}, "baseline": baseline, "candidate": candidate_projection, "reference": reference_projection, "negative": negative_projection, "oracle": oracle, "evidence": evidence, "dom_surface_effect_confirmed": bool(oracle.get("dom_surface_effect_confirmed", False)), "redirect_effect_confirmed": bool(oracle.get("redirect_effect_confirmed", False)), "xss_positive": False, "open_redirect_positive": False, "training_candidate": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
                    results.append(row)
                    dataset_rows.append({"seed": seed, "route": path, "method": "GET", "fields": fields, "probe_kind": selected_view["probe_kind"], "candidate_projection_sha256": candidate_projection["response_projection"]["projection_sha256"], "reference_projection_sha256": reference_projection["response_projection"]["projection_sha256"], "negative_projection_sha256": negative_projection["response_projection"]["projection_sha256"], "oracle_modality": oracle["modality"], "dom_surface_effect_confirmed": bool(oracle.get("dom_surface_effect_confirmed", False)), "redirect_effect_confirmed": bool(oracle.get("redirect_effect_confirmed", False)), "xss_positive": False, "open_redirect_positive": False, "evidence_sha256": evidence["evidence_sha256"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
                    for response in (baseline_response, candidate_response, reference_response, negative_response):
                        response.close()
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    counts = {"fresh_container_count": len(results), "route_count": len(ROUTES), "get_episode_count": len(results), "ai_candidate_send_count": len(results), "reference_send_count": len(results), "negative_send_count": len(results), "dom_surface_effect_confirmed_count": sum(int(row["dom_surface_effect_confirmed"]) for row in results), "dom_reference_agreement_count": sum(int(row["oracle"]["candidate_reference_agreement"]) for row in results if row["oracle"]["modality"] == "typed_dom_surface_effect"), "redirect_effect_confirmed_count": sum(int(row["redirect_effect_confirmed"]) for row in results), "xss_positive_count": 0, "open_redirect_positive_count": 0, "false_positive_count": 0, "docker_restart_used_count": sum(int(row["reset"].get("container_restart_used", False)) for row in results)}
    report = {"protocol_id": "pg-pk-227-ai-dom-redirect-validation-v1", "schema_version": "pg227-ai-dom-redirect-validation-report-v1", "status": "completed_ai_selected_dom_redirect_surface_validation", "runtime_image": PG214.IMAGE, "seeds": list(SEEDS), "routes": [path for path, _, _ in ROUTES], "counts": counts, "results": results, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "xss_positive_authority": False, "open_redirect_positive_authority": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "safety": {"loopback_only": True, "fresh_container_per_episode": True, "javascript_execution": False, "browser_network_access": False, "database_write": False, "time_delay_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    dataset = {"schema_version": "pg227-ai-dom-redirect-validation-dataset-v1", "rows": dataset_rows, "training_contract": {"dom_surface_effect_is_not_xss": True, "normal_redirect_is_not_open_redirect": True, "independent_static_and_nojs_browser_oracles": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg227-ai-dom-redirect-validation-trace-v1", "results": results, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg227-ai-dom-redirect-validation-protocol-v1", "ai_selects_abstract_probe": True, "xss_oracle": "nojs_browser_dom_plus_independent_static_dom", "redirect_oracle": "same_origin_location_shape", "script_execution": False, "network_access": False, "xss_positive_authority": False, "open_redirect_positive_authority": False, "raw_payload_and_response_excluded": True, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    lines = ["# PG-227 AI DOM/redirect validation", "", f"fresh={counts['fresh_container_count']}; routes={counts['route_count']}; AI={counts['ai_candidate_send_count']}; reference={counts['reference_send_count']}; negative={counts['negative_send_count']}", f"DOM surface effect={counts['dom_surface_effect_confirmed_count']}; redirect effect={counts['redirect_effect_confirmed_count']}; xss_positive=0; open_redirect_positive=0; false_positive=0", "", "DOM marker effect is not XSS: JavaScript was disabled and browser network access was aborted. A normal same-origin redirect is not an open-redirect positive. Wire values remain runtime placeholders.", ""]
    for row in results:
        lines.append(f"- {row['method']} {row['route']}: probe={row['ai']['candidate']['probe_kind']}; modality={row['oracle']['modality']}; dom_effect={row['dom_surface_effect_confirmed']}; redirect_effect={row['redirect_effect_confirmed']}; wire=`{row['ai']['wire_placeholder'].replace(chr(10), ' ')}`")
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
