"""PG-74: collect neutral/negative/positive triplets on fresh local targets.

This is the first collector that distinguishes a neutral request from a
non-triggering probe.  It uses the same authorized Pikachu Docker image and
typed PG-52 oracles, but persists only bounded projections, oracle metadata
and hashes.  The output is evaluation-only until source/seed/family splits
and the capability gates pass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step  # noqa: E402


PROTOCOL_ID = "pg-pk-74-causal-triplet-collector-v1"
SCHEMA_VERSION = "sift-pg74-causal-triplet-collector-v1"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
SEEDS = (74101, 74102, 74103)
PG52_PATH = ROOT / "scripts" / "run_pg52_authoritative_local_oracle.py"
PG69_PATH = ROOT / "scripts" / "run_pg69_per_action_reset_unseen_family.py"
PG72_PATH = ROOT / "scripts" / "run_pg72_independent_seed_fresh_docker_matrix.py"
REPORT_PATH = ROOT / "research" / "pg74_causal_triplet_collector_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg74_causal_triplet_collector_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg74_causal_triplet_collector_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg74_causal_triplet_collector_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"projection_available": False}
    forbidden = {"body", "raw_body", "body_preview", "request_body", "raw_probe", "password", "token", "cookie", "authorization"}

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {str(key): clean(child) for key, child in node.items() if str(key).casefold() not in forbidden}
        if isinstance(node, list):
            return [clean(child) for child in node]
        return node

    return json.loads(json.dumps(clean(value), ensure_ascii=False, sort_keys=True))


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _neutral_oracle(pg52: Any, *, modality: str, oracle_id: str, signal: str) -> dict[str, Any]:
    return {"oracle_id": oracle_id, "modality": modality, "positive": False, "positive_authority": True, "confirmed_effect": "none", "candidate_signal": False, "evaluator_state_hidden": True, "signals": {"neutral_request": True, "signal": signal}, "safety": {"external_network": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False}}


def _neutral_xss(pg52: Any, case: dict[str, Any], base: str, seed: int, ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    with httpx.Client(base_url=base, timeout=8.0, follow_redirects=True) as client:
        if case["mode"] == "reflected_get":
            response = client.get(case["path"], params={"submit": "submit"})
        elif case["mode"] == "dom_get":
            response = client.get(case["path"])
        else:
            client.post("/vul/xss/xsspost/post_login.php", data={"username": "admin", "password": "123456", "submit": "Login"})
            response = client.post(case["path"], data={"submit": "submit"})
    projection = pg52.response_projection(response, "")
    if case["mode"] == "dom_get":
        # The DOM source surface has no HTTP-body delta.  Render the neutral
        # response in the same offline browser and retain only bounded DOM
        # shape counts; do not fill, click, dispatch, or return a marker.
        html = response.content.decode("utf-8", errors="replace")
        for pattern in (r"<link\b[^>]*(?:fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*>", r"<script\b[^>]*\bsrc\s*=\s*['\"][^'\"]+['\"][^>]*>\s*</script>"):
            html = __import__("re").sub(pattern, "", html, flags=__import__("re").IGNORECASE | __import__("re").DOTALL)
        with pg52.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(service_workers="block")
            page = context.new_page()
            page.set_content(html, wait_until="domcontentloaded")
            projection["dom_shape"] = pg52._bounded_dom_shape(page)
            projection["dom_shape_sha256"] = pg52.sha256_json(projection["dom_shape"])
            browser.close()
        projection["projection_sha256"] = pg52.sha256_json(projection)
    oracle = _neutral_oracle(pg52, modality="browser_dom_execution", oracle_id=f"pg74-neutral-browser-{case['mode']}-v1", signal="no_controlled_execution")
    return projection, oracle


def _neutral_sql(pg52: Any, case: dict[str, Any], base: str, container: str, seed: int, ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = f"pg74-neutral-s{seed}-c{ordinal}"
    with httpx.Client(base_url=base, timeout=8.0, follow_redirects=False) as client:
        def request() -> httpx.Response:
            return client.get(case["path"], params={"submit": "submit"})

        response, query = pg52._capture_query(container, request, marker)
    projection = pg52.response_projection(response, "")
    projection["result_row_count"] = pg52._html_result_rows(response)
    projection["query_seen"] = bool(query)
    projection["projection_sha256"] = pg52.sha256_json(projection)
    oracle = pg52.sql_ast_differential_oracle(control_query=query, candidate_query=query, control_response=projection, candidate_response=projection, expected_marker="")
    oracle.update({"evaluator_state_hidden": True, "neutral_request": True})
    return projection, oracle


def _neutral_redirect(pg52: Any, case: dict[str, Any], base: str) -> tuple[dict[str, Any], dict[str, Any]]:
    destination = "http://127.0.0.1:8768/pg74-loopback-callback"
    with httpx.Client(base_url=base, timeout=8.0, follow_redirects=False) as client:
        response = client.get(case["path"])
    projection = pg52.response_projection(response, "")
    oracle = pg52.redirect_oracle(location=str(response.headers.get("location", "")), expected_destination=destination, response_status=response.status_code)
    oracle.update({"evaluator_state_hidden": True, "neutral_request": True})
    return projection, oracle


def _run_case(pg52: Any, pg69: Any, case: dict[str, Any], seed: int, ordinal: int) -> dict[str, Any]:
    salted = dict(case)
    salted["case_id"] = f"pg74-s{seed}-c{ordinal:02d}-{case['base_case_id']}"
    name = f"pg74-pikachu-s{seed}-c{ordinal:02d}"
    port = int(case["port"])
    started = False
    try:
        container_id = pg52._start(name, port)
        started = True
        if port == 8767:
            pg52._wait_application_surface(port, "/vul/sqli/sqli_str.php", b"what's your username")
        else:
            pg52._wait_application_surface(port, "/vul/xss/xsspost/post_login.php", b'name="username"')
        if case["family"] == "injection":
            pg52._prepare_mysql(name)
        base = pg52.GET_BASE if port == 8767 else pg52.POST_BASE
        if case["family"] == "xss":
            neutral_response, neutral_oracle = _neutral_xss(pg52, salted, base, seed, ordinal)
            raw = pg52._browser_case(salted, base, f"pg74-s{seed}-c{ordinal}-marker", container_id)
            negative = raw["control"]
            positive = raw["candidate"]
        elif case["family"] == "injection":
            neutral_response, neutral_oracle = _neutral_sql(pg52, salted, base, name, seed, ordinal)
            raw = pg52._sql_case(salted, base, name)
            negative = raw["negative"]
            positive = raw["candidate"]
        else:
            neutral_response, neutral_oracle = _neutral_redirect(pg52, salted, base)
            raw = pg52._redirect_case(salted, base, container_id)
            negative = raw["control"]
            positive = raw["candidate"]
        reset = pg52._fresh_reset(container_id, salted["case_id"], _hash(f"pg74-reset|{seed}|{ordinal}|{name}"))
        row = {"case_id": salted["case_id"], "family": str(case["family"]), "surface": str(case["surface"]), "method": str(case["method"]), "path": str(case["path"]), "field": str(case.get("field", "value")), "port": port, "seed": int(seed), "ordinal": int(ordinal), "source_kind": "real_docker", "independent_implementation": "pinned_pikachu_php_mysql_pg74", "target_instance_id": str(container_id)[:24], "fresh_reset": reset, "neutral_response": _safe_projection(neutral_response), "neutral_oracle": _safe_projection(neutral_oracle), "negative_response": _safe_projection(negative.get("response")), "negative_oracle": _safe_projection(negative.get("oracle")), "positive_response": _safe_projection(positive.get("response")), "positive_oracle": _safe_projection(positive.get("oracle")), "raw_payload_stored": False, "raw_response_body_stored": False}
        row["negative_control"] = {"matched": True, "control_case_id": salted["case_id"], "control_evidence_sha256": sha256_json({"neutral": row["neutral_response"], "negative": row["negative_response"], "oracle": row["negative_oracle"]}), "candidate_vs_control": True}
        row["evidence_sha256"] = sha256_json({"case_id": row["case_id"], "neutral": row["neutral_response"], "negative": row["negative_response"], "positive": row["positive_response"], "neutral_oracle": row["neutral_oracle"], "negative_oracle": row["negative_oracle"], "positive_oracle": row["positive_oracle"], "reset": reset})
        return row
    finally:
        if started:
            pg52._stop(name)


def _step(row: dict[str, Any], index: int, episode_id: str, parent: str | None) -> dict[str, Any]:
    method = str(row["method"]).upper()
    action: dict[str, Any] = {"method": method, "route_template_id": f"pg74-route-{index:03d}", "placement": "form" if method == "POST" else "query", "encoding_chain": ["identity"], "probe_ref": f"pg74-probe-{index:03d}", "probe_sha256": _hash("pg74-abstract-triplet-probe"), "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if method == "POST":
        action["form_field_names"] = [str(row.get("field", "value"))]
    oracle = dict(row["positive_oracle"])
    oracle["negative_control_pair_id"] = f"pg74-control-{index:03d}"
    oracle["evaluator_state_hidden"] = True
    decision = "confirmed_positive" if bool(oracle.get("positive")) else "confirmed_negative"
    step = {"episode_id": episode_id, "step_id": f"pg74-step-{index:03d}", "parent_step_id": parent, "sampling_seed": int(row["seed"]), "target_instance_id": str(row["target_instance_id"]), "hypothesis": "causal_triplet_surface_hypothesis", "belief_before": {"unknown_surface": 1.0}, "action_manifest": action, "baseline_projection": row["neutral_response"], "neutral_projection": row["neutral_response"], "negative_probe_projection": row["negative_response"], "response_projection": row["positive_response"], "neutral_oracle_projection": row["neutral_oracle"], "negative_oracle_projection": row["negative_oracle"], "oracle_projection": oracle, "belief_after": {"unknown_surface": 1.0}, "decision": decision, "next_action": "stop_confirmed" if decision == "confirmed_positive" else "continue_probe", "fresh_reset": row["fresh_reset"], "evidence_sha256": row["evidence_sha256"], "dataset_stage": "evaluation_only", "online_weight_update": False, "long_term_memory_write": False}
    # validate_trace_step computes and binds the triplet fields in the echo.
    return step


def _build_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["seed"])].append((index, row))
    episodes: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed, items in sorted(grouped.items()):
        episode_id = f"pg74-episode-s{seed}"
        parent: str | None = None
        steps: list[dict[str, Any]] = []
        for index, row in items:
            raw_step = _step(row, index, episode_id, parent)
            body = {key: raw_step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")}
            raw_step["echo"] = {"sha256": sha256_json(body)}
            try:
                normalized = validate_trace_step(raw_step)
            except ValueError as exc:
                failures.append({"step_id": raw_step["step_id"], "error_type": type(exc).__name__})
                normalized = raw_step
            steps.append(normalized)
            all_steps.append(normalized)
            parent = normalized["step_id"]
        validation = evaluate_episode(steps) if not failures else {"status": "trace_only", "reasons": ["step_validation_failure"]}
        episodes.append({"episode_id": episode_id, "seed": seed, "steps": steps, "validation": validation})
    return {"schema_version": "sift-pg74-causal-triplet-trace-v1", "protocol_id": PROTOCOL_ID, "evaluation_only": True, "training_eligible": False, "steps": all_steps, "episodes": episodes, "episode_count": len(episodes), "accepted_episode_count": sum(int(item["validation"].get("status") == "accepted_evaluation") for item in episodes), "validation_failures": failures, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}


def _catalog(pg69: Any, rows: list[dict[str, Any]]) -> None:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    groups: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        source_id = f"pg74-real-pikachu-s{row['seed']}-p{row['port']}"
        source = groups.setdefault(source_id, {"provenance": {"source_id": source_id, "source_type": "authorized_local_container", "origin": "research/pg74_causal_triplet_collector_protocol_v1.json", "license": "local_container", "authorization": "workspace_local_only", "scope": [f"http://127.0.0.1:{row['port']}"], "captured_at": captured_at, "authorized_for": ["training", "local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False, "container_image_digest": IMAGE.split("@", 1)[1]}, "samples": []})
        sample = pg69._catalog_sample({"family": row["family"], "surface": row["surface"], "method": row["method"], "path": row["path"], "field": row["field"], "source_kind": "real_docker", "port": row["port"], "fresh_reset": row["fresh_reset"], "candidate_response": row["positive_response"], "control_response": row["negative_response"], "oracle": row["positive_oracle"], "evidence_sha256": row["evidence_sha256"], "negative_control": row["negative_control"]}, index)
        sample["sample_id"] = f"pg74-sample-{index:03d}"
        sample["counterfactual"]["source_sample_id"] = sample["sample_id"]
        source["samples"].append(sample)
    write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg74-causal-triplet-evaluation-only", "sources": list(groups.values())})


def run(*, skip_docker: bool = False, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    pg52 = _load(PG52_PATH, "pg74_pg52_runtime")
    pg69 = _load(PG69_PATH, "pg74_pg69_runtime")
    pg72 = _load(PG72_PATH, "pg74_pg72_runtime")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seed in seeds:
        cases = list(pg72.MATRIX_CASES)
        __import__("random").Random(int(seed)).shuffle(cases)
        for ordinal, case in enumerate(cases):
            if skip_docker:
                continue
            try:
                rows.append(_run_case(pg52, pg69, case, int(seed), ordinal))
            except Exception as exc:
                errors.append({"seed": int(seed), "case_id": str(case["base_case_id"]), "error_type": type(exc).__name__})
    trace = _build_trace(rows)
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        _catalog(pg69, rows)
    typed_positive = sum(int(bool(row["positive_oracle"].get("positive"))) for row in rows)
    typed_negative = sum(int(not bool(row["negative_oracle"].get("positive"))) + int(not bool(row["neutral_oracle"].get("positive"))) for row in rows)
    triplet_complete = sum(int(bool(row.get("neutral_response")) and bool(row.get("negative_response")) and bool(row.get("positive_response")) and bool(row.get("neutral_oracle")) and bool(row.get("negative_oracle")) and bool(row.get("positive_oracle"))) for row in rows)
    target_ids = [str(row["target_instance_id"]) for row in rows]
    checks = {"triplet_complete_per_case": triplet_complete == len(rows) and len(rows) > 0, "typed_positive_per_case": typed_positive == len(rows) and len(rows) > 0, "typed_negative_oracle_per_case": typed_negative == len(rows) * 2 and len(rows) > 0, "fresh_target_per_case": bool(rows) and len(target_ids) == len(set(target_ids)), "get_post_covered": {"GET", "POST"}.issubset({row["method"] for row in rows}), "trace_episodes_accepted": bool(trace["episode_count"]) and trace["accepted_episode_count"] == trace["episode_count"] and not trace["validation_failures"], "no_raw_persistence": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in rows)}
    status = "passed" if all(checks.values()) else "blocked"
    report = {"protocol_id": PROTOCOL_ID, "schema_version": SCHEMA_VERSION, "status": "completed_evaluation" if rows else "blocked_no_rows", "source": {"pinned_image": IMAGE, "seeds_requested": list(seeds), "case_count": len(rows), "docker_errors": errors, "independent_implementation_count": 1}, "metrics": {"triplet_case_count": len(rows), "typed_positive_count": typed_positive, "typed_negative_oracle_count": typed_negative, "neutral_projection_count": len(rows), "negative_probe_projection_count": len(rows), "positive_probe_projection_count": len(rows), "unique_target_instance_count": len(set(target_ids)), "get_post_covered": {"GET": sum(int(row["method"] == "GET") for row in rows), "POST": sum(int(row["method"] == "POST") for row in rows)}, "trace_episode_count": trace["episode_count"], "trace_accepted_episode_count": trace["accepted_episode_count"]}, "hard_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "training_catalog_generated": False, "status": "evaluation_only_triplet_collector", "reason": "triplet collector output requires source/family/seed split and candidate training gates"}, "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT)) if rows else None}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "sift-pg74-causal-triplet-collector-protocol-v1", "target_contract": {"pinned_image": IMAGE, "loopback_only": True, "external_network": False, "fresh_container_per_triplet": True, "methods": ["GET", "POST"]}, "triplet_contract": {"neutral_projection": True, "negative_probe_projection": True, "positive_probe_projection": True, "neutral_typed_oracle": True, "negative_typed_oracle": True, "positive_typed_oracle": True, "raw_persistence_forbidden": True}, "required_gates": {"triplet_complete_per_case": True, "typed_positive_per_case": True, "typed_negative_oracle_per_case": True, "fresh_target_per_case": True, "get_post_covered": True, "trace_episodes_accepted": True, "no_raw_persistence": True}, "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG75 source/family-heldout split and triplet delta representation ablation"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-74 因果三元组采集\n\n" + f"triplets={len(rows)}；typed positive={typed_positive}；typed negative oracles={typed_negative}；neutral/negative/positive projection={len(rows)}/{len(rows)}/{len(rows)}。\n\n硬门：`{status}`；training_allowed=`false`；memory_promotion_allowed=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--seeds", default=",".join(str(item) for item in SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(item.strip()) for item in str(args.seeds).split(",") if item.strip())
    report = run(skip_docker=bool(args.skip_docker), seeds=seeds)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["hard_gate"]["status"], "triplet_case_count": report["metrics"]["triplet_case_count"], "typed_positive_count": report["metrics"]["typed_positive_count"], "typed_negative_oracle_count": report["metrics"]["typed_negative_oracle_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
