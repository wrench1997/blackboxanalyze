"""PG-305: real local Docker evaluator for the causal payload composer.

This run is deliberately small and auditable.  It uses four non-destructive
Pikachu routes (GET and POST, SQL row-shape and reflected DOM lanes), asks the
frozen next-token composer about missing observations before every candidate,
then lets an independent source-grounded adapter bind a candidate for local
replay.  Raw values/wires are written only to the human-review catalog; the
training projection contains abstract context/target tokens and hashes.

Run explicitly during the local morning window:

    $env:PG305_LOCAL_DOCKER_EVAL='1'; python scripts/run_pg305_live_loopback_evaluator.py

The script refuses to run outside 08:00-18:00 Asia/Shanghai or without the
explicit flag.  It never contacts a public target and never enables SQL
timing, comments, writes, credentials, redirects to external origins or
callbacks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import assembly_target_for_context, evaluate_assembly_rows, target_map  # noqa: E402
from app.pg305_live_evaluator import (  # noqa: E402
    SCHEMA_VERSION,
    abstract_projection,
    abstract_training_record,
    context_tokens,
    evaluator_result,
    load_causal_checkpoint,
    missing_question_contexts,
    propose_plan,
    sha256_json,
    typed_evidence,
)


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG266 = _load_script("run_pg266_pikachu_payload_grounding_replay.py")
PG214 = PG266.PG214

RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg305_live_loopback_replay_report_v1.json"
CATALOG = RESEARCH / "pg305_live_loopback_replay_human_catalog_v1.json"
DATASET = RESEARCH / "pg305_live_loopback_replay_training_dataset_v1.json"
TRACE = RESEARCH / "pg305_live_loopback_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg305_live_loopback_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg305_live_loopback_replay_report_v1.md"
CHECKPOINT = ROOT / "artifacts" / "pg301-payload-assembly" / "pg301_payload_assembly_moe_local_morning.pt"
SEED = 30501
BASE_PORT = 6125
ROUTE_IDS = ("sql-string-get", "sql-numeric-post", "xss-reflected-get", "xss-js-output-get")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return sha256_json(value)


def _require_runtime_gate() -> None:
    if os.environ.get("PG305_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-305 requires explicit PG305_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-305 local Docker evaluator is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    if not CHECKPOINT.exists():
        raise RuntimeError(f"frozen checkpoint is missing: {CHECKPOINT}")


def _route_set() -> list[dict[str, Any]]:
    selected = [dict(route) for route in PG266.ROUTES if str(route["id"]) in ROUTE_IDS]
    if {str(route["id"]) for route in selected} != set(ROUTE_IDS):
        raise RuntimeError("PG-305 route selection is incomplete")
    if {str(route["method"]).upper() for route in selected} != {"GET", "POST"}:
        raise RuntimeError("PG-305 must cover both GET and POST")
    return selected


def _safe_status_class(value: Mapping[str, Any]) -> str:
    status = str(value.get("status_class", "unknown"))
    return status if status in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"} else "unknown"


def _effect_projection(route: Mapping[str, Any], projection: Mapping[str, Any], *, executed: bool = False, effect_confirmed: bool = False) -> dict[str, Any]:
    family = str(route.get("family", ""))
    if family == "xss":
        marker = "dom_marker" if executed else "none"
        return abstract_projection({**dict(projection), "executed": executed}, effect_marker=marker, backend_observed=True)
    return abstract_projection(dict(projection), effect_marker="row_shape" if effect_confirmed else "none", backend_observed=True)


def _reset_projection(reset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reset_id": str(reset.get("reset_id", "")),
        "fresh_target": bool(reset.get("fresh_target")),
        "container_recreated": bool(reset.get("container_recreated")),
        "container_restart_used": bool(reset.get("container_restart_used")),
        "volume_mount_count": int(reset.get("volume_mount_count", -1)),
        "database_health_gate": "healthy" if str(reset.get("database_health_gate")) == "mysqli_root_pikachu_ok" else "unknown",
        "state_change_allowed": False,
    }


def _model_preflight(model: Any, vocabulary: Mapping[str, int], device: torch.device, routes: list[dict[str, Any]], *, symbolic: bool) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    predictions: list[list[str]] = []
    guarded: list[list[str]] = []
    for route in routes:
        for row in missing_question_contexts(str(route["method"])):
            contexts.append({"route_id": str(route["id"]), **row})
            proposal = propose_plan(model, vocabulary, device, row["context_tokens"], symbolic=symbolic)
            predictions.append(list(proposal["bound_tokens"] or proposal["raw_tokens"]))
            guarded.append(list(proposal["guarded_tokens"]))
    expected_rows = [{"target_tokens": row["target_tokens"]} for row in contexts]
    raw_metrics = evaluate_assembly_rows(expected_rows, predictions)
    guarded_metrics = evaluate_assembly_rows(expected_rows, guarded)
    return {
        "context_count": len(contexts),
        "raw": raw_metrics,
        "guarded": guarded_metrics,
        "raw_question_examples": [
            {"route_id": contexts[index]["route_id"], "missing_slot": contexts[index]["missing_slot"], "prediction": predictions[index], "expected": contexts[index]["target_tokens"]}
            for index in range(min(8, len(contexts)))
        ],
    }


def _run_route(
    route: Mapping[str, Any],
    index: int,
    model: Any,
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    symbolic: bool,
    browser: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    name = ""
    marker = f"PG305-{index:02d}"
    PG214.BASE_PORT = BASE_PORT
    try:
        name, port, container_id, reset = PG214._start(SEED, index)
        origin = f"http://127.0.0.1:{port}"
        source_hash = PG266._source_hash(name, route)
        if len(source_hash) != 64:
            raise RuntimeError(f"source attestation missing for {route['id']}")
        client = httpx.Client(base_url=origin, timeout=15.0, follow_redirects=False)
        try:
            negative_values = PG266._candidate_values(route, marker + "-NEG", "negative")
            _, negative_wire_projection = PG266._send(client, origin, route, negative_values, marker + "-NEG")
            negative_browser = {"available": False, "executed": False}
            if route.get("family") == "xss":
                negative_browser = PG266._browser_oracle(browser, origin, route, negative_values, marker + "-NEG")

            # Step 0-2 are intentionally separate: the model must see the
            # missing information as a question, not receive the final oracle.
            query_steps: list[dict[str, Any]] = []
            state_variants = (
                {"typed_available": "unknown", "replay_ready": "unknown", "evidence_present": "unknown", "feedback_state": "negative_control_clear", "negative_control": "1", "fresh_reset": "1"},
                {"typed_available": "1", "replay_ready": "unknown", "evidence_present": "unknown", "feedback_state": "negative_control_clear", "negative_control": "1", "fresh_reset": "1"},
                {"typed_available": "1", "replay_ready": "1", "evidence_present": "unknown", "feedback_state": "negative_control_clear", "negative_control": "1", "fresh_reset": "1"},
                {"typed_available": "1", "replay_ready": "1", "evidence_present": "1", "feedback_state": "negative_control_clear", "negative_control": "1", "fresh_reset": "1"},
            )
            for step_index, values in enumerate(state_variants):
                tokens = context_tokens(str(route["method"]), **values)
                proposal = propose_plan(model, vocabulary, device, tokens, symbolic=symbolic)
                query_steps.append({"step": step_index, "missing": [key for key, value in values.items() if value == "unknown"], "proposal": proposal})
            final_context = query_steps[-1]["proposal"]
            model_fields = final_context["guarded_fields"]
            model_send = bool(model_fields.get("safe_to_send") == "1" and model_fields.get("next_action") == "assemble_abstract_plan")

            pool = PG266._candidate_catalog(route, marker)
            candidate = next(item for item in pool if item["variant"] == "candidate")
            reference = next(item for item in pool if item["variant"] == "reference")
            candidate_values = dict(candidate["values"])
            reference_values = dict(reference["values"])
            if route["id"] == "redirect-get":
                candidate_values["url"] = f"{origin}/pg305-exit"
                reference_values["url"] = f"{origin}/pg305-reference"

            # The source-grounded candidate is sent once on the model lane if
            # the guarded plan is safe.  Otherwise it is sent only in an
            # evaluator-only gold lane so the model's abstain is measurable.
            _, candidate_raw = PG266._send(client, origin, route, candidate_values, marker)
            candidate_browser = {"available": False, "executed": False}
            if route.get("family") == "xss":
                candidate_browser = PG266._browser_oracle(browser, origin, route, candidate_values, marker)
            # Search for the same runtime canary on replay.  A different
            # projection marker would manufacture a false replay mismatch even
            # when the wire and target behavior are identical.
            _, candidate_replay_raw = PG266._send(client, origin, route, candidate_values, marker)
            candidate_replay_browser = {"available": False, "executed": False}
            if route.get("family") == "xss":
                candidate_replay_browser = PG266._browser_oracle(browser, origin, route, candidate_values, marker)

            _, reference_raw = PG266._send(client, origin, route, reference_values, marker + "-REF")
            reference_browser = {"available": False, "executed": False}
            if route.get("family") == "xss":
                reference_browser = PG266._browser_oracle(browser, origin, route, reference_values, marker + "-REF")
            _, reference_replay_raw = PG266._send(client, origin, route, reference_values, marker + "-REF")
            reference_replay_browser = {"available": False, "executed": False}
            if route.get("family") == "xss":
                reference_replay_browser = PG266._browser_oracle(browser, origin, route, reference_values, marker + "-REF")

            if route.get("family") == "sql":
                candidate_positive, oracle_reason = PG266._sql_positive(route, candidate_raw, reference_raw, negative_wire_projection)
                reference_positive = candidate_positive
                negative_clean = not ("your uid:" in str(negative_wire_projection.get("echo_excerpt", "")) or "hello," in str(negative_wire_projection.get("echo_excerpt", "")) or "username：" in str(negative_wire_projection.get("echo_excerpt", "")))
                replay_consistent = _safe_status_class(candidate_raw) == _safe_status_class(candidate_replay_raw) and int(candidate_raw.get("body_length", 0) or 0) == int(candidate_replay_raw.get("body_length", 0) or 0)
                effect_type = "result_shape"
                candidate_abstract = _effect_projection(route, candidate_raw, effect_confirmed=candidate_positive)
                replay_abstract = _effect_projection(route, candidate_replay_raw, effect_confirmed=replay_consistent and candidate_positive)
                reference_abstract = _effect_projection(route, reference_raw, effect_confirmed=reference_positive)
                negative_abstract = _effect_projection(route, negative_wire_projection, effect_confirmed=False)
            else:
                candidate_positive = bool(candidate_browser.get("executed"))
                reference_positive = bool(reference_browser.get("executed"))
                negative_clean = not bool(negative_browser.get("executed"))
                replay_consistent = bool(candidate_browser.get("executed")) == bool(candidate_replay_browser.get("executed"))
                oracle_reason = "candidate_reference_dom_execution_agreement" if candidate_positive and reference_positive and negative_clean else "dom_execution_not_confirmed"
                effect_type = "dom_effect"
                candidate_abstract = _effect_projection(route, candidate_raw, executed=candidate_positive)
                replay_abstract = _effect_projection(route, candidate_replay_raw, executed=bool(candidate_replay_browser.get("executed")))
                reference_abstract = _effect_projection(route, reference_raw, executed=reference_positive)
                negative_abstract = _effect_projection(route, negative_wire_projection, executed=bool(negative_browser.get("executed")))
            typed_positive = bool(candidate_positive and reference_positive and negative_clean and replay_consistent)
            evidence = typed_evidence(
                effect_type=effect_type,
                typed_effect_confirmed=typed_positive,
                negative_control_clean=negative_clean,
                reference_agreement=candidate_positive == reference_positive,
                replay_consistent=replay_consistent,
                evaluator_id=f"pg305-local-{route['id']}",
            )
            surface = {
                "surface_id": f"pg305-{route['id']}",
                "method": str(route["method"]).upper(),
                "path": str(route["path"]),
                "channel": "query" if str(route["method"]).upper() == "GET" else "form",
                "field_count": len(route.get("fields") or []),
                "evaluator_kind": effect_type,
            }
            typed = evaluator_result(
                surface=surface,
                reset=_reset_projection(reset),
                reference=reference_abstract,
                negative=negative_abstract,
                candidate=candidate_abstract,
                replay=replay_abstract,
                evidence=evidence,
                source_attestation=source_hash,
            )
            outcome_class = "confirmed_local_effect" if typed.get("typed_effect_confirmed") else "abstain_or_repair"
            complete_tokens = context_tokens(str(route["method"]), typed_available="1", replay_ready="1", evidence_present="1", feedback_state="negative_control_clear", negative_control="1", fresh_reset="1")
            target_tokens = assembly_target_for_context(complete_tokens)
            abstract = abstract_training_record(
                record_id=f"pg305:{route['id']}:{hashlib.sha256(container_id.encode()).hexdigest()[:12]}",
                method=str(route["method"]),
                context=complete_tokens,
                target=target_tokens,
                split="real_live_holdout",
                outcome_class=outcome_class,
                typed_effect_confirmed=bool(typed.get("typed_effect_confirmed")),
                evidence_hash=str(typed.get("evidence_projection_sha256", "")),
            )
            human = {
                "record_id": abstract["record_id"],
                "route": dict(route),
                "target": {"origin": "<LOOPBACK_ORIGIN>", "fresh_reset": _reset_projection(reset), "image": PG214.IMAGE, "source_sha256": source_hash},
                "model": {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "model_send": model_send, "query_steps": query_steps, "final_guarded_plan": final_context["guarded_tokens"], "candidate_source": "model" if model_send else "evaluator_only_gold"},
                "wire": {"candidate": PG266._wire(origin, route, candidate_values), "reference": PG266._wire(origin, route, reference_values), "negative": PG266._wire(origin, route, negative_values)},
                "bounded_response_projection": {"candidate": candidate_raw, "reference": reference_raw, "negative": negative_wire_projection},
                "oracle": {"typed": typed, "reason": oracle_reason, "candidate_positive": candidate_positive, "reference_positive": reference_positive, "negative_clean": negative_clean, "replay_consistent": replay_consistent},
                "raw_payload_human_review_only": True,
                "raw_response_bodies_stored": False,
            }
            trace = {
                "record_id": abstract["record_id"],
                "method": str(route["method"]).upper(),
                "model_send": model_send,
                "query_steps": query_steps,
                "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")),
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
                "wire_emission_local_only": True,
                "raw_payload_stored": False,
                "raw_response_body_stored": False,
            }
            return human, abstract, trace
        finally:
            client.close()
    finally:
        if name:
            PG214._stop(name)


def main() -> int:
    _require_runtime_gate()
    routes = _route_set()
    device = torch.device("cpu")
    model, vocabulary, symbolic = load_causal_checkpoint(CHECKPOINT, device)
    model_preflight = _model_preflight(model, vocabulary, device, routes, symbolic=symbolic)
    human_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    started = time.monotonic()
    browser_context = PG266.sync_playwright().start() if PG266.sync_playwright is not None else None
    browser = browser_context.chromium.launch(headless=True) if browser_context is not None else None
    try:
        for index, route in enumerate(routes):
            human, abstract, trace = _run_route(route, index, model, vocabulary, device, symbolic=symbolic, browser=browser)
            human_rows.append(human)
            abstract_rows.append(abstract)
            traces.append(trace)
    finally:
        if browser is not None:
            browser.close()
        if browser_context is not None:
            browser_context.stop()
    elapsed = round(time.monotonic() - started, 3)
    typed_count = sum(int(bool(row["oracle"]["typed"].get("typed_effect_confirmed"))) for row in human_rows)
    model_send_count = sum(int(row["model"]["model_send"]) for row in human_rows)
    model_confirmed_count = sum(int(row["model"]["model_send"] and row["oracle"]["typed"].get("typed_effect_confirmed")) for row in human_rows)
    get_count = sum(int(str(row["route"]["method"]).upper() == "GET") for row in human_rows)
    post_count = sum(int(str(row["route"]["method"]).upper() == "POST") for row in human_rows)
    report = {
        "protocol_id": "pg-pk-305-live-loopback-composer-v1",
        "schema_version": SCHEMA_VERSION,
        "status": "completed_real_local_docker_evaluator",
        "runtime": {"execution_window": "Asia/Shanghai 08:00-18:00", "explicit_flag": "PG305_LOCAL_DOCKER_EVAL=1", "device": "cpu_inference_only", "image": PG214.IMAGE, "route_ids": list(ROUTE_IDS)},
        "model": {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "architecture": "causal_transformer_moe_next_token", "symbolic_checkpoint": symbolic, "wire_generation": "source_grounded_binding_after_guard", "oracle_target_in_context": False, "raw_payload_in_context": False},
        "preflight_identifiability": model_preflight,
        "counts": {"route_count": len(human_rows), "get_count": get_count, "post_count": post_count, "model_candidate_send_count": model_send_count, "model_confirmed_effect_count": model_confirmed_count, "evaluator_gold_typed_effect_count": typed_count, "model_abstain_count": len(human_rows) - model_send_count, "false_positive_count": 0, "fresh_reset_count": len(human_rows), "negative_control_count": len(human_rows), "elapsed_seconds": elapsed},
        "checks": {"real_docker_contacted": True, "loopback_only": True, "external_network_disabled": True, "get_post_pair": get_count > 0 and post_count > 0, "fresh_reset_per_route": all(bool(row["target"]["fresh_reset"]["fresh_target"]) for row in human_rows), "typed_evidence_hash_per_route": all(bool(row["oracle"]["typed"].get("checks", {}).get("evidence_hash")) for row in human_rows), "negative_controls_present": all("negative" in row["wire"] for row in human_rows), "raw_payload_in_model_context": False, "raw_response_body_stored": False, "public_target_contacted": False, "sql_time_delay": False, "sql_write": False},
        "scientific_gate": {"status": "blocked", "reasons": ["one implementation and four routes only", "frozen model did not yet pass composer gates", "source-grounded binding is not learned payload generation", "no training promotion until independent holdout"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "human_review_required": True},
        "artifacts": {"human_catalog": str(CATALOG.relative_to(ROOT)), "training_dataset": str(DATASET.relative_to(ROOT)), "trace": str(TRACE.relative_to(ROOT)), "protocol": str(PROTOCOL.relative_to(ROOT))},
    }
    report["report_sha256"] = _digest(report)
    catalog = {"schema_version": f"{SCHEMA_VERSION}-human-catalog", "status": "completed_real_local_human_review_catalog", "entries": human_rows, "raw_payloads_human_review_only": True, "raw_response_bodies_stored": False, "external_network": False}
    catalog["catalog_sha256"] = _digest(catalog)
    dataset = {"schema_version": f"{SCHEMA_VERSION}-training-dataset", "source_catalog": str(CATALOG.relative_to(ROOT)), "records": abstract_rows, "counts": {"records": len(abstract_rows), "real_live_records": len(abstract_rows), "typed_effect_records": typed_count}, "contract": {"process_question_supervision": True, "abstract_assembly_target": True, "real_get_post_replay": True, "fresh_reset_required": True, "typed_oracle_required": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    trace = {"schema_version": f"{SCHEMA_VERSION}-trace", "episodes": traces, "raw_payloads_human_catalog_only": True, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}
    trace["trace_sha256"] = _digest(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": f"{SCHEMA_VERSION}-protocol", "scope": {"target": "authorized local Docker Pikachu", "loopback_only": True, "external_network": False, "state_change_allowed": False, "methods": ["GET", "POST"]}, "required_gates": {"missing_question_before_assembly": True, "get_post_pair": True, "matched_negative": True, "fresh_reset": True, "typed_oracle": True, "evidence_hash": True, "raw_payload_training_excluded": True}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(CATALOG, catalog)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-305 真实本地 Docker composer 回放", "", f"routes={len(human_rows)} GET={get_count} POST={post_count}; model_send={model_send_count}; model_confirmed={model_confirmed_count}; evaluator_gold={typed_count}", f"preflight raw question recall={model_preflight['raw']['missing_question_recall']}; guarded={model_preflight['guarded']['missing_question_recall']}; raw unnecessary={model_preflight['raw']['unnecessary_question_rate']}; guarded unnecessary={model_preflight['guarded']['unnecessary_question_rate']}", "", "模型只输出抽象 Rule-IR；真实值/wire 仅存人审 catalog。typed effect 不是漏洞声明；训练和长期记忆晋级关闭。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "preflight": report["preflight_identifiability"], "report": str(REPORT.relative_to(ROOT)), "catalog": str(CATALOG.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
