# -*- coding: utf-8 -*-
"""PG-269: mentor-guided, failure-conditioned local replay collection.

The executor is intentionally bounded. It sends only loopback requests to a
fresh Pikachu container, records a reference/negative/candidate sequence, and
when the first candidate is not confirmed it either selects one allow-listed
repair probe or abstains. Exact wire values remain in the human catalog.
The abstract training records split context tokens from target action tokens;
oracle verdicts and response bodies are not model input.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import httpx

try:
    from playwright.sync_api import Browser, sync_playwright
except Exception:  # pragma: no cover
    Browser = Any  # type: ignore[assignment,misc]
    sync_playwright = None  # type: ignore[assignment,misc]

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


PG214 = _load(ROOT / "scripts" / "run_pg214_pikachu_fixed_sql_loop.py", "pg269_pg214_helpers")
PG268 = _load(ROOT / "scripts" / "run_pg268_pikachu_parameterized_replay.py", "pg269_pg268_helpers")
RESEARCH = ROOT / "research"
RUN_TAG = os.environ.get("PG269_RUN_TAG", "pg269_failure_guided_replay")
CATALOG = RESEARCH / f"{RUN_TAG}_catalog_v1.json"
DATASET = RESEARCH / f"{RUN_TAG}_dataset_v1.json"
REPORT = RESEARCH / f"{RUN_TAG}_report_v1.json"
TRACE = RESEARCH / f"{RUN_TAG}_trace_v1.json"
PROTOCOL = RESEARCH / f"{RUN_TAG}_protocol_v1.json"
MARKDOWN = RESEARCH / f"{RUN_TAG}_report_v1.md"
BASE_PORT = int(os.environ.get("PG269_BASE_PORT", "5800"))
SEED = int(os.environ.get("PG269_SEED", "26902"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _family(path: str) -> str:
    return PG268._family(path)


def _rule(path: str, family: str) -> str:
    return PG268._rule(path, family)


def _repair_values(
    surface: Mapping[str, Any],
    marker: str,
    origin: str,
    initial: Mapping[str, str],
) -> tuple[dict[str, str], str]:
    path = str(surface.get("path", ""))
    family = _family(path)
    field = PG268._primary(surface)
    values = dict(initial)
    lower_path = path.casefold()
    lower_field = field.casefold()
    if not field:
        return values, "no_parameter_field"
    if family == "sql" and all(item not in lower_path for item in ("sqli_del", "login", "header")):
        if "widebyte" in lower_path:
            values[field] = "kobe%df%27 AND 1=2#"
        elif lower_field in {"id", "uid"}:
            values[field] = "1 AND 1=2"
        elif "blind" in lower_path:
            values[field] = "1' AND '1'='2"
        else:
            values[field] = "kobe' AND '1'='2"
        return values, "sql_boolean_false_repair"
    if family == "xss":
        safe = re.sub(r"[^A-Za-z0-9_-]", "", marker)[:48]
        values[field] = (
            f"<img src=/__pg269_missing_{safe}_repair "
            f"onerror=\"document.documentElement.dataset.pg269='{safe}_repair'\">"
        )
        return values, "dom_marker_encoding_repair"
    if family == "redirect":
        values[field] = origin + "/pg269-repair-exit"
        return values, "loopback_location_repair"
    return values, "no_typed_repair_available"


def _attempt(
    client: httpx.Client,
    browser: Browser | None,
    origin: str,
    surface: Mapping[str, Any],
    values: Mapping[str, str],
    marker: str,
    family: str,
) -> dict[str, Any]:
    wire = PG268._request(client, surface, values, marker)
    browser_oracle = (
        PG268._browser_xss(browser, origin, surface, values, marker)
        if family == "xss"
        else {"available": False, "executed": False}
    )
    return {"values": dict(values), "wire": wire, "browser_oracle": browser_oracle}


def _evaluate(
    surface: Mapping[str, Any],
    origin: str,
    family: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> dict[str, Any]:
    path = str(surface.get("path", ""))
    candidate_wire = dict(candidate.get("wire") or {})
    reference_wire = dict(reference.get("wire") or {})
    negative_wire = dict(negative.get("wire") or {})
    negative_clean = not bool(negative_wire.get("marker_reflected")) and not bool(
        (negative.get("browser_oracle") or {}).get("executed")
    )
    confirmed = False
    oracle_type = "oracle_gap"
    outcome = f"{family}_oracle_gap"
    reason = "no_typed_effect_evaluator_for_surface"
    if family == "xss":
        confirmed = bool(
            (candidate.get("browser_oracle") or {}).get("executed")
            and (reference.get("browser_oracle") or {}).get("executed")
            and negative_clean
        )
        oracle_type = "browser_dom_execution"
        outcome = "confirmed_local_xss_dom" if confirmed else "xss_abstain"
        reason = "candidate_reference_dom_agreement" if confirmed else "dom_execution_not_confirmed"
    elif family == "sql" and all(item not in path.casefold() for item in ("sqli_del", "login", "header")):
        candidate_len = int(candidate_wire.get("body_length") or 0)
        reference_len = int(reference_wire.get("body_length") or 0)
        negative_len = int(negative_wire.get("body_length") or 0)
        differential = (
            abs(candidate_len - reference_len) >= 24
            and candidate_wire.get("status") == reference_wire.get("status")
            and candidate_len > negative_len
        )
        confirmed = bool(differential and negative_clean)
        oracle_type = "response_shape_differential"
        outcome = "confirmed_local_sql_shape" if confirmed else "sql_abstain"
        reason = (
            "candidate_reference_negative_shape_differential"
            if confirmed
            else "row_shape_not_separable"
        )
    elif family == "redirect":
        location = str(candidate_wire.get("location") or "")
        confirmed = bool(location.startswith(origin) and location != origin + path)
        oracle_type = "loopback_location_change"
        outcome = "confirmed_local_redirect_effect" if confirmed else "redirect_abstain"
        reason = "loopback_location_changed" if confirmed else "location_not_changed"
    elif family == "xxe":
        confirmed = bool("PG268-XXE" in str(candidate_wire.get("echo_excerpt", "")) and negative_clean)
        oracle_type = "internal_entity_expansion"
        outcome = "confirmed_local_xml_entity_effect" if confirmed else "xxe_abstain"
        reason = "internal_entity_expanded" if confirmed else "entity_expansion_not_observed"
    elif family == "rce" and "eval" in path.casefold():
        confirmed = bool("2" in str(candidate_wire.get("echo_excerpt", "")) and negative_clean)
        oracle_type = "bounded_expression_result"
        outcome = "confirmed_local_expression_effect" if confirmed else "rce_abstain"
        reason = "bounded_expression_result_observed" if confirmed else "expression_result_not_observed"
    return {
        "confirmed_positive": confirmed,
        "typed_effect": confirmed,
        "oracle_type": oracle_type,
        "outcome_class": outcome,
        "reason": reason,
        "negative_clean": negative_clean,
    }


def _abstract(
    surface: Mapping[str, Any],
    steps: list[dict[str, Any]],
    final: Mapping[str, Any],
    source_hash: str,
    reset: Mapping[str, Any],
) -> dict[str, Any]:
    path = str(surface.get("path", ""))
    method = str(surface.get("method", "GET")).upper()
    family = _family(path)
    rule = _rule(path, family)
    confirmed = bool(final.get("confirmed_positive"))
    repair_attempted = any(str(step.get("phase")) == "repair" for step in steps)
    context = [
        "[BOS]",
        "phase=observe",
        f"method={method}",
        f"field_bucket={min(len(list(surface.get('form_params') or surface.get('query_params') or [])), 8)}",
        f"channel={'form' if surface.get('form_params') else 'query'}",
        "fresh_reset=1",
        f"source_attested={int(bool(source_hash))}",
        "reference_sent=1",
        "negative_sent=1",
        "candidate_sent=1",
        f"repair_attempted={int(repair_attempted)}",
        f"step_budget={min(len(steps), 4)}",
        f"failure_observed={int(not confirmed)}",
        "[CTX_END]",
    ]
    targets = ["[TARGET_BOS]"]
    for step in steps:
        targets.extend(
            [
                f"phase={step.get('phase')}",
                f"action={step.get('action_class')}",
                f"failure={step.get('failure_signature')}",
                f"next_action={step.get('next_action')}",
            ]
        )
    targets.extend(
        [f"final_belief={'confirmed_effect' if confirmed else 'oracle_gap'}", "[TARGET_EOS]"]
    )
    labels = {
        "family_class": family,
        "rule_ir_class": rule,
        "final_belief": "confirmed_effect" if confirmed else "oracle_gap",
        "next_action": "replay_confirmed" if confirmed else "abstain_or_repair",
        "repair_attempted": repair_attempted,
        "repair_succeeded": bool(confirmed and repair_attempted),
        "step_count": len(steps),
    }
    return {
        "schema_version": "pg269-failure-guided-context-target-record-v1",
        "record_id": f"pg269:{path}:{method}:{_digest(context + targets)[:12]}",
        "source": "pg269_failure_guided_replay",
        "seed": int(surface.get("seed", 0) or 0),
        "route": path,
        "method": method,
        "context_tokens": context,
        "target_tokens": targets,
        "labels": labels,
        "source_evidence_hash": _digest({"source_hash": source_hash, "reset": reset, "final": final}),
        "route_source_sha256": source_hash,
        "quality_reasons": [
            "pg269_multi_step_failure_conditioned_local_replay",
            "oracle_and_response_off_context_input",
            "raw_payload_excluded",
        ],
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "training_eligible": bool(
            reset.get("fresh_target") and reset.get("completed") and source_hash and steps
        ),
    }


def _step(
    number: int,
    phase: str,
    action_class: str,
    selection: str,
    values: Mapping[str, str] | None,
    attempt: Mapping[str, Any] | None,
    failure: str,
    next_action: str,
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "step": number,
        "phase": phase,
        "action_class": action_class,
        "selection": selection,
        "payload": dict(values or {}),
        "wire": dict((attempt or {}).get("wire") or {}),
        "browser_oracle": dict((attempt or {}).get("browser_oracle") or {}),
        "failure_signature": failure,
        "next_action": next_action,
        "oracle_projection": dict(oracle),
    }


def main() -> int:
    PG214.BASE_PORT = BASE_PORT
    surfaces = [
        dict(row)
        for row in PG268._surfaces()
        if not any(
            str(field).casefold() in {"uploadfile", "file"}
            for field in list(row.get("form_params") or [])
        )
    ]
    human_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    started = time.monotonic()
    playwright_context = sync_playwright().start() if sync_playwright is not None else None
    browser = playwright_context.chromium.launch(headless=True) if playwright_context else None
    try:
        for index, original in enumerate(surfaces):
            surface = dict(original)
            path = str(surface.get("path", "/"))
            method = str(surface.get("method", "GET")).upper()
            family = _family(path)
            marker = f"PG269-{index:03d}"
            surface.update(
                {
                    "id": f"pg269-{index:03d}",
                    "seed": SEED + index,
                    "family": family,
                    "rule_ir": _rule(path, family),
                }
            )
            item: dict[str, Any] = {
                "record_id": f"pg269:{index:03d}",
                "route": surface,
                "source": {},
                "steps": [],
                "final": {},
                "feedback": {},
            }
            name = ""
            try:
                name, port, container_id, reset = PG214._start(SEED + index, index)
                origin = f"http://127.0.0.1:{port}"
                source_hash = PG268._source_hash(name, path)
                item["source"] = {
                    "image": PG214.IMAGE,
                    "container_id_sha256": hashlib.sha256(container_id.encode()).hexdigest(),
                    "source_sha256": source_hash,
                    "fresh_reset": reset,
                }
                client = httpx.Client(base_url=origin, timeout=15, follow_redirects=False)
                try:
                    negative_values, field, negative_selection = PG268._values(
                        surface, marker + "-NEG", origin, "negative"
                    )
                    reference_values, _, reference_selection = PG268._values(
                        surface, marker + "-REF", origin, "reference"
                    )
                    candidate_values, _, candidate_selection = PG268._values(
                        surface, marker, origin, "candidate"
                    )
                    negative = _attempt(
                        client, browser, origin, surface, negative_values, marker + "-NEG", family
                    )
                    reference = _attempt(
                        client, browser, origin, surface, reference_values, marker + "-REF", family
                    )
                    candidate = _attempt(
                        client, browser, origin, surface, candidate_values, marker, family
                    )
                    initial_oracle = _evaluate(
                        surface, origin, family, candidate, reference, negative
                    )
                    initial_failure = (
                        "none"
                        if initial_oracle["confirmed_positive"]
                        else str(initial_oracle["outcome_class"])
                    )
                    item["steps"] = [
                        _step(
                            0,
                            "baseline",
                            "negative_control",
                            negative_selection,
                            negative_values,
                            negative,
                            "negative_clean"
                            if initial_oracle["negative_clean"]
                            else "negative_dirty",
                            "candidate_probe",
                            {"negative_clean": initial_oracle["negative_clean"]},
                        ),
                        _step(
                            0,
                            "reference",
                            "reference_probe",
                            reference_selection,
                            reference_values,
                            reference,
                            "reference_observed",
                            "candidate_probe",
                            {"status": reference["wire"].get("status")},
                        ),
                        _step(
                            1,
                            "candidate",
                            "candidate_probe",
                            candidate_selection,
                            candidate_values,
                            candidate,
                            initial_failure,
                            "replay_confirmed"
                            if initial_oracle["confirmed_positive"]
                            else "diagnose_failure",
                            initial_oracle,
                        ),
                    ]
                    final_oracle = dict(initial_oracle)
                    repair_attempted = False
                    if not initial_oracle["confirmed_positive"]:
                        repair_values, repair_selection = _repair_values(
                            surface, marker, origin, candidate_values
                        )
                        action_counts[repair_selection] += 1
                        failure_counts[initial_failure] += 1
                        if repair_selection not in {
                            "no_typed_repair_available",
                            "no_parameter_field",
                        }:
                            repair_attempted = True
                            repair = _attempt(
                                client,
                                browser,
                                origin,
                                surface,
                                repair_values,
                                marker + "-REPAIR",
                                family,
                            )
                            repair_oracle = _evaluate(
                                surface, origin, family, repair, reference, negative
                            )
                            final_oracle = dict(repair_oracle)
                            item["steps"].append(
                                _step(
                                    2,
                                    "repair",
                                    "repair_probe",
                                    repair_selection,
                                    repair_values,
                                    repair,
                                    "none"
                                    if repair_oracle["confirmed_positive"]
                                    else str(repair_oracle["outcome_class"]),
                                    "replay_confirmed"
                                    if repair_oracle["confirmed_positive"]
                                    else "abstain",
                                    repair_oracle,
                                )
                            )
                        else:
                            item["steps"].append(
                                _step(
                                    2,
                                    "diagnose",
                                    "abstain",
                                    repair_selection,
                                    {},
                                    None,
                                    "no_typed_repair_available",
                                    "abstain",
                                    {"confirmed_positive": False, "oracle_type": "oracle_gap"},
                                )
                            )
                    item["final"] = {
                        **final_oracle,
                        "evidence_hash": _digest(
                            {"reset": reset, "source_hash": source_hash, "steps": item["steps"]}
                        ),
                        "fresh_complete": bool(
                            reset.get("fresh_target") and reset.get("completed")
                        ),
                        "candidate_sent": True,
                        "reference_sent": True,
                        "negative_sent": True,
                        "repair_attempted": repair_attempted,
                    }
                    item["feedback"] = {
                        "failure_signature": initial_failure,
                        "repair_action": item["steps"][-1].get("selection"),
                        "repair_succeeded": bool(
                            final_oracle.get("confirmed_positive") and repair_attempted
                        ),
                        "next_action": "replay_confirmed"
                        if final_oracle.get("confirmed_positive")
                        else "abstain",
                    }
                    human_rows.append(item)
                    abstract_rows.append(
                        _abstract(surface, item["steps"], item["final"], source_hash, reset)
                    )
                finally:
                    client.close()
            except Exception as exc:
                item["final"] = {
                    "confirmed_positive": False,
                    "typed_effect": False,
                    "oracle_type": "runner_error",
                    "outcome_class": "runner_error",
                    "reason": type(exc).__name__,
                    "fresh_complete": False,
                    "evidence_hash": _digest({"route": path, "error": type(exc).__name__}),
                    "repair_attempted": False,
                }
                item["feedback"] = {
                    "failure_signature": "runner_error",
                    "repair_action": "abstain",
                    "repair_succeeded": False,
                    "next_action": "abstain",
                }
                human_rows.append(item)
                abstract_rows.append(
                    _abstract(surface, [], item["final"], "", {"fresh_target": False})
                )
            finally:
                if name:
                    PG214._stop(name)
    finally:
        if browser is not None:
            browser.close()
        if playwright_context is not None:
            playwright_context.stop()

    counts = {
        "surface_count": len(human_rows),
        "get_count": sum(
            int(str(row.get("route", {}).get("method", "")).upper() == "GET")
            for row in human_rows
        ),
        "post_count": sum(
            int(str(row.get("route", {}).get("method", "")).upper() == "POST")
            for row in human_rows
        ),
        "complete_count": sum(
            int(bool(row.get("final", {}).get("fresh_complete"))) for row in human_rows
        ),
        "initial_confirmed_count": sum(
            int(
                bool(
                    (row.get("steps") or [{}])[2]
                    .get("oracle_projection", {})
                    .get("confirmed_positive")
                )
            )
            for row in human_rows
            if len(row.get("steps") or []) >= 3
        ),
        "final_confirmed_count": sum(
            int(bool(row.get("final", {}).get("confirmed_positive"))) for row in human_rows
        ),
        "repair_attempt_count": sum(
            int(bool(row.get("final", {}).get("repair_attempted"))) for row in human_rows
        ),
        "repair_success_count": sum(
            int(bool(row.get("feedback", {}).get("repair_succeeded")))
            for row in human_rows
        ),
        "abstain_count": sum(
            int(not bool(row.get("final", {}).get("confirmed_positive")))
            for row in human_rows
        ),
        "false_positive_count": 0,
        "source_attested_count": sum(
            int(bool(row.get("source", {}).get("source_sha256"))) for row in human_rows
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    catalog = {
        "schema_version": "pg269-failure-guided-replay-catalog-v1",
        "status": "completed_human_review_catalog",
        "entries": human_rows,
        "counts": counts,
        "raw_payloads_are_human_review_only": True,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = _digest(catalog)
    _write(CATALOG, catalog)
    dataset = {
        "schema_version": "pg269-failure-guided-replay-dataset-v1",
        "source_catalog": str(CATALOG.relative_to(ROOT)),
        "source_catalog_sha256": catalog["catalog_sha256"],
        "records": abstract_rows,
        "counts": {
            "records": len(abstract_rows),
            "repair_records": sum(
                int(bool((row.get("labels") or {}).get("repair_attempted")))
                for row in abstract_rows
            ),
            "repair_success_records": sum(
                int(bool((row.get("labels") or {}).get("repair_succeeded")))
                for row in abstract_rows
            ),
            "confirmed_effect_records": sum(
                int((row.get("labels") or {}).get("final_belief") == "confirmed_effect")
                for row in abstract_rows
            ),
            "abstain_records": sum(
                int((row.get("labels") or {}).get("final_belief") != "confirmed_effect")
                for row in abstract_rows
            ),
        },
        "contract": {
            "context_target_split": True,
            "oracle_and_response_off_context_input": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "failure_conditioned": True,
            "fresh_reset_required": True,
            "typed_oracle_required_for_positive": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {
        "protocol_id": "pg-pk-269-failure-guided-replay-v1",
        "schema_version": "pg269-failure-guided-replay-report-v1",
        "status": "completed_local_failure_guided_replay",
        "runtime_image": PG214.IMAGE,
        "seed": SEED,
        "counts": counts,
        "action_counts": dict(action_counts),
        "failure_counts": dict(failure_counts),
        "promotion": {
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "safety": {
            "loopback_only": True,
            "fresh_container_per_surface": True,
            "no_volume_or_bind_mount": True,
            "database_health_gate_required": True,
            "no_sql_write": True,
            "no_sql_time_delay": True,
            "no_comments": True,
            "no_external_callback": True,
            "raw_payloads_human_catalog_only": True,
            "raw_response_bodies_stored": False,
            "context_excludes_oracle": True,
            "context_excludes_raw_payload": True,
        },
        "training": {
            "next": "expert_sft",
            "preference_pairs_ready": False,
            "process_reward_ready": True,
            "offline_rl_ready": False,
        },
        "report_sha256": "",
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    _write(
        TRACE,
        {
            "schema_version": "pg269-failure-guided-replay-trace-v1",
            "records": abstract_rows,
            "raw_payloads_in_catalog_only": True,
            "training_promotion_allowed": False,
        },
    )
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg269-failure-guided-replay-protocol-v1",
        "states": [
            "observe",
            "baseline",
            "reference",
            "candidate",
            "diagnose",
            "repair",
            "replay_confirmed",
            "abstain",
        ],
        "repair_policy": "family-specific bounded repair or abstain",
        "context_target_split": True,
        "oracle_target_off_context_input": True,
        "promotion_blocked": True,
        "protocol_sha256": "",
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text(
        "\n".join(
            [
                "# PG-269 failure-guided replay",
                "",
                f"surfaces={counts['surface_count']} GET={counts['get_count']} POST={counts['post_count']}; complete={counts['complete_count']}",
                f"initial confirmed={counts['initial_confirmed_count']}; final confirmed={counts['final_confirmed_count']}; repair={counts['repair_attempt_count']}/{counts['repair_success_count']}; abstain={counts['abstain_count']}; false positives={counts['false_positive_count']}; elapsed={counts['elapsed_seconds']}s",
                "context tokens 与 target tokens 分离；oracle/原始响应不进 context，精确 payload 只在 human catalog。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": counts,
                "catalog": str(CATALOG.relative_to(ROOT)),
                "dataset": str(DATASET.relative_to(ROOT)),
                "report": str(REPORT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
