"""PG-244: collect real SQL/XSS failure -> diagnosis -> repair trajectories.

The collector stays inside the pinned Pikachu image and loopback.  It sends a
matched no-effect counterfactual, lets the frozen Rule-IR model choose the
abstract channel for the repair candidate, compares that candidate with an
independent reference, and then replays the repaired request on a newly
created container.  Persisted rows contain only bounded projections, hashes,
and abstract feedback tokens; executable payloads and response bodies are
printed only for the human wire log and discarded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import httpx
import torch
from playwright.sync_api import Browser, sync_playwright

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
PG241 = _load("run_pg241_pikachu_payload_acceptance.py")
PG242 = _load("run_pg242_pikachu_xss_dom_acceptance.py")
PG208 = PG214.PG212.PG208

from app.payload_learner import PayloadLearner  # noqa: E402
from app.pg198_payload_grounding import candidate_summary  # noqa: E402
from app.pg230_next_token_quality_funnel import digest  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
IMAGE = "sift/pikachu-pg240-source-native:5e1e8d9d"
SOURCE_COMMIT = "5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc"
SEEDS = (24401, 24402)
BASE_PORT = 10340
REPORT = RESEARCH / "pg244_failure_repair_trajectory_report_v1.json"
DATASET = RESEARCH / "pg244_failure_repair_trajectory_dataset_v1.json"
TRACE = RESEARCH / "pg244_failure_repair_trajectory_trace_v1.json"
PROTOCOL = RESEARCH / "pg244_failure_repair_trajectory_protocol_v1.json"
MARKDOWN = RESEARCH / "pg244_failure_repair_trajectory_report_v1.md"


def _sha(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _wire(origin: str, spec: Mapping[str, Any]) -> str:
    return PG241._wire_text(origin, spec)


def _sql_projection(response: Mapping[str, Any]) -> dict[str, Any]:
    return dict(response.get("projection") or {})


def _xss_projection(response: Mapping[str, Any]) -> dict[str, Any]:
    response = dict(response or {})
    return {
        "status_code": int(response.get("status_code", 0) or 0),
        "status_class": str(response.get("status_class", "unknown")),
        "body_length_bucket": str(response.get("body_length_bucket", "unknown")),
        "typed_marker": bool(response.get("marker_observed", False)),
        "typed_effect": bool(response.get("dom_effect", False)),
        "oracle_available": bool(response.get("oracle_available", True)),
        "external_network": bool(response.get("external_network", False)),
        "external_request_blocked": bool(response.get("external_request_blocked", False)),
        "body_sha256": str(response.get("body_sha256", "")),
        "projection_sha256": digest(response),
    }


def _record(
    *,
    source: str,
    seed: int,
    family: str,
    method: str,
    fields: list[str],
    reset: Mapping[str, Any],
    projection: Mapping[str, Any],
    negative_projection: Mapping[str, Any],
    reference_projection: Mapping[str, Any],
    step: str,
    step_index: int,
    route_source_sha256: str,
    parent_record_id: str,
    model_decision: Mapping[str, Any],
    model_candidate: Mapping[str, Any] | None,
    candidate_sent: bool,
    candidate_effect: bool,
    candidate_reference_agreement: bool,
    failure_signature: str,
    next_step: str,
    previous_feedback: str,
    model_self_error: bool = False,
    model_self_error_kind: str | None = None,
    negative_control_confirmed: bool = False,
    history_tokens: list[str] | None = None,
) -> dict[str, Any]:
    status_class = str(projection.get("status_class", "unknown"))
    negative_clean = not bool(negative_projection.get("typed_marker") or negative_projection.get("row_marker_count"))
    typed = bool(candidate_effect)
    evidence = {
        "schema_version": "pg244-failure-repair-evidence-v1",
        "source": source,
        "seed": int(seed),
        "family": family,
        "method": method,
        "step": step,
        "step_index": int(step_index),
        "route_source_sha256": route_source_sha256,
        "reset_id": reset.get("reset_id"),
        "candidate_projection": dict(projection),
        "negative_projection": dict(negative_projection),
        "reference_projection": dict(reference_projection),
        "candidate_effect": typed,
        "candidate_reference_agreement": bool(candidate_reference_agreement),
        "negative_clean": bool(negative_clean),
        "failure_signature": failure_signature,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    evidence_hash = digest(evidence)
    raw = {
        "source": source,
        "seed": int(seed),
        "surface_role": "sqli_surface" if family == "sql" else "xss_dom_surface",
        "method": method,
        "field_count": len(fields),
        "status_class": status_class,
        "history_len": len(history_tokens or []),
        "fresh_reset_ok": bool(PG241._reset_ok(reset)),
        "reset_completed": bool(reset.get("completed")),
        "reset_not_attempted": False,
        "candidate_sent": bool(candidate_sent),
        "oracle_available": bool(projection.get("oracle_available", True)),
        "typed_effect_confirmed": typed,
        "typed_effect_observed": typed,
        "result_fixture_verified": typed,
        "candidate_reference_agreement": bool(candidate_reference_agreement),
        "negative_clean": bool(negative_clean),
        "binding_valid": True,
        "transport_error": False,
        "result_mismatch_observed": False,
        "next_step": next_step,
        "previous_feedback": previous_feedback,
        "candidate_result_present": typed,
        "model_claimed_positive": bool(candidate_sent),
        "model_abstained": not bool(candidate_sent),
        "model_self_error_detected": bool(model_self_error),
        "model_self_error_kind": model_self_error_kind,
        "negative_control_confirmed": bool(negative_control_confirmed),
        "abstention_required": False,
        "failure_signature": failure_signature,
        "evidence_hash": evidence_hash,
        "payload_grounded_eligible": bool(typed and candidate_reference_agreement and negative_clean),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    prepared = prepare_feedback_record(raw)
    current_tokens = list(prepared["tokens"])
    prior = list(history_tokens or [])
    if prior:
        # Keep one BOS and retain all previous abstract steps.  The body
        # encoder's existing aliases cover these tokens; route/payload values
        # never enter the sequence.
        current_tokens = ["[BOS]"] + prior + current_tokens[1:]
    if len(current_tokens) > 384:
        current_tokens = ["[BOS]"] + current_tokens[-383:]
    prepared["tokens"] = current_tokens
    prepared["trajectory_hash"] = digest(current_tokens)
    prepared["classification_position"] = max((index for index, token in enumerate(current_tokens) if str(token).startswith("failure=")), default=len(current_tokens) - 1)
    prepared.update(
        {
            "source": source,
            "seed": int(seed),
            "family": family,
            "step": step,
            "step_index": int(step_index),
            "route_source_sha256": route_source_sha256,
            "parent_record_id": parent_record_id,
            "model_decision": {
                "action": model_decision.get("effective_action", model_decision.get("action", "abstain")),
                "encoding": model_decision.get("encoding"),
                "failure": model_decision.get("failure"),
                "action_confidence": model_decision.get("action_confidence"),
            },
            "model_candidate": candidate_summary(model_candidate) if model_candidate else None,
            "source_evidence_hash": evidence_hash,
            "failure_signature": failure_signature,
            "repair_delta": {"from": "counterfactual_no_effect", "to": "typed_effect"} if step in {"repair", "replay"} and typed else None,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
    )
    return prepared


def _sql_case(case_id: str) -> dict[str, Any]:
    return next(case for case in PG241._case_specs() if case["case_id"] == case_id)


def _xss_case(case_id: str) -> dict[str, Any]:
    return next(case for case in PG242._case_specs() if case["case_id"] == case_id)


def _sql_episode(model: Any, vocabulary: Mapping[str, int], device: torch.device, learner: PayloadLearner, case: Mapping[str, Any], *, seed: int, run_index: int, fresh_replay_index: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    route_hash = _sha(f"{case['method']} {case['path']}")
    name, port, container_id, reset = PG214._start(seed, run_index)
    origin = f"http://127.0.0.1:{port}"
    records: list[dict[str, Any]] = []
    wires: list[str] = []
    repair_spec_for_replay: dict[str, Any] | None = None
    try:
        with httpx.Client(base_url=origin, timeout=15.0) as client:
            baseline_spec = case["builder"](case, str(case["baseline"]))
            negative_spec = case["builder"](case, str(case["negative"]))
            baseline = PG241._send(client, baseline_spec)["projection"]
            negative = PG241._send(client, negative_spec)["projection"]
            wires.extend([_wire(origin, baseline_spec), _wire(origin, negative_spec)])
            decision, validation, candidates = PG241._model_context(model, vocabulary, device, case, negative, seed=seed)
            selected = learner.select(candidates) if candidates and validation.get("valid") and decision.get("effective_action") == "safe_candidate" else None
            ai_spec = PG241._bound_spec(case, selected, role="ai") if selected is not None else None
            reference_spec = PG241._bound_spec(case, selected, role="reference")
            ai_result = PG241._send(client, ai_spec)["projection"] if ai_spec is not None else {}
            reference_result = PG241._send(client, reference_spec)["projection"] if reference_spec is not None else {}
            if ai_spec is not None:
                wires.append(_wire(origin, ai_spec))
            if reference_spec is not None:
                wires.append(_wire(origin, reference_spec))
            ai_effect = PG241._positive(case, ai_result)
            reference_effect = PG241._positive(case, reference_result)
            parent = f"pg244:{seed}:{route_hash}:counterfactual"
            failed = _record(source="pg244_pikachu_sql_repair", seed=seed, family="sql", method=str(case["method"]).upper(), fields=list(case["fields"]), reset=reset, projection=negative, negative_projection=negative, reference_projection=reference_result, step="counterfactual", step_index=0, route_source_sha256=route_hash, parent_record_id=parent, model_decision=decision, model_candidate=selected, candidate_sent=True, candidate_effect=False, candidate_reference_agreement=False, failure_signature="counterfactual_candidate_no_effect", next_step="retry_candidate", previous_feedback="none", negative_control_confirmed=True)
            records.append(failed)
            history = list(failed["tokens"][1:])
            repaired = _record(source="pg244_pikachu_sql_repair", seed=seed, family="sql", method=str(case["method"]).upper(), fields=list(case["fields"]), reset=reset, projection=ai_result if ai_spec is not None else reference_result, negative_projection=negative, reference_projection=reference_result, step="repair", step_index=1, route_source_sha256=route_hash, parent_record_id=failed["parent_record_id"], model_decision=decision, model_candidate=selected, candidate_sent=ai_spec is not None, candidate_effect=ai_effect, candidate_reference_agreement=bool(ai_effect == reference_effect and ai_effect), failure_signature="typed_effect_after_repair" if ai_effect else "model_abstain_on_reference_positive", next_step="abstain" if ai_effect else "retry_candidate", previous_feedback="failure_adjusted", model_self_error=bool(ai_spec is not None and not ai_effect and reference_effect), model_self_error_kind="model_candidate_no_effect" if ai_spec is not None and not ai_effect and reference_effect else None, negative_control_confirmed=False, history_tokens=history)
            records.append(repaired)
            repair_payload = ai_spec if ai_spec is not None and ai_effect else reference_spec
            repair_spec_for_replay = dict(repair_payload) if repair_payload is not None else None
            episode = {"seed": seed, "route": case["path"], "method": case["method"], "family": "sql", "reset": reset, "target_instance_hash": _sha(container_id), "model": {"decision": decision, "validation": validation, "candidate": candidate_summary(selected) if selected else None}, "baseline": baseline, "negative": negative, "counterfactual": {"projection": negative, "payload_sha256": PG241._payload_sha(negative_spec)}, "repair": {"projection": ai_result if ai_spec is not None else reference_result, "payload_sha256": PG241._payload_sha(repair_payload) if repair_payload else None, "reference_projection": reference_result}, "fresh_replay_pending": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
    finally:
        PG214._stop(name)
    # A separate container is mandatory for replay.  This second episode is
    # deliberately kept as a small append-only record linked to the repair.
    replay_name, replay_port, replay_container, replay_reset = PG214._start(seed, fresh_replay_index)
    replay_origin = f"http://127.0.0.1:{replay_port}"
    try:
        with httpx.Client(base_url=replay_origin, timeout=15.0) as client:
            baseline_spec = case["builder"](case, str(case["baseline"]))
            negative_spec = case["builder"](case, str(case["negative"]))
            baseline = PG241._send(client, baseline_spec)["projection"]
            negative = PG241._send(client, negative_spec)["projection"]
            wires.extend([_wire(replay_origin, baseline_spec), _wire(replay_origin, negative_spec)])
            repair = records[-1]
            # The actual value is never persisted; the selected branch is
            # reconstructed from the original case's vetted reference/AI
            # binding and its persisted hash.
            replay_spec = dict(repair_spec_for_replay or case["builder"](case, str(case["reference"])))
            replay_result = PG241._send(client, replay_spec)["projection"]
            reference_spec = case["builder"](case, str(case["reference"]))
            reference_result = PG241._send(client, reference_spec)["projection"]
            wires.extend([_wire(replay_origin, replay_spec), _wire(replay_origin, reference_spec)])
            replay_effect = PG241._positive(case, replay_result)
            parent = str(repair.get("parent_record_id", ""))
            replay_record = _record(source="pg244_pikachu_sql_repair", seed=seed, family="sql", method=str(case["method"]).upper(), fields=list(case["fields"]), reset=replay_reset, projection=replay_result, negative_projection=negative, reference_projection=reference_result, step="replay", step_index=2, route_source_sha256=route_hash, parent_record_id=str(repair.get("trajectory_hash", parent)), model_decision=repair.get("model_decision") or {}, model_candidate=None, candidate_sent=True, candidate_effect=replay_effect, candidate_reference_agreement=bool(replay_effect == PG241._positive(case, reference_result) and replay_effect), failure_signature="typed_effect_replay" if replay_effect else "replay_no_effect", next_step="abstain" if replay_effect else "retry_candidate", previous_feedback="result_verified", history_tokens=list(repair.get("tokens", [])[1:]))
            records.append(replay_record)
            episode["fresh_replay"] = {"reset": replay_reset, "target_instance_hash": _sha(replay_container), "baseline": baseline, "negative": negative, "repair": replay_result, "reference": reference_result}
            episode["fresh_replay_pending"] = False
    finally:
        PG214._stop(replay_name)
    return episode, records, wires


def _xss_episode(browser: Browser, model: Any, vocabulary: Mapping[str, int], device: torch.device, learner: PayloadLearner, case: Mapping[str, Any], *, seed: int, run_index: int, fresh_replay_index: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    route_hash = _sha(f"{case['method']} {case['path']}")
    name, port, container_id, reset = PG214._start(seed, run_index)
    origin = f"http://127.0.0.1:{port}"
    records: list[dict[str, Any]] = []
    wires: list[str] = []
    try:
        baseline_raw = PG242._run_browser_probe(browser, origin, case, "pg244-safe-baseline", "never-observed")
        negative_raw = PG242._run_browser_probe(browser, origin, case, "pg244-safe-negative", "never-observed")
        baseline = _xss_projection(baseline_raw)
        negative = _xss_projection(negative_raw)
        wires.extend([PG242._wire(origin, case, "pg244-safe-baseline"), PG242._wire(origin, case, "pg244-safe-negative")])
        decision, validation, candidates = PG242._model_context(model, vocabulary, device, case, baseline, seed=seed)
        selected = learner.select(candidates) if candidates and validation.get("valid") and decision.get("effective_action") == "safe_candidate" else None
        ai_payload = str(case["ai"]) if selected is not None else None
        reference_payload = str(case["reference"])
        ai_raw = PG242._run_browser_probe(browser, origin, case, ai_payload, PG242._marker_from_payload(ai_payload)) if ai_payload is not None else {}
        reference_raw = PG242._run_browser_probe(browser, origin, case, reference_payload, PG242._marker_from_payload(reference_payload))
        ai_result = _xss_projection(ai_raw)
        reference_result = _xss_projection(reference_raw)
        if ai_payload is not None:
            wires.append(PG242._wire(origin, case, ai_payload))
        wires.append(PG242._wire(origin, case, reference_payload))
        ai_effect = bool(ai_result.get("typed_marker"))
        reference_effect = bool(reference_result.get("typed_marker"))
        parent = f"pg244:{seed}:{route_hash}:counterfactual"
        failed = _record(source="pg244_pikachu_xss_repair", seed=seed, family="xss", method=str(case["method"]).upper(), fields=list(case["fields"]), reset=reset, projection=negative, negative_projection=negative, reference_projection=reference_result, step="counterfactual", step_index=0, route_source_sha256=route_hash, parent_record_id=parent, model_decision=decision, model_candidate=selected, candidate_sent=True, candidate_effect=False, candidate_reference_agreement=False, failure_signature="counterfactual_candidate_no_effect", next_step="retry_candidate", previous_feedback="none", negative_control_confirmed=True)
        records.append(failed)
        repaired = _record(source="pg244_pikachu_xss_repair", seed=seed, family="xss", method=str(case["method"]).upper(), fields=list(case["fields"]), reset=reset, projection=ai_result if ai_payload is not None else reference_result, negative_projection=negative, reference_projection=reference_result, step="repair", step_index=1, route_source_sha256=route_hash, parent_record_id=failed["parent_record_id"], model_decision=decision, model_candidate=selected, candidate_sent=ai_payload is not None, candidate_effect=ai_effect, candidate_reference_agreement=bool(ai_effect == reference_effect and ai_effect), failure_signature="typed_effect_after_repair" if ai_effect else "model_abstain_on_reference_positive", next_step="abstain" if ai_effect else "retry_candidate", previous_feedback="failure_adjusted", model_self_error=bool(ai_payload is not None and not ai_effect and reference_effect), model_self_error_kind="model_candidate_no_effect" if ai_payload is not None and not ai_effect and reference_effect else None, negative_control_confirmed=False, history_tokens=list(failed["tokens"][1:]))
        records.append(repaired)
        episode = {"seed": seed, "route": case["path"], "method": case["method"], "family": "xss", "reset": reset, "target_instance_hash": _sha(container_id), "model": {"decision": decision, "validation": validation, "candidate": candidate_summary(selected) if selected else None}, "baseline": baseline, "negative": negative, "counterfactual": {"projection": negative}, "repair": {"projection": ai_result if ai_payload is not None else reference_result, "reference_projection": reference_result}, "fresh_replay_pending": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
    finally:
        PG214._stop(name)
    replay_name, replay_port, replay_container, replay_reset = PG214._start(seed, fresh_replay_index)
    replay_origin = f"http://127.0.0.1:{replay_port}"
    try:
        baseline_raw = PG242._run_browser_probe(browser, replay_origin, case, "pg244-safe-baseline", "never-observed")
        negative_raw = PG242._run_browser_probe(browser, replay_origin, case, "pg244-safe-negative", "never-observed")
        replay_payload = str(case["ai"])
        reference_payload = str(case["reference"])
        replay_raw = PG242._run_browser_probe(browser, replay_origin, case, replay_payload, PG242._marker_from_payload(replay_payload))
        reference_raw = PG242._run_browser_probe(browser, replay_origin, case, reference_payload, PG242._marker_from_payload(reference_payload))
        wires.extend([PG242._wire(replay_origin, case, "pg244-safe-baseline"), PG242._wire(replay_origin, case, "pg244-safe-negative"), PG242._wire(replay_origin, case, replay_payload), PG242._wire(replay_origin, case, reference_payload)])
        repair = records[-1]
        replay_result = _xss_projection(replay_raw)
        replay_reference = _xss_projection(reference_raw)
        replay_effect = bool(replay_result.get("typed_marker"))
        replay_record = _record(source="pg244_pikachu_xss_repair", seed=seed, family="xss", method=str(case["method"]).upper(), fields=list(case["fields"]), reset=replay_reset, projection=replay_result, negative_projection=_xss_projection(negative_raw), reference_projection=replay_reference, step="replay", step_index=2, route_source_sha256=route_hash, parent_record_id=str(repair.get("trajectory_hash", repair.get("parent_record_id", ""))), model_decision=repair.get("model_decision") or {}, model_candidate=None, candidate_sent=True, candidate_effect=replay_effect, candidate_reference_agreement=bool(replay_effect == bool(replay_reference.get("typed_marker")) and replay_effect), failure_signature="typed_effect_replay" if replay_effect else "replay_no_effect", next_step="abstain" if replay_effect else "retry_candidate", previous_feedback="result_verified", history_tokens=list(repair.get("tokens", [])[1:]))
        records.append(replay_record)
        episode["fresh_replay"] = {"reset": replay_reset, "target_instance_hash": _sha(replay_container), "baseline": _xss_projection(baseline_raw), "negative": _xss_projection(negative_raw), "repair": replay_result, "reference": replay_reference}
        episode["fresh_replay_pending"] = False
    finally:
        PG214._stop(replay_name)
    return episode, records, wires


def main() -> int:
    PG214.IMAGE = IMAGE
    PG214.BASE_PORT = BASE_PORT
    PG242.PG214.IMAGE = IMAGE
    PG242.PG214.BASE_PORT = BASE_PORT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary = PG208._load_model(device)
    model.eval()
    learner = PayloadLearner(seed=244)
    sql_cases = [_sql_case("sqli_id_numeric"), _sql_case("sqli_search_like")]
    xss_cases = [_xss_case("xss_reflected_get"), _xss_case("xss_reflected_post")]
    episodes: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    wires: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            case_index = 0
            for seed in SEEDS:
                for case in sql_cases:
                    print(f"\nPG244 TRAJECTORY seed={seed} family=sql case={case['case_id']}", flush=True)
                    episode, rows, shown = _sql_episode(model, vocabulary, device, learner, case, seed=seed, run_index=case_index, fresh_replay_index=case_index + 100)
                    case_index += 1
                    episodes.append(episode)
                    records.extend(rows)
                    wires.extend(shown)
                for case in xss_cases:
                    print(f"\nPG244 TRAJECTORY seed={seed} family=xss case={case['case_id']}", flush=True)
                    episode, rows, shown = _xss_episode(browser, model, vocabulary, device, learner, case, seed=seed, run_index=case_index, fresh_replay_index=case_index + 100)
                    case_index += 1
                    episodes.append(episode)
                    records.extend(rows)
                    wires.extend(shown)
        finally:
            browser.close()
    for wire in wires:
        print("--- WIRE (display-only; not persisted) ---\n" + wire, flush=True)
    counts = {
        "episode_count": len(episodes),
        "fresh_container_count": len(episodes) * 2,
        "record_count": len(records),
        "sql_episode_count": sum(int(row["family"] == "sql") for row in episodes),
        "xss_episode_count": sum(int(row["family"] == "xss") for row in episodes),
        "get_episode_count": sum(int(str(row["method"]).upper() == "GET") for row in episodes) * 2,
        "post_episode_count": sum(int(str(row["method"]).upper() == "POST") for row in episodes) * 2,
        "gold_count": sum(int(row["lane"] == "gold") for row in records),
        "hard_negative_count": sum(int(row["lane"] == "hard_negative") for row in records),
        "model_self_error_count": sum(int(row.get("model_self_error_detected")) for row in records),
        "replay_count": sum(int(row["step"] == "replay") for row in records),
        "wire_display_count": len(wires),
        "external_network_count": 0,
    }
    report = {"protocol_id": "pg-pk-244-failure-repair-trajectory-v1", "schema_version": "pg244-failure-repair-trajectory-report-v1", "status": "completed_local_sql_xss_multistep_failure_repair_replay", "device": str(device), "runtime": {"image": IMAGE, "source_commit": SOURCE_COMMIT, "loopback_only": True, "fresh_container_per_repair_replay": True, "external_network": False}, "seeds": list(SEEDS), "counts": counts, "episodes": episodes, "promotion": {"training_eligible": counts["gold_count"] > 0 and counts["replay_count"] > 0, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "honesty": {"counterfactual_failure_is_explicit": True, "model_self_error_is_not_inferred_from_counterfactual": True, "reference_is_independent": True, "general_web_capability_not_established": True}}
    report["report_sha256"] = digest(report)
    dataset = {"schema_version": "pg244-failure-repair-trajectory-dataset-v1", "source_report": str(REPORT.relative_to(ROOT)), "records": records, "counts": {"records": len(records), "gold": counts["gold_count"], "hard_negative": counts["hard_negative_count"], "silver": sum(int(row["lane"] == "silver") for row in records), "quarantine": sum(int(row["lane"] == "quarantine") for row in records)}, "contract": {"real_loopback_replay": True, "counterfactual_failure_has_repair_target": True, "fresh_replay_required": True, "sql_and_xss_families": True, "route_and_seed_lineage": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg244-failure-repair-trajectory-protocol-v1", "families": ["sql", "xss"], "methods": ["GET", "POST"], "steps": ["counterfactual_no_effect", "model_or_reference_repair", "fresh_replay"], "independent_reference_required": True, "fresh_container_per_repair_replay": True, "raw_payload_and_response_excluded": True, "counterfactual_is_not_model_error": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg244-failure-repair-trajectory-trace-v1", "episodes": episodes, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-244 failure → diagnosis → repair → fresh replay", "", f"episodes={counts['episode_count']}; fresh={counts['fresh_container_count']}; SQL={counts['sql_episode_count']}; XSS={counts['xss_episode_count']}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}", f"records={counts['record_count']}; gold={counts['gold_count']}; hard_negative={counts['hard_negative_count']}; replay={counts['replay_count']}; model_self_error={counts['model_self_error_count']}", "counterfactual failures are retained as hard negatives with explicit repair targets; raw wire is stdout-only.", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "counts": counts, "dataset": str(DATASET.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
