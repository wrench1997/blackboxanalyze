"""PG-76: independent unknown-family triplet replay and abstention check."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.detection_payload import build_detection_payload  # noqa: E402
from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step  # noqa: E402


PROTOCOL_ID = "pg-pk-76-independent-unknown-triplet-v1"
IMAGE_SOURCE = "app/pg76_unknown_triplet_fixture.py"
PORT = 8818
SEED = 76101
REPORT_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_report_v1.md"
PG76_FIXTURE = ROOT / "app" / "pg76_unknown_triplet_fixture.py"
PG75_PATH = ROOT / "scripts" / "train_pg75_triplet_delta_ablation.py"
PG71_PATH = ROOT / "scripts" / "train_pg71_trace_abstention_head_v2.py"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg75-triplet-delta" / "trace_triplet_context_head_v4.pt"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _status_class(status: int) -> str:
    return f"{int(status) // 100}xx" if 100 <= int(status) <= 599 else "other"


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"kind": "object", "key_count": len(value), "scalar_count": sum(not isinstance(child, (dict, list)) for child in value.values()), "array_count": sum(isinstance(child, list) for child in value.values())}
    return {"kind": type(value).__name__, "key_count": 0, "scalar_count": 1, "array_count": 0}


def _projection(response: httpx.Response) -> dict[str, Any]:
    body = bytes(response.content)
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
    shape = _shape(value)
    result = {"status_code": int(response.status_code), "status_class": _status_class(response.status_code), "content_type_class": str(response.headers.get("content-type", "")).split(";", 1)[0].casefold(), "body_length_bucket": "0" if not body else "1-255" if len(body) <= 255 else "256-4095" if len(body) <= 4095 else "4096+", "body_sha256": hashlib.sha256(body).hexdigest(), "semantic_body_sha256": sha256_json(shape), "shape": shape, "header_names": sorted({str(key).casefold() for key in response.headers.keys()} & {"content-type", "location", "allow", "x-pg76-variant"}), "state_changed": False, "external_network": False}
    result["projection_sha256"] = sha256_json(result)
    return result


def _oracle(*, positive: bool, signal: str, role: str) -> dict[str, Any]:
    return {"oracle_id": "pg76-typed-workflow-invariant-v1", "modality": "pg76_typed_workflow_invariant_v1", "positive": bool(positive), "positive_authority": True, "confirmed_effect": signal if positive else "none", "candidate_signal": bool(positive), "evaluator_state_hidden": True, "signals": {"role": role, "semantic_contract": signal}, "safety": {"external_network": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False}}


class _FreshTarget:
    def __init__(self, fixture: Any, variant: str, ordinal: int) -> None:
        self.fixture = fixture
        self.variant = variant
        self.ordinal = ordinal
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.client: httpx.Client | None = None
        self.instance_id = _hash(f"pg76-target|{variant}|{ordinal}")[:24]

    def __enter__(self) -> "_FreshTarget":
        self.server = self.fixture.make_server(PORT, self.variant)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                with __import__("socket").create_connection(("127.0.0.1", PORT), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.01)
        else:
            self.close()
            raise RuntimeError("PG-76 fixture did not become ready")
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=3.0, follow_redirects=False)
        return self

    def request(self, route: str, method: str, values: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("PG-76 client is not open")
        if method == "GET":
            response = self.client.get(route, params=values)
        else:
            response = self.client.post(route, data=values)
        projection = _projection(response)
        positive, signal = self.fixture.evaluate(route, values, self.variant)
        return projection, _oracle(positive=positive, signal=signal, role="candidate")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _cases(fixture: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for variant in fixture.VARIANTS:
        for route in fixture.ROUTES:
            for method in ("GET", "POST"):
                if route == "/commit":
                    neutral = {"verb": "wait", "prior": "none", "fresh": "new"}
                    negative = {"verb": "commit", "prior": "pending", "fresh": "new"}
                    candidate = {"verb": "commit", "prior": "verified", "fresh": "new"}
                else:
                    neutral = {"member": "1", "amount": "99"}
                    negative = {"member": "2", "amount": "100"}
                    candidate = {"member": "1", "amount": "100"}
                cases.append({"case_id": f"pg76-{variant}-{route.strip('/').replace('/', '-')}-{method.casefold()}", "variant": variant, "route": route, "method": method, "field_names": sorted(set(neutral) | set(negative) | set(candidate)), "neutral": neutral, "negative": negative, "candidate": candidate})
    return cases


def _step(row: dict[str, Any], index: int, episode_id: str, parent: str | None) -> dict[str, Any]:
    action: dict[str, Any] = {"method": row["method"], "route_template_id": f"pg76-route-{index:03d}", "placement": "form" if row["method"] == "POST" else "query", "encoding_chain": ["identity"], "probe_ref": f"pg76-probe-{index:03d}", "probe_sha256": _hash("pg76-abstract-unknown-triplet"), "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if row["method"] == "POST":
        action["form_field_names"] = row["field_names"][:8]
    oracle = dict(row["positive_oracle"])
    oracle["negative_control_pair_id"] = f"pg76-control-{index:03d}"
    oracle["evaluator_state_hidden"] = True
    step = {"episode_id": episode_id, "step_id": f"pg76-step-{index:03d}", "parent_step_id": parent, "sampling_seed": SEED, "target_instance_id": row["target_instance_id"], "hypothesis": "unknown_workflow_surface_hypothesis", "belief_before": {"unknown_surface": 1.0}, "action_manifest": action, "baseline_projection": row["neutral_response"], "neutral_projection": row["neutral_response"], "negative_probe_projection": row["negative_response"], "response_projection": row["positive_response"], "neutral_oracle_projection": row["neutral_oracle"], "negative_oracle_projection": row["negative_oracle"], "oracle_projection": oracle, "belief_after": {"unknown_surface": 1.0}, "decision": "confirmed_positive", "next_action": "stop_confirmed", "fresh_reset": row["fresh_reset"], "evidence_sha256": row["evidence_sha256"], "dataset_stage": "evaluation_only", "online_weight_update": False, "long_term_memory_write": False}
    echo_body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")}
    step["echo"] = {"sha256": sha256_json(echo_body)}
    return step


def _load_pg75_scorer() -> tuple[Any, Any, Any, Any, float]:
    pg75 = _load(PG75_PATH, "pg76_pg75_candidate")
    v2 = _load(PG71_PATH, "pg76_pg71_features")
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("PG-75 v4 checkpoint is missing")
    checkpoint = __import__("torch").load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    base = v2._load_pg70()
    model = base.TraceDecisionHead(feature_dim=int(checkpoint["feature_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    pg74 = json.loads((ROOT / "research/pg74_causal_triplet_collector_trace_v1.json").read_text(encoding="utf-8"))
    train_rows, _ = pg75._build_rows(v2, [dict(step) for step in pg74["steps"]])
    train_values, _, _ = pg75._normalise(train_rows, train_rows)
    return pg75, v2, model, train_values, float(checkpoint["ood_distance_threshold"])


def _score(pg75: Any, v2: Any, model: Any, reference: Any, threshold: float, step: dict[str, Any]) -> dict[str, Any]:
    import torch

    features = pg75._context_delta_features(v2, step, dict(step["response_projection"]), dict(step["neutral_projection"]))
    # Recreate PG-75 normalization from its accepted training trace.
    pg74 = json.loads((ROOT / "research/pg74_causal_triplet_collector_trace_v1.json").read_text(encoding="utf-8"))
    train_rows, _ = pg75._build_rows(v2, [dict(item) for item in pg74["steps"]])
    _, mean, std = pg75._normalise(train_rows, train_rows)
    values = ((torch.tensor([features], dtype=torch.float32) - mean) / std).clamp(-pg75.CLIP, pg75.CLIP)
    with torch.inference_mode():
        probability = torch.softmax(model(values), dim=-1)[0]
    confidence, predicted = torch.max(probability, dim=0)
    distance = float(torch.cdist(values, reference).min().item())
    raw = pg75.CLASSES[int(predicted)]
    decision = "abstain" if distance >= threshold or float(confidence) < pg75.CONFIDENCE_THRESHOLD else raw
    return {"step_id": step["step_id"], "decision": decision, "raw_prediction": raw, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6), "expected": "abstain"}


def _catalog(fixture: Any, rows: list[dict[str, Any]]) -> None:
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        marker = f"pg76-probe-{index:03d}"
        payload = build_detection_payload(target=f"http://127.0.0.1:{PORT}", method=row["method"], path=row["route"], headers={"accept": "application/json", "x-sift-probe": marker}, marker=marker, probe="workflow_boundary_class", probe_kind="http_canary", form={row["field_names"][0]: marker} if row["method"] == "POST" else {}, expected={"signal": "typed_workflow_invariant", "negative_control": "matched_triplet", "typed_oracle": True})
        samples.append({"sample_id": f"pg76-sample-{index:03d}", "payload": payload, "probe_artifact": {"original": "workflow_boundary_class", "encoding": "abstract_class", "probe_sha256": _hash("workflow_boundary_class")}, "semantic": {"family": "workflow_invariant", "surface": "independent_unknown_workflow", "expected_oracle": "pg69_typed_workflow_invariant_v1", "expected_signal": "typed_workflow_invariant"}, "pair": {"pair_id": f"pg76-pair-{index:03d}", "variant": "abstract_class", "surface_role": "independent_unknown_workflow", "encoding_depth": 0}, "counterfactual": {"kind": "negative_control", "intervention": "matched_triplet", "source_sample_id": f"pg76-sample-{index:03d}"}, "replay": {"target": f"http://127.0.0.1:{PORT}", "method": row["method"], "path": row["route"], "transport": "loopback", "fresh_reset": row["fresh_reset"]}, "response_projection": row["positive_response"], "oracle_projection": row["positive_oracle"], "evidence": {"adapter_evidence_sha256": row["evidence_sha256"], "control_evidence_sha256": row["negative_control"]["control_evidence_sha256"]}, "rule_ir": {"op": "and", "args": [{"op": "eq", "left": {"op": "field", "path": "oracle.positive"}, "right": {"op": "const", "value": True}}]}, "rule_ir_result": True, "evaluator_state_visible": False})
    write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg76-independent-unknown-triplet-evaluation-only", "sources": [{"provenance": {"source_id": "pg76-independent-workflow", "source_type": "in_repo_synthetic", "origin": IMAGE_SOURCE, "license": "in_repo_synthetic", "authorization": "workspace_local_only", "scope": [f"http://127.0.0.1:{PORT}"], "captured_at": captured_at, "authorized_for": ["training", "local_replay", "holdout_evaluation"], "external_network": False, "evaluator_state_visible": False}, "samples": samples}]})


def run() -> dict[str, Any]:
    fixture = _load(PG76_FIXTURE, "pg76_fixture_runtime")
    rows: list[dict[str, Any]] = []
    cases = _cases(fixture)
    for ordinal, case in enumerate(cases):
        with _FreshTarget(fixture, case["variant"], ordinal) as target:
            neutral_response, neutral_oracle = target.request(case["route"], case["method"], case["neutral"])
            negative_response, negative_oracle = target.request(case["route"], case["method"], case["negative"])
            positive_response, positive_oracle = target.request(case["route"], case["method"], case["candidate"])
            reset = {"kind": "pg76-fresh-triplet-target", "reset_id": f"pg76-reset-{ordinal:03d}", "target_instance_id": target.instance_id, "state_epoch": _hash(f"pg76-state|{target.instance_id}"), "reset_adapter_sha256": _hash("pg76-triplet-reset"), "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "state_change_allowed": False, "external_network": False, "read_only_round": True}
        row = {"case_id": case["case_id"], "variant": case["variant"], "route": case["route"], "method": case["method"], "field_names": case["field_names"], "target_instance_id": target.instance_id, "fresh_reset": reset, "neutral_response": neutral_response, "neutral_oracle": neutral_oracle, "negative_response": negative_response, "negative_oracle": negative_oracle, "positive_response": positive_response, "positive_oracle": positive_oracle, "raw_payload_stored": False, "raw_response_body_stored": False}
        row["negative_control"] = {"matched": True, "control_case_id": case["case_id"], "control_evidence_sha256": sha256_json({"neutral": neutral_response, "negative": negative_response, "oracle": negative_oracle})}
        row["evidence_sha256"] = sha256_json({"case": case["case_id"], "neutral": neutral_response, "negative": negative_response, "positive": positive_response, "neutral_oracle": neutral_oracle, "negative_oracle": negative_oracle, "positive_oracle": positive_oracle, "reset": reset})
        rows.append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["variant"]].append(_step(row, index, f"pg76-episode-{row['variant']}", None if len(grouped[row["variant"]]) == 0 else grouped[row["variant"]][-1]["step_id"]))
    all_steps: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant, raw_steps in grouped.items():
        normalized: list[dict[str, Any]] = []
        parent: str | None = None
        for raw in raw_steps:
            raw["parent_step_id"] = parent
            body = {key: raw[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")}
            raw["echo"] = {"sha256": sha256_json(body)}
            try:
                step = validate_trace_step(raw)
            except ValueError as exc:
                failures.append({"step_id": raw["step_id"], "error_type": type(exc).__name__})
                step = raw
            normalized.append(step)
            all_steps.append(step)
            parent = step["step_id"]
        episodes.append({"episode_id": f"pg76-episode-{variant}", "variant": variant, "steps": normalized, "validation": evaluate_episode(normalized) if not failures else {"status": "trace_only", "reasons": ["step_validation_failure"]}})
    trace = {"schema_version": "sift-pg76-independent-unknown-triplet-trace-v1", "protocol_id": PROTOCOL_ID, "evaluation_only": True, "training_eligible": False, "steps": all_steps, "episodes": episodes, "episode_count": len(episodes), "accepted_episode_count": sum(int(item["validation"].get("status") == "accepted_evaluation") for item in episodes), "validation_failures": failures, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "model_input_family_leakage": False}
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _catalog(fixture, rows)
    pg75, v2, model, reference, threshold = _load_pg75_scorer()
    details = [_score(pg75, v2, model, reference, threshold, step) for step in all_steps]
    model_misname = sum(int(item["decision"] != "abstain") for item in details)
    checks = {"triplet_complete_per_case": len(rows) == 12, "typed_positive_per_case": sum(int(bool(row["positive_oracle"].get("positive"))) for row in rows) == len(rows), "typed_negative_oracle_per_case": sum(int(not bool(row["negative_oracle"].get("positive"))) + int(not bool(row["neutral_oracle"].get("positive"))) for row in rows) == len(rows) * 2, "fresh_target_per_case": len({row["target_instance_id"] for row in rows}) == len(rows), "get_post_covered": {"GET", "POST"}.issubset({row["method"] for row in rows}), "trace_episodes_accepted": trace["episode_count"] == trace["accepted_episode_count"] and not failures, "model_unknown_strict_abstain": model_misname == 0, "no_raw_persistence": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in rows)}
    status = "passed" if all(checks.values()) else "blocked"
    report = {"protocol_id": PROTOCOL_ID, "schema_version": "sift-pg76-independent-unknown-triplet-report-v1", "status": "completed_evaluation", "source": {"fixture": IMAGE_SOURCE, "fixture_source_sha256": fixture.source_sha256(), "independent_implementation_count": 1, "family_outside_training_registry": True, "variant_count": len(fixture.VARIANTS), "candidate_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "candidate_checkpoint_sha256": __import__("hashlib").sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()}, "scope": {"case_count": len(rows), "families": [fixture.FAMILY], "methods": ["GET", "POST"], "loopback_only": True, "external_network": False, "raw_payloads_stored": False, "raw_response_bodies_stored": False}, "metrics": {"triplet_case_count": len(rows), "typed_positive_count": sum(int(bool(row["positive_oracle"].get("positive"))) for row in rows), "typed_negative_oracle_count": sum(int(not bool(row["negative_oracle"].get("positive"))) + int(not bool(row["neutral_oracle"].get("positive"))) for row in rows), "unique_target_instance_count": len({row["target_instance_id"] for row in rows}), "trace_episode_count": trace["episode_count"], "trace_accepted_episode_count": trace["accepted_episode_count"], "model_unknown_misname_count": model_misname, "model_unknown_strict_abstain": model_misname == 0, "model_ood_threshold": threshold, "model_max_confidence": max((item["confidence"] for item in details), default=0.0), "model_min_ood_distance": min((item["ood_distance"] for item in details), default=0.0)}, "model_details": details, "hard_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "training_catalog_generated": False, "status": "unknown_triplet_evaluation_only", "reason": "unknown family is a holdout and cannot promote the candidate"}, "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT))}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "sift-pg76-independent-unknown-triplet-protocol-v1", "target_contract": {"fixture": IMAGE_SOURCE, "loopback_port": PORT, "fresh_target_per_triplet": True, "external_network": False, "state_change_allowed": False}, "holdout_contract": {"family": "workflow_invariant", "outside_training_registry": True, "independent_implementation": True, "model_must_not_see_family_or_oracle": True}, "triplet_contract": {"neutral_projection": True, "negative_probe_projection": True, "positive_probe_projection": True, "typed_negative_oracles": True, "typed_positive_oracles": True, "raw_persistence_forbidden": True}, "required_gates": {"triplet_complete_per_case": True, "typed_positive_per_case": True, "typed_negative_oracle_per_case": True, "fresh_target_per_case": True, "get_post_covered": True, "trace_episodes_accepted": True, "model_unknown_strict_abstain": True, "no_raw_persistence": True}, "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG77 cross-implementation known-family triplet replay and candidate family-heldout audit"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-76 独立未知族 triplet 复放\n\n" + f"triplets={len(rows)}；typed positive={report['metrics']['typed_positive_count']}；typed negative={report['metrics']['typed_negative_oracle_count']}；model misname={model_misname}；strict abstain={report['metrics']['model_unknown_strict_abstain']}。\n\n硬门：`{status}`；training_allowed=`false`；memory_promotion_allowed=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["hard_gate"]["status"], "triplet_case_count": result["metrics"]["triplet_case_count"], "model_unknown_misname_count": result["metrics"]["model_unknown_misname_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
