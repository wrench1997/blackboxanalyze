"""Run the current PG-53 Rule IR candidate against PG-42's independent target.

PG-54 is an evaluation-only bridge.  It starts the PG-42 loopback fixture,
performs fresh GET/POST control, screen and confirm requests, and converts the
bounded response projection into the *already reviewed* PG-53 feature slots.
The model is never retrained here.  PG-42 adds a different envelope/layout and
the ``template_injection`` family, which is intentionally outside the current
Rule IR class set and must therefore be handled by abstention rather than a
forced known-family guess.

Only status/shape hashes, anonymous value-type geometry, bounded observations
and typed evaluator attestations are persisted.  Request values and response
bodies stay in process memory and are never written to disk.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg42_independent_semantic_fixture import (  # noqa: E402
    IMPLEMENTATIONS,
    LAYOUTS,
    SURFACE_SPECS,
    VARIANTS,
    make_pg42_server,
)
from app.pg53_cross_source_oracle import generic_effect_geometry, response_projection, sha256_json, surface_observation  # noqa: E402
from app.pg53_rule_ir_candidate import PG53_MODEL_FAMILIES, PG53RuleIRCandidate  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402
from app.web_feature_funnel import audit_feature_funnel, build_feature_dataset, build_feature_row, review_feature_funnel  # noqa: E402


PROTOCOL_ID = "pg-pk-54-pg42-rule-ir-ood-v1"
SCHEMA_VERSION = "pg54-pg42-rule-ir-ood-report-v1"
TRACE_SCHEMA = "pg54-pg42-rule-ir-ood-trace-v1"
TRACE_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_report_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_protocol_v1.json"
FUNNEL_DATASET_PATH = ROOT / "research" / "pg53_web_feature_funnel_dataset_v1.json"
PG53_SOURCE_REPORT_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg53-rule-ir-candidate" / "decoder.pt"
SEEDS = (401, 409, 419)
TARGET_PORT = 32054
MAX_SERVER_START_SECONDS = 5.0
KNOWN_FAMILIES = tuple(PG53_MODEL_FAMILIES[:-1])
UNKNOWN_FAMILY = "template_injection"
NEGATIVE_FAMILY = "ordinary_response"
SURFACES = tuple(sorted(SURFACE_SPECS))
METHODS = ("GET", "POST")


def _worker_port() -> int:
    local = getattr(_worker_port, "local", None)
    if local is None:
        local = threading.local()
        _worker_port.local = local
    if not hasattr(local, "port"):
        lock = getattr(_worker_port, "lock", None)
        if lock is None:
            lock = threading.Lock()
            _worker_port.lock = lock
        with lock:
            index = int(getattr(_worker_port, "counter", 0))
            _worker_port.counter = index + 1
        local.port = TARGET_PORT + index
    return int(local.port)


class _FreshTarget:
    def __init__(self, implementation: str) -> None:
        self.implementation = implementation
        self.port = _worker_port()
        self.server = make_pg42_server(self.port, implementation)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.005}, daemon=True)
        self.client: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        self.thread.start()
        deadline = time.monotonic() + MAX_SERVER_START_SECONDS
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.005)
        else:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)
            raise RuntimeError(f"PG-54 target failed to start: {self.implementation}")
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}",
            timeout=3.0,
            follow_redirects=False,
            headers={"accept": "application/json"},
        )
        return self.client

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.client is not None:
            self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _route(implementation: str, surface: str) -> str:
    return f"{LAYOUTS[implementation]['prefix']}/{surface}"


def _call(
    client: httpx.Client,
    implementation: str,
    surface: str,
    method: str,
    variant: str,
    phase: str,
    *,
    candidate: bool,
) -> httpx.Response:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    values = {
        layout["surface_key"]: surface,
        layout["probe_key"]: spec["probe"] if candidate else "ordinary-observation",
        layout["variant_key"]: variant,
        layout["phase_key"]: phase,
    }
    if method == "GET":
        return client.get(_route(implementation, surface), params=values)
    content_type = layout["post_content_type"]
    headers = {"content-type": content_type}
    if content_type == "application/json":
        content = json.dumps(values, separators=(",", ":")).encode("utf-8")
    else:
        content = "&".join(f"{key}={value}" for key, value in values.items()).encode("utf-8")
    return client.post(_route(implementation, surface), content=content, headers=headers)


def _typed_oracle(
    *,
    source_id: str,
    family: str,
    phase: str,
    response: dict[str, Any],
    body: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate PG-42's bounded effect contract after the probe.

    The top-level ``effect``/``proof`` fields are the evaluator contract.  No
    field names or typed values from the body enter the model feature vector.
    ``template_injection`` is a valid evaluator family but is deliberately not
    a PG-53 model class.
    """

    effect = body.get("effect") if isinstance(body.get("effect"), dict) else {}
    proof = body.get("proof") if isinstance(body.get("proof"), dict) else {}
    safety = {
        "external_network": bool(body.get("external_network", False)),
        "script_execution": bool(body.get("script_execution", False)),
        "database_write": bool(body.get("database_write", False)),
        "state_mutated": bool(body.get("state_mutated", False)),
        "credentials_accessed": bool(body.get("credentials_accessed", False)),
    }
    effect_confirmed = bool(effect.get("confirmed", False))
    proof_bounded = bool(proof.get("bounded", False))
    proof_read_only = bool(proof.get("read_only", False))
    positive = (
        phase == "confirm"
        and family != NEGATIVE_FAMILY
        and effect_confirmed
        and proof_bounded
        and proof_read_only
        and response.get("status_class") == "2xx"
        and not any(safety.values())
    )
    signals = {
        "effect_confirmed": effect_confirmed,
        "proof_bounded": proof_bounded,
        "proof_read_only": proof_read_only,
        "scope_present": bool(str(effect.get("scope", ""))),
    }
    oracle = {
        "oracle_id": f"pg54-{family}-typed-effect-v1",
        "oracle_contract": "pg42-independent-semantic-typed-oracle-v1",
        "source_implementation": source_id,
        "family": family,
        "phase": phase,
        "positive": bool(positive),
        "positive_authority": bool(positive),
        "modality": "bounded_effect_contract" if positive else "negative_control_or_screen",
        "signals": signals,
        "safety": safety,
        "response_projection_sha256": response.get("projection_sha256", ""),
    }
    oracle["evidence_projection_sha256"] = sha256_json(oracle)
    return oracle


def _component(
    response: httpx.Response,
    *,
    source_id: str,
    family: str,
    phase: str,
) -> dict[str, Any]:
    # The parsed body exists only in this function's memory.  The persisted
    # record contains its safe projection and anonymous geometry only.
    body = response.json()
    projection = response_projection(response)
    oracle = _typed_oracle(source_id=source_id, family=family, phase=phase, response=projection, body=body)
    return {
        "response": projection,
        "surface_observation": surface_observation(body),
        "generic_effect_geometry": generic_effect_geometry(body),
        "oracle": oracle,
    }


def _probe_descriptor(implementation: str, surface: str, method: str, phase: str, role: str) -> dict[str, Any]:
    layout = LAYOUTS[implementation]
    spec = SURFACE_SPECS[surface]
    descriptor = {
        "probe_kind": "abstract_channel_class",
        "role": role,
        "phase": phase,
        "method": method,
        "placement": "query" if method == "GET" else "body",
        "encoding_chain": ["identity"],
        "field_name_sha256": sha256_json(sorted(layout.values())),
        "semantic_probe_sha256": hashlib.sha256(f"pg54:{spec['semantic']}:{role}:{phase}".encode()).hexdigest(),
        "raw_probe_stored": False,
    }
    descriptor["descriptor_sha256"] = sha256_json(descriptor)
    return descriptor


def _reset(implementation: str, surface: str, variant: str, seed: int, method: str) -> dict[str, Any]:
    value = {
        "reset_id": f"pg54-reset-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}",
        "kind": "fresh_pg42_independent_http_server",
        "target_instance_id": f"pg54-target-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}",
        "state_epoch": f"pg54-epoch-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}",
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "raw_body_stored": False,
    }
    value["reset_sha256"] = sha256_json(value)
    return value


def _run_case(implementation: str, surface: str, variant: str, seed: int, method: str) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface]
    family = spec["family"]
    source_id = f"pg42-{implementation}-{variant}"
    with _FreshTarget(implementation) as client:
        control_response = _call(client, implementation, surface, method, variant, "confirm", candidate=False)
        screen_response = _call(client, implementation, surface, method, variant, "screen", candidate=True)
        candidate_response = _call(client, implementation, surface, method, variant, "confirm", candidate=True)
        control = _component(control_response, source_id=source_id, family=family, phase="confirm")
        screen = _component(screen_response, source_id=source_id, family=family, phase="screen")
        candidate = _component(candidate_response, source_id=source_id, family=family, phase="confirm")
    reset = _reset(implementation, surface, variant, seed, method)
    negative_control = {
        "matched": bool(control["oracle"]["positive"] is False and control["response"]["status_class"] == "2xx"),
        "control_oracle_positive": bool(control["oracle"]["positive"]),
        "candidate_oracle_positive": bool(candidate["oracle"]["positive"]),
        "evidence_sha256": sha256_json({"control": control["oracle"], "candidate": candidate["oracle"], "reset": reset}),
    }
    decision = "confirmed_positive" if candidate["oracle"]["positive"] else "confirmed_negative"
    row = {
        "schema_version": TRACE_SCHEMA,
        "sample_id": f"pg54-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}",
        "source_id": source_id,
        "implementation": implementation,
        "variant": variant,
        "surface": surface,
        "sampling_seed": int(seed),
        "method": method,
        "family": family,
        "decision": decision,
        "control": control,
        "screen": screen,
        "candidate": candidate,
        "negative_control": negative_control,
        "fresh_reset": reset,
        "probe_descriptors": {
            "control": _probe_descriptor(implementation, surface, method, "confirm", "control"),
            "screen": _probe_descriptor(implementation, surface, method, "screen", "candidate"),
            "candidate": _probe_descriptor(implementation, surface, method, "confirm", "candidate"),
        },
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }
    row["evidence_sha256"] = sha256_json({
        "sample_id": row["sample_id"],
        "candidate": candidate["oracle"],
        "control": control["oracle"],
        "screen": screen["oracle"],
        "negative_control": negative_control,
        "fresh_reset": reset,
    })
    return row


def _all_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for implementation in IMPLEMENTATIONS:
        for surface in SURFACES:
            for variant in VARIANTS:
                for seed in SEEDS:
                    for method in METHODS:
                        rows.append(_run_case(implementation, surface, variant, int(seed), method))
    return rows


def _feature_vector(feature_row: dict[str, Any], selected_features: list[str]) -> torch.Tensor:
    values = feature_row["model_features"]
    vector = [0.0] * FEATURE_DIM
    for offset, name in enumerate(selected_features):
        value = values.get(name, 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        vector[8 + offset] = max(-1.0, min(1.0, numeric / 32.0 if abs(numeric) > 1.0 else numeric))
    return torch.tensor(vector, dtype=torch.float32)


def _load_model(selected_features: list[str]) -> tuple[PG53RuleIRCandidate, torch.Tensor, torch.Tensor, float, float, torch.device, str]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = PG53RuleIRCandidate()
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model, mean, std, float(checkpoint["abstain_threshold"]), float(checkpoint["margin_threshold"]), device, hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()


def _reference_vectors(selected_features: list[str], mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Build source-train/dev vectors for a label-calibrated density gate.

    The gate uses only PG-53 visible projections.  PG-54 labels never enter
    the threshold selection; they are reserved for the independent holdout.
    """

    funnel = json.loads(FUNNEL_DATASET_PATH.read_text(encoding="utf-8"))
    source = json.loads(PG53_SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    feature_map = {str(row["sample_id"]): row for row in funnel.get("rows", [])}
    source_rows = list(source.get("rows", []))
    train_rows = [row for row in source_rows if row.get("implementation") == "pg35" and int(row.get("sampling_seed", 0)) in {5301, 5307}]
    dev_rows = [row for row in source_rows if row.get("implementation") == "pg35" and int(row.get("sampling_seed", 0)) == 5311]
    train_raw = torch.stack([_feature_vector(feature_map[str(row["sample_id"])], selected_features) for row in train_rows])
    dev_raw = torch.stack([_feature_vector(feature_map[str(row["sample_id"])], selected_features) for row in dev_rows])
    return (train_raw - mean) / std, (dev_raw - mean) / std, dev_rows


def _model_emits(output: dict[str, Any], confidence_threshold: float, margin_threshold: float) -> bool:
    return (
        output["candidate_family"] != NEGATIVE_FAMILY
        and float(output["confidence"]) >= float(confidence_threshold)
        and float(output["margin"]) >= float(margin_threshold)
    )


def _fit_density_gate(
    model: PG53RuleIRCandidate,
    train_vectors: torch.Tensor,
    dev_vectors: torch.Tensor,
    dev_rows: list[dict[str, Any]],
    confidence_threshold: float,
    margin_threshold: float,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    with torch.inference_mode():
        outputs = model.decode(dev_vectors.to(device), abstain_threshold=0.0, margin_threshold=0.0)
    distances = torch.cdist(dev_vectors.cpu(), train_vectors.cpu()).min(dim=1).values
    best_threshold = 0.0
    best_hits = -1
    best_emitted = 0
    for threshold in sorted({float(value) for value in distances.tolist()}):
        accepted = [
            index
            for index, output in enumerate(outputs)
            if _model_emits(output, confidence_threshold, margin_threshold) and float(distances[index]) <= threshold
        ]
        false_accepts = sum(
            int(not (dev_rows[index]["decision"] == "confirmed_positive" and outputs[index]["candidate_family"] == dev_rows[index]["family"]))
            for index in accepted
        )
        hits = sum(
            int(dev_rows[index]["decision"] == "confirmed_positive" and outputs[index]["candidate_family"] == dev_rows[index]["family"])
            for index in accepted
        )
        if false_accepts == 0 and (hits > best_hits or (hits == best_hits and threshold < best_threshold)):
            best_threshold = threshold
            best_hits = hits
            best_emitted = len(accepted)
    return best_threshold, {
        "reference_train_rows": int(train_vectors.shape[0]),
        "calibration_dev_rows": len(dev_rows),
        "calibrated_threshold": round(float(best_threshold), 6),
        "calibrated_dev_hits": max(0, int(best_hits)),
        "calibrated_dev_emitted": int(best_emitted),
        "calibrated_dev_false_accepts": 0,
        "calibration_source": "pg53-pg35-dev-only",
    }


def _predictions(rows: list[dict[str, Any]], selected_features: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_rows = [build_feature_row(row) for row in rows]
    model, mean, std, threshold, margin_threshold, device, checkpoint_sha256 = _load_model(selected_features)
    raw = torch.stack([_feature_vector(feature_row, selected_features) for feature_row in feature_rows])
    normalised = (raw - mean) / std
    train_vectors, dev_vectors, dev_rows = _reference_vectors(selected_features, mean, std)
    density_threshold, density_calibration = _fit_density_gate(
        model,
        train_vectors,
        dev_vectors,
        dev_rows,
        threshold,
        margin_threshold,
        device,
    )
    distances = torch.cdist(normalised.cpu(), train_vectors.cpu()).min(dim=1).values
    with torch.inference_mode():
        outputs = model.decode(normalised.to(device), abstain_threshold=0.0, margin_threshold=0.0)
    predictions: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        model_emitted = _model_emits(output, threshold, margin_threshold)
        emitted = model_emitted and float(distances[len(predictions)]) <= density_threshold
        predictions.append({
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "implementation": row["implementation"],
            "variant": row["variant"],
            "sampling_seed": row["sampling_seed"],
            "method": row["method"],
            "surface": row["surface"],
            "expected_family": row["family"],
            "positive": row["decision"] == "confirmed_positive",
            "candidate_family": output["candidate_family"],
            "confidence": output["confidence"],
            "margin": output["margin"],
            "model_emitted_before_density_gate": bool(model_emitted),
            "density_distance": round(float(distances[len(predictions)]), 6),
            "density_threshold": round(float(density_threshold), 6),
            "emitted": bool(emitted),
            "checkpoint_sha256": checkpoint_sha256,
        })
    density_calibration["holdout_distance_min"] = round(float(distances.min()), 6)
    density_calibration["holdout_distance_max"] = round(float(distances.max()), 6)
    density_calibration["gate_enabled"] = True
    return predictions, density_calibration


def _metrics(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(predictions)
    known_positive = [item for item in rows if item["positive"] and item["expected_family"] in KNOWN_FAMILIES]
    unknown_positive = [item for item in rows if item["positive"] and item["expected_family"] == UNKNOWN_FAMILY]
    negatives = [item for item in rows if not item["positive"]]
    known_hits = [item for item in known_positive if item["emitted"] and item["candidate_family"] == item["expected_family"]]
    known_wrong = [item for item in known_positive if item["emitted"] and item["candidate_family"] != item["expected_family"]]
    unknown_misname = [item for item in unknown_positive if item["emitted"]]
    negative_false_accept = [item for item in negatives if item["emitted"]]
    return {
        "count": len(rows),
        "positive_count": sum(int(item["positive"]) for item in rows),
        "negative_count": len(negatives),
        "emitted_count": sum(int(item["emitted"]) for item in rows),
        "known_positive_count": len(known_positive),
        "known_family_recall": round(len(known_hits) / max(len(known_positive), 1), 6),
        "known_wrong_family_count": len(known_wrong),
        "unknown_positive_count": len(unknown_positive),
        "unknown_misname_count": len(unknown_misname),
        "unknown_not_abstain_count": len(unknown_misname),
        "unknown_strict_abstain": len(unknown_misname) == 0,
        "negative_effect_false_accept_count": len(negative_false_accept),
        "negative_effect_false_accept_rate": round(len(negative_false_accept) / max(len(negatives), 1), 6),
        "abstain_rate": round(sum(int(not item["emitted"]) for item in rows) / max(len(rows), 1), 6),
    }


def _group_metrics(predictions: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        groups[str(item[key])].append(item)
    return {name: _metrics(values) for name, values in sorted(groups.items())}


def _feature_review(rows: list[dict[str, Any]], selected_features: list[str]) -> dict[str, Any]:
    dataset = build_feature_dataset(rows)
    dataset["dataset_id"] = "pg54-pg42-rule-ir-ood-feature-audit"
    audit = audit_feature_funnel(dataset)
    review = review_feature_funnel(audit, review_scope="PG-54 PG-42 safe visible response projections only")
    accepted = set(str(name) for name in audit["accepted_features"])
    selected = set(str(name) for name in selected_features)
    selected_revalidated = selected <= accepted
    return {
        "dataset_id": dataset["dataset_id"],
        "row_count": len(rows),
        "stage_counts": audit["stage_counts"],
        "accepted_features": audit["accepted_features"],
        "source_count": audit["source_count"],
        "family_count": audit["family_count"],
        "pg53_selected_features": list(selected_features),
        "selected_features_revalidated_on_pg54": selected_revalidated,
        "selected_features_missing_on_pg54": sorted(selected - accepted),
        "pg54_only_accepted_features": sorted(accepted - selected),
        "feature_transfer_gate": "passed" if selected_revalidated and review["passed"] else "blocked",
        "review": review,
        "training_eligible": False,
        "long_term_memory_write": False,
    }


def _protocol_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["decision"] == "confirmed_positive"]
    negatives = [row for row in rows if row["decision"] == "confirmed_negative"]
    return {
        "case_count": len(rows),
        "confirmed_positive_count": len(positives),
        "confirmed_negative_count": len(negatives),
        "get_post_covered": {method: sum(int(row["method"] == method) for row in rows) for method in METHODS},
        "implementation_count": len({row["implementation"] for row in rows}),
        "variant_count": len({row["variant"] for row in rows}),
        "seed_count": len({row["sampling_seed"] for row in rows}),
        "negative_control_pass_count": sum(int(row["negative_control"]["matched"]) for row in rows),
        "fresh_reset_count": sum(int(row["fresh_reset"]["fresh_target"] and row["fresh_reset"]["completed"]) for row in rows),
    }


def main() -> int:
    funnel = json.loads(FUNNEL_DATASET_PATH.read_text(encoding="utf-8"))
    review_decision = str(funnel.get("review_decision", ""))
    if review_decision != "approved_for_downstream_ood_experiment":
        raise RuntimeError("PG-53 Codex feature review is not approved")
    selected_features = [str(name) for name in funnel.get("accepted_features", [])]
    if not selected_features:
        raise RuntimeError("PG-53 feature funnel accepted no features")

    rows = _all_rows()
    summary = _protocol_summary(rows)
    feature_review = _feature_review(rows, selected_features)
    predictions, density_gate = _predictions(rows, selected_features)
    metrics = _metrics(predictions)
    split_metrics = {
        "all": metrics,
        "implementation_cobalt": _metrics([item for item in predictions if item["implementation"] == "cobalt"]),
        "implementation_quartz": _metrics([item for item in predictions if item["implementation"] == "quartz"]),
        "variant_framed": _metrics([item for item in predictions if item["variant"] == "framed"]),
        "seed_419": _metrics([item for item in predictions if int(item["sampling_seed"]) == 419]),
        "known_families": _metrics([item for item in predictions if item["expected_family"] in KNOWN_FAMILIES]),
        "unknown_template_family": _metrics([item for item in predictions if item["expected_family"] == UNKNOWN_FAMILY]),
        "negative_control": _metrics([item for item in predictions if item["expected_family"] == NEGATIVE_FAMILY]),
    }
    trace = {
        "schema_version": TRACE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "evaluation_only": True,
        "training_eligible": False,
        "long_term_memory_write": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "rows": rows,
        "summary": summary,
        "feature_review_evidence_sha256": funnel.get("review_evidence_sha256", ""),
    }
    trace["trace_sha256"] = hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace_sha256 = hashlib.sha256(TRACE_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only",
        "source": {
            "fixture": "app.pg42_independent_semantic_fixture",
            "implementations": list(IMPLEMENTATIONS),
            "variants": list(VARIANTS),
            "surface_count": len(SURFACES),
            "families": sorted({row["family"] for row in rows}),
            "methods": list(METHODS),
            "sampling_seeds": list(SEEDS),
            "loopback_only": True,
            "external_network": False,
        },
        "summary": summary,
        "feature_review": feature_review,
        "model": {
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": predictions[0]["checkpoint_sha256"] if predictions else "",
            "selected_features": selected_features,
            "feature_funnel_review_evidence_sha256": funnel.get("review_evidence_sha256", ""),
            "oracle_visible_before_probe": False,
            "typed_oracle_in_features": False,
            "family_label_in_features": False,
            "raw_request_response_in_features": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "density_gate": density_gate,
        },
        "splits": split_metrics,
        "unknown_family_policy": {
            "family": UNKNOWN_FAMILY,
            "model_class_present": False,
            "must_abstain": True,
            "unknown_misname_count": metrics["unknown_misname_count"],
            "unknown_not_abstain_count": metrics["unknown_not_abstain_count"],
            "strict_abstain": metrics["unknown_strict_abstain"],
        },
        "trace": {
            "path": str(TRACE_PATH.relative_to(ROOT)),
            "sha256": trace_sha256,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_claim_allowed": False,
            "status": "quarantined_independent_rule_ir_ood",
            "reason": "PG42_is_an_independent_envelope_and_template_family_holdout; feature_transfer_gate_and_capability_metrics_are_diagnostic_only",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PG-54 PG-42 独立实现 Rule IR 族外验证",
        "",
        "只复用 PG-53 Codex 审核后的四个匿名 geometry 特征；不训练、不写长期记忆。PG-42 的 template_injection 不在模型类别中，必须安全 abstain。",
        "",
        f"样本：`{summary['case_count']}`；权威正例：`{summary['confirmed_positive_count']}`；阴性：`{summary['confirmed_negative_count']}`；GET/POST：`{summary['get_post_covered']}`。",
        "",
        "| split | known recall | unknown misname | negative false accept | abstain |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("all", "implementation_cobalt", "implementation_quartz", "variant_framed", "unknown_template_family", "negative_control"):
        item = split_metrics[name]
        lines.append(f"| {name} | {item['known_family_recall']:.3f} | {item['unknown_misname_count']} | {item['negative_effect_false_accept_count']} | {item['abstain_rate']:.3f} |")
    lines.extend([
        "",
        f"PG-54 特征复审：`{feature_review['review']['decision']}`；PG-53 特征迁移门：`{feature_review['feature_transfer_gate']}`；审核证据哈希：`{funnel.get('review_evidence_sha256', '')}`。",
        f"密度 abstain 门：`{report['model']['density_gate']['calibrated_threshold']}`（只用 PG-53 dev 校准，PG-54 不参与阈值选择）。",
        f"未知族严格 abstain：`{metrics['unknown_strict_abstain']}`；训练/长期记忆晋升：`False/False`。",
    ])
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "summary": summary,
        "feature_review": {
            "decision": feature_review["review"]["decision"],
            "accepted_features": feature_review["accepted_features"],
            "selected_features_revalidated_on_pg54": feature_review["selected_features_revalidated_on_pg54"],
            "feature_transfer_gate": feature_review["feature_transfer_gate"],
        },
        "metrics": metrics,
        "unknown_family": report["unknown_family_policy"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
