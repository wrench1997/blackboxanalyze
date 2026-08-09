"""PG-250: drive the real Pikachu payload loop with the PG-249 action adapter.

The existing PG-208 runner already owns the pinned Docker image, crawl-derived
GET/POST field catalog, safe candidate generator, browser/static DOM oracle,
fresh reset, and evidence projection.  This wrapper replaces only its
decision gate with the PG-249 4096 adapter and adds an independent catalog
candidate comparison.  Runtime wires are printed ephemerally for the
researcher; reports retain only bounded projections and hashes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import torch

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


PG208 = _load("run_pg208_pikachu_typed_payload_loop.py")
PG249 = _load("run_pg249_pikachu_route_seed_capacity_training.py")
PG237 = PG249.PG237
PG231 = PG237.PG231
from app.pg252_probe_gate import build_probe_gate_record  # noqa: E402
import app.pg198_payload_grounding as PG198  # noqa: E402


RESEARCH = ROOT / "research"
PG249_REPORT = RESEARCH / "pg249_pikachu_route_seed_capacity_training_report_v1.json"
REPORT = RESEARCH / "pg250_pikachu_pg249_payload_replay_report_v1.json"
TRACE = RESEARCH / "pg250_pikachu_pg249_payload_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg250_pikachu_pg249_payload_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg250_pikachu_pg249_payload_replay_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg250-pikachu-pg249-payload-replay-v1"

ACTION_REPORT = RESEARCH / "pg252_probe_gate_capacity_training_report_v1.json"
# Resolved from the final PG-252 report at runtime so the replay cannot drift
# from the adapter that the independent judge actually selected.
ACTION_ARTIFACT: Path | None = None
ORIGINAL_LOAD_MODEL = PG208._load_model
ORIGINAL_MODEL_DECISION = PG208._model_decision
ORIGINAL_SEND_AI = PG208._send_ai_candidate
WIRE_LOG: list[dict[str, Any]] = []


class _HybridModel:
    def __init__(self, legacy: Any, capacity: Any, capacity_vocab: dict[str, int], input_vocab: dict[str, int], base: Any, device: torch.device) -> None:
        self.legacy = legacy
        self.capacity = capacity
        self.capacity_vocab = capacity_vocab
        self.input_vocab = input_vocab
        self.base = base
        self.device = device


def _load_capacity(device: torch.device) -> tuple[Any, dict[str, int], dict[str, int], Any]:
    global ACTION_ARTIFACT
    PG249._configure()
    base, input_vocab = PG249.PG248.PG247._load_base(device)
    action_report = json.loads(ACTION_REPORT.read_text(encoding="utf-8-sig"))
    ACTION_ARTIFACT = ROOT / str(action_report["selected"]["artifact"])
    artifact = torch.load(ACTION_ARTIFACT, map_location="cpu", weights_only=False)
    target_vocab = {str(key): int(value) for key, value in artifact["token_vocabulary"].items()}
    with torch.no_grad():
        # The real loop uses one live observation at a time, so no context is
        # cached here.  The frozen body remains read-only.
        probe = [build_probe_gate_record({"source": "pg250_live_context", "seed": 25001, "surface_class": "dom_surface", "method": "GET", "field_count": 1, "oracle_available": True, "fresh_reset_ok": True, "reset_completed": True})]
        encoded = PG237._encode(probe, input_vocab, target_vocab, device)
        context = base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
    model = PG237.FrozenXXLFailurePolicy(d_model=int(context.shape[-1]), hidden_dim=int(artifact["hidden_dim"]), vocab_size=len(target_vocab)).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model, target_vocab, input_vocab, base


def _capacity_row(route: Mapping[str, Any], projection: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    typed = bool(route.get("typed_available"))
    status = str(projection.get("status_class", "2xx"))
    return {
        "source": "pg250_live_pikachu_loop",
        "seed": 25001,
        "surface_class": "dom_surface" if typed else "sql_surface",
        "method": str(route.get("method", "GET")).upper(),
        "status_class": status if status in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"} else "unknown",
        "field_count": len(route.get("fields") or []),
        "oracle_available": typed,
        "fresh_reset_ok": True,
        "reset_completed": True,
        "binding_valid": bool(packet.get("field_tokens")),
        "evidence_hash": str(route.get("crawl_evidence_sha256", "")),
    }


def _capacity_decision(bundle: _HybridModel, route: Mapping[str, Any], projection: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    row = build_probe_gate_record(_capacity_row(route, projection, packet))
    encoded = PG237._encode([row], bundle.input_vocab, bundle.capacity_vocab, bundle.device)
    with torch.no_grad():
        context = bundle.base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
        positions = PG237._positions([row], context.shape[1], bundle.device)
        output = bundle.capacity(context, classification_positions=positions)
        probs = output["action"].softmax(-1)[0]
    send_index = PG237.ACTION_INDEX["send_candidate"]
    abstain_index = PG237.ACTION_INDEX["abstain"]
    send_probability = float(probs[send_index].detach().cpu())
    action_index = int(output["action"].argmax(-1)[0].detach().cpu())
    allowed = action_index == send_index and send_probability >= 0.90
    action = "send_candidate" if allowed else "abstain"
    return {"action": action, "send_probability": round(send_probability, 8), "threshold": 0.90, "row_hash": PG237.digest(row), "surface": row["surface_class"], "method": row["method"], "probe_gate_target": bool(row.get("probe_send_eligible")), "model_used_oracle": False, "token_encoder": "pg248-observable-feedback-alias-v1"}


def _hybrid_model_decision(model: _HybridModel, vocabulary: Mapping[str, int], device: torch.device, *, packet: Mapping[str, Any], route: Mapping[str, Any], projection: Mapping[str, Any]) -> dict[str, Any]:
    legacy = ORIGINAL_MODEL_DECISION(model.legacy, vocabulary, device, packet=packet, route=route, projection=projection)
    capacity = _capacity_decision(model, route, projection, packet)
    effective = "safe_candidate" if legacy.get("effective_action") == "safe_candidate" and capacity["action"] == "send_candidate" else "abstain"
    result = dict(legacy)
    result["effective_action"] = effective
    result["capacity_action"] = capacity["action"]
    result["capacity_model"] = capacity
    result["model_used_evaluator"] = False
    if effective == "abstain" and capacity["action"] == "abstain":
        result["abstain_reason"] = "pg249_action_gate"
    return result


def _runtime_wire(client: Any, candidate: Mapping[str, Any], fields: list[str], *, role: str) -> dict[str, Any]:
    payload = PG198.validate_detection_payload(dict(candidate.get("payload") or {}))
    values = PG198._runtime_values(payload=payload, fields=PG198._route_fields(fields))
    method = str(payload["method"]).upper()
    base_url = str(client.base_url).rstrip("/")
    path = str(payload["path"])
    if method == "GET":
        wire = f"GET {base_url}{path}?{urlencode(values)}"
        body = ""
    else:
        wire = f"POST {base_url}{path}"
        body = urlencode(values)
    record = {"role": role, "method": method, "path": path, "field_count": len(values), "wire": wire, "body": body, "payload_sha256": str(payload.get("payload_sha256", "")), "values_sha256": PG208._digest(values)}
    # This is intentionally stdout-only.  The JSON reports contain only the
    # hashes and bounded response projections.
    print(f"[PG250-EPHEMERAL-{role.upper()}-WIRE] {wire}{(' body=' + body) if body else ''}")
    WIRE_LOG.append({key: record[key] for key in ("role", "method", "path", "field_count", "payload_sha256", "values_sha256")})
    return record


def _send_ai_with_reference(client: Any, learner: Any, candidates: list[dict[str, Any]], *, route: Mapping[str, Any], baseline_status: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = learner.select(candidates)
    reference = next((candidate for candidate in candidates if str(candidate.get("candidate_id")) != str(selected.get("candidate_id"))), selected)
    reference_result = PG208.send_grounded_candidate(client, candidate=reference, fields=list(route["fields"]), layout_variant=str(route["layout"]), baseline_status=baseline_status, typed_available=True)
    _runtime_wire(client, reference, list(route["fields"]), role="reference")
    ai_result = PG208.send_grounded_candidate(client, candidate=selected, fields=list(route["fields"]), layout_variant=str(route["layout"]), baseline_status=baseline_status, typed_available=True)
    _runtime_wire(client, selected, list(route["fields"]), role="ai")
    signal = bool((ai_result.get("signal") or {}).get("candidate_signal", False))
    feedback = learner.observe(selected, status="candidate" if signal else "dead_end", evidence=ai_result.get("evidence"), evaluator_confirmed=False)
    ai_result["ai_decision"] = {"candidate_id": str(selected["candidate_id"]), "selection_score": float(selected.get("selection_score", 0.0)), "status_feedback": feedback["status"], "model_used_evaluator": False}
    ai_result["reference_comparison"] = {"reference_candidate": PG208.candidate_summary(reference), "reference_result": reference_result, "reference_role": "independent_catalog_candidate", "wire_hashes_only": True}
    return ai_result, selected


def _load_model(device: torch.device) -> tuple[_HybridModel, dict[str, int]]:
    legacy, legacy_vocab = ORIGINAL_LOAD_MODEL(device)
    capacity, capacity_vocab, input_vocab, base = _load_capacity(device)
    return _HybridModel(legacy, capacity, capacity_vocab, input_vocab, base, device), legacy_vocab


def main() -> int:
    PG208._load_model = _load_model
    PG208._model_decision = _hybrid_model_decision
    PG208._send_ai_candidate = _send_ai_with_reference
    PG208.REPORT_PATH = REPORT
    PG208.TRACE_PATH = TRACE
    PG208.PROTOCOL_PATH = PROTOCOL
    PG208.MARKDOWN_PATH = MARKDOWN
    PG208.ARTIFACT_DIR = ARTIFACT_DIR
    PG208.main()
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    route_runs = list(report.get("route_runs") or [])
    sent = [row for row in route_runs if row.get("candidate_sent")]
    # The wrapper stores the comparison inside candidate_result so that the
    # top-level route keeps the PG-208 episode schema.  Read that nested
    # record here rather than silently reporting zero comparisons.
    pairs = [row for row in sent if (row.get("candidate_result") or {}).get("reference_comparison")]
    ai_effect = sum(int(bool((row.get("candidate_result") or {}).get("oracle", {}).get("dual_agreement"))) for row in pairs)
    ref_effect = sum(int(bool((((row.get("candidate_result") or {}).get("reference_comparison") or {}).get("reference_result") or {}).get("oracle", {}).get("dual_agreement"))) for row in pairs)
    agreement = sum(int(bool((row.get("candidate_result") or {}).get("oracle", {}).get("dual_agreement")) == bool((((row.get("candidate_result") or {}).get("reference_comparison") or {}).get("reference_result") or {}).get("oracle", {}).get("dual_agreement"))) for row in pairs)
    action_artifact = ACTION_ARTIFACT
    if action_artifact is None:
        raise RuntimeError("PG-250 action artifact was not resolved")
    action_checkpoint = torch.load(action_artifact, map_location="cpu", weights_only=False)
    report.update({"protocol_id": "pg-pk-250-pg252-pikachu-payload-replay-v1", "schema_version": "pg250-pikachu-pg252-payload-replay-v1", "status": "completed_pg252_probe_gate_pikachu_get_post_payload_replay", "model": {**dict(report.get("model") or {}), "action_adapter": str(action_artifact.relative_to(ROOT)), "action_adapter_hidden_dim": int(action_checkpoint["hidden_dim"]), "action_probability_threshold": 0.90, "model_participated_in_send_gate": True, "action_target": "safe_probe_availability"}, "reference_comparison": {"pair_count": len(pairs), "ai_typed_effect_count": ai_effect, "reference_typed_effect_count": ref_effect, "ai_reference_oracle_agreement_count": agreement, "reference_is_independent_catalog_candidate": True, "raw_wires_persisted": False}, "ephemeral_wire_count": len(WIRE_LOG), "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "safety": {**dict(report.get("safety") or {}), "pg252_probe_gate": True, "reference_sent_on_loopback_only": True, "raw_wires_stdout_only": True, "external_network_target": False}})
    report["report_sha256"] = PG208._digest(report)
    protocol.update({"protocol_id": report["protocol_id"], "schema_version": "pg250-pikachu-pg252-payload-replay-protocol-v1", "model_action_gate": "pg252_probe_gate_selected_adapter_with_fixed_0.90_threshold", "reference_comparison": "catalog_candidate_sent_separately_on_same_fresh_loopback_target", "get_post_parameterized_replay": True, "raw_wire_displayed_ephemerally": True, "raw_payload_and_response_excluded": True, "promotion_blocked": True})
    protocol["protocol_sha256"] = PG208._digest(protocol)
    trace.update({"schema_version": "pg250-pikachu-pg252-payload-replay-trace-v1", "action_adapter": str(ACTION_ARTIFACT.relative_to(ROOT)), "reference_comparison": report["reference_comparison"], "ephemeral_wires": WIRE_LOG, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG208._write(REPORT, report)
    PG208._write(PROTOCOL, protocol)
    PG208._write(TRACE, trace)
    MARKDOWN.write_text("\n".join(["# PG-250 PG-252 probe-gated Pikachu payload replay", "", f"routes={len(route_runs)}; candidate sends={len(sent)}; GET={report['counts']['get_route_count']}; POST={report['counts']['post_route_count']}", f"AI typed effects={ai_effect}; reference typed effects={ref_effect}; oracle agreement={agreement}; ephemeral wires={len(WIRE_LOG)}", "", "AI 先经过 PG-252 safe-probe gate，再由 PG-208 safe catalog 生成候选；reference candidate 单独发包。wire 只 stdout 展示，报告只保存 projection/hash。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "reference_comparison": report["reference_comparison"], "ephemeral_wire_count": len(WIRE_LOG), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
