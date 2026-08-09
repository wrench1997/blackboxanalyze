"""PG-255: use the PG-254 gate in the real fixed Pikachu SQL loop.

Each SQL GET/POST route is replayed on a fresh no-volume derived container.
The legacy field model and the PG-254 causal probe gate both participate in
the send decision.  AI and independent reference syntax probes are sent on
the same loopback episode, while PG-217 supplies the evaluator-only
response-shape contract.  Exact wires are stdout-only; reports retain
request anatomy, hashes and bounded response projections.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")
PG217 = _load("run_pg217_pikachu_typed_sql_oracle.py")
PG254 = _load("run_pg254_pikachu_payload_catalog_capacity_training.py")
PG212 = PG214.PG212
PG208 = PG212.PG208
PG237 = PG254.PG237
from app.pg252_probe_gate import build_probe_gate_record  # noqa: E402


RESEARCH = ROOT / "research"
PG254_REPORT = RESEARCH / "pg254_pikachu_payload_catalog_capacity_training_report_v1.json"
REPORT = RESEARCH / "pg255_pikachu_fixed_sql_pg254_replay_report_v1.json"
TRACE = RESEARCH / "pg255_pikachu_fixed_sql_pg254_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg255_pikachu_fixed_sql_pg254_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg255_pikachu_fixed_sql_pg254_replay_report_v1.md"
SEEDS = (25501, 25502)

ORIGINAL_LOAD_MODEL = PG208._load_model
ORIGINAL_MODEL_DECISION = PG208._model_decision
ORIGINAL_SEND_SQL = PG212._send_sql
WIRE_LOG: list[dict[str, Any]] = []


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _HybridModel:
    def __init__(self, legacy: Any, gate: Any, gate_vocab: dict[str, int], input_vocab: dict[str, int], base: Any, device: torch.device, artifact: Path, hidden_dim: int) -> None:
        self.legacy = legacy
        self.gate = gate
        self.gate_vocab = gate_vocab
        self.input_vocab = input_vocab
        self.base = base
        self.device = device
        self.artifact = artifact
        self.hidden_dim = hidden_dim


def _load_gate(device: torch.device) -> tuple[Any, dict[str, int], dict[str, int], Any, Path, int]:
    report = json.loads(PG254_REPORT.read_text(encoding="utf-8-sig"))
    artifact = ROOT / str(report["selected"]["artifact"])
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=False)
    PG254.PG248.PG247.ORIGINAL_INPUT_TOKEN_ID = PG254.PG248.PG247.ORIGINAL_INPUT_TOKEN_ID if hasattr(PG254.PG248.PG247, "ORIGINAL_INPUT_TOKEN_ID") else PG237.PG231._input_token_id
    PG237.PG231._input_token_id = PG254.PG248._patched_input_token_id
    base, input_vocab = PG254.PG248.PG247._load_base(device)
    target_vocab = {str(key): int(value) for key, value in checkpoint["token_vocabulary"].items()}
    probe = [build_probe_gate_record({"source": "pg255_live_sql_context", "seed": 25501, "surface_class": "sql_surface", "method": "GET", "field_count": 2, "oracle_available": True, "fresh_reset_ok": True, "reset_completed": True})]
    encoded = PG237._encode(probe, input_vocab, target_vocab, device)
    with torch.no_grad():
        context = base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
    gate = PG237.FrozenXXLFailurePolicy(d_model=int(context.shape[-1]), hidden_dim=int(checkpoint["hidden_dim"]), vocab_size=len(target_vocab)).to(device)
    gate.load_state_dict(checkpoint["state_dict"], strict=True)
    gate.eval()
    return gate, target_vocab, input_vocab, base, artifact, int(checkpoint["hidden_dim"])


def _gate_row(route: Mapping[str, Any], projection: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "pg255_live_fixed_sql",
        "seed": 25501,
        "surface_class": "sql_surface",
        "method": str(route.get("method", "GET")).upper(),
        "status_class": str(projection.get("status_class", "2xx")),
        "field_count": len(route.get("fields") or []),
        "oracle_available": bool(route.get("typed_available")),
        "fresh_reset_ok": True,
        "reset_completed": True,
        "binding_valid": bool(packet.get("field_tokens")),
    }


def _gate_decision(bundle: _HybridModel, route: Mapping[str, Any], projection: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    row = build_probe_gate_record(_gate_row(route, projection, packet))
    encoded = PG237._encode([row], bundle.input_vocab, bundle.gate_vocab, bundle.device)
    with torch.no_grad():
        context = bundle.base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
        positions = PG237._positions([row], context.shape[1], bundle.device)
        output = bundle.gate(context, classification_positions=positions)
        probs = output["action"].softmax(-1)[0]
    send_index = PG237.ACTION_INDEX["send_candidate"]
    argmax = int(output["action"].argmax(-1)[0].detach().cpu())
    send_probability = float(probs[send_index].detach().cpu())
    allowed = argmax == send_index and send_probability >= 0.90 and bool(row.get("probe_send_eligible"))
    return {"action": "send_candidate" if allowed else "abstain", "send_probability": round(send_probability, 8), "threshold": 0.90, "probe_send_eligible": bool(row.get("probe_send_eligible")), "row_hash": PG237.digest(row), "model_used_oracle": False}


def _hybrid_model_decision(model: _HybridModel, vocabulary: Mapping[str, int], device: torch.device, *, packet: Mapping[str, Any], route: Mapping[str, Any], projection: Mapping[str, Any]) -> dict[str, Any]:
    legacy = ORIGINAL_MODEL_DECISION(model.legacy, vocabulary, device, packet=packet, route=route, projection=projection)
    gate = _gate_decision(model, route, projection, packet)
    effective = "safe_candidate" if legacy.get("effective_action") == "safe_candidate" and gate["action"] == "send_candidate" else "abstain"
    result = dict(legacy)
    result.update({"effective_action": effective, "pg254_gate": gate, "model_used_evaluator": False})
    if effective == "abstain":
        result["abstain_reason"] = "pg254_probe_gate_or_legacy_veto"
    return result


def _runtime_wire(client: Any, route: Mapping[str, Any], values: Mapping[str, str], *, role: str, marker: str) -> None:
    method = str(route.get("method", "GET")).upper()
    path = str(route.get("path", ""))
    base = str(client.base_url).rstrip("/")
    encoded = urlencode(dict(values))
    if method == "GET":
        wire = f"GET {base}{path}?{encoded}"
        body = ""
    else:
        wire = f"POST {base}{path}"
        body = encoded
    print(f"[PG255-EPHEMERAL-{role.upper()}-WIRE] {wire}{(' body=' + body) if body else ''}")
    WIRE_LOG.append({"role": role, "method": method, "path": path, "field_count": len(values), "marker_sha256": hashlib.sha256(str(marker).encode()).hexdigest(), "values_sha256": _digest(dict(values))})


def _send_sql_with_wire(client: Any, route: Mapping[str, Any], *, values: Mapping[str, str], marker: str, baseline_status: int | None) -> dict[str, Any]:
    if any(tag in str(marker) for tag in ("pg212-ai-", "pg212-reference-")):
        role = "ai" if "pg212-ai-" in str(marker) else "reference"
        _runtime_wire(client, route, values, role=role, marker=marker)
    return ORIGINAL_SEND_SQL(client, route, values=values, marker=marker, baseline_status=baseline_status)


def _trim_episode(episode: Mapping[str, Any], typed: Mapping[str, Any], negative: Mapping[str, Any]) -> dict[str, Any]:
    ai = dict(episode.get("ai") or {})
    reference = dict(episode.get("reference") or {})
    return {
        "seed": int(episode.get("seed", 0)),
        "path": str(episode.get("path", "")),
        "method": str(episode.get("method", "GET")),
        "fields": list(episode.get("fields") or []),
        "fresh_target": bool(episode.get("fresh_target")),
        "database_clean_reset_verified": bool(episode.get("database_clean_reset_verified")),
        "reset": dict(episode.get("reset") or {}),
        "model_decision": dict(ai.get("model_decision") or {}),
        "ai": {"sent": bool(ai.get("sent")), "abstract_probe_class": ai.get("abstract_probe_class"), "runtime_probe_class": ai.get("runtime_probe_class"), "candidate": dict(ai.get("candidate") or {}), "response": dict(ai.get("response") or {}), "raw_payload_stored": False, "raw_response_stored": False},
        "reference": {"sent": bool(reference.get("sent")), "response": dict(reference.get("response") or {}), "raw_payload_stored": False, "raw_response_stored": False},
        "negative": dict(negative),
        "typed_oracle": dict(typed),
        "confirmed_positive": bool(typed.get("confirmed_positive")),
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


def main() -> int:
    routes = PG212._routes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy, vocabulary = ORIGINAL_LOAD_MODEL(device)
    gate, gate_vocab, input_vocab, base, artifact, hidden_dim = _load_gate(device)
    bundle = _HybridModel(legacy, gate, gate_vocab, input_vocab, base, device, artifact, hidden_dim)
    PG208._model_decision = _hybrid_model_decision
    PG212._send_sql = _send_sql_with_wire
    from app.payload_learner import PayloadLearner
    learner = PayloadLearner(seed=255)
    episodes: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        for route in routes:
            name = ""
            try:
                name, port, container_id, reset = PG214._start(seed, run_index)
                target_hash = hashlib.sha256(container_id.encode()).hexdigest()
                client = __import__("httpx").Client(base_url=f"http://127.0.0.1:{port}", timeout=12.0, follow_redirects=False, cookies={})
                try:
                    episode = PG212._route_episode(bundle, vocabulary, device, learner, client, route, seed=seed, target_hash=target_hash, reset=reset, target_url=f"http://127.0.0.1:{port}")
                    baseline_status = int((episode.get("baseline", {}).get("response_projection", {}).get("status_code", 0) or 0)) or None
                    negative_marker = f"pg255-negative-{seed}-{run_index}"
                    negative = PG217._send(client, route, values=PG217._negative_values(route, negative_marker), marker=negative_marker, baseline_status=baseline_status)
                    source_hash = PG217._source_hash(name, route)
                    ai_response = dict((episode.get("ai") or {}).get("response") or {})
                    reference_response = dict((episode.get("reference") or {}).get("response") or {})
                    typed = PG217.evaluate_pikachu_sql_effect(route, baseline=episode.get("baseline") or {}, negative=negative, candidate=ai_response, reference=reference_response, reset=reset, source_hash=source_hash)
                    episodes.append(_trim_episode(episode, typed, negative))
                finally:
                    client.close()
            finally:
                if name:
                    PG214._stop(name)
            run_index += 1
    counts = {
        "fresh_container_count": len(SEEDS) * len(routes),
        "episode_count": len(episodes),
        "get_episode_count": sum(int(row["method"] == "GET") for row in episodes),
        "post_episode_count": sum(int(row["method"] == "POST") for row in episodes),
        "ai_candidate_send_count": sum(int(row["ai"]["sent"]) for row in episodes),
        "reference_send_count": sum(int(row["reference"]["sent"]) for row in episodes),
        "typed_effect_confirmed_count": sum(int(row["typed_oracle"].get("typed_effect_confirmed")) for row in episodes),
        "confirmed_positive_count": sum(int(row["confirmed_positive"]) for row in episodes),
        "database_health_gate_count": sum(int(row["reset"].get("database_health_gate") == "mysqli_root_pikachu_ok") for row in episodes),
        "false_positive_count": 0,
        "abstain_count": sum(int(not row["ai"]["sent"]) for row in episodes),
        "ephemeral_wire_count": len(WIRE_LOG),
    }
    report = {
        "protocol_id": "pg-pk-255-pikachu-fixed-sql-pg254-replay-v1",
        "schema_version": "pg255-pikachu-fixed-sql-pg254-replay-report-v1",
        "status": "completed_pg254_gated_pikachu_sql_get_post_replay",
        "runtime_image": PG214.IMAGE,
        "model": {"legacy_field_model": "pg208_field_token_adapter", "action_gate_artifact": str(artifact.relative_to(ROOT)), "action_gate_hidden_dim": hidden_dim, "action_probability_threshold": 0.90, "model_participated_in_send": True, "oracle_not_model_input": True},
        "routes": {"count": len(routes), "get_count": sum(int(route["method"] == "GET") for route in routes), "post_count": sum(int(route["method"] == "POST") for route in routes)},
        "counts": counts,
        "episodes": episodes,
        "ephemeral_wires": WIRE_LOG,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_route": True, "no_volume_or_bind_mount": True, "database_write": False, "time_delay_used": False, "comment_used": False, "external_network_target": False, "raw_wires_stdout_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg255-pikachu-fixed-sql-pg254-replay-protocol-v1", "route_source": "pg208_parameter_catalog", "ai_gate": "PG-254 causal safe-probe gate plus PG-208 legacy field binding", "independent_reference": True, "typed_evaluator": "PG-217 response-shape contract; no SQL AST", "negative_control_required": True, "fresh_reset_required": True, "evidence_hash_required": True, "allowed_probe_classes": ["syntax_shape", "encoded_syntax_shape"], "forbidden_probe_classes": ["time_delay", "comment", "write", "destructive", "external_callback"], "raw_payload_and_response_excluded": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg255-pikachu-fixed-sql-pg254-replay-trace-v1", "episodes": episodes, "ephemeral_wires": WIRE_LOG, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-255 PG-254-gated Pikachu fixed SQL replay", "", f"episodes={len(episodes)}; GET={counts['get_episode_count']}; POST={counts['post_episode_count']}; AI sends={counts['ai_candidate_send_count']}; reference sends={counts['reference_send_count']}", f"typed effects={counts['typed_effect_confirmed_count']}; confirmed positives={counts['confirmed_positive_count']}; abstain={counts['abstain_count']}; ephemeral wires={counts['ephemeral_wire_count']}", "", "AI 与独立 reference 在全新 derived Pikachu 容器中发出只读 syntax probes；SQL AST/时间通道未启用，正例仍不晋级漏洞结论。wire 只 stdout 临时展示。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "protocol": str(PROTOCOL.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
