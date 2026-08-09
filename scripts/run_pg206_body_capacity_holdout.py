"""PG-206: matched field-token training with large and XXL frozen bodies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG205 = _load_script("run_pg205_field_token_training_and_replay.py")
from app.pg205_field_token_decoder import FieldTokenGroundingDecoder, evaluate_field_aware, train_field_aware, warm_start_from_pg203  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg206-body-capacity-v1"
REPORT_PATH = RESEARCH / "pg206_body_capacity_holdout_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg206_body_capacity_holdout_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg206_body_capacity_holdout_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg206_body_capacity_holdout_report_v1.md"
PG203_ARTIFACT = ROOT / "artifacts" / "pg203-token-aware-adapter-v1" / "xxl_token_aware_adapter.pt"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _runtime_rows() -> list[dict[str, Any]]:
    report_path = RESEARCH / "pg205_field_token_training_and_replay_report_v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    encoding = {"identity": 0, "dom_markup": 1, "encoded_dom": 2, "abstract_sql": 3}
    failure = {"no_effect": 0, "status_changed": 1, "redirect_shape": 2, "validation_shape": 3, "server_shape": 4, "oracle_unknown": 5}
    rows: list[dict[str, Any]] = []
    for item in report["route_runs"]:
        decision = dict(item["model_decision"])
        features = dict(decision.get("features") or {})
        failure_name = str(decision.get("failure", "no_effect"))
        fields = list(item.get("fields") or [])
        projection = dict(item.get("control_projection") or {})
        rows.append({
            "method": str(item["method"]).upper(),
            "redirect_hops": int(projection.get("redirect_hop_count", 0) or 0),
            "status_class": str(features.get("status_class", projection.get("status_class", "2xx"))),
            "candidate_signal": 1,
            "typed_available": int(bool(features.get("typed_available", False))),
            "negative_control": 1,
            "budget_remaining": 1,
            "failure_kind": str(features.get("failure_kind", "no_effect")),
            "label": 2 if item.get("candidate_sent") else 3,
            "encoding_label": encoding.get(str(decision.get("encoding", "identity")), 0),
            "failure_label": failure.get(failure_name, 0),
            "field_names": fields,
            "response_projection": projection,
            "source": "pg205_fresh_route_replay",
            "seed": int(item.get("seed", 0)),
        })
    return rows


def _build_body(name: str, vocabulary: dict[str, int], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name == "large":
        model = PG205.PG197.PG191._build_model("large", vocabulary, device)
        return model, {"variant": "large", "body_parameter_count": int(sum(p.numel() for p in model.parameters())), "source": "artifacts/pg189-structured-get-trace-action-v1/large.pt"}
    risk_decoder, _ = PG205.PG197._load_decoder(device, vocabulary)
    return risk_decoder.frozen_base, {"variant": "xxl", "body_parameter_count": int(sum(p.numel() for p in risk_decoder.frozen_base.parameters())), "source": "artifacts/pg203-token-aware-adapter-v1/xxl_token_aware_adapter.pt"}


def main() -> int:
    train, replay, holdout, vocabulary, device = PG205._load_vocabulary()
    runtime = _runtime_rows()
    augmented = PG205._augment_rows(train)
    train_rows = train + augmented
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    pg203_state = torch.load(PG203_ARTIFACT, map_location="cpu", weights_only=False)["model_state"]
    results: list[dict[str, Any]] = []
    for index, name in enumerate(("large", "xxl")):
        torch.manual_seed(20601 + index)
        base, base_meta = _build_body(name, vocabulary, device)
        adapter = FieldTokenGroundingDecoder(base, hidden_dim=96).to(device)
        warm = warm_start_from_pg203(adapter, pg203_state) if name == "xxl" else {"source": "fresh_large_field_adapter", "copied_keys": [], "field_projection_initialized": True}
        training = train_field_aware(adapter, train_rows, holdout, ids, mask, epochs=60)
        replay_metrics = evaluate_field_aware(adapter, replay, ids, mask)
        route_metrics = evaluate_field_aware(adapter, runtime, ids, mask)
        result = {**base_meta, "adapter_parameter_count": int(sum(p.numel() for p in adapter.parameters())), "warm_start": warm, "training": training, "replay": replay_metrics, "fresh_route_replay": route_metrics, "catastrophic_forgetting_detected": bool(replay_metrics["action_accuracy"] < 0.7 or replay_metrics["unsafe_allow_count"] > 0), "unsafe_allow_total": int(training["holdout"]["unsafe_allow_count"] + replay_metrics["unsafe_allow_count"] + route_metrics["unsafe_allow_count"])}
        results.append(result)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": f"pg206-{name}-field-token-v1", "vocabulary": vocabulary, "model_state": adapter.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / f"{name}_field_token_adapter.pt")
        del adapter, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    large, xxl = results
    xxl_better = bool(
        xxl["training"]["holdout"]["action_accuracy"] >= large["training"]["holdout"]["action_accuracy"]
        and xxl["training"]["holdout"]["encoding_accuracy"] >= large["training"]["holdout"]["encoding_accuracy"]
        and xxl["fresh_route_replay"]["action_accuracy"] >= large["fresh_route_replay"]["action_accuracy"]
        and xxl["unsafe_allow_total"] <= large["unsafe_allow_total"]
        and (xxl["training"]["holdout"]["action_accuracy"] > large["training"]["holdout"]["action_accuracy"] or xxl["training"]["holdout"]["encoding_accuracy"] > large["training"]["holdout"]["encoding_accuracy"] or xxl["fresh_route_replay"]["action_accuracy"] > large["fresh_route_replay"]["action_accuracy"])
    )
    report = {
        "protocol_id": "pg-pk-206-body-capacity-holdout-v1",
        "schema_version": "pg206-body-capacity-holdout-report-v1",
        "status": "completed_matched_large_xxl_field_token_holdout",
        "device": str(device),
        "data": {"train_rows": len(train_rows), "base_rows": len(train), "augmentation_rows": len(augmented), "holdout_rows": len(holdout), "old_replay_rows": len(replay), "fresh_route_replay_rows": len(runtime), "source_split": "PG-199 seed 19901 train / PG-199 seed 19902 replay / PG-200 holdout / PG-205 fresh route replay"},
        "variants": results,
        "capacity_101m_better": xxl_better,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "selected_variant": "xxl" if xxl_better else None},
        "safety": {"local_projection_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "script_execution": False, "database_write": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg206-body-capacity-holdout-protocol-v1", "body_variants": ["large", "xxl"], "same_train_holdout_required": True, "source_route_seed_holdout_required": True, "capacity_gain_requires_repeated_new_route_improvement": True, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg206-body-capacity-holdout-trace-v1", "evaluation_only": True, "data": report["data"], "variants": results, "training_eligible": False, "memory_promotion_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN_PATH.write_text("\n".join(["# PG-206 body capacity holdout", "", f"device={device}; train={len(train_rows)}; holdout={len(holdout)}; fresh route replay={len(runtime)}", f"large body={large['body_parameter_count']}; holdout action/encoding/failure={large['training']['holdout']['action_accuracy']}/{large['training']['holdout']['encoding_accuracy']}/{large['training']['holdout']['failure_accuracy']}; fresh action={large['fresh_route_replay']['action_accuracy']}", f"xxl body={xxl['body_parameter_count']}; holdout action/encoding/failure={xxl['training']['holdout']['action_accuracy']}/{xxl['training']['holdout']['encoding_accuracy']}/{xxl['training']['holdout']['failure_accuracy']}; fresh action={xxl['fresh_route_replay']['action_accuracy']}", f"capacity_101m_better={xxl_better}; selected={report['promotion']['selected_variant']}", "", "No capacity variant is promoted solely from one split; raw payloads/responses remain excluded.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "large_body_parameters": large["body_parameter_count"], "xxl_body_parameters": xxl["body_parameter_count"], "large_holdout": large["training"]["holdout"], "xxl_holdout": xxl["training"]["holdout"], "large_fresh_route": large["fresh_route_replay"], "xxl_fresh_route": xxl["fresh_route_replay"], "capacity_101m_better": xxl_better, "selected_variant": report["promotion"]["selected_variant"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
