"""PG-209: train on PG-208 parameterized traces with route/seed holdout.

The experiment asks whether real GET/POST response projections add useful
signal to the field-token adapter.  It compares a roughly 19M large body with
the selected 101M XXL body under the same adapter and the same holdout.  The
Pikachu trace is used as a diagnostic source only: oracle outcomes remain
quarantined and no raw request/response text is loaded into the model.
"""

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
PG208_DATASET = RESEARCH / "pg208_pikachu_parameterized_trace_dataset_v1.json"
PG208_REPORT = RESEARCH / "pg208_pikachu_typed_payload_loop_report_v1.json"
PG203_ARTIFACT = ROOT / "artifacts" / "pg203-token-aware-adapter-v1" / "xxl_token_aware_adapter.pt"
ARTIFACT_DIR = ROOT / "artifacts" / "pg209-parameterized-trace-training-v1"
REPORT_PATH = RESEARCH / "pg209_parameterized_trace_training_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg209_parameterized_trace_training_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg209_parameterized_trace_training_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg209_parameterized_trace_training_report_v1.md"

ROUTE_HOLDOUT = frozenset({"/vul/xss/xss_04.php", "/vul/xss/xss_dom_x.php", "/vul/sqli/sqli_x.php"})


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _normalize_pg208(row: dict[str, Any]) -> dict[str, Any]:
    decision = dict(row.get("model_decision") or {})
    features = dict(decision.get("features") or {})
    projection = dict(row.get("control_projection") or {})
    encoding = {"http_canary": 0, "inert_dom_markup": 1, "encoded_dom_markup": 2, "sql_channel_class": 3}
    failure = {"no_effect": 0, "status_changed": 1, "redirect_shape": 2, "validation_shape": 3, "server_shape": 4, "oracle_unknown": 5}
    probe_kind = str((row.get("candidate_summary") or {}).get("probe_kind", "http_canary"))
    failure_name = str(decision.get("failure", "no_effect"))
    return {
        "method": str(row.get("method", "GET")).upper(),
        "redirect_hops": int(projection.get("redirect_hop_count", 0) or 0),
        "status_class": str(features.get("status_class", projection.get("status_class", "2xx"))),
        "candidate_signal": 1,
        "typed_available": int(str(row.get("typed_oracle", "")) == "dom_nojs_dual"),
        "negative_control": 1,
        "budget_remaining": 1,
        "failure_kind": str(features.get("failure_kind", "no_effect")),
        "label": 2 if row.get("candidate_sent") else 3,
        "encoding_label": encoding.get(probe_kind, 0),
        "failure_label": failure.get(failure_name, 0),
        "field_names": list(row.get("fields") or []),
        "response_projection": projection,
        "source": "pg208_fresh_pikachu_parameterized_projection",
        "seed": int(row.get("seed", 0)),
        "path": str(row.get("path", "")),
    }


def _load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int], torch.device, dict[str, Any]]:
    base_train, old_replay, old_holdout, vocabulary, device = PG205._load_vocabulary()
    report = json.loads(PG208_REPORT.read_text(encoding="utf-8-sig"))
    all_rows = [_normalize_pg208(dict(row)) for row in report["route_runs"]]
    # Seed 20801 contributes training data except for route holdout paths;
    # seed 20802 plus the held-out paths are never seen during training.
    train_new = [row for row in all_rows if int(row["seed"]) == 20801 and row["path"] not in ROUTE_HOLDOUT]
    holdout_new = [row for row in all_rows if int(row["seed"]) == 20802 or row["path"] in ROUTE_HOLDOUT]
    # Keep the old PG-205 route source in training, but do not reuse its replay
    # seed as the new holdout.  The old independent holdout stays separate.
    train_rows = list(base_train) + train_new
    holdout_rows = list(old_holdout) + holdout_new
    metadata = {
        "base_train_rows": len(base_train), "pg208_train_rows": len(train_new), "train_rows": len(train_rows),
        "old_holdout_rows": len(old_holdout), "pg208_holdout_rows": len(holdout_new), "holdout_rows": len(holdout_rows),
        "route_holdout": sorted(ROUTE_HOLDOUT), "seed_train": 20801, "seed_holdout": 20802,
    }
    return train_rows, old_replay, holdout_rows, vocabulary, device, metadata


def _build_body(name: str, vocabulary: dict[str, int], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name == "large":
        body = PG205.PG197.PG191._build_model("large", vocabulary, device)
        return body, {"variant": "large", "body_parameter_count": int(sum(p.numel() for p in body.parameters())), "source": "pg189_large_body"}
    decoder, _ = PG205.PG197._load_decoder(device, vocabulary)
    return decoder.frozen_base, {"variant": "xxl", "body_parameter_count": int(sum(p.numel() for p in decoder.frozen_base.parameters())), "source": "pg206_xxl_body"}


def _train_variant(name: str, train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], old_replay: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device, pg203_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(20901 if name == "large" else 20902)
    body, body_meta = _build_body(name, vocabulary, device)
    adapter = FieldTokenGroundingDecoder(body, hidden_dim=96).to(device)
    warm = warm_start_from_pg203(adapter, pg203_state) if name == "xxl" else {"source": "fresh_large_field_adapter", "copied_keys": []}
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    training = train_field_aware(adapter, train_rows, holdout_rows, ids, mask, epochs=60)
    replay_metrics = evaluate_field_aware(adapter, old_replay, ids, mask)
    pg208_holdout = evaluate_field_aware(adapter, [row for row in holdout_rows if str(row.get("source", "")).startswith("pg208")], ids, mask)
    result = {
        **body_meta,
        "adapter_parameter_count": int(sum(p.numel() for p in adapter.parameters())),
        "warm_start": warm,
        "training": training,
        "old_replay": replay_metrics,
        "pg208_holdout": pg208_holdout,
        "catastrophic_forgetting_detected": bool(replay_metrics["action_accuracy"] < 0.7 or replay_metrics["unsafe_allow_count"] > 0),
        "unsafe_allow_total": int(training["holdout"]["unsafe_allow_count"] + replay_metrics["unsafe_allow_count"] + pg208_holdout["unsafe_allow_count"]),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": f"pg209-{name}-field-token-v1", "vocabulary": vocabulary, "model_state": adapter.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / f"{name}_field_token_adapter.pt")
    del adapter, body
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, {"variant": name, "training": training, "old_replay": replay_metrics, "pg208_holdout": pg208_holdout}


def main() -> int:
    train_rows, old_replay, holdout_rows, vocabulary, device, data = _load_rows()
    pg203_state = torch.load(PG203_ARTIFACT, map_location="cpu", weights_only=False)["model_state"]
    variants: list[dict[str, Any]] = []
    for name in ("large", "xxl"):
        result, _ = _train_variant(name, train_rows, holdout_rows, old_replay, vocabulary, device, pg203_state)
        variants.append(result)
    large, xxl = variants
    capacity_better = bool(
        xxl["training"]["holdout"]["action_accuracy"] >= large["training"]["holdout"]["action_accuracy"]
        and xxl["training"]["holdout"]["encoding_accuracy"] >= large["training"]["holdout"]["encoding_accuracy"]
        and xxl["pg208_holdout"]["action_accuracy"] >= large["pg208_holdout"]["action_accuracy"]
        and xxl["unsafe_allow_total"] <= large["unsafe_allow_total"]
        and (xxl["pg208_holdout"]["encoding_accuracy"] > large["pg208_holdout"]["encoding_accuracy"] or xxl["training"]["holdout"]["failure_accuracy"] > large["training"]["holdout"]["failure_accuracy"])
    )
    report = {
        "protocol_id": "pg-pk-209-parameterized-trace-training-v1",
        "schema_version": "pg209-parameterized-trace-training-report-v1",
        "status": "completed_pg208_route_seed_holdout_capacity_sweep",
        "device": str(device),
        "data": data,
        "variants": variants,
        "capacity_101m_better": capacity_better,
        "selected_variant": "xxl" if capacity_better else None,
        "promotion": {"training_eligible": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"raw_payload_strings_in_model": False, "raw_response_bodies_in_model": False, "oracle_labels_as_model_inputs": False, "external_network_targets": False, "script_execution": False, "database_write": False, "online_weight_update": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg209-parameterized-trace-training-protocol-v1", "train_source": "PG-205 train plus PG-208 seed 20801 non-holdout routes", "holdout_source": "PG-200 holdout plus PG-208 seed 20802 and route holdout", "capacity_variants": ["large", "xxl"], "same_adapter_and_optimizer": True, "catastrophic_forgetting_gate": True, "oracle_labels_as_model_inputs": False, "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg209-parameterized-trace-training-trace-v1", "evaluation_only": True, "data": data, "variants": variants, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN_PATH.write_text("\n".join(["# PG-209 parameterized trace training", "", f"device={device}; train={data['train_rows']}; holdout={data['holdout_rows']}; route holdout={data['route_holdout']}", f"large body={large['body_parameter_count']}; holdout={large['training']['holdout']}; PG-208={large['pg208_holdout']}", f"xxl body={xxl['body_parameter_count']}; holdout={xxl['training']['holdout']}; PG-208={xxl['pg208_holdout']}", f"capacity_101m_better={capacity_better}; selected={report['selected_variant']}", "", "This is a route/seed holdout diagnostic. Checkpoints remain quarantined until an independent typed SQL oracle and fresh OOD gate are added.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "train_rows": data["train_rows"], "holdout_rows": data["holdout_rows"], "large_body": large["body_parameter_count"], "xxl_body": xxl["body_parameter_count"], "large_pg208_holdout": large["pg208_holdout"], "xxl_pg208_holdout": xxl["pg208_holdout"], "capacity_101m_better": capacity_better, "selected_variant": report["selected_variant"], "training_eligible": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
