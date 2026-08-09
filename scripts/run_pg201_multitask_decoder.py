"""PG-201: train a multi-task adapter and measure cross-source forgetting."""

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

from app.pg196_failure_action_decoder import encode_features  # noqa: E402
from app.pg201_multitask_decoder import (  # noqa: E402
    MultiTaskGroundingDecoder,
    evaluate_multitask,
    train_multitask,
)


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG197 = _load_script("run_pg197_risk_aware_cross_evaluator.py")

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg201-multitask-decoder-v1"
REPORT_PATH = RESEARCH / "pg201_multitask_decoder_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg201_multitask_decoder_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg201_multitask_decoder_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg201_multitask_decoder_report_v1.md"


ENCODING_LABEL = {"http_canary": 0, "inert_dom_markup": 1, "encoded_dom_markup": 2, "sql_channel_class": 3}
FAILURE_LABEL = {"no_effect": 0, "status_changed": 1, "redirect_chain": 2, "post_validation": 3, "server_shape": 4, "oracle_unknown": 5}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _state_row(state: dict[str, Any], *, label: int, encoding_label: int, failure_label: int, source: str, seed: int | None = None) -> dict[str, Any]:
    row = {
        "method": str(state["method"]),
        "redirect_hops": int(state.get("redirect_hops", 0)),
        "status_class": str(state.get("status_class", "2xx")),
        "candidate_signal": int(state.get("candidate_signal", 0)),
        "typed_available": int(state.get("typed_available", 0)),
        "negative_control": int(state.get("negative_control", 0)),
        "budget_remaining": int(state.get("budget_remaining", 1)),
        "failure_kind": str(state.get("failure_kind", "no_effect")),
        "label": int(label),
        "encoding_label": int(encoding_label),
        "failure_label": int(failure_label),
        "source": source,
    }
    if seed is not None:
        row["seed"] = int(seed)
    return row


def _pg199_rows() -> list[dict[str, Any]]:
    report = json.loads((RESEARCH / "pg199_xxl_grounding_matrix_report_v1.json").read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for episode in report["route_runs"]:
        state = dict(episode["model_decision"]["state"])
        state["failure_kind"] = state.get("failure_kind", "no_effect")
        result = dict(episode.get("candidate_result") or {})
        candidate = dict(result.get("candidate") or {})
        encoding = ENCODING_LABEL.get(str(candidate.get("probe_kind", "http_canary")), 0)
        label = 2 if bool(episode.get("candidate_sent")) else 3
        failure = "post_validation" if str(episode["method"]).upper() == "POST" else str(state.get("failure_kind", "no_effect"))
        if failure not in FAILURE_LABEL:
            failure = "no_effect"
        rows.append(_state_row(state, label=label, encoding_label=encoding, failure_label=FAILURE_LABEL[failure], source="pg199_crawl_grounding", seed=int(episode["seed"])))
    return rows


def _pg200_rows() -> list[dict[str, Any]]:
    report = json.loads((RESEARCH / "pg200_source_heldout_report_v1.json").read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for item in report["sql_v6_runs"]:
        projection = item["oracle"]
        mode = str(item["mode"])
        if mode == "syntax":
            failure = "status_changed"
            failure_label = FAILURE_LABEL["status_changed"]
        elif mode == "error_redirect":
            failure = "redirect_chain"
            failure_label = FAILURE_LABEL["redirect_chain"]
        else:
            failure = "no_effect"
            failure_label = FAILURE_LABEL["no_effect"]
        state = {
            "method": item["method"],
            "redirect_hops": 0,
            "status_class": f"{int(projection.get('status_code', 200)) // 100}xx",
            "candidate_signal": 1,
            "typed_available": 1,
            "negative_control": 1,
            "budget_remaining": 1,
            "failure_kind": failure,
        }
        rows.append(_state_row(state, label=2, encoding_label=ENCODING_LABEL["sql_channel_class"], failure_label=failure_label, source="pg200_sql_v6"))
    for item in report["post_failure_runs"]:
        projection = item["failure"]
        mode = str(item["mode"])
        failure = "redirect_chain" if mode == "redirect_loop" else "post_validation"
        failure_label = FAILURE_LABEL[failure]
        state = {
            "method": "POST",
            "redirect_hops": 1 if mode == "redirect_loop" else 0,
            "status_class": str(item["model"]["state"]["status_class"]),
            "candidate_signal": 0,
            "typed_available": 0,
            "negative_control": 1,
            "budget_remaining": 1,
            "failure_kind": "redirect_chain" if mode == "redirect_loop" else "post_validation",
        }
        rows.append(_state_row(state, label=3, encoding_label=ENCODING_LABEL["http_canary"], failure_label=failure_label, source="pg200_post_failure"))
    return rows


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG197.PG191.PG189._load_rows()
    vocabulary = PG197.PG191.PG189._vocabulary(train, PG197.PG191.PG189._load_body_vocab())
    risk_decoder, decoder_training = PG197._load_decoder(device, vocabulary)
    model = MultiTaskGroundingDecoder(risk_decoder.frozen_base).to(device)
    for parameter in model.frozen_base.parameters():
        parameter.requires_grad = False
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    pg199 = _pg199_rows()
    pg200 = _pg200_rows()
    train_rows = [row for row in pg199 if int(row.get("seed", 0)) == 19901]
    replay_rows = [row for row in pg199 if int(row.get("seed", 0)) == 19902]
    training = train_multitask(model, train_rows, pg200, ids, mask, epochs=80)
    replay_metrics = evaluate_multitask(model, replay_rows, ids, mask)
    report = {
        "protocol_id": "pg-pk-201-multitask-decoder-v1",
        "schema_version": "pg201-multitask-decoder-report-v1",
        "status": "completed_multitask_adapter_source_split_and_forgetting_check",
        "device": str(device),
        "model": {
            "variant": "xxl_multitask_adapter",
            "base_parameter_count": int(sum(p.numel() for p in model.frozen_base.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "frozen_base": True,
            "online_weight_update": False,
        },
        "source_split": {
            "train_source": "pg199_crawl_grounding_seed_19901",
            "replay_source": "pg199_crawl_grounding_seed_19902",
            "holdout_source": ["pg200_sql_v6", "pg200_post_failure"],
            "train_rows": len(train_rows),
            "replay_rows": len(replay_rows),
            "holdout_rows": len(pg200),
        },
        "decoder_training": decoder_training,
        "multitask_training": training,
        "replay_metrics": replay_metrics,
        "counts": {
            "train_rows": len(train_rows),
            "replay_rows": len(replay_rows),
            "holdout_rows": len(pg200),
            "holdout_unsafe_allow_count": training["holdout"]["unsafe_allow_count"],
            "replay_unsafe_allow_count": replay_metrics["unsafe_allow_count"],
            "catastrophic_forgetting_detected": bool(replay_metrics["action_accuracy"] < 0.7),
        },
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "safety": {
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_policy_input": False,
            "external_network": False,
            "online_weight_update": False,
        },
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg201-multitask-decoder-protocol-v1",
        "model": "101M XXL frozen body + multitask action/encoding/failure adapter",
        "tasks": ["next_action", "encoding_class", "failure_class"],
        "source_split": report["source_split"],
        "catastrophic_forgetting_metric": "replay_action_accuracy_and_unsafe_allow_count",
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {
        "schema_version": "pg201-multitask-decoder-trace-v1",
        "evaluation_only": True,
        "source_split": report["source_split"],
        "training": training,
        "replay_metrics": replay_metrics,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    })
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg201-multitask-decoder-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_multitask_adapter.pt")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-201 multitask decoder",
        "",
        f"device={device}; base parameters={report['model']['base_parameter_count']}; total parameters={report['model']['total_parameter_count']}",
        f"train={len(train_rows)}; replay={len(replay_rows)}; source-heldout={len(pg200)}",
        f"holdout unsafe allow={training['holdout']['unsafe_allow_count']}; replay unsafe allow={replay_metrics['unsafe_allow_count']}; forgetting={report['counts']['catastrophic_forgetting_detected']}",
        "",
        "The adapter trains action, encoding and failure heads jointly while the XXL body remains frozen. No raw payload/response material enters the model or artifact.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "device": str(device),
        "base_parameters": report["model"]["base_parameter_count"],
        "total_parameters": report["model"]["total_parameter_count"],
        "train_rows": len(train_rows),
        "replay_rows": len(replay_rows),
        "holdout_rows": len(pg200),
        "holdout_unsafe_allow": training["holdout"]["unsafe_allow_count"],
        "replay_unsafe_allow": replay_metrics["unsafe_allow_count"],
        "forgetting": report["counts"]["catastrophic_forgetting_detected"],
        "training_eligible": False,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
