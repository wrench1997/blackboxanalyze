"""PG-203: train and evaluate the explicit structural-token adapter."""

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

from app.pg203_token_aware_decoder import TokenAwareGroundingDecoder, evaluate_token_aware, train_token_aware  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG197 = _load_script("run_pg197_risk_aware_cross_evaluator.py")
PG201 = _load_script("run_pg201_multitask_decoder.py")
PG202 = _load_script("run_pg202_balanced_auxiliary.py")

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg203-token-aware-adapter-v1"
REPORT_PATH = RESEARCH / "pg203_token_aware_adapter_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg203_token_aware_adapter_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg203_token_aware_adapter_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg203_token_aware_adapter_report_v1.md"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG197.PG191.PG189._load_rows()
    vocabulary = PG197.PG191.PG189._vocabulary(train, PG197.PG191.PG189._load_body_vocab())
    risk_decoder, decoder_training = PG197._load_decoder(device, vocabulary)
    model = TokenAwareGroundingDecoder(risk_decoder.frozen_base).to(device)
    for parameter in model.frozen_base.parameters():
        parameter.requires_grad = False
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    pg199 = PG201._pg199_rows()
    pg200 = PG201._pg200_rows()
    base_train = [row for row in pg199 if int(row.get("seed", 0)) == 19901]
    replay_rows = [row for row in pg199 if int(row.get("seed", 0)) == 19902]
    augmented = PG202._augmentation()
    train_rows = base_train + augmented
    training = train_token_aware(model, train_rows, pg200, ids, mask, epochs=80)
    replay_metrics = evaluate_token_aware(model, replay_rows, ids, mask)
    report = {
        "protocol_id": "pg-pk-203-token-aware-adapter-v1",
        "schema_version": "pg203-token-aware-adapter-report-v1",
        "status": "completed_explicit_encoding_failure_token_adapter",
        "device": str(device),
        "model": {
            "variant": "xxl_token_aware_adapter",
            "base_parameter_count": int(sum(p.numel() for p in model.frozen_base.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "token_feature_dim": 10,
            "frozen_base": True,
            "online_weight_update": False,
        },
        "data": {
            "base_train_rows": len(base_train),
            "augmentation_rows": len(augmented),
            "train_rows": len(train_rows),
            "replay_rows": len(replay_rows),
            "holdout_rows": len(pg200),
            "structural_tokens": ["encoding_chain", "failure_projection"],
            "evaluator_labels_in_tokens": False,
        },
        "decoder_training": decoder_training,
        "token_aware_training": training,
        "replay_metrics": replay_metrics,
        "counts": {
            "holdout_action_accuracy": training["holdout"]["action_accuracy"],
            "holdout_encoding_accuracy": training["holdout"]["encoding_accuracy"],
            "holdout_failure_accuracy": training["holdout"]["failure_accuracy"],
            "holdout_unsafe_allow_count": training["holdout"]["unsafe_allow_count"],
            "replay_action_accuracy": replay_metrics["action_accuracy"],
            "replay_encoding_accuracy": replay_metrics["encoding_accuracy"],
            "replay_failure_accuracy": replay_metrics["failure_accuracy"],
            "replay_unsafe_allow_count": replay_metrics["unsafe_allow_count"],
            "catastrophic_forgetting_detected": bool(replay_metrics["action_accuracy"] < 0.7 or replay_metrics["unsafe_allow_count"] > 0),
        },
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "safety": {
            "structural_tokens_only": True,
            "evaluator_labels_in_policy_input": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "external_network": False,
            "online_weight_update": False,
        },
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg203-token-aware-adapter-protocol-v1",
        "model": "101M XXL frozen body + explicit encoding/failure token adapter",
        "tokens": ["encoding_chain", "failure_projection"],
        "source_holdout": "pg200_sql_v6_and_post_failure remain unseen during base training",
        "catastrophic_forgetting_gate": "replay_action_accuracy >= 0.7 and unsafe_allow_count == 0",
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {
        "schema_version": "pg203-token-aware-adapter-trace-v1",
        "evaluation_only": True,
        "data": report["data"],
        "training": training,
        "replay_metrics": replay_metrics,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    })
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg203-token-aware-adapter-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_token_aware_adapter.pt")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-203 token-aware adapter",
        "",
        f"device={device}; base parameters={report['model']['base_parameter_count']}; total parameters={report['model']['total_parameter_count']}",
        f"train={len(train_rows)}; replay={len(replay_rows)}; source-heldout={len(pg200)}; token features={report['model']['token_feature_dim']}",
        f"holdout action={training['holdout']['action_accuracy']}; encoding={training['holdout']['encoding_accuracy']}; failure={training['holdout']['failure_accuracy']}; unsafe={training['holdout']['unsafe_allow_count']}",
        f"replay action={replay_metrics['action_accuracy']}; replay unsafe={replay_metrics['unsafe_allow_count']}; forgetting={report['counts']['catastrophic_forgetting_detected']}",
        "",
        "Encoding chain and failure projection are explicit structural tokens; family labels and raw payload/response text remain absent.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "device": str(device),
        "base_parameters": report["model"]["base_parameter_count"],
        "total_parameters": report["model"]["total_parameter_count"],
        "holdout_action": training["holdout"]["action_accuracy"],
        "holdout_encoding": training["holdout"]["encoding_accuracy"],
        "holdout_failure": training["holdout"]["failure_accuracy"],
        "holdout_unsafe_allow": training["holdout"]["unsafe_allow_count"],
        "replay_action": replay_metrics["action_accuracy"],
        "replay_unsafe_allow": replay_metrics["unsafe_allow_count"],
        "forgetting": report["counts"]["catastrophic_forgetting_detected"],
        "training_eligible": False,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
