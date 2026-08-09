"""PG-202: balanced abstract auxiliary data for encoding/failure heads."""

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

from app.pg201_multitask_decoder import MultiTaskGroundingDecoder, evaluate_multitask, train_multitask  # noqa: E402


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

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg202-balanced-auxiliary-v1"
REPORT_PATH = RESEARCH / "pg202_balanced_auxiliary_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg202_balanced_auxiliary_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg202_balanced_auxiliary_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg202_balanced_auxiliary_report_v1.md"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _row(*, method: str, typed: int, candidate: int, status: str, failure_kind: str, label: int, encoding: int, failure: int, source: str) -> dict[str, Any]:
    return {
        "method": method,
        "redirect_hops": 1 if failure_kind == "redirect_chain" else 0,
        "status_class": status,
        "candidate_signal": candidate,
        "typed_available": typed,
        "negative_control": 1,
        "budget_remaining": 1,
        "failure_kind": failure_kind,
        "label": label,
        "encoding_label": encoding,
        "failure_label": failure,
        "source": source,
    }


def _augmentation() -> list[dict[str, Any]]:
    """Create balanced abstract rows, with no route/payload identity."""

    rows: list[dict[str, Any]] = []
    # Four encoding classes, alternating method and typed state.
    for encoding in range(4):
        for index in range(8):
            method = "GET" if index % 2 == 0 else "POST"
            typed = 0 if encoding == 0 else 1
            rows.append(_row(method=method, typed=typed, candidate=1 if typed else 0, status="2xx", failure_kind="no_effect", label=2 if typed else 3, encoding=encoding, failure=0, source="pg202_balanced_encoding"))
    # Six failure classes are represented through the four model-visible
    # failure kinds; the auxiliary label retains the finer distinction.
    failure_specs = (
        ("no_effect", "2xx", 0),
        ("status_changed", "4xx", 1),
        ("redirect_chain", "3xx", 2),
        ("post_validation", "4xx", 3),
    )
    for failure_kind, status, failure_label in failure_specs:
        for index in range(12):
            method = "POST" if failure_kind in {"post_validation", "redirect_chain"} else ("GET" if index % 2 == 0 else "POST")
            rows.append(_row(method=method, typed=0, candidate=0, status=status, failure_kind=failure_kind, label=3, encoding=0, failure=failure_label, source="pg202_balanced_failure"))
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
    pg199 = PG201._pg199_rows()
    pg200 = PG201._pg200_rows()
    base_train = [row for row in pg199 if int(row.get("seed", 0)) == 19901]
    replay_rows = [row for row in pg199 if int(row.get("seed", 0)) == 19902]
    augmented = _augmentation()
    train_rows = base_train + augmented
    training = train_multitask(model, train_rows, pg200, ids, mask, epochs=120)
    replay_metrics = evaluate_multitask(model, replay_rows, ids, mask)
    report = {
        "protocol_id": "pg-pk-202-balanced-auxiliary-v1",
        "schema_version": "pg202-balanced-auxiliary-report-v1",
        "status": "completed_balanced_encoding_failure_auxiliary_training",
        "device": str(device),
        "model": {
            "variant": "xxl_multitask_adapter",
            "base_parameter_count": int(sum(p.numel() for p in model.frozen_base.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "frozen_base": True,
            "online_weight_update": False,
        },
        "data": {
            "base_train_rows": len(base_train),
            "augmentation_rows": len(augmented),
            "train_rows": len(train_rows),
            "replay_rows": len(replay_rows),
            "holdout_rows": len(pg200),
            "augmentation_sources": ["pg202_balanced_encoding", "pg202_balanced_failure"],
        },
        "decoder_training": decoder_training,
        "multitask_training": training,
        "replay_metrics": replay_metrics,
        "counts": {
            "holdout_unsafe_allow_count": training["holdout"]["unsafe_allow_count"],
            "replay_unsafe_allow_count": replay_metrics["unsafe_allow_count"],
            "holdout_action_accuracy": training["holdout"]["action_accuracy"],
            "holdout_encoding_accuracy": training["holdout"]["encoding_accuracy"],
            "holdout_failure_accuracy": training["holdout"]["failure_accuracy"],
            "replay_action_accuracy": replay_metrics["action_accuracy"],
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
            "abstract_augmentation_only": True,
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
        "schema_version": "pg202-balanced-auxiliary-protocol-v1",
        "model": "101M XXL frozen body + balanced auxiliary action/encoding/failure adapter",
        "augmentation": "balanced abstract rows for encoding classes and failure shapes",
        "source_holdout": "pg200_sql_v6_and_post_failure_remain_unseen",
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
        "schema_version": "pg202-balanced-auxiliary-trace-v1",
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
    torch.save({"schema_version": "pg202-balanced-auxiliary-v1", "vocabulary": vocabulary, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_balanced_auxiliary_adapter.pt")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-202 balanced auxiliary training",
        "",
        f"device={device}; base parameters={report['model']['base_parameter_count']}; total parameters={report['model']['total_parameter_count']}",
        f"base train={len(base_train)}; augmentation={len(augmented)}; source-heldout={len(pg200)}; replay={len(replay_rows)}",
        f"holdout action={training['holdout']['action_accuracy']}; encoding={training['holdout']['encoding_accuracy']}; failure={training['holdout']['failure_accuracy']}; unsafe={training['holdout']['unsafe_allow_count']}",
        f"replay action={replay_metrics['action_accuracy']}; replay unsafe={replay_metrics['unsafe_allow_count']}; forgetting={report['counts']['catastrophic_forgetting_detected']}",
        "",
        "Augmentation is abstract and bounded; it adds no raw payload or response content.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "device": str(device),
        "base_parameters": report["model"]["base_parameter_count"],
        "total_parameters": report["model"]["total_parameter_count"],
        "base_train_rows": len(base_train),
        "augmentation_rows": len(augmented),
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
