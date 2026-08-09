"""Evaluate frozen PG-295 causal MoE on unseen missing-observation patterns."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS, sha256_json  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, build_vocabulary, evaluate_causal_moe  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg294_active_repair_dataset_v1.json"
TRAINING_REPORT = RESEARCH / "pg295_causal_moe_training_report_v1_local_morning.json"
CHECKPOINT = ROOT / "artifacts" / "pg295-causal-moe" / "pg295_causal_moe_selected_local_morning.pt"
DATASET = RESEARCH / "pg296_missing_pattern_dataset_v1.json"
REPORT = RESEARCH / "pg296_missing_pattern_eval_report_v1.json"
PROTOCOL = RESEARCH / "pg296_missing_pattern_eval_protocol_v1.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def target() -> list[str]:
    return [TARGET_BOS, "next_action=recheck_oracle", "repair_action=recheck_oracle", "question=ask_typed_availability", "safe_to_send=0", TARGET_EOS]


def make_row(base: dict[str, Any], pattern: str) -> dict[str, Any]:
    context = [str(token) for token in base.get("context_tokens", []) if not str(token).startswith(("method=", "channel=", "status=", "field_bucket="))]
    if pattern == "get_query_order_shift":
        prefix = ["[BOS]", "phase=observe", "method=GET", "channel=query", "status=302", "field_bucket=unknown"]
        context = prefix + [token for token in context if token not in {"[BOS]", "[EOS]"}]
    elif pattern == "post_form_decoy":
        prefix = ["[BOS]", "phase=diagnose", "method=POST", "channel=form", "status=204", "candidate_error_shape=1", "field_bucket=unknown"]
        context = prefix + [token for token in context if token not in {"[BOS]", "[EOS]"}]
    elif pattern == "permuted_missing":
        core = [token for token in context if token not in {"[BOS]", "[EOS]"}]
        context = ["[BOS]"] + list(reversed(core)) + ["phase=diagnose", "typed_available=unknown", "feedback_state=unknown", "replay_ready=unknown", "evidence_present=unknown", "[EOS]"]
    else:
        raise ValueError(pattern)
    # Ensure the information-gap slots are present and have no verdict.
    context = [token for token in context if not str(token).startswith(("typed_available=", "feedback_state=", "replay_ready=", "evidence_present="))]
    context = context[:-1] + ["typed_available=unknown", "feedback_state=unknown", "replay_ready=unknown", "evidence_present=unknown", "[EOS]"]
    row = {
        "schema_version": "pg296-missing-pattern-v1",
        "record_id": f"pg296:{pattern}:{sha256_json(context)[:16]}",
        "source_group": "independent_missing_pattern",
        "split": "implementation_holdout",
        "pattern": pattern,
        "context_tokens": context,
        "target_tokens": target(),
        "next_action": "recheck_oracle",
        "repair_action": "recheck_oracle",
        "question": "ask_typed_availability",
        "safe_to_send": False,
        "hard_negative": False,
        "oracle_label_in_context": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "route_identity_stored": False,
        "family_identity_stored": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
    }
    row["record_sha256"] = sha256_json(row)
    return row


def main() -> None:
    source = load(SOURCE)
    base = [row for row in source.get("records", []) if row.get("split") in {"source_holdout", "seed_holdout"}][:12]
    patterns = ("get_query_order_shift", "post_form_decoy", "permuted_missing")
    records = [make_row(row, pattern) for row in base for pattern in patterns]
    dataset = {"schema_version": "pg296-missing-pattern-dataset-v1", "purpose": "frozen causal MoE OOD missing-observation questioning", "source_sha256": source.get("dataset_sha256"), "records": records, "counts": {"total": len(records), "patterns": list(patterns), "training_eligible": 0}, "contract": {"oracle_blind": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "wire_emission_allowed": False, "memory_promotion_allowed": False}}
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    source_report = load(TRAINING_REPORT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    selected_name = str(source_report["selection"]["config_name"])
    selected_variant = next(item for item in source_report["variants"] if item["config_name"] == selected_name)
    config = CausalMoEConfig(**selected_variant["config"])
    vocab = build_vocabulary([row for row in source.get("records", []) if row.get("split") == "train" and row.get("training_eligible") is True])
    model = CausalMoELanguageModel(vocab_size=len(vocab), config=config)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    device = torch.device("cpu")
    metrics = evaluate_causal_moe(model, records, vocab, device)
    report = {"protocol_id": "pg296-missing-pattern-eval-v1", "schema_version": "pg296-missing-pattern-eval-report-v1", "status": "completed_frozen_causal_moe_ood_eval", "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], "training_report": str(TRAINING_REPORT.relative_to(ROOT).as_posix()), "checkpoint": str(CHECKPOINT.relative_to(ROOT).as_posix()), "literal_payload_in_context": False, "wire_emission": False}, "model": {"architecture": "causal_transformer_moe", "config_name": selected_name, "frozen": True}, "split": dataset["counts"], "metrics": metrics, "gate": {"question_required": True, "hard_negative_not_run": True, "promotion_blocked": True, "claim_allowed": False}, "conclusion": "OOD missing patterns test question composition only; no real evaluator or payload capability is established."}
    report["report_sha256"] = sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg296-missing-pattern-eval-protocol-v1", "frozen_checkpoint": True, "patterns": list(patterns), "oracle_blind": True, "training_eligible": False, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-296B: add independent implementation missingness and same-context hard-negative before any retraining."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "report": str(REPORT.relative_to(ROOT).as_posix()), "dataset": str(DATASET.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
