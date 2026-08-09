"""Plan-only structured Rule-IR candidate for the PG-388 logic dataset.

The existing PG-388 token smoke uses independent heads.  This contract plans
an autoregressive slot decoder so that invariant, transition, action and
repair slots are composed in order instead of treated as unrelated labels.
It is deliberately stdlib-only: no optimizer, Docker, network, payload or
evaluator access is present in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg388_logic_canary_trajectory_dataset_v1.json"
SCHEMA_VERSION = "pg388-logic-composed-candidate-plan-v1"
SLOT_ORDER = (
    "question",
    "ask_reason",
    "logic_invariant_ref",
    "state_transition_ref",
    "precondition_ref",
    "counterfactual_ref",
    "probe_variant_ref",
    "next_action",
    "repair_action",
    "oracle_ref",
    "safe_to_send",
)
FORBIDDEN_MARKERS = (
    "http://",
    "https://",
    "payload=",
    "wire=",
    "response_body=",
    "raw_",
    "evaluator=",
)
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("status") != "abstract_canary_trajectory_candidate_only":
        raise ValueError("pg388_dataset_status_mismatch")
    if not isinstance(value.get("rows"), list) or not value["rows"]:
        raise ValueError("pg388_rows_missing")
    return value


def _parse_slots(tokens: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if any(marker in text.casefold() for marker in FORBIDDEN_MARKERS):
            raise ValueError("pg388_model_firewall_marker")
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in SLOT_ORDER:
            parsed[key] = value
    missing = [slot for slot in SLOT_ORDER if slot not in parsed]
    if missing:
        raise ValueError("pg388_slot_missing:" + ",".join(missing))
    return parsed


def _safe_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    context = raw.get("context_tokens")
    target = raw.get("target_tokens")
    if not isinstance(context, list) or not isinstance(target, list) or not context or not target:
        raise ValueError("pg388_row_tokens_missing")
    if raw.get("raw_source_stored") is not False or raw.get("raw_payload_stored") is not False:
        raise ValueError("pg388_raw_source_firewall_open")
    if raw.get("raw_response_body_stored") is not False or raw.get("oracle_answer_in_context") is not False:
        raise ValueError("pg388_oracle_firewall_open")
    context_values = [str(token) for token in context]
    if any(any(marker in token.casefold() for marker in FORBIDDEN_MARKERS) for token in context_values):
        raise ValueError("pg388_context_firewall_open")
    slots = _parse_slots([str(token) for token in target])
    return {"context_tokens": context_values, "slots": slots, "split": str(raw.get("split", ""))}


def _vocabulary(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(token) for row in rows for token in row["context_tokens"]}


def _slot_classes(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    return {slot: {str(row["slots"][slot]) for row in rows} for slot in SLOT_ORDER}


def build_plan(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    d_model: int = 256,
    n_layers: int = 4,
    experts: int = 4,
    expert_hidden: int = 512,
    slot_decoder_layers: int = 2,
    max_length: int = 256,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    rows = [_safe_row(raw) for raw in dataset["rows"] if isinstance(raw, Mapping)]
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "implementation_holdout"]
    if not train or not holdout:
        raise ValueError("pg388_split_empty")
    vocab = _vocabulary(train)
    holdout_unknown = sorted({token for row in holdout for token in row["context_tokens"] if token not in vocab})
    train_classes = _slot_classes(train)
    unknown_slots = {
        slot: sorted({row["slots"][slot] for row in holdout if row["slots"][slot] not in train_classes[slot]})
        for slot in SLOT_ORDER
    }
    unknown_slots = {slot: values for slot, values in unknown_slots.items() if values}
    required_window = max((len(row["context_tokens"]) for row in rows), default=1)
    source_contract = dataset.get("source_contract") if isinstance(dataset.get("source_contract"), Mapping) else {}
    failures = []
    if dataset.get("training_eligible") != 0:
        failures.append("dataset_training_flag_mismatch")
    if any(source_contract.get(key) is not True for key in ("candidate_reference_negative_replay",)):
        failures.append("role_contract_incomplete")
    # The PG-388 trajectory is intentionally synthetic/abstract and is not a
    # typed source-row grant.  Keep this gate explicit instead of inferring
    # training permission from counts or clean vocabulary coverage.
    failures.extend(["typed_evaluator_not_attested", "fresh_role_reset_not_attested", "operator_review_not_attested"])
    if holdout_unknown:
        failures.append("holdout_context_vocabulary_gap")
    if unknown_slots:
        failures.append("holdout_slot_vocabulary_gap")
    if max_length < required_window:
        failures.append("configured_context_window_too_small")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_capability_contract" if failures else "ready_static_candidate_design",
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "counts": {"records": len(rows), "train": len(train), "implementation_holdout": len(holdout), "slots": len(SLOT_ORDER)},
        "slot_order": list(SLOT_ORDER),
        "train_only_vocabulary": {"size": len(vocab), "unknown_holdout_context_count": len(holdout_unknown)},
        "unknown_holdout_slot_values": {slot: _sha(values) for slot, values in sorted(unknown_slots.items())},
        "required_context_window": required_window,
        "model_design": {
            "backbone": "decoder_only_causal_moe",
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "experts": int(experts),
            "expert_hidden": int(expert_hidden),
            "slot_decoder": "autoregressive_causal_previous_slot_conditioned",
            "slot_decoder_layers": int(slot_decoder_layers),
            "auxiliary_heads": ["ask", "repair", "negative_control"],
            "target_mask": "target_only_next_token_plus_teacher_forced_slot_composition",
        },
        "gate": {"failures": sorted(set(failures)), "optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_contacted": False, "wire_created": False},
        "training_eligible": 0,
        "capability_training_allowed": False,
        "promotion": dict(PROMOTION),
    }
    report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def write_plan(path: str | Path, **kwargs: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(**kwargs), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="research/pg388_logic_composed_candidate_plan_v1.json")
    args = parser.parse_args()
    output = write_plan(args.output, dataset_path=Path(args.dataset))
    print(json.dumps(json.loads(output.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
