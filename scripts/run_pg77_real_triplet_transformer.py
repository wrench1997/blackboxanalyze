"""PG-77: real causal-triplet bridge for the abstract Trace Transformer.

This is an evaluation-only bridge from the fresh PG-74/PG-76 traces into the
existing abstract causal Transformer.  It deliberately keeps the typed oracle
after ``ORACLE_TARGET`` and trains a small Rule-IR head on the hidden state at
that boundary.  Family names, source identifiers, route words, raw probes and
response bodies never enter model tokens.  PG-76 remains a strict unknown
holdout and is never used for weight updates or memory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


PG56_CHECKPOINT = ROOT / "artifacts" / "pg56-causal-trace-transformer" / "model.pt"
PG74_TRACE = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
PG76_TRACE = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
PG72_TRACE = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg77_real_triplet_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg77_real_triplet_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg77_real_triplet_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg77_real_triplet_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg77_real_triplet_transformer_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg77-real-triplet-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
SEED = 20770403
TRAIN_SEEDS = (74101, 74102)
DEV_SEEDS = (74103,)
CONFIDENCE_THRESHOLD = 0.70
OOD_MARGIN = 0.25
CLIP = 6.0
CLASSES = ("confirm", "reject")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bucket(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "u"
    if number <= 0:
        return "0"
    if number == 1:
        return "1"
    if number == 2:
        return "2"
    return "3p"


def _signed_delta(before: Any, after: Any) -> str:
    try:
        delta = float(after) - float(before)
    except (TypeError, ValueError):
        delta = 0.0
    return "neg" if delta < 0 else "pos" if delta > 0 else "zero"


def _status_class(response: dict[str, Any]) -> str:
    value = str(response.get("status_class", "unknown")).upper().replace(" ", "_")
    return value if value else "UNKNOWN"


def _content_class(response: dict[str, Any]) -> str:
    value = str(response.get("content_type_class", response.get("content_type", "unknown"))).lower()
    if "html" in value:
        return "HTML"
    if "json" in value:
        return "JSON"
    if "text" in value:
        return "TEXT"
    return "OTHER" if value else "UNKNOWN"


def _shape_tokens(prefix: str, response: dict[str, Any]) -> list[str]:
    shape = response.get("shape") or response.get("json_shape") or {}
    dom = response.get("dom_shape") or {}
    return [
        f"{prefix}_STATUS_{_status_class(response)}",
        f"{prefix}_CONTENT_{_content_class(response)}",
        f"{prefix}_LENGTH_{str(response.get('body_length_bucket', 'unknown')).upper().replace('-', '_').replace('+', 'P')}",
        f"{prefix}_HTML_{_bucket(response.get('html_tag_count'))}",
        f"{prefix}_FORMS_{_bucket(response.get('form_count'))}",
        f"{prefix}_INPUTS_{_bucket(response.get('input_count'))}",
        f"{prefix}_SCRIPTS_{_bucket(response.get('script_count'))}",
        f"{prefix}_ROWS_{_bucket(response.get('result_row_count'))}",
        f"{prefix}_JSON_KIND_{str(shape.get('kind', 'none')).upper()}",
        f"{prefix}_JSON_KEYS_{_bucket(shape.get('key_count'))}",
        f"{prefix}_JSON_SCALARS_{_bucket(shape.get('scalar_count'))}",
        f"{prefix}_JSON_ARRAYS_{_bucket(shape.get('array_count'))}",
        f"{prefix}_DOM_NODES_{_bucket(dom.get('node_count'))}",
        f"{prefix}_DOM_SVG_{_bucket(dom.get('svg_count'))}",
        f"{prefix}_DOM_EVENTS_{_bucket(dom.get('event_handler_attribute_count'))}",
        f"{prefix}_LOCATION_{str(response.get('location_origin', 'none')).upper()}",
    ]


def _delta_tokens(prefix: str, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    shape_before = before.get("shape") or before.get("json_shape") or {}
    shape_after = after.get("shape") or after.get("json_shape") or {}
    dom_before = before.get("dom_shape") or {}
    dom_after = after.get("dom_shape") or {}
    pairs = [
        ("HTML", before.get("html_tag_count", 0), after.get("html_tag_count", 0)),
        ("FORMS", before.get("form_count", 0), after.get("form_count", 0)),
        ("INPUTS", before.get("input_count", 0), after.get("input_count", 0)),
        ("SCRIPTS", before.get("script_count", 0), after.get("script_count", 0)),
        ("ROWS", before.get("result_row_count", 0), after.get("result_row_count", 0)),
        ("JSON_KEYS", shape_before.get("key_count", 0), shape_after.get("key_count", 0)),
        ("JSON_SCALARS", shape_before.get("scalar_count", 0), shape_after.get("scalar_count", 0)),
        ("JSON_ARRAYS", shape_before.get("array_count", 0), shape_after.get("array_count", 0)),
        ("DOM_NODES", dom_before.get("node_count", 0), dom_after.get("node_count", 0)),
        ("DOM_SVG", dom_before.get("svg_count", 0), dom_after.get("svg_count", 0)),
        ("DOM_EVENTS", dom_before.get("event_handler_attribute_count", 0), dom_after.get("event_handler_attribute_count", 0)),
    ]
    tokens = [f"{prefix}_STATUS_CHANGE_{int(_status_class(before) != _status_class(after))}", f"{prefix}_CONTENT_CHANGE_{int(_content_class(before) != _content_class(after))}", f"{prefix}_LENGTH_CHANGE_{int(before.get('body_length_bucket') != after.get('body_length_bucket'))}", f"{prefix}_REFLECTION_CHANGE_{int(bool(before.get('marker_reflected')) != bool(after.get('marker_reflected')))}", f"{prefix}_LOCATION_CHANGE_{int(str(before.get('location_origin', 'none')) != str(after.get('location_origin', 'none')))}"]
    tokens.extend(f"{prefix}_{name}_{_signed_delta(left, right)}" for name, left, right in pairs)
    return tokens


def _oracle_modality(oracle: dict[str, Any]) -> str:
    modality = str(oracle.get("modality", "unknown")).lower()
    if "dom" in modality or "browser" in modality:
        return "DOM"
    if "sql" in modality or "ast" in modality:
        return "AST"
    if "redirect" in modality:
        return "REDIRECT"
    if "workflow" in modality or "boundary" in modality:
        return "BOUNDARY"
    return "OTHER"


def _row_tokens(step: dict[str, Any], candidate: dict[str, Any], oracle: dict[str, Any]) -> tuple[list[str], int]:
    method = str((step.get("action_manifest") or {}).get("method", "GET")).upper()
    neutral = dict(step.get("neutral_projection") or step.get("baseline_projection") or {})
    negative = dict(step.get("negative_probe_projection") or neutral)
    tokens = [
        "BOS",
        f"CHANNEL_{method}",
        "CONTROL_TARGET",
        *_shape_tokens("CONTROL", neutral),
        "SCREEN_TARGET",
        *_shape_tokens("SCREEN", negative),
        *_delta_tokens("SCREEN_DIFF", neutral, negative),
        "CANDIDATE_TARGET",
        *_shape_tokens("CANDIDATE", candidate),
        *_delta_tokens("CANDIDATE_DIFF", neutral, candidate),
        f"BELIEF_ACTION_{method}",
        "BELIEF_IG_0",
        "BELIEF_DUP_0",
        "NEXT_ACTION_TARGET",
        f"ACTION_{method}_CANDIDATE_CANDIDATE",
        "ORACLE_TARGET",
    ]
    oracle_index = len(tokens) - 1
    modality = _oracle_modality(oracle)
    outcome = bool(oracle.get("positive"))
    tokens.extend([f"ORACLE_MODALITY_{modality}", f"ORACLE_OUTCOME_{'POSITIVE' if outcome else 'NEGATIVE'}", "RULE_IR_TARGET", f"RULE_EFFECT_{'CONFIRMED' if outcome else 'REJECTED'}", f"RULE_TRANSPORT_{method}", f"RULE_ORACLE_{modality}", "EOS"])
    return tokens, oracle_index


def _make_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pg74 = _read(PG74_TRACE)
    pg76 = _read(PG76_TRACE)
    pg72 = _read(PG72_TRACE)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []

    def append_pair(target: list[dict[str, Any]], step: dict[str, Any], split: str, unknown_holdout: bool = False) -> None:
        variants = [("positive", dict(step.get("response_projection") or {}), dict(step.get("oracle_projection") or {}))]
        if not unknown_holdout and step.get("negative_probe_projection") is not None:
            variants.append(("negative", dict(step.get("negative_probe_projection") or {}), dict(step.get("negative_oracle_projection") or {})))
        for role, projection, oracle in variants:
            tokens, oracle_index = _row_tokens(step, projection, oracle)
            target.append({"trace_id": f"{step['step_id']}-{role}", "split": split, "role": role, "sampling_seed": step.get("sampling_seed"), "tokens": tokens, "oracle_index": oracle_index, "expected": "abstain" if unknown_holdout else ("confirm" if bool(oracle.get("positive")) else "reject"), "source_trace": "pg76" if unknown_holdout else "pg74", "raw_probe_stored": False, "raw_response_stored": False})

    for step in pg74.get("steps", []):
        seed = int(step.get("sampling_seed", -1))
        append_pair(train if seed in TRAIN_SEEDS else dev if seed in DEV_SEEDS else dev, step, "train" if seed in TRAIN_SEEDS else "dev")
    for step in pg76.get("steps", []):
        append_pair(unknown, step, "unknown_family_holdout", unknown_holdout=True)
    for step in pg72.get("steps", []):
        tokens, oracle_index = _row_tokens(step, dict(step.get("response_projection") or {}), {"positive": True, "modality": "unknown"})
        external.append({"trace_id": f"{step['step_id']}-external", "split": "external_known_schema_diagnostic", "role": "positive", "sampling_seed": step.get("sampling_seed"), "tokens": tokens, "oracle_index": oracle_index, "expected": "confirm", "source_trace": "pg72", "raw_probe_stored": False, "raw_response_stored": False})
    return train, dev, unknown, external


class RuleIRHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def _load_pretrained(vocabulary: dict[str, int], device: torch.device) -> CausalTraceTransformer:
    checkpoint = torch.load(PG56_CHECKPOINT, map_location="cpu", weights_only=False)
    model = CausalTraceTransformer(len(vocabulary), d_model=96, nhead=4, layers=2, max_len=128)
    old_vocab = dict(checkpoint["vocabulary"])
    old_state = checkpoint["model_state"]
    new_state = model.state_dict()
    for key, value in old_state.items():
        if key in {"token_embedding.weight", "lm_head.weight", "lm_head.bias"}:
            continue
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key].copy_(value)
    for token, old_index in old_vocab.items():
        if token not in vocabulary:
            continue
        new_index = vocabulary[token]
        new_state["token_embedding.weight"][new_index].copy_(old_state["token_embedding.weight"][old_index])
        new_state["lm_head.weight"][new_index].copy_(old_state["lm_head.weight"][old_index])
        new_state["lm_head.bias"][new_index].copy_(old_state["lm_head.bias"][old_index])
    model.load_state_dict(new_state)
    return model.to(device)


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad = vocabulary["<PAD>"]
    unk = vocabulary["<UNK>"]
    max_len = max(len(row["tokens"]) for row in rows)
    if max_len > 128:
        raise RuntimeError(f"PG-77 sequence exceeds transformer max_len: {max_len}")
    ids = torch.full((len(rows), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    oracle_positions: list[int] = []
    for index, row in enumerate(rows):
        encoded = [vocabulary.get(token, unk) for token in row["tokens"]]
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
        oracle_positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(oracle_positions, dtype=torch.long)


def _lm_loss(model: CausalTraceTransformer, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(ids, mask)
    target = ids[:, 1:]
    valid = mask[:, 1:]
    return nn.functional.cross_entropy(logits[:, :-1][valid], target[valid])


def _hidden_at_oracle(model: CausalTraceTransformer, ids: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    hidden = model.encode(ids, mask)
    return hidden[torch.arange(len(ids)), positions]


def _calibrate_threshold(train_hidden: torch.Tensor, dev_hidden: torch.Tensor) -> float:
    if len(train_hidden) > 1:
        distances = torch.cdist(train_hidden, train_hidden)
        distances.fill_diagonal_(float("inf"))
        train_max = float(distances.min(dim=1).values.max())
    else:
        train_max = 0.0
    dev_max = float(torch.cdist(dev_hidden, train_hidden).min(dim=1).values.max()) if len(dev_hidden) else train_max
    return round(max(train_max, dev_max) + OOD_MARGIN, 6)


def _evaluate_head(head: RuleIRHead, hidden: torch.Tensor, rows: list[dict[str, Any]], reference: torch.Tensor, threshold: float, unknown: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with torch.inference_mode():
        probabilities = torch.softmax(head(hidden), dim=-1)
    details: list[dict[str, Any]] = []
    for index, (row, probability) in enumerate(zip(rows, probabilities)):
        confidence, predicted = torch.max(probability, dim=0)
        distance = float(torch.cdist(hidden[index:index + 1], reference).min()) if len(reference) else float("inf")
        raw = CLASSES[int(predicted)]
        # Unknown rows must pass the same calibrated OOD/confidence rule as
        # known rows; ``unknown`` only changes aggregation, never the action.
        decision = "abstain" if distance >= threshold or float(confidence) < CONFIDENCE_THRESHOLD else raw
        details.append({"trace_id": row["trace_id"], "role": row["role"], "expected": row["expected"], "raw_prediction": raw, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6), "unknown": unknown})
    if unknown:
        return {"count": len(details), "misname_count": sum(int(item["decision"] != "abstain") for item in details), "strict_abstain": all(item["decision"] == "abstain" for item in details), "min_ood_distance": round(min((item["ood_distance"] for item in details), default=0.0), 6)}, details
    positives = [item for item in details if item["expected"] == "confirm"]
    return {"count": len(details), "accuracy": round(sum(int(item["decision"] == item["expected"]) for item in details) / max(len(details), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in positives) / max(len(positives), 1), 6), "abstain_count": sum(int(item["decision"] == "abstain") for item in details)}, details


def run() -> dict[str, Any]:
    if not PG56_CHECKPOINT.exists():
        raise RuntimeError("PG-77 requires the PG-56 pretrained checkpoint")
    train_rows, dev_rows, unknown_rows, external_rows = _make_rows()
    all_tokens = sorted({token for rows in (train_rows, dev_rows, unknown_rows, external_rows) for row in rows for token in row["tokens"]})
    old_checkpoint = torch.load(PG56_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = dict(old_checkpoint["vocabulary"])
    for token in all_tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    dataset = {"schema_version": "pg77-real-triplet-trace-dataset-v1", "dataset_id": "pg77-real-causal-triplets", "training_eligible": False, "evaluation_only": True, "model_input_contract": {"family_name_in_tokens": False, "source_id_in_tokens": False, "implementation_in_tokens": False, "route_words_in_tokens": False, "raw_probe_in_tokens": False, "raw_response_body_in_tokens": False, "typed_oracle_before_target_marker": False, "evaluator_target_is_metadata_only": True}, "split_counts": {"train": len(train_rows), "dev": len(dev_rows), "unknown_family_holdout": len(unknown_rows), "external_known_schema_diagnostic": len(external_rows)}, "vocabulary_size": len(vocabulary), "rows": train_rows + dev_rows + unknown_rows + external_rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    dataset["dataset_sha256"] = hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = _load_pretrained(vocabulary, device)
    train_ids, train_mask, train_positions = _encode(train_rows, vocabulary)
    dev_ids, dev_mask, dev_positions = _encode(dev_rows, vocabulary)
    unknown_ids, unknown_mask, unknown_positions = _encode(unknown_rows, vocabulary)
    external_ids, external_mask, external_positions = _encode(external_rows, vocabulary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, 251):
        model.train()
        loss = _lm_loss(model, train_ids.to(device), train_mask.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            model.eval()
            with torch.inference_mode():
                dev_loss = float(_lm_loss(model, dev_ids.to(device), dev_mask.to(device)).detach().cpu())
            history.append({"epoch": epoch, "train_loss": round(float(loss.detach().cpu()), 6), "dev_loss": round(dev_loss, 6)})
            if dev_loss < best_dev:
                best_dev = dev_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_hidden = _hidden_at_oracle(model, train_ids.to(device), train_mask.to(device), train_positions.to(device)).detach()
        dev_hidden = _hidden_at_oracle(model, dev_ids.to(device), dev_mask.to(device), dev_positions.to(device)).detach()
        unknown_hidden = _hidden_at_oracle(model, unknown_ids.to(device), unknown_mask.to(device), unknown_positions.to(device)).detach()
        external_hidden = _hidden_at_oracle(model, external_ids.to(device), external_mask.to(device), external_positions.to(device)).detach()
    labels = torch.tensor([0 if row["expected"] == "confirm" else 1 for row in train_rows], dtype=torch.long, device=device)
    dev_labels = torch.tensor([0 if row["expected"] == "confirm" else 1 for row in dev_rows], dtype=torch.long, device=device)
    head = RuleIRHead(train_hidden.shape[-1]).to(device)
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.02)
    best_head: dict[str, torch.Tensor] | None = None
    best_head_loss = float("inf")
    for epoch in range(1, 401):
        head.train()
        head_loss = nn.functional.cross_entropy(head(train_hidden.detach()), labels)
        head_optimizer.zero_grad(set_to_none=True)
        head_loss.backward()
        head_optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            head.eval()
            with torch.inference_mode():
                loss_value = float(nn.functional.cross_entropy(head(dev_hidden.detach()), dev_labels).detach().cpu())
            if loss_value < best_head_loss:
                best_head_loss = loss_value
                best_head = copy.deepcopy({key: value.detach().cpu() for key, value in head.state_dict().items()})
    if best_head is not None:
        head.load_state_dict(best_head)
    threshold = _calibrate_threshold(train_hidden.detach(), dev_hidden.detach())
    train_metrics, train_details = _evaluate_head(head, train_hidden.detach(), train_rows, train_hidden.detach(), threshold)
    dev_metrics, dev_details = _evaluate_head(head, dev_hidden.detach(), dev_rows, train_hidden.detach(), threshold)
    unknown_metrics, unknown_details = _evaluate_head(head, unknown_hidden.detach(), unknown_rows, train_hidden.detach(), threshold, unknown=True)
    external_metrics, external_details = _evaluate_head(head, external_hidden.detach(), external_rows, train_hidden.detach(), threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg77-real-triplet-transformer-checkpoint-v1", "vocabulary": vocabulary, "max_len": 128, "transformer_state": model.state_dict(), "rule_ir_head_state": head.state_dict(), "hidden_dim": int(train_hidden.shape[-1]), "ood_distance_threshold": threshold, "confidence_threshold": CONFIDENCE_THRESHOLD, "source_checkpoint": str(PG56_CHECKPOINT.relative_to(ROOT)), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}, CHECKPOINT_PATH)
    checks = {"dev_confirm_recall": float(dev_metrics.get("confirm_recall", 0.0)) >= 0.80, "dev_false_accept_zero": int(dev_metrics.get("false_accept_count", 1)) == 0, "unknown_strict_abstain": bool(unknown_metrics.get("strict_abstain", False)), "external_known_recall_min": float(external_metrics.get("confirm_recall", 0.0)) >= 0.80}
    report = {"protocol_id": "pg-pk-77-real-triplet-transformer-v1", "schema_version": "pg77-real-triplet-transformer-report-v1", "status": "candidate_evaluation_completed", "source": {"training_trace": str(PG74_TRACE.relative_to(ROOT)), "unknown_holdout_trace": str(PG76_TRACE.relative_to(ROOT)), "external_known_diagnostic_trace": str(PG72_TRACE.relative_to(ROOT)), "base_pretraining_checkpoint": str(PG56_CHECKPOINT.relative_to(ROOT)), "model_retrained_on_unknown_family": False, "family_in_tokens": False, "oracle_in_tokens_before_target": False, "device": str(device)}, "dataset": dataset["split_counts"] | {"train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS), "vocabulary_size": len(vocabulary), "max_sequence_length": max(len(row["tokens"]) for row in dataset["rows"])}, "metrics": {"train": train_metrics, "dev_holdout": dev_metrics, "unknown_family_holdout": unknown_metrics, "external_known_schema_diagnostic": external_metrics}, "details": {"train": train_details, "dev_holdout": dev_details, "unknown_family_holdout": unknown_details, "external_known_schema_diagnostic": external_details}, "training": {"pretraining_epochs": 250, "rule_ir_head_epochs": 400, "history_tail": history[-5:], "ood_distance_threshold": threshold, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "online_weight_update": False, "long_term_memory_write": False}, "capability_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_evaluation_only", "reason": "PG-77 uses a tiny real-triplet bridge; require fresh multi-implementation replay and projection-contract gate before promotion"}, "formal_claim": {"allowed": False, "reason": "not a broad web vulnerability detector"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg77-real-triplet-transformer-trace-v1", "evaluation_only": True, "training_eligible": False, "rows": [{"trace_id": row["trace_id"], "split": row["split"], "role": row["role"], "source_trace": row["source_trace"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False} for row in dataset["rows"]], "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": "pg-pk-77-real-triplet-transformer-v1", "schema_version": "pg77-real-triplet-transformer-protocol-v1", "input_contract": dataset["model_input_contract"], "split_contract": {"train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS), "unknown_family_training_forbidden": True, "external_known_schema_diagnostic_only": True, "typed_oracle_after_target_marker_only": True, "raw_persistence_forbidden": True}, "required_gates": {"dev_confirm_recall_min": 0.80, "dev_false_accept_zero": True, "unknown_strict_abstain": True, "external_known_recall_min": 0.80, "fresh_multi_implementation_replay_required": True, "projection_contract_gate_required": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG78 fresh multi-implementation unified triplet replay"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-77 Real triplet Trace Transformer bridge\n\n" + f"train/dev/unknown/external={len(train_rows)}/{len(dev_rows)}/{len(unknown_rows)}/{len(external_rows)}；device={device}；vocab={len(vocabulary)}；OOD={threshold}。\n\ndev recall={dev_metrics.get('confirm_recall', 0.0)}；unknown strict abstain={unknown_metrics.get('strict_abstain', False)}；external known recall={external_metrics.get('confirm_recall', 0.0)}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "capability_gate": result["capability_gate"]["status"], "dev_confirm_recall": result["metrics"]["dev_holdout"].get("confirm_recall", 0.0), "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"].get("strict_abstain", False), "external_known_recall": result["metrics"]["external_known_schema_diagnostic"].get("confirm_recall", 0.0), "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
