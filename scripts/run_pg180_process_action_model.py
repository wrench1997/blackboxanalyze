"""PG-180: train a capacity-swept next-action token model on PG-179B.

The experiment trains only the abstract process policy.  It never turns a
surface signal into a vulnerability label and never stores raw probe or body
values.  Splits are by episode/surface so a held-out route cannot leak through
the model input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg180_process_action_model import (  # noqa: E402
    ACTION_TOKENS,
    ALLOWED_ACTIONS,
    MODEL_VARIANTS,
    SCHEMA_VERSION,
    build_model,
    build_vocabulary,
    collate,
    encode_examples,
    example_tokens,
    last_logits,
    parameter_count,
    restrict_action,
)


TRACE_PATH = ROOT / "research" / "pg179b_pikachu_iterative_trace_v1.json"
CATALOG_PATH = ROOT / "research" / "pg179b_pikachu_iterative_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg180_process_action_model_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg180_process_action_model_protocol_v1.json"
TRACE_OUT_PATH = ROOT / "research" / "pg180_process_action_model_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg180_process_action_model_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg180-process-action-v1"

SEEDS = (18001, 18002)
EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 12
SPLITS = (
    {"name": "url_holdout", "test": ("url_redirect_get",), "dev": ("xss_blind_post",)},
    {"name": "xss_holdout", "test": ("xss_stored_post", "xss_blind_post"), "dev": ("sqli_id_post",)},
    {"name": "injection_holdout", "test": ("sqli_delete_post", "sqli_header_post", "sqli_id_post", "sqli_widebyte_post"), "dev": ("url_redirect_get",)},
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _surface_for_episode(episode_id: str) -> str:
    prefix = "pg179b-pikachu-"
    if not str(episode_id).startswith(prefix):
        raise ValueError("PG-180 received an ungrounded episode id")
    return str(episode_id)[len(prefix):]


def _load_examples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not bool(trace.get("evaluation_only")) or bool(trace.get("training_eligible")):
        raise RuntimeError("PG-180 requires the evaluation-only PG-179B trace")
    if bool(trace.get("raw_probe_strings_stored")) or bool(trace.get("raw_response_bodies_stored")):
        raise RuntimeError("PG-180 refuses a trace with raw request/response retention")
    if bool(trace.get("invented_parameter_names")) or bool(catalog.get("channel_grounding", {}).get("invented_parameter_names")):
        raise RuntimeError("PG-180 refuses a trace with invented parameter names")
    rows: list[dict[str, Any]] = []
    episode_summaries = {str(item["episode_id"]): dict(item) for item in trace.get("episodes", [])}
    grouped_steps: dict[str, list[dict[str, Any]]] = {}
    for raw_step in trace.get("steps", []):
        grouped_steps.setdefault(str(raw_step["episode_id"]), []).append(dict(raw_step))
    for episode_id, episode_steps in grouped_steps.items():
        surface = _surface_for_episode(episode_id)
        history: list[dict[str, Any]] = []
        contract = dict(episode_summaries.get(episode_id, {}).get("method_contract") or {})
        for step in episode_steps:
            context, target = example_tokens(step, history=history)
            rows.append(
                {
                    "surface": surface,
                    "episode_id_hash": hashlib.sha256(str(step["episode_id"]).encode("utf-8")).hexdigest(),
                    "context": context,
                    "target": target,
                    "single_channel": not bool(contract.get("dual_channel")),
                    "parameterized_method": str(contract.get("parameterized_method", "unknown")),
                }
            )
            history.append(dict(step))
    if len(rows) != 35:
        raise RuntimeError(f"PG-180 expected 35 grounded process rows, got {len(rows)}")
    # A vocabulary is schema-level only; targets are restricted to action
    # tokens and no route/family/response value is placed in it.
    vocabulary = build_vocabulary([(row["context"], row["target"]) for row in rows])
    return rows, {"trace": trace, "catalog": catalog, "vocabulary": vocabulary}


def _split_rows(rows: list[dict[str, Any]], split: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    test_surfaces = set(split["test"])
    dev_surfaces = set(split["dev"])
    if test_surfaces & dev_surfaces:
        raise ValueError("PG-180 split overlaps dev and test surfaces")
    train = [row for row in rows if row["surface"] not in test_surfaces | dev_surfaces]
    dev = [row for row in rows if row["surface"] in dev_surfaces]
    test = [row for row in rows if row["surface"] in test_surfaces]
    if not train or not dev or not test:
        raise ValueError(f"PG-180 split is empty: {split}")
    return train, dev, test


def _batch_rows(rows: list[dict[str, Any]], vocabulary: dict[str, int], batch_size: int, *, shuffle: bool, seed: int) -> list[list[dict[str, Any]]]:
    ordered = list(rows)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    return [ordered[index:index + batch_size] for index in range(0, len(ordered), batch_size)]


def _evaluate(model: torch.nn.Module, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        return {"count": 0}, []
    encoded = encode_examples([(row["context"], row["target"]) for row in rows], vocabulary)
    model.eval()
    details: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(0, len(rows), BATCH_SIZE):
            batch_rows = encoded[index:index + BATCH_SIZE]
            ids, mask, _ = collate(batch_rows)
            logits = last_logits(model, ids.to(device), mask.to(device)).detach().cpu()
            for offset, vector in enumerate(logits):
                row = rows[index + offset]
                raw_action, confidence = restrict_action(vector, vocabulary)
                safe_action, _ = restrict_action(vector, vocabulary, context={"single_channel": row["single_channel"]})
                expected = row["target"].split("::", 1)[1]
                details.append(
                    {
                        "surface": row["surface"],
                        "expected_action": expected,
                        "raw_action": raw_action,
                        "safe_action": safe_action,
                        "confidence": round(confidence, 6),
                        "single_channel": bool(row["single_channel"]),
                    }
                )
    targets = [item["expected_action"] for item in details]
    safe_predictions = [item["safe_action"] for item in details]
    raw_predictions = [item["raw_action"] for item in details]
    abstain_targets = [index for index, value in enumerate(targets) if value == "abstain_unknown_oracle"]
    metrics = {
        "count": len(details),
        "safe_accuracy": round(sum(int(a == b) for a, b in zip(safe_predictions, targets)) / len(details), 6),
        "raw_accuracy": round(sum(int(a == b) for a, b in zip(raw_predictions, targets)) / len(details), 6),
        "abstain_recall": round(sum(int(safe_predictions[index] == "abstain_unknown_oracle") for index in abstain_targets) / max(len(abstain_targets), 1), 6),
        "single_channel_probe_block_count": sum(int(item["single_channel"] and item["raw_action"] == "probe_candidate_other_method" and item["safe_action"] == "abstain_unknown_oracle") for item in details),
        "safe_action_not_allowlisted_count": sum(int(item["safe_action"] not in ALLOWED_ACTIONS) for item in details),
        "target_distribution": dict(Counter(targets)),
        "prediction_distribution": dict(Counter(safe_predictions)),
        "mean_confidence": round(statistics.mean(item["confidence"] for item in details), 6),
    }
    return metrics, details


def _train_one(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], vocabulary: dict[str, int], *, variant: str, seed: int, device: torch.device, split_name: str) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(len(vocabulary), variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if variant != "moe_large" else 2e-4, weight_decay=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_loss = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_values: list[float] = []
        for batch_index, batch_rows in enumerate(_batch_rows(train_rows, vocabulary, BATCH_SIZE, shuffle=True, seed=seed + epoch)):
            batch_encoded = encode_examples([(row["context"], row["target"]) for row in batch_rows], vocabulary)
            ids, mask, targets = collate(batch_encoded)
            ids = ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            logits = last_logits(model, ids, mask)
            action_ids = torch.tensor([vocabulary[token] for token in ACTION_TOKENS], device=device)
            action_logits = logits.index_select(1, action_ids)
            target_indices = torch.tensor([list(ACTION_TOKENS).index(row["target"]) for row in batch_rows], dtype=torch.long, device=device)
            loss = F.cross_entropy(action_logits, target_indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            dev_loss_values: list[float] = []
            with torch.inference_mode():
                for batch_index, batch_rows in enumerate(_batch_rows(dev_rows, vocabulary, BATCH_SIZE, shuffle=False, seed=seed)):
                    batch_encoded = encode_examples([(row["context"], row["target"]) for row in batch_rows], vocabulary)
                    ids, mask, targets = collate(batch_encoded)
                    logits = last_logits(model, ids.to(device), mask.to(device))
                    action_ids = torch.tensor([vocabulary[token] for token in ACTION_TOKENS], device=device)
                    action_logits = logits.index_select(1, action_ids)
                    target_indices = torch.tensor([list(ACTION_TOKENS).index(row["target"]) for row in batch_rows], dtype=torch.long, device=device)
                    dev_loss_values.append(float(F.cross_entropy(action_logits, target_indices).detach().cpu()))
            dev_loss = statistics.mean(dev_loss_values) if dev_loss_values else float("inf")
            history.append({"epoch": epoch, "train_loss": round(statistics.mean(loss_values), 8), "dev_loss": round(dev_loss, 8)})
            if dev_loss < best_dev_loss - 1e-6:
                best_dev_loss = dev_loss
                stale = 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics, _ = _evaluate(model, train_rows, vocabulary, device)
    dev_metrics, _ = _evaluate(model, dev_rows, vocabulary, device)
    test_metrics, test_details = _evaluate(model, test_rows, vocabulary, device)
    checkpoint_path = ARTIFACT_DIR / split_name / f"{variant}_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "variant": variant, "seed": seed, "vocabulary": vocabulary, "model_config": MODEL_VARIANTS[variant], "model_state": model.state_dict(), "raw_input_retained": False}, checkpoint_path)
    result = {
        "split": split_name,
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "parameter_count": parameter_count(model),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "test_count": len(test_rows),
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "history_tail": history[-5:],
        "train": train_metrics,
        "dev": dev_metrics,
        "test": test_metrics,
        "test_details": test_details,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_input_retained": False,
        "oracle_in_input": False,
        "family_in_input": False,
        "route_in_input": False,
        "vulnerability_label_in_input": False,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    rows, loaded = _load_examples()
    vocabulary = loaded["vocabulary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    split_metadata: list[dict[str, Any]] = []
    for split in SPLITS:
        train_rows, dev_rows, test_rows = _split_rows(rows, split)
        split_metadata.append({"name": split["name"], "train_surfaces": sorted({row["surface"] for row in train_rows}), "dev_surfaces": sorted({row["surface"] for row in dev_rows}), "test_surfaces": sorted({row["surface"] for row in test_rows}), "train_count": len(train_rows), "dev_count": len(dev_rows), "test_count": len(test_rows)})
        for variant in MODEL_VARIANTS:
            for seed in SEEDS:
                results.append(_train_one(train_rows, dev_rows, test_rows, vocabulary, variant=variant, seed=seed, device=device, split_name=str(split["name"])))
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in MODEL_VARIANTS:
        variant_results = [item for item in results if item["variant"] == variant]
        test_scores = [float(item["test"]["safe_accuracy"]) for item in variant_results]
        abstain_scores = [float(item["test"]["abstain_recall"]) for item in variant_results]
        by_variant[variant] = {"parameter_count": variant_results[0]["parameter_count"], "run_count": len(variant_results), "mean_test_safe_accuracy": round(statistics.mean(test_scores), 6), "min_test_safe_accuracy": round(min(test_scores), 6), "mean_test_abstain_recall": round(statistics.mean(abstain_scores), 6), "all_safe_action_allowlisted": all(int(item["test"]["safe_action_not_allowlisted_count"]) == 0 for item in variant_results), "all_input_contracts_clean": all(not item["raw_input_retained"] and not item["oracle_in_input"] and not item["family_in_input"] and not item["route_in_input"] and not item["vulnerability_label_in_input"] for item in variant_results)}
    capacity_order = sorted(by_variant, key=lambda key: by_variant[key]["parameter_count"])
    report = {
        "protocol_id": "pg-pk-180-process-action-model-v1",
        "schema_version": "pg180-process-action-model-report-v1",
        "status": "completed_process_policy_diagnostic",
        "objective": "根据脱敏失败/响应形状 token 预测下一安全抽象动作；不预测漏洞阳性",
        "source": {"trace": str(TRACE_PATH.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT)), "trace_sha256": _sha256_json(loaded["trace"]), "catalog_sha256": _sha256_json(loaded["catalog"]), "device": str(device)},
        "dataset": {"row_count": len(rows), "episode_count": len(loaded["trace"]["episodes"]), "vocabulary_size": len(vocabulary), "action_vocabulary": list(ALLOWED_ACTIONS), "splits": split_metadata, "model_input_contract": ["method_token", "placement_token", "encoding_token", "failure_token", "failed_gate_token", "candidate_signal_token", "typed_availability_token", "response_shape_token", "status_chain_length_token", "belief_token", "bounded_budget_token"], "excluded_from_model_input": ["route_path", "family_label", "raw_probe", "raw_response_body", "oracle_authority", "vulnerability_label"]},
        "capacity": by_variant,
        "capacity_order": capacity_order,
        "runs": results,
        "selection": {"selected_variant": None, "promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "process-only labels and no typed vulnerability oracle"},
        "request_loop": {"model_action_manifest_generation": "diagnostic_only", "network_replay_in_this_run": False, "next_replay_requires_manifest_validator": True, "safe_canary_only": True},
        "safety": {"loopback_only": True, "external_network": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "script_execution": False, "database_write": False, "credential_access": False, "oracle_in_model_input": False, "family_in_model_input": False, "vulnerability_labels_in_model_input": False, "memory_promotion_allowed": False},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    trace_out = {"schema_version": "pg180-process-action-model-trace-v1", "evaluation_only": True, "training_eligible": False, "source_trace": str(TRACE_PATH.relative_to(ROOT)), "run_count": len(results), "runs": [{"split": item["split"], "variant": item["variant"], "seed": item["seed"], "parameter_count": item["parameter_count"], "test": item["test"], "raw_input_retained": item["raw_input_retained"], "oracle_in_input": item["oracle_in_input"], "family_in_input": item["family_in_input"], "route_in_input": item["route_in_input"], "vulnerability_label_in_input": item["vulnerability_label_in_input"]} for item in results], "online_weight_update": False, "long_term_memory_write": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}
    _write(TRACE_OUT_PATH, trace_out)
    protocol = {"protocol_id": "pg-pk-180-process-action-model-v1", "schema_version": "pg180-process-action-model-protocol-v1", "source_trace": str(TRACE_PATH.relative_to(ROOT)), "splits": list(SPLITS), "seeds": list(SEEDS), "variants": MODEL_VARIANTS, "epochs": EPOCHS, "patience": PATIENCE, "action_vocabulary": list(ALLOWED_ACTIONS), "required_gates": {"leave_surface_out": True, "family_and_route_excluded_from_input": True, "raw_probe_and_response_excluded": True, "typed_positive_excluded": True, "single_channel_probe_must_be_blocked": True, "safe_action_allowlist": True, "capacity_comparison_requires_multiple_seeds": True, "promotion_allowed": False, "memory_promotion_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    lines = ["# PG-180 process action token model", "", f"device={device}; rows={len(rows)}; vocabulary={len(vocabulary)}; runs={len(results)}", "", "| variant | parameters | mean test accuracy | mean abstain recall |", "|---|---:|---:|---:|"]
    for variant in capacity_order:
        item = by_variant[variant]
        lines.append(f"| {variant} | {item['parameter_count']} | {item['mean_test_safe_accuracy']} | {item['mean_test_abstain_recall']} |")
    lines.extend(["", "模型只预测 allow-listed 抽象动作；不含路径、漏洞族、原始 probe/response 或 oracle 权威。网络复放不在本轮执行，所有晋升门保持 false。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "rows": len(rows), "runs": len(results), "capacity": by_variant, "selection": report["selection"], "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_OUT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
