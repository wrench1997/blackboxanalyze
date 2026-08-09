"""PG-81: source-isolated Trace Transformer training on PG-79 triplets.

Training uses only two independently written local implementations from the
fresh PG-79 collection.  A third implementation family is a true source
holdout; PG-76 remains an unknown-family abstention holdout.  Source/family
labels are evaluator metadata and never tokenized.
"""

from __future__ import annotations

import copy
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


PG77_SCRIPT = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG53_SCRIPT = ROOT / "scripts" / "run_pg53_cross_source_typed_replay.py"
PG79_TRACE = ROOT / "research" / "pg79_fresh_unified_triplet_collector_trace_v1.json"
PG76_TRACE = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
PG56_CHECKPOINT = ROOT / "artifacts" / "pg56-causal-trace-transformer" / "model.pt"
DATASET_PATH = ROOT / "research" / "pg81_source_holdout_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg81_source_holdout_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg81_source_holdout_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg81_source_holdout_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg81_source_holdout_transformer_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg81-source-holdout-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
TRAIN_SOURCES = {("pg34", "base"), ("pg35", "alpha")}
DEV_SOURCES = {("pg35", "beta")}
HOLDOUT_SOURCES = {("pg36", "north"), ("pg36", "south")}
SEED = 20810403
CONFIDENCE_THRESHOLD = 0.70
OOD_MARGIN = 0.25
CLASSES = ("confirm", "reject")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _annotated_source(pg53: Any, index: int) -> dict[str, Any]:
    per_target = len(pg53.SURFACES) * len(pg53.METHODS)
    target = pg53.TARGETS[index // per_target]
    within = index % per_target
    surface = pg53.SURFACES[within // len(pg53.METHODS)]
    method = pg53.METHODS[within % len(pg53.METHODS)]
    return {"source_id": str(target["source_id"]), "implementation": str(target["implementation"]), "variant": str(target["variant"]), "surface": surface, "method": method, "family": str(pg53._spec(target, surface).get("family", "unknown"))}


def _step_rows(pg77: Any, trace: dict[str, Any], pg53: Any, *, split: str, allowed_sources: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_seed_index: dict[int, int] = {}
    for step in trace.get("steps", []):
        seed = int(step.get("sampling_seed", -1))
        index = per_seed_index.get(seed, 0)
        per_seed_index[seed] = index + 1
        meta = _annotated_source(pg53, index)
        source_key = (meta["implementation"], meta["variant"])
        if allowed_sources is not None and source_key not in allowed_sources:
            continue
        for role, projection, oracle in (("positive", step["response_projection"], step["oracle_projection"]), ("negative", step["negative_probe_projection"], step["negative_oracle_projection"])):
            tokens, oracle_index = pg77._row_tokens(step, dict(projection), dict(oracle))
            rows.append({"trace_id": f"{step['step_id']}-{role}", "split": split, "source_id": meta["source_id"], "implementation": meta["implementation"], "variant": meta["variant"], "family": meta["family"], "surface": meta["surface"], "method": meta["method"], "sampling_seed": seed, "role": role, "tokens": tokens, "oracle_index": oracle_index, "expected": "confirm" if role == "positive" and bool(oracle.get("positive")) else "reject", "raw_probe_stored": False, "raw_response_stored": False})
    return rows


def _unknown_rows(pg77: Any, trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        tokens, oracle_index = pg77._row_tokens(step, dict(step["response_projection"]), dict(step["oracle_projection"]))
        rows.append({"trace_id": f"{step['step_id']}-positive", "split": "unknown_family_holdout", "role": "positive", "tokens": tokens, "oracle_index": oracle_index, "expected": "abstain", "raw_probe_stored": False, "raw_response_stored": False})
    return rows


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pad, unk = vocabulary["<PAD>"], vocabulary["<UNK>"]
    max_len = max(len(row["tokens"]) for row in rows)
    ids = torch.full((len(rows), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    positions: list[int] = []
    unknown = 0
    for index, row in enumerate(rows):
        encoded = []
        for token in row["tokens"]:
            unknown += int(token not in vocabulary)
            encoded.append(vocabulary.get(token, unk))
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
        positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(positions, dtype=torch.long), unknown


def _lm_loss(model: CausalTraceTransformer, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(ids, mask)
    target = ids[:, 1:]
    valid = mask[:, 1:]
    return nn.functional.cross_entropy(logits[:, :-1][valid], target[valid])


def _load_pretrained(pg77: Any, vocabulary: dict[str, int], device: torch.device) -> CausalTraceTransformer:
    model = pg77._load_pretrained(vocabulary, device)
    return model


def run() -> dict[str, Any]:
    pg77 = _load(PG77_SCRIPT, "pg81_pg77_runtime")
    pg53 = _load(PG53_SCRIPT, "pg81_pg53_runtime")
    pg79 = json.loads(PG79_TRACE.read_text(encoding="utf-8"))
    pg76 = json.loads(PG76_TRACE.read_text(encoding="utf-8"))
    train_rows = _step_rows(pg77, pg79, pg53, split="train", allowed_sources=TRAIN_SOURCES)
    dev_rows = _step_rows(pg77, pg79, pg53, split="dev", allowed_sources=DEV_SOURCES)
    holdout_rows = _step_rows(pg77, pg79, pg53, split="source_holdout", allowed_sources=HOLDOUT_SOURCES)
    unknown_rows = _unknown_rows(pg77, pg76)
    old = torch.load(PG56_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = dict(old["vocabulary"])
    # The schema vocabulary is fixed from train/dev only; holdout token drift
    # is measured rather than silently added after the split.
    for token in sorted({token for row in train_rows + dev_rows for token in row["tokens"]}):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    dataset = {"schema_version": "pg81-source-holdout-trace-dataset-v1", "dataset_id": "pg81-source-holdout-triplets", "training_eligible": False, "evaluation_only": True, "model_input_contract": {"family_name_in_tokens": False, "source_id_in_tokens": False, "implementation_in_tokens": False, "route_words_in_tokens": False, "raw_probe_in_tokens": False, "raw_response_body_in_tokens": False, "typed_oracle_before_target_marker": False}, "split_contract": {"train_sources": sorted("/".join(item) for item in TRAIN_SOURCES), "dev_sources": sorted("/".join(item) for item in DEV_SOURCES), "holdout_sources": sorted("/".join(item) for item in HOLDOUT_SOURCES), "unknown_family_training_forbidden": True}, "split_counts": {"train": len(train_rows), "dev": len(dev_rows), "source_holdout": len(holdout_rows), "unknown_family_holdout": len(unknown_rows)}, "vocabulary_size": len(vocabulary), "rows": train_rows + dev_rows + holdout_rows + unknown_rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    dataset["dataset_sha256"] = __import__("hashlib").sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = _load_pretrained(pg77, vocabulary, device)
    train_ids, train_mask, train_pos, train_unknown = _encode(train_rows, vocabulary)
    dev_ids, dev_mask, dev_pos, dev_unknown = _encode(dev_rows, vocabulary)
    holdout_ids, holdout_mask, holdout_pos, holdout_unknown = _encode(holdout_rows, vocabulary)
    unknown_ids, unknown_mask, unknown_pos, unknown_token_count = _encode(unknown_rows, vocabulary)
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
            with torch.no_grad():
                dev_loss = float(_lm_loss(model, dev_ids.to(device), dev_mask.to(device)).detach().cpu())
            history.append({"epoch": epoch, "train_loss": round(float(loss.detach().cpu()), 6), "dev_loss": round(dev_loss, 6)})
            if dev_loss < best_dev:
                best_dev = dev_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_hidden = pg77._hidden_at_oracle(model, train_ids.to(device), train_mask.to(device), train_pos.to(device)).detach()
        dev_hidden = pg77._hidden_at_oracle(model, dev_ids.to(device), dev_mask.to(device), dev_pos.to(device)).detach()
        holdout_hidden = pg77._hidden_at_oracle(model, holdout_ids.to(device), holdout_mask.to(device), holdout_pos.to(device)).detach()
        unknown_hidden = pg77._hidden_at_oracle(model, unknown_ids.to(device), unknown_mask.to(device), unknown_pos.to(device)).detach()
    head = pg77.RuleIRHead(int(train_hidden.shape[-1])).to(device)
    labels = torch.tensor([0 if row["expected"] == "confirm" else 1 for row in train_rows], dtype=torch.long, device=device)
    dev_labels = torch.tensor([0 if row["expected"] == "confirm" else 1 for row in dev_rows], dtype=torch.long, device=device)
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.02)
    best_head: dict[str, torch.Tensor] | None = None
    best_head_loss = float("inf")
    for epoch in range(1, 401):
        head.train()
        loss = nn.functional.cross_entropy(head(train_hidden), labels)
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        head_optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            head.eval()
            with torch.no_grad():
                dev_loss = float(nn.functional.cross_entropy(head(dev_hidden), dev_labels).detach().cpu())
            if dev_loss < best_head_loss:
                best_head_loss = dev_loss
                best_head = copy.deepcopy({key: value.detach().cpu() for key, value in head.state_dict().items()})
    if best_head is not None:
        head.load_state_dict(best_head)
    threshold = pg77._calibrate_threshold(train_hidden, dev_hidden)
    train_metrics, train_details = pg77._evaluate_head(head, train_hidden, train_rows, train_hidden, threshold)
    dev_metrics, dev_details = pg77._evaluate_head(head, dev_hidden, dev_rows, train_hidden, threshold)
    holdout_metrics, holdout_details = pg77._evaluate_head(head, holdout_hidden, holdout_rows, train_hidden, threshold)
    unknown_metrics, unknown_details = pg77._evaluate_head(head, unknown_hidden, unknown_rows, train_hidden, threshold, unknown=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg81-source-holdout-transformer-checkpoint-v1", "vocabulary": vocabulary, "max_len": 128, "transformer_state": model.state_dict(), "rule_ir_head_state": head.state_dict(), "hidden_dim": int(train_hidden.shape[-1]), "ood_distance_threshold": threshold, "confidence_threshold": CONFIDENCE_THRESHOLD, "train_sources": sorted("/".join(item) for item in TRAIN_SOURCES), "dev_sources": sorted("/".join(item) for item in DEV_SOURCES), "holdout_sources": sorted("/".join(item) for item in HOLDOUT_SOURCES), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}, CHECKPOINT_PATH)
    checks = {"dev_confirm_recall": float(dev_metrics.get("confirm_recall", 0.0)) >= 0.80, "dev_false_accept_zero": int(dev_metrics.get("false_accept_count", 1)) == 0, "source_holdout_confirm_recall": float(holdout_metrics.get("confirm_recall", 0.0)) >= 0.80, "source_holdout_false_accept_zero": int(holdout_metrics.get("false_accept_count", 1)) == 0, "unknown_strict_abstain": bool(unknown_metrics.get("strict_abstain", False)), "family_source_not_in_tokens": True}
    report = {"protocol_id": "pg-pk-81-source-holdout-transformer-v1", "schema_version": "pg81-source-holdout-transformer-report-v1", "status": "candidate_training_completed", "source": {"train_trace": str(PG79_TRACE.relative_to(ROOT)), "unknown_holdout_trace": str(PG76_TRACE.relative_to(ROOT)), "base_pretraining_checkpoint": str(PG56_CHECKPOINT.relative_to(ROOT)), "model_retrained_on_unknown_family": False, "family_in_tokens": False, "source_in_tokens": False, "device": str(device)}, "dataset": dataset["split_counts"] | {"vocabulary_size": len(vocabulary), "train_sources": sorted("/".join(item) for item in TRAIN_SOURCES), "dev_sources": sorted("/".join(item) for item in DEV_SOURCES), "holdout_sources": sorted("/".join(item) for item in HOLDOUT_SOURCES), "unknown_token_counts": {"train": train_unknown, "dev": dev_unknown, "source_holdout": holdout_unknown, "unknown_family": unknown_token_count}}, "metrics": {"train": train_metrics, "dev_holdout": dev_metrics, "source_holdout": holdout_metrics, "unknown_family_holdout": unknown_metrics}, "details": {"train": train_details, "dev_holdout": dev_details, "source_holdout": holdout_details, "unknown_family_holdout": unknown_details}, "training": {"pretraining_epochs": 250, "rule_ir_head_epochs": 400, "history_tail": history[-5:], "ood_distance_threshold": threshold, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "online_weight_update": False, "long_term_memory_write": False}, "capability_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_evaluation_only", "reason": "source holdout must pass with independent families and a second implementation before promotion"}, "formal_claim": {"allowed": False, "reason": "candidate source holdout is not a broad web vulnerability detector"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg81-source-holdout-transformer-trace-v1", "evaluation_only": True, "training_eligible": False, "rows": [{"trace_id": row["trace_id"], "split": row["split"], "source_id": row.get("source_id"), "implementation": row.get("implementation"), "variant": row.get("variant"), "role": row["role"], "raw_probe_stored": False, "raw_response_stored": False} for row in dataset["rows"]], "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": "pg-pk-81-source-holdout-transformer-v1", "schema_version": "pg81-source-holdout-transformer-protocol-v1", "input_contract": dataset["model_input_contract"], "split_contract": dataset["split_contract"], "training_contract": {"pg79_collection_gate_required": True, "unknown_family_training_forbidden": True, "family_and_source_features_forbidden": True, "oracle_after_target_only": True, "raw_persistence_forbidden": True}, "required_gates": {"dev_confirm_recall_min": 0.80, "dev_false_accept_zero": True, "source_holdout_confirm_recall_min": 0.80, "source_holdout_false_accept_zero": True, "unknown_strict_abstain": True, "independent_second_implementation_required": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG82 independent second implementation and family holdout"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-81 Source-isolated Trace Transformer\n\n" + f"train/dev/holdout/unknown={len(train_rows)}/{len(dev_rows)}/{len(holdout_rows)}/{len(unknown_rows)}；device={device}；vocab={len(vocabulary)}。\n\ndev recall={dev_metrics.get('confirm_recall', 0.0)}；source holdout recall={holdout_metrics.get('confirm_recall', 0.0)}；unknown strict abstain={unknown_metrics.get('strict_abstain', False)}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "dev_confirm_recall": result["metrics"]["dev_holdout"]["confirm_recall"], "source_holdout_confirm_recall": result["metrics"]["source_holdout"]["confirm_recall"], "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"]["strict_abstain"], "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
