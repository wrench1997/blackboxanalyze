"""PG-189: train a large manifest decoder with structured real GET traces.

The training rows are derived from completed local replays but contain only
abstract action/response/failure tokens and safety-policy labels.  Raw marker
values, response bodies, routes, families, and vulnerability authority are
excluded from model input.  Two Pikachu GET routes remain a route holdout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from app.pg181_manifest_decoder import (  # noqa: E402
    MANIFEST_ACTION_TOKENS,
    build_manifest_examples,
    pre_action_tokens,
)


RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg189_structured_get_trace_action_training_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg189_structured_get_trace_action_training_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg189_structured_get_trace_action_training_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg189_structured_get_trace_action_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg189-structured-get-trace-action-v1"
PG179B_TRACE = RESEARCH / "pg179b_pikachu_iterative_trace_v1.json"
PG185_REPORT = RESEARCH / "pg185_pikachu_dom_replay_report_v1.json"
PG186_REPORT = RESEARCH / "pg186_pikachu_dom_capacity_encoding_report_v1.json"
PG187_REPORT = RESEARCH / "pg187_pikachu_cross_route_holdout_report_v1.json"
PG147_DATASET = RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json"
BODY_CHECKPOINTS = {
    "large": ROOT / "artifacts" / "pg163-large-typed-mix-v1" / "large_typed_mix.pt",
    "xxl": ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt",
}
SEED = 18901
MAX_LEN = 128
BATCH_SIZE = 8
ACTION_NAMES = ("baseline", "matched_control", "safe_candidate", "abstain")
ACTION_TOKENS = tuple(f"manifest::{name}" for name in ACTION_NAMES)
ROUTE_HOLDOUT = frozenset({"xss_01_unseen_get", "xss_04_unseen_get"})


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_pg180_helper() -> Any:
    path = ROOT / "scripts" / "run_pg180_process_action_model.py"
    spec = importlib.util.spec_from_file_location("pg180_for_pg189", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-180 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG180 = _load_pg180_helper()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_body_vocab() -> list[str]:
    checkpoint = torch.load(BODY_CHECKPOINTS["xxl"], map_location="cpu", weights_only=False)
    return [str(item) for item in checkpoint["vocabulary"]]


def _structured_rows(report_path: Path, *, source: str, exclude_route_holdout: bool = True) -> list[dict[str, Any]]:
    report = _load_json(report_path)
    rows: list[dict[str, Any]] = []
    for run in report.get("runs", []):
        surface = str(run.get("surface", "unknown"))
        if exclude_route_holdout and surface in ROUTE_HOLDOUT:
            continue
        history: list[dict[str, Any]] = []
        for step in run.get("steps", []):
            previous = history[-1] if history else None
            context = pre_action_tokens(previous, history=history[:-1])
            step_index = int(step.get("step_index", len(history) + 1))
            controller = str(step.get("controller_decision", ""))
            oracle = dict(step.get("oracle_projection") or {})
            typed_effect = bool(step.get("typed_surface_effect", False))
            candidate_signal = bool(oracle.get("candidate_signal", False))
            if step_index == 1:
                target = "manifest::baseline"
            elif typed_effect or candidate_signal:
                # No vulnerability oracle is available: a surface signal is a
                # safety-policy abstain target, not a candidate label.
                target = "manifest::abstain"
            elif controller == "send_safe_canary":
                target = "manifest::matched_control"
            else:
                target = "manifest::abstain"
            rows.append({
                "row_id": f"{source}:{surface}:{step_index}:{len(rows)}",
                "source": source,
                "surface": surface,
                "context": context,
                "target": target,
                "typed_surface_effect": typed_effect,
                "candidate_signal": candidate_signal,
                "policy_label": True,
            })
            history.append(dict(step))
    return rows


def _load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trace = _load_json(PG179B_TRACE)
    canonical = build_manifest_examples(trace)
    canonical_rows = [
        {
            "row_id": f"pg179b:{row['surface']}:{row['step_index']}",
            "source": "pg179b_canonical",
            "surface": str(row["surface"]),
            "context": list(row["context"]),
            "target": str(row["target"]),
            "typed_surface_effect": False,
            "candidate_signal": False,
            "policy_label": False,
        }
        for row in canonical
    ]
    real_rows = _structured_rows(PG185_REPORT, source="pg185_structured") + _structured_rows(PG186_REPORT, source="pg186_structured")
    # PG-187 is the intentionally unseen-route evaluation source.  Do not
    # apply the training-side route filter here, otherwise the holdout would
    # silently become empty and the gate would be meaningless.
    holdout_rows = _structured_rows(PG187_REPORT, source="pg187_holdout", exclude_route_holdout=False)
    # Keep only a route-disjoint training set. PG187 is never used for fitting.
    train = canonical_rows + real_rows
    dev = [row for row in real_rows if row["surface"] == "xss_reflected_get"]
    holdout = holdout_rows
    return train, dev, holdout, {"canonical_count": len(canonical_rows), "real_structured_count": len(real_rows), "holdout_count": len(holdout_rows), "holdout_surfaces": sorted({row["surface"] for row in holdout_rows})}


def _vocabulary(train_rows: list[dict[str, Any]], body_vocab: list[str]) -> dict[str, int]:
    tokens = set(body_vocab)
    tokens.update({"[PAD]", "[UNK]", "[BOS]", "[EOS]"})
    tokens.update(ACTION_TOKENS)
    for row in train_rows:
        tokens.update(str(item) for item in row["context"])
        tokens.add(str(row["target"]))
    ordered = ["[PAD]", "[UNK]"] + sorted(tokens - {"[PAD]", "[UNK]"})
    return {token: index for index, token in enumerate(ordered)}


def _encode_token(token: str, vocabulary: Mapping[str, int]) -> int:
    if token in vocabulary:
        return int(vocabulary[token])
    if token.startswith("history::encoding::") and "history::encoding::identity" in vocabulary:
        return int(vocabulary["history::encoding::identity"])
    return int(vocabulary.get("[UNK]", 1))


def _batch(rows: list[dict[str, Any]], vocabulary: Mapping[str, int], *, shuffle: bool, seed: int, batch_size: int = BATCH_SIZE) -> list[dict[str, Any]]:
    ordered = list(rows)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    batches: list[dict[str, Any]] = []
    for start in range(0, len(ordered), batch_size):
        subset = ordered[start:start + batch_size]
        encoded = [[_encode_token(str(token), vocabulary) for token in row["context"][:MAX_LEN]] for row in subset]
        width = max(len(item) for item in encoded)
        ids = torch.zeros((len(encoded), width), dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for index, values in enumerate(encoded):
            ids[index, :len(values)] = torch.tensor(values, dtype=torch.long)
            mask[index, :len(values)] = True
        batches.append({"ids": ids, "mask": mask, "rows": subset, "labels": torch.tensor([ACTION_NAMES.index(str(row["target"]).split("::", 1)[1]) for row in subset], dtype=torch.long)})
    return batches


def _lm_batches(rows: list[dict[str, Any]], vocabulary: Mapping[str, int], *, seed: int, limit: int = 2048) -> list[dict[str, Any]]:
    selected = [row for row in rows[:limit] if isinstance(row.get("tokens"), list)]
    projected = [{"context": list(row["tokens"]), "target": "manifest::abstain"} for row in selected]
    return _batch(projected, vocabulary, shuffle=True, seed=seed, batch_size=4)


def _init_body(body_name: str, vocabulary: Mapping[str, int], device: torch.device) -> tuple[CausalTraceTransformer, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(BODY_CHECKPOINTS[body_name], map_location="cpu", weights_only=False)
    old_vocab = [str(item) for item in checkpoint["vocabulary"]]
    config = dict(checkpoint["config"])
    body = CausalTraceTransformer(len(vocabulary), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    new_state = body.state_dict()
    old_state = checkpoint["model_state_dict"]
    old_index = {token: index for index, token in enumerate(old_vocab)}
    for key in new_state:
        if key not in old_state:
            continue
        if key in {"token_embedding.weight", "lm_head.weight"}:
            with torch.no_grad():
                for index, token in enumerate(vocabulary):
                    if token in old_index:
                        new_state[key][index].copy_(old_state[key][old_index[token]])
        elif key == "lm_head.bias":
            with torch.no_grad():
                for index, token in enumerate(vocabulary):
                    if token in old_index:
                        new_state[key][index].copy_(old_state[key][old_index[token]])
        elif tuple(new_state[key].shape) == tuple(old_state[key].shape):
            new_state[key].copy_(old_state[key])
    body.load_state_dict(new_state)
    return body, {"d_model": int(config["d_model"]), "nhead": int(config["nhead"]), "layers": int(config["layers"]), "max_len": MAX_LEN}, {"checkpoint_sha256": hashlib.sha256(BODY_CHECKPOINTS[body_name].read_bytes()).hexdigest(), "old_vocab_size": len(old_vocab)}


class _ManifestModel(nn.Module):
    def __init__(self, body: CausalTraceTransformer, d_model: int) -> None:
        super().__init__()
        self.body = body
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(ACTION_NAMES)))

    def logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.body.encode(ids, mask)
        lengths = mask.long().sum(dim=1).clamp_min(1)
        last = hidden[torch.arange(ids.shape[0], device=ids.device), lengths - 1]
        return self.head(last)


def _metrics(model: _ManifestModel, rows: list[dict[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    model.eval()
    count = correct = abstain_expected = abstain_predicted = abstain_true_positive = false_candidate_on_abstain = 0
    with torch.inference_mode():
        for batch in _batch(rows, vocabulary, shuffle=False, seed=0):
            predictions = model.logits(batch["ids"].to(device), batch["mask"].to(device)).argmax(-1).detach().cpu().tolist()
            labels = batch["labels"].tolist()
            for predicted, expected in zip(predictions, labels):
                count += 1
                correct += int(predicted == expected)
                abstain_expected += int(expected == ACTION_NAMES.index("abstain"))
                abstain_predicted += int(predicted == ACTION_NAMES.index("abstain"))
                abstain_true_positive += int(expected == ACTION_NAMES.index("abstain") and predicted == ACTION_NAMES.index("abstain"))
                false_candidate_on_abstain += int(expected == ACTION_NAMES.index("abstain") and predicted == ACTION_NAMES.index("safe_candidate"))
    return {"count": count, "accuracy": round(correct / max(count, 1), 8), "expected_abstain_count": abstain_expected, "predicted_abstain_count": abstain_predicted, "abstain_true_positive_count": abstain_true_positive, "abstain_recall": round(abstain_true_positive / max(abstain_expected, 1), 8), "false_candidate_on_abstain_count": false_candidate_on_abstain}


def _lm_metrics(body: CausalTraceTransformer, lm_batches: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    body.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in lm_batches:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            if ids.shape[1] < 2:
                continue
            logits = body(ids[:, :-1], mask[:, :-1])
            targets = ids[:, 1:]
            total_loss += float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum", ignore_index=0).detach().cpu())
            total_tokens += int(mask[:, 1:].sum().item())
    loss = total_loss / max(total_tokens, 1)
    return {"loss": round(loss, 8), "perplexity": round(float(torch.exp(torch.tensor(loss))), 8), "token_count": total_tokens}


def _train_variant(body_name: str, train: list[dict[str, Any]], dev: list[dict[str, Any]], holdout: list[dict[str, Any]], vocabulary: dict[str, int], lm_rows: list[dict[str, Any]], device: torch.device, seed_offset: int) -> dict[str, Any]:
    torch.manual_seed(SEED + seed_offset)
    random.seed(SEED + seed_offset)
    body, config, init_meta = _init_body(body_name, vocabulary, device)
    model = _ManifestModel(body, config["d_model"]).to(device)
    lm_batches = _lm_batches(lm_rows, vocabulary, seed=SEED + seed_offset)
    before_lm = _lm_metrics(model.body, lm_batches, device)
    optimizer = torch.optim.AdamW([{"params": model.body.parameters(), "lr": 8e-6}, {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=0.01)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, 4):
        model.train()
        losses: list[float] = []
        replay_index = 0
        for batch in _batch(train, vocabulary, shuffle=True, seed=SEED + seed_offset + epoch):
            action_loss = nn.functional.cross_entropy(model.logits(batch["ids"].to(device), batch["mask"].to(device)), batch["labels"].to(device))
            replay = lm_batches[replay_index % len(lm_batches)]
            replay_index += 1
            replay_ids = replay["ids"].to(device)
            replay_mask = replay["mask"].to(device)
            if replay_ids.shape[1] > 1:
                lm_logits = model.body(replay_ids[:, :-1], replay_mask[:, :-1])
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), replay_ids[:, 1:].reshape(-1), ignore_index=0)
            else:
                lm_loss = torch.zeros((), device=device)
            loss = action_loss + 0.10 * lm_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_loss": round(statistics.mean(losses), 8), "dev": _metrics(model, dev, vocabulary, device)})
    after_lm = _lm_metrics(model.body, lm_batches, device)
    relative = (after_lm["perplexity"] - before_lm["perplexity"]) / max(before_lm["perplexity"], 1e-9)
    result = {"body": body_name, "parameter_count": int(sum(p.numel() for p in model.parameters())), "train_count": len(train), "dev_count": len(dev), "holdout_count": len(holdout), "history": history, "train": _metrics(model, train, vocabulary, device), "dev": _metrics(model, dev, vocabulary, device), "holdout": _metrics(model, holdout, vocabulary, device), "language_replay": {"before": before_lm, "after": after_lm, "relative_perplexity_increase": round(relative, 8), "catastrophic_forgetting_detected": bool(relative > 0.20)}, "init": init_meta, "elapsed_seconds": round(time.perf_counter() - started, 3), "raw_payloads_in_model": False, "raw_responses_in_model": False, "target_route_in_input": False, "vulnerability_label_in_input": False}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg189-structured-get-trace-action-v1", "body": body_name, "vocabulary": vocabulary, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / f"{body_name}.pt")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    train, dev, holdout, row_stats = _load_rows()
    body_vocab = _load_body_vocab()
    vocabulary = _vocabulary(train, body_vocab)
    lm_dataset = _load_json(PG147_DATASET)
    lm_rows = [row for row in lm_dataset["splits"]["train"][:2048] if isinstance(row.get("tokens"), list)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [_train_variant(name, train, dev, holdout, vocabulary, lm_rows, device, offset) for offset, name in enumerate(("large", "xxl"), start=1)]
    eligible = [row for row in results if not row["language_replay"]["catastrophic_forgetting_detected"] and row["holdout"]["abstain_recall"] >= 0.95 and row["holdout"]["false_candidate_on_abstain_count"] == 0]
    selected = max(eligible, key=lambda row: float(row["holdout"]["accuracy"]))["body"] if eligible else None
    report = {"protocol_id": "pg-pk-189-structured-get-trace-action-training-v1", "schema_version": "pg189-structured-get-trace-action-training-report-v1", "status": "completed_structured_real_get_trace_action_training", "device": str(device), "row_stats": row_stats, "training_rows": len(train), "dev_rows": len(dev), "holdout_rows": len(holdout), "vocabulary_size": len(vocabulary), "lm_replay_rows": len(lm_rows), "route_holdout": sorted(ROUTE_HOLDOUT), "results": results, "selection": {"selected_variant": selected, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "gate": "holdout abstain recall >= .95, false_candidate_on_abstain=0, forgetting=false"}, "safety": {"loopback_only": True, "external_network": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "target_route_in_input": False, "vulnerability_label_in_input": False, "script_execution": False, "database_write": False, "target_trace_raw_persistence": False}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg189-structured-get-trace-action-training-trace-v1", "structured_projection_only": True, "route_holdout": sorted(ROUTE_HOLDOUT), "training_rows": len(train), "holdout_rows": len(holdout), "results": [{"body": row["body"], "holdout": row["holdout"], "language_replay": row["language_replay"], "raw_payloads_in_model": False, "raw_responses_in_model": False} for row in results], "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False})
    protocol = {"protocol_id": "pg-pk-189-structured-get-trace-action-training-v1", "schema_version": "pg189-structured-get-trace-action-training-protocol-v1", "structured_sources": [str(PG185_REPORT.relative_to(ROOT)), str(PG186_REPORT.relative_to(ROOT))], "holdout_source": str(PG187_REPORT.relative_to(ROOT)), "route_holdout": sorted(ROUTE_HOLDOUT), "model_variants": ["large", "xxl"], "model_output_vocabulary": list(ACTION_NAMES), "raw_payload_and_response_excluded": True, "target_route_excluded_from_input": True, "vulnerability_labels_excluded": True, "lm_replay_rows": len(lm_rows), "gates": {"holdout_abstain_recall_min": 0.95, "false_candidate_on_abstain_max": 0, "catastrophic_forgetting_blocks_selection": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-189 structured GET trace action training", "", f"device={device}; train={len(train)}; dev={len(dev)}; holdout={len(holdout)}; vocab={len(vocabulary)}; selected={selected}", "", "| body | parameters | holdout accuracy | abstain recall | false candidate | forgetting |", "|---|---:|---:|---:|---:|---|"] + [f"| {r['body']} | {r['parameter_count']} | {r['holdout']['accuracy']} | {r['holdout']['abstain_recall']} | {r['holdout']['false_candidate_on_abstain_count']} | {r['language_replay']['catastrophic_forgetting_detected']} |" for r in results] + ["", "结构化 real GET trace 只作为 safety-policy action 数据；不包含原始 payload、响应正文或漏洞阳性标签。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "train_rows": len(train), "holdout_rows": len(holdout), "vocabulary_size": len(vocabulary), "results": [{"body": r["body"], "parameter_count": r["parameter_count"], "holdout_accuracy": r["holdout"]["accuracy"], "abstain_recall": r["holdout"]["abstain_recall"], "false_candidate_on_abstain": r["holdout"]["false_candidate_on_abstain_count"], "forgetting": r["language_replay"]["catastrophic_forgetting_detected"]} for r in results], "selected_variant": selected, "training_artifact_promotion_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
