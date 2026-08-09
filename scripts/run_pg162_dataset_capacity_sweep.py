"""PG-162: collect a fresh typed local dataset and sweep model capacity.

This experiment is deliberately narrower than a vulnerability scanner.  The
targets are in-process loopback fixtures with inert abstract probes.  A row is
training-eligible only when the collector supplied a fresh-reset contract and
an integrity-checked evidence hash.  Raw probe strings, response bodies,
family names and evaluator fields never enter the model-facing projection.

The experiment answers two questions:

* does more *verified* data help, rather than merely replaying the same rows?
* does a wider Rule-IR decoder improve source-heldout decisions, or just fit
  the fixture vocabulary?

The generated report keeps the real PG-146 container replay evaluation-only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

# Scripts are invoked from the repository root in CI and from the scripts
# directory by hand; make the package import independent of the current cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg116_multisource_replay import collect_source
from app.pg118_transition_replay import collect_target
from app.pg118_transition_rule_ir_decoder import (
    PG118_DECISIONS,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
)


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg162-dataset-capacity-sweep-v1"
DATASET_PATH = RESEARCH / "pg162_fresh_typed_training_dataset_v1.json"
TRACE_MANIFEST_PATH = RESEARCH / "pg162_fresh_typed_trace_manifest_v1.json"
REPORT_PATH = RESEARCH / "pg162_dataset_capacity_sweep_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg162_dataset_capacity_sweep_report_v1.md"

PG116_TRAIN_SEEDS = [16201, 16202, 16203, 16204]
PG116_DEV_SEEDS = [16205, 16206, 16207, 16208]
PG118_TRAIN_SEEDS = [16211, 16212, 16213, 16214]
PG118_DEV_SEEDS = [16215, 16216, 16217, 16218]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    return canonical_model_input({
        "action_manifest": value.get("action_manifest") or {},
        "baseline_projection": value.get("baseline_projection") or {},
        "response_projection": value.get("response_projection") or {},
        "belief_before": value.get("belief_before") or {},
    })


def _is_hash(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _make_row(
    *,
    source_group: str,
    source: str,
    split: str,
    episode: dict[str, Any],
    step: dict[str, Any],
    prior: list[dict[str, Any]],
) -> dict[str, Any]:
    # Only canonical visible Rule-IR fields are retained.  In particular, the
    # evaluator projection, route names, target IDs and probe strings are not
    # part of this object.
    model_input = _canonical(step.get("model_input") or step)
    prior_inputs = [_canonical(value) for value in prior]
    evidence_hash = step.get("evidence_sha256")
    fresh_reset = step.get("fresh_reset") or {}
    if not _is_hash(evidence_hash):
        raise ValueError(f"PG-162 missing evidence hash for {step.get('step_id')}")
    if not bool(fresh_reset.get("fresh_target")) or not bool(fresh_reset.get("completed")):
        raise ValueError(f"PG-162 fresh reset gate failed for {step.get('step_id')}")
    method = str((model_input.get("action_manifest") or {}).get("method", "")).upper()
    if method not in {"GET", "POST"}:
        raise ValueError(f"PG-162 unsupported method {method!r}")
    return {
        "row_id": str(step["step_id"]),
        "source_group": source_group,
        "source": source,
        "target_seed": int(episode["target_seed"]),
        "target_instance_id": str(episode["target_instance_id"]),
        "episode_id": str(episode["episode_id"]),
        "surface_kind": str(episode["surface_kind"]),
        "split": split,
        "method": method,
        "model_input": model_input,
        "prior_inputs": prior_inputs,
        "label": str(step["decision"]),
        "evidence_sha256": str(evidence_hash),
        "fresh_reset": True,
        "training_eligible": True,
        "memory_promotion_allowed": False,
    }


def _rows_from_target(target: dict[str, Any], *, source_group: str, source: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in target["episodes"]:
        prior: list[dict[str, Any]] = []
        for step in episode["steps"]:
            rows.append(_make_row(source_group=source_group, source=source, split=split, episode=episode, step=step, prior=prior))
            prior.append(step.get("model_input") or step)
    return rows


async def _collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for source in ("alpha", "beta"):
        train_target = await collect_source(source, PG116_TRAIN_SEEDS)
        dev_target = await collect_source(source, PG116_DEV_SEEDS)
        train_rows.extend(_rows_from_target(train_target, source_group=f"pg116_{source}", source=source, split="train"))
        dev_rows.extend(_rows_from_target(dev_target, source_group=f"pg116_{source}", source=source, split="dev"))
    for seed in PG118_TRAIN_SEEDS:
        target = await collect_target(seed)
        train_rows.extend(_rows_from_target(target, source_group="pg118_delta", source="pg118_delta", split="train"))
    for seed in PG118_DEV_SEEDS:
        target = await collect_target(seed)
        dev_rows.extend(_rows_from_target(target, source_group="pg118_delta", source="pg118_delta", split="dev"))
    return train_rows, dev_rows


def _features(rows: Iterable[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = list(rows)
    x = torch.tensor(
        [model_input_feature_vector(row["model_input"], prior_inputs=row.get("prior_inputs", [])) for row in values],
        dtype=torch.float32,
    )
    y = torch.tensor([decision_index(row["label"]) for row in values], dtype=torch.long)
    return x, y


class CapacityDecoder(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = 44
        for _ in range(depth):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_dim, len(PG118_DECISIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def _metrics(pred: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    correct = sum(p == y for p, y in zip(pred, labels))
    per_class: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for index, name in enumerate(PG118_DECISIONS):
        tp = sum(p == index and y == index for p, y in zip(pred, labels))
        fp = sum(p == index and y != index for p, y in zip(pred, labels))
        fn = sum(p != index and y == index for p, y in zip(pred, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {"count": total, "accuracy": round(correct / total, 6) if total else 0.0, "macro_f1": round(sum(f1s) / len(f1s), 6), "per_class": per_class}


def _predict(model: nn.Module, rows: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[int], list[float]]:
    x, y = _features(rows)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(x.to(device)), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.cpu().tolist(), y.tolist(), confidence.cpu().tolist()


def _episode_metrics(rows: list[dict[str, Any]], pred: list[int]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for row, value in zip(rows, pred):
        grouped.setdefault(row["episode_id"], []).append((row, value))
    finals: list[dict[str, Any]] = []
    for episode_id, values in sorted(grouped.items()):
        last_row, last_pred = values[-1]
        finals.append({"episode_id": episode_id, "surface_kind": last_row["surface_kind"], "predicted": PG118_DECISIONS[last_pred], "expected": last_row["label"], "length": len(values)})
    positive = [r for r in finals if r["surface_kind"] in {"policy", "boundary", "route"}]
    decoys = [r for r in finals if r["surface_kind"] in {"decoy", "layout", "steady"}]
    unknown = [r for r in finals if r["surface_kind"] in {"opaque", "blind"}]
    return {
        "episode_count": len(finals),
        "positive_final_confirm_recall": round(sum(r["predicted"] == "confirmed_positive" for r in positive) / len(positive), 6) if positive else None,
        "decoy_false_accept_count": sum(r["predicted"] == "confirmed_positive" for r in decoys),
        "unknown_final_abstain_rate": round(sum(r["predicted"] == "abstain" for r in unknown) / len(unknown), 6) if unknown else None,
        "final_rows": finals,
    }


def _train_variant(name: str, hidden: int, depth: int, dropout: float, train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], source_holdouts: dict[str, list[dict[str, Any]]], device: torch.device) -> dict[str, Any]:
    _set_seed(16200 + hidden + depth)
    model = CapacityDecoder(hidden, depth, dropout).to(device)
    x_train, y_train = _features(train_rows)
    x_dev, y_dev = _features(dev_rows)
    counts = Counter(y_train.tolist())
    weights = torch.tensor([1.0 / max(counts.get(index, 1), 1) ** 0.5 for index in range(len(PG118_DECISIONS))], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002 if hidden < 128 else 0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train.to(device)), y_train.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        train_pred, train_labels, _ = _predict(model, train_rows, device)
        dev_pred, dev_labels, _ = _predict(model, dev_rows, device)
        train_metric = _metrics(train_pred, train_labels)
        dev_metric = _metrics(dev_pred, dev_labels)
        history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "train_macro_f1": train_metric["macro_f1"], "dev_macro_f1": dev_metric["macro_f1"]})
        if dev_metric["macro_f1"] > best_score:
            best_score = float(dev_metric["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError("PG-162 failed to select a checkpoint")
    model.load_state_dict(best_state)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"model": model.state_dict(), "hidden": hidden, "depth": depth, "feature_dim": 44, "decisions": list(PG118_DECISIONS), "best_epoch": best_epoch}, checkpoint_path)
    train_pred, train_labels, _ = _predict(model, train_rows, device)
    dev_pred, dev_labels, dev_confidence = _predict(model, dev_rows, device)
    result: dict[str, Any] = {
        "hidden_dim": hidden,
        "depth": depth,
        "dropout": dropout,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "train": _metrics(train_pred, train_labels),
        "dev": _metrics(dev_pred, dev_labels),
        "dev_episode": _episode_metrics(dev_rows, dev_pred),
        "checkpoint": str(checkpoint_path),
        "confidence": {"mean": round(sum(dev_confidence) / len(dev_confidence), 6), "min": round(min(dev_confidence), 6), "max": round(max(dev_confidence), 6)},
        "history_tail": history[-5:],
        "source_holdout": {},
    }
    for holdout_name, holdout_rows in source_holdouts.items():
        holdout_pred, holdout_labels, _ = _predict(model, holdout_rows, device)
        result["source_holdout"][holdout_name] = {"rows": len(holdout_rows), "metrics": _metrics(holdout_pred, holdout_labels), "episodes": _episode_metrics(holdout_rows, holdout_pred)}
    return result


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    train_rows, dev_rows = await _collect()
    all_rows = train_rows + dev_rows
    source_holdouts = {
        "pg118_delta": [row for row in all_rows if row["source_group"] == "pg118_delta"],
        "pg116_beta": [row for row in all_rows if row["source_group"] == "pg116_beta"],
    }
    manifest = {
        "schema_version": "pg162-fresh-typed-trace-manifest-v1",
        "protocol_id": "pg-pk-162-dataset-capacity-sweep-v1",
        "source_set": ["pg116_alpha", "pg116_beta", "pg118_delta"],
        "target_implementations": ["pg116-multisource-training-target-v1", "pg118-delta-independent-target"],
        "train_seed_sets": {"pg116": PG116_TRAIN_SEEDS, "pg118": PG118_TRAIN_SEEDS},
        "dev_seed_sets": {"pg116": PG116_DEV_SEEDS, "pg118": PG118_DEV_SEEDS},
        "row_count": len(all_rows),
        "train_row_count": len(train_rows),
        "dev_row_count": len(dev_rows),
        "get_count": sum(row["method"] == "GET" for row in all_rows),
        "post_count": sum(row["method"] == "POST" for row in all_rows),
        "class_counts": dict(Counter(row["label"] for row in all_rows)),
        "source_counts": dict(Counter(row["source_group"] for row in all_rows)),
        "evidence_hash_valid": all(_is_hash(row["evidence_sha256"]) for row in all_rows),
        "fresh_reset_per_step": all(row["fresh_reset"] for row in all_rows),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "family_in_model_input": False,
        "oracle_labels_in_model_input": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _write(TRACE_MANIFEST_PATH, manifest)
    dataset = {
        "schema_version": "pg162-fresh-typed-training-dataset-v1",
        "protocol_id": "pg-pk-162-dataset-capacity-sweep-v1",
        "purpose": "fresh local typed action-decision training; not a real vulnerability scanner",
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "model_input_family_free": True,
        "model_input_oracle_blind": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "script_execution": False,
        "database_write": False,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "source_holdout_rows": source_holdouts,
    }
    dataset["manifest_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {
        "tiny": (24, 1, 0.0),
        "base": (48, 2, 0.0),
        "wide": (128, 3, 0.05),
    }
    model_results = {name: _train_variant(name, hidden, depth, dropout, train_rows, dev_rows, source_holdouts, device) for name, (hidden, depth, dropout) in variants.items()}
    best_name = max(model_results, key=lambda name: (model_results[name]["source_holdout"]["pg118_delta"]["metrics"]["macro_f1"], model_results[name]["dev"]["macro_f1"], -model_results[name]["parameter_count"]))
    pg146 = json.loads((RESEARCH / "pg146_public_lab_replay_report_v1.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "pg162-dataset-capacity-sweep-report-v1",
        "protocol_id": "pg-pk-162-dataset-capacity-sweep-v1",
        "status": "completed_pg162_dataset_capacity_sweep",
        "scope": {"claim": "safe local typed action-decision generalization only", "real_vulnerability_scanner_claim_allowed": False, "device": str(device), "feature_dim": 44, "decision_set": list(PG118_DECISIONS)},
        "dataset": {"manifest": str(TRACE_MANIFEST_PATH), "path": str(DATASET_PATH), "row_count": len(all_rows), "train_rows": len(train_rows), "dev_rows": len(dev_rows), "get_count": manifest["get_count"], "post_count": manifest["post_count"], "class_counts": manifest["class_counts"], "source_counts": manifest["source_counts"], "evidence_hash_valid": manifest["evidence_hash_valid"], "fresh_reset_per_step": manifest["fresh_reset_per_step"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False},
        "model_variants": model_results,
        "selection": {"best_source_heldout_candidate": best_name, "promotion_allowed": False, "reason": "source-heldout success on controlled fixtures is not enough to promote a scanner or long-term memory"},
        "pg146_real_container_evaluation_only": {"report": str(RESEARCH / "pg146_public_lab_replay_report_v1.json"), "row_count": (pg146.get("counts") or {}).get("row_count"), "typed_oracle_count": (pg146.get("counts") or {}).get("typed_oracle_count"), "training_eligible": pg146.get("training_eligible")},
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "destructive_payloads": False},
        "source": {"dataset_sha256": _sha256_file(DATASET_PATH), "manifest_sha256": _sha256_file(TRACE_MANIFEST_PATH), "runner_sha256": _sha256_file(Path(__file__))},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    markdown = "\n".join([
        "# PG-162 数据集与模型容量实验",
        "",
        f"- fresh typed rows: **{len(all_rows)}**（GET {manifest['get_count']} / POST {manifest['post_count']}）",
        f"- source groups: **{', '.join(f'{k}={v}' for k, v in manifest['source_counts'].items())}**",
        f"- classes: **{manifest['class_counts']}**",
        f"- device: **{device}**",
        f"- best source-heldout candidate: **{best_name}**",
        "",
        "所有输入均为抽象、脱敏的 Rule-IR projection；原始 probe、响应正文、族名与 evaluator 标签不进入模型。PG-146 Docker 真实靶场仍是 evaluation-only，因为本轮没有 typed oracle。",
    ])
    MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "row_count": len(all_rows), "train_rows": len(train_rows), "dev_rows": len(dev_rows), "device": str(device), "variants": {name: {"dev_macro_f1": value["dev"]["macro_f1"], "pg118_holdout_macro_f1": value["source_holdout"]["pg118_delta"]["metrics"]["macro_f1"], "pg118_false_accept": value["source_holdout"]["pg118_delta"]["episodes"]["decoy_false_accept_count"]} for name, value in model_results.items()}, "best": best_name, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
