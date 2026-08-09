"""PG-370 shared-backbone multi-task Rule-IR candidate.

PG-369 showed that a plain next-token objective collapsed on the combined
dataset.  PG-370 keeps one causal Transformer-MoE backbone but adds separate
heads for:

* full ordered next-token Rule-IR supervision;
* each of the 13 Rule-IR slots;
* ASK/question state;
* repair/next-action state; and
* safe/negative state.

The model sees abstract context tokens only for the structured heads.  The
next-token lane receives ``context + target`` as teacher-forced labels, with
loss applied only to target positions.  Evaluator projections, source
metadata, route names, raw payloads and response bodies are discarded by the
loader before a batch is constructed.

This runner defaults to ``--plan-only``.  ``--cpu-smoke`` is the only local
execution path used by the current research phase; it runs a tiny bounded
subset to validate tensor wiring.  A future remote CUDA run is explicitly
gated by ``BLACKBOX_REMOTE_A800_TRAIN=1`` and ``CUDA_VISIBLE_DEVICES=0``.
No Docker, network, evaluator or raw-wire path exists in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# Direct ``python scripts/run_pg370_...py`` execution places only ``scripts``
# on sys.path.  Add the repository root before importing the shared app
# modules; this does not execute training or contact any external target.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, TARGET_BOS, TARGET_EOS, UNK
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel
from scripts.plan_pg369_multitask_moe_candidate import (
    DEFAULT_INPUTS,
    DEFAULT_AUDITS,
    PROMOTION_KEYS,
    SLOTS,
    _audit_dataset,
    _audit_report,
    _load_json,
    _sha_file,
    derive_multitask_labels,
)

SCHEMA_VERSION = "pg370-multitask-moe-candidate-v1"
SEEDS = (37001, 37002, 37003)
RAW_FRAGMENT_MARKERS = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "response_body_text=",
    "raw_response=",
    "wire=",
    "evaluator=",
    "oracle=",
    "route_literal=",
    "family=",
    "http://",
    "https://",
    "pg367-runtime-canary",
)
LOCKED_DATASET_SHA256 = {
    "pg362": "2f2e270a9143e9488a7e9206cace944abb3b0f37b31bd8b3059bf2bc3c3f4d35",
    "pg367": "aef0788f65b8870bd5ee2a26419e876589d4d8ac4af39cc0ef5f5a97d1df4913",
}
LOCKED_AUDIT_SHA256 = {
    "pg362": "2d1d76179575d0b5de80442a05b094def02a3d02e76f5f38cc719b0a7acfb081",
    "pg367": "060b1f7ba7e0573f611df2c89f780166fc3e422f9efa4245cf97ddd2637c35e5",
}


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _target_values(target_tokens: Sequence[str]) -> dict[str, str]:
    labels = derive_multitask_labels(target_tokens)
    if labels is None:
        raise ValueError("target does not satisfy full Rule-IR slot contract")
    return dict(labels["next_token"])


def _safe_abstract_row(row: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Copy only model-visible abstract fields from a reviewed row."""

    context = row.get("context_tokens")
    target = row.get("target_tokens")
    if not isinstance(context, list) or not isinstance(target, list):
        raise ValueError("row context/target must be lists")
    values = [str(token) for token in [*context, *target]]
    if any(any(marker in token.casefold() for marker in RAW_FRAGMENT_MARKERS) for token in values):
        raise ValueError("raw/evaluator fragment reached model row")
    if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False or row.get("oracle_answer_in_context") is not False:
        raise ValueError("row raw context firewall is not closed")
    if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        raise ValueError("row context firewall mismatch")
    target_values = _target_values([str(token) for token in target])
    # ``source`` is an audit sidecar used only for aggregate accounting.  It
    # is not passed to the model or persisted in a batch/checkpoint.
    return {
        "context_tokens": [str(token) for token in context],
        "target_tokens": [str(token) for token in target],
        "split": str(row.get("split", "")),
        "_source": source,
        "_target_values": target_values,
    }


def load_locked_rows(
    *,
    dataset_paths: Mapping[str, str | Path] = DEFAULT_INPUTS,
    audit_paths: Mapping[str, str | Path] = DEFAULT_AUDITS,
    require_locked_hashes: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load PG-362/PG-367 abstract rows and reject sidecar/raw leakage."""

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    locks: dict[str, Any] = {
        "datasets": {},
        "audits": {},
        "row_audits": {},
        "report_audits": {},
        "declared_vocabulary": set(),
        "declared_slot_values": {key: set() for key in SLOTS},
    }
    for name in ("pg362", "pg367"):
        dataset_path = Path(dataset_paths[name])
        audit_path = Path(audit_paths[name])
        dataset_hash = _sha_file(dataset_path)
        audit_hash = _sha_file(audit_path)
        if require_locked_hashes and dataset_hash != LOCKED_DATASET_SHA256[name]:
            raise ValueError(f"{name} dataset hash is not locked")
        if require_locked_hashes and audit_hash != LOCKED_AUDIT_SHA256[name]:
            raise ValueError(f"{name} audit hash is not locked")
        dataset = _load_json(dataset_path)
        audit_report = _load_json(audit_path)
        row_audit = _audit_dataset(dataset, source_name=name)
        report_audit = _audit_report(audit_report, source_name=name)
        if row_audit["status"] != "passed_candidate_input_audit" or report_audit["status"] != "passed_candidate_audit":
            raise ValueError(f"{name} input audit blocked")
        locks["datasets"][name] = dataset_hash
        locks["audits"][name] = audit_hash
        locks["row_audits"][name] = row_audit
        locks["report_audits"][name] = report_audit
        declared = dataset.get("vocabulary")
        if not isinstance(declared, Mapping):
            raise ValueError(f"{name} dataset is missing declared vocabulary")
        for token in [*declared.get("context_tokens", []), *declared.get("target_tokens", [])]:
            locks["declared_vocabulary"].add(str(token))
        for key, values in _declared_slot_values(dataset).items():
            locks["declared_slot_values"][key].update(values)
        for raw in dataset["records"]:
            # No evaluator_projection/source_meta is read here.
            item = _safe_abstract_row(raw, source=name)
            if item["split"] == "train":
                train.append(item)
            elif item["split"] == "implementation_holdout":
                holdout.append(item)
            else:
                raise ValueError(f"{name} row has unsupported split")
    # Sets are convenient while merging manifests but are not JSON-safe.  The
    # sorted lists also make the lock deterministic and hashable in reports.
    locks["declared_vocabulary"] = sorted(locks["declared_vocabulary"])
    locks["declared_slot_values"] = {
        key: sorted(values) for key, values in sorted(locks["declared_slot_values"].items())
    }
    locks["declared_vocabulary_sha256"] = _sha_json(locks["declared_vocabulary"])
    locks["declared_slot_values_sha256"] = _sha_json(locks["declared_slot_values"])
    return train, holdout, locks


def build_vocabulary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in row["context_tokens"])
        tokens.update(str(token) for token in row["target_tokens"])
    return {token: index for index, token in enumerate([PAD, UNK, *sorted(tokens - {PAD, UNK})])}


def build_declared_vocabulary(tokens: Sequence[str]) -> dict[str, int]:
    """Build an append-only vocabulary from a frozen ontology manifest.

    The manifest is a category inventory, not a row scan.  It is therefore
    allowed to contain a category that happens to occur only in an
    implementation holdout, while the holdout rows themselves remain unread
    by the vocabulary builder.  Callers must record the manifest hash and
    keep this mode separate from the strict train-only diagnostic mode.
    """

    normalized = {str(token) for token in tokens if str(token)}
    normalized.update((PAD, UNK))
    return {token: index for index, token in enumerate([PAD, UNK, *sorted(normalized - {PAD, UNK})])}


def _declared_slot_values(dataset: Mapping[str, Any]) -> dict[str, list[str]]:
    """Extract slot categories from a dataset's declared target inventory."""

    vocabulary = dataset.get("vocabulary")
    target_tokens = vocabulary.get("target_tokens") if isinstance(vocabulary, Mapping) else None
    if not isinstance(target_tokens, list):
        raise ValueError("dataset is missing declared target vocabulary")
    values: dict[str, set[str]] = {key: set() for key in SLOTS}
    for raw in target_tokens:
        token = str(raw)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in values and value:
            values[key].add(value)
    missing = [key for key, items in values.items() if not items]
    if missing:
        raise ValueError(f"declared target vocabulary missing slots: {','.join(missing)}")
    return {key: sorted(items) for key, items in values.items()}


def _slot_classes_from_values(values: Mapping[str, Sequence[str]]) -> dict[str, dict[str, int]]:
    classes: dict[str, dict[str, int]] = {}
    for key in SLOTS:
        choices = sorted({str(value) for value in values.get(key, ()) if str(value)})
        if not choices:
            raise ValueError(f"declared slot vocabulary missing {key}")
        classes[key] = {value: index for index, value in enumerate(choices)}
    return classes


def _vocabulary_gaps(
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    unknown_tokens = sorted(
        {
            str(token)
            for row in holdout_rows
            for token in [*row["context_tokens"], *row["target_tokens"]]
            if str(token) not in vocabulary
        }
    )
    unknown_slots: dict[str, list[str]] = {}
    for key in SLOTS:
        values = sorted(
            {
                str((row.get("_target_values") or _target_values(row["target_tokens"]))[key])
                for row in holdout_rows
                if str((row.get("_target_values") or _target_values(row["target_tokens"]))[key]) not in slot_classes[key]
            }
        )
        if values:
            unknown_slots[key] = values
    return {
        "unknown_token_count": len(unknown_tokens),
        "unknown_tokens_sha256": _sha_json(unknown_tokens),
        "unknown_slot_value_count": sum(len(items) for items in unknown_slots.values()),
        "unknown_slot_values": {key: _sha_json(items) for key, items in sorted(unknown_slots.items())},
        "blocked": bool(unknown_tokens or unknown_slots),
    }


def _slot_classes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    classes: dict[str, list[str]] = {key: [] for key in SLOTS}
    for row in rows:
        values = row.get("_target_values") or _target_values(row["target_tokens"])
        for key in SLOTS:
            if values[key] not in classes[key]:
                classes[key].append(values[key])
    return {key: {value: index for index, value in enumerate(sorted(values))} for key, values in classes.items()}


class SharedCausalMoEMultiTask(nn.Module):
    """Shared PG-295 backbone with independent abstract supervision heads."""

    def __init__(self, *, vocab_size: int, config: CausalMoEConfig, slot_classes: Mapping[str, Mapping[str, int]]) -> None:
        super().__init__()
        self.backbone = CausalMoELanguageModel(vocab_size=vocab_size, config=config)
        self.slot_heads = nn.ModuleDict({key: nn.Linear(config.d_model, len(values)) for key, values in slot_classes.items()})
        self.ask_head = nn.Linear(config.d_model, 2)
        self.repair_head = nn.Linear(config.d_model, 4)
        self.negative_head = nn.Linear(config.d_model, 2)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        lm_ids: torch.Tensor | None = None,
        lm_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        context_hidden, context_balance = self.backbone.forward_hidden(context_ids, valid_mask=context_mask)
        lengths = context_mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = context_hidden[torch.arange(context_hidden.shape[0], device=context_hidden.device), lengths]
        output: dict[str, Any] = {
            "slot": {key: head(boundary) for key, head in self.slot_heads.items()},
            "ask": self.ask_head(boundary),
            "repair": self.repair_head(boundary),
            "negative": self.negative_head(boundary),
            "balance": context_balance,
        }
        if lm_ids is not None:
            lm_hidden, lm_balance = self.backbone.forward_hidden(lm_ids, valid_mask=lm_mask)
            output["lm"] = self.backbone.lm_head(lm_hidden)
            output["balance"] = output["balance"] + lm_balance
        return output


def _pad_context(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    sequences = [[int(vocabulary.get(token, vocabulary[UNK])) for token in row["context_tokens"]] for row in rows]
    width = max((len(item) for item in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    lengths: list[int] = []
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = True
        lengths.append(len(sequence))
    return ids, mask, lengths


def _pad_lm(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [[int(vocabulary.get(token, vocabulary[UNK])) for token in [*row["context_tokens"], *row["target_tokens"]]] for row in rows]
    width = max((len(item) for item in sequences), default=2)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    target_mask = torch.zeros((len(sequences), max(width - 1, 1)), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = True
        context_len = len(rows[index]["context_tokens"])
        # labels[j] is predicted by logits[j].  The first target label is at
        # original sequence position ``context_len`` and is therefore emitted
        # from the logit immediately before it (``context_len - 1``).  Keep
        # the complete target span, including TARGET_EOS; silently dropping
        # the first/last label makes the next-token objective look healthier
        # than it is and is especially harmful for short smoke rows.
        if len(sequence) > context_len:
            start = max(context_len - 1, 0)
            end = len(sequence) - 1
            target_mask[index, start:end] = True
    return ids, mask, target_mask[:, : max(width - 1, 1)]


def _batch_labels(rows: Sequence[Mapping[str, Any]], slot_classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, Any]:
    values = [row.get("_target_values") or _target_values(row["target_tokens"]) for row in rows]
    labels = {
        "slot": {key: torch.tensor([slot_classes[key][item[key]] for item in values], dtype=torch.long, device=device) for key in SLOTS},
        "ask": torch.tensor([int(item["question"].startswith("ask_")) for item in values], dtype=torch.long, device=device),
        "repair": torch.tensor([{"select_probe_variant": 0, "replay": 1, "repair": 2, "abstain": 3}.get(item["next_action"], 0) for item in values], dtype=torch.long, device=device),
        "negative": torch.tensor([int(item["safe_to_send"] == "1") for item in values], dtype=torch.long, device=device),
    }
    return labels


def _loss_components(model: SharedCausalMoEMultiTask, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], slot_classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, torch.Tensor]:
    context_ids, context_mask, _ = _pad_context(rows, vocabulary, device)
    lm_ids, lm_mask, target_mask = _pad_lm(rows, vocabulary, device)
    output = model(context_ids, context_mask, lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1])
    labels = _batch_labels(rows, slot_classes, device)
    # Target-only next-token loss keeps long page contexts from dominating the
    # structured objectives, while the complete context remains in the input.
    lm_labels = lm_ids[:, 1:]
    lm_loss = F.cross_entropy(output["lm"].reshape(-1, output["lm"].shape[-1]), lm_labels.reshape(-1), ignore_index=int(vocabulary[PAD]), reduction="none").reshape(lm_labels.shape)
    lm_loss = lm_loss[target_mask].mean() if bool(target_mask.any()) else lm_loss.mean() * 0.0
    slot_loss = torch.stack([F.cross_entropy(output["slot"][key], labels["slot"][key]) for key in SLOTS]).mean()
    ask_loss = F.cross_entropy(output["ask"], labels["ask"])
    repair_loss = F.cross_entropy(output["repair"], labels["repair"])
    negative_loss = F.cross_entropy(output["negative"], labels["negative"])
    return {"next_token": lm_loss, "slot_query": slot_loss, "ask": ask_loss, "repair": repair_loss, "negative": negative_loss, "balance": output["balance"]}


def _entropy(logits: torch.Tensor) -> float:
    probabilities = logits.softmax(dim=-1)
    return float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)).mean().detach().cpu())


def evaluate_multitask(model: SharedCausalMoEMultiTask, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], slot_classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "sequence_exact": None, "slot_accuracy": None, "ask_recall": None, "repair_recall": None, "positive_recall": None, "negative_false_allow": 0, "predictive_entropy": None}
    model.eval()
    reverse_slots = {key: {index: value for value, index in classes.items()} for key, classes in slot_classes.items()}
    totals = Counter()
    entropies: list[float] = []
    with torch.inference_mode():
        for row in rows:
            context_ids, context_mask, context_lengths = _pad_context([row], vocabulary, device)
            lm_ids, lm_mask, target_mask = _pad_lm([row], vocabulary, device)
            output = model(context_ids, context_mask, lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1])
            values = row.get("_target_values") or _target_values(row["target_tokens"])
            predicted_values = {key: reverse_slots[key][int(output["slot"][key].argmax(-1).item())] for key in SLOTS}
            totals["slot_total"] += len(SLOTS)
            totals["slot_correct"] += sum(int(predicted_values[key] == values[key]) for key in SLOTS)
            predicted_ask = int(output["ask"].argmax(-1).item()) == 1
            expected_ask = values["question"].startswith("ask_")
            totals["ask_total"] += int(expected_ask)
            totals["ask_correct"] += int(predicted_ask and expected_ask)
            expected_repair = values["next_action"] == "repair"
            predicted_repair = int(output["repair"].argmax(-1).item()) == 2
            totals["repair_total"] += int(expected_repair)
            totals["repair_correct"] += int(predicted_repair and expected_repair)
            expected_safe = values["safe_to_send"] == "1"
            predicted_safe = int(output["negative"].argmax(-1).item()) == 1
            totals["positive_total"] += int(expected_safe)
            totals["positive_correct"] += int(predicted_safe and expected_safe)
            totals["negative_total"] += int(not expected_safe)
            totals["negative_false_allow"] += int(predicted_safe and not expected_safe)
            lm_labels = lm_ids[:, 1:]
            input_target_mask = target_mask
            positions = torch.nonzero(input_target_mask[0], as_tuple=False).flatten()
            if len(positions):
                logits = output["lm"][0, positions]
                labels = lm_labels[0, positions]
                entropies.append(_entropy(logits))
                totals["token_total"] += len(labels)
                totals["token_correct"] += int((logits.argmax(-1) == labels).sum().item())
                totals["sequence_total"] += 1
                totals["sequence_exact"] += int(bool(torch.all(logits.argmax(-1) == labels).item()))
    return {
        "rows": len(rows),
        "token_accuracy": round(totals["token_correct"] / max(totals["token_total"], 1), 6),
        "sequence_exact": round(totals["sequence_exact"] / max(totals["sequence_total"], 1), 6),
        "slot_accuracy": round(totals["slot_correct"] / max(totals["slot_total"], 1), 6),
        "ask_recall": round(totals["ask_correct"] / max(totals["ask_total"], 1), 6) if totals["ask_total"] else None,
        "repair_recall": round(totals["repair_correct"] / max(totals["repair_total"], 1), 6) if totals["repair_total"] else None,
        "positive_recall": round(totals["positive_correct"] / max(totals["positive_total"], 1), 6) if totals["positive_total"] else None,
        "negative_false_allow": int(totals["negative_false_allow"]),
        "negative_total": int(totals["negative_total"]),
        "predictive_entropy": round(sum(entropies) / max(len(entropies), 1), 6),
    }


def train_one_seed(
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    *,
    seed: int,
    slot_classes: Mapping[str, Mapping[str, int]],
    config: CausalMoEConfig,
    epochs: int,
    microbatch: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    baseline = SharedCausalMoEMultiTask(vocab_size=len(vocabulary), config=config, slot_classes=slot_classes).to(device)
    baseline_metrics = evaluate_multitask(baseline, holdout_rows, vocabulary, slot_classes, device)
    model = SharedCausalMoEMultiTask(vocab_size=len(vocabulary), config=config, slot_classes=slot_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    order = list(range(len(train_rows)))
    for _ in range(max(1, int(epochs))):
        random.Random(int(seed) + _).shuffle(order)
        for start in range(0, len(order), max(1, int(microbatch))):
            batch = [train_rows[index] for index in order[start : start + max(1, int(microbatch))]]
            components = _loss_components(model, batch, vocabulary, slot_classes, device)
            loss = components["next_token"] + components["slot_query"] + 1.5 * components["ask"] + 1.5 * components["repair"] + 2.0 * components["negative"] + 0.01 * components["balance"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    post_metrics = evaluate_multitask(model, holdout_rows, vocabulary, slot_classes, device)
    checkpoint_sha256 = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Only abstract model state and the declared token coordinate system
        # are saved.  No source metadata, evaluator answer, wire or payload is
        # ever part of a PG-370 checkpoint.
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": int(seed),
                "config": dict(config.__dict__),
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "vocabulary": dict(vocabulary),
                "slot_classes": {key: dict(values) for key, values in slot_classes.items()},
            },
            checkpoint_path,
        )
        checkpoint_sha256 = _sha_file(checkpoint_path)
    return {
        "seed": int(seed),
        "baseline": baseline_metrics,
        "post": post_metrics,
        "entropy_relative_drop": round((float(baseline_metrics["predictive_entropy"]) - float(post_metrics["predictive_entropy"])) / max(abs(float(baseline_metrics["predictive_entropy"])), 1e-12), 6),
        "checkpoint": {"path": str(checkpoint_path) if checkpoint_path is not None else None, "sha256": checkpoint_sha256},
    }


def run_candidate(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int] = SEEDS,
    device: str = "cpu",
    epochs: int = 1,
    microbatch: int = 2,
    config: CausalMoEConfig | None = None,
    declared_vocabulary: Sequence[str] | None = None,
    declared_slot_values: Mapping[str, Sequence[str]] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    if device != "cpu" and os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("non-CPU PG-370 execution requires explicit remote training flag")
    torch_device = torch.device(device)
    if device != "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("PG-370 remote lane requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("PG-370 remote lane requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        if "A800" not in torch.cuda.get_device_name(0):
            raise RuntimeError("PG-370 remote lane requires an NVIDIA A800 GPU")
    vocabulary_scope = "train_only"
    if declared_vocabulary is not None:
        vocabulary = build_declared_vocabulary(declared_vocabulary)
        slot_classes = _slot_classes_from_values(declared_slot_values or {})
        vocabulary_scope = "declared_ontology_manifest"
    else:
        # Strict default: an implementation holdout is never allowed to
        # enlarge the token or class set.  This catches the common leakage
        # where a builder silently scans all records before splitting.
        vocabulary = build_vocabulary(train_rows)
        slot_classes = _slot_classes(train_rows)
    vocabulary_gaps = _vocabulary_gaps(train_rows, holdout_rows, vocabulary, slot_classes)
    diagnostic_union = False
    if vocabulary_gaps["blocked"]:
        if device != "cpu":
            raise ValueError("holdout vocabulary is not covered by the locked training/ontology vocabulary")
        # CPU smoke is a tensor-wiring diagnostic only.  It may use a
        # temporary union so the heads can be exercised, but it is explicitly
        # marked and can never be used as a capability result or checkpoint.
        vocabulary = build_vocabulary([*train_rows, *holdout_rows])
        slot_classes = _slot_classes([*train_rows, *holdout_rows])
        diagnostic_union = True
    effective_config = config or CausalMoEConfig(d_model=32, n_heads=4, n_layers=1, experts=2, expert_hidden=64, max_length=768)
    required_window = max(
        (len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]),
        default=1,
    )
    if int(effective_config.max_length) < required_window:
        raise ValueError(f"PG-370 max_length={effective_config.max_length} is below required context window {required_window}")
    results = [
        train_one_seed(
            train_rows,
            holdout_rows,
            vocabulary,
            seed=int(seed),
            slot_classes=slot_classes,
            config=effective_config,
            epochs=epochs,
            microbatch=microbatch,
            device=torch_device,
            checkpoint_path=(checkpoint_dir / f"pg370_seed_{int(seed)}.pt" if checkpoint_dir is not None else None),
        )
        for seed in seeds
    ]
    worst = {
        "sequence_exact_min": min(float(item["post"]["sequence_exact"]) for item in results),
        "slot_accuracy_min": min(float(item["post"]["slot_accuracy"]) for item in results),
        "ask_recall_min": min(float(item["post"]["ask_recall"] or 0.0) for item in results),
        "repair_recall_min": min(float(item["post"]["repair_recall"] or 0.0) for item in results),
        "positive_recall_min": min(float(item["post"]["positive_recall"] or 0.0) for item in results),
        "negative_false_allow_max": max(int(item["post"]["negative_false_allow"]) for item in results),
        "entropy_relative_drop_max": max(float(item["entropy_relative_drop"]) for item in results),
    }
    return {
        "status": "cpu_smoke_candidate_only" if device == "cpu" else "remote_candidate_only",
        "training": {
            "device": device,
            "seeds": [int(seed) for seed in seeds],
            "epochs": int(epochs),
            "microbatch": int(microbatch),
            "action_balance_replication": False,
            "config": effective_config.__dict__,
            "vocabulary_scope": vocabulary_scope,
            "vocabulary_size": len(vocabulary),
            "vocabulary_gaps": vocabulary_gaps,
            "cpu_diagnostic_union": diagnostic_union,
            "required_context_window": int(required_window),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        },
        "candidates": results,
        "worst_seed": worst,
        "promotion": {key: False for key in PROMOTION_KEYS},
        "scientific_gate": {
            "typed_live_replay_with_model_selected_wire": False,
            "independent_implementation": False,
            "claim_allowed": False,
            "holdout_vocabulary_closed": not vocabulary_gaps["blocked"],
        },
    }


def build_plan_report(*, train_rows: Sequence[Mapping[str, Any]], holdout_rows: Sequence[Mapping[str, Any]], locks: Mapping[str, Any], runner_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "plan_only",
        "locks": {"datasets": locks.get("datasets", {}), "audits": locks.get("audits", {}), "runner_sha256": _sha_file(runner_path), "base_model_sha256": _sha_file(ROOT / "app" / "pg295_causal_moe.py")},
        "data": {"train_rows": len(train_rows), "implementation_holdout_rows": len(holdout_rows), "full_context": True, "raw_rows_loaded": False},
        "training": {"seeds": list(SEEDS), "microbatch": 2, "action_balance_replication": False, "device": "not_run"},
        "evaluation": {"metrics": ["predictive_entropy", "sequence_exact", "slot_accuracy", "ask_recall", "repair_recall", "positive_recall", "negative_false_allow"], "status": "not_run"},
        "execution": {"trainer_invoked": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False},
        "promotion": {key: False for key in PROMOTION_KEYS},
        "scientific_gate": {"status": "blocked_candidate_only", "claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-370 shared-backbone multi-task MoE candidate")
    parser.add_argument("--pg362-dataset", type=Path, default=DEFAULT_INPUTS["pg362"])
    parser.add_argument("--pg367-dataset", type=Path, default=DEFAULT_INPUTS["pg367"])
    parser.add_argument("--pg362-audit", type=Path, default=DEFAULT_AUDITS["pg362"])
    parser.add_argument("--pg367-audit", type=Path, default=DEFAULT_AUDITS["pg367"])
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg370_multitask_moe_candidate_v1.json")
    parser.add_argument("--plan-only", action="store_true", default=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--remote-candidate", action="store_true", help="run only on the explicitly gated remote A800 lane")
    parser.add_argument("--smoke-rows", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-hidden", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    train, holdout, locks = load_locked_rows(dataset_paths={"pg362": args.pg362_dataset, "pg367": args.pg367_dataset}, audit_paths={"pg362": args.pg362_audit, "pg367": args.pg367_audit})
    if args.cpu_smoke:
        cap = max(1, int(args.smoke_rows))
        result = run_candidate(train_rows=train[:cap], holdout_rows=holdout[:cap], seeds=SEEDS, device="cpu", epochs=args.epochs, microbatch=args.microbatch)
    elif args.remote_candidate:
        if args.checkpoint_dir is None:
            args.checkpoint_dir = ROOT / "artifacts" / "pg370-multitask-moe-candidate"
        config = CausalMoEConfig(
            d_model=int(args.d_model),
            n_heads=4,
            n_layers=int(args.n_layers),
            experts=int(args.experts),
            expert_hidden=int(args.expert_hidden),
            max_length=768,
        )
        result = run_candidate(
            train_rows=train,
            holdout_rows=holdout,
            seeds=SEEDS,
            device="cuda:0",
            epochs=int(args.epochs),
            microbatch=int(args.microbatch),
            config=config,
            declared_vocabulary=locks["declared_vocabulary"],
            declared_slot_values=locks["declared_slot_values"],
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        result = build_plan_report(train_rows=train, holdout_rows=holdout, locks=locks, runner_path=Path(__file__))
    result["schema_version"] = SCHEMA_VERSION
    result["locks"] = result.get("locks", locks)
    result["locks"]["rules_sha256"] = _sha_file(ROOT / "research" / "improvement_rules.json")
    result["report_sha256"] = _sha_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
