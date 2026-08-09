"""Train/evaluate pair-invariant Rule IR decoders on the Pikachu catalog.

The experiment compares the same architecture with and without an embedding
consistency term.  Pair ids, family labels, and surface roles are used only
by the training/evaluation harness; the neural input is the visible catalog
projection.  Outputs remain grammar-checked abstract Rule IR templates.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import (  # noqa: E402
    CATALOG_DECODER_FAMILIES,
    CatalogRuleIRDecoderV2,
    catalog_feature_vector,
    catalog_visible_trace,
)
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.pikachu_replay_collector import PIKACHU_BASE_URL, PIKACHU_IMAGE_DIGEST  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM, trace_feature_vector, validate_abstract_rule_ir  # noqa: E402


PROTOCOL_ID = "pg-pk-02-pair-invariance-training-v1"
CATALOG_PATH = ROOT / "research" / "pikachu_paired_catalog_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg-pk-02-pair-invariance"
REPORT_PATH = ROOT / "research" / "pikachu_pair_invariance_training_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_pair_invariance_training_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pikachu_pair_invariance_protocol_v1.json"
SEED = 20260803
EPOCHS = 260
PAIR_WEIGHT = 0.35
ABSTAIN_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.10
# trace_feature_vector layout: string flag 4/5 and the 24 bounded
# decode-depth flags.  These features remain available to the pair embedding
# consistency term, but the family classifier can be trained without them.
ENCODING_FEATURE_INDICES = (59, 60, *range(120, 144))


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)


def _normalise(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _mask_encoding_features(features: torch.Tensor) -> torch.Tensor:
    masked = features.clone()
    masked[:, list(ENCODING_FEATURE_INDICES)] = 0.0
    return masked


def _pair_consistency_loss(embeddings: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    """Pull same pair_id together while retaining surface/encoding views."""

    normalized = F.normalize(embeddings, dim=-1)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair"]["pair_id"])].append(index)
    losses: list[torch.Tensor] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for left, right in itertools.combinations(indices, 2):
            losses.append(1.0 - (normalized[left] * normalized[right]).sum())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def _train(
    train_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    pair_weight: float,
    encoding_invariant: bool,
    seed: int,
) -> tuple[CatalogRuleIRDecoderV2, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    raw = _features(train_rows)
    base, mean, std = _normalise(raw)
    labels = torch.tensor([CATALOG_DECODER_FAMILIES.index(row["semantic"]["family"]) for row in train_rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(CATALOG_DECODER_FAMILIES)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    features = base.to(device)
    target = labels.to(device)
    model = CatalogRuleIRDecoderV2(branch_dim=112, embedding_dim=72, dropout=0.06).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=0.025)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        classifier_features = _mask_encoding_features(features) if encoding_invariant else features
        logits = model(classifier_features)
        embeddings = model.encode(features)
        classification_loss = loss_fn(logits, target)
        pair_loss = _pair_consistency_loss(embeddings, train_rows)
        loss = classification_loss + float(pair_weight) * pair_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_logits = model(features).cpu()
            accuracy = float(train_logits.argmax(dim=-1).eq(labels).float().mean())
        loss_value = float(loss.detach().cpu())
        history.append({
            "epoch": epoch,
            "loss": round(loss_value, 6),
            "classification_loss": round(float(classification_loss.detach().cpu()), 6),
            "pair_loss": round(float(pair_loss.detach().cpu()), 6),
            "train_accuracy": round(accuracy, 6),
        })
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss_value < best_loss):
            best_accuracy = accuracy
            best_loss = loss_value
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "train_accuracy": best_accuracy,
        "family_support": {
            family: int(counts[index]) for index, family in enumerate(CATALOG_DECODER_FAMILIES) if counts[index] > 0
        },
        "pair_weight": pair_weight,
        "encoding_invariant": encoding_invariant,
        "history_tail": history[-5:],
    }


def _embedding_cosine(model: CatalogRuleIRDecoderV2, rows: list[dict[str, Any]], features: torch.Tensor, *, device: torch.device) -> dict[str, float]:
    if not rows:
        return {}
    with torch.inference_mode():
        embeddings = F.normalize(model.encode(features.to(device)), dim=-1).cpu()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair = row["pair"]
        key = f"{pair['pair_id']}::{pair['surface_role']}"
        groups[key].append(index)
    scores: dict[str, float] = {}
    for key, indices in groups.items():
        values = [float((embeddings[left] * embeddings[right]).sum()) for left, right in itertools.combinations(indices, 2)]
        if values:
            scores[key] = round(sum(values) / len(values), 6)
    return scores


def _evaluate(
    model: CatalogRuleIRDecoderV2,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    fit: dict[str, Any],
    *,
    split: str,
    encoding_invariant: bool,
    pair_agreement_gate: bool = False,
    device: torch.device,
) -> dict[str, Any]:
    mean = torch.tensor(fit["normalisation_mean"])
    std = torch.tensor(fit["normalisation_std"])
    train_features = (_features(train_rows) - mean) / std
    test_features = (_features(test_rows) - mean) / std
    classifier_features = _mask_encoding_features(test_features) if encoding_invariant else test_features
    decoded = model.decode(
        classifier_features.to(device),
        abstain_threshold=ABSTAIN_THRESHOLD,
        margin_threshold=MARGIN_THRESHOLD,
    )
    predictions: list[dict[str, Any]] = []
    for row, output in zip(test_rows, decoded):
        prediction = dict(output)
        expected = row["semantic"]["family"]
        predicted = prediction.get("family")
        if prediction.get("rule_ir") is not None:
            validate_abstract_rule_ir(prediction["rule_ir"])
        predictions.append({
            "sample_id": row["sample_id"],
            "pair_id": row["pair"]["pair_id"],
            "surface_role": row["pair"]["surface_role"],
            "variant": row["pair"]["variant"],
            "expected_family": expected,
            "predicted_family": predicted,
            "candidate_family": prediction.get("candidate_family"),
            "confidence": prediction.get("confidence", 0.0),
            "margin": prediction.get("margin", 0.0),
            "abstained": bool(prediction.get("abstained")),
            "exit_found": predicted == expected and prediction.get("rule_ir") is not None,
            "false_positive": predicted is not None and predicted != expected,
            "rule_ir_emitted": prediction.get("rule_ir") is not None,
        })
    if pair_agreement_gate:
        # A final Rule IR emission requires agreement across the safe encoding
        # views of the same surface.  Disagreement is useful feedback for the
        # active-probe controller, but is not a reason to guess.
        agreement_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            agreement_groups[f"{row['pair_id']}::{row['surface_role']}"] .append(row)
        for group in agreement_groups.values():
            if len(group) < 2 or len({row["candidate_family"] for row in group}) <= 1:
                continue
            for row in group:
                row["predicted_family"] = None
                row["abstained"] = True
                row["abstain_reason"] = "pair_variant_disagreement"
                row["exit_found"] = False
                row["false_positive"] = False
                row["rule_ir_emitted"] = False
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[f"{row['pair_id']}::{row['surface_role']}"] .append(row)
    consistency_values: list[float] = []
    accepted_consistency_values: list[float] = []
    for group in groups.values():
        candidates = [row["candidate_family"] for row in group]
        accepted = [row["predicted_family"] for row in group if row["predicted_family"] is not None]
        consistency_values.append(float(len(set(candidates)) == 1))
        if accepted:
            accepted_consistency_values.append(float(len(set(accepted)) == 1))
    total = len(predictions)
    embedding_scores = _embedding_cosine(model, test_rows, test_features, device=device)
    return {
        "split": split,
        "total": total,
        "exit_found_rate": round(sum(row["exit_found"] for row in predictions) / total, 6) if total else 0.0,
        "false_positive_rate": round(sum(row["false_positive"] for row in predictions) / total, 6) if total else 0.0,
        "abstain_rate": round(sum(row["abstained"] for row in predictions) / total, 6) if total else 0.0,
        "rule_ir_emission_rate": round(sum(row["rule_ir_emitted"] for row in predictions) / total, 6) if total else 0.0,
        "pair_agreement_gate": pair_agreement_gate,
        "candidate_family_consistency_rate": round(sum(consistency_values) / len(consistency_values), 6) if consistency_values else 0.0,
        "accepted_family_consistency_rate": round(sum(accepted_consistency_values) / len(accepted_consistency_values), 6) if accepted_consistency_values else 0.0,
        "mean_pair_embedding_cosine": round(sum(embedding_scores.values()) / len(embedding_scores), 6) if embedding_scores else 0.0,
        "pair_embedding_cosines": embedding_scores,
        "predictions": predictions,
    }


def _split(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants_train = {"plain", "url_percent"}
    variants_test = {"html_entity", "double_html_entity"}
    train_surface = {"reflected_get", "sqli_str", "sqli_search"}
    test_surface = {"dom_value_source", "sqli_blind_boolean", "sqli_blind_time"}
    if split == "encoding_holdout":
        return (
            [row for row in rows if row["pair"]["variant"] in variants_train],
            [row for row in rows if row["pair"]["variant"] in variants_test],
        )
    if split == "surface_holdout":
        return (
            [row for row in rows if row["pair"]["surface_role"] in train_surface],
            [row for row in rows if row["pair"]["surface_role"] in test_surface],
        )
    if split == "joint_holdout":
        return (
            [row for row in rows if row["pair"]["variant"] in variants_train and row["pair"]["surface_role"] in train_surface],
            [row for row in rows if row["pair"]["variant"] in variants_test and row["pair"]["surface_role"] in test_surface],
        )
    raise ValueError(f"unknown pair split: {split}")


def _ablate_visible_trace(row: dict[str, Any], mode: str) -> dict[str, Any]:
    trace = copy.deepcopy(catalog_visible_trace(row))
    if mode == "no_encoding":
        trace["input"]["encoding"] = ""
    elif mode == "no_probe":
        trace["input"]["probe"] = ""
        trace["input"]["probe_kind"] = "generic_canary"
        trace["input"]["encoding"] = ""
    elif mode == "response_only":
        trace["input"] = {}
    elif mode == "no_oracle_shape":
        trace["context"]["oracle_shape"] = {"field_count": 0}
    elif mode != "full":
        raise ValueError(f"unknown feature audit mode: {mode}")
    return trace


def _feature_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether observable feature groups are carrying useful signal."""

    modes = ("full", "no_encoding", "no_probe", "response_only", "no_oracle_shape")
    result: dict[str, Any] = {
        "visible_groups": [
            "request method",
            "neutral route shape (depth/extension/query flag; no route tokens)",
            "probe modality and inert representation",
            "encoding descriptor",
            "response status/content type/body length",
            "bounded structural/oracle field count",
        ],
        "excluded_groups": [
            "raw response body",
            "cookies and credentials",
            "source code and evaluator state",
            "pair_id, family label, surface role",
            "raw route/family tokens",
        ],
        "nearest_neighbor": {},
    }
    for split in ("encoding_holdout", "joint_holdout"):
        train_rows, test_rows = _split(rows, split)
        split_result: dict[str, Any] = {}
        for mode in modes:
            train_features = torch.tensor([trace_feature_vector([_ablate_visible_trace(row, mode)]) for row in train_rows])
            test_features = torch.tensor([trace_feature_vector([_ablate_visible_trace(row, mode)]) for row in test_rows])
            distances = torch.cdist(test_features, train_features)
            nearest = distances.argmin(dim=1).tolist()
            predicted = [train_rows[index]["semantic"]["family"] for index in nearest]
            expected = [row["semantic"]["family"] for row in test_rows]
            split_result[mode] = {
                "accuracy": round(sum(left == right for left, right in zip(predicted, expected)) / len(expected), 6),
                "sample_count": len(expected),
            }
        result["nearest_neighbor"][split] = split_result
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-02 配对不变性训练",
        "",
        "每个 split 比较无配对损失（baseline）和加入同 pair embedding consistency 的模型。输入只包含可观察的 action/probe/response shape/encoding descriptor；pair id、family、surface role 只在训练/评估 harness 中使用。",
        "",
        "| split | model | exit | false positive | abstain | candidate consistency | embedding cosine |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["evaluations"]:
        lines.append(
            f"| {row['split']} | {row['model']} | {row['exit_found_rate']:.2f} | {row['false_positive_rate']:.2f} | "
            f"{row['abstain_rate']:.2f} | {row['candidate_family_consistency_rate']:.2f} | {row['mean_pair_embedding_cosine']:.2f} |"
        )
    lines.extend([
        "",
        "解释：配对损失的目标是让同一抽象族的编码/表面变体在表示空间接近，而不是强迫模型对新表面硬猜。错误或高不确定性结果仍必须 abstain。",
        "",
        "## 特征相关性审计",
        "",
        "下表是 1-NN 的快速消融，不是最终模型分数；它用于发现明显的捷径或无关输入。",
        "",
        "| split | full | no encoding | no probe | response only | no oracle shape |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for split, rows in report["feature_audit"]["nearest_neighbor"].items():
        lines.append(
            f"| {split} | {rows['full']['accuracy']:.2f} | {rows['no_encoding']['accuracy']:.2f} | "
            f"{rows['no_probe']['accuracy']:.2f} | {rows['response_only']['accuracy']:.2f} | {rows['no_oracle_shape']['accuracy']:.2f} |"
        )
    lines.extend([
        "",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    rows = flatten_catalog(load_catalog(CATALOG_PATH))
    visible = json.dumps([catalog_visible_trace(row) for row in rows], ensure_ascii=False).casefold()
    assert "pair_id" not in visible and "evaluator" not in visible and "intended_output" not in visible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluations: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    for split_index, split in enumerate(("encoding_holdout", "surface_holdout", "joint_holdout")):
        train_rows, test_rows = _split(rows, split)
        model_configs = (
            ("baseline", 0.0, False),
            ("pair_consistency", PAIR_WEIGHT, False),
            ("pair_encoding_invariant", PAIR_WEIGHT, True),
        )
        for config_index, (model_name, pair_weight, encoding_invariant) in enumerate(model_configs):
            model, fit = _train(
                train_rows,
                device=device,
                pair_weight=pair_weight,
                encoding_invariant=encoding_invariant,
                seed=SEED + split_index * 10 + config_index,
            )
            evaluation = _evaluate(
                model,
                train_rows,
                test_rows,
                fit,
                split=split,
                encoding_invariant=encoding_invariant,
                device=device,
            )
            evaluation["model"] = model_name
            evaluation["train_count"] = len(train_rows)
            evaluation["test_count"] = len(test_rows)
            evaluations.append(evaluation)
            if model_name == "pair_encoding_invariant":
                consensus = _evaluate(
                    model,
                    train_rows,
                    test_rows,
                    fit,
                    split=split,
                    encoding_invariant=encoding_invariant,
                    pair_agreement_gate=True,
                    device=device,
                )
                consensus["model"] = "pair_encoding_invariant_consensus"
                consensus["train_count"] = len(train_rows)
                consensus["test_count"] = len(test_rows)
                evaluations.append(consensus)
            checkpoint_dir = OUTPUT_DIR / split / model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "decoder.pt"
            torch.save({
                "schema_version": "sift-pikachu-pair-invariance-decoder-v1",
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "feature_dim": FEATURE_DIM,
                "families": list(CATALOG_DECODER_FAMILIES),
                "normalisation_mean": fit["normalisation_mean"],
                "normalisation_std": fit["normalisation_std"],
                "pair_weight": pair_weight,
                "encoding_invariant": encoding_invariant,
                "abstain_threshold": ABSTAIN_THRESHOLD,
                "margin_threshold": MARGIN_THRESHOLD,
                "seed": SEED + split_index * 10 + int(pair_weight > 0),
                "device_at_training": str(device),
            }, checkpoint_path)
            checkpoints.append(str(checkpoint_path.relative_to(ROOT)))
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-pair-invariance-training-report-v1",
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": load_catalog(CATALOG_PATH)["catalog_sha256"],
            "sample_count": len(rows),
        },
        "model": {
            "class": "CatalogRuleIRDecoderV2",
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "feature_dim": FEATURE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "pair_loss_weight": PAIR_WEIGHT,
            "encoding_feature_indices_masked_for_invariant_head": list(ENCODING_FEATURE_INDICES),
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "free_form_payload_generation": False,
        },
        "split_design": {
            "encoding_holdout": "train plain+url_percent; test html_entity+double_html_entity",
            "surface_holdout": "train reflected_get+sqli_str+sqli_search; test dom_value_source+sqli_blind_boolean+sqli_blind_time",
            "joint_holdout": "encoding and surface held out together",
            "pair_metadata_hidden_from_decoder": True,
        },
        "evaluations": evaluations,
        "feature_audit": _feature_audit(rows),
        "checkpoints": checkpoints,
        "target_scope": {
            "base_url": PIKACHU_BASE_URL,
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
            "independent_target": False,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
        },
        "evaluator_confirmation_count": 0,
        "public_corpus_ingested": False,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "model": report["model"],
        "evaluations": [{key: value for key, value in row.items() if key not in {"predictions", "pair_embedding_cosines"}} for row in evaluations],
        "checkpoints": checkpoints,
        "report": report["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
