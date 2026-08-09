"""PG-168: add discriminative, family-free Rule-IR slots and train a 101M model.

The generator creates local, inert abstract traces.  It deliberately contains
no raw probe, response body, vulnerability family, oracle decision, or payload.
Two complete sink-slot values are held out so OOD can be checked by exact
projection signature, not by a random row split.  The experiment compares a
replay-only branch with the same 101M checkpoint plus the new slot corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg168-discriminative-slot-augmentation-v1"
DATASET_PATH = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg168_discriminative_slot_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg168_discriminative_slot_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg168_discriminative_slot_report_v1.md"
MAX_LEN = 128
SEED = 16801
TRAIN_ROWS = 8000
DEV_ROWS = 1000
OOD_ROWS = 1000

TRAIN_SINKS = (
    "html_dom_sink",
    "query_ast_boundary",
    "xml_entity_parser",
    "html_attribute",
    "html_text",
    "dom_template",
)
OOD_SINKS = ("sql_ast_boundary", "access_transition")
STYLES = ("govuk_like", "material_like", "primer_like", "dashboard_like", "dense_like", "minimal_like")
JS_SHAPES = ("progressive_enhancement", "client_router", "event_listener", "fetch_form", "filter_table_callback", "none")
METHODS = ("GET", "POST")
PLACEMENTS = ("query", "form", "json", "body_field", "path")
ENCODING_DEPTHS = (1, 2, 3)
ROUTE_CLASSES = ("api_surface", "comment_surface", "login_surface", "profile_surface", "search_surface", "upload_surface")
CONTENT_TYPES = ("html", "json", "text", "xml", "unknown")
STATUS_CLASSES = ("2xx", "3xx", "4xx")
SHAPE_CLASSES = ("stable", "steady", "transition-v2", "visual-shift-v2", "policy-transition")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slot_tokens(rng: random.Random, sink: str) -> list[str]:
    method = rng.choice(METHODS)
    placement = rng.choice(PLACEMENTS)
    depth = rng.choice(ENCODING_DEPTHS)
    return [
        "[BOS]",
        "[STEP]",
        "[SRC_HTML]",
        f"src.html.style={rng.choice(STYLES)}",
        "src.html.syntax=bounded_template",
        "[SRC_JAVASCRIPT]",
        f"src.javascript.shape={rng.choice(JS_SHAPES)}",
        "[SRC_TRANSPORT]",
        f"src.transport.method={method}",
        f"src.transport.placement={placement}",
        f"src.transport.encoding_depth={depth}",
        f"src.transport.route_class={rng.choice(ROUTE_CLASSES)}",
        "src.transport.route=loopback_allowlisted",
        "[IR]",
        "ir.surface.family_free=true",
        f"ir.surface.sink_class={sink}",
        "ir.probe.shape=bounded_marker",
        f"ir.response.content_type={rng.choice(CONTENT_TYPES)}",
        f"ir.response.status_class={rng.choice(STATUS_CLASSES)}",
        f"ir.response.shape_class={rng.choice(SHAPE_CLASSES)}",
        "ir.response.candidate_signal=false",
        "ir.response.transition_delta=unknown",
        "[OBS]",
        "obs.oracle=unknown_oracle",
        "[EOS]",
    ]


def _generate_rows(vocab: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    counts = {"train": 0, "dev": 0, "ood": 0}
    # The sink slot defines the held-out axis.  Other slots vary independently
    # to prevent the model from memorising a single style or transport surface.
    targets = [("train", TRAIN_ROWS), ("dev", DEV_ROWS), ("ood", OOD_ROWS)]
    for split, target in targets:
        sinks = TRAIN_SINKS if split != "ood" else OOD_SINKS
        attempts = 0
        while counts[split] < target:
            attempts += 1
            if attempts > target * 100:
                raise RuntimeError(f"could not generate unique {split} projections")
            sink = rng.choice(sinks)
            tokens = _slot_tokens(rng, sink)
            signature = tuple(tokens)
            if signature in seen:
                continue
            seen.add(signature)
            if any(token not in vocab for token in tokens):
                raise RuntimeError(f"token missing from frozen vocabulary: {next(token for token in tokens if token not in vocab)}")
            rows.append({"row_id": f"pg168-{split}-{counts[split] + 1:05d}", "split": split, "tokens": tokens, "projection_sha256": _sha256_json(tokens)})
            counts[split] += 1
    return rows, counts


class _Dataset(Dataset[dict[str, list[int]]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.ids = [[stoi.get(token, stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]] for row in rows]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"ids": self.ids[index]}


def _collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {"ids": ids, "mask": ids.ne(0)}


def _metrics(model: nn.Module, rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, float | int]:
    loader = DataLoader(_Dataset(rows, stoi), batch_size=8, shuffle=False, collate_fn=_collate)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            logits = model(ids[:, :-1], mask[:, :-1])
            targets = ids[:, 1:]
            valid = targets.ne(0)
            total_loss += float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum").item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 8), "perplexity": round(math.exp(min(mean, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _load_model(checkpoint: dict[str, Any], device: torch.device) -> CausalTraceTransformer:
    config = checkpoint["config"]
    model = CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _train(name: str, checkpoint: dict[str, Any], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = _load_model(checkpoint, device)
    loader = DataLoader(_Dataset(rows, stoi), batch_size=8, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    started = time.perf_counter()
    model.train()
    loss_sum = 0.0
    token_sum = 0
    for batch in loader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(ids[:, :-1], mask[:, :-1])
            targets = ids[:, 1:]
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        valid = int(targets.ne(0).sum().item())
        loss_sum += float(loss.detach().cpu()) * valid
        token_sum += valid
    result = {"strategy": name, "train_row_count": len(rows), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "slot_dev": _metrics(model, eval_rows["slot_dev"], stoi, device), "slot_ood": _metrics(model, eval_rows["slot_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg168-discriminative-slot-augmentation-v1", "strategy": name, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    vocab = list(checkpoint["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocab)}
    rows, counts = _generate_rows(set(vocab))
    train_slots = [row for row in rows if row["split"] == "train"]
    dev_slots = [row for row in rows if row["split"] == "dev"]
    ood_slots = [row for row in rows if row["split"] == "ood"]
    train_base = [{"tokens": row["tokens"]} for row in base["train_rows"]]
    train_augmented = train_base + train_slots
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "slot_dev": dev_slots, "slot_ood": ood_slots}
    all_projection_overlap = len({tuple(row["tokens"]) for row in dev_slots} & {tuple(row["tokens"]) for row in ood_slots})
    if all_projection_overlap != 0:
        raise RuntimeError("slot OOD projection collision detected")
    dataset = {"schema_version": "pg168-discriminative-slot-dataset-v1", "purpose": "family-free abstract Rule-IR slot augmentation", "source": {"generator": "local_deterministic_synthetic", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "base_checkpoint": str(CHECKPOINT.relative_to(ROOT))}, "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "model_vocabulary_frozen": True, "memory_promotion_allowed": False}, "slot_axes": {"train_sink_classes": list(TRAIN_SINKS), "ood_sink_classes": list(OOD_SINKS), "styles": list(STYLES), "methods": list(METHODS), "placements": list(PLACEMENTS), "route_classes": list(ROUTE_CLASSES)}, "counts": counts, "rows": rows}
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {key: _metrics(baseline_model, value, stoi, device) for key, value in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = {"replay_only": _train("replay_only", checkpoint, train_base, eval_rows, stoi, device), "slot_augmented": _train("slot_augmented", checkpoint, train_augmented, eval_rows, stoi, device)}
    report = {"schema_version": "pg168-discriminative-slot-report-v1", "protocol_id": "pg-pk-168-discriminative-slot-v1", "status": "completed_pg168_discriminative_slot_augmentation", "scope": {"claim": "abstract Rule-IR slot information and OOD collision diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"slot_train_count": len(train_slots), "slot_dev_count": len(dev_slots), "slot_ood_count": len(ood_slots), "replay_train_count": len(train_base), "augmented_train_count": len(train_augmented), "projection_overlap_dev_ood": all_projection_overlap}, "baseline": baseline, "variants": results, "interpretation": {"slot_ood_projection_isolated": all_projection_overlap == 0, "vulnerability_claim_allowed": False, "promotion_allowed": False, "next_token_lm_only": True}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "base_dataset_sha256": base.get("dataset_sha256"), "slot_dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-168-discriminative-slot-v1", "schema_version": "pg168-discriminative-slot-protocol-v1", "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "model_parameter_count": 101380329, "strategies": {"replay_only": {"rows": len(train_base), "lr": 2e-5, "epochs": 1}, "slot_augmented": {"rows": len(train_augmented), "lr": 2e-5, "epochs": 1}}, "ood_gate": {"requires_projection_overlap_zero": True, "observed_projection_overlap": all_projection_overlap, "claim_allowed": all_projection_overlap == 0}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-168 discriminative Rule-IR slots", "", f"- slot train/dev/OOD: **{len(train_slots)} / {len(dev_slots)} / {len(ood_slots)}**", f"- replay-only OOD PPL: **{results['replay_only']['slot_ood']['perplexity']}**", f"- slot-augmented OOD PPL: **{results['slot_augmented']['slot_ood']['perplexity']}**", f"- dev/OOD projection overlap: **{all_projection_overlap}**", "", "该轮只验证抽象 slot 信息量与下一个 token 训练；不产生漏洞标签，不晋级长期记忆。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "counts": counts, "projection_overlap": all_projection_overlap, "replay_only": {"dev_ppl": results["replay_only"]["slot_dev"]["perplexity"], "ood_ppl": results["replay_only"]["slot_ood"]["perplexity"]}, "slot_augmented": {"dev_ppl": results["slot_augmented"]["slot_dev"]["perplexity"], "ood_ppl": results["slot_augmented"]["slot_ood"]["perplexity"]}, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
