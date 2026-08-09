"""PG-170: cross-generator OOD with a fixed 101M replay-anchored model."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg168_discriminative_slot_augmentation import _Dataset, _collate, _load_model, _metrics  # noqa: E402


RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PRIOR_SLOT_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
DATASET_PATH = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg170_cross_generator_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg170_cross_generator_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg170_cross_generator_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg170-cross-generator-ood-v1"
SEED = 17001
MAX_LEN = 128
TRAIN_COUNT = 4000
DEV_COUNT = 1000
OOD_COUNT = 1000
TRAIN_SINKS = ("html_dom_sink", "query_ast_boundary", "xml_entity_parser", "html_attribute", "html_text")
OOD_SINKS = ("dom_template", "sql_ast_boundary", "access_transition")
SYNTAXES = ("component_tree", "mixed_markup", "server_rendered")
STYLES = ("govuk_like", "material_like", "primer_like", "dashboard_like", "dense_like", "minimal_like")
JS_SHAPES = ("progressive_enhancement", "client_router", "event_listener", "fetch_form", "filter_table_callback", "none")
METHODS = ("GET", "POST")
PLACEMENTS = ("query", "form", "json", "body_field", "path")
DEPTHS = (1, 2, 3)
ROUTES = ("api_surface", "comment_surface", "login_surface", "profile_surface", "search_surface", "upload_surface")
CONTENT_TYPES = ("html", "json", "text", "xml", "unknown")
STATUS = ("2xx", "3xx", "4xx")
SHAPES = ("stable", "steady", "transition-v2", "visual-shift-v2", "policy-transition")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tokens(rng: random.Random, sink: str) -> list[str]:
    return [
        "[BOS]", "[STEP]", "[SRC_HTML]", f"src.html.style={rng.choice(STYLES)}", f"src.html.syntax={rng.choice(SYNTAXES)}",
        "[SRC_JAVASCRIPT]", f"src.javascript.shape={rng.choice(JS_SHAPES)}", "[SRC_TRANSPORT]", f"src.transport.method={rng.choice(METHODS)}",
        f"src.transport.placement={rng.choice(PLACEMENTS)}", f"src.transport.encoding_depth={rng.choice(DEPTHS)}", f"src.transport.route_class={rng.choice(ROUTES)}",
        "src.transport.route=loopback_allowlisted", "[IR]", "ir.surface.family_free=true", f"ir.surface.sink_class={sink}",
        "ir.probe.shape=bounded_marker", f"ir.response.content_type={rng.choice(CONTENT_TYPES)}", f"ir.response.status_class={rng.choice(STATUS)}",
        f"ir.response.shape_class={rng.choice(SHAPES)}", "ir.response.candidate_signal=false", "ir.response.transition_delta=unknown", "[OBS]", "obs.oracle=unknown_oracle", "[EOS]",
    ]


def _generate(vocab: set[str], prior_signatures: set[tuple[str, ...]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    seen = set(prior_signatures)
    counts = {"train": 0, "dev": 0, "ood": 0}
    for split, target in (("train", TRAIN_COUNT), ("dev", DEV_COUNT), ("ood", OOD_COUNT)):
        sinks = TRAIN_SINKS if split != "ood" else OOD_SINKS
        attempts = 0
        while counts[split] < target:
            attempts += 1
            if attempts > target * 200:
                raise RuntimeError(f"unable to generate collision-free {split} rows")
            tokens = _tokens(rng, rng.choice(sinks))
            signature = tuple(tokens)
            if signature in seen:
                continue
            missing = [token for token in tokens if token not in vocab]
            if missing:
                raise RuntimeError(f"frozen vocabulary missing token: {missing[0]}")
            seen.add(signature)
            counts[split] += 1
            rows.append({"row_id": f"pg170-{split}-{counts[split]:05d}", "generator": "independent_syntax_v2", "split": split, "tokens": tokens, "projection_sha256": _sha256_json(tokens)})
    return rows, counts


def _train(checkpoint: dict[str, Any], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
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
    result = {"train_row_count": len(rows), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "generator_dev": _metrics(model, eval_rows["generator_dev"], stoi, device), "generator_ood": _metrics(model, eval_rows["generator_ood"], stoi, device), "prior_slot_ood": _metrics(model, eval_rows["prior_slot_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "cross_generator.pt"
    torch.save({"schema_version": "pg170-cross-generator-ood-v1", "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_SLOT_DATASET.read_text(encoding="utf-8"))
    vocab = set(checkpoint["vocabulary"])
    prior_signatures = {tuple(row["tokens"]) for row in prior["rows"]}
    rows, counts = _generate(vocab, prior_signatures)
    train_rows = [{"tokens": row["tokens"]} for row in rows if row["split"] == "train"]
    dev_rows = [{"tokens": row["tokens"]} for row in rows if row["split"] == "dev"]
    ood_rows = [{"tokens": row["tokens"]} for row in rows if row["split"] == "ood"]
    prior_ood = [{"tokens": row["tokens"]} for row in prior["rows"] if row["split"] == "ood"]
    current_dev_sig = {tuple(row["tokens"]) for row in dev_rows}
    current_ood_sig = {tuple(row["tokens"]) for row in ood_rows}
    prior_ood_sig = {tuple(row["tokens"]) for row in prior_ood}
    if current_dev_sig & current_ood_sig or current_ood_sig & prior_signatures:
        raise RuntimeError("cross-generator OOD projection collision detected")
    base_rows = [{"tokens": row["tokens"]} for row in base["train_rows"]]
    rng = random.Random(SEED)
    replay_rows = rng.sample(base_rows, len(train_rows)) + train_rows
    rng.shuffle(replay_rows)
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "generator_dev": dev_rows, "generator_ood": ood_rows, "prior_slot_ood": prior_ood}
    dataset = {"schema_version": "pg170-cross-generator-dataset-v1", "purpose": "independent family-free Rule-IR generator for cross-generator OOD", "generator": {"id": "independent_syntax_v2", "syntax_axis": list(SYNTAXES), "train_sink_classes": list(TRAIN_SINKS), "ood_sink_classes": list(OOD_SINKS)}, "counts": counts, "prior_projection_overlap": len(current_ood_sig & prior_signatures), "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "memory_promotion_allowed": False}, "rows": rows}
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {key: _metrics(baseline_model, value, stoi, device) for key, value in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = _train(checkpoint, replay_rows, eval_rows, stoi, device)
    report = {"schema_version": "pg170-cross-generator-report-v1", "protocol_id": "pg-pk-170-cross-generator-v1", "status": "completed_pg170_cross_generator_ood", "scope": {"claim": "cross-generator abstract Rule-IR OOD diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"train_count": len(train_rows), "dev_count": len(dev_rows), "ood_count": len(ood_rows), "prior_slot_ood_count": len(prior_ood), "train_projection_overlap_prior": len({tuple(row["tokens"]) for row in train_rows} & prior_signatures), "ood_projection_overlap_prior": len(current_ood_sig & prior_signatures), "dev_ood_projection_overlap": len(current_dev_sig & current_ood_sig)}, "baseline": baseline, "cross_generator": result, "interpretation": {"cross_generator_ood_isolated": len(current_ood_sig & prior_signatures) == 0 and len(current_dev_sig & current_ood_sig) == 0, "vulnerability_claim_allowed": False, "promotion_allowed": False, "next_token_lm_only": True}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "base_dataset_sha256": base.get("dataset_sha256"), "prior_slot_dataset_sha256": prior.get("dataset_sha256"), "dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-170-cross-generator-v1", "schema_version": "pg170-cross-generator-protocol-v1", "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "prior_dataset": str(PRIOR_SLOT_DATASET.relative_to(ROOT)), "replay_ratio": "1:1", "model_parameter_count": 101380329, "ood_gate": report["interpretation"], "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-170 cross-generator OOD", "", f"- train/dev/OOD: **{len(train_rows)} / {len(dev_rows)} / {len(ood_rows)}**", f"- generator OOD PPL: **{result['generator_ood']['perplexity']}**", f"- prior slot OOD PPL: **{result['prior_slot_ood']['perplexity']}**", f"- projection overlap: **{len(current_ood_sig & prior_signatures)}**", "", "该轮只验证独立生成器的抽象 Rule-IR 泛化，不产生漏洞标签，也不晋级长期记忆。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "counts": counts, "projection_overlap_prior": len(current_ood_sig & prior_signatures), "generator_ood_ppl": result["generator_ood"]["perplexity"], "prior_slot_ood_ppl": result["prior_slot_ood"]["perplexity"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
