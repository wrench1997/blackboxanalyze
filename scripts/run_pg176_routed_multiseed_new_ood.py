"""PG-176: multi-seed replay routing and fourth-generator OOD audit."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg175_joint_routing_loss_search import _load_sources, _make_train_rows, _route  # noqa: E402


RESEARCH = ROOT / "research"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PG168_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
PG170_DATASET = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PG172_DATASET = RESEARCH / "pg172_third_generator_dataset_v1.json"
START_CHECKPOINT = ROOT / "artifacts" / "pg173-matched-budget-capacity-v1" / "160m_epoch4.pt"
DATASET_PATH = RESEARCH / "pg176_fourth_generator_ood_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg176_routed_multiseed_new_ood_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg176_routed_multiseed_new_ood_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg176_routed_multiseed_new_ood_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg176-routed-multiseed-new-ood-v1"
SEEDS = (17601, 17602, 17603)
GENERATOR_SEED = 17600
ROWS_PER_SOURCE = 250
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_LEN = 128
BASELINE_AGGREGATE = 2.51077335
PER_SPLIT_TOLERANCE = 0.005


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _Dataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.rows = rows
        self.stoi = stoi

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"ids": [self.stoi.get(token, self.stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]], "source": row.get("source", "eval")}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    sources = []
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        sources.append(item["source"])
    return {"ids": ids, "mask": ids.ne(0), "source": sources}


def _metrics(model: nn.Module, rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, float | int]:
    loader = DataLoader(_Dataset(rows, stoi), batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=_collate)
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
            total_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum").item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 8), "perplexity": round(math.exp(min(mean, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _load_model(checkpoint: dict[str, Any], device: torch.device) -> CausalTraceTransformer:
    config = checkpoint["config"]
    model = CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _fourth_generator_rows(vocab: set[str], prior_signatures: set[tuple[str, ...]]) -> list[dict[str, Any]]:
    rng = random.Random(GENERATOR_SEED)
    styles = ("govuk_like", "material_like", "primer_like", "dashboard_like", "dense_like", "minimal_like")
    syntax = ("bounded_template", "component_tree", "mixed_markup", "server_rendered")
    sinks = ("html_dom_sink", "query_ast_boundary", "xml_entity_parser", "html_attribute", "html_text", "dom_template", "sql_ast_boundary", "access_transition")
    methods = ("GET", "POST")
    placements = ("query", "form", "json", "body_field", "path")
    depths = (1, 2, 3)
    routes = ("api_surface", "comment_surface", "login_surface", "profile_surface", "search_surface", "upload_surface")
    js = ("progressive_enhancement", "client_router", "event_listener", "fetch_form", "filter_table_callback", "none")
    response_content = ("html", "json", "text", "xml", "unknown")
    response_status = ("2xx", "3xx", "4xx")
    response_shape = ("stable", "steady", "transition-v2", "visual-shift-v2", "policy-transition")
    tags = ("1-4", "5-16", "17+")
    forms = ("0", "1-2", "3+")
    inputs = ("0", "1-4", "5+")
    seen = set(prior_signatures)
    rows: list[dict[str, Any]] = []
    while len(rows) < 1000:
        style = rng.choice(styles)
        tokens = ["[BOS]", "[STEP]", "[SRC_HTML]", f"src.html.style={style}", f"src.html.syntax={rng.choice(syntax)}", f"src.html.tag_count={rng.choice(tags)}", f"src.html.form_count={rng.choice(forms)}", f"src.html.input_count={rng.choice(inputs)}", "src.html.script_count=1-4", f"src.html.form_method={rng.choice(methods)}", f"src.html.attribute={rng.choice(('method', 'name'))}", f"src.html.text_length_bucket=1-4", "[SRC_JAVASCRIPT]", f"src.javascript.shape={rng.choice(js)}", "[SRC_TRANSPORT]", f"src.transport.method={rng.choice(methods)}", f"src.transport.placement={rng.choice(placements)}", f"src.transport.encoding_depth={rng.choice(depths)}", f"src.transport.route_class={rng.choice(routes)}", "src.transport.route=loopback_allowlisted", "[IR]", "ir.surface.family_free=true", f"ir.surface.style={style}", f"ir.surface.sink_class={rng.choice(sinks)}", "ir.probe.shape=bounded_marker", f"ir.response.content_type={rng.choice(response_content)}", f"ir.response.status_class={rng.choice(response_status)}", f"ir.response.shape_class={rng.choice(response_shape)}", "ir.response.candidate_signal=false", "ir.response.transition_delta=unknown", "[OBS]", "obs.oracle=unknown_oracle", "[EOS]"]
        signature = tuple(tokens)
        if signature in seen:
            continue
        missing = [token for token in tokens if token not in vocab]
        if missing:
            raise RuntimeError(f"frozen vocabulary missing token: {missing[0]}")
        seen.add(signature)
        rows.append({"row_id": f"pg176-fourth-ood-{len(rows) + 1:05d}", "generator": "independent_hybrid_v4", "split": "ood", "tokens": tokens, "projection_sha256": _sha256_json(tokens)})
    return rows


def _aggregate(value: dict[str, Any], keys: tuple[str, ...]) -> float:
    return round(sum(value[key]["perplexity"] for key in keys) / len(keys), 8)


def _train(seed: int, checkpoint: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = _load_model(checkpoint, device)
    ordered = _route(train_rows, "interleave")
    loader = DataLoader(_Dataset(ordered, stoi), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=_collate)
    weights = {"pg163_base": 1.5, "pg168_slots": 1.0, "pg170_generator": 1.0, "pg172_generator": 1.0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
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
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="none").reshape(targets.shape)
            valid = targets.ne(0)
            batch_weights = torch.tensor([weights.get(source, 1.0) for source in batch["source"]], device=device, dtype=losses.dtype).unsqueeze(1)
            loss = (losses * valid * batch_weights).sum() / (valid * batch_weights).sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        valid_count = int(valid.sum().item())
        loss_sum += float(loss.detach().cpu()) * valid_count
        token_sum += valid_count
    keys = ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood")
    metrics = {key: _metrics(model, eval_rows[key], stoi, device) for key in (*keys, "fourth_generator_ood")}
    result = {"seed": seed, "train_loss": round(loss_sum / max(token_sum, 1), 8), "train_row_count": len(ordered), **metrics, "aggregate_existing": _aggregate(metrics, keys), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"seed_{seed}.pt"
    torch.save({"schema_version": "pg176-routed-multiseed-new-ood-v1", "seed": seed, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu")
    source_rows, _, source_hashes = _load_sources()
    train_rows = _make_train_rows(source_rows)
    # Build the prior projection set from all old train/OOD rows so the fourth
    # generator is a genuine projection holdout.
    prior_signatures = {tuple(row["tokens"]) for rows in source_rows.values() for row in rows}
    for path in (PG168_DATASET, PG170_DATASET, PG172_DATASET):
        data = json.loads(path.read_text(encoding="utf-8"))
        prior_signatures.update(tuple(row["tokens"]) for row in data["rows"])
    vocab = set(checkpoint["vocabulary"])
    fourth_rows = _fourth_generator_rows(vocab, prior_signatures)
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    p168 = json.loads(PG168_DATASET.read_text(encoding="utf-8"))
    p170 = json.loads(PG170_DATASET.read_text(encoding="utf-8"))
    p172 = json.loads(PG172_DATASET.read_text(encoding="utf-8"))
    eval_rows = {"base_holdout": [{"tokens": row["tokens"]} for row in base["base_holdout_rows"]], "typed_holdout": [{"tokens": row["tokens"]} for row in base["typed_holdout_rows"]], "pg168_ood": [{"tokens": row["tokens"]} for row in p168["rows"] if row["split"] == "ood"], "pg170_ood": [{"tokens": row["tokens"]} for row in p170["rows"] if row["split"] == "ood"], "pg172_ood": [{"tokens": row["tokens"]} for row in p172["rows"] if row["split"] == "ood"], "fourth_generator_ood": [{"tokens": row["tokens"]} for row in fourth_rows]}
    dataset = {"schema_version": "pg176-fourth-generator-ood-dataset-v1", "purpose": "fourth independent family-free Rule-IR OOD", "generator": "independent_hybrid_v4", "row_count": len(fourth_rows), "projection_overlap_prior": 0, "source_dataset_sha256": source_hashes, "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "memory_promotion_allowed": False}, "rows": fourth_rows}
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {key: _metrics(baseline_model, rows, stoi, device) for key, rows in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = [_train(seed, checkpoint, train_rows, eval_rows, stoi, device) for seed in SEEDS]
    keys = ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood")
    base_agg = _aggregate({key: baseline[key] for key in keys}, keys)
    for result in results:
        result["aggregate_existing"] = _aggregate(result, keys)
    gates = []
    for result in results:
        split_ok = all(result[key]["perplexity"] <= baseline[key]["perplexity"] * (1.0 + PER_SPLIT_TOLERANCE) for key in keys)
        gates.append({"seed": result["seed"], "existing_split_gate": split_ok, "existing_aggregate_gate": result["aggregate_existing"] < base_agg, "fourth_ood_gate": result["fourth_generator_ood"]["perplexity"] < baseline["fourth_generator_ood"]["perplexity"], "pass": split_ok and result["aggregate_existing"] < base_agg and result["fourth_generator_ood"]["perplexity"] < baseline["fourth_generator_ood"]["perplexity"]})
    report = {"schema_version": "pg176-routed-multiseed-new-ood-report-v1", "protocol_id": "pg-pk-176-routed-multiseed-new-ood-v1", "status": "completed_pg176_routed_multiseed_new_ood", "scope": {"claim": "multi-seed routed_low_lr stability and fourth-generator OOD diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"train_row_count": len(train_rows), "full_eval_counts": {key: len(rows) for key, rows in eval_rows.items()}, "fourth_generator_projection_overlap_prior": 0, "source_dataset_sha256": source_hashes}, "baseline": baseline, "baseline_existing_aggregate_ppl": base_agg, "seed_results": results, "gates": gates, "selection": {"all_seeds_pass": all(gate["pass"] for gate in gates), "promotion_allowed": all(gate["pass"] for gate in gates), "vulnerability_claim_allowed": False}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(START_CHECKPOINT.read_bytes()).hexdigest(), "dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-176-routed-multiseed-new-ood-v1", "schema_version": "pg176-routed-multiseed-new-ood-protocol-v1", "start_checkpoint": str(START_CHECKPOINT.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "seeds": list(SEEDS), "rows_per_source": ROWS_PER_SOURCE, "full_holdout": report["dataset"]["full_eval_counts"], "hard_gate": {"baseline_existing_aggregate_ppl": base_agg, "per_split_tolerance": PER_SPLIT_TOLERANCE, "fourth_ood_must_improve": True}, "promotion": {"training_artifact_promotion_allowed": report["selection"]["promotion_allowed"], "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-176 routed multi-seed/new OOD", "", f"- seeds: **{SEEDS}**", f"- fourth generator rows: **{len(fourth_rows)}**", f"- baseline existing aggregate: **{base_agg}**", f"- gates: **{gates}**", f"- all seeds pass: **{report['selection']['all_seeds_pass']}**", "", "该轮仍只验证抽象 Rule-IR 训练泛化，不产生漏洞标签。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "baseline_existing_aggregate_ppl": base_agg, "seed_gates": gates, "fourth_ood_baseline": baseline["fourth_generator_ood"]["perplexity"], "fourth_ood_seed_ppl": [result["fourth_generator_ood"]["perplexity"] for result in results], "all_seeds_pass": report["selection"]["all_seeds_pass"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
