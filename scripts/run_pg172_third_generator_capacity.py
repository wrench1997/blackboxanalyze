"""PG-172: third independent Rule-IR generator and 101M/160M capacity audit.

Both capacity variants start from scratch on the same 1:1 old/new corpus so
the comparison changes model capacity, not checkpoint initialization.  The
third generator adds independent HTML shape slots and reserves three sink
slots for OOD.  It is abstract next-token training only.
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
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PRIOR_DATASETS = [RESEARCH / "pg168_discriminative_slot_dataset_v1.json", RESEARCH / "pg170_cross_generator_dataset_v1.json"]
CHECKPOINT_REFERENCE = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
DATASET_PATH = RESEARCH / "pg172_third_generator_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg172_third_generator_capacity_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg172_third_generator_capacity_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg172_third_generator_capacity_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg172-third-generator-capacity-v1"
SEED = 17201
MAX_LEN = 128
TRAIN_COUNT = 4000
DEV_COUNT = 1000
OOD_COUNT = 1000
TRAIN_SINKS = ("html_dom_sink", "query_ast_boundary", "xml_entity_parser", "html_attribute", "html_text")
OOD_SINKS = ("dom_template", "sql_ast_boundary", "access_transition")
STYLES = ("govuk_like", "material_like", "primer_like", "dashboard_like", "dense_like", "minimal_like")
JS_SHAPES = ("progressive_enhancement", "client_router", "event_listener", "fetch_form", "filter_table_callback", "none")
METHODS = ("GET", "POST")
PLACEMENTS = ("query", "form", "json", "body_field", "path")
DEPTHS = (1, 2, 3)
ROUTES = ("api_surface", "comment_surface", "login_surface", "profile_surface", "search_surface", "upload_surface")
CONTENT_TYPES = ("html", "json", "text", "xml", "unknown")
STATUS = ("2xx", "3xx", "4xx")
SHAPES = ("stable", "steady", "transition-v2", "visual-shift-v2", "policy-transition")
SYNTAXES = ("bounded_template", "component_tree", "mixed_markup", "server_rendered")
HTML_TAGS = ("form",)
TAG_COUNTS = ("1-4", "5-16", "17+")
FORM_COUNTS = ("0", "1-2", "3+")
INPUT_COUNTS = ("0", "1-4", "5+")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tokens(rng: random.Random, sink: str) -> list[str]:
    return [
        "[BOS]", "[STEP]", "[SRC_HTML]", f"src.html.style={rng.choice(STYLES)}", f"src.html.syntax={rng.choice(SYNTAXES)}",
        f"src.html.tag={rng.choice(HTML_TAGS)}", f"src.html.tag_count={rng.choice(TAG_COUNTS)}", f"src.html.form_count={rng.choice(FORM_COUNTS)}",
        f"src.html.input_count={rng.choice(INPUT_COUNTS)}", "src.html.script_count=1-4", f"src.html.form_method={rng.choice(METHODS)}",
        f"src.html.attribute={rng.choice(('method', 'name'))}", "[SRC_JAVASCRIPT]", f"src.javascript.shape={rng.choice(JS_SHAPES)}", "[SRC_TRANSPORT]",
        f"src.transport.method={rng.choice(METHODS)}", f"src.transport.placement={rng.choice(PLACEMENTS)}", f"src.transport.encoding_depth={rng.choice(DEPTHS)}",
        f"src.transport.route_class={rng.choice(ROUTES)}", "src.transport.route=loopback_allowlisted", "[IR]", "ir.surface.family_free=true",
        f"ir.surface.sink_class={sink}", "ir.probe.shape=bounded_marker", f"ir.response.content_type={rng.choice(CONTENT_TYPES)}",
        f"ir.response.status_class={rng.choice(STATUS)}", f"ir.response.shape_class={rng.choice(SHAPES)}", "ir.response.candidate_signal=false",
        "ir.response.transition_delta=unknown", "[OBS]", "obs.oracle=unknown_oracle", "[EOS]",
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
            if attempts > target * 300:
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
            rows.append({"row_id": f"pg172-{split}-{counts[split]:05d}", "generator": "independent_html_shape_v3", "split": split, "tokens": tokens, "projection_sha256": _sha256_json(tokens)})
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


def _make_model(vocab_size: int, config: dict[str, int], device: torch.device) -> CausalTraceTransformer:
    return CausalTraceTransformer(vocab_size, d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], max_len=MAX_LEN).to(device)


def _train(name: str, config: dict[str, int], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = _make_model(len(stoi), config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loader = DataLoader(_Dataset(rows, stoi), batch_size=8, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.01)
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
    result = {"strategy": name, "config": config, "parameter_count": parameter_count, "train_row_count": len(rows), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "third_generator_dev": _metrics(model, eval_rows["third_generator_dev"], stoi, device), "third_generator_ood": _metrics(model, eval_rows["third_generator_ood"], stoi, device), "prior_generator_ood": _metrics(model, eval_rows["prior_generator_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg172-third-generator-capacity-v1", "strategy": name, "config": config, "vocabulary": list(stoi), "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    reference = torch.load(CHECKPOINT_REFERENCE, map_location="cpu")
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    prior_rows: list[dict[str, Any]] = []
    prior_hashes = []
    for path in PRIOR_DATASETS:
        data = json.loads(path.read_text(encoding="utf-8"))
        prior_rows.extend(data["rows"])
        prior_hashes.append(data.get("dataset_sha256"))
    vocab = set(reference["vocabulary"])
    rows, counts = _generate(vocab, {tuple(row["tokens"]) for row in prior_rows})
    third_train = [{"tokens": row["tokens"]} for row in rows if row["split"] == "train"]
    third_dev = [{"tokens": row["tokens"]} for row in rows if row["split"] == "dev"]
    third_ood = [{"tokens": row["tokens"]} for row in rows if row["split"] == "ood"]
    prior_ood = [{"tokens": row["tokens"]} for row in prior_rows if row.get("split") == "ood"]
    if {tuple(row["tokens"]) for row in third_dev} & {tuple(row["tokens"]) for row in third_ood}:
        raise RuntimeError("third generator dev/OOD projection collision")
    if {tuple(row["tokens"]) for row in third_ood} & {tuple(row["tokens"]) for row in prior_rows}:
        raise RuntimeError("third generator/prior projection collision")
    base_all = [{"tokens": row["tokens"]} for row in base["train_rows"]]
    selected_base = random.Random(SEED).sample(base_all, len(third_train))
    train_rows = selected_base + third_train
    random.Random(SEED).shuffle(train_rows)
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "third_generator_dev": third_dev, "third_generator_ood": third_ood, "prior_generator_ood": prior_ood}
    dataset = {"schema_version": "pg172-third-generator-dataset-v1", "purpose": "third independent family-free Rule-IR generator with HTML shape slots", "generator": {"id": "independent_html_shape_v3", "train_sink_classes": list(TRAIN_SINKS), "ood_sink_classes": list(OOD_SINKS), "new_html_slots": ["src.html.tag", "src.html.tag_count", "src.html.form_count", "src.html.input_count", "src.html.script_count", "src.html.form_method", "src.html.attribute"]}, "counts": counts, "prior_projection_overlap": 0, "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "memory_promotion_allowed": False}, "rows": rows}
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    stoi = {token: index for index, token in enumerate(reference["vocabulary"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = {"capacity_101m_scratch": {"d_model": 1024, "nhead": 16, "layers": 8}, "capacity_160m_scratch": {"d_model": 1152, "nhead": 18, "layers": 10}}
    results: dict[str, Any] = {}
    for name, config in configs.items():
        results[name] = _train(name, config, train_rows, eval_rows, stoi, device)
    report = {"schema_version": "pg172-third-generator-capacity-report-v1", "protocol_id": "pg-pk-172-third-generator-capacity-v1", "status": "completed_pg172_third_generator_capacity", "scope": {"claim": "third-generator OOD and 101M versus 160M scratch capacity diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"base_replay_count": len(selected_base), "third_train_count": len(third_train), "third_dev_count": len(third_dev), "third_ood_count": len(third_ood), "prior_ood_count": len(prior_ood), "projection_overlap_third_prior": 0, "projection_overlap_third_dev_ood": 0}, "variants": results, "capacity_comparison": {"same_training_rows": True, "same_optimizer": True, "same_seed": True, "initialization": "from_scratch_for_both", "causal_capacity_claim_allowed": True}, "interpretation": {"third_generator_ood_isolated": True, "vulnerability_claim_allowed": False, "promotion_allowed": False, "next_token_lm_only": True}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "reference_checkpoint_sha256": hashlib.sha256(CHECKPOINT_REFERENCE.read_bytes()).hexdigest(), "base_dataset_sha256": base.get("dataset_sha256"), "prior_dataset_sha256": prior_hashes, "dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-172-third-generator-capacity-v1", "schema_version": "pg172-third-generator-capacity-protocol-v1", "reference_checkpoint": str(CHECKPOINT_REFERENCE.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "prior_datasets": [str(path.relative_to(ROOT)) for path in PRIOR_DATASETS], "configs": configs, "optimizer": {"name": "AdamW", "lr": 7e-5, "epochs": 1}, "capacity_comparison": report["capacity_comparison"], "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    small = results["capacity_101m_scratch"]
    large = results["capacity_160m_scratch"]
    MARKDOWN_PATH.write_text("\n".join(["# PG-172 third generator and capacity", "", f"- third generator train/dev/OOD: **{len(third_train)} / {len(third_dev)} / {len(third_ood)}**", f"- 101M OOD/base/typed PPL: **{small['third_generator_ood']['perplexity']} / {small['base_holdout']['perplexity']} / {small['typed_holdout']['perplexity']}**", f"- 160M OOD/base/typed PPL: **{large['third_generator_ood']['perplexity']} / {large['base_holdout']['perplexity']} / {large['typed_holdout']['perplexity']}**", "", "两种容量从零开始在相同数据上训练；结果只用于容量因果诊断，不产生漏洞标签。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "counts": counts, "capacity": {name: {"parameters": result["parameter_count"], "third_ood_ppl": result["third_generator_ood"]["perplexity"], "base_ppl": result["base_holdout"]["perplexity"], "typed_ppl": result["typed_holdout"]["perplexity"]} for name, result in results.items()}, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
