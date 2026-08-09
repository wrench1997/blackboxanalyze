"""PG-147: train several causal Transformer capacities on abstract traces.

This is a representation-learning experiment, not a scanner and not a claim
that perplexity equals vulnerability capability.  The corpus combines
existing bounded Rule-IR sequences with procedurally generated, high-entropy
abstract trajectories.  No raw payload, response body, target identity or
external URL is generated.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg147-model-capacity-sweep-v1"
REPORT = RESEARCH / "pg147_model_capacity_sweep_report_v1.json"
DATASET = RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json"
PROTOCOL = RESEARCH / "pg147_model_capacity_sweep_protocol_v1.json"
SEED = 14701
GENERATED_TARGET = 12000
MAX_LEN = 128


SPECIAL = ("[PAD]", "[UNK]")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "pretrain_sequences" in raw:
        return [row for row in raw["pretrain_sequences"] if isinstance(row.get("tokens"), list)]
    if "rows" in raw:
        return [row for row in raw["rows"] if isinstance(row.get("tokens"), list)]
    return []


def _generated_sequence(rng: random.Random, index: int) -> list[str]:
    styles = ("govuk_like", "material_like", "primer_like", "dashboard_like", "minimal_like", "dense_like")
    html_syntax = ("bounded_template", "server_rendered", "component_tree", "mixed_markup")
    js_shapes = ("progressive_enhancement", "fetch_form", "event_listener", "client_router", "none")
    js_apis = ("fetch", "xhr", "form_submit", "history_api", "none")
    methods = ("GET", "POST")
    placements = ("query", "form", "json", "path", "fragment")
    routes = ("login_surface", "search_surface", "profile_surface", "upload_surface", "comment_surface", "api_surface")
    sinks = ("html_text", "html_attribute", "dom_template", "sql_ast_boundary", "xml_entity_parser", "access_transition")
    failures = ("no_surface_delta", "status_only_delta", "shape_delta", "redirect_delta", "timeout_signature", "parse_error_signature", "auth_boundary")
    beliefs = ("prior", "candidate", "negative_control", "uncertain", "recovery")
    statuses = ("2xx", "3xx", "4xx", "5xx", "transport_error")
    content_types = ("html", "json", "text", "xml", "unknown")
    weights = ("0.5", "1.0", "1.25", "1.5", "2.0")
    steps = rng.randint(1, 4)
    tokens = ["[BOS]"]
    for step in range(steps):
        method = rng.choice(methods)
        style = rng.choice(styles)
        js_shape = rng.choice(js_shapes)
        tokens.extend(
            [
                "[STEP]",
                "[SRC_HTML]",
                f"src.html.style={style}",
                f"src.html.syntax={rng.choice(html_syntax)}",
                f"src.html.tag_count={rng.choice(('1-4', '5-16', '17+'))}",
                f"src.html.form_count={rng.choice(('0', '1-2', '3+'))}",
                f"src.html.input_count={rng.choice(('0', '1-4', '5+'))}",
                "[SRC_JAVASCRIPT]",
                f"src.javascript.shape={js_shape}",
                f"src.javascript.api={rng.choice(js_apis)}",
                f"src.javascript.script_count={rng.choice(('0', '1-4', '5+'))}",
                "[SRC_TRANSPORT]",
                f"src.transport.method={method}",
                f"src.transport.placement={rng.choice(placements)}",
                f"src.transport.route_class={rng.choice(routes)}",
                "[IR]",
                f"ir.surface.style={style}",
                f"ir.surface.sink_class={rng.choice(sinks)}",
                f"ir.failure.kind={rng.choice(failures)}",
                f"ir.belief.phase={rng.choice(beliefs)}",
                f"ir.response.status_class={rng.choice(statuses)}",
                f"ir.response.content_type={rng.choice(content_types)}",
                f"ir_weight={rng.choice(weights)}",
                "[OBS]",
                f"obs.method_seen={method}",
                f"obs.step_progress=step_{step + 1}_of_{steps}",
                "obs.oracle=unknown_oracle",
            ]
        )
    tokens.append("[EOS]")
    return tokens[:MAX_LEN]


def _build_corpus() -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_paths = [
        RESEARCH / "pg136_causal_token_lm_dataset_v1.json",
        RESEARCH / "pg145_local_multisurface_model_dataset_v1.json",
        RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json",
    ]
    rows = _load_rows(base_paths[0]) + _load_rows(base_paths[1]) + _load_rows(base_paths[2])
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sequence = tuple(str(token) for token in row["tokens"][:MAX_LEN])
        if sequence:
            unique.setdefault(sequence, {"row_id": f"base-{index}", "tokens": list(sequence), "source": "existing"})
    rng = random.Random(SEED)
    generated = 0
    attempts = 0
    while generated < GENERATED_TARGET and attempts < GENERATED_TARGET * 20:
        attempts += 1
        sequence = tuple(_generated_sequence(rng, generated))
        if sequence in unique:
            continue
        unique[sequence] = {"row_id": f"generated-{generated:06d}", "tokens": list(sequence), "source": "procedural_abstract_trace"}
        generated += 1
    corpus = list(unique.values())
    rng.shuffle(corpus)
    train_count = int(len(corpus) * 0.8)
    dev_count = int(len(corpus) * 0.1)
    splits = {"train": train_count, "dev": train_count + dev_count}
    for index, row in enumerate(corpus):
        row["split"] = "train" if index < splits["train"] else "dev" if index < splits["dev"] else "holdout"
    return corpus, {"base_count": len(rows), "unique_count": len(corpus), "generated_count": generated, "duplicate_count": len(rows) + generated - len(corpus)}


class _TraceDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.rows = rows
        self.stoi = stoi

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        ids = [self.stoi.get(token, self.stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]]
        return {"ids": ids, "row_id": row["row_id"]}


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {"ids": ids, "mask": ids.ne(0)}


def _metrics(model: CausalTraceTransformer, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": round(mean_loss, 8), "perplexity": round(math.exp(min(mean_loss, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _train_variant(name: str, config: dict[str, int], train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    model = CausalTraceTransformer(len(stoi), d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], max_len=MAX_LEN).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    batch_size = config["batch_size"]
    train_loader = DataLoader(_TraceDataset(train_rows, stoi), batch_size=batch_size, shuffle=True, collate_fn=_collate, drop_last=False)
    dev_loader = DataLoader(_TraceDataset(dev_rows, stoi), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    holdout_loader = DataLoader(_TraceDataset(holdout_rows, stoi), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        token_sum = 0
        for batch in train_loader:
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
            valid_tokens = int(targets.ne(0).sum().item())
            loss_sum += float(loss.item()) * valid_tokens
            token_sum += valid_tokens
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(token_sum, 1), 8), "dev": _metrics(model, dev_loader, device)})
    train_metrics = _metrics(model, train_loader, device)
    dev_metrics = _metrics(model, dev_loader, device)
    holdout_metrics = _metrics(model, holdout_loader, device)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg147-causal-transformer-v1", "variant": name, "config": config, "vocabulary": sorted(stoi, key=stoi.get), "model_state_dict": model.state_dict()}, checkpoint)
    return {"variant": name, "config": config, "parameter_count": params, "checkpoint": str(checkpoint.relative_to(ROOT)), "train": train_metrics, "dev": dev_metrics, "holdout": holdout_metrics, "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    corpus, corpus_stats = _build_corpus()
    train_rows = [row for row in corpus if row["split"] == "train"]
    dev_rows = [row for row in corpus if row["split"] == "dev"]
    holdout_rows = [row for row in corpus if row["split"] == "holdout"]
    tokens = sorted({token for row in train_rows for token in row["tokens"]})
    itos = list(SPECIAL) + [token for token in tokens if token not in SPECIAL]
    stoi = {token: index for index, token in enumerate(itos)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {
        "tiny_transformer": {"d_model": 96, "nhead": 4, "layers": 2, "batch_size": 128, "epochs": 4, "lr": 3e-4},
        "medium_transformer": {"d_model": 256, "nhead": 8, "layers": 4, "batch_size": 96, "epochs": 4, "lr": 2e-4},
        "large_transformer": {"d_model": 512, "nhead": 8, "layers": 6, "batch_size": 48, "epochs": 4, "lr": 1.5e-4},
        "xl_transformer": {"d_model": 768, "nhead": 12, "layers": 8, "batch_size": 24, "epochs": 3, "lr": 1e-4},
    }
    results = [_train_variant(name, config, train_rows, dev_rows, holdout_rows, stoi, device) for name, config in variants.items()]
    report = {
        "protocol_id": "pg-pk-147-model-capacity-sweep-v1",
        "schema_version": "pg147-model-capacity-sweep-report-v1",
        "status": "completed_pg147_model_capacity_sweep",
        "device": str(device),
        "seed": SEED,
        "corpus": {**corpus_stats, "train_count": len(train_rows), "dev_count": len(dev_rows), "holdout_count": len(holdout_rows), "vocabulary_size": len(stoi), "max_length": MAX_LEN, "source_files": ["research/pg136_causal_token_lm_dataset_v1.json", "research/pg145_local_multisurface_model_dataset_v1.json", "research/pg146_public_lab_replay_model_dataset_v1.json"]},
        "variants": results,
        "objective": "capacity_and_next_token_representation_comparison",
        "raw_payload_or_response_in_corpus": False,
        "external_network_targets": False,
        "capability_claim_allowed": False,
        "long_term_memory_promotion_allowed": False,
        "report_sha256": "",
    }
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg147-model-capacity-sweep-dataset-v1", "corpus_stats": corpus_stats, "vocabulary": itos, "splits": {"train": train_rows, "dev": dev_rows, "holdout": holdout_rows}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-147-model-capacity-sweep-v1", "schema_version": "pg147-model-capacity-sweep-protocol-v1", "objective": report["objective"], "variants": variants, "data_policy": {"procedural_abstract_trace_generation": True, "raw_payloads": False, "raw_responses": False, "labels_in_next_token_input": False}, "promotion": {"capability_claim_allowed": False, "long_term_memory_promotion_allowed": False}}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "corpus": report["corpus"], "variants": [{"variant": row["variant"], "parameter_count": row["parameter_count"], "dev_perplexity": row["dev"]["perplexity"], "holdout_perplexity": row["holdout"]["perplexity"], "holdout_next_token_accuracy": row["holdout"]["next_token_accuracy"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
