"""PG-177: expand abstract Rule-IR data and compare 160M/200M capacity.

This is a next-token language-model experiment over family-free, bounded Rule-IR
tokens.  It deliberately keeps raw payloads, response bodies, evaluator labels,
and vulnerability-family labels out of model inputs.  The new generators are
projection-disjoint from all earlier PG163/168/170/172/176 data.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg173_matched_budget_capacity import _load_sources  # noqa: E402


RESEARCH = ROOT / "research"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PG168_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
PG170_DATASET = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PG172_DATASET = RESEARCH / "pg172_third_generator_dataset_v1.json"
PG176_DATASET = RESEARCH / "pg176_fourth_generator_ood_dataset_v1.json"
START_CHECKPOINT = ROOT / "artifacts" / "pg176-routed-multiseed-new-ood-v1" / "seed_17601.pt"
DATASET_PATH = RESEARCH / "pg177_data_capacity_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg177_data_capacity_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg177_data_capacity_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg177_data_capacity_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg177-data-capacity-v1"

SEEDS = (17701, 17702)
GENERATOR_SEED = 17700
OLD_ROWS_PER_SOURCE = 600
NEW_TRAIN_PER_GENERATOR = 1200
NEW_DEV_PER_GENERATOR = 300
NEW_OOD_PER_GENERATOR = 300
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_LEN = 128
PER_SPLIT_TOLERANCE = 0.005

OLD_SOURCE_ORDER = ("pg163_base", "pg168_slots", "pg170_generator", "pg172_generator")
NEW_GENERATORS = (
    "surface_permutation_v7",
    "failure_recovery_v8",
    "transport_matrix_v9",
    "semantic_delta_v10",
)

CAPACITY_CONFIGS = {
    "160m_scratch": {"d_model": 1152, "nhead": 18, "layers": 10},
    "200m_scratch": {"d_model": 1280, "nhead": 20, "layers": 10},
}


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
    sources: list[str] = []
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


def _new_generator_tokens(rng: random.Random, generator: str, split: str) -> list[str]:
    styles = ("govuk_like", "material_like", "primer_like", "dashboard_like", "dense_like", "minimal_like")
    syntaxes = ("bounded_template", "component_tree", "mixed_markup", "server_rendered")
    tags = ("1-4", "5-16", "17+")
    forms = ("0", "1-2", "3+")
    inputs = ("0", "1-4", "5+")
    methods = ("GET", "POST")
    attributes = ("method", "name")
    js_shapes = ("progressive_enhancement", "client_router", "event_listener", "fetch_form", "filter_table_callback", "responsive_component_state", "web_component_event", "none")
    js_apis = ("fetch", "form_submit", "history_api", "none", "xhr")
    routes = ("api_surface", "comment_surface", "login_surface", "profile_surface", "search_surface", "upload_surface")
    placements = ("query", "form", "json", "body_field", "path", "fragment")
    depths = (1, 2, 3)
    contents = ("html", "json", "text", "xml", "unknown")
    statuses = ("2xx", "3xx", "4xx", "5xx", "transport_error")
    shapes = ("stable", "steady", "transition-v2", "visual-shift-v2", "policy-transition", "steady-v2", "transition-v3", "decision-v5", "layout-decoy")
    sinks = ("html_dom_sink", "query_ast_boundary", "xml_entity_parser", "html_attribute", "html_text", "dom_template", "sql_ast_boundary", "access_transition")
    failures = ("matched_negative_control", "surface_delta", "typed_effect")
    failure_kinds = ("auth_boundary", "candidate_without_typed_effect", "no_surface_delta", "oracle_unavailable", "parse_error_signature", "redirect_delta", "shape_delta", "status_only_delta", "timeout_signature", "typed_positive")
    phases = ("candidate", "negative_control", "prior", "recovery", "uncertain")
    effects = ("0-0.2", "0.2-0.4", "0.4-0.6")
    input_effects = ("0-0.2", "0.2-0.4")
    progress = ("step_1_of_4", "step_1_of_6", "step_2_of_4", "step_2_of_6")
    step_progress = ("step_1_of_1", "step_1_of_2", "step_1_of_3", "step_1_of_4", "step_2_of_2", "step_2_of_3", "step_2_of_4", "step_3_of_3", "step_3_of_4", "step_4_of_4")
    method = rng.choice(methods)
    other_method = "POST" if method == "GET" else "GET"
    style = rng.choice(styles)
    sink = rng.choice(sinks)
    common_html = [
        "[SRC_HTML]",
        f"src.html.style={style}",
        f"src.html.syntax={rng.choice(syntaxes)}",
        "src.html.tag=form",
        f"src.html.tag_count={rng.choice(tags)}",
        f"src.html.form_count={rng.choice(forms)}",
        f"src.html.input_count={rng.choice(inputs)}",
        "src.html.script_count=1-4",
        f"src.html.form_method={method}",
        f"src.html.attribute={rng.choice(attributes)}",
        "src.html.text_length_bucket=1-4",
    ]
    common_js = [
        "[SRC_JAVASCRIPT]",
        f"src.javascript.shape={rng.choice(js_shapes)}",
        f"src.javascript.api={rng.choice(js_apis)}",
        f"src.javascript.script_count={rng.choice(('0', '1-4', '5+'))}",
        "src.javascript.keyword=const",
        "src.javascript.length_bucket=17+",
    ]
    common_transport = [
        "[SRC_TRANSPORT]",
        f"src.transport.method={method}",
        f"src.transport.placement={rng.choice(placements)}",
        f"src.transport.encoding_depth={rng.choice(depths)}",
        f"src.transport.route_class={rng.choice(routes)}",
        "src.transport.route=loopback_allowlisted",
        "src.transport.form_field_count=1-4",
        "src.transport.route_template=hash_present",
    ]
    common_ir = [
        "[IR]",
        "ir.surface.family_free=true",
        "ir.surface.modalities=html+javascript+transport",
        f"ir.surface.style={style}",
        f"ir.surface.sink_class={sink}",
        "ir.probe.shape=bounded_marker",
        f"ir.response.content_type={rng.choice(contents)}",
        f"ir.response.status_class={rng.choice(statuses)}",
        f"ir.response.shape_class={rng.choice(shapes)}",
        f"ir.response.transition_delta={rng.choice(('none', 'location', 'scope', 'unknown'))}",
        f"ir.response.candidate_signal={rng.choice(('false', 'true'))}",
    ]
    common_obs = [
        "[OBS]",
        f"obs.oracle=unknown_oracle",
        f"obs.method_seen={method}",
        f"obs.status_class={rng.choice(('2xx', '4xx'))}",
        f"obs.body_length={rng.choice(('1-255', '256-4095', '4096-65535'))}",
        f"obs.step_progress={rng.choice(step_progress)}",
        f"obs.step_index={rng.choice(('1', '2', '3', '4'))}",
        "obs.failure.transport=false",
    ]
    if generator == "surface_permutation_v7":
        body = ["[STEP]"] + common_html + common_transport + common_js + common_ir + ["ir.transport.methods_seen=GET+POST"] + common_obs
    elif generator == "failure_recovery_v8":
        body = ["[RESET]", "[STEP]", "[BELIEF]", f"belief.effect={rng.choice(effects)}", f"belief.input_only={rng.choice(input_effects)}", f"belief.no_effect={rng.choice(effects)}", "belief.unknown=0.2-0.4"] + common_transport + common_html + ["[IR]", "ir.surface.family_free=true", f"ir.belief.phase={rng.choice(phases)}", f"ir.failure.failed_gate={rng.choice(failures)}", f"ir.failure.kind={rng.choice(failure_kinds)}", f"ir.failure.recovery_phase={rng.choice(('failure_adjusted', 'forward_baseline'))}", "ir.failure.weight=1.0", "ir.probe.remaining_budget=1-4"] + common_ir[2:] + common_obs + ["ir.trajectory.progress=" + rng.choice(progress)]
    elif generator == "transport_matrix_v9":
        body = ["[STEP]"] + common_transport + ["transport.channel=loopback", f"transport.method={method}", f"transport.method={other_method}"] + common_js + common_html + ["[IR]", "ir.transport.methods_seen=GET+POST", "ir.surface.family_free=true", "ir.surface.modalities=html+javascript+transport", f"ir.surface.sink_class={sink}", f"ir.response.location_changed={rng.choice(('false', 'true'))}", f"ir.response.metadata_changed={rng.choice(('false', 'true'))}", f"ir.response.authorization_changed={rng.choice(('false', 'true'))}", "ir.response.authorization_status=unknown", f"ir.response.policy_header_changed={rng.choice(('false', 'true'))}"] + common_obs
    else:
        body = ["[STEP]"] + common_js + common_html + common_transport + ["[IR]", "ir.surface.family_free=true", f"ir.surface.style={style}", f"ir.surface.sink_class={sink}", "ir.probe.shape=bounded_marker", f"ir.response.effect={rng.choice(('candidate', 'no_effect', 'unknown'))}", f"ir.response.shape_changed={rng.choice(('false', 'true'))}", f"ir.response.shape_class={rng.choice(shapes)}", f"ir.failure.failed_gate={rng.choice(failures)}", f"ir.failure.kind={rng.choice(failure_kinds)}", "ir.failure.recovery_phase=forward_baseline", "ir.failure.weight=1.0"] + common_obs + ["[BELIEF]", f"belief.effect={rng.choice(effects)}", f"belief.input_only={rng.choice(input_effects)}", f"belief.no_effect={rng.choice(effects)}"]
    return ["[BOS]"] + body + ["[EOS]"]


def _generate_new_rows(vocab: set[str], prior_signatures: set[tuple[str, ...]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    rng = random.Random(GENERATOR_SEED)
    seen = set(prior_signatures)
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for generator_index, generator in enumerate(NEW_GENERATORS):
        counts[generator] = {}
        for split, target in (("train", NEW_TRAIN_PER_GENERATOR), ("dev", NEW_DEV_PER_GENERATOR), ("ood", NEW_OOD_PER_GENERATOR)):
            count = 0
            attempts = 0
            while count < target:
                attempts += 1
                if attempts > target * 500:
                    raise RuntimeError(f"unable to generate collision-free {generator}/{split}")
                local = random.Random(rng.randrange(1 << 30) + generator_index * 100003 + count)
                tokens = _new_generator_tokens(local, generator, split)
                signature = tuple(tokens)
                if signature in seen:
                    continue
                missing = [token for token in tokens if token not in vocab]
                if missing:
                    raise RuntimeError(f"frozen vocabulary missing token: {missing[0]}")
                seen.add(signature)
                count += 1
                rows.append({"row_id": f"pg177-{generator}-{split}-{count:05d}", "generator": generator, "split": split, "tokens": tokens, "projection_sha256": _sha256_json(tokens)})
            counts[generator][split] = count
    return rows, counts


def _prior_signatures() -> set[tuple[str, ...]]:
    prior: set[tuple[str, ...]] = set()
    for path in (BASE_DATASET, PG168_DATASET, PG170_DATASET, PG172_DATASET, PG176_DATASET):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = list(data.get("rows") or []) + list(data.get("train_rows") or []) + list(data.get("base_holdout_rows") or []) + list(data.get("typed_holdout_rows") or [])
        prior.update(tuple(row["tokens"]) for row in rows if isinstance(row, dict) and "tokens" in row)
    return prior


def _load_model(checkpoint: dict[str, Any], device: torch.device, config: dict[str, int] | None = None) -> CausalTraceTransformer:
    selected = config or {key: int(value) for key, value in checkpoint["config"].items() if key in {"d_model", "nhead", "layers"}}
    return CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=selected["d_model"], nhead=selected["nhead"], layers=selected["layers"], max_len=MAX_LEN).to(device)


def _route(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = list(OLD_SOURCE_ORDER) + [f"pg177_{generator}" for generator in NEW_GENERATORS]
    groups = {source: [row for row in rows if row.get("source") == source] for source in order}
    cursors = {source: 0 for source in order}
    routed: list[dict[str, Any]] = []
    while any(cursors[source] < len(groups[source]) for source in order):
        for source in order:
            group = groups[source]
            if cursors[source] >= len(group):
                continue
            take = min(TRAIN_BATCH_SIZE, len(group) - cursors[source])
            routed.extend(group[cursors[source] : cursors[source] + take])
            cursors[source] += take
    return routed


def _aggregate(metrics: dict[str, Any], keys: tuple[str, ...]) -> float:
    return round(sum(metrics[key]["perplexity"] for key in keys) / len(keys), 8)


def _prepare_data(checkpoint: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, str]]:
    source_rows, _, source_hashes = _load_sources()
    prior = _prior_signatures()
    new_rows, new_counts = _generate_new_rows(set(checkpoint["vocabulary"]), prior)
    train_rows: list[dict[str, Any]] = []
    for offset, source in enumerate(OLD_SOURCE_ORDER):
        pool = source_rows[source]
        if len(pool) < OLD_ROWS_PER_SOURCE:
            raise RuntimeError(f"source {source} has only {len(pool)} rows")
        train_rows.extend(random.Random(GENERATOR_SEED + offset).sample(pool, OLD_ROWS_PER_SOURCE))
    for generator in NEW_GENERATORS:
        train_rows.extend({**row, "source": f"pg177_{generator}"} for row in new_rows if row["generator"] == generator and row["split"] == "train")
    if len(train_rows) != OLD_ROWS_PER_SOURCE * len(OLD_SOURCE_ORDER) + NEW_TRAIN_PER_GENERATOR * len(NEW_GENERATORS):
        raise RuntimeError("unexpected PG177 training row count")
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    p168 = json.loads(PG168_DATASET.read_text(encoding="utf-8"))
    p170 = json.loads(PG170_DATASET.read_text(encoding="utf-8"))
    p172 = json.loads(PG172_DATASET.read_text(encoding="utf-8"))
    p176 = json.loads(PG176_DATASET.read_text(encoding="utf-8"))
    eval_rows: dict[str, list[dict[str, Any]]] = {
        "base_holdout": [{"tokens": row["tokens"]} for row in base["base_holdout_rows"]],
        "typed_holdout": [{"tokens": row["tokens"]} for row in base["typed_holdout_rows"]],
        "pg168_ood": [{"tokens": row["tokens"]} for row in p168["rows"] if row["split"] == "ood"],
        "pg170_ood": [{"tokens": row["tokens"]} for row in p170["rows"] if row["split"] == "ood"],
        "pg172_ood": [{"tokens": row["tokens"]} for row in p172["rows"] if row["split"] == "ood"],
        "pg176_ood": [{"tokens": row["tokens"]} for row in p176["rows"] if row["split"] == "ood"],
    }
    for generator in NEW_GENERATORS:
        eval_rows[f"pg177_{generator}_dev"] = [{"tokens": row["tokens"]} for row in new_rows if row["generator"] == generator and row["split"] == "dev"]
        eval_rows[f"pg177_{generator}_ood"] = [{"tokens": row["tokens"]} for row in new_rows if row["generator"] == generator and row["split"] == "ood"]
    old_keys = ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood", "pg176_ood")
    new_keys = tuple(f"pg177_{generator}_ood" for generator in NEW_GENERATORS)
    dataset = {
        "schema_version": "pg177-data-capacity-dataset-v1",
        "purpose": "expanded family-free Rule-IR corpus with four independent generator surfaces",
        "generator_ids": list(NEW_GENERATORS),
        "counts": {"old_rows_per_source": OLD_ROWS_PER_SOURCE, "new_train_per_generator": NEW_TRAIN_PER_GENERATOR, "new_dev_per_generator": NEW_DEV_PER_GENERATOR, "new_ood_per_generator": NEW_OOD_PER_GENERATOR, "train_row_count": len(train_rows), "new_row_count": len(new_rows)},
        "new_generator_counts": new_counts,
        "full_eval_counts": {key: len(rows) for key, rows in eval_rows.items()},
        "projection_overlap_prior": 0,
        "source_dataset_sha256": source_hashes,
        "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "memory_promotion_allowed": False},
        "training_source_order": list(OLD_SOURCE_ORDER) + [f"pg177_{generator}" for generator in NEW_GENERATORS],
        "holdout_keys": {"existing": list(old_keys), "new_ood": list(new_keys)},
        "rows": new_rows,
    }
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    return train_rows, eval_rows, dataset, source_hashes


def _train_one(
    variant: str,
    seed: int,
    checkpoint: dict[str, Any],
    config: dict[str, int],
    train_rows: list[dict[str, Any]],
    eval_rows: dict[str, list[dict[str, Any]]],
    stoi: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if variant == "160m_continued":
        model = _load_model(checkpoint, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        learning_rate = 5e-6
        weight_by_source = {source: 1.5 for source in OLD_SOURCE_ORDER}
    else:
        model = CausalTraceTransformer(len(stoi), d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], max_len=MAX_LEN).to(device)
        learning_rate = 7e-5
        weight_by_source = {}
    ordered = _route(train_rows)
    loader = DataLoader(_Dataset(ordered, stoi), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
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
            row_weights = torch.tensor([weight_by_source.get(source, 1.0) for source in batch["source"]], device=device, dtype=losses.dtype).unsqueeze(1)
            loss = (losses * valid * row_weights).sum() / (valid * row_weights).sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = int(valid.sum().item())
        loss_sum += float(loss.detach().cpu()) * count
        token_sum += count
    keys = tuple(eval_rows.keys())
    metrics = {key: _metrics(model, eval_rows[key], stoi, device) for key in keys}
    old_keys = ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood", "pg176_ood")
    new_ood_keys = tuple(f"pg177_{generator}_ood" for generator in NEW_GENERATORS)
    result: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "config": config,
        "learning_rate": learning_rate,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_row_count": len(ordered),
        "train_token_count": token_sum,
        "train_loss": round(loss_sum / max(token_sum, 1), 8),
        **metrics,
        "aggregate_existing": _aggregate(metrics, old_keys),
        "aggregate_new_ood": _aggregate(metrics, new_ood_keys),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"seed_{seed}_{variant}.pt"
    torch.save({"schema_version": "pg177-data-capacity-v1", "variant": variant, "seed": seed, "config": config, "vocabulary": list(stoi), "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu")
    train_rows, eval_rows, dataset, source_hashes = _prepare_data(checkpoint)
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    for row in train_rows:
        missing = [token for token in row["tokens"] if token not in stoi]
        if missing:
            raise RuntimeError(f"frozen vocabulary missing token: {missing[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {key: _metrics(baseline_model, rows, stoi, device) for key, rows in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        results.append(_train_one("160m_continued", seed, checkpoint, {"d_model": 1152, "nhead": 18, "layers": 10}, train_rows, eval_rows, stoi, device))
        for variant, config in CAPACITY_CONFIGS.items():
            if variant == "160m_continued":
                continue
            results.append(_train_one(variant, seed, checkpoint, config, train_rows, eval_rows, stoi, device))
    old_keys = ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood", "pg176_ood")
    new_ood_keys = tuple(f"pg177_{generator}_ood" for generator in NEW_GENERATORS)
    continued = [result for result in results if result["variant"] == "160m_continued"]
    gates = []
    for result in continued:
        split_ok = all(result[key]["perplexity"] <= baseline[key]["perplexity"] * (1.0 + PER_SPLIT_TOLERANCE) for key in old_keys)
        new_ood_ok = result["aggregate_new_ood"] < _aggregate(baseline, new_ood_keys)
        gates.append({"seed": result["seed"], "existing_split_gate": split_ok, "new_ood_gate": new_ood_ok, "existing_aggregate": result["aggregate_existing"], "new_ood_aggregate": result["aggregate_new_ood"], "pass": split_ok and new_ood_ok})
    scratch = {(result["variant"], result["seed"]): result for result in results if result["variant"] in {"160m_scratch", "200m_scratch"}}
    capacity_comparison = [{"seed": seed, "160m_scratch_new_ood": scratch[("160m_scratch", seed)]["aggregate_new_ood"], "200m_scratch_new_ood": scratch[("200m_scratch", seed)]["aggregate_new_ood"], "200m_better": scratch[("200m_scratch", seed)]["aggregate_new_ood"] < scratch[("160m_scratch", seed)]["aggregate_new_ood"]} for seed in SEEDS]
    report: dict[str, Any] = {
        "schema_version": "pg177-data-capacity-report-v1",
        "protocol_id": "pg-pk-177-data-capacity-v1",
        "status": "completed_pg177_data_capacity_sweep",
        "scope": {"claim": "expanded family-free Rule-IR data with 160M continued, 160M scratch and 200M scratch diagnostics", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)},
        "dataset": {"train_row_count": len(train_rows), "full_eval_counts": {key: len(rows) for key, rows in eval_rows.items()}, "new_generator_counts": dataset["new_generator_counts"], "projection_overlap_prior": 0, "source_dataset_sha256": source_hashes},
        "baseline": baseline,
        "baseline_existing_aggregate": _aggregate(baseline, old_keys),
        "baseline_new_ood_aggregate": _aggregate(baseline, new_ood_keys),
        "variants": results,
        "capacity_comparison": capacity_comparison,
        "gates": gates,
        "selection": {"selected_variant": "160m_continued" if all(gate["pass"] for gate in gates) else None, "promotion_allowed": all(gate["pass"] for gate in gates), "vulnerability_claim_allowed": False},
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False},
        "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(START_CHECKPOINT.read_bytes()).hexdigest(), "dataset_sha256": dataset["dataset_sha256"]},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-177-data-capacity-v1", "schema_version": "pg177-data-capacity-protocol-v1", "start_checkpoint": str(START_CHECKPOINT.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "seeds": list(SEEDS), "capacity_configs": CAPACITY_CONFIGS, "train_row_count": len(train_rows), "hard_gate": {"existing_holdout_keys": list(old_keys), "per_split_tolerance": PER_SPLIT_TOLERANCE, "new_ood_aggregate_must_improve": True}, "promotion": {"training_artifact_promotion_allowed": report["selection"]["promotion_allowed"], "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    summary_lines = ["# PG-177 data and capacity sweep", "", f"- train rows: **{len(train_rows)}**", f"- generators: **{', '.join(NEW_GENERATORS)}**", f"- baseline existing/new-OOD PPL: **{report['baseline_existing_aggregate']} / {report['baseline_new_ood_aggregate']}**", f"- gates: **{gates}**", f"- capacity comparison: **{capacity_comparison}**", "", "该轮只验证抽象 Rule-IR 的 next-token 学习与跨生成器泛化，不生成漏洞标签或攻击 payload。", ""]
    MARKDOWN_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "train_rows": len(train_rows), "gates": gates, "capacity_comparison": capacity_comparison, "selected_variant": report["selection"]["selected_variant"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
