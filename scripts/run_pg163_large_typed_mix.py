"""PG-163: mix fresh typed traces into a large causal-token training run.

PG-147 already demonstrated that the repository can train 19M/57M causal
Transformers on abstract traces.  PG-163 adds independently collected PG-116
and PG-118 episodes as *model-facing, family-free Rule-IR tokens*.  It then
trains Large and XL models from scratch and reports both ordinary language
holdout metrics and a typed-trace holdout.  Oracle labels, raw probes,
response bodies, target identities and evidence hashes are kept out of the
token stream.

This is representation/pretraining research.  It is not a claim that the
result is a real vulnerability scanner.
"""

from __future__ import annotations

import asyncio
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
from app.pg116_multisource_replay import collect_source  # noqa: E402
from app.pg118_transition_replay import collect_target  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg163-large-typed-mix-v1"
DATASET_PATH = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg163_large_typed_mix_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg163_large_typed_mix_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg163_large_typed_mix_report_v1.md"
BASE_DATASET = RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json"
BASE_REPORT = RESEARCH / "pg147_model_capacity_sweep_report_v1.json"
PG119_VISIBLE = RESEARCH / "pg119_metadata_visible_dataset_v1.json"
PG123_DATASET = RESEARCH / "pg123_authorization_slot_training_dataset_v1.json"
MAX_LEN = 128
SEED = 16301
PG116_TRAIN_SEEDS = list(range(16301, 16311))
PG116_HOLDOUT_SEEDS = list(range(16311, 16316))
PG118_TRAIN_SEEDS = list(range(16321, 16331))
PG118_HOLDOUT_SEEDS = list(range(16331, 16336))
SPECIAL = ("[PAD]", "[UNK]")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    # Importing the decoder here would expose no evaluator data; this local
    # projection is intentionally stricter and only retains bounded fields.
    action = value.get("action_manifest") or {}
    baseline = value.get("baseline_projection") or {}
    response = value.get("response_projection") or {}
    belief = value.get("belief_before") or {}
    bsp = response.get("bsp_core_projection") or {}
    return {
        "action": {
            "method": str(action.get("method", "")).upper(),
            "placement": str(action.get("placement", "unknown")),
            "encoding_depth": min(len(action.get("encoding_chain") or []), 4),
        },
        "baseline": {"status_class": str(baseline.get("status_class", "unknown")), "body_length_bucket": str(baseline.get("body_length_bucket", "unknown"))},
        "response": {
            "candidate_signal": bool(response.get("candidate_signal")),
            "noise_bucket": min(max(int(response.get("noise_bucket", 0) or 0), 0), 16),
            "policy_header_changed": bool(response.get("policy_header_changed")),
            "shape_changed": bool(response.get("shape_changed")),
            "location_changed": bool(response.get("location_changed")),
            "transition_delta": str(response.get("transition_delta", "unknown")) if response.get("transition_delta") in {"location", "none"} else "unknown",
            "metadata_changed": bool(response.get("metadata_changed")),
            "authorization_changed": bool(response.get("authorization_changed")),
            "authorization_status": str(response.get("authorization_status", "unknown")),
            "shape_class": str(response.get("shape_class", "unknown")),
            "status_class": str(response.get("status_class", "unknown")),
            "bsp_leaf_count": min(len(bsp.get("selected_leaf_ids") or []), 8),
            "bsp_topology_version": min(max(int(bsp.get("topology_version", 0) or 0), 0), 8),
        },
        "belief": {key: float(belief.get(key, 0.0) or 0.0) for key in ("effect", "input_only", "no_effect", "unknown")},
    }


def _bucket(value: float) -> str:
    if value < 0.2:
        return "0-0.2"
    if value < 0.4:
        return "0.2-0.4"
    if value < 0.6:
        return "0.4-0.6"
    if value < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _episode_tokens(episode: dict[str, Any]) -> list[str]:
    tokens = ["[BOS]"]
    steps = episode.get("steps") or []
    for index, step in enumerate(steps):
        value = _canonical(step.get("model_input") or step)
        action = value["action"]
        baseline = value["baseline"]
        response = value["response"]
        belief = value["belief"]
        tokens.extend([
            "[STEP]",
            "[RESET]",
            "[SRC_TRANSPORT]",
            f"src.transport.method={action['method']}",
            f"src.transport.placement={action['placement']}",
            f"src.transport.encoding_depth={action['encoding_depth']}",
            "[IR]",
            f"ir.baseline.status_class={baseline['status_class']}",
            f"ir.baseline.body_length_bucket={baseline['body_length_bucket']}",
            f"ir.response.candidate_signal={str(response['candidate_signal']).lower()}",
            f"ir.response.noise_bucket={response['noise_bucket']}",
            f"ir.response.policy_header_changed={str(response['policy_header_changed']).lower()}",
            f"ir.response.shape_changed={str(response['shape_changed']).lower()}",
            f"ir.response.location_changed={str(response['location_changed']).lower()}",
            f"ir.response.transition_delta={response['transition_delta']}",
            f"ir.response.metadata_changed={str(response['metadata_changed']).lower()}",
            f"ir.response.authorization_changed={str(response['authorization_changed']).lower()}",
            f"ir.response.authorization_status={response['authorization_status']}",
            f"ir.response.shape_class={response['shape_class']}",
            f"ir.response.status_class={response['status_class']}",
            f"ir.bsp.leaf_count={response['bsp_leaf_count']}",
            f"ir.bsp.topology_version={response['bsp_topology_version']}",
            "[BELIEF]",
            f"belief.effect={_bucket(belief['effect'])}",
            f"belief.input_only={_bucket(belief['input_only'])}",
            f"belief.no_effect={_bucket(belief['no_effect'])}",
            f"belief.unknown={_bucket(belief['unknown'])}",
            "[OBS]",
            f"obs.step_index={index + 1}",
        ])
    tokens.append("[EOS]")
    return tokens[:MAX_LEN]


async def _collect_typed() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    evidence: list[str] = []
    source_episode_counts: dict[str, int] = {}
    for source in ("alpha", "beta"):
        train_target = await collect_source(source, PG116_TRAIN_SEEDS)
        holdout_target = await collect_source(source, PG116_HOLDOUT_SEEDS)
        for target, destination, split in ((train_target, train, "train"), (holdout_target, holdout, "typed_holdout")):
            for episode in target["episodes"]:
                tokens = _episode_tokens(episode)
                destination.append({"row_id": episode["episode_id"], "tokens": tokens, "source_group": f"pg116_{source}", "split": split})
                evidence.extend(step["evidence_sha256"] for step in episode["steps"])
        source_episode_counts[f"pg116_{source}"] = len(train_target["episodes"]) + len(holdout_target["episodes"])
    for seed in PG118_TRAIN_SEEDS:
        target = await collect_target(seed)
        for episode in target["episodes"]:
            train.append({"row_id": episode["episode_id"], "tokens": _episode_tokens(episode), "source_group": "pg118_delta", "split": "train"})
            evidence.extend(step["evidence_sha256"] for step in episode["steps"])
    for seed in PG118_HOLDOUT_SEEDS:
        target = await collect_target(seed)
        for episode in target["episodes"]:
            holdout.append({"row_id": episode["episode_id"], "tokens": _episode_tokens(episode), "source_group": "pg118_delta", "split": "typed_holdout"})
            evidence.extend(step["evidence_sha256"] for step in episode["steps"])
    source_episode_counts["pg118_delta"] = len(PG118_TRAIN_SEEDS) * 4 + len(PG118_HOLDOUT_SEEDS) * 4
    return train, holdout, {"evidence_hash_count": len(evidence), "evidence_hash_valid": all(isinstance(x, str) and len(x) == 64 for x in evidence), "source_episode_counts": source_episode_counts}


def _load_approved_slot_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Load already-attested visible projections without importing labels.

    PG-119 and PG-123 are approved local typed datasets.  Only their bounded
    ``model_input`` projection is converted; ``training_label``, source names,
    oracle fields and any trace metadata remain outside the token stream.
    """

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    visible = json.loads(PG119_VISIBLE.read_text(encoding="utf-8"))
    for row in visible.get("rows", []):
        output = train if row.get("split") == "train" else holdout
        output.append({
            "row_id": f"pg119-{row['row_id']}",
            "tokens": _episode_tokens({"steps": [{"model_input": row["model_input"]}]}),
            "source_group": "pg119_metadata",
            "split": "train" if row.get("split") == "train" else "typed_holdout",
        })
    counts["pg119_metadata_train"] = sum(row.get("split") == "train" for row in visible.get("rows", []))
    counts["pg119_metadata_holdout"] = sum(row.get("split") == "dev" for row in visible.get("rows", []))
    dataset = json.loads(PG123_DATASET.read_text(encoding="utf-8"))
    for source_key, output, split in (("train_rows", train, "train"), ("dev_rows", holdout, "typed_holdout")):
        for row in dataset.get(source_key, []):
            output.append({
                "row_id": f"pg123-{row['row_id']}",
                "tokens": _episode_tokens({"steps": [{"model_input": row["model_input"]}]}),
                "source_group": "pg123_authorization",
                "split": split,
            })
        counts[f"pg123_authorization_{split}"] = len(dataset.get(source_key, []))
    return train, holdout, counts


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


def _metrics(model: CausalTraceTransformer, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
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


def _train_variant(name: str, config: dict[str, int | float], train_rows: list[dict[str, Any]], base_dev_rows: list[dict[str, Any]], base_holdout_rows: list[dict[str, Any]], typed_holdout_rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    torch.manual_seed(SEED + int(config["d_model"]))
    random.seed(SEED + int(config["d_model"]))
    model = CausalTraceTransformer(len(stoi), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    train_loader = DataLoader(_TraceDataset(train_rows, stoi), batch_size=int(config["batch_size"]), shuffle=True, collate_fn=_collate)
    base_dev_loader = DataLoader(_TraceDataset(base_dev_rows, stoi), batch_size=int(config["batch_size"]), shuffle=False, collate_fn=_collate)
    base_holdout_loader = DataLoader(_TraceDataset(base_holdout_rows, stoi), batch_size=int(config["batch_size"]), shuffle=False, collate_fn=_collate)
    typed_holdout_loader = DataLoader(_TraceDataset(typed_holdout_rows, stoi), batch_size=int(config["batch_size"]), shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
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
            valid = int(targets.ne(0).sum().item())
            loss_sum += float(loss.detach().cpu()) * valid
            token_sum += valid
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_dev": _metrics(model, base_dev_loader, device), "typed_holdout": _metrics(model, typed_holdout_loader, device)})
    result = {
        "variant": name,
        "config": config,
        "parameter_count": params,
        "train": _metrics(model, train_loader, device),
        "base_dev": _metrics(model, base_dev_loader, device),
        "base_holdout": _metrics(model, base_holdout_loader, device),
        "typed_holdout": _metrics(model, typed_holdout_loader, device),
        "history_tail": history[-3:],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg163-large-typed-mix-v1", "variant": name, "config": config, "vocabulary": sorted(stoi, key=stoi.get), "model_state_dict": model.state_dict()}, checkpoint)
    result["checkpoint"] = str(checkpoint.relative_to(ROOT))
    return result


async def main() -> None:
    typed_train, typed_holdout, typed_stats = await _collect_typed()
    approved_train, approved_holdout, approved_counts = _load_approved_slot_rows()
    typed_train.extend(approved_train)
    typed_holdout.extend(approved_holdout)
    typed_stats["approved_slot_row_counts"] = approved_counts
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    base_train = list(base["splits"]["train"])
    base_dev = list(base["splits"]["dev"])
    base_holdout = list(base["splits"]["holdout"])
    # Deduplicate only by model-visible token sequence; metadata/evidence
    # cannot be used to create artificial language examples.
    seen_train = {tuple(row["tokens"][:MAX_LEN]) for row in base_train}
    unique_typed_train: list[dict[str, Any]] = []
    for row in typed_train:
        key = tuple(row["tokens"][:MAX_LEN])
        if key not in seen_train:
            seen_train.add(key)
            unique_typed_train.append(row)
    # A source/seed holdout may not reuse a model-visible sequence that was
    # already seen in training; otherwise the LM score would overstate
    # cross-target generalization.
    seen_holdout = {tuple(row["tokens"][:MAX_LEN]) for row in base_holdout} | seen_train
    unique_typed_holdout: list[dict[str, Any]] = []
    for row in typed_holdout:
        key = tuple(row["tokens"][:MAX_LEN])
        if key not in seen_holdout:
            seen_holdout.add(key)
            unique_typed_holdout.append(row)
    train_rows = base_train + unique_typed_train
    # New typed holdout is kept out of the ordinary dev and holdout metrics.
    all_train_tokens = {token for row in train_rows for token in row["tokens"]}
    vocabulary = list(SPECIAL) + sorted(token for token in all_train_tokens if token not in SPECIAL)
    stoi = {token: index for index, token in enumerate(vocabulary)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {
        "large_typed_mix": {"d_model": 512, "nhead": 8, "layers": 6, "batch_size": 48, "epochs": 2, "lr": 1.5e-4},
        "xl_typed_mix": {"d_model": 768, "nhead": 12, "layers": 8, "batch_size": 24, "epochs": 2, "lr": 1e-4},
    }
    results = {name: _train_variant(name, config, train_rows, base_dev, base_holdout, unique_typed_holdout, stoi, device) for name, config in variants.items()}
    best = min(results, key=lambda name: results[name]["typed_holdout"]["perplexity"])
    dataset = {
        "schema_version": "pg163-large-typed-mix-dataset-v1",
        "base_dataset": str(BASE_DATASET.relative_to(ROOT)),
        "train_rows": train_rows,
        "base_dev_rows": base_dev,
        "base_holdout_rows": base_holdout,
        "typed_holdout_rows": unique_typed_holdout,
        "vocabulary": vocabulary,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_labels_in_tokens": False,
        "target_identity_in_tokens": False,
        "family_labels_in_tokens": False,
        "memory_promotion_allowed": False,
    }
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    protocol = {
        "protocol_id": "pg-pk-163-large-typed-mix-v1",
        "schema_version": "pg163-large-typed-mix-protocol-v1",
        "objective": "用 fresh typed GET/POST Rule-IR projection 增强大规模 causal token 预训练，并观察 typed holdout 泛化和基础语言保持。",
        "source_set": ["pg147_abstract_corpus", "pg116_alpha_fresh", "pg116_beta_fresh", "pg118_delta_fresh", "pg119_metadata_approved", "pg123_authorization_approved"],
        "train_seed_sets": {"pg116": PG116_TRAIN_SEEDS, "pg118": PG118_TRAIN_SEEDS},
        "holdout_seed_sets": {"pg116": PG116_HOLDOUT_SEEDS, "pg118": PG118_HOLDOUT_SEEDS},
        "sequence_contract": {"max_len": MAX_LEN, "get_post_balanced": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "oracle_labels_in_tokens": False, "target_identity_in_tokens": False, "family_labels_in_tokens": False},
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False},
        "training_artifact_promotion_allowed": False,
        "memory_promotion_allowed": False,
    }
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    base_report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    report = {
        "schema_version": "pg163-large-typed-mix-report-v1",
        "protocol_id": "pg-pk-163-large-typed-mix-v1",
        "status": "completed_pg163_large_typed_mix",
        "scope": {"claim": "causal Rule-IR representation learning only", "real_vulnerability_scanner_claim_allowed": False, "device": str(device), "max_len": MAX_LEN},
        "fresh_typed_collection": {**typed_stats, "train_projection_count": len(typed_train), "typed_holdout_projection_count": len(typed_holdout), "typed_train_unique_count": len(unique_typed_train), "typed_holdout_unique_count": len(unique_typed_holdout), "get_step_count": typed_stats["evidence_hash_count"] // 2, "post_step_count": typed_stats["evidence_hash_count"] // 2},
        "corpus": {"base_train_count": len(base_train), "mixed_train_count": len(train_rows), "base_dev_count": len(base_dev), "base_holdout_count": len(base_holdout), "typed_holdout_count": len(unique_typed_holdout), "vocabulary_size": len(vocabulary), "base_pg147_report": str(BASE_REPORT.relative_to(ROOT))},
        "variants": results,
        "selection": {"best_typed_holdout_variant": best, "promotion_allowed": False, "reason": "大模型在受控 typed projection 上的语言建模结果不能直接升级为漏洞扫描能力"},
        "baseline_reference": {"pg147_large_holdout_perplexity": next((item["holdout"]["perplexity"] for item in base_report.get("variants", []) if item.get("variant") == "large_transformer"), None), "pg147_xl_holdout_perplexity": next((item["holdout"]["perplexity"] for item in base_report.get("variants", []) if item.get("variant") == "xl_transformer"), None), "comparison_note": "仅作历史参考；本轮 vocabulary 与训练混合已变化，不把数值差异解释成严格遗忘因果。"},
        "safety": protocol["safety"],
        "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "dataset_sha256": dataset["dataset_sha256"], "protocol_sha256": protocol["protocol_sha256"]},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-163 大模型 typed mix 实验",
        "",
        f"- fresh typed episodes: **{len(typed_train) + len(typed_holdout)}**（train {len(typed_train)} / holdout {len(typed_holdout)}）",
        f"- mixed train sequences: **{len(train_rows)}**；typed holdout: **{len(unique_typed_holdout)}**",
        f"- device: **{device}**；vocabulary: **{len(vocabulary)}**",
        f"- best typed holdout: **{best}**",
        "",
        "模型输入只含抽象 Rule-IR token；原始 probe、响应正文、目标身份、族名和 oracle 标签均不进 token。结果仍是表示学习，不是漏洞扫描认证。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "fresh_typed_episodes": len(typed_train) + len(typed_holdout), "mixed_train_sequences": len(train_rows), "typed_holdout_sequences": len(unique_typed_holdout), "vocabulary_size": len(vocabulary), "variants": {name: {"params": value["parameter_count"], "base_holdout_ppl": value["base_holdout"]["perplexity"], "typed_holdout_ppl": value["typed_holdout"]["perplexity"], "typed_holdout_acc": value["typed_holdout"]["next_token_accuracy"]} for name, value in results.items()}, "best": best, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
