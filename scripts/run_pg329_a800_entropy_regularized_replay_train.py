"""PG-329: A800 replay training with predictive-entropy regularization.

PG-328 showed a 34.9% collapse in predictive entropy despite good symbolic
scores.  This follow-up changes only the training objective: the initial
frozen checkpoint's predictive entropy on the same abstract training batch is
used as a soft reference.  The run is still a research candidate and never
emits a request or promotes a payload/memory entry.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _entropy_regularized_train(pg295: Any, records: Any, vocabulary: Any, device: torch.device, *, seed: int, config: Any, epochs: int, learning_rate: float, token_weights: Any = None, initial_state: Any = None) -> Any:
    """Train the same small LM while anchoring entropy to its initial state."""
    torch.manual_seed(int(seed))
    model = pg295.CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    if initial_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in initial_state.items()})
    ids, valid = pg295._batch(records, vocabulary, device)
    pad = int(vocabulary[pg295.PAD])
    weight_vector = torch.ones((len(vocabulary),), dtype=torch.float32, device=device)
    for token, weight in (token_weights or {}).items():
        if str(token) in vocabulary:
            weight_vector[int(vocabulary[str(token)])] = float(weight)
    with torch.inference_mode():
        reference_logits, _ = model(ids[:, :-1], valid_mask=valid[:, :-1])
        reference_probs = torch.softmax(reference_logits, dim=-1).clamp_min(1e-12)
        reference_entropy = -(reference_probs * reference_probs.log()).sum(dim=-1)
        reference_mask = valid[:, 1:]
        reference_entropy_mean = reference_entropy[reference_mask].mean().detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        logits, balance = model(ids[:, :-1], valid_mask=valid[:, :-1])
        labels = ids[:, 1:]
        label_valid = valid[:, 1:]
        labels = labels.masked_fill(~label_valid, pad)
        per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=pad, reduction="none")
        label_weights = weight_vector[labels.reshape(-1)]
        valid_loss = per_token[label_valid.reshape(-1)] * label_weights[label_valid.reshape(-1)]
        probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy_mean = entropy[label_valid].mean()
        # Keep the abstract outline's uncertainty scale near the frozen
        # baseline without forcing token identities or raw payload strings.
        entropy_penalty = (entropy_mean - reference_entropy_mean).pow(2)
        loss = valid_loss.mean() + 0.01 * balance + 2.0 * entropy_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> int:
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-329 requires BLACKBOX_REMOTE_A800_TRAIN=1 and CUDA_VISIBLE_DEVICES=0")
    pg328 = _load_module("pg328_helpers_for_pg329", ROOT / "scripts" / "run_pg328_a800_entropy_replay_train.py")
    runner = _load_module("pg327_runner_for_pg329", ROOT / "scripts" / "run_pg327_a800_replay_train.py")
    parent = runner._load()
    from app import pg295_causal_moe as pg295

    parent.SEEDS = (31907, 31908, 31909)
    parent.DATASET = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_v1.json"
    parent.AUDIT = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_audit_v1.json"
    base_source_dir = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
    parent.BASE_DIR = ROOT / "artifacts" / "pg329-a800-entropy-regularized" / "base"
    parent.BASE_DIR.mkdir(parents=True, exist_ok=True)
    for run_seed, base_seed in zip(parent.SEEDS, (31901, 31902, 31903)):
        alias = parent.BASE_DIR / f"pg322_cross_impl_decoy_seed_{run_seed}.pt"
        source = base_source_dir / f"pg322_cross_impl_decoy_seed_{base_seed}.pt"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        try:
            alias.symlink_to(source)
        except OSError:
            alias.write_bytes(source.read_bytes())
    parent.BASE_PREFIX = "pg322_cross_impl_decoy_seed_"
    parent.OUT_DIR = ROOT / "artifacts" / "pg329-a800-entropy-regularized" / "seeds"
    parent.CHECKPOINT = ROOT / "artifacts" / "pg329-a800-entropy-regularized" / "pg329_a800_entropy_regularized_candidate.pt"
    parent.REPORT = ROOT / "research" / "pg329_a800_entropy_regularized_training_report_v1.json"
    # Parent still evaluates and saves exactly the same lanes; only its
    # optimizer function is replaced by the entropy-anchored objective.
    parent.PG313.train_causal_moe = lambda records, vocabulary, device, *, seed, config, epochs=70, learning_rate=0.00035, token_weights=None, initial_state=None: _entropy_regularized_train(
        pg295, records, vocabulary, device, seed=seed, config=config, epochs=epochs, learning_rate=learning_rate, token_weights=token_weights, initial_state=initial_state
    )
    os.environ["BLACKBOX_REMOTE_A800_TRAIN"] = "1"
    result = int(parent.main())

    checkpoint_hashes: dict[str, str] = {}
    for path in sorted(parent.OUT_DIR.glob("*.pt")):
        checkpoint_hashes[str(path.relative_to(ROOT))] = runner._rewrite_checkpoint(path)
    checkpoint_hashes[str(parent.CHECKPOINT.relative_to(ROOT))] = runner._rewrite_checkpoint(parent.CHECKPOINT)

    dataset = json.loads(parent.DATASET.read_text(encoding="utf-8-sig"))
    train_rows = [dict(row) for row in dataset.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    role = json.loads((ROOT / "research" / "pg321_variant_role_lattice_dataset_v1.json").read_text(encoding="utf-8-sig"))
    lattice = json.loads((ROOT / "research" / "pg320_observation_lattice_dataset_v1.json").read_text(encoding="utf-8-sig"))
    role_rows = [dict(row) for row in role.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    lattice_rows = [dict(row) for row in lattice.get("records", []) if row.get("split") == "train" and row.get("training_eligible")][::3]
    mix_rows = train_rows + role_rows + lattice_rows
    probe_rows = [dict(row) for row in dataset.get("records", []) if row.get("split") in {"implementation_holdout", "third_surface_holdout", "ask_holdout", "hard_negative_eval"}]
    device = torch.device("cuda:0")
    before_values: list[dict[str, Any]] = []
    after_values: list[dict[str, Any]] = []
    for run_seed, base_seed in zip(parent.SEEDS, (31901, 31902, 31903)):
        before_model, before_vocab = pg328._load_checkpoint(base_source_dir / f"pg322_cross_impl_decoy_seed_{base_seed}.pt", device, pg295)
        after_model, after_vocab = pg328._load_checkpoint(parent.OUT_DIR / f"pg322_cross_impl_decoy_seed_{run_seed}.pt", device, pg295)
        before_values.append(pg328._predictive_entropy(before_model, before_vocab, probe_rows, device, pg295))
        after_values.append(pg328._predictive_entropy(after_model, after_vocab, probe_rows, device, pg295))

    def mean(values: list[dict[str, Any]]) -> dict[str, Any]:
        nums = [float(row["nats"]) for row in values if row.get("nats") is not None]
        return {"nats": round(sum(nums) / len(nums), 6) if nums else None, "count": sum(int(row.get("count") or 0) for row in values), "status": "measured" if nums else "not_applicable"}

    predictive_before = mean(before_values)
    predictive_after = mean(after_values)
    guard = pg328._entropy_guard(predictive_before, predictive_after)
    report = json.loads(parent.REPORT.read_text(encoding="utf-8-sig"))
    report.update({"protocol_id": "pg-pk-329-a800-entropy-regularized-replay-v1", "schema_version": "pg329-a800-entropy-regularized-training-report-v1", "status": "completed_remote_a800_pg329_entropy_regularized_candidate"})
    report["training"].update({"execution_mode": "remote_a800_gpu0", "device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0), "seeds": list(parent.SEEDS), "optimizer_change": "entropy_anchored_to_frozen_initial_predictive_distribution", "entropy_regularization_weight": 2.0, "raw_payload_in_context": False, "raw_response_body_in_context": False})
    report["information_entropy"] = {"outline_input_context": pg328._empirical_entropy(mix_rows, "context_tokens"), "outline_target_tokens": pg328._empirical_entropy(mix_rows, "target_tokens"), "probe_rows": len(probe_rows), "predictive_before": predictive_before, "predictive_after": predictive_after, "per_seed_before": before_values, "per_seed_after": after_values, "guard": guard, "interpretation": "PG-329 只测试熵保持正则是否抑制抽象 token 分布塌缩；不等价于漏洞检测或 payload 成功。"}
    report["hypothesis_gate"]["checks"]["information_entropy_preserved"] = bool(guard.get("passed", False))
    report["hypothesis_gate"]["status"] = "blocked"
    report["hypothesis_gate"]["claim_allowed"] = False
    report["promotion"] = {"training_allowed": True, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only", "promotion_blocked": True}
    report["provenance"] = {"training_script_sha256": _sha256(Path(__file__).resolve()), "pg328_script_sha256": _sha256(ROOT / "scripts" / "run_pg328_a800_entropy_replay_train.py"), "parent_runner_sha256": _sha256(ROOT / "scripts" / "run_pg327_a800_replay_train.py"), "parent_loop_sha256": _sha256(ROOT / "scripts" / "run_pg322_cross_impl_decoy_moe.py"), "model_impl_sha256": _sha256(ROOT / "app" / "pg295_causal_moe.py"), "dataset_file_sha256": _sha256(parent.DATASET), "audit_file_sha256": _sha256(parent.AUDIT), "checkpoint_sha256": checkpoint_hashes}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    parent.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": report["training"]["device"], "gpu": report["training"]["gpu_name"], "seeds": report["training"]["seeds"], "entropy_guard": guard, "metrics": report.get("metrics"), "promotion_blocked": True, "checkpoint": str(parent.CHECKPOINT.relative_to(ROOT)), "report": str(parent.REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
