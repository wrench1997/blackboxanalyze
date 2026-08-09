"""PG-330: teacher-distribution replay to stop abstract-token collapse.

PG-328 and PG-329 kept symbolic holdout scores but lost about 35% of the
fixed-holdout predictive entropy.  This candidate uses the frozen PG-322
checkpoint as a teacher at every abstract training position (token-level KL,
not a family/route label).  It is an offline A800 experiment only; no target
or wire data is available to the model.
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


def _teacher_kl_train(pg295: Any, records: Any, vocabulary: Any, device: torch.device, *, seed: int, config: Any, epochs: int, learning_rate: float, token_weights: Any = None, initial_state: Any = None) -> Any:
    torch.manual_seed(int(seed))
    model = pg295.CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    if initial_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in initial_state.items()})
    ids, valid = pg295._batch(records, vocabulary, device)
    pad = int(vocabulary[pg295.PAD])
    weights = torch.ones((len(vocabulary),), dtype=torch.float32, device=device)
    for token, weight in (token_weights or {}).items():
        if str(token) in vocabulary:
            weights[int(vocabulary[str(token)])] = float(weight)
    # Use no_grad rather than inference_mode: the KL target must be a normal
    # detached tensor that autograd can save while differentiating the student.
    with torch.no_grad():
        teacher_logits, _ = model(ids[:, :-1], valid_mask=valid[:, :-1])
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1).detach()
        # ``inference_mode`` tensors cannot be captured by autograd as a
        # constant target; clone creates an ordinary detached tensor for KL.
        teacher_probs = teacher_log_probs.exp().detach().clone()
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
        label_weights = weights[labels.reshape(-1)]
        supervised = (per_token[label_valid.reshape(-1)] * label_weights[label_valid.reshape(-1)]).mean()
        student_log_probs = torch.log_softmax(logits, dim=-1)
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none", log_target=False).sum(dim=-1)
        teacher_kl = kl_per_token[label_valid].mean()
        # Token-level distillation keeps the full abstract next-token
        # distribution, while the weighted CE still trains ASK/Rule-IR targets.
        loss = supervised + 0.35 * teacher_kl + 0.01 * balance
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
        raise RuntimeError("PG-330 requires BLACKBOX_REMOTE_A800_TRAIN=1 and CUDA_VISIBLE_DEVICES=0")
    pg328 = _load_module("pg328_helpers_for_pg330", ROOT / "scripts" / "run_pg328_a800_entropy_replay_train.py")
    runner = _load_module("pg327_runner_for_pg330", ROOT / "scripts" / "run_pg327_a800_replay_train.py")
    parent = runner._load()
    from app import pg295_causal_moe as pg295

    parent.SEEDS = (31910, 31911, 31912)
    parent.DATASET = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_v1.json"
    parent.AUDIT = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_audit_v1.json"
    base_source_dir = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
    parent.BASE_DIR = ROOT / "artifacts" / "pg330-a800-teacher-kl" / "base"
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
    parent.OUT_DIR = ROOT / "artifacts" / "pg330-a800-teacher-kl" / "seeds"
    parent.CHECKPOINT = ROOT / "artifacts" / "pg330-a800-teacher-kl" / "pg330_a800_teacher_kl_candidate.pt"
    parent.REPORT = ROOT / "research" / "pg330_a800_teacher_kl_training_report_v1.json"
    parent.PG313.train_causal_moe = lambda records, vocabulary, device, *, seed, config, epochs=70, learning_rate=0.00035, token_weights=None, initial_state=None: _teacher_kl_train(
        pg295, records, vocabulary, device, seed=seed, config=config, epochs=epochs, learning_rate=learning_rate, token_weights=token_weights, initial_state=initial_state
    )
    os.environ["BLACKBOX_REMOTE_A800_TRAIN"] = "1"
    result = int(parent.main())

    hashes: dict[str, str] = {}
    for path in sorted(parent.OUT_DIR.glob("*.pt")):
        hashes[str(path.relative_to(ROOT))] = runner._rewrite_checkpoint(path)
    hashes[str(parent.CHECKPOINT.relative_to(ROOT))] = runner._rewrite_checkpoint(parent.CHECKPOINT)
    dataset = json.loads(parent.DATASET.read_text(encoding="utf-8-sig"))
    train_rows = [dict(row) for row in dataset.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    role = json.loads((ROOT / "research" / "pg321_variant_role_lattice_dataset_v1.json").read_text(encoding="utf-8-sig"))
    lattice = json.loads((ROOT / "research" / "pg320_observation_lattice_dataset_v1.json").read_text(encoding="utf-8-sig"))
    mix_rows = train_rows + [dict(row) for row in role.get("records", []) if row.get("split") == "train" and row.get("training_eligible")] + [dict(row) for row in lattice.get("records", []) if row.get("split") == "train" and row.get("training_eligible")][::3]
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
    report.update({"protocol_id": "pg-pk-330-a800-teacher-kl-replay-v1", "schema_version": "pg330-a800-teacher-kl-training-report-v1", "status": "completed_remote_a800_pg330_teacher_kl_candidate"})
    report["training"].update({"execution_mode": "remote_a800_gpu0", "device": "cuda:0", "gpu_name": torch.cuda.get_device_name(0), "seeds": list(parent.SEEDS), "optimizer_change": "token_level_teacher_distribution_kl_from_frozen_pg322", "teacher_kl_weight": 0.35, "raw_payload_in_context": False, "raw_response_body_in_context": False})
    report["information_entropy"] = {"outline_input_context": pg328._empirical_entropy(mix_rows, "context_tokens"), "outline_target_tokens": pg328._empirical_entropy(mix_rows, "target_tokens"), "probe_rows": len(probe_rows), "predictive_before": predictive_before, "predictive_after": predictive_after, "per_seed_before": before_values, "per_seed_after": after_values, "guard": guard, "interpretation": "PG-330 仅验证逐 token teacher KL 是否保住抽象分布；不等价于漏洞检测或 payload 成功。"}
    report["hypothesis_gate"]["checks"]["information_entropy_preserved"] = bool(guard.get("passed", False))
    report["hypothesis_gate"]["status"] = "blocked"
    report["hypothesis_gate"]["claim_allowed"] = False
    report["promotion"] = {"training_allowed": True, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only", "promotion_blocked": True}
    report["provenance"] = {"training_script_sha256": _sha256(Path(__file__).resolve()), "pg328_script_sha256": _sha256(ROOT / "scripts" / "run_pg328_a800_entropy_replay_train.py"), "parent_runner_sha256": _sha256(ROOT / "scripts" / "run_pg327_a800_replay_train.py"), "parent_loop_sha256": _sha256(ROOT / "scripts" / "run_pg322_cross_impl_decoy_moe.py"), "model_impl_sha256": _sha256(ROOT / "app" / "pg295_causal_moe.py"), "dataset_file_sha256": _sha256(parent.DATASET), "audit_file_sha256": _sha256(parent.AUDIT), "checkpoint_sha256": hashes}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    parent.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": report["training"]["device"], "gpu": report["training"]["gpu_name"], "seeds": report["training"]["seeds"], "entropy_guard": guard, "metrics": report.get("metrics"), "promotion_blocked": True, "checkpoint": str(parent.CHECKPOINT.relative_to(ROOT)), "report": str(parent.REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
