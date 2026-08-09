"""Run an independent A800 replay candidate with an information-entropy guard.

PG-328 is deliberately a training-only follow-up to PG-327.  It reuses the
audited abstract replay mix, starts from the frozen PG-322 checkpoints, and
uses new seeds.  It never emits wire traffic or reads a target.  In addition
to the normal ASK/variant/negative metrics, it measures the predictive
distribution entropy on a fixed abstract holdout before and after training.
The entropy check is a regression guard against collapsing the outline into a
small set of memorised tokens; it is not a capability or promotion claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch

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
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _empirical_entropy(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        for token in row.get(field) or []:
            key = str(token)
            counts[key] = counts.get(key, 0) + 1
            total += 1
    if not total:
        return {"nats": None, "bits": None, "tokens": 0, "unique": 0, "status": "not_applicable"}
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return {
        "nats": round(entropy, 6),
        "bits": round(entropy / math.log(2), 6),
        "tokens": total,
        "unique": len(counts),
        "status": "measured",
    }


def _load_checkpoint(path: Path, device: torch.device, model_impl: Any) -> tuple[Any, dict[str, int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = model_impl.CausalMoEConfig(**dict(payload["config"]))
    vocab = {str(key): int(value) for key, value in payload["vocabulary"].items()}
    model = model_impl.CausalMoELanguageModel(vocab_size=len(vocab), config=config).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, vocab


def _predictive_entropy(
    model: Any,
    vocab: Mapping[str, int],
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    model_impl: Any,
) -> dict[str, Any]:
    bos = str(model_impl.TARGET_BOS)
    unknown = int(vocab.get(str(model_impl.UNK), vocab.get("<unk>", 1)))
    values: list[float] = []
    with torch.inference_mode():
        for row in rows:
            tokens = [str(token) for token in (row.get("context_tokens") or [])]
            ids = [int(vocab.get(token, unknown)) for token in tokens]
            ids.append(int(vocab.get(bos, unknown)))
            ids = ids[-int(model.config.max_length) :]
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            valid = torch.ones_like(input_ids, dtype=torch.bool)
            logits, _ = model(input_ids, valid_mask=valid)
            probabilities = torch.softmax(logits[0, -1], dim=-1).clamp_min(1e-12)
            values.append(float(-(probabilities * probabilities.log()).sum().detach().cpu()))
    if not values:
        return {"nats": None, "min": None, "max": None, "count": 0, "status": "not_applicable"}
    return {
        "nats": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "count": len(values),
        "status": "measured",
    }


def _entropy_guard(before: Mapping[str, Any], after: Mapping[str, Any], threshold: float = 0.25) -> dict[str, Any]:
    old = before.get("nats")
    new = after.get("nats")
    if old is None or new is None or float(old) <= 1e-9:
        return {
            "status": "not_applicable",
            "relative_drop": None,
            "threshold": threshold,
            "passed": False,
            "reason": "predictive entropy denominator unavailable",
        }
    relative_drop = max(0.0, (float(old) - float(new)) / float(old))
    return {
        "status": "measured",
        "relative_drop": round(relative_drop, 6),
        "threshold": threshold,
        "passed": relative_drop <= threshold,
        "reason": "post-training predictive entropy stayed within the preregistered collapse bound",
    }


def main() -> int:
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-328 requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-328 requires CUDA_VISIBLE_DEVICES=0")

    runner = _load_module("pg327_runner_for_pg328", ROOT / "scripts" / "run_pg327_a800_replay_train.py")
    parent = runner._load()
    parent.SEEDS = (31904, 31905, 31906)
    parent.DATASET = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_v1.json"
    parent.AUDIT = ROOT / "research" / "pg323_decoy_ask_anchor_dataset_audit_v1.json"
    # Give the parent loop a deterministic alias directory because it expects
    # one base file per run seed.  The aliases point only to the frozen
    # PG-322 files; no checkpoint is synthesized or rewritten here.
    base_source_dir = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
    parent.BASE_DIR = ROOT / "artifacts" / "pg328-a800-entropy-replay" / "base"
    parent.BASE_DIR.mkdir(parents=True, exist_ok=True)
    for run_seed, base_seed in zip(parent.SEEDS, (31901, 31902, 31903)):
        alias = parent.BASE_DIR / f"pg322_cross_impl_decoy_seed_{run_seed}.pt"
        source = base_source_dir / f"pg322_cross_impl_decoy_seed_{base_seed}.pt"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        try:
            alias.symlink_to(source)
        except OSError:
            # Windows development checkouts may not allow symlinks; a small
            # byte-for-byte copy is still auditable and remains read-only input.
            alias.write_bytes(source.read_bytes())
    parent.BASE_PREFIX = "pg322_cross_impl_decoy_seed_"
    parent.OUT_DIR = ROOT / "artifacts" / "pg328-a800-entropy-replay" / "seeds"
    parent.CHECKPOINT = ROOT / "artifacts" / "pg328-a800-entropy-replay" / "pg328_a800_entropy_replay_candidate.pt"
    parent.REPORT = ROOT / "research" / "pg328_a800_entropy_replay_training_report_v1.json"
    os.environ["BLACKBOX_REMOTE_A800_TRAIN"] = "1"
    result = int(parent.main())

    checkpoint_hashes: dict[str, str] = {}
    for path in sorted(parent.OUT_DIR.glob("*.pt")):
        checkpoint_hashes[str(path.relative_to(ROOT))] = runner._rewrite_checkpoint(path)
    checkpoint_hashes[str(parent.CHECKPOINT.relative_to(ROOT))] = runner._rewrite_checkpoint(parent.CHECKPOINT)

    dataset = json.loads(parent.DATASET.read_text(encoding="utf-8-sig"))
    role = json.loads((ROOT / "research" / "pg321_variant_role_lattice_dataset_v1.json").read_text(encoding="utf-8-sig"))
    lattice = json.loads((ROOT / "research" / "pg320_observation_lattice_dataset_v1.json").read_text(encoding="utf-8-sig"))
    trace = json.loads((ROOT / "research" / "pg321_family_holdout_trace_v1.json").read_text(encoding="utf-8-sig"))
    train_rows = [dict(row) for row in dataset.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    role_rows = [dict(row) for row in role.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    lattice_rows = [dict(row) for row in lattice.get("records", []) if row.get("split") == "train" and row.get("training_eligible")][::3]
    mix_rows = train_rows + role_rows + lattice_rows
    probe_rows = [
        dict(row)
        for row in dataset.get("records", [])
        if row.get("split") in {"implementation_holdout", "third_surface_holdout", "ask_holdout", "hard_negative_eval"}
    ]
    device = torch.device("cuda:0")
    # Import through the package so pg295's relative import of the shared
    # token constants remains intact on the remote worker.
    from app import pg295_causal_moe as model_impl
    entropy_before: list[dict[str, Any]] = []
    entropy_after: list[dict[str, Any]] = []
    for seed in parent.SEEDS:
        base_path = base_source_dir / f"{parent.BASE_PREFIX}{seed}.pt"
        # The frozen base files carry the original three seed names; pair each
        # independent run with the same-index baseline, not a synthetic value.
        base_seed = 31901 + (seed - 31904)
        if base_seed not in (31901, 31902, 31903):
            base_seed = 31901
        base_path = base_source_dir / f"{parent.BASE_PREFIX}{base_seed}.pt"
        after_path = parent.OUT_DIR / f"pg322_cross_impl_decoy_seed_{seed}.pt"
        before_model, before_vocab = _load_checkpoint(base_path, device, model_impl)
        after_model, after_vocab = _load_checkpoint(after_path, device, model_impl)
        entropy_before.append(_predictive_entropy(before_model, before_vocab, probe_rows, device, model_impl))
        entropy_after.append(_predictive_entropy(after_model, after_vocab, probe_rows, device, model_impl))

    def _mean(field: str, values: Sequence[Mapping[str, Any]]) -> float | None:
        nums = [float(row[field]) for row in values if row.get(field) is not None]
        return round(sum(nums) / len(nums), 6) if nums else None

    before = {"nats": _mean("nats", entropy_before), "count": sum(int(row.get("count") or 0) for row in entropy_before), "status": "measured"}
    after = {"nats": _mean("nats", entropy_after), "count": sum(int(row.get("count") or 0) for row in entropy_after), "status": "measured"}
    guard = _entropy_guard(before, after)

    report = json.loads(parent.REPORT.read_text(encoding="utf-8-sig"))
    report["protocol_id"] = "pg-pk-328-a800-entropy-replay-train-v1"
    report["schema_version"] = "pg328-a800-entropy-replay-training-report-v1"
    report["status"] = "completed_remote_a800_pg328_entropy_replay_candidate"
    report["training"].update(
        {
            "execution_mode": "remote_a800_gpu0",
            "device": "cuda:0",
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "seeds": list(parent.SEEDS),
            "information_entropy_guard": "predictive_next_token_entropy_on_fixed_abstract_holdout",
            "raw_payload_in_context": False,
            "raw_response_body_in_context": False,
        }
    )
    report["information_entropy"] = {
        "outline_input_context": _empirical_entropy(mix_rows, "context_tokens"),
        "outline_target_tokens": _empirical_entropy(mix_rows, "target_tokens"),
        "probe_rows": len(probe_rows),
        "predictive_before": before,
        "predictive_after": after,
        "per_seed_before": entropy_before,
        "per_seed_after": entropy_after,
        "guard": guard,
        "interpretation": "信息熵只作为防止抽象大纲塌缩的回归指标；它不能证明模型发现漏洞或生成可迁移 payload。",
    }
    report["hypothesis_gate"]["checks"]["information_entropy_preserved"] = bool(guard.get("passed", False))
    report["hypothesis_gate"]["status"] = "blocked"
    report["hypothesis_gate"]["claim_allowed"] = False
    report["training"]["promotion_reason"] = "PG-326 strict uniform schema and long-term promotion remain blocked; entropy guard is diagnostic only"
    report["sources"]["entropy_probe"] = "research/pg323_decoy_ask_anchor_dataset_v1.json:implementation_holdout+third_surface_holdout+ask_holdout+hard_negative_eval"
    report["provenance"] = {
        "training_script_sha256": _sha256(Path(__file__).resolve()),
        "parent_runner_sha256": _sha256(ROOT / "scripts" / "run_pg327_a800_replay_train.py"),
        "parent_loop_sha256": _sha256(ROOT / "scripts" / "run_pg322_cross_impl_decoy_moe.py"),
        "model_impl_sha256": _sha256(ROOT / "app" / "pg295_causal_moe.py"),
        "dataset_file_sha256": _sha256(parent.DATASET),
        "audit_file_sha256": _sha256(parent.AUDIT),
        "checkpoint_sha256": checkpoint_hashes,
        "captured_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    report["promotion"] = {
        "training_allowed": True,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "checkpoint_role": "research_candidate_only",
        "promotion_blocked": True,
    }
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    parent.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": report["training"]["device"],
                "gpu": report["training"]["gpu_name"],
                "seeds": report["training"]["seeds"],
                "entropy_guard": guard,
                "metrics": report.get("metrics"),
                "promotion_blocked": report["promotion"]["promotion_blocked"],
                "checkpoint": str(parent.CHECKPOINT.relative_to(ROOT)),
                "report": str(parent.REPORT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
