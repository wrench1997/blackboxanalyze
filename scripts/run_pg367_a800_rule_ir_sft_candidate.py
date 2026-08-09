"""PG-367 weighted Rule-IR SFT candidate on the authorized A800 GPU0 lane.

This is the next learning stage after the plain causal next-token smoke.  It
still trains only on abstract context/target tokens; critical process slots
receive more loss weight, while all evaluator/raw fields remain unread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg295_causal_moe import CausalMoEConfig, train_causal_moe  # noqa: E402
from scripts.run_pg367_a800_process_candidate import _entropy, _load_rows, _target_exact, _vocabulary  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (36711, 36712, 36713)
CRITICAL_PREFIXES = ("question=", "ask_reason=", "next_action=", "repair_action=", "safe_to_send=", "oracle_ref=", "negative_control_presence_ref=", "probe_variant_ref=")


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights(vocabulary: dict[str, int], *, critical: float, context: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for token in vocabulary:
        if any(str(token).startswith(prefix) for prefix in CRITICAL_PREFIXES):
            result[str(token)] = float(critical)
        elif str(token).startswith("[") or "=" not in str(token):
            result[str(token)] = 1.0
        else:
            result[str(token)] = float(context)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-367 weighted Rule-IR SFT candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--critical-weight", type=float, default=3.0)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-367 SFT requires explicit remote A800 GPU0 flags")
    if datetime.now(TZ).weekday() < 5:
        raise RuntimeError("PG-367 SFT is weekend A800 lane only")
    if args.epochs < 1 or args.epochs > 32 or not 0 < args.learning_rate <= 0.01 or args.critical_weight <= 0 or args.context_weight <= 0:
        raise ValueError("invalid SFT bounds")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    train_rows, train_failures = _load_rows(dataset, "train")
    holdout_rows, holdout_failures = _load_rows(dataset, "implementation_holdout")
    if not train_rows or not holdout_rows or train_failures or holdout_failures:
        raise RuntimeError("invalid abstract rows")
    if str(audit.get("status")) != "passed_diagnostic_only":
        raise RuntimeError("audit gate blocked")
    import torch
    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    if not (device_info["cuda_available"] and device_info["visible_device_count"] == 1 and device_info["current_device"] == 0 and "A800" in device_info["name"]):
        raise RuntimeError("A800 GPU0 gate blocked")
    vocabulary = _vocabulary(train_rows)
    max_length = max(len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows])
    config = CausalMoEConfig(d_model=256, n_heads=4, n_layers=4, experts=4, expert_hidden=512, max_length=max_length)
    token_weights = _weights(vocabulary, critical=args.critical_weight, context=args.context_weight)
    device = torch.device("cuda:0")
    candidates: list[dict[str, object]] = []
    states: dict[str, object] = {}
    for seed in SEEDS:
        model = train_causal_moe(train_rows, vocabulary, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, token_weights=token_weights, normalize_weighted_loss=True, batch_size=args.batch_size)
        entropy, entropy_tokens = _entropy(model, holdout_rows, vocabulary, device, max_length=max_length)
        candidates.append({"seed": seed, "holdout": _target_exact(model, holdout_rows, vocabulary, device), "holdout_predictive_entropy_nats": entropy, "entropy_token_count": entropy_tokens})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    report: dict[str, object] = {
        "schema_version": "pg367-a800-rule-ir-sft-candidate-v1",
        "status": "blocked_candidate_only",
        "locks": {"dataset": _file_sha(args.dataset), "audit": _file_sha(args.audit), "rules": _file_sha(args.rules), "runner": _file_sha(Path(__file__)), "model": _file_sha(ROOT / "app" / "pg295_causal_moe.py")},
        "training": {"device": "cuda:0", "gpu": device_info["name"], "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "d_model": 256, "n_layers": 4, "experts": 4, "expert_hidden": 512, "objective": "weighted_rule_ir_sft", "critical_weight": args.critical_weight, "context_weight": args.context_weight, "normalize_weighted_loss": True, "raw_context_only": True},
        "counts": {"train_rows": len(train_rows), "implementation_holdout_rows": len(holdout_rows), "vocabulary_size": len(vocabulary), "critical_weighted_tokens": sum(any(str(token).startswith(prefix) for prefix in CRITICAL_PREFIXES) for token in vocabulary)},
        "candidates": candidates,
        "worst_seed": {"sequence_exact_min": min(float(item["holdout"]["sequence_exact"]) for item in candidates), "typed_oracle": "not_run", "fresh_replay": "not_run"},
        "scientific_gate": {"single_synthetic_implementation": True, "typed_live_replay_with_model_selected_wire": False, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "加权 Rule-IR SFT 只检验关键过程槽位的学习目标，不等于模型可迁移漏洞或原始 payload 能力。",
    }
    report["report_sha256"] = _sha(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": report["schema_version"], "config": config.__dict__, "vocabulary": vocabulary, "token_weights": token_weights, "states": states, "promotion": report["promotion"]}, args.checkpoint)
    print(json.dumps(report if args.json else {"status": report["status"], "worst_seed": report["worst_seed"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
