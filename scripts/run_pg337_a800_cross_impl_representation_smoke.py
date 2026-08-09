"""Context-only PG-337 cross-implementation representation smoke on A800 GPU0.

The runner reads context tokens from the PG-337 train split and the
implementation holdout only.  It never reads target tokens, evaluator
sidecars, wire values or response bodies.  This is a candidate representation
pretrain, not Rule-IR capability/SFT/RL promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, train_causal_moe  # noqa: E402
from scripts.run_pg336_a800_real_failure_representation_smoke import _context_rows, _metrics, _weekend  # noqa: E402


SCHEMA_VERSION = "pg337-a800-cross-impl-representation-smoke-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (33701, 33702, 33703)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _gate(*, dataset: Mapping[str, Any], audit: Mapping[str, Any], vocabulary: Mapping[str, Any], env: Mapping[str, str], device: Mapping[str, Any], locks: Mapping[str, str], train_rows: list[dict[str, Any]], train_failures: list[str], holdout_rows: list[dict[str, Any]], holdout_failures: list[str], now: datetime) -> dict[str, Any]:
    vocab_tokens = {str(token) for token in list(vocabulary.get("context_tokens") or [])}
    unknown_train = sorted({token for row in train_rows for token in row["context_tokens"]} - vocab_tokens)
    unknown_holdout = sorted({token for row in holdout_rows for token in row["context_tokens"]} - vocab_tokens)
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and device.get("visible_device_count") == 1 and device.get("current_device") == 0 and "A800" in str(device.get("name", ""))),
        "train_context_rows_valid": bool(train_rows) and not train_failures,
        "implementation_holdout_context_rows_valid": bool(holdout_rows) and not holdout_failures,
        "context_vocabulary_locked": bool(vocab_tokens) and not unknown_train and not unknown_holdout,
        "audit_is_diagnostic_only": str(audit.get("status")) == "diagnostic_only",
        "independent_implementation_holdout": dataset.get("source", {}).get("independent_implementation_holdout") is True,
        "data_code_vocab_rules_hashes_locked": all(len(str(value)) == 64 for value in locks.values()),
    }
    failures = [key for key, value in checks.items() if not value] + train_failures + holdout_failures
    if unknown_train or unknown_holdout:
        failures.append("context_vocabulary_unknown_token")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_representation_pretrain_candidate" if not failures else "blocked",
        "checks": checks,
        "failures": sorted(set(failures)),
        "representation_training_allowed": not failures,
        "information_gate_status": str(audit.get("status", "missing")),
        "information_promotion_gate_passed": False,
        "train_split": "train",
        "holdout_split": "implementation_holdout",
        "split_counts": {"train": len(train_rows), "implementation_holdout": len(holdout_rows)},
        "context_row_count": len(train_rows),
        "source_implementation_holdout_recorded": True,
        "unknown_context_token_count": len(unknown_train) + len(unknown_holdout),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-337 cross-implementation context-only A800 candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--information-audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit, vocabulary = load(args.dataset), load(args.information_audit), load(args.vocabulary)
    locks = {"dataset": _sha(args.dataset), "information_audit": _sha(args.information_audit), "vocabulary": _sha(args.vocabulary), "rules": _sha(args.rules), "script": _sha(Path(__file__)), "model": _sha(ROOT / "app" / "pg295_causal_moe.py")}
    train_rows, train_failures = _context_rows(dataset, "train")
    holdout_rows, holdout_failures = _context_rows(dataset, "implementation_holdout")
    import torch

    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = _gate(dataset=dataset, audit=audit, vocabulary=vocabulary, env=os.environ, device=device_info, locks=locks, train_rows=train_rows, train_failures=train_failures, holdout_rows=holdout_rows, holdout_failures=holdout_failures, now=datetime.now(TZ))
    if not gate["representation_training_allowed"]:
        raise RuntimeError("PG-337 representation candidate gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32:
        raise ValueError("learning rate/epochs outside conservative smoke bounds")
    vocab_tokens = [PAD, UNK] + [str(token) for token in list(vocabulary.get("context_tokens") or []) if str(token) not in {PAD, UNK}]
    vocabulary_map = {token: index for index, token in enumerate(dict.fromkeys(vocab_tokens))}
    max_length = max([len(row["context_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    config = CausalMoEConfig(d_model=128, n_layers=2, experts=2, expert_hidden=256, max_length=max_length)
    device = torch.device("cuda:0")
    candidates: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = train_causal_moe(train_rows, vocabulary_map, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate)
        candidates.append({"seed": seed, "train": _metrics(model, train_rows, vocabulary_map, device), "heldout": {**_metrics(model, holdout_rows, vocabulary_map, device), "context_row_count": len(holdout_rows), "unknown_context_token_count": 0, "split": "implementation_holdout"}, "forgetting": "not_applicable_no_prior_candidate"})
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "representation_pretrain_candidate_only": True, "vocabulary": vocabulary_map, "promotion": gate["promotion"]}, args.checkpoint)
    report = {"schema_version": SCHEMA_VERSION, "status": "representation_pretrain_candidate_only", "gate": gate, "locks": locks, "training": {"device": "cuda:0", "seeds": list(SEEDS), "context_only": True, "target_tokens_read": False, "learning_rate": args.learning_rate, "epochs": args.epochs}, "context_capacity_requirement": {"train_max": max((len(row["context_tokens"]) for row in train_rows), default=0), "implementation_holdout_max": max((len(row["context_tokens"]) for row in holdout_rows), default=0), "required_max_length": max_length}, "loss": candidates, "forgetting": "not_applicable_no_prior_candidate", "promotion": gate["promotion"]}
    report["report_sha256"] = _json_hash(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "train_rows": len(train_rows), "holdout_rows": len(holdout_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
