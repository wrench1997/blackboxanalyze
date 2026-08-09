"""Context-only PG-331 representation pretrain candidate for A800 GPU0.

This is not capability training: it consumes only already-de-identified
``context_tokens`` and never opens target/evaluator/payload/response fields.
Diagnostic rows may be used when their context firewall and full field manifest
pass, even when strict typed-evaluator eligibility is incomplete.  Information
audit status is recorded as a promotion blocker, never silently bypassed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, train_causal_moe  # noqa: E402

SCHEMA_VERSION = "pg331-a800-representation-smoke-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (33121, 33122, 33123)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 1
# ``response_body_length``/``response_body_shape`` are legitimate ontology
# projections.  Reject only literal/raw side-channel token keys.
FORBIDDEN_PREFIXES = ("raw_", "payload=", "payload_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "family=", "route_literal=")


def _forbidden_context(tokens: list[Any]) -> bool:
    return any(str(token).casefold().startswith(FORBIDDEN_PREFIXES) for token in tokens)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weekend(now: datetime) -> bool:
    return (now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)).weekday() >= 5


def _context_rows(dataset: Mapping[str, Any] | None, *, train_split: str = "train") -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("records"), list): return [], ["dataset_missing_or_invalid"], {"train": 0, "implementation_holdout": 0}
    accepted: list[dict[str, Any]] = []; failures: list[str] = []
    counts = {"train": 0, "implementation_holdout": 0}
    for index, raw in enumerate(dataset["records"]):
        if not isinstance(raw, Mapping): failures.append(f"row_{index}_not_mapping"); continue
        split = str(raw.get("split", ""))
        if split in counts: counts[split] += 1
        # Do not inspect context/target fields for any holdout row.  Split is
        # provenance metadata, not model input, and is only counted here.
        if split != train_split: continue
        context = raw.get("context_tokens"); manifest = raw.get("field_capture_manifest")
        flags = (raw.get("raw_payload_stored"), raw.get("raw_response_body_stored"), raw.get("oracle_answer_in_context"))
        firewall = raw.get("context_firewall")
        if not isinstance(context, list) or len(context) < 2: failures.append(f"row_{index}_context_missing"); continue
        if not isinstance(manifest, Mapping) or len(manifest) != 7: failures.append(f"row_{index}_field_manifest_missing"); continue
        firewall_count = firewall.get("forbidden_token_count") if isinstance(firewall, Mapping) else None
        if not isinstance(firewall, Mapping) or firewall_count != 0 or firewall.get("sidecars_off_context") is not True or any(flag is not False for flag in flags) or _forbidden_context(context): failures.append(f"row_{index}_context_firewall"); continue
        statuses = [str(status) for fields in manifest.values() if isinstance(fields, Mapping) for status in fields.values()]
        if not statuses or any(status not in {"observed", "absent", "not_observed", "unknown"} for status in statuses): failures.append(f"row_{index}_field_manifest_invalid"); continue
        # target_tokens are intentionally neither read nor copied.
        accepted.append({"context_tokens": [str(token) for token in context], "target_tokens": []})
    if not accepted: failures.append("no_raw_context_manifest_valid_rows")
    return accepted, sorted(set(failures)), counts


def _predictive_metrics(model: Any, rows: list[dict[str, Any]], vocabulary: Mapping[str, int], device: Any) -> dict[str, Any]:
    """Mean loss/entropy over every valid next-token, never only the tail."""
    import torch
    losses: list[float] = []; entropies: list[float] = []; total = 0
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([[int(vocabulary[token]) for token in row["context_tokens"]]], device=device)
            logits, _ = model(ids[:, :-1])
            labels = ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            losses.extend((-log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)).reshape(-1).detach().cpu().tolist())
            entropies.extend((-(probs * log_probs).sum(-1)).reshape(-1).detach().cpu().tolist())
            total += int(labels.numel())
    return {"next_token_count": total, "mean_next_token_loss": round(sum(losses) / max(len(losses), 1), 6) if losses else None, "mean_predictive_entropy_nats": round(sum(entropies) / max(len(entropies), 1), 6) if entropies else None}


def _required_context_window(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Measure both splits before optimization; never truncate holdout eval."""
    train_max = max((len(row["context_tokens"]) for row in train_rows), default=0)
    holdout_max = max((len(row["context_tokens"]) for row in holdout_rows), default=0)
    return {"train_context_max_length": train_max, "holdout_context_max_length": holdout_max, "required_max_length": max(train_max, holdout_max)}


def evaluate_representation_gate(*, now: datetime, env: Mapping[str, str], dataset: Mapping[str, Any] | None, information_audit: Mapping[str, Any] | None, vocabulary: Mapping[str, Any] | None, device: Mapping[str, Any], locks: Mapping[str, str], train_split: str = "train") -> dict[str, Any]:
    rows, row_failures, split_counts = _context_rows(dataset, train_split=train_split)
    vocab_tokens = list((vocabulary or {}).get("context_tokens") or []) if isinstance(vocabulary, Mapping) else []
    row_tokens = {token for row in rows for token in row["context_tokens"]}
    unknown_count = len(row_tokens - {str(token) for token in vocab_tokens})
    checks = {"weekend_remote_lane": _weekend(now), "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1", "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0", "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and device.get("visible_device_count") == 1 and device.get("current_device") == 0 and "A800" in str(device.get("name", ""))), "raw_context_manifest_rows": bool(rows) and not row_failures, "context_vocabulary_locked": bool(vocab_tokens) and unknown_count == 0, "data_code_vocab_rules_hashes_locked": all(len(str(value)) == 64 for value in locks.values())}
    failures = [key for key, ok in checks.items() if not ok] + row_failures
    return {"schema_version": SCHEMA_VERSION, "status": "ready_representation_pretrain_candidate" if not failures else "blocked", "checks": checks, "failures": sorted(set(failures)), "representation_training_allowed": not failures, "information_gate_status": str((information_audit or {}).get("status", "missing")), "information_promotion_gate_passed": bool((information_audit or {}).get("status") == "passed"), "train_split": train_split, "split_counts": split_counts, "source_implementation_holdout_recorded": split_counts["implementation_holdout"] > 0, "context_row_count": len(rows), "unknown_context_token_count": unknown_count, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-331 context-only A800 representation candidate smoke")
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--information-audit", type=Path, required=True); parser.add_argument("--vocabulary", type=Path, required=True); parser.add_argument("--rules", type=Path, default=ROOT / "research" / "improvement_rules.json"); parser.add_argument("--report", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--train-split", default="train"); parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE); parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS); parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, info, vocab = load(args.dataset), load(args.information_audit), load(args.vocabulary)
    locks = {"dataset": _sha(args.dataset), "information_audit": _sha(args.information_audit), "vocabulary": _sha(args.vocabulary), "rules": _sha(args.rules), "script": _sha(Path(__file__)), "model": _sha(ROOT / "app" / "pg295_causal_moe.py")}
    import torch
    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = evaluate_representation_gate(now=datetime.now(TZ), env=os.environ, dataset=dataset, information_audit=info, vocabulary=vocab, device=device_info, locks=locks, train_split=args.train_split)
    if not gate["representation_training_allowed"]: raise RuntimeError("PG-331 representation candidate gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32: raise ValueError("learning rate/epochs outside conservative smoke bounds")
    rows, _, _ = _context_rows(dataset, train_split=args.train_split); holdout_rows, holdout_failures, holdout_counts = _context_rows(dataset, train_split="implementation_holdout"); context_capacity = _required_context_window(rows, holdout_rows); ordered = [PAD, UNK] + [str(token) for token in list(vocab.get("context_tokens") or []) if str(token) not in {PAD, UNK}]; vocabulary_map = {token: index for index, token in enumerate(dict.fromkeys(ordered))}
    config = CausalMoEConfig(d_model=128, n_layers=2, experts=2, expert_hidden=256, max_length=context_capacity["required_max_length"])
    device = torch.device("cuda:0"); candidates = []
    for seed in SEEDS:
        model = train_causal_moe(rows, vocabulary_map, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate)
        train_metrics = _predictive_metrics(model, rows, vocabulary_map, device)
        # Holdout is read only after the optimizer has completed; it cannot
        # influence vocabulary construction or any parameter update.
        holdout_tokens = {token for row in holdout_rows for token in row["context_tokens"]}
        unknown_holdout = sorted(holdout_tokens - set(vocabulary_map))
        holdout_metrics = _predictive_metrics(model, holdout_rows, vocabulary_map, device) if holdout_rows and not unknown_holdout else {"next_token_count": 0, "mean_next_token_loss": None, "mean_predictive_entropy_nats": None}
        candidates.append({"seed": seed, "train": train_metrics, "heldout": {**holdout_metrics, "context_row_count": len(holdout_rows), "unknown_context_token_count": len(unknown_holdout), "failures": holdout_failures + (["holdout_token_not_in_fixed_vocabulary"] if unknown_holdout else []), "split_counts": holdout_counts}, "forgetting": "not_applicable_no_prior_candidate"})
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"schema_version": SCHEMA_VERSION, "representation_pretrain_candidate_only": True, "vocabulary": vocabulary_map, "promotion": gate["promotion"]}, args.checkpoint)
    holdout_unknown = any(int(item["heldout"]["unknown_context_token_count"]) > 0 for item in candidates)
    report = {"schema_version": SCHEMA_VERSION, "status": "representation_pretrain_candidate_only" if not holdout_unknown else "representation_pretrain_candidate_holdout_blocked", "gate": gate, "locks": locks, "training": {"device": "cuda:0", "seeds": list(SEEDS), "context_only": True, "target_tokens_read": False, "learning_rate": args.learning_rate, "epochs": args.epochs}, "context_capacity_requirement": context_capacity, "loss": candidates, "entropy": candidates, "forgetting": "not_applicable_no_prior_candidate", "holdout_vocabulary_gate_passed": not holdout_unknown, "promotion": gate["promotion"]}; report["report_sha256"] = _json_hash(report); args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report if args.json else {"status": report["status"]}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
