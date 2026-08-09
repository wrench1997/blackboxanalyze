"""Short A800 candidate smoke for the PG-367 WAF process trajectory.

This is a next-token representation/process experiment, not a vulnerability
claim.  It consumes only abstract context/target tokens from the diagnostic
dataset.  No evaluator projection, raw request, response body, URL, or
payload literal is read by the optimizer.  The candidate is permanently
non-promotable and is intended to use the authorized weekend A800 GPU0 lane
without pretending that a single synthetic implementation is a holdout.
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
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, TARGET_EOS, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, generate_target, train_causal_moe  # noqa: E402
from app.pg367_waf_staircase import ALLOWED_ROLES  # noqa: E402

SCHEMA_VERSION = "pg367-a800-process-candidate-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (36701, 36702, 36703)
RAW_PREFIXES = ("payload=", "raw_", "response_body=", "wire=", "oracle=", "evaluator=", "route_literal=")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(dataset: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping) or raw.get("split") != split:
            continue
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        if not isinstance(context, list) or not isinstance(target, list) or len(context) < 2 or len(target) < 2:
            failures.append(f"row_{index}:token_shape")
            continue
        if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:context_firewall")
            continue
        if any(raw.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}:raw_flag")
            continue
        tokens = [str(token) for token in [*context, *target]]
        if any(token.casefold().startswith(RAW_PREFIXES) for token in tokens):
            failures.append(f"row_{index}:raw_token")
            continue
        rows.append({"context_tokens": [str(token) for token in context], "target_tokens": [str(token) for token in target]})
    return rows, sorted(set(failures))


def _vocabulary(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in train_rows:
        tokens.update(str(token) for token in [*(row.get("context_tokens") or []), *(row.get("target_tokens") or [])])
    ordered = [PAD, UNK] + sorted(tokens - {PAD, UNK})
    return {token: index for index, token in enumerate(dict.fromkeys(ordered))}


def _entropy(model: Any, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any, *, max_length: int) -> tuple[float, int]:
    import torch
    from torch.nn import functional as F
    from app.pg295_causal_moe import _batch
    values: list[float] = []
    unknown = int(vocabulary[UNK])
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            ids, valid = _batch(batch, vocabulary, device, max_length=max_length)
            # _batch uses UNK for fixed-vocabulary holdout values by design.
            logits, _ = model(ids[:, :-1], valid_mask=valid[:, :-1])
            probs = F.softmax(logits, dim=-1)
            valid_next = valid[:, 1:]
            values.extend((-(probs * probs.clamp_min(1e-12).log()).sum(-1)[valid_next]).detach().cpu().tolist())
    _ = unknown
    return (round(sum(values) / max(len(values), 1), 6) if values else 0.0, len(values))


def _target_exact(model: Any, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any) -> dict[str, Any]:
    exact = 0
    unknown = 0
    for row in rows:
        target = [str(token) for token in row["target_tokens"]]
        unknown += sum(token not in vocabulary for token in [*row["context_tokens"], *target])
        decoded = generate_target(model, row["context_tokens"], len(target), vocabulary, device)
        exact += int(decoded == target)
    return {"sequence_exact": round(exact / max(len(rows), 1), 6), "rows": len(rows), "unknown_token_count": unknown}


def _weekend(now: datetime) -> bool:
    local = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    return local.weekday() >= 5


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-367 A800 abstract WAF process candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-367 A800 lane requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-367 A800 lane requires CUDA_VISIBLE_DEVICES=0")
    now = datetime.now(TZ)
    if not _weekend(now):
        raise RuntimeError("PG-367 candidate is weekend A800 lane only")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    train_rows, train_failures = _load_rows(dataset, "train")
    holdout_rows, holdout_failures = _load_rows(dataset, "implementation_holdout")
    if not train_rows or not holdout_rows or train_failures or holdout_failures:
        raise RuntimeError("PG-367 token rows invalid: " + ",".join([*train_failures, *holdout_failures]))
    if str(audit.get("status")) != "passed_diagnostic_only":
        raise RuntimeError("PG-367 audit is not passed_diagnostic_only")
    if not 1 <= args.epochs <= 16 or not 0 < args.learning_rate <= 0.01 or not 1 <= args.batch_size <= 128:
        raise ValueError("training bounds invalid")
    import torch
    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    checks = {"cuda": device_info["cuda_available"], "one_visible_gpu": device_info["visible_device_count"] == 1, "gpu0": device_info["current_device"] == 0, "a800": "A800" in str(device_info["name"]), "weekend": _weekend(now), "raw_free": not train_failures and not holdout_failures, "diagnostic_only": True}
    if not all(checks.values()):
        raise RuntimeError("PG-367 A800 device gate blocked: " + ",".join(key for key, value in checks.items() if not value))
    vocabulary = _vocabulary(train_rows)
    max_length = max(len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows])
    if args.d_model <= 0 or args.d_model % 4 or args.n_layers <= 0 or args.experts <= 0 or args.expert_hidden <= 0:
        raise ValueError("model dimensions invalid")
    config = CausalMoEConfig(d_model=args.d_model, n_heads=4, n_layers=args.n_layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=max_length)
    device = torch.device("cuda:0")
    candidates: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    for seed in SEEDS:
        model = train_causal_moe(train_rows, vocabulary, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)
        entropy, entropy_tokens = _entropy(model, holdout_rows, vocabulary, device, max_length=max_length)
        metrics = _target_exact(model, holdout_rows, vocabulary, device)
        candidates.append({"seed": seed, "holdout": metrics, "holdout_predictive_entropy_nats": entropy, "entropy_token_count": entropy_tokens})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    worst_exact = min(float(item["holdout"]["sequence_exact"]) for item in candidates)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_candidate_only",
        "locks": {"dataset": _sha_file(args.dataset), "audit": _sha_file(args.audit), "rules": _sha_file(args.rules), "runner": _sha_file(Path(__file__)), "model": _sha_file(ROOT / "app" / "pg295_causal_moe.py")},
        "training": {"device": "cuda:0", "gpu": device_info["name"], "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "d_model": args.d_model, "n_layers": args.n_layers, "experts": args.experts, "expert_hidden": args.expert_hidden, "context_only_raw_free": True, "target_tokens_used_as_labels_only": True, "required_context_window": max_length},
        "checks": checks,
        "counts": {"train_rows": len(train_rows), "implementation_holdout_rows": len(holdout_rows), "vocabulary_size": len(vocabulary)},
        "candidates": candidates,
        "worst_seed": {"sequence_exact_min": worst_exact, "negative_oracle_metrics": "not_run_evaluator_sidecar_only", "fresh_replay": "not_run"},
        "scientific_gate": {"single_synthetic_implementation": True, "typed_live_replay_with_model_selected_wire": False, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "A800 只完成抽象 WAF 过程 next-token candidate smoke；无 typed evaluator/跨实现/负对照能力证明，不能声称会生成任意原始 payload。",
    }
    report["report_sha256"] = _sha(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "config": config.__dict__, "vocabulary": vocabulary, "states": states, "promotion": report["promotion"]}, args.checkpoint)
    print(json.dumps(report if args.json else {"status": report["status"], "worst_seed": report["worst_seed"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
