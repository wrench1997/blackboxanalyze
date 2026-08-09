"""Run a bounded PG-380 abstract reasoning-SFT candidate.

This runner reuses the shared PG-370 multi-task MoE wiring, but its input is
the isolated PG-380 synthetic abstract matrix.  The declared vocabulary is an
ontology manifest, not a scan of holdout examples.  The run can therefore
exercise ASK/repair/negative/replay heads without turning the result into a
capability, payload, or vulnerability claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg295_causal_moe import CausalMoEConfig  # noqa: E402
from scripts.plan_pg369_multitask_moe_candidate import SLOTS, derive_multitask_labels  # noqa: E402
from scripts.run_pg370_multitask_moe_candidate import run_candidate  # noqa: E402

SCHEMA_VERSION = "pg380-abstract-reasoning-sft-candidate-v1"
DEFAULT_DATASET = ROOT / "research/pg380_abstract_adversarial_reasoning_dataset_v1.json"
SEEDS = (38001, 38002, 38003)
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("PG-380 dataset root must be an object")
    return value


def _safe_rows(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset.get("status") != "abstract_adversarial_candidate_only":
        raise ValueError("PG-380 dataset status is not abstract candidate-only")
    if dataset.get("safety", {}).get("raw_payload_in_context") is not False:
        raise ValueError("PG-380 raw payload gate is not closed")
    rows: list[dict[str, Any]] = []
    for raw in dataset.get("records", []):
        if not isinstance(raw, dict):
            raise ValueError("PG-380 record must be an object")
        if raw.get("raw_payload_stored") is not False or raw.get("raw_response_body_stored") is not False or raw.get("oracle_answer_in_context") is not False:
            raise ValueError("PG-380 record raw/evaluator gate is open")
        if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            raise ValueError("PG-380 record firewall is incomplete")
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        if not isinstance(context, list) or not isinstance(target, list):
            raise ValueError("PG-380 context/target must be lists")
        if derive_multitask_labels([str(token) for token in target]) is None:
            raise ValueError("PG-380 target does not satisfy 13-slot contract")
        tokens = [str(token) for token in [*context, *target]]
        forbidden = ("http://", "https://", "javascript:", "<script", "document.cookie", "route_literal=", "oracle_answer=")
        if any(any(marker in token.casefold() for marker in forbidden) for token in tokens):
            raise ValueError("PG-380 raw/evaluator marker reached model row")
        rows.append({"context_tokens": tokens[: len(context)], "target_tokens": tokens[len(context) :], "split": str(raw.get("split"))})
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "implementation_holdout"]
    if not train or not holdout:
        raise ValueError("PG-380 train/holdout split is empty")
    return train, holdout


def _declared_manifest(dataset: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    vocabulary = dataset.get("vocabulary")
    if not isinstance(vocabulary, dict) or vocabulary.get("scope") != "declared_abstract_ontology":
        raise ValueError("PG-380 declared abstract ontology vocabulary is required")
    context_tokens = vocabulary.get("context_tokens")
    target_tokens = vocabulary.get("target_tokens")
    if not isinstance(context_tokens, list) or not isinstance(target_tokens, list):
        raise ValueError("PG-380 vocabulary manifest is incomplete")
    declared = [str(token) for token in [*context_tokens, *target_tokens]]
    slot_values: dict[str, set[str]] = {key: set() for key in SLOTS}
    for token in target_tokens:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in slot_values:
            slot_values[key].add(value)
    if any(not values for values in slot_values.values()):
        raise ValueError("PG-380 declared slot inventory is incomplete")
    return declared, {key: sorted(values) for key, values in slot_values.items()}


def _execution_gate(device: str) -> dict[str, Any]:
    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    weekend = now.weekday() >= 5
    allowed_window = weekend or (8 <= now.hour < 18)
    explicit = os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") == "1"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return {
        "timestamp": now.isoformat(),
        "weekend": weekend,
        "allowed_time_window": allowed_window,
        "explicit_remote_flag": explicit,
        "cuda_visible_devices": visible,
        "device": device,
        "passed": device == "cpu" or (weekend and allowed_window and explicit and visible == "0"),
    }


def run_candidate_report(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    device: str = "cpu",
    epochs: int = 1,
    microbatch: int = 4,
    d_model: int = 32,
    n_layers: int = 1,
    experts: int = 2,
    expert_hidden: int = 64,
    max_length: int = 128,
    checkpoint_dir: Path | None = None,
    row_limit: int | None = None,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    train, holdout = _safe_rows(dataset)
    if row_limit is not None:
        limit = max(1, int(row_limit))
        train, holdout = train[:limit], holdout[:limit]
    declared, slot_values = _declared_manifest(dataset)
    gate = _execution_gate(device)
    if device != "cpu" and not gate["passed"]:
        raise RuntimeError("PG-380 remote A800 gate failed")
    config = CausalMoEConfig(
        d_model=int(d_model),
        n_heads=4 if int(d_model) < 256 else 8,
        n_layers=int(n_layers),
        experts=int(experts),
        expert_hidden=int(expert_hidden),
        max_length=int(max_length),
        top_k=min(2, int(experts)),
    )
    result = run_candidate(
        train_rows=train,
        holdout_rows=holdout,
        seeds=SEEDS,
        device=device,
        epochs=int(epochs),
        microbatch=int(microbatch),
        config=config,
        declared_vocabulary=declared,
        declared_slot_values=slot_values,
        checkpoint_dir=checkpoint_dir,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_reasoning_sft_candidate_only" if device != "cpu" else "cpu_smoke_candidate_only",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha_file(dataset_path),
        "dataset_internal_sha256": dataset.get("dataset_sha256"),
        "data": {"train_rows": len(train), "holdout_rows": len(holdout), "vocabulary_scope": "declared_abstract_ontology", "raw_rows_loaded": False, "target_slots": len(SLOTS)},
        "execution_gate": gate,
        "training": result.get("training", {}),
        "candidates": result.get("candidates", []),
        "worst_seed": result.get("worst_seed", {}),
        "scientific_gate": {**dict(result.get("scientific_gate", {})), "abstract_reasoning_only": True, "model_selected_wire_replay": False},
        "promotion": dict(PROMOTION),
        "interpretation": "PG-380 只验证抽象 ASK/repair/negative/replay 学习；没有真实 payload、target contact 或漏洞能力晋级。",
    }
    report["report_sha256"] = _sha_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg380_abstract_reasoning_sft_candidate_v1.json")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--remote-candidate", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-hidden", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/pg380-abstract-reasoning-sft")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    if not args.cpu_smoke and not args.remote_candidate:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "plan_only",
            "dataset": str(args.dataset),
            "dataset_sha256": _sha_file(args.dataset),
            "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False},
            "promotion": dict(PROMOTION),
        }
    else:
        device = "cuda:0" if args.remote_candidate else "cpu"
        if args.cpu_smoke:
            report = run_candidate_report(dataset_path=args.dataset, device=device, epochs=min(args.epochs, 1), microbatch=min(args.microbatch, 2), d_model=min(args.d_model, 32), n_layers=min(args.n_layers, 1), experts=min(args.experts, 2), expert_hidden=min(args.expert_hidden, 64), max_length=args.max_length, row_limit=args.row_limit or 32, checkpoint_dir=None)
        else:
            report = run_candidate_report(dataset_path=args.dataset, device=device, epochs=args.epochs, microbatch=args.microbatch, d_model=args.d_model, n_layers=args.n_layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=args.max_length, row_limit=args.row_limit, checkpoint_dir=args.checkpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "dataset_sha256": report.get("dataset_sha256"), "worst_seed": report.get("worst_seed"), "promotion": report.get("promotion")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_candidate_report"]
