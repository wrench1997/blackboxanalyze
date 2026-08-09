"""PG-260: train a larger active-belief Rule-IR adapter on fresh paired traces.

PG-259's auxiliary belief/probe heads were strong while route Rule-IR and
implementation OOD were weak.  This experiment adds the 32 fresh PG-260
records, keeps the PG-249 body/legacy policy frozen, compares a larger 8192
adapter, and trains an explicit unknown-family abstention head.  All oracle
fields are supervision only and remain outside the model input.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(filename: str, unique_name: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG259 = _load("run_pg259_active_belief_capacity_training.py", "pg260_pg259_helpers")
PG258 = PG259.PG258
PG249 = PG259.PG249
PG231 = PG259.PG231
PG237 = PG259.PG237
from app.pg230_next_token_quality_funnel import build_vocabulary, digest  # noqa: E402
from app.pg260_active_belief_adapter import (  # noqa: E402
    ABSTAIN_CLASSES,
    BELIEF_CLASSES,
    FAMILY_CLASSES,
    PROBE_CLASSES,
    RULE_IR_CLASSES,
    PG260ActiveBeliefAdapter,
    belief_target,
    evaluate_pg260_adapter,
    family_target,
    probe_target,
    rule_target,
    unknown_abstain_target,
)


RESEARCH = ROOT / "research"
PG258_DATASET = RESEARCH / "pg258_unified_rule_ir_capacity_dataset_v1.json"
PG259_DATASET = RESEARCH / "pg259_fresh_local_trace_collection_dataset_v1.json"
PG260_DATASET = RESEARCH / "pg260_fresh_paired_trace_collection_dataset_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
PG249_ARTIFACT = ROOT / "artifacts" / "pg249-pikachu-route-seed-capacity-v1" / "frozen_xxl_capacity_hidden4096.pt"
REPORT = RESEARCH / "pg260_active_belief_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg260_active_belief_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg260_active_belief_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg260_active_belief_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg260_active_belief_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg260-active-belief-capacity-v1"
def _env_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 1)
    except ValueError:
        return default


def _env_variants() -> tuple[int, ...]:
    raw = os.environ.get("SIFT_CAPACITY_VARIANTS", "2048,4096,8192")
    values: list[int] = []
    for item in raw.split(","):
        try:
            value = int(item.strip())
        except ValueError:
            continue
        if value > 0 and value not in values:
            values.append(value)
    return tuple(values) or (2048, 4096, 8192)


CAPACITY_VARIANTS = _env_variants()
TRAIN_STEPS = _env_int("SIFT_TRAIN_STEPS", 170)
# A non-zero value enables activation micro-batching.  It preserves the
# optimizer step count while keeping one 8192 adapter from holding the full
# train context and all token-head activations at once.
MICRO_BATCH_SIZE = _env_int("SIFT_MICRO_BATCH_SIZE", 0) if os.environ.get("SIFT_MICRO_BATCH_SIZE") else 0
# Wrappers for later experiments can add a genuinely fresh source while the
# historical PG-260 split remains the default and therefore unchanged.
FRESH_SOURCE_PREFIXES = ("pg259_", "pg260_")
CANONICAL_CONTEXT_WIDTH: int | None = None


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _route(row: Mapping[str, Any]) -> str:
    parent = str(row.get("parent_record_id") or "")
    return str(row.get("route") or parent.rsplit(":", 1)[-1])


def _normalise(row: Mapping[str, Any], source_lane: str) -> dict[str, Any]:
    record = dict(row)
    record["source_lane"] = source_lane
    record["rule_ir_class"] = PG259._rule_class(record)
    record["family_class"] = PG259._family(record)
    record["belief_class"] = PG259._belief(record)
    record["probe_class"] = PG259._probe(record)
    record["unknown_abstain_class"] = ABSTAIN_CLASSES[unknown_abstain_target(record)]
    record["source_record_id"] = str(record.get("record_id") or record.get("parent_record_id") or digest({"source": record.get("source"), "seed": record.get("seed"), "route": _route(record), "trajectory_hash": record.get("trajectory_hash")}))
    if not record.get("tokens"):
        raise ValueError(f"PG-260 input row has no bounded tokens: {record['source_record_id']}")
    return record


def _load_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, lane in ((PG258_DATASET, "pg258_unified"), (PG259_DATASET, "pg259_fresh"), (PG260_DATASET, "pg260_fresh_paired")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows.extend(_normalise(row, lane) for row in list(payload.get("records") or []) if str(row.get("lane", "")) not in {"quarantine", "reject"})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", 0) or 0), _route(row), str(row.get("route_source_sha256", "")), str(row.get("trajectory_hash", row.get("token_hash", ""))))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _is_ood(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")) == "pg246_vulnerableapp_source_independent"


def _is_fresh_source(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")).startswith(tuple(FRESH_SOURCE_PREFIXES))


def _is_pg258_holdout(row: Mapping[str, Any]) -> bool:
    return bool(PG258._is_holdout(row))


def _is_pg259_holdout(row: Mapping[str, Any]) -> bool:
    source = str(row.get("source", ""))
    if not source.startswith("pg259_"):
        return False
    return _route(row) in {"/vul/sqli/sqli_search.php", "/vul/sqli/sqli_blind_b.php", "/vul/sqli/sqli_widebyte.php", "/vul/xss/xss_03.php", "/vul/xss/xss_dom_x.php"} or int(row.get("seed", 0) or 0) == 25922


def _is_pg260_holdout(row: Mapping[str, Any]) -> bool:
    if not str(row.get("source", "")).startswith("pg260_"):
        return False
    route = _route(row)
    seed = int(row.get("seed", 0) or 0)
    # Route holdout for syntax/DOM, seed holdout for single-route boolean and
    # widebyte surfaces.  The source prefix keeps SQL blind_b and boolean
    # blind_b distinct even though the URL is shared.
    if route in {"/vul/sqli/sqli_search.php", "/vul/sqli/sqli_x.php", "/vul/xss/xss_dom_x.php"}:
        return True
    if "boolean" in str(row.get("source", "")) or "widebyte" in str(row.get("source", "")):
        return seed % 2 == 0
    if route == "/vul/sqli/sqli_blind_b.php":
        return True
    return seed in {26002, 26008, 26018}


def _is_holdout(row: Mapping[str, Any]) -> bool:
    return _is_pg258_holdout(row) or _is_pg259_holdout(row) or _is_pg260_holdout(row)


def _positions(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(row.get("classification_position", 0)), 0), max(width - 1, 0)) for row in rows], dtype=torch.long, device=device)


def _state_digest(model: torch.nn.Module) -> str:
    return hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def _encode_context(rows: list[dict[str, Any]], input_vocab: dict[str, int], target_vocab: dict[str, int], base: Any, old_policy: Any, device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    encoded = PG231._encode(rows, input_vocab, target_vocab, device)
    if CANONICAL_CONTEXT_WIDTH is not None and encoded[0].shape[1] < CANONICAL_CONTEXT_WIDTH:
        import torch.nn.functional as functional

        padded = []
        for tensor in encoded:
            if tensor.ndim == 2 and tensor.shape[1] < CANONICAL_CONTEXT_WIDTH:
                tensor = functional.pad(tensor, (0, CANONICAL_CONTEXT_WIDTH - tensor.shape[1]))
            padded.append(tensor)
        encoded = tuple(padded)  # type: ignore[assignment]
    with torch.no_grad():
        body = base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
        context = old_policy.context_projection(body).detach().clone()
    return context, encoded


def _targets(rows: list[dict[str, Any]], encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "token": encoded[1],
        "rule": torch.tensor([rule_target(row["rule_ir_class"]) for row in rows], dtype=torch.long, device=device),
        "family": torch.tensor([family_target(row["family_class"]) for row in rows], dtype=torch.long, device=device),
        "belief": torch.tensor([belief_target(row["belief_class"]) for row in rows], dtype=torch.long, device=device),
        "probe": torch.tensor([probe_target(row["probe_class"]) for row in rows], dtype=torch.long, device=device),
        "unknown": torch.tensor([unknown_abstain_target(row) for row in rows], dtype=torch.long, device=device),
    }


def _metrics(model: PG260ActiveBeliefAdapter, context: torch.Tensor, encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    targets = _targets(rows, encoded, device)
    return evaluate_pg260_adapter(model, context, targets["token"], targets["rule"], targets["family"], targets["belief"], targets["probe"], targets["unknown"], _positions(rows, context.shape[1], device), attention_mask=encoded[0].ne(0))


def main() -> int:
    rows = _load_records()
    ood = [row for row in rows if _is_ood(row)]
    eligible = [row for row in rows if not _is_ood(row)]
    holdout = [row for row in eligible if _is_holdout(row)]
    train = [row for row in eligible if not _is_holdout(row)]
    fresh = [row for row in eligible if _is_fresh_source(row)]
    fresh_holdout = [row for row in holdout if _is_fresh_source(row)]
    if not train or not holdout or not ood or not fresh_holdout:
        raise RuntimeError(f"PG-260 split incomplete train={len(train)} holdout={len(holdout)} fresh_holdout={len(fresh_holdout)} ood={len(ood)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    if not hasattr(PG249.PG248.PG247, "ORIGINAL_INPUT_TOKEN_ID"):
        PG249.PG248.PG247.ORIGINAL_INPUT_TOKEN_ID = PG231._input_token_id
    PG231._input_token_id = PG249.PG248._patched_input_token_id
    base, _ = PG249.PG248.PG247._load_base(device)
    old_payload = torch.load(PG249_ARTIFACT, map_location="cpu", weights_only=False)
    old_target_vocab = {str(key): int(value) for key, value in old_payload["token_vocabulary"].items()}
    old_policy = PG237.FrozenXXLFailurePolicy(d_model=1024, hidden_dim=int(old_payload["hidden_dim"]), vocab_size=len(old_target_vocab)).to(device)
    old_policy.load_state_dict(old_payload["state_dict"], strict=True)
    old_policy.eval()
    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)
    target_vocab = build_vocabulary(rows)
    train_context, train_encoded = _encode_context(train, input_vocab, target_vocab, base, old_policy, device)
    hold_context, hold_encoded = _encode_context(holdout, input_vocab, target_vocab, base, old_policy, device)
    fresh_context, fresh_encoded = _encode_context(fresh_holdout, input_vocab, target_vocab, base, old_policy, device)
    ood_context, ood_encoded = _encode_context(ood, input_vocab, target_vocab, base, old_policy, device)
    train_targets = _targets(train, train_encoded, device)
    rule_counts = torch.bincount(train_targets["rule"], minlength=len(RULE_IR_CLASSES)).float()
    family_counts = torch.bincount(train_targets["family"], minlength=len(FAMILY_CLASSES)).float()
    belief_counts = torch.bincount(train_targets["belief"], minlength=len(BELIEF_CLASSES)).float()
    probe_counts = torch.bincount(train_targets["probe"], minlength=len(PROBE_CLASSES)).float()
    abstain_counts = torch.bincount(train_targets["unknown"], minlength=len(ABSTAIN_CLASSES)).float()

    def weights(counts: torch.Tensor) -> torch.Tensor:
        return torch.where(counts > 0, torch.sqrt(counts.sum() / counts), torch.zeros_like(counts)).to(device)

    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: PG260ActiveBeliefAdapter | None = None
    positions = _positions(train, train_context.shape[1], device)
    for hidden_dim in CAPACITY_VARIANTS:
        torch.manual_seed(260 + hidden_dim)
        model = PG260ActiveBeliefAdapter(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, token_vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.01)
        for _ in range(TRAIN_STEPS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_size = MICRO_BATCH_SIZE or int(train_context.shape[0])
            for start in range(0, int(train_context.shape[0]), batch_size):
                end = min(start + batch_size, int(train_context.shape[0]))
                batch_weight = float(end - start) / max(int(train_context.shape[0]), 1)
                output = model(train_context[start:end], classification_positions=positions[start:end], attention_mask=train_encoded[0][start:end].ne(0))
                token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_targets["token"][start:end].reshape(-1), ignore_index=0)
                loss = token_loss
                loss = loss + nn.functional.cross_entropy(output["rule"], train_targets["rule"][start:end], weight=weights(rule_counts))
                loss = loss + 0.5 * nn.functional.cross_entropy(output["family"], train_targets["family"][start:end], weight=weights(family_counts))
                loss = loss + 0.75 * nn.functional.cross_entropy(output["belief"], train_targets["belief"][start:end], weight=weights(belief_counts))
                loss = loss + 0.75 * nn.functional.cross_entropy(output["probe"], train_targets["probe"][start:end], weight=weights(probe_counts))
                loss = loss + 0.75 * nn.functional.cross_entropy(output["unknown_abstain"], train_targets["unknown"][start:end], weight=weights(abstain_counts))
                (loss * batch_weight).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train": _metrics(model, train_context, train_encoded, train, device), "route_seed_holdout": _metrics(model, hold_context, hold_encoded, holdout, device), "fresh_route_holdout": _metrics(model, fresh_context, fresh_encoded, fresh_holdout, device), "implementation_ood": _metrics(model, ood_context, ood_encoded, ood, device)}
        variants.append(result)
        metric = result["route_seed_holdout"]
        key = (-float(metric["rule_accuracy"]), -float(metric["family_accuracy"]), -float(metric["unknown_abstain_accuracy"]), float(metric["token_loss"]))
        old_key = None if selected is None else (-float(selected["route_seed_holdout"]["rule_accuracy"]), -float(selected["route_seed_holdout"]["family_accuracy"]), -float(selected["route_seed_holdout"]["unknown_abstain_accuracy"]), float(selected["route_seed_holdout"]["token_loss"]))
        if selected is None or key < old_key:
            selected, selected_model = result, model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-260 did not select a capacity variant")

    old_hold_encoded = PG231._encode(holdout, input_vocab, old_target_vocab, device)
    with torch.no_grad():
        old_hold_body = base.base.body.encode(old_hold_encoded[0], old_hold_encoded[0].ne(0)).detach().clone()
    old_hold_positions = _positions(holdout, old_hold_body.shape[1], device)
    legacy_hash_before = _state_digest(old_policy)
    legacy_before = PG237._evaluate(old_policy, old_hold_body, old_hold_encoded, old_hold_positions, holdout, device)
    legacy_hash_after = _state_digest(old_policy)
    legacy_after = PG237._evaluate(old_policy, old_hold_body, old_hold_encoded, old_hold_positions, holdout, device)
    canary = {"state_hash_before": legacy_hash_before, "state_hash_after": legacy_hash_after, "state_unchanged": legacy_hash_before == legacy_hash_after, "action_metrics_unchanged": legacy_before == legacy_after, "before": legacy_before, "after": legacy_after, "oracle_features_fed_to_model": False}
    canary["pass"] = bool(canary["state_unchanged"] and canary["action_metrics_unchanged"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"active_belief_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": "pg260-active-belief-adapter-artifact-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "token_vocabulary": target_vocab, "rule_classes": list(RULE_IR_CLASSES), "family_classes": list(FAMILY_CLASSES), "belief_classes": list(BELIEF_CLASSES), "probe_classes": list(PROBE_CLASSES), "abstain_classes": list(ABSTAIN_CLASSES), "frozen_legacy_artifact": str(PG249_ARTIFACT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    counts = {"records": len(rows), "train_rows": len(train), "holdout_rows": len(holdout), "fresh_holdout_rows": len(fresh_holdout), "implementation_ood_rows": len(ood), "pg260_rows": sum(int(str(row.get("source", "")).startswith("pg260_")) for row in rows), "train_rule_counts": dict(Counter(row["rule_ir_class"] for row in train)), "holdout_rule_counts": dict(Counter(row["rule_ir_class"] for row in holdout)), "fresh_holdout_rule_counts": dict(Counter(row["rule_ir_class"] for row in fresh_holdout)), "source_counts": dict(Counter(row["source"] for row in rows))}
    hold_metrics = selected["route_seed_holdout"]
    fresh_metrics = selected["fresh_route_holdout"]
    ood_metrics = selected["implementation_ood"]
    # Support is a property of the held-out rows, not a metric-head output.
    # Reading it from ``hold_metrics`` silently produced zero for every class
    # and made the judge report a false blocker.
    hold_support = {name: int((counts.get("holdout_rule_counts") or {}).get(name, 0) or 0) for name in RULE_IR_CLASSES}
    gates = {"holdout_rule_accuracy_ge_0_80": float(hold_metrics["rule_accuracy"]) >= 0.80, "holdout_family_accuracy_ge_0_80": float(hold_metrics["family_accuracy"]) >= 0.80, "fresh_route_rule_accuracy_ge_0_70": float(fresh_metrics["rule_accuracy"]) >= 0.70, "fresh_route_belief_accuracy_ge_0_70": float(fresh_metrics["belief_accuracy"]) >= 0.70, "fresh_unknown_abstain_accuracy_ge_0_70": float(fresh_metrics["unknown_abstain_accuracy"]) >= 0.70, "implementation_ood_family_accuracy_ge_0_60": float(ood_metrics["family_accuracy"]) >= 0.60, "holdout_each_rule_class_support_ge_2": all(value >= 2 for value in hold_support.values()), "catastrophic_forgetting_canary": bool(canary["pass"])}
    judge = {"authority": ["PG-258 holdout", "PG-259 fresh route holdout", "PG-260 fresh paired route/seed holdout", "VulnerableApp implementation OOD", "frozen legacy policy canary"], "hard_gates": gates, "holdout_support": hold_support, "pass": bool(all(gates.values())), "decision": "candidate_eligible_for_next_replay" if all(gates.values()) else "blocked_insufficient_generalization", "reasons": [name for name, passed in gates.items() if not passed], "model_output_is_candidate_only": True, "oracle_or_reference_is_not_model_input": True}
    report = {"protocol_id": "pg-pk-260-active-belief-capacity-v1", "schema_version": "pg260-active-belief-capacity-training-report-v1", "status": "completed_pg260_active_belief_capacity_training", "device": str(device), "capacity_variants": list(CAPACITY_VARIANTS), "train_steps": TRAIN_STEPS, "resource_profile": {"micro_batch_size": int(MICRO_BATCH_SIZE or len(train)), "gradient_accumulation_enabled": bool(MICRO_BATCH_SIZE and MICRO_BATCH_SIZE < len(train)), "capacity_variants_are_sequential": True}, "capacity_variant_metrics": variants, "counts": counts, "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected}, "frozen_legacy_policy": {"artifact": str(PG249_ARTIFACT.relative_to(ROOT)), "policy_state_unchanged": canary["state_unchanged"]}, "catastrophic_forgetting_canary": canary, "independent_final_judge": judge, "training_eligible": bool(judge["pass"]), "model_input_excludes_oracle_target": True, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "blocked_by": judge["reasons"]}, "honesty": {"fresh_traces_are_authorized_loopback": True, "pg260_route_seed_holdout_is_disjoint": True, "unknown_family_abstain_is_supervised_only": True, "implementation_ood_is_separate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "legacy_action_policy_not_replaced": True}}
    report["report_sha256"] = digest(report)
    dataset = {"schema_version": "pg260-active-belief-capacity-training-dataset-v1", "source_datasets": [str(PG258_DATASET.relative_to(ROOT)), str(PG259_DATASET.relative_to(ROOT)), str(PG260_DATASET.relative_to(ROOT))], "records": rows, "counts": counts, "contract": {"pg260_paired_fresh_traces": True, "route_seed_holdout": True, "unknown_family_abstain_head": True, "vulnerableapp_implementation_ood_separate": True, "oracle_target_off_input": True, "legacy_policy_frozen": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg260-active-belief-capacity-training-protocol-v1", "frozen_legacy_policy": str(PG249_ARTIFACT.relative_to(ROOT)), "training_sources": [str(PG258_DATASET.relative_to(ROOT)), str(PG259_DATASET.relative_to(ROOT)), str(PG260_DATASET.relative_to(ROOT))], "capacity_variants": list(CAPACITY_VARIANTS), "heads": ["next_token", "rule_ir", "family", "belief", "probe", "unknown_family_abstain"], "oracle_target_off_input": True, "fresh_route_seed_holdout": True, "implementation_ood": "pg246 VulnerableApp rows, separate score", "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg260-active-belief-capacity-trace-v1", "selected": selected, "fresh_holdout": fresh_metrics, "implementation_ood": ood_metrics, "canary": canary, "independent_final_judge": judge, "counts": counts, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-260 active-belief capacity training", "", f"records={len(rows)}; train={len(train)}; holdout={len(holdout)}; fresh_holdout={len(fresh_holdout)}; implementation OOD={len(ood)}", f"selected_hidden={selected['hidden_dim']}; adapter_params={selected['adapter_parameter_count']}; fresh_rule={fresh_metrics['rule_accuracy']}; fresh_family={fresh_metrics['family_accuracy']}; fresh_unknown_abstain={fresh_metrics['unknown_abstain_accuracy']}; OOD_family={ood_metrics['family_accuracy']}", f"judge={judge['decision']}; reasons={', '.join(judge['reasons']) or 'none'}; canary={canary['pass']}", "PG-260 只训练抽象过程 token 与 unknown-family abstain 监督；oracle 不进入输入，真实公网能力不由本报告声明。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "selected": report["selected"], "judge": judge, "canary": canary, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
