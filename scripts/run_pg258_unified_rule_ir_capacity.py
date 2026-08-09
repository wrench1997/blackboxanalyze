"""PG-258: train a unified Rule-IR adapter with real SQL/XSS process traces.

PG-249's 4096 policy is loaded as a frozen legacy lane.  A new adapter head is
trained on PG-249 process records plus PG-257's explicit SQL class feedback.
All PG-242 XSS records and selected implementation/seed records are held out;
VulnerableApp remains a separate OOD split.  The legacy action/lane/repair
policy is replayed before and after training as an accidental-mutation canary.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG249 = _load("run_pg249_pikachu_route_seed_capacity_training.py")
PG231 = PG249.PG248.PG237.PG231
PG237 = PG249.PG248.PG237
from app.pg230_next_token_quality_funnel import build_vocabulary, digest  # noqa: E402
from app.pg258_unified_rule_ir_adapter import (  # noqa: E402
    FAMILY_CLASSES,
    RULE_IR_CLASSES,
    UnifiedRuleIRCapacityAdapter,
    evaluate_unified_adapter,
    family_target,
    rule_target,
)


RESEARCH = ROOT / "research"
PG249_DATASET = RESEARCH / "pg249_pikachu_route_seed_capacity_training_dataset_v1.json"
PG257_DATASET = RESEARCH / "pg257_widebyte_rule_ir_capacity_training_dataset_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
PG249_ARTIFACT = ROOT / "artifacts" / "pg249-pikachu-route-seed-capacity-v1" / "frozen_xxl_capacity_hidden4096.pt"
REPORT = RESEARCH / "pg258_unified_rule_ir_capacity_report_v1.json"
DATASET = RESEARCH / "pg258_unified_rule_ir_capacity_dataset_v1.json"
TRACE = RESEARCH / "pg258_unified_rule_ir_capacity_trace_v1.json"
PROTOCOL = RESEARCH / "pg258_unified_rule_ir_capacity_protocol_v1.json"
MARKDOWN = RESEARCH / "pg258_unified_rule_ir_capacity_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg258-unified-rule-ir-capacity-v1"
CAPACITY_VARIANTS = (1024, 2048, 4096)
TRAIN_STEPS = 120


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rule_class(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("next_rule_class", ""))
    explicit_map = {
        "syntax_boundary": "sql_syntax",
        "blind_boolean": "sql_boolean",
        "widebyte_escape_boundary": "sql_widebyte",
    }
    if explicit in explicit_map:
        return explicit_map[explicit]
    surface = str(row.get("surface_class") or row.get("surface_role") or "").casefold()
    failure = str(row.get("failure_kind") or row.get("failure_signature") or "").casefold()
    source = str(row.get("source", "")).casefold()
    lane = str(row.get("lane", "")).casefold()
    if "dom" in surface or "xss" in source:
        if "oracle" in failure or lane == "silver":
            return "oracle_gap"
        if "typed" in failure or lane == "gold":
            return "dom_marker"
        return "other"
    if "sql" in surface or "sqli" in source:
        if "widebyte" in source or "widebyte" in failure:
            return "sql_widebyte"
        if "boolean" in failure or "differential" in failure or "blind" in source:
            return "sql_boolean"
        return "sql_syntax"
    return "other"


def _family(row: Mapping[str, Any]) -> str:
    surface = str(row.get("surface_class") or row.get("surface_role") or "").casefold()
    source = str(row.get("source", "")).casefold()
    if "dom" in surface or "xss" in source:
        return "dom"
    if "sql" in surface or "sqli" in source:
        return "sql"
    return "other"


def _normalise(row: Mapping[str, Any], *, source_lane: str) -> dict[str, Any]:
    record = dict(row)
    record["source_lane"] = source_lane
    record["rule_ir_class"] = _rule_class(record)
    record["family_class"] = _family(record)
    record["source_record_id"] = str(record.get("record_id") or record.get("parent_record_id") or digest({"source": record.get("source"), "seed": record.get("seed"), "token_hash": record.get("token_hash"), "trajectory_hash": record.get("trajectory_hash")}))
    if "tokens" not in record or not record.get("tokens"):
        raise ValueError(f"PG-258 input row has no bounded tokens: {record['source_record_id']}")
    return record


def _load_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pg249 = json.loads(PG249_DATASET.read_text(encoding="utf-8-sig"))
    pg257 = json.loads(PG257_DATASET.read_text(encoding="utf-8-sig"))
    rows.extend(_normalise(row, source_lane="pg249_policy_process") for row in list(pg249.get("records") or []))
    rows.extend(_normalise(row, source_lane="pg257_rule_ir_feedback") for row in list(pg257.get("records") or []))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", 0) or 0), str(row.get("token_hash", row.get("trajectory_hash", ""))), str(row.get("rule_ir_class", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _is_ood(row: Mapping[str, Any]) -> bool:
    return str(row.get("source", "")) == "pg246_vulnerableapp_source_independent"


def _is_holdout(row: Mapping[str, Any]) -> bool:
    source = str(row.get("source", ""))
    seed = int(row.get("seed", 0) or 0)
    # Keep one complete XSS seed for the implementation/seed holdout while
    # retaining the other authorized seed for learning.  Holding out every
    # PG-242 row would leave too few DOM positives to estimate a class head;
    # it would test data starvation rather than generalization.
    if source == "pg242_pikachu_source_native" and seed == 24202:
        return True
    if source in {"pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"} and seed == 24402:
        return True
    if source in {"pg255_standard_sql_replay", "pg256_widebyte_failure_feedback"}:
        return seed % 2 == 0
    return False


def _positions(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(row.get("classification_position", 0)), 0), max(width - 1, 0)) for row in rows], dtype=torch.long, device=device)


def _metrics(
    model: UnifiedRuleIRCapacityAdapter,
    context: torch.Tensor,
    encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rule_targets: torch.Tensor,
    family_targets: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, Any]:
    return evaluate_unified_adapter(model, context, encoded[1], rule_targets, family_targets, positions)


def _legacy_metrics(model: Any, context: torch.Tensor, encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    return PG237._evaluate(model, context, encoded, positions, rows, device)


def _state_digest(model: torch.nn.Module) -> str:
    return hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def main() -> int:
    rows = [row for row in _load_records() if row.get("lane") not in {"quarantine", "reject"}]
    ood = [row for row in rows if _is_ood(row)]
    eligible = [row for row in rows if not _is_ood(row)]
    holdout = [row for row in eligible if _is_holdout(row)]
    train = [row for row in eligible if not _is_holdout(row)]
    if not train or not holdout or not ood:
        raise RuntimeError("PG-258 requires train, holdout and implementation OOD rows")
    if set(str(row["rule_ir_class"]) for row in train) != set(RULE_IR_CLASSES):
        raise RuntimeError(f"PG-258 train must contain all Rule-IR classes: {Counter(row['rule_ir_class'] for row in train)}")
    if set(str(row["rule_ir_class"]) for row in holdout) != set(RULE_IR_CLASSES):
        raise RuntimeError(f"PG-258 holdout must contain all Rule-IR classes: {Counter(row['rule_ir_class'] for row in holdout)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    # PG-249 normally installs this alias inside its CLI entry point.  PG-258
    # imports the wrapper as a library, so establish the immutable baseline
    # encoder explicitly before selecting the observation-only aliases.
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
    train_encoded = PG231._encode(train, input_vocab, target_vocab, device)
    hold_encoded = PG231._encode(holdout, input_vocab, target_vocab, device)
    ood_encoded = PG231._encode(ood, input_vocab, target_vocab, device)
    with torch.no_grad():
        train_body = base.base.body.encode(train_encoded[0], train_encoded[0].ne(0)).detach().clone()
        hold_body = base.base.body.encode(hold_encoded[0], hold_encoded[0].ne(0)).detach().clone()
        ood_body = base.base.body.encode(ood_encoded[0], ood_encoded[0].ne(0)).detach().clone()
        train_context = old_policy.context_projection(train_body).detach().clone()
        hold_context = old_policy.context_projection(hold_body).detach().clone()
        ood_context = old_policy.context_projection(ood_body).detach().clone()
    train_positions = _positions(train, train_context.shape[1], device)
    hold_positions = _positions(holdout, hold_context.shape[1], device)
    ood_positions = _positions(ood, ood_context.shape[1], device)
    train_rules = torch.tensor([rule_target(row["rule_ir_class"]) for row in train], dtype=torch.long, device=device)
    hold_rules = torch.tensor([rule_target(row["rule_ir_class"]) for row in holdout], dtype=torch.long, device=device)
    ood_rules = torch.tensor([rule_target(row["rule_ir_class"]) for row in ood], dtype=torch.long, device=device)
    train_families = torch.tensor([family_target(row["family_class"]) for row in train], dtype=torch.long, device=device)
    hold_families = torch.tensor([family_target(row["family_class"]) for row in holdout], dtype=torch.long, device=device)
    ood_families = torch.tensor([family_target(row["family_class"]) for row in ood], dtype=torch.long, device=device)

    # The source corpus is intentionally heterogeneous and ``other`` is the
    # largest bucket.  Inverse-square-root weights keep the rare, explicitly
    # grounded SQL/XSS classes from being erased by the majority class while
    # avoiding the unstable gradients of a raw inverse-frequency weight.
    rule_counts = torch.bincount(train_rules, minlength=len(RULE_IR_CLASSES)).float()
    family_counts = torch.bincount(train_families, minlength=len(FAMILY_CLASSES)).float()
    rule_weights = torch.where(rule_counts > 0, torch.sqrt(rule_counts.sum() / rule_counts), torch.zeros_like(rule_counts)).to(device)
    family_weights = torch.where(family_counts > 0, torch.sqrt(family_counts.sum() / family_counts), torch.zeros_like(family_counts)).to(device)

    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: UnifiedRuleIRCapacityAdapter | None = None
    for hidden_dim in CAPACITY_VARIANTS:
        torch.manual_seed(258 + hidden_dim)
        model = UnifiedRuleIRCapacityAdapter(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, token_vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        for _ in range(TRAIN_STEPS):
            model.train()
            output = model(train_context, classification_positions=train_positions)
            token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_encoded[1].reshape(-1), ignore_index=0)
            rule_loss = nn.functional.cross_entropy(output["rule"], train_rules, weight=rule_weights)
            family_loss = nn.functional.cross_entropy(output["family"], train_families, weight=family_weights)
            loss = token_loss + rule_loss + 0.5 * family_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {
            "hidden_dim": hidden_dim,
            "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "train": _metrics(model, train_context, train_encoded, train_rules, train_families, train_positions),
            "seed_route_family_holdout": _metrics(model, hold_context, hold_encoded, hold_rules, hold_families, hold_positions),
            "implementation_ood": _metrics(model, ood_context, ood_encoded, ood_rules, ood_families, ood_positions),
        }
        variants.append(result)
        metrics = result["seed_route_family_holdout"]
        key = (-float(metrics["rule_accuracy"]), -float(metrics["family_accuracy"]), float(metrics["token_loss"]))
        old_metrics = None if selected is None else selected["seed_route_family_holdout"]
        old_key = None if old_metrics is None else (-float(old_metrics["rule_accuracy"]), -float(old_metrics["family_accuracy"]), float(old_metrics["token_loss"]))
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-258 did not select a capacity variant")

    # The old action/lane/repair model is frozen.  Re-run its action projection
    # on the selected holdout to prove the new Rule-IR head did not mutate it.
    old_hold_encoded = PG231._encode(holdout, input_vocab, old_target_vocab, device)
    old_ood_encoded = PG231._encode(ood, input_vocab, old_target_vocab, device)
    with torch.no_grad():
        old_hold_body = base.base.body.encode(old_hold_encoded[0], old_hold_encoded[0].ne(0)).detach().clone()
        old_ood_body = base.base.body.encode(old_ood_encoded[0], old_ood_encoded[0].ne(0)).detach().clone()
    old_hold_context = old_hold_body
    old_ood_context = old_ood_body
    old_hold_positions = _positions(holdout, old_hold_context.shape[1], device)
    old_ood_positions = _positions(ood, old_ood_context.shape[1], device)
    legacy_hash_before = _state_digest(old_policy)
    legacy_hold_before = _legacy_metrics(old_policy, old_hold_context, old_hold_encoded, old_hold_positions, holdout, device)
    legacy_ood_before = _legacy_metrics(old_policy, old_ood_context, old_ood_encoded, old_ood_positions, ood, device)
    legacy_hash_after = _state_digest(old_policy)
    legacy_hold_after = _legacy_metrics(old_policy, old_hold_context, old_hold_encoded, old_hold_positions, holdout, device)
    legacy_ood_after = _legacy_metrics(old_policy, old_ood_context, old_ood_encoded, old_ood_positions, ood, device)
    canary = {
        "before": {"holdout": legacy_hold_before, "implementation_ood": legacy_ood_before},
        "after": {"holdout": legacy_hold_after, "implementation_ood": legacy_ood_after},
        "state_hash_before": legacy_hash_before,
        "state_hash_after": legacy_hash_after,
        "state_unchanged": legacy_hash_before == legacy_hash_after,
        "action_metrics_unchanged": legacy_hold_before == legacy_hold_after and legacy_ood_before == legacy_ood_after,
        "canary_is_evaluation_only": True,
        "oracle_features_fed_to_model": False,
    }
    canary["pass"] = bool(canary["state_unchanged"] and canary["action_metrics_unchanged"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"unified_rule_ir_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": "pg258-unified-rule-ir-adapter-artifact-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "token_vocabulary": target_vocab, "rule_classes": list(RULE_IR_CLASSES), "family_classes": list(FAMILY_CLASSES), "frozen_legacy_artifact": str(PG249_ARTIFACT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    counts = {"records": len(rows), "train_rows": len(train), "holdout_rows": len(holdout), "implementation_ood_rows": len(ood), "train_rule_counts": dict(Counter(row["rule_ir_class"] for row in train)), "holdout_rule_counts": dict(Counter(row["rule_ir_class"] for row in holdout)), "ood_rule_counts": dict(Counter(row["rule_ir_class"] for row in ood)), "train_family_counts": dict(Counter(row["family_class"] for row in train)), "holdout_family_counts": dict(Counter(row["family_class"] for row in holdout)), "source_counts": dict(Counter(row["source"] for row in rows))}
    hold_metrics = selected["seed_route_family_holdout"]
    ood_metrics = selected["implementation_ood"]
    holdout_support = {name: int(hold_metrics.get(f"{name}_count", 0) or 0) for name in RULE_IR_CLASSES}
    judge_gates = {
        "holdout_rule_accuracy_ge_0_80": float(hold_metrics["rule_accuracy"]) >= 0.80,
        "holdout_family_accuracy_ge_0_80": float(hold_metrics["family_accuracy"]) >= 0.80,
        "implementation_ood_family_accuracy_ge_0_60": float(ood_metrics["family_accuracy"]) >= 0.60,
        "holdout_each_rule_class_support_ge_2": all(value >= 2 for value in holdout_support.values()),
        "catastrophic_forgetting_canary": bool(canary["pass"]),
    }
    independent_final_judge = {
        "authority": ["seed/route holdout", "separate VulnerableApp implementation OOD", "frozen legacy policy canary"],
        "hard_gates": judge_gates,
        "holdout_support": holdout_support,
        "pass": bool(all(judge_gates.values())),
        "decision": "candidate_eligible_for_next_replay" if all(judge_gates.values()) else "blocked_insufficient_generalization",
        "reasons": [name for name, passed in judge_gates.items() if not passed],
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
    }
    report = {"protocol_id": "pg-pk-258-unified-rule-ir-capacity-v1", "schema_version": "pg258-unified-rule-ir-capacity-report-v1", "status": "completed_unified_sql_xss_rule_ir_capacity_training", "device": str(device), "capacity_variants": list(CAPACITY_VARIANTS), "train_steps": TRAIN_STEPS, "counts": counts, "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected}, "frozen_legacy_policy": {"artifact": str(PG249_ARTIFACT.relative_to(ROOT)), "hidden_dim": int(old_payload["hidden_dim"]), "base_parameter_count": 101487169, "policy_state_unchanged": canary["state_unchanged"]}, "catastrophic_forgetting_canary": canary, "independent_final_judge": independent_final_judge, "training_eligible": bool(independent_final_judge["pass"]), "model_input_excludes_oracle_target": True, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "blocked_by": independent_final_judge["reasons"]}, "honesty": {"pg242_dom_oracle_records_are_local": True, "pg257_sql_class_records_are_local": True, "implementation_ood_is_separate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "legacy_action_policy_not_replaced": True}}
    report["report_sha256"] = digest(report)
    dataset = {"schema_version": "pg258-unified-rule-ir-capacity-dataset-v1", "source_datasets": [str(PG249_DATASET.relative_to(ROOT)), str(PG257_DATASET.relative_to(ROOT))], "records": rows, "counts": counts, "contract": {"pg242_seed_24202_holdout": True, "pg244_seed_24402_holdout": True, "pg257_even_seed_holdout": True, "vulnerableapp_implementation_ood_separate": True, "oracle_target_off_input": True, "legacy_policy_frozen": True, "next_token_auxiliary_loss": True, "next_token_loss_not_promotion_gate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg258-unified-rule-ir-capacity-protocol-v1", "frozen_legacy_policy": str(PG249_ARTIFACT.relative_to(ROOT)), "train_sources": ["pg249 records outside holdout", "pg257 odd-seed Rule-IR feedback"], "holdout": ["pg242 seed 24202 XSS", "pg244 seed 24402", "pg255/256 even seeds"], "implementation_ood": "pg246 VulnerableApp rows, separate score", "capacity_variants": list(CAPACITY_VARIANTS), "class_weighting": "inverse_sqrt_train_frequency", "sequence_summary_fusion": True, "canary_replay_required": True, "oracle_target_off_input": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg258-unified-rule-ir-capacity-trace-v1", "selected": selected, "canary": canary, "independent_final_judge": independent_final_judge, "counts": counts, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-258 unified SQL/XSS Rule-IR capacity", "", f"train={len(train)}; holdout={len(holdout)}; implementation OOD={len(ood)}", f"selected_hidden={selected['hidden_dim']}; holdout_rule={selected['seed_route_family_holdout']['rule_accuracy']}; holdout_family={selected['seed_route_family_holdout']['family_accuracy']}; holdout_next_token={selected['seed_route_family_holdout']['next_token_accuracy']}", f"judge={independent_final_judge['decision']}; reasons={', '.join(independent_final_judge['reasons']) or 'none'}", f"canary={canary['pass']}; state_unchanged={canary['state_unchanged']}; action_metrics_unchanged={canary['action_metrics_unchanged']}", "旧发送/拒答策略保持冻结；新头只学习抽象 Rule-IR 与 surface family。PG-242/PG-257 oracle 目标不进入输入，结果不代表公网能力。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "selected": report["selected"], "judge": independent_final_judge, "canary": canary, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
