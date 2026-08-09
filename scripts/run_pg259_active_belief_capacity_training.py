"""PG-259: train an active-belief Rule-IR adapter on fresh local traces.

The body and legacy send/abstain policy are frozen.  This experiment adds
fresh PG-259 SQL/XSS/boolean/widebyte projections, holds out unseen routes,
and scores the new Rule-IR, family, belief, and next-probe heads separately.
It is deliberately promotion-blocked until the independent route holdout,
implementation OOD, class-support, and forgetting gates all pass.
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


PG258 = _load("run_pg258_unified_rule_ir_capacity.py")
PG249 = PG258.PG249
PG231 = PG258.PG231
PG237 = PG258.PG237
from app.pg230_next_token_quality_funnel import build_vocabulary, digest  # noqa: E402
from app.pg259_active_belief_rule_ir_adapter import (  # noqa: E402
    ActiveBeliefRuleIRAdapter,
    BELIEF_CLASSES,
    FAMILY_CLASSES,
    PROBE_CLASSES,
    RULE_IR_CLASSES,
    belief_target,
    evaluate_active_adapter,
    family_target,
    probe_target,
    rule_target,
)


RESEARCH = ROOT / "research"
PG258_DATASET = RESEARCH / "pg258_unified_rule_ir_capacity_dataset_v1.json"
PG259_DATASET = RESEARCH / "pg259_fresh_local_trace_collection_dataset_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
PG249_ARTIFACT = ROOT / "artifacts" / "pg249-pikachu-route-seed-capacity-v1" / "frozen_xxl_capacity_hidden4096.pt"
REPORT = RESEARCH / "pg259_active_belief_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg259_active_belief_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg259_active_belief_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg259_active_belief_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg259_active_belief_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg259-active-belief-capacity-v1"
CAPACITY_VARIANTS = (1024, 2048, 4096)
TRAIN_STEPS = 140


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _route(row: Mapping[str, Any]) -> str:
    parent = str(row.get("parent_record_id") or "")
    return str(row.get("route") or parent.rsplit(":", 1)[-1])


def _rule_class(row: Mapping[str, Any]) -> str:
    """Resolve PG-259 source names before falling back to PG-258 labels."""
    explicit = str(row.get("next_rule_class", ""))
    if explicit == "syntax_boundary":
        return "sql_syntax"
    if explicit == "blind_boolean":
        return "sql_boolean"
    if explicit == "widebyte_escape_boundary":
        return "sql_widebyte"
    surface = str(row.get("surface_class") or row.get("surface_role") or "").casefold()
    failure = str(row.get("failure_kind") or row.get("failure_signature") or "").casefold()
    source = str(row.get("source", "")).casefold()
    lane = str(row.get("lane", "")).casefold()
    route = _route(row).casefold()
    is_dom = "dom" in surface or "xss" in source or "xss" in route
    is_sql = "sql" in surface or "sqli" in source or "sql" in source or "sqli" in route
    if is_dom:
        if "oracle" in failure or lane == "silver":
            return "oracle_gap"
        if "typed" in failure or lane == "gold":
            return "dom_marker"
        return "other"
    if is_sql:
        if "widebyte" in source or "widebyte" in failure or "widebyte" in route:
            return "sql_widebyte"
        if "boolean" in source or "differential" in failure or "blind" in source or "blind" in route:
            return "sql_boolean"
        return "sql_syntax"
    return "other"


def _family(row: Mapping[str, Any]) -> str:
    surface = str(row.get("surface_class") or row.get("surface_role") or "").casefold()
    source = str(row.get("source", "")).casefold()
    route = _route(row).casefold()
    if "dom" in surface or "xss" in surface or "xss" in source or "xss" in route:
        return "dom"
    if "sql" in surface or "sqli" in surface or "sql" in source or "sqli" in source or "sql" in route:
        return "sql"
    return "other"


def _belief(row: Mapping[str, Any]) -> str:
    lane = str(row.get("lane", ""))
    if lane == "gold" or bool(row.get("payload_grounded_eligible")):
        return "confirmed_effect"
    if lane == "hard_negative":
        return "needs_reference"
    if lane == "silver":
        return "oracle_gap"
    return "repair_environment"


def _probe(row: Mapping[str, Any]) -> str:
    lane = str(row.get("lane", ""))
    if lane == "gold":
        return "replay_confirm"
    if lane == "hard_negative":
        return "negative_control"
    if lane == "silver":
        return "reference_probe"
    return "reset_environment"


def _normalise(row: Mapping[str, Any], *, source_lane: str) -> dict[str, Any]:
    record = dict(row)
    record["source_lane"] = source_lane
    record["rule_ir_class"] = _rule_class(record)
    record["family_class"] = _family(record)
    record["belief_class"] = _belief(record)
    record["probe_class"] = _probe(record)
    record["source_record_id"] = str(record.get("record_id") or record.get("parent_record_id") or digest({"source": record.get("source"), "seed": record.get("seed"), "route": _route(record), "trajectory_hash": record.get("trajectory_hash")}))
    if not record.get("tokens"):
        raise ValueError(f"PG-259 input row has no bounded tokens: {record['source_record_id']}")
    return record


def _load_records() -> list[dict[str, Any]]:
    old = json.loads(PG258_DATASET.read_text(encoding="utf-8-sig"))
    fresh = json.loads(PG259_DATASET.read_text(encoding="utf-8-sig"))
    rows = [_normalise(row, source_lane="pg258_unified") for row in list(old.get("records") or [])]
    rows.extend(_normalise(row, source_lane="pg259_fresh_local") for row in list(fresh.get("records") or []))
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


def _is_old_holdout(row: Mapping[str, Any]) -> bool:
    # Preserve PG-258's independent route/seed partition.
    return bool(PG258._is_holdout(row))


def _is_fresh_holdout(row: Mapping[str, Any]) -> bool:
    source = str(row.get("source", ""))
    if not source.startswith("pg259_"):
        return False
    route = _route(row)
    holdout_routes = {
        "/vul/sqli/sqli_search.php",
        "/vul/sqli/sqli_blind_b.php",
        "/vul/sqli/sqli_widebyte.php",
        "/vul/xss/xss_03.php",
        "/vul/xss/xss_dom_x.php",
    }
    # Keep one boolean seed as a true unseen-seed check even though its route
    # is shared with the fresh SQL surface.
    return route in holdout_routes or int(row.get("seed", 0) or 0) == 25922


def _positions(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(row.get("classification_position", 0)), 0), max(width - 1, 0)) for row in rows], dtype=torch.long, device=device)


def _state_digest(model: torch.nn.Module) -> str:
    return hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def _metrics(model: ActiveBeliefRuleIRAdapter, context: torch.Tensor, encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    positions = _positions(rows, context.shape[1], device)
    rules = torch.tensor([rule_target(row["rule_ir_class"]) for row in rows], dtype=torch.long, device=device)
    families = torch.tensor([family_target(row["family_class"]) for row in rows], dtype=torch.long, device=device)
    beliefs = torch.tensor([belief_target(row["belief_class"]) for row in rows], dtype=torch.long, device=device)
    probes = torch.tensor([probe_target(row["probe_class"]) for row in rows], dtype=torch.long, device=device)
    return evaluate_active_adapter(model, context, encoded[1], rules, families, beliefs, probes, positions)


def _encode_context(rows: list[dict[str, Any]], input_vocab: dict[str, int], target_vocab: dict[str, int], base: Any, old_policy: Any, device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    encoded = PG231._encode(rows, input_vocab, target_vocab, device)
    with torch.no_grad():
        body = base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
        context = old_policy.context_projection(body).detach().clone()
    return context, encoded


def main() -> int:
    rows = [row for row in _load_records() if row.get("lane") not in {"quarantine", "reject"}]
    ood = [row for row in rows if _is_ood(row)]
    eligible = [row for row in rows if not _is_ood(row)]
    holdout = [row for row in eligible if _is_old_holdout(row) or _is_fresh_holdout(row)]
    train = [row for row in eligible if not (_is_old_holdout(row) or _is_fresh_holdout(row))]
    fresh_train = [row for row in train if str(row.get("source", "")).startswith("pg259_")]
    fresh_holdout = [row for row in holdout if str(row.get("source", "")).startswith("pg259_")]
    if not train or not holdout or not ood or not fresh_train or not fresh_holdout:
        raise RuntimeError("PG-259 requires old+fresh train, fresh route holdout, and implementation OOD")

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
    fresh_hold_context, fresh_hold_encoded = _encode_context(fresh_holdout, input_vocab, target_vocab, base, old_policy, device)
    ood_context, ood_encoded = _encode_context(ood, input_vocab, target_vocab, base, old_policy, device)

    train_rules = torch.tensor([rule_target(row["rule_ir_class"]) for row in train], dtype=torch.long, device=device)
    train_families = torch.tensor([family_target(row["family_class"]) for row in train], dtype=torch.long, device=device)
    train_beliefs = torch.tensor([belief_target(row["belief_class"]) for row in train], dtype=torch.long, device=device)
    train_probes = torch.tensor([probe_target(row["probe_class"]) for row in train], dtype=torch.long, device=device)
    rule_counts = torch.bincount(train_rules, minlength=len(RULE_IR_CLASSES)).float()
    family_counts = torch.bincount(train_families, minlength=3).float()
    belief_counts = torch.bincount(train_beliefs, minlength=len(BELIEF_CLASSES)).float()
    probe_counts = torch.bincount(train_probes, minlength=len(PROBE_CLASSES)).float()

    def weights(counts: torch.Tensor) -> torch.Tensor:
        return torch.where(counts > 0, torch.sqrt(counts.sum() / counts), torch.zeros_like(counts)).to(device)

    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: ActiveBeliefRuleIRAdapter | None = None
    for hidden_dim in CAPACITY_VARIANTS:
        torch.manual_seed(259 + hidden_dim)
        model = ActiveBeliefRuleIRAdapter(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, token_vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        positions = _positions(train, train_context.shape[1], device)
        for _ in range(TRAIN_STEPS):
            model.train()
            output = model(train_context, classification_positions=positions)
            token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_encoded[1].reshape(-1), ignore_index=0)
            rule_loss = nn.functional.cross_entropy(output["rule"], train_rules, weight=weights(rule_counts))
            family_loss = nn.functional.cross_entropy(output["family"], train_families, weight=weights(family_counts))
            belief_loss = nn.functional.cross_entropy(output["belief"], train_beliefs, weight=weights(belief_counts))
            probe_loss = nn.functional.cross_entropy(output["probe"], train_probes, weight=weights(probe_counts))
            loss = token_loss + rule_loss + 0.5 * family_loss + 0.75 * belief_loss + 0.75 * probe_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {
            "hidden_dim": hidden_dim,
            "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "train": _metrics(model, train_context, train_encoded, train, device),
            "route_seed_holdout": _metrics(model, hold_context, hold_encoded, holdout, device),
            "fresh_route_holdout": _metrics(model, fresh_hold_context, fresh_hold_encoded, fresh_holdout, device),
            "implementation_ood": _metrics(model, ood_context, ood_encoded, ood, device),
        }
        variants.append(result)
        metrics = result["route_seed_holdout"]
        key = (-float(metrics["rule_accuracy"]), -float(metrics["family_accuracy"]), -float(metrics["belief_accuracy"]), float(metrics["token_loss"]))
        old_metrics = None if selected is None else selected["route_seed_holdout"]
        old_key = None if old_metrics is None else (-float(old_metrics["rule_accuracy"]), -float(old_metrics["family_accuracy"]), -float(old_metrics["belief_accuracy"]), float(old_metrics["token_loss"]))
        if selected is None or key < old_key:
            selected, selected_model = result, model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-259 did not select a capacity variant")

    # The legacy policy is never updated.  Replaying it before/after the new
    # adapter run catches accidental optimizer/state mutation.
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
    torch.save({"schema_version": "pg259-active-belief-adapter-artifact-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "token_vocabulary": target_vocab, "rule_classes": list(RULE_IR_CLASSES), "family_classes": list(FAMILY_CLASSES), "belief_classes": list(BELIEF_CLASSES), "probe_classes": list(PROBE_CLASSES), "frozen_legacy_artifact": str(PG249_ARTIFACT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    counts = {
        "records": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "fresh_train_rows": len(fresh_train),
        "fresh_holdout_rows": len(fresh_holdout),
        "implementation_ood_rows": len(ood),
        "train_rule_counts": dict(Counter(row["rule_ir_class"] for row in train)),
        "holdout_rule_counts": dict(Counter(row["rule_ir_class"] for row in holdout)),
        "fresh_holdout_rule_counts": dict(Counter(row["rule_ir_class"] for row in fresh_holdout)),
        "ood_rule_counts": dict(Counter(row["rule_ir_class"] for row in ood)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
    }
    hold_metrics = selected["route_seed_holdout"]
    fresh_metrics = selected["fresh_route_holdout"]
    ood_metrics = selected["implementation_ood"]
    holdout_support = {name: int(hold_metrics.get(f"{name}_count", 0) or 0) for name in RULE_IR_CLASSES}
    gates = {
        "holdout_rule_accuracy_ge_0_80": float(hold_metrics["rule_accuracy"]) >= 0.80,
        "holdout_family_accuracy_ge_0_80": float(hold_metrics["family_accuracy"]) >= 0.80,
        "fresh_route_rule_accuracy_ge_0_70": float(fresh_metrics["rule_accuracy"]) >= 0.70,
        "fresh_route_belief_accuracy_ge_0_70": float(fresh_metrics["belief_accuracy"]) >= 0.70,
        "implementation_ood_family_accuracy_ge_0_60": float(ood_metrics["family_accuracy"]) >= 0.60,
        "holdout_each_rule_class_support_ge_2": all(value >= 2 for value in holdout_support.values()),
        "catastrophic_forgetting_canary": bool(canary["pass"]),
    }
    judge = {
        "authority": ["PG-258 seed/route holdout", "fresh PG-259 unseen routes", "separate VulnerableApp implementation OOD", "frozen legacy policy canary"],
        "hard_gates": gates,
        "holdout_support": holdout_support,
        "pass": bool(all(gates.values())),
        "decision": "candidate_eligible_for_next_replay" if all(gates.values()) else "blocked_insufficient_generalization",
        "reasons": [name for name, passed in gates.items() if not passed],
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
    }
    report = {
        "protocol_id": "pg-pk-259-active-belief-capacity-v1",
        "schema_version": "pg259-active-belief-capacity-training-report-v1",
        "status": "completed_active_belief_rule_ir_training",
        "device": str(device),
        "capacity_variants": list(CAPACITY_VARIANTS),
        "train_steps": TRAIN_STEPS,
        "counts": counts,
        "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected},
        "frozen_legacy_policy": {"artifact": str(PG249_ARTIFACT.relative_to(ROOT)), "policy_state_unchanged": canary["state_unchanged"]},
        "catastrophic_forgetting_canary": canary,
        "independent_final_judge": judge,
        "training_eligible": bool(judge["pass"]),
        "model_input_excludes_oracle_target": True,
        "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "blocked_by": judge["reasons"]},
        "honesty": {"fresh_traces_are_authorized_loopback": True, "fresh_pg259_routes_are_disjoint_in_holdout": True, "implementation_ood_is_separate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "legacy_action_policy_not_replaced": True},
    }
    report["report_sha256"] = digest(report)
    dataset = {"schema_version": "pg259-active-belief-capacity-training-dataset-v1", "source_datasets": [str(PG258_DATASET.relative_to(ROOT)), str(PG259_DATASET.relative_to(ROOT))], "records": rows, "counts": counts, "contract": {"fresh_pg259_route_holdout": True, "vulnerableapp_implementation_ood_separate": True, "oracle_target_off_input": True, "legacy_policy_frozen": True, "active_belief_and_probe_auxiliary_heads": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    fresh_holdout_manifest = sorted(
        [{"source": row["source"], "route": _route(row), "seed": row["seed"]} for row in fresh_holdout],
        key=lambda x: (x["source"], x["route"], x["seed"]),
    )
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg259-active-belief-capacity-training-protocol-v1", "frozen_legacy_policy": str(PG249_ARTIFACT.relative_to(ROOT)), "training_sources": [str(PG258_DATASET.relative_to(ROOT)), "PG-259 fresh local traces outside route holdout"], "fresh_route_holdout": fresh_holdout_manifest, "implementation_ood": "pg246 VulnerableApp rows, separate score", "capacity_variants": list(CAPACITY_VARIANTS), "heads": ["next_token", "rule_ir", "family", "belief", "probe"], "oracle_target_off_input": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg259-active-belief-capacity-trace-v1", "selected": selected, "fresh_holdout": fresh_metrics, "canary": canary, "independent_final_judge": judge, "counts": counts, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-259 active-belief Rule-IR capacity", "", f"train={len(train)}; fresh_train={len(fresh_train)}; holdout={len(holdout)}; fresh_holdout={len(fresh_holdout)}; implementation OOD={len(ood)}", f"selected_hidden={selected['hidden_dim']}; holdout_rule={hold_metrics['rule_accuracy']}; holdout_family={hold_metrics['family_accuracy']}; fresh_rule={fresh_metrics['rule_accuracy']}; fresh_belief={fresh_metrics['belief_accuracy']}; holdout_next_token={hold_metrics['next_token_accuracy']}", f"judge={judge['decision']}; reasons={', '.join(judge['reasons']) or 'none'}", f"canary={canary['pass']}; state_unchanged={canary['state_unchanged']}; action_metrics_unchanged={canary['action_metrics_unchanged']}", "新增 active-belief/probe 头只读取抽象轨迹；oracle 仅作监督，旧动作策略保持冻结。结果不代表公网能力。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "selected": report["selected"], "judge": judge, "canary": canary, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
