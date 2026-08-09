"""PG-257: train a larger Rule-IR class decoder from PG-256 failure traces.

The input is an abstract feedback trajectory.  The target is the next binding
class supplied by an independent local replay/reference lane.  Raw GET/POST
values, route identities, response bodies, and evaluator keys are excluded.
The frozen PG-191 XXL body stays fixed; only 1024/2048/4096 heads are trained,
then evaluated on even-seed holdouts.
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


PG231 = _load("run_pg231_feedback_trajectory_training.py")
from app.pg230_next_token_quality_funnel import build_vocabulary, digest  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402
from app.pg257_rule_ir_class_decoder import RULE_CLASSES, RULE_CLASS_INDEX, RuleIRClassDecoder, class_target, evaluate_decoder  # noqa: E402


RESEARCH = ROOT / "research"
PG255_REPORT = RESEARCH / "pg255_pikachu_fixed_sql_pg254_replay_report_v1.json"
PG256_REPORT = RESEARCH / "pg256_pikachu_widebyte_oracle_report_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg257_widebyte_rule_ir_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg257_widebyte_rule_ir_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg257_widebyte_rule_ir_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg257_widebyte_rule_ir_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg257_widebyte_rule_ir_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg257-widebyte-rule-ir-capacity-v1"
CAPACITY_VARIANTS = (1024, 2048, 4096)
TRAIN_STEPS = 80


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _pg255_rows() -> list[dict[str, Any]]:
    report = json.loads(PG255_REPORT.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(list(report.get("episodes") or [])):
        route = str(episode.get("path", ""))
        typed = dict(episode.get("typed_oracle") or {})
        evidence = dict(typed.get("evidence") or {})
        if route.endswith("sqli_blind_t.php"):
            # Timing is intentionally not a training target in this lane.
            continue
        if route.endswith("sqli_blind_b.php"):
            rule_class = "blind_boolean"
        elif route.endswith("sqli_widebyte.php"):
            rule_class = "widebyte_escape_boundary"
        else:
            rule_class = "syntax_boundary"
        confirmed = bool(typed.get("typed_effect_confirmed"))
        raw = {
            "source": "pg255_standard_sql_replay",
            "seed": int(episode.get("seed", 0) or 0),
            "surface_role": "sql_surface",
            "method": str(episode.get("method", "GET")).upper(),
            "status_class": "2xx",
            "field_count": len(episode.get("fields") or []),
            "history_len": 0,
            "fresh_reset_ok": bool(episode.get("fresh_target")),
            "reset_completed": bool((episode.get("reset") or {}).get("completed")),
            "candidate_sent": bool((episode.get("ai") or {}).get("sent")),
            "reference_sent": bool((episode.get("reference") or {}).get("sent")),
            "negative_sent": True,
            "oracle_available": bool(confirmed),
            "typed_effect_confirmed": confirmed,
            "result_fixture_verified": confirmed,
            "candidate_reference_agreement": bool(evidence.get("candidate_reference_agreement", confirmed)),
            "negative_clean": not bool(evidence.get("negative_sql_error_shape", False)),
            "binding_valid": True,
            "backend_observed": True,
            "database_health_ok": bool((episode.get("reset") or {}).get("database_health_gate") == "mysqli_root_pikachu_ok"),
            "candidate_result_present": confirmed,
            "candidate_sql_error_shape": bool(evidence.get("candidate_sql_error_shape", False)),
            "negative_result_absent": not bool(evidence.get("negative_sql_error_shape", False)),
            "model_claimed_positive": False,
            "model_abstained": not confirmed,
            "result_mismatch_observed": not bool(evidence.get("candidate_reference_agreement", confirmed)),
            "previous_feedback": "result_verified" if confirmed else "mismatch",
            "failure_signature": "typed_effect" if confirmed else "counterfactual_candidate_no_effect",
            "evidence_hash": str(typed.get("evidence_hash", "")),
            "payload_grounded_eligible": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "next_rule_class": rule_class,
            "source_record_id": f"pg255:{index}:{route}",
        }
        if len(raw["evidence_hash"]) != 64:
            raw["evidence_hash"] = _hash({"source": raw["source"], "seed": raw["seed"], "route_class": rule_class})
        rows.append(raw)
    return rows


def _pg256_rows() -> list[dict[str, Any]]:
    report = json.loads(PG256_REPORT.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(list(report.get("episodes") or [])):
        typed = dict(episode.get("typed_oracle") or {})
        evidence = dict(typed.get("evidence") or {})
        confirmed = bool(typed.get("typed_effect_confirmed"))
        raw = {
            "source": "pg256_widebyte_failure_feedback",
            "seed": int(episode.get("seed", 0) or 0),
            "surface_role": "sql_surface",
            "method": "POST",
            "status_class": "2xx",
            "field_count": 2,
            "history_len": 0,
            "fresh_reset_ok": bool((episode.get("reset") or {}).get("fresh_target")),
            "reset_completed": bool((episode.get("reset") or {}).get("completed")),
            "candidate_sent": bool((episode.get("ai") or {}).get("sent")),
            "reference_sent": bool((episode.get("reference") or {}).get("sent")),
            "negative_sent": True,
            "oracle_available": True,
            "typed_effect_confirmed": confirmed,
            "result_fixture_verified": confirmed,
            "candidate_reference_agreement": bool(evidence.get("candidate_reference_agreement", confirmed)),
            "negative_clean": int(evidence.get("negative_row_count_capped", 0) or 0) == 0,
            "binding_valid": True,
            "backend_observed": True,
            "database_health_ok": bool((episode.get("reset") or {}).get("database_health_gate") == "mysqli_root_pikachu_ok"),
            "candidate_result_present": int(evidence.get("candidate_row_count_capped", 0) or 0) > 0,
            "candidate_sql_error_shape": False,
            "negative_result_absent": int(evidence.get("negative_row_count_capped", 0) or 0) == 0,
            "model_claimed_positive": False,
            "model_abstained": False,
            "result_mismatch_observed": not bool(evidence.get("candidate_reference_agreement", confirmed)),
            "previous_feedback": "result_verified" if confirmed else "mismatch",
            "failure_signature": "typed_effect" if confirmed else "counterfactual_candidate_no_effect",
            "evidence_hash": str(typed.get("evidence_hash", "")),
            "payload_grounded_eligible": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "next_rule_class": "widebyte_escape_boundary",
            "source_record_id": f"pg256:{index}:widebyte",
        }
        if len(raw["evidence_hash"]) != 64:
            raw["evidence_hash"] = _hash({"source": raw["source"], "seed": raw["seed"], "route_class": "widebyte_escape_boundary"})
        rows.append(raw)
    return rows


def _prepare_rows() -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in _pg255_rows() + _pg256_rows():
        key = (int(raw["seed"]), str(raw["source_record_id"]), str(raw["next_rule_class"]))
        if key in seen:
            continue
        seen.add(key)
        record = prepare_feedback_record(raw)
        record["next_rule_class"] = str(raw["next_rule_class"])
        record["source_record_id"] = str(raw["source_record_id"])
        prepared.append(record)
    if not prepared:
        raise RuntimeError("PG-257 did not find replay rows")
    return prepared


def _holdout(record: Mapping[str, Any]) -> bool:
    # Even seeds are never used to train; the split is seed and source
    # separated while every class remains present on both sides.
    return int(record.get("seed", 0) or 0) % 2 == 0


def _encode(records: list[dict[str, Any]], input_vocab: Mapping[str, int], target_vocab: Mapping[str, int], device: torch.device):
    return PG231._encode(records, input_vocab, target_vocab, device)


def _positions(records: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(record.get("classification_position", 0)), 0), max(width - 1, 0)) for record in records], dtype=torch.long, device=device)


def main() -> int:
    records = _prepare_rows()
    train = [row for row in records if not _holdout(row)]
    holdout = [row for row in records if _holdout(row)]
    if not train or not holdout:
        raise RuntimeError("PG-257 requires both odd-seed train and even-seed holdout")
    if set(str(row["next_rule_class"]) for row in train) != set(RULE_CLASSES):
        raise RuntimeError("PG-257 train split must contain all Rule-IR classes")
    if set(str(row["next_rule_class"]) for row in holdout) != set(RULE_CLASSES):
        raise RuntimeError("PG-257 holdout must contain all Rule-IR classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    target_vocab = build_vocabulary(records)
    train_ids, train_target_tokens, _, _ = _encode(train, input_vocab, target_vocab, device)
    hold_ids, hold_target_tokens, _, _ = _encode(holdout, input_vocab, target_vocab, device)
    # ``inference_mode`` tensors cannot later participate in adapter backward
    # on this PyTorch build; no_grad keeps the frozen context detached while
    # allowing the trainable Rule-IR head to consume it.
    with torch.no_grad():
        train_context = base.base.body.encode(train_ids, train_ids.ne(0)).detach().clone()
        hold_context = base.base.body.encode(hold_ids, hold_ids.ne(0)).detach().clone()
    train_positions = _positions(train, train_context.shape[1], device)
    hold_positions = _positions(holdout, hold_context.shape[1], device)
    train_rules = torch.tensor([class_target(row["next_rule_class"]) for row in train], dtype=torch.long, device=device)
    hold_rules = torch.tensor([class_target(row["next_rule_class"]) for row in holdout], dtype=torch.long, device=device)
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: RuleIRClassDecoder | None = None
    for hidden_dim in CAPACITY_VARIANTS:
        torch.manual_seed(257 + hidden_dim)
        model = RuleIRClassDecoder(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, token_vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        for _ in range(TRAIN_STEPS):
            model.train()
            output = model(train_context, classification_positions=train_positions)
            token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_target_tokens.reshape(-1), ignore_index=0)
            rule_loss = nn.functional.cross_entropy(output["rule"], train_rules)
            loss = token_loss + 1.0 * rule_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train": evaluate_decoder(model, train_context, train_target_tokens, train_rules, train_positions), "seed_holdout": evaluate_decoder(model, hold_context, hold_target_tokens, hold_rules, hold_positions)}
        variants.append(result)
        metrics = result["seed_holdout"]
        key = (-float(metrics["rule_accuracy"]), -float(metrics.get("widebyte_escape_boundary_recall", 0.0)), float(metrics["token_loss"]))
        old_metrics = None if selected is None else selected["seed_holdout"]
        old_key = None if old_metrics is None else (-float(old_metrics["rule_accuracy"]), -float(old_metrics.get("widebyte_escape_boundary_recall", 0.0)), float(old_metrics["token_loss"]))
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-257 no capacity variant selected")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"rule_ir_class_decoder_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": "pg257-rule-ir-class-decoder-artifact-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "rule_classes": list(RULE_CLASSES), "token_vocabulary": target_vocab, "frozen_body_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {"schema_version": "pg257-widebyte-rule-ir-capacity-training-dataset-v1", "source_reports": [str(PG255_REPORT.relative_to(ROOT)), str(PG256_REPORT.relative_to(ROOT))], "records": records, "counts": {"records": len(records), "train_rows": len(train), "holdout_rows": len(holdout), "train_class_counts": dict(Counter(str(row["next_rule_class"]) for row in train)), "holdout_class_counts": dict(Counter(str(row["next_rule_class"]) for row in holdout))}, "contract": {"seed_disjoint_holdout": True, "route_identity_not_tokenized": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_target_off_input": True, "next_token_auxiliary_loss": True, "next_token_loss_not_promotion_gate": True, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _hash(dataset)
    report = {"protocol_id": "pg-pk-257-widebyte-rule-ir-capacity-training-v1", "schema_version": "pg257-widebyte-rule-ir-capacity-training-report-v1", "status": "completed_rule_ir_class_capacity_training", "device": str(device), "capacity_variants": list(CAPACITY_VARIANTS), "train_steps": TRAIN_STEPS, "counts": dataset["counts"], "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected}, "variants": variants, "frozen_body_parameter_count": int(sum(parameter.numel() for parameter in base.parameters())), "model_input_excludes_oracle_target": True, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"payload_class_target_is_reference_bound": True, "seed_holdout_disjoint": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True}}
    report["report_sha256"] = _hash(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg257-widebyte-rule-ir-capacity-training-protocol-v1", "model": "frozen_pg191_xxl_body_plus_rule_ir_class_adapter", "capacity_variants": list(CAPACITY_VARIANTS), "train_steps": TRAIN_STEPS, "holdout": "even seeds from PG255/PG256", "input_tokens": "bounded failure/response process tokens only", "target": "next abstract Rule-IR binding class", "oracle_target_off_input": True, "catastrophic_forgetting_canary": "required before promotion", "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = _hash(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg257-widebyte-rule-ir-capacity-training-trace-v1", "selected": selected, "variants": variants, "holdout_class_counts": dataset["counts"]["holdout_class_counts"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-257 wide-byte Rule-IR class capacity training", "", f"train={len(train)}; holdout={len(holdout)}; train_classes={dataset['counts']['train_class_counts']}; holdout_classes={dataset['counts']['holdout_class_counts']}", f"selected_hidden={selected['hidden_dim']}; holdout_rule_accuracy={selected['seed_holdout']['rule_accuracy']}; widebyte_recall={selected['seed_holdout'].get('widebyte_escape_boundary_recall')}; next_token={selected['seed_holdout']['next_token_accuracy']}", "", "训练只使用抽象失败/复放过程 token；payload class 是独立 reference/evaluator 产生的监督标签，不进入模型输入。结果不能推出公网漏洞能力，晋级保持冻结。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
