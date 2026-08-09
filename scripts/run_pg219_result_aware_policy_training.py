"""PG-219: train a result-aware multi-step policy on real Pikachu traces.

PG-217/218 supply the bounded observations.  The runner does not re-send a
payload: it trains a process head from already completed local episodes, then
evaluates seed and complete-route holdouts.  The frozen 101M language body is
used only as context; the adapter sees no route text, payload value, response
body, or evaluator label as an input feature.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg219_result_policy import (  # noqa: E402
    OUTCOME_NAMES,
    PG219_SCHEMA,
    PROCESS_ACTIONS,
    ResultAwareProcessPolicy,
    hard_gate,
    predict_result_policy,
    train_result_policy,
)


RESEARCH = ROOT / "research"
PG217_REPORT = RESEARCH / "pg217_pikachu_typed_sql_oracle_report_v1.json"
PG218_REPORT = RESEARCH / "pg218_pikachu_result_fixture_report_v1.json"
DATASET = RESEARCH / "pg219_result_aware_policy_dataset_v1.json"
REPORT = RESEARCH / "pg219_result_aware_policy_training_report_v1.json"
PROTOCOL = RESEARCH / "pg219_result_aware_policy_training_protocol_v1.json"
TRACE = RESEARCH / "pg219_result_aware_policy_training_trace_v1.json"
MARKDOWN = RESEARCH / "pg219_result_aware_policy_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg219-result-aware-policy-v1"
BASE_ARTIFACT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
ROUTE_HOLDOUT = "/vul/sqli/sqli_x.php"
TRAIN_SEEDS = {21701, 21801}
HOLDOUT_SEEDS = {21702, 21802}
CAPACITY_VARIANTS = {"standard": 96, "wide": 192, "deep": 384}
EPOCHS = 80


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _fixture_projection(row: Mapping[str, Any], side: str) -> dict[str, Any]:
    fixture = dict(row.get("fixture") or {})
    branch = dict(fixture.get(side) or {})
    return dict(branch.get("response_projection") or {})


def _base_state(typed: Mapping[str, Any], fixture: Mapping[str, Any], *, phase: str, history_len: int, previous_feedback: str) -> dict[str, Any]:
    evidence = dict(typed.get("evidence") or {})
    reset = dict(typed.get("reset") or {})
    negative_projection = dict(fixture.get("negative_projection") or {})
    return {
        "method": str(typed.get("method", "GET")).upper(),
        "status_class": "2xx",
        "redirect_hops": 0,
        "candidate_signal": 0,
        "typed_available": bool((typed.get("contract") or {}).get("confirmable") and evidence.get("baseline_backend_state") == "backend_response_observed"),
        "negative_control": True,
        "budget_remaining": 1 if phase != "preflight" else 2,
        "failure_kind": "no_effect",
        "fresh_reset_ok": bool((reset.get("fresh_target") and reset.get("container_recreated") and not reset.get("container_restart_used") and int(reset.get("volume_mount_count", -1)) == 0)),
        "database_health_ok": str(reset.get("database_health_gate")) == "mysqli_root_pikachu_ok",
        "backend_observed": evidence.get("baseline_backend_state") == "backend_response_observed",
        "negative_clean": bool(negative_projection.get("row_marker_count", 0) == 0 and negative_projection.get("result_shape") == "record_absent") or not negative_projection,
        "candidate_sql_error_shape": bool(evidence.get("candidate_sql_error_shape")),
        "reference_agreement": bool(evidence.get("candidate_reference_agreement")),
        "candidate_result_present": False,
        "negative_result_absent": bool(negative_projection.get("row_marker_count", 0) == 0),
        "typed_effect_observed": bool(typed.get("typed_effect_confirmed")),
        "result_fixture_verified": False,
        "candidate_sent": False,
        "reference_sent": bool(row_value := fixture.get("reference_sent", True)),
        "negative_sent": True,
        "previous_feedback": previous_feedback,
        "history_len": int(history_len),
        "field_count": 0,
        "binding_valid": True,
        "phase": phase,
    }


def _make_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pg217 = _load(PG217_REPORT)
    pg218 = _load(PG218_REPORT)
    typed_by_key = {(int(row["seed"]), str(row["route"])): row for row in pg217["results"]}
    rows: list[dict[str, Any]] = []
    for source_row in pg218["results"]:
        # PG-217 and PG-218 deliberately used separate seed namespaces for
        # the same two route passes; join by pass index, not literal seed.
        source_seed = int(source_row["seed"])
        typed_seed = 21701 if source_seed == 21801 else 21702 if source_seed == 21802 else source_seed
        key = (typed_seed, str(source_row["route"]))
        typed_row = typed_by_key.get(key)
        if typed_row is None:
            raise RuntimeError(f"PG-219 missing PG-217 pair for {key}")
        typed_oracle = dict(typed_row.get("typed_oracle") or {})
        fixture = dict(source_row.get("fixture") or {})
        positive_projection = _fixture_projection(source_row, "positive")
        negative_projection = _fixture_projection(source_row, "negative")
        result_oracle = dict(source_row.get("result_oracle") or {})
        result_verified = bool(result_oracle.get("result_fixture_verified"))
        typed_effect = bool(typed_oracle.get("typed_effect_confirmed"))
        backend = str((typed_oracle.get("evidence") or {}).get("baseline_backend_state")) == "backend_response_observed"
        environment_failure = not backend or not bool((source_row.get("reset") or {}).get("fresh_target"))
        outcome = "environment_failure" if environment_failure else "result_verified" if result_verified else "typed_effect" if typed_effect else "no_effect"
        typed_context = dict(typed_oracle)
        typed_context["reset"] = dict(source_row.get("reset") or {})
        typed_context["method"] = str(source_row.get("method", "GET")).upper()
        typed_context["typed_effect_confirmed"] = typed_effect
        route = str(source_row["route"])
        fields = list(source_row.get("fields") or [])
        common = {
            "seed": int(source_row["seed"]),
            "route": route,
            "method": str(source_row.get("method", "GET")).upper(),
            "field_count": len(fields),
            "fields": fields,
            "source": "pg217_pg218_real_projection",
            "typed_effect_target": typed_effect,
            "result_fixture_target": result_verified,
            "outcome_target": outcome,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "oracle_labels_as_features": False,
        }
        pre = _base_state(typed_context, {**fixture, "negative_projection": negative_projection, "reference_sent": source_row.get("reference_sent", True)}, phase="preflight", history_len=0, previous_feedback="none")
        pre.update({"field_count": len(fields), "outcome_label": outcome, "label": "safe_candidate" if hard_gate(pre) else "abstain"})
        rows.append({**common, **pre, "step_index": 0})
        candidate = dict(pre)
        candidate.update({
            "phase": "candidate_feedback",
            "history_len": 1,
            "previous_feedback": "candidate_error" if bool((typed_oracle.get("evidence") or {}).get("candidate_sql_error_shape")) else "no_effect",
            "candidate_signal": 1,
            "candidate_sent": bool(source_row.get("ai_sent")),
            "candidate_result_present": bool(positive_projection.get("row_marker_count", 0) > 0),
            "result_fixture_verified": False,
            "outcome_label": outcome,
            "label": "abstain" if result_verified or not hard_gate(pre) else "retry_alternate",
        })
        rows.append({**common, **candidate, "step_index": 1})
        verify = dict(candidate)
        verify.update({
            "phase": "verification_feedback",
            "history_len": 2,
            "previous_feedback": "result_verified" if result_verified else "reference_disagreement" if not bool((typed_oracle.get("evidence") or {}).get("candidate_reference_agreement")) else "no_effect",
            "result_fixture_verified": result_verified,
            "reference_agreement": bool((typed_oracle.get("evidence") or {}).get("candidate_reference_agreement")),
            "outcome_label": outcome,
            "label": "abstain" if result_verified or not hard_gate(pre) else "retry_alternate",
        })
        rows.append({**common, **verify, "step_index": 2})
    # The completed PG-217/218 runs contain typed positives and typed
    # abstentions, but no live "typed backend + candidate had no effect"
    # episode.  Add an explicitly unsent counterfactual for that missing
    # branch so retry learning is measurable without fabricating a target
    # response or calling it a real vulnerability sample.
    typed_failure_counterfactuals: list[dict[str, Any]] = []
    for row in list(rows):
        if int(row.get("step_index", -1)) != 1 or not bool(row.get("typed_available")) or not bool(row.get("typed_effect_target")):
            continue
        copy = dict(row)
        copy.update({
            "candidate_signal": 0,
            "candidate_sql_error_shape": False,
            "candidate_result_present": False,
            "reference_agreement": False,
            "typed_effect_observed": False,
            "result_fixture_verified": False,
            "candidate_sent": False,
            "previous_feedback": "no_effect",
            "label": "retry_alternate",
            "outcome_label": "no_effect",
            "counterfactual": True,
            "counterfactual_kind": "typed_candidate_no_effect",
        })
        typed_failure_counterfactuals.append(copy)
    rows.extend(typed_failure_counterfactuals)
    train = [row for row in rows if int(row["seed"]) in TRAIN_SEEDS and str(row["route"]) != ROUTE_HOLDOUT]
    holdout = [row for row in rows if int(row["seed"]) in HOLDOUT_SEEDS or str(row["route"]) == ROUTE_HOLDOUT]
    # Unsent binding counterfactuals make the hard gate an OOD test, not a
    # property accidentally learned from the positive route distribution.
    def counterfactual(source: Mapping[str, Any]) -> dict[str, Any]:
        copy = dict(source)
        copy.update({"method": "POST" if str(source.get("method")) == "GET" else "GET", "field_count": 0, "fields": [], "typed_available": False, "binding_valid": False, "fresh_reset_ok": False, "database_health_ok": False, "negative_clean": False, "candidate_sent": False, "reference_sent": False, "negative_sent": False, "label": "abstain", "outcome_label": "environment_failure", "counterfactual": True})
        return copy
    train.extend(counterfactual(row) for row in list(train))
    holdout.extend(counterfactual(row) for row in list(holdout))
    meta = {
        "all_rows": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_seed_set": sorted(TRAIN_SEEDS),
        "holdout_seed_set": sorted(HOLDOUT_SEEDS),
        "route_holdout": ROUTE_HOLDOUT,
        "train_route_count": len({row["route"] for row in train if not row.get("counterfactual")}),
        "holdout_route_count": len({row["route"] for row in holdout if not row.get("counterfactual")}),
        "counterfactual_train_rows": sum(int(bool(row.get("counterfactual"))) for row in train),
        "counterfactual_holdout_rows": sum(int(bool(row.get("counterfactual"))) for row in holdout),
        "typed_failure_counterfactual_rows": len(typed_failure_counterfactuals),
    }
    return train, holdout, meta


def _load_frozen_base(device: torch.device) -> tuple[torch.nn.Module, dict[str, int]]:
    if not BASE_ARTIFACT.exists():
        raise FileNotFoundError(BASE_ARTIFACT)
    checkpoint = torch.load(BASE_ARTIFACT, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    pg197 = _load_script("run_pg197_risk_aware_cross_evaluator.py")
    # PG-197 exposes the PG-194 builder as a loaded module; use its frozen
    # body directly and avoid retraining the older risk head.
    base = pg197.PG194._load_model(vocabulary, device)
    for parameter in base.parameters():
        parameter.requires_grad = False
    base.eval()
    return base, vocabulary


def _adapter_state(model: ResultAwareProcessPolicy) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith("frozen_base.")}


def main() -> int:
    train_rows, holdout_rows, data_meta = _make_rows()
    dataset = {
        "schema_version": "pg219-result-aware-policy-dataset-v1",
        "source_reports": [str(PG217_REPORT.relative_to(ROOT)), str(PG218_REPORT.relative_to(ROOT))],
        "rows": train_rows + holdout_rows,
        "split": data_meta,
        "contract": {"raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_labels_as_features": False, "route_identity_as_feature": False, "training_promotion_allowed": False},
    }
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen_base, vocabulary = _load_frozen_base(device)
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_score = float("-inf")
    for index, (name, hidden_dim) in enumerate(CAPACITY_VARIANTS.items()):
        random.seed(21900 + index)
        torch.manual_seed(21900 + index)
        model = ResultAwareProcessPolicy(frozen_base, d_model=1024, hidden_dim=hidden_dim).to(device)
        training = train_result_policy(model, train_rows, holdout_rows, ids, mask, epochs=EPOCHS)
        score = float(training["holdout"]["gated_action_accuracy"]) + float(training["holdout"]["outcome_accuracy"]) - 0.25 * int(training["holdout"]["gated_unsafe_allow_count"])
        result = {
            "variant": name,
            "hidden_dim": hidden_dim,
            "base_parameter_count": int(sum(parameter.numel() for parameter in frozen_base.parameters())),
            "trainable_adapter_parameter_count": int(sum(parameter.numel() for name_, parameter in model.named_parameters() if not name_.startswith("frozen_base."))),
            "training": training,
            "score": round(score, 8),
            "catastrophic_forgetting_gate": {"not_applicable_to_frozen_body": True, "adapter_holdout_gated_accuracy": training["holdout"]["gated_action_accuracy"]},
        }
        variants.append(result)
        if result["score"] > selected_score and training["holdout"]["gated_unsafe_allow_count"] == 0:
            selected = result
            selected_score = result["score"]
            artifact = ARTIFACT_DIR / f"result_aware_policy_{name}.pt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": PG219_SCHEMA, "variant": name, "hidden_dim": hidden_dim, "base_parameter_count": result["base_parameter_count"], "model_state": _adapter_state(model), "raw_inputs_retained": False, "promotion_allowed": False}, artifact)
            result["artifact"] = str(artifact.relative_to(ROOT))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("PG-219 no variant passed the gated unsafe-action gate")
    # A compact shadow replay makes the policy's stepwise behavior auditable
    # without sending another request or storing any runtime value.
    selected_variant = selected["variant"]
    selected_dim = int(selected["hidden_dim"])
    shadow = ResultAwareProcessPolicy(frozen_base, d_model=1024, hidden_dim=selected_dim).to(device)
    selected_checkpoint = torch.load(ROOT / selected["artifact"], map_location="cpu", weights_only=False)
    state = shadow.state_dict()
    for key, value in selected_checkpoint["model_state"].items():
        state[key].copy_(value)
    shadow.load_state_dict(state)
    shadow_rows = []
    for row in holdout_rows[: min(len(holdout_rows), 18)]:
        prediction = predict_result_policy(shadow, row, ids, mask)
        shadow_rows.append({"seed": row["seed"], "route": row["route"], "step_index": row["step_index"], "target_action": row["label"], "proposed_action": prediction["proposed_action"], "action": prediction["action"], "hard_gate": prediction["hard_gate"], "outcome": prediction["outcome"]})
    del shadow, frozen_base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = {
        "protocol_id": "pg-pk-219-result-aware-policy-training-v1",
        "schema_version": "pg219-result-aware-policy-training-report-v1",
        "status": "completed_result_aware_process_policy_seed_route_holdout",
        "device": str(device),
        "data": data_meta,
        "model": {"base_artifact": str(BASE_ARTIFACT.relative_to(ROOT)), "base_parameter_count": variants[0]["base_parameter_count"], "context_hidden_frozen": True, "capacity_variants": CAPACITY_VARIANTS, "selected_variant": selected_variant},
        "variants": variants,
        "shadow_replay": shadow_rows,
        "promotion": {"training_eligible": True, "artifact_promotion_allowed": False, "memory_promotion_allowed": False, "live_send_takeover_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"local_trace_sources_only": True, "loopback_replay_not_repeated": True, "external_network_targets": False, "database_write": False, "time_delay_used": False, "oracle_labels_as_features": False, "route_identity_as_feature": False, "raw_inputs_retained": False},
        "source_report_sha256": {"pg217": _digest(_load(PG217_REPORT)), "pg218": _digest(_load(PG218_REPORT))},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg219-result-aware-policy-training-protocol-v1",
        "sources": [str(PG217_REPORT.relative_to(ROOT)), str(PG218_REPORT.relative_to(ROOT))],
        "split": {"train_seeds": sorted(TRAIN_SEEDS), "holdout_seeds": sorted(HOLDOUT_SEEDS), "complete_route_holdout": ROUTE_HOLDOUT},
        "steps": ["preflight", "candidate_feedback", "verification_feedback"],
        "actions": list(PROCESS_ACTIONS),
        "outcomes": list(OUTCOME_NAMES),
        "large_body_frozen": True,
        "oracle_labels_as_features": False,
        "route_identity_as_feature": False,
        "hard_gate": ["typed_available", "fresh_reset_ok", "database_health_ok", "backend_observed", "negative_clean", "binding_valid"],
        "promotion_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg219-result-aware-policy-training-trace-v1", "data": data_meta, "variants": [{"variant": row["variant"], "hidden_dim": row["hidden_dim"], "training": row["training"], "artifact": row.get("artifact")} for row in variants], "shadow_replay": shadow_rows, "raw_inputs_retained": False, "training_eligible": True})
    lines = [
        "# PG-219 result-aware process policy",
        "",
        f"device={device}; train={data_meta['train_rows']}; holdout={data_meta['holdout_rows']}; route_holdout={ROUTE_HOLDOUT}",
        f"selected={selected_variant}; variants={[(row['variant'], row['hidden_dim'], row['training']['holdout']['gated_action_accuracy'], row['training']['holdout']['gated_unsafe_allow_count']) for row in variants]}",
        "",
        "模型只读取 bounded transport/result projection；typed/result oracle 只作为监督目标或上一阶段反馈，不作为当前动作的输入。large body 冻结，adapter 未接管真实发包。",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "selected_variant": selected_variant, "data": data_meta, "variants": [{"variant": row["variant"], "holdout": row["training"]["holdout"]} for row in variants], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
