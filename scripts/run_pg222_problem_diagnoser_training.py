"""PG-222: train a small AI that diagnoses why a probe episode failed.

The source reports are already bounded process traces.  This runner turns
their observable fields into a diagnostic dataset, adds explicitly marked
counterfactual failures, and trains a compact diagnosis head with a
seed/route holdout.  It never stores executable payloads or response bodies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg222_problem_diagnoser import (  # noqa: E402
    DIAGNOSIS_NAMES,
    PG222_SCHEMA,
    ProblemDiagnoser,
    diagnose_features,
    hard_diagnostic_gate,
    train_problem_diagnoser,
)


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg222-problem-diagnoser-v1"
REPORT = RESEARCH / "pg222_problem_diagnoser_training_report_v1.json"
DATASET = RESEARCH / "pg222_problem_diagnoser_dataset_v1.json"
PROTOCOL = RESEARCH / "pg222_problem_diagnoser_protocol_v1.json"
TRACE = RESEARCH / "pg222_problem_diagnoser_training_trace_v1.json"
MARKDOWN = RESEARCH / "pg222_problem_diagnoser_training_report_v1.md"

SOURCE_REPORTS = (
    RESEARCH / "pg217_pikachu_typed_sql_oracle_report_v1.json",
    RESEARCH / "pg218_pikachu_result_fixture_report_v1.json",
    RESEARCH / "pg220_live_shadow_replay_report_v1.json",
    RESEARCH / "pg221_pikachu_boolean_blind_oracle_report_v1.json",
)
HOLDOUT_SEEDS = {21702, 21802, 22002, 22102}
HOLDOUT_ROUTES = {
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/sqli/sqli_x.php",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _projection_from_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    ai = result.get("ai") or {}
    if isinstance(ai.get("response_projection"), Mapping):
        return ai["response_projection"]
    fixture = result.get("fixture") or {}
    positive = fixture.get("positive") or {}
    return positive.get("response_projection") or {}


def _negative_projection(result: Mapping[str, Any]) -> Mapping[str, Any]:
    negative = result.get("negative") or {}
    if isinstance(negative.get("projection"), Mapping):
        return negative["projection"].get("response_projection") or {}
    response = negative.get("response") or {}
    return response.get("response_projection") or {}


def _bool(value: Any) -> bool:
    return bool(value)


def _normalise_feedback(value: Any) -> str:
    text = str(value or "none")
    return {"dead_end": "no_effect", "result_verified": "result_verified", "evaluator_confirmed": "result_verified"}.get(text, text if text in {"none", "no_effect", "environment_failure", "reference_disagreement", "result_verified"} else "none")


def _common_row(result: Mapping[str, Any], *, source: str, shadow: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reset = result.get("reset") or {}
    projection = _projection_from_result(result)
    negative = _negative_projection(result)
    typed = result.get("typed_oracle") or {}
    typed_evidence = typed.get("evidence") or {}
    result_oracle = result.get("result_oracle") or {}
    result_evidence = result_oracle.get("evidence") or {}
    boolean_oracle = result.get("oracle") or {}
    boolean_evidence = boolean_oracle.get("evidence") or {}
    method = str(result.get("method", "GET")).upper()
    fields = list(result.get("fields") or [])
    typed_available = bool((typed.get("contract") or {}).get("confirmable"))
    oracle_available = bool(
        typed_available
        or boolean_oracle.get("boolean_effect_confirmed")
        or result_oracle.get("result_fixture_verified")
    )
    typed_effect = bool(typed.get("typed_effect_confirmed") or boolean_oracle.get("boolean_effect_confirmed"))
    result_verified = bool(result_oracle.get("result_fixture_verified"))
    boolean_differential = bool(boolean_evidence.get("candidate_boolean_differential"))
    candidate_reference_agreement = typed_evidence.get("candidate_reference_agreement")
    if candidate_reference_agreement is None:
        candidate_reference_agreement = boolean_evidence.get("candidate_reference_agreement")
    if candidate_reference_agreement is None:
        candidate_reference_agreement = result.get("ai_reference_shape_agreement", True)
    reset_ok = bool(reset.get("fresh_target") and reset.get("completed") and reset.get("container_recreated"))
    health_ok = reset.get("database_health_gate") == "mysqli_root_pikachu_ok"
    backend_observed = projection.get("backend_state") in {"backend_response_observed", None}
    if result.get("ai_sent") is False:
        backend_observed = True
    candidate_sent = bool(result.get("ai_sent", (result.get("ai") or {}).get("sent", True)))
    reference_sent = bool(result.get("reference_sent", (result.get("reference") or {}).get("sent", True)))
    negative_sent = bool(result.get("negative_sent", (result.get("negative") or {}).get("sent", True)))
    model_claimed_positive = bool((result.get("ai") or {}).get("feedback", {}).get("evaluator_confirmed"))
    if shadow and str(shadow.get("target_action")) == "safe_candidate":
        model_claimed_positive = True
    model_abstained = bool(shadow and str(shadow.get("action")) == "abstain")
    row = {
        "seed": int(result.get("seed", 0)),
        "route": str(result.get("route", "")),
        "method": method,
        "field_count": len(fields),
        "source": source,
        "fresh_reset_ok": reset_ok,
        "reset_completed": bool(reset.get("completed")),
        "database_health_ok": health_ok,
        "backend_observed": backend_observed,
        "transport_error": bool(projection.get("transport_error")),
        "container_restart_used": bool(reset.get("container_restart_used")),
        "status_class": str(projection.get("status_class", "2xx")),
        "binding_valid": True,
        "candidate_sent": candidate_sent,
        "reference_sent": reference_sent,
        "negative_sent": negative_sent,
        "oracle_available": oracle_available,
        "typed_effect_observed": typed_effect,
        "result_fixture_verified": result_verified,
        "boolean_differential": boolean_differential,
        "candidate_reference_agreement": bool(candidate_reference_agreement),
        "negative_clean": int(negative.get("row_marker_count", 0) or 0) == 0 and not bool(negative.get("marker", {}).get("reflected")),
        "candidate_result_present": bool((boolean_evidence.get("true_candidate_shape") == "record_present") or result_verified),
        "negative_result_absent": str(negative.get("result_shape", "")) == "record_absent",
        "candidate_sql_error_shape": bool(typed_evidence.get("candidate_sql_error_shape")),
        "result_mismatch_observed": bool(typed_effect != result_verified) if result_oracle else False,
        "model_claimed_positive": model_claimed_positive,
        "model_abstained": model_abstained,
        "previous_feedback": _normalise_feedback((result.get("ai") or {}).get("feedback", {}).get("status")),
        "history_len": int((result.get("ai") or {}).get("feedback", {}).get("step", 0) or 0),
        "source_hash": str(result.get("route_source_sha256") or typed.get("source_attestation_sha256") or ""),
        "evidence_hash": str(result.get("evidence_hash") or typed.get("evidence_hash") or boolean_oracle.get("evidence_hash") or result_oracle.get("evidence_hash") or ""),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    row["hard_gate_observed"] = hard_diagnostic_gate(row)
    return row


def _label_real(row: Mapping[str, Any]) -> str:
    """Assign a target from independent oracle outcomes, never from model confidence."""

    if row.get("model_claimed_positive") and not hard_diagnostic_gate(row):
        return "model_decision_error"
    if not row.get("fresh_reset_ok") or not row.get("database_health_ok") or row.get("transport_error") or row.get("container_restart_used"):
        return "environment_failure"
    if not row.get("binding_valid"):
        return "binding_failure"
    if row.get("result_mismatch_observed"):
        return "result_mismatch"
    if row.get("candidate_reference_agreement") is False:
        return "reference_disagreement"
    if not row.get("oracle_available") and not row.get("typed_effect_observed") and not row.get("result_fixture_verified") and not row.get("boolean_differential"):
        return "oracle_unavailable"
    if hard_diagnostic_gate(row):
        return "confirmed_local_effect"
    if row.get("candidate_sent") and row.get("oracle_available") and row.get("negative_clean") and not row.get("typed_effect_observed") and not row.get("result_fixture_verified") and not row.get("boolean_differential"):
        return "candidate_no_effect"
    return "inconclusive"


def _counterfactual(row: Mapping[str, Any], diagnosis: str, index: int) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    item["counterfactual"] = True
    item["counterfactual_kind"] = diagnosis
    item["counterfactual_id"] = f"cf-{index:04d}-{diagnosis}"
    item["model_claimed_positive"] = False
    item["model_abstained"] = False
    item["result_mismatch_observed"] = False
    if diagnosis == "environment_failure":
        item.update(fresh_reset_ok=False, reset_completed=False, database_health_ok=False, backend_observed=False, transport_error=True)
    elif diagnosis == "binding_failure":
        item.update(binding_valid=False, field_count=0, candidate_sent=False, reference_sent=False, negative_sent=False)
    elif diagnosis == "oracle_unavailable":
        item.update(oracle_available=False, typed_effect_observed=False, result_fixture_verified=False, boolean_differential=False, candidate_sent=True, reference_sent=True, negative_sent=True, negative_clean=True)
    elif diagnosis == "candidate_no_effect":
        item.update(oracle_available=True, typed_effect_observed=False, result_fixture_verified=False, boolean_differential=False, candidate_sent=True, reference_sent=True, negative_sent=True, negative_clean=True, candidate_reference_agreement=True, candidate_result_present=False)
    elif diagnosis == "reference_disagreement":
        item.update(oracle_available=True, typed_effect_observed=False, result_fixture_verified=False, boolean_differential=False, candidate_sent=True, reference_sent=True, negative_sent=True, negative_clean=True, candidate_reference_agreement=False)
    elif diagnosis == "result_mismatch":
        item.update(oracle_available=True, typed_effect_observed=True, result_fixture_verified=False, boolean_differential=False, candidate_sent=True, reference_sent=True, negative_sent=True, negative_clean=True, candidate_reference_agreement=True, result_mismatch_observed=True)
    elif diagnosis == "model_decision_error":
        item.update(oracle_available=False, typed_effect_observed=False, result_fixture_verified=False, boolean_differential=False, candidate_sent=True, reference_sent=True, negative_sent=True, negative_clean=True, candidate_reference_agreement=True, model_claimed_positive=True)
    elif diagnosis == "confirmed_local_effect":
        item.update(fresh_reset_ok=True, reset_completed=True, database_health_ok=True, backend_observed=True, transport_error=False, container_restart_used=False, binding_valid=True, candidate_sent=True, reference_sent=True, negative_sent=True, oracle_available=True, typed_effect_observed=True, result_fixture_verified=True, boolean_differential=True, candidate_reference_agreement=True, negative_clean=True, negative_result_absent=True, source_hash="0" * 64, evidence_hash="1" * 64)
    elif diagnosis == "inconclusive":
        item.update(candidate_sent=False, reference_sent=False, negative_sent=False, oracle_available=False, typed_effect_observed=False, result_fixture_verified=False, boolean_differential=False, candidate_reference_agreement=True, negative_clean=False)
    item["diagnosis"] = diagnosis
    item["next_step"] = {"environment_failure": "inspect_environment", "binding_failure": "inspect_binding", "oracle_unavailable": "recheck_oracle", "candidate_no_effect": "retry_candidate", "reference_disagreement": "compare_reference", "result_mismatch": "recheck_oracle", "model_decision_error": "abstain", "confirmed_local_effect": "abstain", "inconclusive": "abstain"}[diagnosis]
    return item


def _build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shadow_by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for path in SOURCE_REPORTS:
        report = _load(path)
        for shadow in report.get("shadow", []) or []:
            key = (int(shadow.get("seed", 0)), str(shadow.get("route", "")), str(shadow.get("method", "GET")))
            shadow_by_key.setdefault(key, shadow)
        source = path.stem
        for result in report.get("results", []) or report.get("episodes", []):
            key = (int(result.get("seed", 0)), str(result.get("route", "")), str(result.get("method", "GET")))
            row = _common_row(result, source=source, shadow=shadow_by_key.get(key))
            row["diagnosis"] = _label_real(row)
            row["next_step"] = {"environment_failure": "inspect_environment", "binding_failure": "inspect_binding", "oracle_unavailable": "recheck_oracle", "candidate_no_effect": "retry_candidate", "reference_disagreement": "compare_reference", "result_mismatch": "recheck_oracle", "model_decision_error": "abstain", "confirmed_local_effect": "abstain", "inconclusive": "abstain"}[row["diagnosis"]]
            rows.append(row)
    # Every live observation receives the same bounded set of explicit
    # counterfactuals.  They are labelled as synthetic and are never claimed
    # to be live target behavior.
    kinds = [name for name in DIAGNOSIS_NAMES if name != "confirmed_local_effect"] + ["confirmed_local_effect"]
    index = 0
    for base in list(rows):
        for kind in kinds:
            rows.append(_counterfactual(base, kind, index))
            index += 1
    return rows


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in rows:
        is_holdout = int(row.get("seed", 0)) in HOLDOUT_SEEDS or str(row.get("route", "")) in HOLDOUT_ROUTES
        (holdout if is_holdout else train).append(row)
    if not train or not holdout:
        raise RuntimeError("PG-222 split produced an empty partition")
    return train, holdout


def _safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    safe = dict(row)
    # Keep only the public process projection; these are either booleans,
    # hashes, route templates, or bounded categorical values.
    safe.pop("raw_payload", None)
    safe.pop("payload", None)
    safe.pop("raw_response", None)
    safe.pop("response_body", None)
    return safe


def main() -> int:
    random.seed(222)
    torch.manual_seed(222)
    rows = _build_rows()
    train_rows, holdout_rows = _split(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: ProblemDiagnoser | None = None
    for hidden_dim in (32, 64, 128):
        torch.manual_seed(222 + hidden_dim)
        model = ProblemDiagnoser(hidden_dim=hidden_dim)
        result = train_problem_diagnoser(model, train_rows, holdout_rows, epochs=120, learning_rate=2e-3, device=device)
        result["hidden_dim"] = hidden_dim
        result["device"] = str(device)
        variants.append(result)
        if selected is None or (
            result["holdout"]["guarded_positive_false_accept_count"],
            -result["holdout"]["guarded_diagnosis_accuracy"],
            -result["holdout"]["next_step_accuracy"],
        ) < (
            selected["holdout"]["guarded_positive_false_accept_count"],
            -selected["holdout"]["guarded_diagnosis_accuracy"],
            -selected["holdout"]["next_step_accuracy"],
        ):
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-222 did not train a variant")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACT_DIR / f"problem_diagnoser_hidden{selected['hidden_dim']}.pt"
    torch.save({"state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "schema_version": PG222_SCHEMA}, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_hashes = [_digest(_load(path)) for path in SOURCE_REPORTS]
    dataset = {
        "schema_version": "pg222-problem-diagnoser-dataset-v1",
        "source_reports": [str(path.relative_to(ROOT)) for path in SOURCE_REPORTS],
        "source_report_hashes": source_hashes,
        "rows": [_safe_row(row) for row in rows],
        "split": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "holdout_seeds": sorted(HOLDOUT_SEEDS),
            "holdout_routes": sorted(HOLDOUT_ROUTES),
            "route_overlap": sorted({str(row.get("route")) for row in train_rows}.intersection(str(row.get("route")) for row in holdout_rows)),
        },
        "contract": {
            "feature_names": list(__import__("app.pg222_problem_diagnoser", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES),
            "evaluator_targets_not_features": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "counterfactual_rows_marked": True,
            "local_only": True,
        },
    }
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {
        "protocol_id": "pg-pk-222-problem-diagnoser-v1",
        "schema_version": PG222_SCHEMA,
        "status": "completed_structured_problem_diagnosis_training",
        "device": str(device),
        "source_reports": [str(path.relative_to(ROOT)) for path in SOURCE_REPORTS],
        "source_report_hashes": source_hashes,
        "dataset": str(DATASET.relative_to(ROOT)),
        "row_counts": {"total": len(rows), "train": len(train_rows), "holdout": len(holdout_rows), "counterfactual": sum(bool(row.get("counterfactual")) for row in rows)},
        "label_counts": {"all": dict(Counter(str(row["diagnosis"]) for row in rows)), "train": dict(Counter(str(row["diagnosis"]) for row in train_rows)), "holdout": dict(Counter(str(row["diagnosis"]) for row in holdout_rows))},
        "variants": variants,
        "selected": {"hidden_dim": selected["hidden_dim"], "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash, "holdout": selected["holdout"]},
        "guardrails": {
            "raw_positive_false_accept_count": selected["holdout"]["raw_positive_false_accept_count"],
            "guarded_positive_false_accept_count": selected["holdout"]["guarded_positive_false_accept_count"],
            "confirmed_local_effect_is_not_a_vulnerability_claim": True,
            "payload_generation": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "honesty": {
            "counterfactuals_are_not_live_evidence": True,
            "small_dataset": True,
            "source_and_route_holdout_used": True,
            "general_website_capability_not_established": True,
        },
        "safety": {"loopback_only": True, "external_network": False, "database_write": False, "time_delay_used": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg222-problem-diagnoser-protocol-v1",
        "objective": "detect process/environment/model errors before they become training or memory claims",
        "diagnosis_names": list(DIAGNOSIS_NAMES),
        "target_fields_not_features": ["diagnosis", "typed_effect_target", "result_fixture_target", "outcome_label"],
        "counterfactuals_marked": True,
        "seed_and_route_holdout": True,
        "positive_guard": "fresh reset + health + binding + reference + negative + typed/result/boolean evidence",
        "fail_closed": True,
        "raw_payload_and_response_excluded": True,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg222-problem-diagnoser-trace-v1", "selected": selected, "variants": variants, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join([
        "# PG-222 Problem Diagnoser",
        "",
        f"device={device}; rows={len(rows)} (train={len(train_rows)}, holdout={len(holdout_rows)}); counterfactual={sum(bool(row.get('counterfactual')) for row in rows)}",
        f"selected hidden={selected['hidden_dim']}; guarded holdout accuracy={selected['holdout']['guarded_diagnosis_accuracy']}; guarded positive false accepts={selected['holdout']['guarded_positive_false_accept_count']}",
        "",
        "模型只判断过程问题：环境、绑定、oracle、候选无效、参考不一致、结果不匹配、模型自身决策错误或本地效果已被复放确认。它不生成 payload，也不把确认结果升级成任意网站漏洞结论。",
        "",
        "PG-221 的真实修复被保留为过程教训：先前真假值构造多拼了结束引号，导致两条分支都无回显；修复后 fresh replay 变为 2/2。该轨迹说明诊断头需要检查绑定/候选构造，而不能直接责怪靶场。",
        "",
        "反事实行已显式标记，只用于训练分类边界，不是新的靶场证据；raw payload/response body 未保存；memory promotion 和 vulnerability claim 均关闭。",
        "",
    ]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "row_counts": report["row_counts"], "selected": report["selected"], "holdout": selected["holdout"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
