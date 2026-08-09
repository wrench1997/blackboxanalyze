"""PG-228: train a frozen-XXL diagnoser on typed and untyped local traces.

This experiment joins three evidence tiers without leaking route names, payload
values, response bodies, or evaluator targets into the feature encoder:

* PG-226 SQL rows passed an independent typed effect and result fixture.
* PG-227 DOM/redirect rows are deliberately oracle-unavailable for a
  vulnerability claim; a DOM surface effect is not XSS.
* PG-224 projection-only rows teach abstention when no typed oracle exists.

For each real row a bounded ``model_decision_error`` counterfactual records the
case where an agent claims a positive before the evidence gate.  These rows are
explicitly marked synthetic and never authorize a payload or memory update.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg223_large_problem_diagnoser import LargeProblemDiagnoserAdapter, PG223_SCHEMA, train_large_adapter  # noqa: E402


RESEARCH = ROOT / "research"
PG222_DATASET = RESEARCH / "pg222_problem_diagnoser_dataset_v1.json"
PG224_REPORT = RESEARCH / "pg224_pikachu_parameter_surface_collection_report_v1.json"
PG226_REPORT = RESEARCH / "pg226_ai_sql_payload_validation_report_v1.json"
PG227_REPORT = RESEARCH / "pg227_ai_dom_redirect_validation_report_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"

REPORT = RESEARCH / "pg228_grounded_diagnoser_training_report_v1.json"
DATASET = RESEARCH / "pg228_grounded_diagnoser_dataset_v1.json"
TRACE = RESEARCH / "pg228_grounded_diagnoser_trace_v1.json"
PROTOCOL = RESEARCH / "pg228_grounded_diagnoser_protocol_v1.json"
MARKDOWN = RESEARCH / "pg228_grounded_diagnoser_report_v1.md"


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")
PG222 = _load_script("run_pg222_problem_diagnoser_training.py")
PG223 = _load_script("run_pg223_large_problem_diagnoser.py")

NEXT_STEP = {
    "environment_failure": "inspect_environment",
    "binding_failure": "inspect_binding",
    "oracle_unavailable": "recheck_oracle",
    "candidate_no_effect": "retry_candidate",
    "reference_disagreement": "compare_reference",
    "result_mismatch": "recheck_oracle",
    "model_decision_error": "abstain",
    "confirmed_local_effect": "abstain",
    "inconclusive": "abstain",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _projection(result: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = result.get(key) or {}
    if isinstance(value, Mapping):
        projection = value.get("response_projection")
        if isinstance(projection, Mapping):
            return dict(projection)
    return {}


def _status_class(projection: Mapping[str, Any]) -> str:
    value = str(projection.get("status_class", "2xx"))
    return value if value in {"2xx", "3xx", "4xx", "5xx"} else "4xx"


def _database_health(reset: Mapping[str, Any]) -> bool:
    gate = str(reset.get("database_health_gate", ""))
    # SQL runs have an explicit mysqli gate; other read-only lanes have no
    # database assertion but still carry a successful fresh reset.
    return gate in {"mysqli_root_pikachu_ok", "not_applicable", ""} and bool(reset.get("completed", True))


def _base_row(*, result: Mapping[str, Any], source: str, diagnosis: str, grounding_status: str) -> dict[str, Any]:
    reset = result.get("reset") or result.get("fresh_reset") or {}
    if not isinstance(reset, Mapping):
        reset = {}
    fields = list(result.get("fields") or [])
    candidate = _projection(result, "candidate")
    negative = _projection(result, "negative")
    ai = result.get("ai") or {}
    sent = bool(ai.get("sent"))
    oracle = result.get("oracle") or {}
    typed_oracle = result.get("typed_oracle") or {}
    result_oracle = result.get("result_oracle") or {}
    evidence = result.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        evidence = {}
    candidate_reference_agreement = evidence.get("ai_reference_binding_match")
    if candidate_reference_agreement is None:
        candidate_reference_agreement = oracle.get("candidate_reference_agreement", True)
    negative_clean = oracle.get("negative_clean")
    if negative_clean is None:
        negative_clean = not bool((negative.get("marker") or {}).get("reflected", False))
    typed_effect = bool(result.get("typed_effect_confirmed") or result.get("dom_surface_effect_confirmed") and grounding_status == "typed_surface_only")
    result_verified = bool(result.get("result_fixture_verified"))
    if grounding_status == "typed_sql_result":
        typed_effect = bool(result.get("typed_effect_confirmed"))
        result_verified = bool(result.get("result_fixture_verified"))
    oracle_available = bool(typed_effect or result_verified or result.get("boolean_effect_confirmed"))
    if grounding_status in {"projection_only", "dom_surface_only", "redirect_surface_only"}:
        # No vulnerability authority is available in these lanes.  The DOM
        # and redirect projections are retained as observable context only.
        oracle_available = False
        typed_effect = False
        result_verified = False
    backend_observed = bool(sent and not candidate.get("transport_error", False))
    if not sent and isinstance(result.get("baseline"), Mapping):
        backend_observed = bool((_projection(result, "baseline")).get("backend_state") == "backend_response_observed")
    source_hash = str(evidence.get("route_source_sha256") or result.get("source_row_sha256") or result.get("target_instance_hash") or "")
    evidence_hash = str(evidence.get("evidence_sha256") or result.get("report_sha256") or "")
    if len(source_hash) != 64:
        source_hash = _digest({"source": source, "seed": result.get("seed", 0), "route": result.get("route", "")})
    if len(evidence_hash) != 64:
        evidence_hash = _digest({"source": source, "source_hash": source_hash, "status": grounding_status})
    preflight_only = bool(grounding_status == "projection_only" and not sent)
    # A blocked preflight is not an environment failure: no fresh target was
    # attempted yet.  Keep that distinction explicit so the hard gate does
    # not rewrite ``inconclusive`` into ``environment_failure``.
    fresh_reset_ok = bool(result.get("fresh_reset", reset.get("fresh_target", True)))
    reset_completed = bool(reset.get("completed", result.get("fresh_reset", True)))
    database_health_ok = _database_health(reset)
    if preflight_only:
        fresh_reset_ok = True
        reset_completed = False
        database_health_ok = True
    row: dict[str, Any] = {
        "seed": int(result.get("seed", 0)),
        "route": str(result.get("route", "")),
        "method": str(result.get("method", "GET")).upper(),
        "field_count": len(fields),
        "source": source,
        "grounding_status": grounding_status,
        "fresh_reset_ok": fresh_reset_ok,
        "reset_completed": reset_completed,
        "reset_not_attempted": preflight_only,
        "database_health_ok": database_health_ok,
        "backend_observed": backend_observed,
        "transport_error": bool(candidate.get("transport_error", False)),
        "container_restart_used": bool(reset.get("container_restart_used", False)),
        "status_class": _status_class(candidate),
        "binding_valid": bool(fields),
        "candidate_sent": sent,
        "reference_sent": bool(result.get("reference") is not None and sent),
        "negative_sent": bool(result.get("negative") is not None and sent),
        "oracle_available": oracle_available,
        "typed_effect_observed": typed_effect,
        "result_fixture_verified": result_verified,
        "boolean_differential": bool(result.get("boolean_effect_confirmed", False)),
        "candidate_reference_agreement": bool(candidate_reference_agreement),
        "negative_clean": bool(negative_clean),
        "candidate_result_present": bool((candidate.get("marker") or {}).get("reflected", False) or result.get("dom_surface_effect_confirmed", False)),
        "negative_result_absent": not bool((negative.get("marker") or {}).get("reflected", False)),
        "candidate_sql_error_shape": bool((candidate.get("shape") or {}).get("kind") == "sql_error"),
        "result_mismatch_observed": bool(result.get("result_mismatch_observed", False)),
        "model_claimed_positive": False,
        "model_abstained": not sent,
        "previous_feedback": "result_verified" if (typed_effect or result_verified) else ("no_effect" if sent else "none"),
        "history_len": 1 if sent else 0,
        "source_hash": source_hash,
        "evidence_hash": evidence_hash,
        "diagnosis": diagnosis,
        "next_step": NEXT_STEP[diagnosis],
        "payload_grounded_eligible": grounding_status == "typed_sql_result" and typed_effect and result_verified and bool(result.get("training_candidate", False)),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "real_trace": True,
    }
    return row


def _self_error(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    item.update(
        {
            "source": f"{row.get('source', 'unknown')}_self_error_counterfactual",
            "grounding_status": "self_error_counterfactual",
            "counterfactual": True,
            "counterfactual_id": f"pg228-self-error-{index:04d}",
            "oracle_available": False,
            "typed_effect_observed": False,
            "result_fixture_verified": False,
            "boolean_differential": False,
            "candidate_sent": True,
            "reference_sent": True,
            "negative_sent": True,
            "negative_clean": True,
            "candidate_reference_agreement": True,
            "model_claimed_positive": True,
            "model_abstained": False,
            "previous_feedback": "none",
            "diagnosis": "model_decision_error",
            "next_step": "abstain",
            "payload_grounded_eligible": False,
        }
    )
    item["evidence_hash"] = _digest({"self_error": item["counterfactual_id"], "source_hash": item.get("source_hash", "")})
    return item


def _new_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    pg224 = json.loads(PG224_REPORT.read_text(encoding="utf-8-sig"))
    pg226 = json.loads(PG226_REPORT.read_text(encoding="utf-8-sig"))
    pg227 = json.loads(PG227_REPORT.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for result in pg224.get("results", []):
        sent = bool((result.get("ai") or {}).get("sent"))
        rows.append(_base_row(result=result, source="pg224_real_surface_projection", diagnosis="oracle_unavailable" if sent else "inconclusive", grounding_status="projection_only"))
    for result in pg226.get("results", []):
        rows.append(_base_row(result=result, source="pg226_typed_sql_result", diagnosis="confirmed_local_effect", grounding_status="typed_sql_result"))
    for result in pg227.get("results", []):
        route = str(result.get("route", ""))
        grounding = "dom_surface_only" if route.startswith("/vul/xss/") else "redirect_surface_only"
        rows.append(_base_row(result=result, source="pg227_dom_redirect_surface", diagnosis="oracle_unavailable", grounding_status=grounding))
    real = list(rows)
    for index, row in enumerate(real):
        # One explicit self-error per real row makes the model learn the
        # earliest process failure: positive claim before a typed oracle.
        rows.append(_self_error(row, index))
    counts = {
        "pg224_projection_rows": sum(int(row["source"] == "pg224_real_surface_projection") for row in real),
        "pg226_typed_sql_result_rows": sum(int(row["source"] == "pg226_typed_sql_result") for row in real),
        "pg227_dom_redirect_rows": sum(int(row["source"] == "pg227_dom_redirect_surface") for row in real),
        "self_error_counterfactual_rows": len(rows) - len(real),
    }
    return rows, counts


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # New traces use even seeds as fresh seed holdout.  Keep the existing
    # PG-222 split for the inherited corpus, and add two unseen route anchors
    # to prevent a route-specific memorization shortcut.
    heldout_routes = {"/vul/sqli/sqli_search.php", "/vul/xss/xss_reflected_get.php"}
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row.get("seed", 0))
        route = str(row.get("route", ""))
        existing_holdout = seed in {21702, 21802, 22002, 22102} or route in {"/vul/sqli/sqli_blind_b.php", "/vul/sqli/sqli_blind_t.php", "/vul/sqli/sqli_x.php"}
        new_holdout = seed in {22402, 22602, 22702} or route in heldout_routes
        (holdout if existing_holdout or new_holdout else train).append(row)
    if not train or not holdout:
        raise RuntimeError("PG-228 split produced an empty partition")
    return train, holdout


def _safe(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"raw_payload", "payload", "raw_response", "response_body"}}


def main() -> int:
    torch.manual_seed(228)
    base_rows = PG222._build_rows()
    new_rows, new_counts = _new_rows()
    rows = base_rows + new_rows
    train_rows, holdout_rows = _split(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG191._build_model("xxl", vocabulary, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_context = PG223._frozen_context(train_rows, vocabulary, base, device)
    hold_context = PG223._frozen_context(holdout_rows, vocabulary, base, device)
    frozen_count = int(sum(parameter.numel() for parameter in base.parameters()))
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: LargeProblemDiagnoserAdapter | None = None
    for hidden_dim in (64, 128, 256):
        torch.manual_seed(228 + hidden_dim)
        model = LargeProblemDiagnoserAdapter(d_model=int(train_context.shape[1]), hidden_dim=hidden_dim).to(device)
        result = train_large_adapter(model, train_context, hold_context, train_rows, holdout_rows, epochs=100, learning_rate=1e-3)
        result.update({"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "frozen_parameter_count": frozen_count, "device": str(device)})
        variants.append(result)
        key = (result["holdout"]["guarded_positive_false_accept_count"], -result["holdout"]["guarded_diagnosis_accuracy"], -result["holdout"]["next_step_accuracy"])
        old_key = None if selected is None else (selected["holdout"]["guarded_positive_false_accept_count"], -selected["holdout"]["guarded_diagnosis_accuracy"], -selected["holdout"]["next_step_accuracy"])
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-228 no adapter selected")
    artifact_dir = ROOT / "artifacts" / "pg228-grounded-diagnoser-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"grounded_diagnoser_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": "pg228-grounded-diagnoser-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "frozen_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {
        "schema_version": "pg228-grounded-diagnoser-dataset-v1",
        "source_datasets": [str(PG222_DATASET.relative_to(ROOT)), str(PG224_REPORT.relative_to(ROOT)), str(PG226_REPORT.relative_to(ROOT)), str(PG227_REPORT.relative_to(ROOT))],
        "source_counts": {"pg222_rows": len(base_rows), **new_counts, "total": len(rows), "train": len(train_rows), "holdout": len(holdout_rows)},
        "rows": [_safe(row) for row in rows],
        "split": {"train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "seed_and_route_holdout": True, "route_identity_as_feature": False, "new_even_seeds_held_out": [22402, 22602, 22702]},
        "contract": {"typed_sql_result_rows_only_payload_grounded": True, "dom_effect_is_not_xss": True, "projection_only_rows_train_abstention": True, "self_error_counterfactuals_marked": True, "evaluator_targets_as_features": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False},
    }
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {
        "protocol_id": "pg-pk-228-grounded-diagnoser-training-v1",
        "schema_version": "pg228-grounded-diagnoser-v1",
        "status": "completed_typed_untyped_self_error_diagnoser_training",
        "device": str(device),
        "frozen_parameter_count": frozen_count,
        "row_counts": {"total": len(rows), "train": len(train_rows), "holdout": len(holdout_rows), "pg222_rows": len(base_rows), **new_counts},
        "label_counts": {"all": dict(Counter(str(row["diagnosis"]) for row in rows)), "train": dict(Counter(str(row["diagnosis"]) for row in train_rows)), "holdout": dict(Counter(str(row["diagnosis"]) for row in holdout_rows))},
        "payload_grounded_eligible_count": sum(int(row.get("payload_grounded_eligible", False)) for row in rows),
        "variants": variants,
        "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "holdout": selected["holdout"]},
        "comparison": {"pg225_report": str((RESEARCH / "pg225_enriched_large_diagnoser_report_v1.json").relative_to(ROOT)), "current_uses_typed_sql_result_rows": True, "current_uses_dom_redirect_failure_rows": True, "current_has_self_error_counterfactuals": True},
        "promotion": {"adapter_promotion_allowed": False, "payload_grounded_catalog_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_generation": False},
        "honesty": {"typed_sql_result_is_local_only": True, "dom_surface_effect_is_not_xss": True, "projection_only_is_not_evidence": True, "self_error_rows_are_synthetic": True, "general_website_capability_not_established": True},
        "safety": {"loopback_only": True, "external_network": False, "database_write": False, "time_delay_used": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg228-grounded-diagnoser-protocol-v1", "objective": "teach a larger process model to identify its own premature positive decisions and abstain when the oracle is unavailable", "sources": ["PG-222 structured diagnostics", "PG-224 projection-only surfaces", "PG-226 typed SQL/result local traces", "PG-227 DOM/redirect surface traces"], "typed_rows_can_be_payload_grounded": True, "dom_and_redirect_rows_are_not_vulnerability_positives": True, "self_error_counterfactuals_marked": True, "seed_and_route_holdout": True, "evaluator_targets_as_features": False, "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg228-grounded-diagnoser-trace-v1", "selected": selected, "variants": variants, "new_counts": new_counts, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-228 grounded diagnoser training", "", f"device={device}; total={len(rows)}; train={len(train_rows)}; holdout={len(holdout_rows)}", f"new PG-224={new_counts['pg224_projection_rows']}; PG-226 typed SQL/result={new_counts['pg226_typed_sql_result_rows']}; PG-227 DOM/redirect={new_counts['pg227_dom_redirect_rows']}; self-error counterfactuals={new_counts['self_error_counterfactual_rows']}", f"selected hidden={selected['hidden_dim']}; guarded holdout accuracy={selected['holdout']['guarded_diagnosis_accuracy']}; next-step accuracy={selected['holdout']['next_step_accuracy']}; guarded positive false accepts={selected['holdout']['guarded_positive_false_accept_count']}", "", "只有 PG-226 同时通过 typed SQL effect 和 result fixture 的行标记为 payload-grounded eligible；PG-224 projection、PG-227 DOM/redirect surface 和 self-error 对照都不产生漏洞阳性或长期记忆。", "" ]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "row_counts": report["row_counts"], "selected": report["selected"], "payload_grounded_eligible_count": report["payload_grounded_eligible_count"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
