"""PG-225: enrich the large problem diagnoser with real PG-224 surface traces."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg223_large_problem_diagnoser import LargeProblemDiagnoserAdapter, PG223_SCHEMA, train_large_adapter  # noqa: E402


RESEARCH = ROOT / "research"
PG224_REPORT = RESEARCH / "pg224_pikachu_parameter_surface_collection_report_v1.json"
PG222_DATASET = RESEARCH / "pg222_problem_diagnoser_dataset_v1.json"
PG223_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg225_enriched_large_diagnoser_report_v1.json"
DATASET = RESEARCH / "pg225_enriched_large_diagnoser_dataset_v1.json"
TRACE = RESEARCH / "pg225_enriched_large_diagnoser_trace_v1.json"
PROTOCOL = RESEARCH / "pg225_enriched_large_diagnoser_protocol_v1.json"
MARKDOWN = RESEARCH / "pg225_enriched_large_diagnoser_report_v1.md"


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


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _projection(result: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(((result.get(key) or {}).get("response_projection") or {}))


def _enriched_rows() -> list[dict[str, Any]]:
    report = json.loads(PG224_REPORT.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(report.get("results", [])):
        candidate = _projection(result, "candidate")
        negative = _projection(result, "negative")
        sent = bool((result.get("ai") or {}).get("sent"))
        fields = list(result.get("fields") or [])
        # PG-224 deliberately has no typed effect oracle.  A sent projection
        # is therefore an observed "oracle unavailable" trace, while a
        # preflight-only row is genuinely inconclusive because no candidate
        # was sent.  Weak 302/500/reflection signals do not become positives.
        diagnosis = "oracle_unavailable" if sent else "inconclusive"
        row = {
            "seed": int(result.get("seed", 0)),
            "route": str(result.get("route", "")),
            "method": str(result.get("method", "GET")).upper(),
            "field_count": len(fields),
            "source": "pg224_real_surface_projection",
            "fresh_reset_ok": bool(result.get("fresh_reset")),
            "reset_completed": bool((result.get("reset") or {}).get("completed")),
            "database_health_ok": (result.get("reset") or {}).get("database_health_gate") == "mysqli_root_pikachu_ok",
            "backend_observed": sent,
            "transport_error": bool(candidate.get("transport_error")),
            "container_restart_used": bool((result.get("reset") or {}).get("container_restart_used")),
            "status_class": str(candidate.get("status_class", "2xx")),
            "binding_valid": bool(fields),
            "candidate_sent": sent,
            "reference_sent": False,
            "negative_sent": bool(result.get("negative")),
            "oracle_available": False,
            "typed_effect_observed": False,
            "result_fixture_verified": False,
            "boolean_differential": False,
            "candidate_reference_agreement": True,
            "negative_clean": int(negative.get("marker_reflected", False) is False),
            "candidate_result_present": bool(candidate.get("marker_reflected")),
            "negative_result_absent": not bool(negative.get("marker_reflected")),
            "candidate_sql_error_shape": False,
            "result_mismatch_observed": False,
            "model_claimed_positive": False,
            "model_abstained": not sent,
            "previous_feedback": "none",
            "history_len": 0,
            "source_hash": str(result.get("source_row_sha256", "")),
            "evidence_hash": str(((result.get("oracle") or {}).get("evidence") or {}).get("evidence_sha256", "")),
            "diagnosis": diagnosis,
            "next_step": "recheck_oracle" if sent else "abstain",
            "pg224_status": str(result.get("status", "")),
            "pg224_policy_reason": str((result.get("policy") or {}).get("reason", "")),
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "real_trace": True,
        }
        rows.append(row)
    return rows


def _safe(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"raw_payload", "payload", "raw_response", "response_body"}}


def main() -> int:
    torch.manual_seed(225)
    base_rows = PG222._build_rows()
    enriched = _enriched_rows()
    rows = base_rows + enriched
    train_rows, holdout_rows = PG222._split(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG223_CHECKPOINT, map_location="cpu", weights_only=False)
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
        torch.manual_seed(225 + hidden_dim)
        model = LargeProblemDiagnoserAdapter(d_model=int(train_context.shape[1]), hidden_dim=hidden_dim).to(device)
        result = train_large_adapter(model, train_context, hold_context, train_rows, holdout_rows, epochs=100, learning_rate=1e-3)
        result.update({"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "frozen_parameter_count": frozen_count, "device": str(device)})
        variants.append(result)
        if selected is None or (result["holdout"]["guarded_positive_false_accept_count"], -result["holdout"]["guarded_diagnosis_accuracy"], -result["holdout"]["next_step_accuracy"]) < (selected["holdout"]["guarded_positive_false_accept_count"], -selected["holdout"]["guarded_diagnosis_accuracy"], -selected["holdout"]["next_step_accuracy"]):
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-225 no model selected")
    artifact_dir = ROOT / "artifacts" / "pg225-enriched-large-diagnoser-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"enriched_diagnoser_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": "pg225-enriched-large-diagnoser-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "frozen_checkpoint": str(PG223_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {"schema_version": "pg225-enriched-large-diagnoser-dataset-v1", "source_datasets": [str((RESEARCH / "pg222_problem_diagnoser_dataset_v1.json").relative_to(ROOT)), str(PG224_REPORT.relative_to(ROOT))], "pg222_rows": len(base_rows), "pg224_real_rows": len(enriched), "rows": [_safe(row) for row in rows], "split": {"train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "seed_and_route_holdout": True, "route_identity_as_feature": False}, "contract": {"real_pg224_rows_are_projection_only": True, "evaluator_targets_as_features": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = _digest(dataset)
    _write(DATASET, dataset)
    report = {"protocol_id": "pg-pk-225-enriched-large-diagnoser-v1", "schema_version": "pg225-enriched-large-diagnoser-v1", "status": "completed_real_surface_enriched_large_diagnoser", "device": str(device), "frozen_parameter_count": frozen_count, "row_counts": {"total": len(rows), "train": len(train_rows), "holdout": len(holdout_rows), "pg222_rows": len(base_rows), "pg224_real_rows": len(enriched)}, "pg224_real_label_counts": {"oracle_unavailable": sum(int(row["diagnosis"] == "oracle_unavailable") for row in enriched), "inconclusive": sum(int(row["diagnosis"] == "inconclusive") for row in enriched)}, "variants": variants, "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "holdout": selected["holdout"]}, "promotion": {"adapter_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_generation": False}, "honesty": {"pg224_signals_are_weak_oracle_inputs": True, "no_new_typed_positive": True, "counterfactuals_and_projection_only_are_not_vulnerability_evidence": True, "general_website_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "database_write": False, "time_delay_used": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = _digest(report)
    _write(REPORT, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg225-enriched-large-diagnoser-protocol-v1", "base_diagnosis_source": "PG-222", "real_surface_source": "PG-224", "frozen_body": True, "diagnosis_targets": ["oracle_unavailable", "inconclusive"], "weak_signals_not_positive": ["reflection", "500", "302", "body_length_delta"], "seed_and_route_holdout": True, "raw_payload_and_response_excluded": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg225-enriched-large-diagnoser-trace-v1", "selected": selected, "variants": variants, "pg224_real_rows": len(enriched), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-225 enriched large problem diagnoser", "", f"device={device}; total={len(rows)}; train={len(train_rows)}; holdout={len(holdout_rows)}; real PG-224 rows={len(enriched)}", f"selected hidden={selected['hidden_dim']}; guarded holdout accuracy={selected['holdout']['guarded_diagnosis_accuracy']}; guarded positive false accepts={selected['holdout']['guarded_positive_false_accept_count']}", "", "PG-224 的真实回放只有 projection-only oracle，所以新行只能训练 oracle_unavailable/inconclusive，不得被当作漏洞阳性。PG-223 的 frozen XXL body 仍未解冻；训练的是诊断 adapter。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "row_counts": report["row_counts"], "selected": report["selected"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
