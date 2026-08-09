"""PG-99: audit bounded surface novelty and prove unknown-family overlap.

The OOD component is deliberately conservative: it knows only generic
surface summaries and abstains on fingerprints outside the PG-94 design
support.  A second evaluator-only audit checks whether PG42 known positives
and template_injection positives share the same visible fingerprint.  If
they do, no classifier restricted to that projection can both confirm the
known class and strictly abstain on the unknown class.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.surface_novelty_discriminator import (  # noqa: E402
    SCHEMA_VERSION,
    SurfaceNoveltyDiscriminator,
    make_surface_observation,
    observation_fingerprint,
)


TRAIN_TRACE_PATH = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
EVAL_TRACE_PATH = ROOT / "research" / "pg42_independent_semantic_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg99_surface_novelty_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg99_surface_novelty_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg99_surface_novelty_visible_dataset_v1.json"
TRACE_OUT_PATH = ROOT / "research" / "pg99_surface_novelty_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg99_surface_novelty_report_v1.md"
PROTOCOL_ID = "pg-pk-99-surface-novelty-audit-v1"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _phase(step: dict[str, Any]) -> str:
    match = re.search(r"-(screen|confirm|error|timeout)-(?:control|candidate)$", str(step.get("step_id", "")))
    if not match:
        raise ValueError("PG-99 phase is not allow-listed")
    return match.group(1)


def _implementation(step: dict[str, Any]) -> str:
    route = str((step.get("action_manifest") or {}).get("route_template_id", ""))
    match = re.match(r"pg\d+-([A-Za-z0-9]+)-", route)
    if not match:
        raise ValueError("PG-99 implementation id is not bounded")
    return match.group(1)


def _safe_reset(step: dict[str, Any]) -> bool:
    reset = step.get("fresh_reset") or {}
    return bool(reset.get("completed")) and bool(reset.get("fresh_target")) and not bool(reset.get("external_network")) and str(reset.get("transport", "")) == "httpx_loopback"


def _pairs(path: Path, *, source: str) -> list[dict[str, Any]]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(step["step_id"]): step for step in trace.get("steps", [])}
    rows: list[dict[str, Any]] = []
    for candidate in trace.get("steps", []):
        step_id = str(candidate.get("step_id", ""))
        if "-candidate" not in step_id:
            continue
        control = by_id.get(step_id.replace("-candidate", "-control", 1))
        if control is None:
            raise ValueError(f"PG-99 missing matched control for {step_id}")
        action = candidate.get("action_manifest") or {}
        safety = action.get("safety") or {}
        encoding = "->".join(str(value) for value in (action.get("encoding_chain") or []))
        observation = make_surface_observation(
            control.get("response_projection") or {},
            candidate.get("response_projection") or {},
            method=str(action.get("method", "")),
            encoding_class=encoding,
            phase=_phase(candidate),
            safe_probe=bool(safety.get("no_external_network")) and bool(safety.get("does_not_execute")) and bool(safety.get("no_database_write")) and bool(safety.get("no_credential_access")),
        )
        evidence_hash = str(candidate.get("evidence_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
            raise ValueError(f"PG-99 evidence hash is invalid for {step_id}")
        rows.append({
            "observation": observation,
            "fingerprint": observation_fingerprint(observation),
            "source": source,
            "implementation": _implementation(candidate),
            "seed": int(candidate.get("sampling_seed", -1)),
            "family": str(candidate.get("hypothesis", "")),
            "positive": bool((candidate.get("oracle_projection") or {}).get("positive")) and bool((candidate.get("oracle_projection") or {}).get("positive_authority")),
            "evidence_sha256": evidence_hash,
            "fresh_reset": _safe_reset(candidate) and _safe_reset(control),
            "negative_control_matched": str(candidate.get("parent_step_id", "")) == str(control.get("step_id", "")),
        })
    return rows


def _novelty_metrics(rows: Iterable[dict[str, Any]], discriminator: SurfaceNoveltyDiscriminator) -> dict[str, Any]:
    rows = list(rows)
    counts = defaultdict(int)
    for row in rows:
        result = discriminator.predict(row["observation"])
        key = "positive" if row["positive"] else "negative"
        counts[f"{key}_{result['decision']}"] += 1
    positive = sum(1 for row in rows if row["positive"])
    negative = len(rows) - positive
    positive_abstain = counts["positive_novel_surface"]
    negative_abstain = counts["negative_novel_surface"]
    return {
        "count": len(rows),
        "typed_positive_count": positive,
        "typed_negative_count": negative,
        "positive_novel_surface_abstain_rate": round(positive_abstain / positive, 6) if positive else 0.0,
        "negative_novel_surface_abstain_rate": round(negative_abstain / negative, 6) if negative else 0.0,
        "positive_in_domain_count": counts["positive_in_domain"],
        "negative_in_domain_count": counts["negative_in_domain"],
        "all_rows_abstain": bool(rows) and positive_abstain + negative_abstain == len(rows),
        "support_size": discriminator.support_size,
    }


def _overlap_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = {row["fingerprint"] for row in rows if row["positive"] and row["family"] != "template_injection"}
    unknown = {row["fingerprint"] for row in rows if row["positive"] and row["family"] == "template_injection"}
    classes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["positive"]:
            classes[row["fingerprint"]].add("unknown" if row["family"] == "template_injection" else "known")
    conflicts = sorted(fp for fp, labels in classes.items() if labels == {"known", "unknown"})
    unknown_rows = [row for row in rows if row["positive"] and row["family"] == "template_injection"]
    return {
        "known_positive_fingerprint_count": len(known),
        "unknown_positive_fingerprint_count": len(unknown),
        "known_unknown_fingerprint_overlap_count": len(known & unknown),
        "unknown_positive_rows": len(unknown_rows),
        "unknown_rows_with_known_positive_fingerprint": sum(row["fingerprint"] in known for row in unknown_rows),
        "unknown_overlap_rate": round(sum(row["fingerprint"] in known for row in unknown_rows) / len(unknown_rows), 6) if unknown_rows else 0.0,
        "equivalence_class_conflict_count": len(conflicts),
        "impossibility_witness": bool(conflicts),
        "reason": "same visible fingerprint carries known-positive and template_injection-positive oracle outcomes" if conflicts else "no exact conflict observed",
    }


def run() -> dict[str, Any]:
    train_rows = [row for row in _pairs(TRAIN_TRACE_PATH, source="pg94") if row["seed"] in {361, 367}]
    eval_rows = _pairs(EVAL_TRACE_PATH, source="pg42")
    discriminator = SurfaceNoveltyDiscriminator().fit([row["observation"] for row in train_rows])
    novelty = _novelty_metrics(eval_rows, discriminator)
    overlap = _overlap_audit(eval_rows)
    checks = {
        "model_input_has_no_oracle_or_family": all("family" not in row["observation"] for row in train_rows + eval_rows),
        "fresh_reset_per_pair": all(row["fresh_reset"] for row in eval_rows),
        "negative_control_matched": all(row["negative_control_matched"] for row in eval_rows),
        "evidence_hashes_valid": all(re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"]) for row in eval_rows),
        "get_post_covered": sorted({row["observation"]["method"] for row in eval_rows}) == ["GET", "POST"],
        "unknown_overlap_audited": overlap["impossibility_witness"],
        "not_all_abstain": not novelty["all_rows_abstain"],
        "known_positive_not_all_abstain": novelty["positive_in_domain_count"] > 0,
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "blocked" if overlap["impossibility_witness"] or not checks["known_positive_not_all_abstain"] else ("passed" if not blocked else "blocked")
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg99-surface-novelty-report-v1",
        "status": status,
        "source": {
            "training_design": "pg94 seeds 361/367",
            "evaluation_source": "pg42 cobalt/quartz seeds 401/409/419",
            "training_excludes_pg42": True,
            "surface_projection_schema": SCHEMA_VERSION,
            "oracle_after_surface_projection": True,
            "training": False,
            "memory_write": False,
        },
        "metrics": {"pg42_novelty": novelty, "pg42_known_unknown_overlap": overlap},
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked + (["known_unknown_visible_equivalence_impossible"] if overlap["impossibility_witness"] else []), "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "novelty_audit_only",
            "reason": "surface novelty is an abstention aid, not a typed vulnerability detector; exact known/unknown overlap blocks family claims",
        },
        "safety": {
            "loopback_only": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_model_input": False,
            "typed_oracle_used_only_after_projection": True,
            "evidence_hashes_verified": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible_rows = []
    trace_rows = []
    for row in eval_rows:
        result = discriminator.predict(row["observation"])
        visible = dict(row["observation"])
        visible["evaluation_source"] = "pg42"
        visible["surface_fingerprint"] = row["fingerprint"]
        visible_rows.append(visible)
        trace_rows.append({
            "trace_id": _digest(row["fingerprint"] + row["evidence_sha256"])[:24],
            "surface_novelty_decision": result["decision"],
            "abstain": result["abstain"],
            "surface_fingerprint": row["fingerprint"],
            "fresh_reset": row["fresh_reset"],
            "negative_control_matched": row["negative_control_matched"],
            "evidence_sha256": row["evidence_sha256"],
        })
    DATASET_PATH.write_text(json.dumps({
        "schema_version": "pg99-surface-novelty-visible-dataset-v1",
        "dataset_id": "pg99-surface-novelty-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "surface_fingerprint_is_bounded_hash": True,
            "visible_projection_schema": SCHEMA_VERSION,
        },
        "training_excludes_pg42": True,
        "rows": visible_rows,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUT_PATH.write_text(json.dumps({
        "schema_version": "pg99-surface-novelty-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "surface_projection_schema": SCHEMA_VERSION,
        "steps": trace_rows,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg99-surface-novelty-protocol-v1",
        "purpose": "determine whether bounded surface novelty can support unknown-family abstention without family leakage",
        "training_contract": {"source": "pg94 seeds 361/367", "pg42_excluded": True, "oracle_visible": False},
        "evaluation_contract": {"source": "pg42 cobalt/quartz seeds 401/409/419", "family_labels_used_after_projection_only": True},
        "visible_fields": ["method", "encoding_class", "phase", "safe_probe", "baseline_summary", "candidate_summary"],
        "forbidden_fields": ["family", "oracle_projection", "marker", "raw_probe", "raw_response", "route_identity", "hash identifiers"],
        "discriminator": {"type": "exact_bounded_support_ood", "in_domain_action": "continue_to_typed_oracle", "novel_action": "abstain"},
        "impossibility_check": {"requires_known_unknown_fingerprint_overlap_audit": True, "promotion_on_overlap": False},
        "safety_contract": {"loopback_only": True, "get_post_required": True, "fresh_reset_required": True, "negative_control_required": True, "evidence_sha256_required": True},
        "result": {"status": status, "blocking_reasons": report["capability_gate"]["blocking_reasons"]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-99 Surface Novelty / OOD 审计\n\n"
        f"状态：`{status}`；PG42 novelty 全量弃权：`{novelty['all_rows_abstain']}`；已知/未知正例指纹重叠率：`{overlap['unknown_overlap_rate']}`。\n\n"
        f"等价类冲突数：`{overlap['equivalence_class_conflict_count']}`；结论：`{overlap['reason']}`。\n\n"
        "该组件只负责发现表面新颖性，不能替代 typed oracle，也不会生成训练样本或长期记忆。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    novelty = result["metrics"]["pg42_novelty"]
    overlap = result["metrics"]["pg42_known_unknown_overlap"]
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "support_size": novelty["support_size"],
        "all_rows_abstain": novelty["all_rows_abstain"],
        "known_unknown_overlap_rate": overlap["unknown_overlap_rate"],
        "equivalence_class_conflict_count": overlap["equivalence_class_conflict_count"],
        "impossibility_witness": overlap["impossibility_witness"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
