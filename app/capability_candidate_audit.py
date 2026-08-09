"""Offline audit for model-capability candidate reports.

This module is deliberately narrower than :mod:`app.model_capability_gate`.
The capability gate evaluates an already structured evidence object; this
auditor decides whether an experiment report contains that evidence at all.
It never executes a trainer, loads a checkpoint, performs network I/O, or
infers dataset evidence from a convenient ``status``/``accuracy`` field.

That last property is important: a frozen-checkpoint report that happens to
contain a family holdout score is not automatically a PG-30 capability
claim.  The producer must explicitly attest the independent dataset rows and
their lineage in ``capability_evidence`` (or provide the same object at the
top level).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from app.model_capability_gate import evaluate_model_capability


AUDIT_SCHEMA = "sift-capability-candidate-audit-v1"
_EVIDENCE_KEYS = (
    "capability_evidence",
    "model_capability_evidence",
    "capability_gate_input",
)
_DIRECT_EVIDENCE_KEYS = frozenset(
    {
        "dataset_tests",
        "baseline_metrics",
        "candidate_metrics",
        "unit_tests_passed",
        "oracle_validated",
        "data_lineage_complete",
        "authorized_sources_attested",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load one JSON report without executing any report-provided code."""

    try:
        if not path.is_file():
            return None, "report_not_found"
        # Reports are projections, not data dumps.  Keep the offline auditor
        # bounded so a mistaken path cannot turn this command into an unbounded
        # ingestion job.
        if path.stat().st_size > 32 * 1024 * 1024:
            return None, "report_too_large"
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "report_unreadable"
    if not isinstance(value, dict):
        return None, "report_root_not_object"
    return value, None


def _explicit_evidence(report: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return only explicitly labelled PG-30 evidence.

    Existing training/evaluation reports commonly contain ``holdout`` and
    ``accuracy`` fields, but those are intentionally not interpreted here.
    Doing so would manufacture independent-dataset evidence and undermine the
    gate this tool is meant to protect.
    """

    for key in _EVIDENCE_KEYS:
        if key in report:
            value = report[key]
            if not isinstance(value, dict):
                return None, f"{key}_not_object"
            return value, None
    # A producer may emit the PG-30 object as the report itself.  Requiring the
    # complete identifying subset prevents a generic report with one
    # ``dataset_tests`` field from being silently reinterpreted.
    if _DIRECT_EVIDENCE_KEYS.issubset(report):
        return report, None
    return None, "explicit_capability_evidence_missing"


def audit_capability_reports(
    paths: Iterable[str | Path],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit report files and evaluate one explicit PG-30 evidence object.

    Multiple paths are allowed for a batch audit, but evidence is never
    merged across reports.  At most one report may claim capability evidence;
    combining independently produced reports would make source/seed
    separation unverifiable and is therefore fail-closed.
    """

    path_values = [Path(value) for value in paths]
    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    structural_reasons: list[str] = []
    for path in path_values:
        report, load_error = _load_report(path)
        item: dict[str, Any] = {
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
            "loaded": report is not None,
        }
        if load_error:
            item["error"] = load_error
            structural_reasons.append(f"{path.name}:{load_error}")
            summaries.append(item)
            continue
        assert report is not None
        item["schema_version"] = report.get("schema_version")
        item["status"] = report.get("status")
        evidence, evidence_error = _explicit_evidence(report)
        item["explicit_capability_evidence"] = evidence is not None
        if evidence_error:
            item["evidence_error"] = evidence_error
            if evidence_error != "explicit_capability_evidence_missing":
                structural_reasons.append(f"{path.name}:{evidence_error}")
        else:
            assert evidence is not None
            candidates.append({"path": path, "evidence": evidence})
        summaries.append(item)

    if len(candidates) > 1:
        structural_reasons.append("multiple_capability_evidence_reports_not_merged")
    if not candidates:
        structural_reasons.append("no_explicit_independent_dataset_evidence")

    gate_report: dict[str, Any] | None = None
    if len(candidates) == 1 and not structural_reasons:
        # The gate performs strict schema validation and returns a structured
        # ``blocked``/``no_proven_gain`` result rather than authorising a
        # training action.  Validation errors are converted to a blocked
        # audit result below so this CLI remains fail-closed.
        try:
            gate_report = evaluate_model_capability(candidates[0]["evidence"], policy=policy)
        except (TypeError, ValueError, KeyError) as error:
            structural_reasons.append(f"capability_evidence_invalid:{type(error).__name__}")

    if structural_reasons:
        status = "blocked"
        reasons = sorted(set(structural_reasons + (["independent_dataset_tests_required"] if not candidates else [])))
        gate_summary = None
    elif gate_report is not None:
        status = str(gate_report["status"])
        reasons = list(gate_report["reasons"])
        gate_summary = gate_report
    else:  # defensive; the branches above should cover this
        status = "blocked"
        reasons = ["audit_inconclusive"]
        gate_summary = None

    allowed = status == "pass"
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": status,
        "claim_allowed": allowed,
        "training_allowed": allowed,
        "memory_promotion_allowed": allowed,
        "reasons": sorted(set(reasons)),
        "reports": summaries,
        "evaluated_gate": gate_summary,
        "actions": {
            "trainer_invoked": False,
            "checkpoint_written": False,
            "training_dataset_generated": False,
            "memory_write_attempted": False,
        },
        "unit_tests_are_not_capability_evidence": True,
        "independent_dataset_tests_required": True,
    }


__all__ = ["AUDIT_SCHEMA", "audit_capability_reports"]

