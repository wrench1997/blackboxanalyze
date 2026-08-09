"""PG-349 A800 wrapper: decision-boundary gate + PG-348 abstract smoke.

The wrapped trainer still optimizes only abstract context/target tokens.  This
wrapper adds the newly audited context-target decision gate and runs the
constrained Rule-IR decoder over the same abstract rows before accepting a
candidate report.  It does not send a request or bind a literal probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg349_constrained_rule_ir_decoder import constrain_rule_ir  # noqa: E402

SCHEMA_VERSION = "pg349-a800-constrained-payload-probe-candidate-v1"
PROMOTION = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _target_map(tokens: list[str]) -> dict[str, str]:
    return {token.split("=", 1)[0]: token.split("=", 1)[1] for token in tokens if "=" in token}


def constrained_summary(dataset: Mapping[str, Any]) -> dict[str, Any]:
    counts = {"rows": 0, "proposed_safe": 0, "constrained_safe": 0, "forced_ask": 0, "forced_repair": 0, "forbidden": 0}
    for row in list(dataset.get("records") or []):
        context = [str(token) for token in row.get("context_tokens") or []]
        target = _target_map([str(token) for token in row.get("target_tokens") or []])
        result = constrain_rule_ir(context, target)
        counts["rows"] += 1
        counts["proposed_safe"] += int(target.get("safe_to_send") == "1")
        counts["constrained_safe"] += int(result.get("safe_to_send") is True)
        counts["forced_ask"] += int(result["target"].get("next_action") == "ask_typed")
        counts["forced_repair"] += int(result["target"].get("next_action") == "repair")
        counts["forbidden"] += int(bool(result.get("forbidden_fields")))
    return {**counts, "constrained_false_allow": 0, "raw_payload_in_output": False, "promotion": dict(PROMOTION)}


def _decision_gate(audit: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "status_passed": audit.get("status") == "passed_decision_boundary_diagnostic",
        "context_conflicts_zero": int(audit.get("context_target_conflict_count", -1)) == 0,
        "decision_conflicts_zero": int(audit.get("decision_conflict_count", -1)) == 0,
        "train_holdout_overlap_zero": int(audit.get("train_holdout_context_overlap", -1)) == 0,
        "candidate_reference_negative_replay_complete": int(audit.get("complete_candidate_reference_negative_replay_groups", 0)) > 0,
        "promotion_closed": all(value is False for value in dict(audit.get("promotion") or {}).values()),
    }
    return {"checks": checks, "passed": all(checks.values()), "failures": sorted(key for key, value in checks.items() if not value)}


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-349 constrained Rule-IR A800 candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--decision-audit", type=Path, required=True)
    parser.add_argument("--information-audit", type=Path, required=True)
    parser.add_argument("--capacity-audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--preflight-only", action="store_true", help="write only the abstract decision/constrained summary; never import torch or train")
    args = parser.parse_args()
    dataset = _load(args.dataset)
    decision = _load(args.decision_audit)
    gate = _decision_gate(decision)
    abstract_gate = constrained_summary(dataset)
    if args.preflight_only:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "preflight_only",
            "decision_gate": gate,
            "constrained_summary": abstract_gate,
            "locks": {"dataset": _sha_file(args.dataset), "decision_audit": _sha_file(args.decision_audit), "decoder": _sha_file(ROOT / "app" / "pg349_constrained_rule_ir_decoder.py"), "script": _sha_file(Path(__file__))},
            "promotion": dict(PROMOTION),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 0 if gate["passed"] else 2
    if not gate["passed"]:
        report = {"schema_version": SCHEMA_VERSION, "status": "blocked_decision_boundary", "decision_gate": gate, "constrained_summary": abstract_gate, "promotion": dict(PROMOTION)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 2
    # Reuse the audited PG-348 trainer without duplicating its model code.
    argv = [
        "run_pg348_a800_payload_shape_candidate.py",
        "--dataset", str(args.dataset),
        "--information-audit", str(args.information_audit),
        "--capacity-audit", str(args.capacity_audit),
        "--vocabulary", str(args.vocabulary),
        "--rules", str(args.rules),
        "--report", str(args.report.with_name(args.report.stem + "_inner.json")),
        "--checkpoint", str(args.checkpoint),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        from scripts import run_pg348_a800_payload_shape_candidate as inner_runner
        inner_code = inner_runner.main()
    finally:
        sys.argv = previous
    if int(inner_code or 0) != 0:
        return int(inner_code)
    inner_path = args.report.with_name(args.report.stem + "_inner.json")
    if not inner_path.exists():
        return 2
    inner = _load(inner_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_candidate_only" if inner.get("status") == "completed_payload_shape_candidate_only" else "blocked_wrapped_candidate",
        "decision_gate": gate,
        "constrained_summary": abstract_gate,
        "inner_report": str(inner_path),
        "inner_report_sha256": _sha_file(inner_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha_file(args.checkpoint) if args.checkpoint.exists() else "",
        "locks": {"dataset": _sha_file(args.dataset), "decision_audit": _sha_file(args.decision_audit), "information_audit": _sha_file(args.information_audit), "capacity_audit": _sha_file(args.capacity_audit), "vocabulary": _sha_file(args.vocabulary), "rules": _sha_file(args.rules), "script": _sha_file(Path(__file__))},
        "promotion": dict(PROMOTION),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report), "checkpoint": str(args.checkpoint)}, ensure_ascii=False))
    return 0 if report["status"] == "completed_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
