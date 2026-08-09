"""PG-98: frozen neural goal/label proposal on an independent implementation.

The decoder is trained only on the PG-94 design split after a universal
canonical delta projection is applied.  It is then frozen and evaluated on
the previously unused PG-42 independent semantic fixture (cobalt/quartz,
three seeds, GET/POST, and template_injection as an unknown family).
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.canonical_delta_projection import SCHEMA_VERSION as DELTA_SCHEMA, canonical_delta_tokens  # noqa: E402
from app.goal_label_decoder import NeuralGoalLabelDecoder  # noqa: E402


TRAIN_TRACE_PATH = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
EVAL_TRACE_PATH = ROOT / "research" / "pg42_independent_semantic_trace_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg98_cross_implementation_neural_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg98_cross_implementation_neural_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg98_cross_implementation_neural_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg98_cross_implementation_visible_dataset_v1.json"
TRACE_OUT_PATH = ROOT / "research" / "pg98_cross_implementation_trace_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg98-cross-implementation-neural-proposal" / "model.pt"
MARKDOWN_PATH = ROOT / "research" / "pg98_cross_implementation_neural_report_v1.md"
PROTOCOL_ID = "pg-pk-98-cross-implementation-neural-proposal-v1"
SEED = 20260803
CANONICAL_VOCABULARY = (
    "DELTA_EFFECT_INCREASE",
    "DELTA_EFFECT_DECREASE",
    "DELTA_EFFECT_CHANGE",
    "DELTA_RESPONSE_INCREASE",
    "DELTA_RESPONSE_DECREASE",
    "DELTA_RESPONSE_CHANGE",
    "DELTA_BOUNDARY_INCREASE",
    "DELTA_BOUNDARY_DECREASE",
    "DELTA_BOUNDARY_CHANGE",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _phase(step: dict[str, Any]) -> str:
    match = re.search(r"-(screen|confirm|error|timeout)-(?:control|candidate)$", str(step.get("step_id", "")))
    if not match:
        raise ValueError(f"unable to derive an allow-listed phase from {step.get('step_id')}")
    return match.group(1)


def _implementation(step: dict[str, Any]) -> str:
    route = str((step.get("action_manifest") or {}).get("route_template_id", ""))
    match = re.match(r"pg\d+-([A-Za-z0-9]+)-", route)
    if not match:
        raise ValueError("route implementation is not bounded")
    return match.group(1)


def _safe_reset(step: dict[str, Any]) -> bool:
    reset = step.get("fresh_reset") or {}
    return (
        bool(reset.get("completed"))
        and bool(reset.get("fresh_target"))
        and not bool(reset.get("external_network"))
        and str(reset.get("transport", "")) == "httpx_loopback"
    )


def _visible(control: dict[str, Any], candidate: dict[str, Any], *, source: str) -> dict[str, Any]:
    action = candidate.get("action_manifest") or {}
    safety = action.get("safety") or {}
    method = str(action.get("method", ""))
    if method not in {"GET", "POST"}:
        raise ValueError("PG-98 only accepts GET/POST")
    encoding = tuple(str(value) for value in (action.get("encoding_chain") or []))
    if not encoding:
        raise ValueError("empty encoding chain")
    tokens = list(canonical_delta_tokens(control.get("response_projection") or {}, candidate.get("response_projection") or {}))
    return {
        "schema_version": "pg98-visible-pair-v1",
        "method": method,
        "encoding_class": "->".join(encoding),
        "phase": _phase(candidate),
        "safe_probe": bool(safety.get("no_external_network"))
        and bool(safety.get("does_not_execute"))
        and bool(safety.get("no_database_write"))
        and bool(safety.get("no_credential_access")),
        "delta_tokens": tokens,
        "delta_count": len(tokens),
        "has_observed_change": bool(tokens),
        "row_id": _digest(f"{source}|{candidate.get('step_id', '')}")[:24],
        "context_group": _digest(f"{source}|{_implementation(candidate)}|{candidate.get('sampling_seed')}|{method}|{_phase(candidate)}")[:24],
    }


def _pairs(path: Path, *, source: str) -> list[dict[str, Any]]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(step["step_id"]): step for step in trace.get("steps", [])}
    result: list[dict[str, Any]] = []
    for candidate in trace.get("steps", []):
        step_id = str(candidate.get("step_id", ""))
        if "-candidate" not in step_id:
            continue
        control = by_id.get(step_id.replace("-candidate", "-control", 1))
        if control is None:
            raise ValueError(f"missing matched negative control for {step_id}")
        visible = _visible(control, candidate, source=source)
        evidence = str(candidate.get("evidence_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", evidence):
            raise ValueError(f"missing canonical evidence hash for {step_id}")
        oracle = dict(candidate.get("oracle_projection") or {})
        result.append({
            "visible": visible,
            "source": source,
            "implementation": _implementation(candidate),
            "seed": int(candidate.get("sampling_seed", -1)),
            "family": str(candidate.get("hypothesis", "")),
            "phase": visible["phase"],
            "method": visible["method"],
            "oracle": oracle,
            "evidence_sha256": evidence,
            "fresh_reset": _safe_reset(candidate) and _safe_reset(control),
            "episode_id": str(candidate.get("episode_id", "")),
            "negative_control_matched": str(candidate.get("parent_step_id", "")) == str(control.get("step_id", "")),
        })
    return result


def _metric(rows: Iterable[dict[str, Any]], decoder: NeuralGoalLabelDecoder) -> dict[str, Any]:
    rows = list(rows)
    positive = negative = confirm = false_accept = abstain = 0
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_impl: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unknown_count = unknown_abstain = 0
    for item in rows:
        output = decoder.predict(item["visible"])
        positive_authority = bool(item["oracle"].get("positive_authority")) and bool(item["oracle"].get("positive"))
        if positive_authority:
            positive += 1
            by_family[item["family"]][0] += 1
            by_impl[item["implementation"]][0] += 1
            if output["decision"] == "confirm_candidate":
                confirm += 1
                by_family[item["family"]][1] += 1
                by_impl[item["implementation"]][1] += 1
        else:
            negative += 1
            if output["decision"] == "confirm_candidate":
                false_accept += 1
        abstain += int(output["decision"] == "abstain")
        if item["family"] == "template_injection":
            unknown_count += 1
            unknown_abstain += int(output["decision"] == "abstain")
    family_recall = {k: round(v[1] / v[0], 6) if v[0] else 0.0 for k, v in sorted(by_family.items())}
    implementation_recall = {k: round(v[1] / v[0], 6) if v[0] else 0.0 for k, v in sorted(by_impl.items())}
    known_values = [value for family, value in family_recall.items() if family != "template_injection"]
    return {
        "count": len(rows),
        "typed_positive_count": positive,
        "typed_negative_count": negative,
        "confirm_recall": round(confirm / positive, 6) if positive else 0.0,
        "known_family_confirm_recall": round(sum(known_values) / len(known_values), 6) if known_values else 0.0,
        "false_accept_count": false_accept,
        "abstain_count": abstain,
        "not_all_abstain": bool(rows) and abstain < len(rows),
        "family_confirm_recall": family_recall,
        "family_min_confirm_recall": min(family_recall.values()) if family_recall else 0.0,
        "implementation_confirm_recall": implementation_recall,
        "unknown_family_count": unknown_count,
        "unknown_family_strict_abstain": bool(unknown_count) and unknown_abstain == unknown_count,
    }


def run() -> dict[str, Any]:
    train_pairs = _pairs(TRAIN_TRACE_PATH, source="pg94")
    eval_pairs = _pairs(EVAL_TRACE_PATH, source="pg42")
    train_design = [item for item in train_pairs if item["seed"] in {361, 367}]
    if not train_design or not eval_pairs:
        raise RuntimeError("PG-98 missing training design or independent evaluation rows")
    decoder = NeuralGoalLabelDecoder(seed=SEED, epochs=80).fit([item["visible"] for item in train_design], extra_tokens=CANONICAL_VOCABULARY)
    proposal = decoder.proposal(design_row_count=len(train_design))
    proposal["cross_implementation_eval_source"] = "pg42"
    proposal["canonical_delta_projection_schema"] = DELTA_SCHEMA
    proposal["predeclared_grammar_vocabulary"] = list(CANONICAL_VOCABULARY)
    # Recompute the digest after adding provenance fields, before writing it.
    proposal.pop("proposal_sha256", None)
    proposal["proposal_sha256"] = hashlib.sha256(json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = decoder.checkpoint()
    checkpoint["canonical_delta_projection_schema"] = DELTA_SCHEMA
    checkpoint["predeclared_grammar_vocabulary"] = list(CANONICAL_VOCABULARY)
    checkpoint["cross_implementation_eval_source_excluded_from_training"] = True
    torch.save(checkpoint, CHECKPOINT_PATH)

    metrics = {
        "pg94_design": _metric(train_design, decoder),
        "pg42_cross_implementation": _metric(eval_pairs, decoder),
        "pg42_cobalt": _metric([item for item in eval_pairs if item["implementation"] == "cobalt"], decoder),
        "pg42_quartz": _metric([item for item in eval_pairs if item["implementation"] == "quartz"], decoder),
        "pg42_seed_401": _metric([item for item in eval_pairs if item["seed"] == 401], decoder),
        "pg42_seed_409": _metric([item for item in eval_pairs if item["seed"] == 409], decoder),
        "pg42_seed_419": _metric([item for item in eval_pairs if item["seed"] == 419], decoder),
    }
    checks = {
        "proposal_did_not_see_oracle": proposal["proposal_inputs"]["oracle_visible"] is False,
        "proposal_did_not_see_family": proposal["proposal_inputs"]["family_visible"] is False,
        "proposal_did_not_see_raw": proposal["proposal_inputs"]["raw_probe_visible"] is False and proposal["proposal_inputs"]["raw_response_visible"] is False,
        "canonical_delta_schema_declared": proposal["canonical_delta_projection_schema"] == DELTA_SCHEMA,
        "get_post_covered": sorted({item["method"] for item in eval_pairs}) == ["GET", "POST"],
        "fresh_reset_per_pair": all(item["fresh_reset"] for item in eval_pairs),
        "negative_control_matched": all(item["negative_control_matched"] for item in eval_pairs),
        "evidence_hashes_valid": all(re.fullmatch(r"[0-9a-f]{64}", item["evidence_sha256"]) for item in eval_pairs),
        "cross_implementation_recall_min": metrics["pg42_cross_implementation"]["confirm_recall"] >= 0.80,
        "cross_implementation_false_accept_zero": metrics["pg42_cross_implementation"]["false_accept_count"] == 0,
        "seed_min_recall": min(metrics[f"pg42_seed_{seed}"]["confirm_recall"] for seed in (401, 409, 419)) >= 0.75,
        "implementation_min_recall": min(metrics[f"pg42_{impl}"]["confirm_recall"] for impl in ("cobalt", "quartz")) >= 0.75,
        "unknown_family_strict_abstain": metrics["pg42_cross_implementation"]["unknown_family_strict_abstain"],
        "not_all_abstain": metrics["pg42_cross_implementation"]["not_all_abstain"],
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg98-cross-implementation-neural-report-v1",
        "status": status,
        "source": {
            "training_trace": str(TRAIN_TRACE_PATH.relative_to(ROOT)),
            "training_design": "pg94 seeds 361/367 only",
            "evaluation_trace": str(EVAL_TRACE_PATH.relative_to(ROOT)),
            "evaluation_source": "pg42 cobalt/quartz seeds 401/409/419",
            "cross_implementation_eval_source_excluded_from_training": True,
            "canonical_delta_projection_schema": DELTA_SCHEMA,
            "device": str(decoder.device),
            "oracle_after_proposal": True,
            "training": "self_supervised_visible_delta_only",
            "memory_write": False,
        },
        "proposal": {
            "proposal_file": str(PROPOSAL_PATH.relative_to(ROOT)),
            "proposal_sha256": proposal["proposal_sha256"],
            "architecture": proposal["model"]["architecture"],
            "training_row_count": len(train_design),
            "evaluation_row_count": len(eval_pairs),
        },
        "metrics": metrics,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "cross_implementation_candidate_quarantined",
            "reason": "PG42 is a fresh independent evaluation source; unknown-family abstention and Codex review remain mandatory",
        },
        "safety": {
            "loopback_only": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_proposal_input": False,
            "typed_oracle_labels_used_only_for_evaluation": True,
            "evidence_hashes_verified": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    visible_rows = []
    trace_rows = []
    for item in eval_pairs:
        output = decoder.predict(item["visible"])
        row = dict(item["visible"])
        row["evaluation_source"] = "pg42"
        row["neural_auto_label"] = output["label_id"]
        visible_rows.append(row)
        trace_rows.append({
            "trace_id": item["visible"]["row_id"],
            "source": "pg42",
            "implementation": item["implementation"],
            "seed": item["seed"],
            "method": item["method"],
            "phase": item["phase"],
            "neural_auto_label": output["label_id"],
            "decision": output["decision"],
            "unknown_tokens": output.get("unknown_tokens", []),
            "fresh_reset": item["fresh_reset"],
            "negative_control_matched": item["negative_control_matched"],
            "evidence_sha256": item["evidence_sha256"],
            "safety": item["visible"]["safe_probe"],
        })
    DATASET_PATH.write_text(json.dumps({
        "schema_version": "pg98-cross-implementation-visible-dataset-v1",
        "dataset_id": "pg98-cross-implementation-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "canonical_delta_projection_schema": DELTA_SCHEMA,
            "visible_fields": ["method", "encoding_class", "phase", "delta_tokens", "delta_count", "has_observed_change", "safe_probe"],
        },
        "training_excludes_pg42": True,
        "proposal_sha256": proposal["proposal_sha256"],
        "rows": visible_rows,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUT_PATH.write_text(json.dumps({
        "schema_version": "pg98-cross-implementation-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "steps": trace_rows,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg98-cross-implementation-neural-protocol-v1",
        "purpose": "frozen neural automatic goal/label proposal on a fresh independent implementation",
        "training_contract": {"source": "pg94", "seeds": [361, 367], "pg42_excluded": True, "typed_oracle_visible": False},
        "evaluation_contract": {"source": "pg42", "implementations": ["cobalt", "quartz"], "seeds": [401, 409, 419], "families_hidden_to_model": True},
        "canonical_delta_projection": {"schema": DELTA_SCHEMA, "field_names_discarded": True, "marker_and_oracle_discarded": True},
        "safety_contract": {"loopback_only": True, "get_post_required": True, "fresh_reset_required": True, "negative_control_required": True, "evidence_sha256_required": True},
        "gate": {"minimum_cross_implementation_recall": 0.80, "minimum_seed_recall": 0.75, "minimum_implementation_recall": 0.75, "false_accept_count": 0, "unknown_family_strict_abstain": True, "not_all_abstain": True, "promotion_on_pass": False},
        "result": {"status": status, "blocking_reasons": blocked},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-98 跨实现神经目标/标签盲测\n\n"
        f"状态：`{status}`；PG94 训练设计 → PG42 cobalt/quartz 冻结盲测；表示：`{DELTA_SCHEMA}`。\n\n"
        f"跨实现召回：`{metrics['pg42_cross_implementation']['confirm_recall']}`；误报：`{metrics['pg42_cross_implementation']['false_accept_count']}`；未知族严格弃权：`{metrics['pg42_cross_implementation']['unknown_family_strict_abstain']}`。\n\n"
        f"阻塞项：{', '.join(blocked) if blocked else '无'}。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    metrics = result["metrics"]["pg42_cross_implementation"]
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "device": result["source"]["device"],
        "cross_implementation_recall": metrics["confirm_recall"],
        "known_family_recall": metrics["known_family_confirm_recall"],
        "false_accept_count": metrics["false_accept_count"],
        "unknown_family_strict_abstain": metrics["unknown_family_strict_abstain"],
        "get_post_covered": result["capability_gate"]["checks"]["get_post_covered"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
