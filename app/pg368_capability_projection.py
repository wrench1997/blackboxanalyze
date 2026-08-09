"""Read-only capability projection for the PG-367 demonstration.

PG-367 has two different kinds of evidence:

* the model-selected abstract Rule-IR rows, which were bound by an
  allowlisted local adapter and replayed against a disposable evaluator; and
* the complete WAF staircase replay, which exercises the GET/POST,
  candidate/reference/negative and repair contracts without claiming that a
  generic web site is vulnerable.

This module turns those reports into a small, presentation-safe object.  It
never starts a target, loads a checkpoint, sends a request, or exposes a wire
value.  Report paths are accepted only by :func:`load_pg368_capability`; the
pure :func:`project_pg368_capability` function is useful to the research UI
and to tests.

The projection intentionally keeps provenance hashes and abstract slots, but
drops binding templates, request paths, canaries, raw values, response bodies,
and evaluator literals.  All promotion flags are forced to ``False``: this is
evidence for a demo, not a general vulnerability claim or a training/gold
dataset promotion.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pg368-capability-projection-v1"
MODEL_REPORT_SCHEMA = "pg367-model-binder-replay-v1"
WAF_REPORT_SCHEMA = "pg367-waf-staircase-replay-v1"
_HASH_KEYS = frozenset(
    {
        "model_checkpoint_sha256",
        "dataset_sha256",
        "registry_sha256",
        "report_sha256",
        "policy_hashes",
        "evidence_sha256",
        "source_hash",
        "checkpoint_sha256",
    }
)
_SLOT_KEYS = (
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "probe_variant_ref",
    "payload_shape_ref",
    "oracle_ref",
    "syntax_category_ref",
    "safe_to_send",
)
_FORBIDDEN_TEXT = (
    "pg367-runtime-canary",
    "http://127.0.0.1:",
    "https://",
    "raw_payload",
    "response_body",
    "oracle_answer",
    "evaluator_answer",
)
_PROMOTION_KEYS = (
    "training_allowed",
    "memory_promotion_allowed",
    "payload_catalog_promotion_allowed",
    "vulnerability_claim_allowed",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # A presentation projection must not accidentally become an ingestion
    # path for a huge/raw report.
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"report is too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"report root must be an object: {path}")
    return value


def _text_scan(value: Any, *, path: str = "") -> list[str]:
    """Return forbidden presentation strings, without retaining their values."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_text_scan(child, path=f"{path}.{key}" if path else str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_text_scan(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        folded = value.casefold()
        for marker in _FORBIDDEN_TEXT:
            if marker.casefold() in folded:
                found.append(path or "<value>")
                break
    return found


def _as_bool(value: Any) -> bool:
    return value is True or (isinstance(value, (int, float)) and value == 1)


def _promotion_false(report: Mapping[str, Any]) -> dict[str, bool]:
    """Project promotion state conservatively and fail closed.

    A malformed or missing source flag is represented as ``False``.  The
    projection itself never grants authority based on a report field.
    """

    source = report.get("promotion")
    if not isinstance(source, Mapping):
        source = {}
    return {key: False for key in _PROMOTION_KEYS if key in source or key == "vulnerability_claim_allowed"}


def _project_slots(rows: Sequence[Any], *, limit: int = 12) -> dict[str, Any]:
    slot_counts: Counter[str] = Counter()
    slot_value_counts: dict[str, Counter[str]] = {key: Counter() for key in _SLOT_KEYS}
    examples: list[dict[str, Any]] = []
    seen_examples: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        rule = raw.get("rule_ir")
        if not isinstance(rule, Mapping):
            continue
        projected: dict[str, Any] = {}
        valid = True
        for key in _SLOT_KEYS:
            if key not in rule:
                valid = False
                break
            value = rule[key]
            # Rule-IR slots are bounded enums/bits.  Do not copy arbitrary
            # model output into a presentation artifact.
            if key == "safe_to_send":
                if not isinstance(value, bool):
                    valid = False
                    break
                rendered: Any = value
            else:
                if not isinstance(value, str) or not value or len(value) > 64:
                    valid = False
                    break
                rendered = value
            projected[key] = rendered
            slot_value_counts[key][str(rendered)] += 1
        if not valid:
            continue
        slot_counts[json.dumps(projected, sort_keys=True, separators=(",", ":"))] += 1
        key = json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in seen_examples and len(examples) < max(0, int(limit)):
            examples.append(projected)
            seen_examples.add(key)
    return {
        "unique_rule_ir_count": len(slot_counts),
        "slot_occurrences": int(sum(slot_counts.values())),
        "slot_value_counts": {key: dict(sorted(counter.items())) for key, counter in slot_value_counts.items()},
        "examples": examples,
    }


def _model_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("rows") if isinstance(report.get("rows"), Sequence) else []
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    bound_rows = [row for row in rows if isinstance(row, Mapping) and row.get("reason") == "bound_and_replayed"]
    evidence_rows = [row for row in bound_rows if isinstance(row.get("evidence_sha256"), str)]
    fresh_rows = [row for row in bound_rows if row.get("replay_consistent") is True]
    negative_rows = [row for row in bound_rows if row.get("negative_control_clean") is True]
    summary_counts = {
        "holdout_rows": int(counts.get("holdout_rows_considered", len(rows))),
        "decoded_exact": int(counts.get("decoded_exact", 0)),
        "bindable": int(counts.get("bound_rows", len(bound_rows))),
        "confirmed_positive": int(counts.get("confirmed_positive", 0)),
        "abstain": int(counts.get("abstain_rows", 0)),
        "safe_abstain": int(counts.get("safe_abstain_correct", 0)),
        "unsafe_allow": int(counts.get("unsafe_allow", 0)),
        "evidence": len(evidence_rows),
        "fresh_replay": len(fresh_rows),
        "negative_clean": len(negative_rows),
    }
    return {
        "status": str(report.get("status", "incomplete")),
        "counts": summary_counts,
        "rates": {
            "decode_exact": (summary_counts["decoded_exact"] / summary_counts["holdout_rows"]) if summary_counts["holdout_rows"] else 0.0,
            "bind_rate": (summary_counts["bindable"] / summary_counts["holdout_rows"]) if summary_counts["holdout_rows"] else 0.0,
            "confirmed_positive_rate": (summary_counts["confirmed_positive"] / summary_counts["bindable"]) if summary_counts["bindable"] else 0.0,
            "safe_abstain_rate": (summary_counts["safe_abstain"] / summary_counts["abstain"]) if summary_counts["abstain"] else 0.0,
        },
        "rule_ir": _project_slots(rows),
        "provenance": {
            key: report[key]
            for key in ("model_checkpoint_sha256", "dataset_sha256", "registry_sha256", "report_sha256")
            if isinstance(report.get(key), str)
        },
        "promotion": _promotion_false(report),
        "scope": {
            "model_selected_abstract_binding": True,
            "evaluator_only": True,
            "raw_wire_exposed": False,
            "general_vulnerability_claim": False,
        },
    }


def _waf_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    contract = report.get("contract") if isinstance(report.get("contract"), Mapping) else {}
    fresh = report.get("fresh_reset_contract") if isinstance(report.get("fresh_reset_contract"), Mapping) else {}
    scope = report.get("scientific_scope") if isinstance(report.get("scientific_scope"), Mapping) else {}
    return {
        "status": str(report.get("status", "incomplete")),
        "counts": {
            key: int(value)
            for key, value in counts.items()
            if key in {
                "episodes", "get_episodes", "post_episodes", "repair_action_changed", "failure_action_change",
                "candidate_typed", "reference_typed", "negative_clean", "negative_violation", "replay_consistent",
                "fresh_reset_rows", "evidence_sha256_rows", "confirmed_positive",
            } and isinstance(value, (int, float))
        },
        "contract": {str(key): bool(value) for key, value in contract.items() if isinstance(value, bool)},
        "fresh_reset": {str(key): bool(value) for key, value in fresh.items() if isinstance(value, bool)},
        "scope": {
            "synthetic_evaluator_only": bool(scope.get("synthetic_evaluator_only", True)),
            "independent_implementation": bool(scope.get("independent_implementation", False)),
            "general_vulnerability_claim": False,
            "raw_wire_exposed": False,
        },
        "provenance": {
            "policy_hashes": report.get("policy_hashes") if isinstance(report.get("policy_hashes"), Mapping) else {},
        },
        "promotion": _promotion_false(report),
    }


def _training_summary(report: Mapping[str, Any], *, candidate_name: str) -> dict[str, Any]:
    gate = report.get("scientific_gate") if isinstance(report.get("scientific_gate"), Mapping) else {}
    worst = report.get("worst_seed") if isinstance(report.get("worst_seed"), Mapping) else {}
    training = report.get("training") if isinstance(report.get("training"), Mapping) else {}
    candidates = report.get("candidates") if isinstance(report.get("candidates"), Sequence) else []
    sequence = [row.get("holdout", {}).get("sequence_exact") for row in candidates if isinstance(row, Mapping) and isinstance(row.get("holdout"), Mapping)]
    locks = report.get("locks") if isinstance(report.get("locks"), Mapping) else {}
    lock_hashes = {
        str(key): str(value)
        for key, value in locks.items()
        if isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)
    }
    return {
        "name": candidate_name,
        "status": str(report.get("status", "incomplete")),
        "seeds": list(training.get("seeds", [])) if isinstance(training.get("seeds"), Sequence) else [],
        "holdout_sequence_exact": [float(value) for value in sequence if isinstance(value, (int, float))],
        "worst_seed": {key: worst[key] for key in ("sequence_exact_min", "typed_oracle", "fresh_replay") if key in worst},
        "scientific_gate": {str(key): bool(value) for key, value in gate.items() if isinstance(value, bool)},
        "limits": [
            "single_synthetic_implementation",
            "typed_oracle_not_run_for_model_candidate",
            "fresh_replay_not_run_for_model_candidate",
            "candidate_only_no_promotion",
        ],
        "promotion": {key: False for key in _PROMOTION_KEYS},
        "provenance": {
            "report_sha256": report.get("report_sha256") if isinstance(report.get("report_sha256"), str) else None,
            "lock_hashes": lock_hashes,
        },
    }


def project_pg368_capability(
    model_report: Mapping[str, Any],
    waf_report: Mapping[str, Any],
    *,
    process_report: Mapping[str, Any] | None = None,
    sft_report: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a presentation-safe, read-only capability summary."""

    if not isinstance(model_report, Mapping) or not isinstance(waf_report, Mapping):
        raise TypeError("PG-368 reports must be mappings")
    forbidden = _text_scan({"model": model_report, "waf": waf_report})
    if forbidden:
        raise ValueError("PG-368 projection rejects raw wire/canary text")
    model = _model_summary(model_report)
    waf = _waf_summary(waf_report)
    training: list[dict[str, Any]] = []
    if process_report is not None:
        training.append(_training_summary(process_report, candidate_name="next_token_process_candidate"))
    if sft_report is not None:
        training.append(_training_summary(sft_report, candidate_name="weighted_rule_ir_sft_candidate"))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_demo_evidence",
        "headline": "模型已能选择抽象 Rule-IR 并在受控本地回放中完成正/负/复放对照；尚未证明通用网址漏洞能力。",
        "model_replay": model,
        "waf_replay": waf,
        "training_candidates": training,
        "scope": {
            "authorized_loopback_only": True,
            "synthetic_implementation_only": True,
            "independent_second_implementation": False,
            "wire_literals_included": False,
            "general_vulnerability_claim": False,
        },
        "promotion": {key: False for key in _PROMOTION_KEYS},
        "source_hashes": dict(source_hashes or {}),
        "next_step": "加入第二个独立实现，保持 GET/POST、正负 oracle、fresh reset 后再讨论晋级。",
    }
    # Re-scan the generated object as a defensive output firewall.  This only
    # records marker names, never the offending value.
    leaked = _text_scan(result)
    if leaked:
        raise ValueError("PG-368 projection generated forbidden raw text")
    result["projection_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def load_pg368_capability(
    model_report_path: str | Path,
    waf_report_path: str | Path,
    *,
    process_report_path: str | Path | None = None,
    sft_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load bounded JSON reports and return :func:`project_pg368_capability`."""

    paths = [Path(model_report_path), Path(waf_report_path)]
    model = _load_json(paths[0])
    waf = _load_json(paths[1])
    process = _load_json(Path(process_report_path)) if process_report_path is not None else None
    sft = _load_json(Path(sft_report_path)) if sft_report_path is not None else None
    source_hashes = {
        "model_report": _sha256_path(paths[0]),
        "waf_report": _sha256_path(paths[1]),
    }
    if process_report_path is not None:
        source_hashes["process_report"] = _sha256_path(Path(process_report_path))
    if sft_report_path is not None:
        source_hashes["sft_report"] = _sha256_path(Path(sft_report_path))
    return project_pg368_capability(model, waf, process_report=process, sft_report=sft, source_hashes=source_hashes)


# Descriptive aliases keep the projection convenient for a dashboard without
# creating a second implementation or a second schema.
project_capability_summary = project_pg368_capability
load_capability_projection = load_pg368_capability


__all__ = [
    "SCHEMA_VERSION",
    "project_pg368_capability",
    "load_pg368_capability",
    "project_capability_summary",
    "load_capability_projection",
]
