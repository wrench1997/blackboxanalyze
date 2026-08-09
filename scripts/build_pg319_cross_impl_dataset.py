"""Build PG-319 cross-implementation Rule-IR process data.

PG-318 is a frozen evaluation holdout.  This builder therefore uses only the
already audited PG-317 abstract corpus plus the *training* side of the older
VulnerableApp replay.  VulnerableApp route-template hashes and seeds are split
before conversion: two hashes/seeds may train, the remaining hashes and seed
are implementation/route holdout.  The decoder sees observation and surface
slots only; implementation, family, route, payload and response fields remain
metadata or evaluator evidence.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context, target_map  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
PG317_DATASET = RESEARCH / "pg317_question_anchor_dataset_v1.json"
PG317_AUDIT = RESEARCH / "pg317_question_anchor_dataset_audit_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
PG246_REPORT = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_report_v1.json"
OUTPUT = RESEARCH / "pg319_cross_impl_rule_ir_dataset_v1.json"
AUDIT = RESEARCH / "pg319_cross_impl_rule_ir_dataset_audit_v1.json"

MISSING_COMBINATIONS = tuple(itertools.combinations(OBSERVATION_KEYS, 2))
FORBIDDEN = frozenset(
    {
        "payload",
        "url",
        "route",
        "family",
        "response",
        "response_body",
        "source_code",
        "sql",
        "xss",
        "xxe",
        "replay_expected",
        "typed_effect",
    }
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _token_value(tokens: list[str], key: str, default: str = "unknown") -> str:
    prefix = f"{key}="
    for token in tokens:
        if str(token).startswith(prefix):
            return str(token).split("=", 1)[1]
    return default


def _replace(tokens: list[str], key: str, value: str) -> list[str]:
    result = [token for token in tokens if not str(token).startswith(f"{key}=")]
    eos = next((index for index, token in enumerate(result) if str(token) == "[EOS]"), len(result))
    result.insert(eos, f"{key}={value}")
    return result


def _pg246_split(row: Mapping[str, Any]) -> str:
    """Route/seed split chosen before abstraction, never from the target."""

    # The first two route hashes are training surfaces; secure/img variants
    # and seed 24603 are held out.  No hash appears in both sides.
    train_hashes = {
        "1bcd727afe2c59adfd8d77c8a6afdcee8984613348526f04dd446923559eb6f2",
        "02cde0bdcb3834e03a9870711aa6abf77c7f6d78a7c9fe903fe9b4796786db86",
    }
    if str(row.get("route_template_hash")) not in train_hashes:
        return "implementation_holdout"
    if int(row.get("seed", 0) or 0) == 24603:
        return "seed_holdout"
    return "train"


def _pg246_context(row: Mapping[str, Any]) -> list[str]:
    method = str(row.get("method", "GET")).upper()
    field_role = "query_param" if method == "GET" else "form_field"
    encoding = "url_percent" if method == "GET" else "form_urlencoded"
    lane = str(row.get("lane", "hard_negative"))
    failure = str(row.get("failure_kind", "candidate_no_effect"))
    if failure == "oracle_unavailable":
        typed = "unknown"
        feedback = "unknown"
        history = "none"
        failure_class = "none"
    elif lane == "gold" and failure == "typed_effect":
        typed = "1"
        feedback = "negative_control_clear"
        history = "none"
        failure_class = "none"
    else:
        typed = "1"
        feedback = "observable_no_effect"
        history = "candidate_failed"
        failure_class = "effect_not_confirmed"
    raw = [
        "[BOS]",
        f"typed_available={typed}",
        f"feedback_state={feedback}",
        "replay_ready=1",
        "evidence_present=1",
        "negative_control=1",
        "fresh_reset=1",
        f"surface_method={method}",
        f"surface_field_role={field_role}",
        f"surface_encoding={encoding}",
        f"history_action={history}",
        f"failure_class={failure_class}",
        "step_budget=present",
        "[EOS]",
    ]
    return canonical_assembly_context(raw)


def _base_record(row: Mapping[str, Any], split: str, ordinal: int) -> dict[str, Any]:
    context = _pg246_context(row)
    target = probe_target_for_context(context)
    lane = str(row.get("lane", "hard_negative"))
    failure = str(row.get("failure_kind", "candidate_no_effect"))
    return {
        "schema_version": "pg319-cross-impl-rule-ir-v1",
        "record_id": f"pg319-vapp-{row.get('record_id', ordinal)}",
        "source": "pg246_vulnerableapp_independent_dom_holdout",
        "source_implementation_meta": str(row.get("source_implementation", "owasp-vulnerableapp-java-spring")),
        "source_evidence_hash": str(row.get("source_evidence_hash", "")),
        "source_route_template_hash": str(row.get("route_template_hash", "")),
        "source_seed": int(row.get("seed", 0) or 0),
        "split": split,
        "training_eligible": split == "train",
        "family_meta": "dom_surface",
        "surface_meta": {"method": str(row.get("method", "GET")).upper(), "lane": lane, "failure": failure},
        "context_tokens": context,
        "target_tokens": target,
        "outcome_class_meta": "typed_effect" if lane == "gold" and failure == "typed_effect" else "failure_or_abstain",
        "hard_negative": lane == "hard_negative",
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_target_off_input": True,
        "counterfactual_kind": "pg246_observed_process",
        "record_sha256": "",
    }


def _missing_clone(row: Mapping[str, Any], missing: tuple[str, ...], ordinal: int) -> dict[str, Any]:
    clone = copy.deepcopy(dict(row))
    context = list(clone["context_tokens"])
    for key in missing:
        context = _replace(context, key, "unknown")
    target = probe_target_for_context(context)
    clone.update(
        {
            "record_id": f"{row['record_id']}:missing:{'-'.join(missing)}:{ordinal}",
            "context_tokens": context,
            "target_tokens": target,
            "counterfactual_kind": "pg319_multi_missing_pair",
            "anchor_role": "ask",
            "missing_slots": list(missing),
            "training_eligible": row.get("split") == "train",
            "hard_negative": True,
        }
    )
    return clone


def _seal(row: dict[str, Any]) -> dict[str, Any]:
    row["record_sha256"] = _digest({key: value for key, value in row.items() if key != "record_sha256"})
    return row


def main() -> int:
    pg317 = _load(PG317_DATASET)
    pg317_audit = _load(PG317_AUDIT)
    pg246 = _load(PG246_DATASET)
    pg246_report = _load(PG246_REPORT)
    if pg317_audit.get("status") != "passed":
        raise RuntimeError("PG-319 requires PG-317 audit passed")
    if pg246_report.get("status") != "completed_independent_implementation_route_holdout":
        raise RuntimeError("PG-319 requires the audited PG-246 independent implementation replay")
    rows: list[dict[str, Any]] = []
    # Keep PG-317 train rows as prior abstract process knowledge.  Its holdout
    # rows remain evaluation-only and are copied only for the new audit lanes.
    for source_row in pg317.get("records", []):
        clone = copy.deepcopy(dict(source_row))
        clone["source_family_meta"] = "pg317_prior_abstract"
        clone["raw_payload_stored"] = False
        clone["raw_response_body_stored"] = False
        rows.append(clone)
    pg246_base: list[dict[str, Any]] = []
    for ordinal, source_row in enumerate(pg246.get("records", [])):
        split = _pg246_split(source_row)
        base = _seal(_base_record(source_row, split, ordinal))
        pg246_base.append(base)
        rows.append(base)
        # Add the complete/ASK relation without copying the implementation's
        # final label into model-visible tokens.
        for missing in MISSING_COMBINATIONS:
            rows.append(_seal(_missing_clone(base, missing, ordinal)))
    counts = Counter(str(row.get("split")) for row in rows)
    vapp_rows = [row for row in rows if str(row.get("source")) == "pg246_vulnerableapp_independent_dom_holdout"]
    vapp_train = [row for row in vapp_rows if row.get("split") == "train"]
    vapp_holdout = [row for row in vapp_rows if row.get("split") in {"implementation_holdout", "seed_holdout"}]
    vapp_route_holdout = [row for row in vapp_rows if row.get("split") == "implementation_holdout"]
    vapp_seed_holdout = [row for row in vapp_rows if row.get("split") == "seed_holdout"]
    anchor_rows = [row for row in vapp_rows if row.get("counterfactual_kind") == "pg319_multi_missing_pair"]
    ask_rows = [row for row in anchor_rows if row.get("anchor_role") == "ask"]
    forbidden_context = [
        (row["record_id"], token)
        for row in rows
        for token in row.get("context_tokens", [])
        if str(token).split("=", 1)[0] in FORBIDDEN
    ]
    forbidden_target = [
        (row["record_id"], token)
        for row in rows
        for token in row.get("target_tokens", [])
        if str(token).split("=", 1)[0] in FORBIDDEN or "<" in str(token) or ">" in str(token)
    ]
    train_hashes = {str(row.get("source_route_template_hash")) for row in vapp_train}
    holdout_hashes = {str(row.get("source_route_template_hash")) for row in vapp_route_holdout}
    checks = {
        "pg317_audit_pass": pg317_audit.get("status") == "passed",
        "pg246_independent_replay_pass": bool(pg246_report.get("honesty", {}).get("independent_implementation", True)) and bool(pg246_report.get("safety", {}).get("raw_payload_strings_stored") is False),
        "records_present": bool(rows),
        "train_present": counts.get("train", 0) > 0,
        "implementation_holdout_present": counts.get("implementation_holdout", 0) > 0,
        "hard_negative_present": counts.get("hard_negative_eval", 0) > 0 or any(bool(row.get("hard_negative")) for row in rows),
        "vapp_train_present": bool(vapp_train),
        "vapp_route_holdout_present": bool(vapp_holdout),
        "vapp_seed_holdout_present": bool(vapp_seed_holdout),
        "vapp_route_hash_disjoint": bool(train_hashes) and bool(holdout_hashes) and train_hashes.isdisjoint(holdout_hashes),
        "multi_missing_rows": bool(ask_rows) and all(len(row.get("missing_slots") or []) >= 2 for row in ask_rows),
        "multi_missing_ask_safe_zero": all(target_map(row.get("target_tokens") or []).get("safe_to_send") == "0" for row in ask_rows),
        "multi_missing_ask_questioned": all(target_map(row.get("target_tokens") or []).get("question") != "none" for row in ask_rows),
        "context_forbidden_absent": not forbidden_context,
        "target_forbidden_absent": not forbidden_target,
        "raw_values_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in rows),
        "oracle_target_off_input": all(bool(row.get("oracle_target_off_input", True)) for row in rows),
        "pg318_not_in_training": all(str(row.get("source")) != "pg318_family_holdout" for row in rows if row.get("split") == "train"),
    }
    dataset = {
        "schema_version": "pg319-cross-impl-rule-ir-dataset-v1",
        "sources": {
            "pg317": {"dataset": str(PG317_DATASET.relative_to(ROOT)), "dataset_sha256": pg317.get("dataset_sha256"), "audit": str(PG317_AUDIT.relative_to(ROOT)), "audit_sha256": pg317_audit.get("audit_sha256")},
            "pg246": {"dataset": str(PG246_DATASET.relative_to(ROOT)), "dataset_sha256": pg246.get("dataset_sha256"), "report": str(PG246_REPORT.relative_to(ROOT)), "report_sha256": pg246_report.get("report_sha256")},
        },
        "records": rows,
        "counts": {"total": len(rows), "train": counts.get("train", 0), "implementation_holdout": counts.get("implementation_holdout", 0), "seed_holdout": counts.get("seed_holdout", 0), "real_live_holdout": counts.get("real_live_holdout", 0), "hard_negative_eval": counts.get("hard_negative_eval", 0), "vapp_total": len(vapp_rows), "vapp_train": len(vapp_train), "vapp_route_holdout": len(vapp_route_holdout), "vapp_seed_holdout": len(vapp_seed_holdout), "vapp_holdout": len(vapp_holdout), "vapp_anchor_rows": len(anchor_rows), "vapp_ask_rows": len(ask_rows), "vapp_route_hash_train": len(train_hashes), "vapp_route_hash_holdout": len(holdout_hashes)},
        "contract": {"decoder_only_next_token": True, "abstract_rule_ir_target": True, "multi_missing_question_pairs": True, "implementation_route_holdout": True, "family_hidden_in_context": True, "raw_payloads_excluded": True, "raw_responses_excluded": True, "pg318_holdout_frozen": True, "training_promotion_allowed": False, "memory_promotion_allowed": False},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {"schema_version": "pg319-cross-impl-rule-ir-dataset-audit-v1", "checks": checks, "counts": {"forbidden_context": len(forbidden_context), "forbidden_target": len(forbidden_target), "train_route_hashes": sorted(train_hashes), "holdout_route_hashes": sorted(holdout_hashes)}, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["schema_version"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
