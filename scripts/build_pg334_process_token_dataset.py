"""Build PG-334: abstract process-token diagnostics.

PG-278 is a controlled loopback fixture, not real vulnerability gold.  This
adapter deliberately removes family/implementation/slot literals and keeps
only process-relevant tokens: method, placement, encoding, reset/control
availability, unknown-observation state, generic response shape, and the
question/repair decision.  The resulting rows are useful for a fast
representation smoke, but are never training- or promotion-eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "pg278_multifamily_question_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg334_process_token_diagnostic_v1.json"

AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_replay",
)
FORBIDDEN_PREFIXES = (
    "family=", "implementation=", "route=", "route_literal=", "slot=",
    "unknown_slot=", "bound_slot=", "oracle=", "evaluator=", "payload=",
    "raw_", "response_body=", "response_body_text=", "expected_",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _safe_token(token: Any) -> str | None:
    """Map a fixture token to an abstract token, or drop it."""
    value = str(token)
    if value in {"[BOS]", "[CTX_END]"}:
        return value
    if any(value.casefold().startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return None
    if value.startswith("unknown_role="):
        return "unknown_role=observation"
    if value.startswith("unknown_slot_count="):
        return value
    if value.startswith("question_asked="):
        return "question_asked=observation"
    if value.startswith("observed_"):
        # Shape buckets are safe; literal marker names are not.
        return value if len(value) <= 64 else value.split("=", 1)[0] + "=bucketed"
    if value.startswith("phase="):
        return value.replace("pre_question", "pre_observation")
    if value == "scope=loopback_fixture":
        return "scope=controlled_fixture"
    return value


def abstract_context(tokens: list[Any], *, stage: str) -> list[str]:
    result: list[str] = []
    for token in tokens:
        safe = _safe_token(token)
        if safe is not None:
            result.append(safe)
    # Keep a generic stage marker even when the fixture has an empty projection.
    if "stage=" + stage not in result:
        result.insert(1 if result and result[0] == "[BOS]" else 0, "stage=" + stage)
    return list(dict.fromkeys(result))


def _manifest(stage: str) -> dict[str, dict[str, str]]:
    status = {
        "document_structure": "not_observed",
        "navigation": "not_observed",
        "request_transport": "observed",
        "response_transport": "observed" if stage == "post" else "not_observed",
        "javascript_surface": "not_observed",
        "failure_feedback": "observed",
        "belief_replay": "observed",
    }
    return {axis: {"presence": value} for axis, value in status.items()}


def _target(*, stage: str, negative: bool) -> list[str]:
    if stage == "pre":
        values = [
            "[TARGET_BOS]", "question=ask_missing_observation",
            "next_action=ask_observation", "repair_action=observe",
            "failure_class=missing_observation", "action_changed=1",
            "safe_to_send=0", "[TARGET_EOS]",
        ]
    else:
        values = [
            "[TARGET_BOS]", "question=review_evidence",
            "next_action=abstain" if negative else "next_action=request_replay",
            "repair_action=none", "failure_class=none",
            "action_changed=1", "safe_to_send=0", "[TARGET_EOS]",
        ]
    return values


def build_dataset(source: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = list(source.get("records") or []) if isinstance(source, Mapping) else []
    records: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            continue
        negative = not bool(dict(row.get("labels") or {}).get("expected_positive"))
        split = "train" if str(row.get("split")) == "implementation_train" else "implementation_holdout"
        base = {
            "source_row_digest": digest({"index": index, "record_id": row.get("record_id"), "source": row.get("source")}),
            "source_kind": "controlled_loopback_fixture",
            "real_gold": False,
            "negative_control": negative,
        }
        for stage, source_field in (("pre", "pre_question_context_tokens"), ("post", "post_observation_context_tokens")):
            context = abstract_context(list(row.get(source_field) or []), stage=stage)
            record_identity = digest({"source": base["source_row_digest"], "stage": stage})
            pair_digest = digest({"pair": row.get("pair_id"), "split": split, "stage": stage})
            target = _target(stage=stage, negative=negative)
            records.append({
                "schema_version": "pg334-process-token-row-v1",
                "record_id": "pg334:" + record_identity,
                "paired_id": "pg334:" + pair_digest,
                "split": split,
                "stage": stage,
                "context_tokens": context,
                "target_tokens": target,
                "target_projection": {
                    "question": "ask_missing_observation" if stage == "pre" else "review_evidence",
                    "next_action": "ask_observation" if stage == "pre" else ("abstain" if negative else "request_replay"),
                    "repair_action": "observe" if stage == "pre" else "none",
                    "safe_to_send": False,
                    "action_changed": True,
                },
                "process_metadata": {
                    "pre_state": "missing_observation" if stage == "pre" else "observed_projection",
                    "failure_class": "missing_observation" if stage == "pre" else "none",
                    "negative_control": negative,
                    "source_row_digest": base["source_row_digest"],
                },
                "field_capture_manifest": _manifest(stage),
                "context_firewall": {
                    "forbidden_token_count": 0,
                    "sidecars_off_context": True,
                    "raw_payload_stored": False,
                    "raw_response_body_stored": False,
                    "oracle_answer_in_context": False,
                },
                # Keep the runner's explicit top-level firewall contract as
                # well as the nested audit projection.
                "raw_payload_stored": False,
                "raw_response_body_stored": False,
                "oracle_answer_in_context": False,
                "source": base,
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "payload_catalog_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
            })
    context_tokens = sorted({token for row in records for token in row["context_tokens"]})
    target_tokens = sorted({token for row in records for token in row["target_tokens"]})
    payload = {
        "schema_version": "pg334-process-token-diagnostic-v1",
        "purpose": "fast abstract process-token representation smoke for ASK/failure-repair/negative diagnostics",
        "source": {"dataset": "research/pg278_multifamily_question_dataset_v1.json", "real_gold_rows": 0, "controlled_fixture_only": True},
        "records": records,
        "counts": {
            "total": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in records),
            "pre": sum(row["stage"] == "pre" for row in records),
            "post": sum(row["stage"] == "post" for row in records),
            "negative": sum(bool(row["process_metadata"]["negative_control"]) for row in records),
        },
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "token_policy": {"family_literals_removed": True, "implementation_literals_removed": True, "slot_literals_removed": True, "raw_oracle_removed": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    payload["dataset_sha256"] = digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-334 abstract process-token diagnostic dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build_dataset(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "diagnostic_only", "records": len(result["records"]), "dataset_sha256": result["dataset_sha256"], "output": str(args.output.relative_to(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
