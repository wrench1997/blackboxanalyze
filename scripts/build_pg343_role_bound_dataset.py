"""Build the PG-343 role/step-bound full-axis diagnostic dataset.

This is a bounded, offline transformation.  It consumes only already
collected authorized source rows and adds two abstract process tokens using
an explicit source-side role/step contract.  It never derives role from
target_tokens, never copies source metadata/evaluator sidecars into model
context, and keeps every promotion flag false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg343_role_step_binding import ALLOWED_ROLES, bind_context_tokens, binding_present  # noqa: E402

RESEARCH = ROOT / "research"
DEFAULT_SOURCES = (
    RESEARCH / "pg343_webgoat_role_step_binding_source_rows_v1.json",
    RESEARCH / "pg332_dvwa_pikachu_cross_impl_source_rows_v1.json",
    RESEARCH / "pg337_dvwa_failure_repair_source_rows_v1.json",
)
DEFAULT_OUTPUT = RESEARCH / "pg343_role_bound_full_axis_target_conditioned_dataset_v1.json"
SCHEMA_VERSION = "pg343-role-bound-full-axis-target-conditioned-dataset-v1"
AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)
FORBIDDEN = (
    "payload=",
    "payload_",
    "response_body=",
    "response_body_text=",
    "raw_",
    "oracle=",
    "evaluator=",
    "family=",
    "route=",
    "route_literal=",
    "implementation=",
    "image=",
    "url=",
    "path=",
    "source=",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"source is not an object: {path}")
    return value


def _role_from_record_id(record_id: Any) -> str:
    parts = str(record_id).replace(":", "-").split("-")
    roles = [part.casefold() for part in parts if part.casefold() in ALLOWED_ROLES]
    if len(roles) != 1:
        raise ValueError("role_attestation_missing_or_ambiguous")
    return roles[0]


def _context_value(tokens: list[str], prefix: str) -> str | None:
    values = [token[len(prefix) :] for token in tokens if token.startswith(prefix)]
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"conflicting_context_{prefix[:-1]}")
    return values[0]


def _step_from_source_contract(raw: Mapping[str, Any], context: list[str]) -> str:
    """Read step from source observation/collector contract, never target."""

    if binding_present(context):
        value = _context_value(context, "belief_process_step=")
        if value is not None:
            return value
    failure_next = _context_value(context, "failure_next_action=")
    collector = str(dict(raw.get("source_meta") or {}).get("collector_id", "")).casefold()
    if failure_next == "repair" or "failure-repair" in collector:
        return "repair"
    if failure_next == "abstain":
        return "failure"
    # These collectors are source/evaluator contracts for a fresh preflight
    # observation.  The step is taken from the collector attestation, never
    # inferred from target_tokens or the target projection.
    if (
        "typed-source-row" in collector
        or "typed-get" in collector
        or "typed-method-shape" in collector
        or "typed-stored-post" in collector
    ):
        return "preflight"
    raise ValueError("step_attestation_missing")


def _implementation_hash(raw: Mapping[str, Any]) -> str:
    source_meta = raw.get("source_meta") if isinstance(raw.get("source_meta"), Mapping) else {}
    digest = str(source_meta.get("image_digest", ""))
    if len(digest) != 64:
        # Existing diagnostic projections may already carry a one-way hash.
        digest = str(raw.get("source_implementation_hash", ""))
    if len(digest) != 64:
        raise ValueError("implementation_attestation_missing")
    return _sha({"image_digest": digest})


def _project(raw: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    context = [str(token) for token in raw.get("context_tokens") or []]
    target = [str(token) for token in raw.get("target_tokens") or []]
    if len(context) < 2 or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
        raise ValueError("context_or_target_missing")
    if any(any(marker in token.casefold() for marker in FORBIDDEN) for token in [*context, *target]):
        raise ValueError("context_firewall")
    if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        raise ValueError("firewall_metadata")
    if any(raw.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
        raise ValueError("raw_oracle_flag")
    raw_presence = raw.get("axis_presence")
    if not isinstance(raw_presence, Mapping):
        raise ValueError("full_axis_presence_missing")
    presence_keys = {
        "document_structure": "document_presence",
        "navigation": "navigation_presence",
        "request_transport": "request_transport_presence",
        "response_transport": "response_transport_presence",
        "javascript_surface": "javascript_presence",
        "failure_feedback": "failure_feedback_presence",
        "belief_and_replay": "belief_replay_presence",
    }
    presence = {
        axis: raw_presence.get(axis, raw_presence.get(key))
        for axis, key in presence_keys.items()
    }
    if any(value is None for value in presence.values()):
        raise ValueError("full_axis_presence_missing")
    role = _role_from_record_id(raw.get("record_id"))
    step = _step_from_source_contract(raw, context)
    bound_context = bind_context_tokens(context, role=role, step=step)
    source_record = str(raw.get("record_sha256", raw.get("source_record_sha256", "")))
    if len(source_record) != 64:
        raise ValueError("source_record_hash_missing")
    impl_hash = _implementation_hash(raw)
    source_split = str(raw.get("split", "unknown"))
    split = "train" if source_split == "train" else "implementation_holdout"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg343-{_sha({'source': source_record, 'role': role, 'step': step})[:24]}",
        "split": split,
        "source_split": source_split,
        "source_artifact": source_path.name,
        "source_record_sha256": source_record,
        "source_implementation_hash": impl_hash,
        "context_tokens": bound_context,
        "target_tokens": target,
        "target_projection": dict(raw.get("target_projection") or {}),
        "field_capture_manifest": raw.get("field_capture_manifest"),
        "axis_presence": {axis: str(presence.get(axis)) for axis in AXES},
        "role_step_binding": {"role": role, "step": step, "source_attested": True},
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def build(sources: tuple[Path, ...] = DEFAULT_SOURCES) -> dict[str, Any]:
    accepted: dict[str, dict[str, Any]] = {}
    rejection_reasons: Counter[str] = Counter()
    source_manifest: list[dict[str, Any]] = []
    for source_path in sources:
        data = _load(source_path)
        source_manifest.append({"name": source_path.name, "file_sha256": _file_sha(source_path), "dataset_sha256": str(data.get("dataset_sha256", ""))})
        for raw in data.get("records") or []:
            if not isinstance(raw, Mapping):
                rejection_reasons["row_not_mapping"] += 1
                continue
            try:
                row = _project(raw, source_path=source_path)
            except ValueError as exc:
                rejection_reasons[str(exc)] += 1
                continue
            key = _sha({"context": row["context_tokens"], "target": row["target_tokens"]})
            if key in accepted:
                rejection_reasons["duplicate_context_target"] += 1
                continue
            row["context_target_sha256"] = key
            row["record_sha256"] = _sha(row)
            accepted[key] = row
    records = sorted(accepted.values(), key=lambda item: str(item["record_sha256"]))
    counts = Counter((row["split"], row["role_step_binding"]["role"], row["role_step_binding"]["step"]) for row in records)
    implementation_groups = defaultdict(lambda: {"train": 0, "implementation_holdout": 0})
    for row in records:
        implementation_groups[row["source_implementation_hash"]][row["split"]] += 1
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only_pending_target_audit",
        "purpose": "full-axis target-conditioned diagnostic with explicit source-attested role/step context binding",
        "sources": source_manifest,
        "records": records,
        "counts": {
            "input_rows": sum(len((_load(path).get("records") or [])) for path in sources),
            "accepted_rows": len(records),
            "train_rows": sum(row["split"] == "train" for row in records),
            "implementation_holdout_rows": sum(row["split"] != "train" for row in records),
            "duplicate_context_target_rows": rejection_reasons["duplicate_context_target"],
            "rejected_rows": sum(value for key, value in rejection_reasons.items() if key != "duplicate_context_target"),
            "accepted_training_rows": 0,
        },
        "role_step_counts": {f"{split}:{role}:{step}": count for (split, role, step), count in sorted(counts.items())},
        "implementation_hash_groups": {key: dict(value) for key, value in sorted(implementation_groups.items())},
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "isolation": {
            "source_split_preserved": True,
            "implementation_split_disjoint": all(sum(1 for value in groups.values() if value > 0) <= 1 for groups in implementation_groups.values()),
            "holdout_excluded_from_training": True,
            "role_step_not_inferred_from_target": True,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "dataset_sha256": "",
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-343 role-bound full-axis diagnostic dataset")
    parser.add_argument("--source", action="append", type=Path, dest="sources")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = tuple(args.sources) if args.sources else DEFAULT_SOURCES
    result = build(sources)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
