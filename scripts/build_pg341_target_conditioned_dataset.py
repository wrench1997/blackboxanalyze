"""Build the PG-341 target-conditioned diagnostic corpus.

PG-341 deliberately keeps two representation views separate:

* ``coarse_process`` comes from PG-337 and has real abstract ASK/repair/
  negative targets, but its context is not a full web-page observation;
* ``full_axis`` comes from PG-338 and preserves the seven-axis page token
  stream, but its training split does not contain ASK/repair/negative target
  supervision.

The builder never invents targets, never joins rows by visual similarity and
never copies evaluator sidecars, routes, payloads or response bodies.  The
result is a diagnostic corpus.  Every row remains ineligible for promotion;
the coarse branch may be used only by an explicitly named diagnostic
target-decoder smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COARSE = ROOT / "research" / "pg337_cross_impl_process_token_v1.json"
DEFAULT_FULL_AXIS = ROOT / "research" / "pg338_information_preserving_process_token_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg341_target_conditioned_process_full_axis_dataset_v1.json"

SCHEMA_VERSION = "pg341-target-conditioned-process-full-axis-dataset-v1"
FORBIDDEN_FRAGMENTS = (
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
ALLOWED_TARGET_PREFIXES = (
    "[TARGET_BOS]",
    "[TARGET_EOS]",
    "question=",
    "next_action=",
    "repair_action=",
    "action_changed=",
    "failure_class=",
    "safe_to_send=",
    "transport_ref=",
    "field_role_ref=",
    "encoding_ref=",
    "probe_variant_ref=",
)
AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_replay",
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
        raise ValueError(f"dataset must be an object: {path}")
    return value


def _token_forbidden(token: Any) -> bool:
    value = str(token).casefold()
    return any(fragment in value for fragment in FORBIDDEN_FRAGMENTS)


def _safe_target(tokens: Any) -> list[str]:
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("target_tokens missing")
    result = [str(token) for token in tokens]
    if result[0] != "[TARGET_BOS]" or result[-1] != "[TARGET_EOS]":
        raise ValueError("target boundary missing")
    if any(_token_forbidden(token) for token in result):
        raise ValueError("forbidden target token")
    if any(not token.startswith(ALLOWED_TARGET_PREFIXES) for token in result):
        raise ValueError("non-abstract target token")
    return result


def _safe_context(tokens: Any) -> list[str]:
    if not isinstance(tokens, list) or len(tokens) < 2:
        raise ValueError("context_tokens missing")
    result = [str(token) for token in tokens]
    if any(_token_forbidden(token) for token in result):
        raise ValueError("forbidden context token")
    return result


def _target_map(tokens: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        if "=" in token and token.split("=", 1)[0] in {
            "question",
            "next_action",
            "repair_action",
            "action_changed",
            "safe_to_send",
            "transport_ref",
            "field_role_ref",
            "encoding_ref",
            "probe_variant_ref",
        }:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def _axis_presence(raw: Mapping[str, Any], view: str) -> dict[str, str]:
    if view == "full_axis":
        value = raw.get("axis_presence")
        if isinstance(value, Mapping):
            return {axis: str(value.get(f"{axis}_presence", value.get(axis, "unknown"))) for axis in AXES}
    manifest = raw.get("field_capture_manifest")
    result: dict[str, str] = {}
    if isinstance(manifest, Mapping):
        for axis in AXES:
            item = manifest.get(axis)
            if isinstance(item, Mapping):
                result[axis] = str(item.get("presence", "unknown"))
            else:
                result[axis] = "unknown"
    else:
        result = {axis: "unknown" for axis in AXES}
    return result


def _method_class(context: list[str]) -> str:
    for token in context:
        if token in {"surface_method=GET", "surface_method=POST", "request_method=get", "request_method=post"}:
            return token.split("=", 1)[1].upper()
    return "unknown"


def _abstract_row(raw: Mapping[str, Any], *, view: str, source_dataset_sha256: str, source_file_sha256: str) -> dict[str, Any]:
    context = _safe_context(raw.get("context_tokens"))
    target = _safe_target(raw.get("target_tokens"))
    source_record_sha256 = str(raw.get("source_record_sha256", ""))
    if len(source_record_sha256) != 64:
        raise ValueError("source_record_sha256 missing")
    source_implementation_hash = str(raw.get("source_implementation_hash", ""))
    if source_implementation_hash and len(source_implementation_hash) != 64:
        raise ValueError("source implementation hash malformed")
    split = str(raw.get("split", ""))
    if split not in {"train", "implementation_holdout", "seed_holdout"}:
        raise ValueError("unsupported split")
    target_values = _target_map(target)
    process_metadata = raw.get("process_metadata") if isinstance(raw.get("process_metadata"), Mapping) else {}
    # Only abstract booleans are retained; evaluator-side identities and
    # evidence remain outside the model-facing corpus.
    abstract_metadata = {
        "real_failure_trace": bool(process_metadata.get("real_failure_trace", False)),
        "real_negative_evaluator_trace": bool(process_metadata.get("real_negative_evaluator_trace", False)),
        "real_ask_preflight": bool(process_metadata.get("real_ask_preflight", False)),
        "evidence_hash_present_sidecar": bool(process_metadata.get("evidence_hash_present_sidecar", False)),
        "method_class": _method_class(context),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg341-{view}-{_sha({'view': view, 'source_record_sha256': source_record_sha256, 'target': target})[:20]}",
        "view": view,
        "split": split,
        "source_split": str(raw.get("source_split", split)),
        "diagnostic_kind": str(raw.get("diagnostic_kind", "unknown")),
        "source_grounded": raw.get("source_grounded") is True,
        "synthetic_counterfactual": raw.get("synthetic_counterfactual") is True,
        "source_record_sha256": source_record_sha256,
        "source_dataset_sha256": source_dataset_sha256,
        "source_file_sha256": source_file_sha256,
        "source_implementation_hash": source_implementation_hash,
        "context_view_schema": "pg337_coarse_process_tokens_v1" if view == "coarse_process" else "pg338_full_axis_tokens_v1",
        "context_tokens": context,
        "target_tokens": target,
        "target_projection": target_values,
        "axis_presence": _axis_presence(raw, view),
        "process_metadata": abstract_metadata,
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        # This flag is intentionally false.  PG-341 is a target-conditioned
        # diagnostic corpus, not a promotion-ready source-row dataset.
        "training_eligible": False,
        "target_conditioned_diagnostic_eligible": view == "coarse_process",
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def build(coarse: Mapping[str, Any], full_axis: Mapping[str, Any], *, coarse_path: Path = DEFAULT_COARSE, full_axis_path: Path = DEFAULT_FULL_AXIS) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for source, view, path in ((coarse, "coarse_process", coarse_path), (full_axis, "full_axis", full_axis_path)):
        dataset_sha256 = str(source.get("dataset_sha256", ""))
        if len(dataset_sha256) != 64:
            dataset_sha256 = _sha({"schema_version": source.get("schema_version"), "records": len(source.get("records") or [])})
        source_file_sha256 = _file_sha(path)
        for index, raw in enumerate(source.get("records") or []):
            try:
                if not isinstance(raw, Mapping):
                    raise ValueError("row not mapping")
                records.append(_abstract_row(raw, view=view, source_dataset_sha256=dataset_sha256, source_file_sha256=source_file_sha256))
            except ValueError as exc:
                key = f"{view}:{str(exc)}"
                rejected[key] = rejected.get(key, 0) + 1
    counts = {
        "total": len(records),
        "coarse_process": sum(row["view"] == "coarse_process" for row in records),
        "full_axis": sum(row["view"] == "full_axis" for row in records),
        "coarse_train": sum(row["view"] == "coarse_process" and row["split"] == "train" for row in records),
        "coarse_holdout": sum(row["view"] == "coarse_process" and row["split"] != "train" for row in records),
        "full_axis_train": sum(row["view"] == "full_axis" and row["split"] == "train" for row in records),
        "full_axis_holdout": sum(row["view"] == "full_axis" and row["split"] != "train" for row in records),
        "rejected": sum(rejected.values()),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_target_conditioned_two_view",
        "purpose": "target-conditioned Rule-IR/ASK/repair learning diagnostic without collapsing full-axis context",
        "sources": {
            "coarse_process": {"path": str(coarse_path.relative_to(ROOT).as_posix()), "dataset_sha256": coarse.get("dataset_sha256"), "context_view": "coarse_process", "full_axis": False},
            "full_axis": {"path": str(full_axis_path.relative_to(ROOT).as_posix()), "dataset_sha256": full_axis.get("dataset_sha256"), "context_view": "full_axis", "full_axis": True},
        },
        "records": records,
        "counts": counts,
        "rejection_reason_counts": rejected,
        "track_contract": {
            "coarse_process_target_decoder_smoke_allowed": True,
            "full_axis_target_decoder_smoke_allowed": False,
            "full_axis_requires_ask_repair_train_coverage": True,
            "views_must_not_be_merged_for_capability_claim": True,
            "target_tokens_abstract_only": True,
            "raw_payloads_excluded": True,
            "raw_responses_excluded": True,
            "evaluator_sidecars_off_context": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "information_gate": {
            "full_axis_context_preserved": counts["full_axis"] > 0,
            "coarse_target_diversity_present": counts["coarse_process"] > 0,
            "full_axis_train_target_coverage_pending": True,
            "status": "pending_audit",
        },
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "dataset_sha256": "",
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-341 two-view target-conditioned diagnostic corpus")
    parser.add_argument("--coarse", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--full-axis", type=Path, default=DEFAULT_FULL_AXIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(_load(args.coarse), _load(args.full_axis), coarse_path=args.coarse, full_axis_path=args.full_axis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
