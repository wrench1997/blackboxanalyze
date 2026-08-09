"""Build a larger full-slot diagnostic dataset without adding raw wire data.

PG-347 merges the two existing abstract full-axis sources (PG-338 process
rows and PG-345 role-bound rows), normalizes the older source's explicit
process/evaluator attestation into a sidecar, and reassigns split by frozen
source-implementation hash groups.  It never infers a payload or oracle
answer, and it keeps every source row diagnostic-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pg347-multi-implementation-full-slot-dataset-v1"
TARGET_PREFIXES = ("[TARGET_BOS]", "[TARGET_EOS]", "question=", "next_action=", "repair_action=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=", "safe_to_send=")
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
SPLIT_BY_IMPL = {
    # PG-338 source groups: one observed implementation and one held-out
    # failure/negative implementation.
    "40bdd27f85c795ec0eada01fbd30f0f5ba95a437a2a4a466bca54fb25fcd226e": "train",
    "853bdaae8abf99154475d53684a48597692fbbcfcb9ca988316ff456265382af": "implementation_holdout",
    # PG-345 source groups are retained as source-hash groups, not claimed to
    # be independent products without a separate implementation attestation.
    "a60f75b1234fdb2bf422725a4968e919d0abceca421bf19cb83caf29a5447232": "train",
    "8fd17d70f910c4e0f96f2ebd26fa2892ae5f7c751400b5ac141f8a16dbd444c6": "train",
    "9ae6a3c02cf053298249f15c8c6ba023f926b254f1d8fae9cfe5710e949b8c36": "implementation_holdout",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_abstract_tokens(tokens: Sequence[Any], *, target: bool = False) -> bool:
    if not isinstance(tokens, list) or not tokens:
        return False
    if target and (str(tokens[0]) != "[TARGET_BOS]" or str(tokens[-1]) != "[TARGET_EOS]"):
        return False
    return all(any(str(token).startswith(prefix) for prefix in TARGET_PREFIXES) for token in tokens) if target else all("=" in str(token) or str(token).startswith("[") for token in tokens)


def _normalize(row: Mapping[str, Any], *, source_artifact: str, source_file_sha256: str) -> dict[str, Any] | None:
    implementation = str(row.get("source_implementation_hash") or "")
    split = SPLIT_BY_IMPL.get(implementation)
    context = row.get("context_tokens")
    target = row.get("target_tokens")
    if split is None or not _valid_abstract_tokens(context or []) or not _valid_abstract_tokens(target or [], target=True):
        return None
    existing_binding = dict(row.get("role_step_binding") or {})
    if row.get("source_grounded") is not True and existing_binding.get("source_attested") is not True:
        return None
    if row.get("synthetic_counterfactual") is True:
        return None
    firewall = row.get("context_firewall")
    if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        return None
    if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False or row.get("oracle_answer_in_context") is not False:
        return None
    process = dict(row.get("process_metadata") or {})
    target_projection = dict(row.get("target_projection") or {})
    source_sidecar = dict(row.get("evaluator_sidecar_ref") or {})
    # For PG-338 the process/evaluator fields are explicit source attestations;
    # normalize them to the same abstract role/step contract without copying
    # source track names or evaluator literals into context.
    source_attested = bool(existing_binding.get("source_attested") is True or (process.get("full_axis_context") is True and process.get("process_kind") and len(str(source_sidecar.get("evidence_sha256", ""))) == 64 and source_sidecar.get("typed_available") is True and source_sidecar.get("fresh_reset") is True))
    if not source_attested:
        return None
    context_tokens = [str(token) for token in context]
    if not any(token.startswith("belief_probe_role=") for token in context_tokens):
        context_tokens.append("belief_probe_role=source_observed")
    if not any(token.startswith("belief_process_step=") for token in context_tokens):
        context_tokens.append("belief_process_step=" + str(process.get("process_kind") or existing_binding.get("step") or "preflight"))
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "record_id": "pg347-" + hashlib.sha256((source_file_sha256 + ":" + str(row.get("record_id"))).encode("utf-8")).hexdigest()[:28],
        "split": split,
        "source_split": str(row.get("split") or row.get("source_split") or "unknown"),
        "source_artifact": source_artifact,
        "source_file_sha256": source_file_sha256,
        "source_record_sha256": str(row.get("source_record_sha256") or row.get("record_sha256") or ""),
        "source_implementation_hash": implementation,
        "context_tokens": context_tokens,
        "target_tokens": [str(token) for token in target],
        "target_projection": {key: target_projection[key] for key in ("question", "next_action", "repair_action", "safe_to_send") if key in target_projection},
        "field_capture_manifest": row.get("field_capture_manifest") or {},
        "axis_presence": row.get("axis_presence") or {},
        "role_step_binding": {"source_attested": True, "normalization": "source_process_and_evaluator_attested", "source_process_kind": str(process.get("process_kind") or existing_binding.get("step") or "preflight")},
        "context_firewall": firewall,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "evaluator_sidecar_ref": {key: source_sidecar[key] for key in ("evidence_sha256", "typed_available", "fresh_reset", "negative_control", "confirmed_positive") if key in source_sidecar},
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    normalized["context_target_sha256"] = _sha({"context_tokens": normalized["context_tokens"], "target_tokens": normalized["target_tokens"]})
    normalized["record_sha256"] = _sha({key: value for key, value in normalized.items() if key != "record_sha256"})
    return normalized


def build_dataset(*, pg338: Mapping[str, Any], pg345: Mapping[str, Any], pg338_file_sha256: str, pg345_file_sha256: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rejected = Counter()
    for source, artifact, digest in ((pg338, "pg338_information_preserving_process_token_v1.json", pg338_file_sha256), (pg345, "pg345_decision_boundary_role_bound_dataset_v1.json", pg345_file_sha256)):
        for row in source.get("records") or []:
            normalized = _normalize(row, source_artifact=artifact, source_file_sha256=digest)
            if normalized is None:
                rejected[artifact] += 1
            else:
                rows.append(normalized)
    # Exact context+target duplicates are excluded, with the first source
    # retained; split is never silently changed.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicates = 0
    for row in rows:
        key = str(row["context_target_sha256"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(row)
    impl_counts = Counter(str(row["source_implementation_hash"]) for row in deduped)
    split_counts = Counter(str(row["split"]) for row in deduped)
    context_splits = {}
    for row in deduped:
        context_splits.setdefault(str(row["context_target_sha256"]), set()).add(str(row["split"]))
    source_splits = {}
    for row in deduped:
        source_splits.setdefault(str(row["source_record_sha256"]), set()).add(str(row["split"]))
    axis_stats = {}
    for axis in AXES:
        sequences = set()
        for row in deduped:
            tokens = tuple(token for token in row["context_tokens"] if token in (f"axis_begin={axis}", f"axis_end={axis}") or token.startswith(f"{axis}_field_"))
            sequences.add(tokens)
        axis_stats[axis] = {"rows": len(deduped), "unique_sequences": len(sequences), "status": "measured" if sequences else "missing"}
    audit = {
        "status": "diagnostic_passed_not_training_eligible" if deduped else "blocked",
        "rows": len(deduped),
        "train_rows": split_counts.get("train", 0),
        "implementation_holdout_rows": split_counts.get("implementation_holdout", 0),
        "implementation_groups": len(impl_counts),
        "implementation_split_leaks": 0,
        "context_split_leaks": sum(1 for values in context_splits.values() if len(values) > 1),
        "source_record_split_leaks": sum(1 for values in source_splits.values() if len(values) > 1),
        "axis_token_sequence_entropy": axis_stats,
        "failures": [],
        "context_target_duplicates_removed": duplicates,
        "source_record_duplicates_same_split": sum(1 for count in Counter(str(row["source_record_sha256"]) for row in deduped).values() if count > 1),
        "rejected_by_source": dict(sorted(rejected.items())),
        "independent_implementation_attestation": False,
        "training_eligible_rows": 0,
    }
    audit["audit_sha256"] = _sha(audit)
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only",
        "sources": {"pg338": pg338_file_sha256, "pg345": pg345_file_sha256},
        "records": deduped,
        "counts": audit,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "这是 source-hash 分层的 full-slot 诊断数据；未宣称独立产品实现，所有 rows 保持 training_eligible=false，不能直接晋级。",
    }
    output["dataset_sha256"] = _sha({key: value for key, value in output.items() if key != "dataset_sha256"})
    return output


def build_vocabulary(dataset: Mapping[str, Any]) -> dict[str, Any]:
    context = sorted({str(token) for row in dataset.get("records") or [] for token in row.get("context_tokens") or []})
    target = sorted({str(token) for row in dataset.get("records") or [] for token in row.get("target_tokens") or []})
    vocabulary = {"schema_version": "pg347-multi-implementation-full-slot-vocabulary-v1", "status": "diagnostic_only", "context_tokens": context, "target_tokens": target, "context_vocabulary_size": len(context), "target_vocabulary_size": len(target), "append_only": True, "forbidden_tokens": [], "source_dataset_sha256": dataset.get("dataset_sha256"), "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    vocabulary["vocabulary_sha256"] = _sha(vocabulary)
    return vocabulary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-347 full-slot diagnostic dataset")
    parser.add_argument("--pg338", type=Path, required=True)
    parser.add_argument("--pg345", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--vocabulary-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset = build_dataset(pg338=load(args.pg338), pg345=load(args.pg345), pg338_file_sha256=_sha_file(args.pg338), pg345_file_sha256=_sha_file(args.pg345))
    vocabulary = build_vocabulary(dataset)
    audit = dict(dataset["counts"])
    audit["dataset_file_sha256"] = None
    audit["vocabulary_file_sha256"] = None
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.vocabulary_output.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit["dataset_file_sha256"] = _sha_file(args.dataset_output)
    audit["vocabulary_file_sha256"] = _sha_file(args.vocabulary_output)
    audit["audit_sha256"] = _sha(audit)
    audit_doc = {"schema_version": "pg347-multi-implementation-full-slot-audit-v1", "status": audit["status"], "dataset": str(args.dataset_output), "vocabulary": str(args.vocabulary_output), "counts": audit, "failures": [], "information_gate": "diagnostic_not_promotion", "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}, "audit_sha256": audit["audit_sha256"]}
    args.audit_output.write_text(json.dumps(audit_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(args.dataset_output), "vocabulary": str(args.vocabulary_output), "audit": str(args.audit_output), "counts": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
