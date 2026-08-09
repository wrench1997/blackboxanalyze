"""Build an information-preserving context-index view for PG-351.

The index is a deterministic, append-only projection of observations already
present in ``context_tokens``.  It is deliberately not a target or evaluator
answer: no action, oracle, payload shape, or safe-to-send value is derived.
The full original context remains byte-for-byte present before the index.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg359_context_index_dataset_v1.json"
INDEX_BEGIN = "[CONTEXT_INDEX_BOS]"
INDEX_END = "[CONTEXT_INDEX_EOS]"
FORBIDDEN = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "raw_response=",
    "wire=",
    "oracle=",
    "route_literal=",
    "family=",
    "evaluator=",
)

# These are observations, not labels.  Keep this list explicit so a future
# edit cannot accidentally turn the index into a target-side shortcut.
INDEX_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("typed", "belief_typed_available", ("present", "absent", "unknown", "not_observed")),
    ("evidence", "belief_evidence_present", ("present", "absent", "unknown", "not_observed")),
    ("fresh", "belief_fresh_reset", ("present", "absent", "unknown", "not_observed")),
    ("negative", "belief_negative_control", ("present", "absent", "unknown", "not_observed")),
    ("reference", "belief_reference_present", ("present", "absent", "unknown", "not_observed")),
    ("candidate", "belief_candidate_present", ("present", "absent", "unknown", "not_observed")),
    ("replay", "belief_replay_ready", ("present", "absent", "unknown", "not_observed")),
    ("probe_role", "belief_probe_role", ("candidate", "reference", "negative", "replay", "unknown")),
    ("process", "belief_process_step", ("baseline", "dynamic_observe", "failure", "repair", "replay", "unknown")),
    ("failure", "failure_failure_class", ("none", "unknown", "blocked_variant", "blocked", "parse", "environment")),
    ("request_axis", "request_transport_presence", ("observed", "not_observed", "unknown", "absent")),
    ("response_axis", "response_transport_presence", ("observed", "not_observed", "unknown", "absent")),
    ("javascript_axis", "javascript_presence", ("observed", "not_observed", "unknown", "absent")),
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(tokens: Sequence[str], prefix: str, allowed: Sequence[str]) -> str:
    marker = prefix + "="
    value = next((str(token)[len(marker) :] for token in tokens if str(token).startswith(marker)), "unknown")
    return value if value in set(allowed) else "unknown"


def context_index(context_tokens: Sequence[str]) -> list[str]:
    tokens = [str(token) for token in context_tokens]
    result = [INDEX_BEGIN]
    for name, source_prefix, allowed in INDEX_SPECS:
        result.append(f"index_{name}={_value(tokens, source_prefix, allowed)}")
    result.append(INDEX_END)
    return result


def _raw_free(tokens: Sequence[str]) -> bool:
    return not any(any(fragment in str(token).casefold() for fragment in FORBIDDEN) for token in tokens)


def build(dataset: Mapping[str, Any], *, input_sha256: str, input_path: str) -> dict[str, Any]:
    source_records = list(dataset.get("records") or [])
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    added_tokens: set[str] = {INDEX_BEGIN, INDEX_END}
    for position, source in enumerate(source_records):
        if not isinstance(source, Mapping):
            failures.append(f"row_{position}:not_mapping")
            continue
        row = copy.deepcopy(dict(source))
        context = row.get("context_tokens")
        target = row.get("target_tokens")
        if not isinstance(context, list) or not context or not isinstance(target, list):
            failures.append(f"row_{position}:stream")
            continue
        if not _raw_free(context) or not _raw_free(target):
            failures.append(f"row_{position}:raw_token")
            continue
        index = context_index(context)
        row["context_tokens"] = [*map(str, context), *index]
        row["context_index"] = {
            "schema_version": "pg359-context-index-v1",
            "source_context_sha256": _sha([str(token) for token in context]),
            "index_sha256": _sha(index),
            "derived_only_from_context": True,
            "target_tokens_read": False,
            "evaluator_sidecar_read": False,
        }
        added_tokens.update(index)
        row["record_sha256"] = ""
        body = dict(row)
        body.pop("record_sha256", None)
        row["record_sha256"] = _sha(body)
        records.append(row)

    base_vocab = dict(dataset.get("vocabulary") or {})
    context_vocab = list(dict.fromkeys(str(token) for token in base_vocab.get("context_tokens") or []))
    target_vocab = list(dict.fromkeys(str(token) for token in base_vocab.get("target_tokens") or []))
    for token in sorted(added_tokens):
        if token not in context_vocab:
            context_vocab.append(token)
    shared = sorted(set(context_vocab) | set(target_vocab) | {str(token) for token in base_vocab.get("shared_tokens") or []})
    result = {
        "schema_version": "pg359-context-index-dataset-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "source_dataset": input_path,
        "source_dataset_sha256": input_sha256,
        "records": records,
        "vocabulary": {
            **base_vocab,
            "context_tokens": sorted(context_vocab),
            "target_tokens": sorted(target_vocab),
            "shared_tokens": shared,
            "context_index_tokens": sorted(added_tokens),
            "append_only": True,
        },
        "context_index_contract": {
            "schema_version": "pg359-context-index-v1",
            "specs": [{"name": name, "source_prefix": prefix, "allowed": list(allowed)} for name, prefix, allowed in INDEX_SPECS],
            "original_context_preserved": True,
            "derived_only_from_context": True,
            "target_tokens_read": False,
            "evaluator_sidecar_read": False,
            "target_information_added": False,
            "raw_payload_in_context": False,
        },
        "counts": {
            "input_records": len(source_records),
            "records": len(records),
            "invalid_records": len(failures),
            "train_rows": sum(str(row.get("split")) == "train" for row in records),
            "implementation_holdout_rows": sum(str(row.get("split")) == "implementation_holdout" for row in records),
            "context_tokens_added_per_row": len(INDEX_SPECS) + 2,
            "context_vocabulary_added": len(added_tokens),
            "raw_payload_in_context": 0,
        },
        "failures": sorted(failures),
        "provenance": {"builder": "scripts/build_pg359_context_index_dataset.py", "builder_sha256": _file_sha(Path(__file__))},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build(dataset, input_sha256=_file_sha(args.input), input_path=str(args.input.resolve().relative_to(ROOT.resolve())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
