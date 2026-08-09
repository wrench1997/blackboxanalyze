"""Build the append-only PG-343 context/target vocabulary manifest.

Context inventory may include abstract token *types* observed in the held-out
implementation so the model does not fail merely because a structural enum is
new.  Holdout sequences and target labels are never used for optimization or
target-vocabulary derivation.  The manifest contains no row IDs, responses,
payloads, routes, or evaluator answers.
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

RESEARCH = ROOT / "research"
DEFAULT_DATASET = RESEARCH / "pg343_role_bound_full_axis_target_conditioned_dataset_v1.json"
DEFAULT_BASE = RESEARCH / "pg331_web_token_vocabulary_v1.json"
DEFAULT_OUTPUT = RESEARCH / "pg343_role_bound_vocabulary_v1.json"
SCHEMA_VERSION = "pg343-role-bound-vocabulary-v1"
FORBIDDEN = (
    "payload=",
    "payload_",
    "response_body=",
    "response_body_text=",
    "raw_",
    "oracle=",
    "evaluator=",
    "route_literal=",
    "route_name=",
    "family=",
    "image=",
    "url=",
    "path=",
)
ROLE_STEP = tuple(
    [f"belief_probe_role={value}" for value in ("candidate", "reference", "negative", "replay")]
    + [f"belief_process_step={value}" for value in ("preflight", "baseline", "failure", "repair", "replay")]
)
TARGET_RESERVED = (
    "[TARGET_BOS]",
    "[TARGET_EOS]",
    "question=none",
    "question=ask_typed",
    "question=ask_failure",
    "question=review_negative",
    "next_action=select_probe_variant",
    "next_action=send_probe",
    "next_action=assemble_rule_ir",
    "next_action=repair",
    "next_action=abstain",
    "repair_action=none",
    "repair_action=observe",
    "safe_to_send=0",
    "safe_to_send=1",
    "transport_ref=request_method",
    "field_role_ref=parameter_role",
    "encoding_ref=encoding_chain",
    "probe_variant_ref=source_attested_candidate",
    "probe_variant_ref=reference",
    "probe_variant_ref=negative_control",
    "probe_variant_ref=none",
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
        raise ValueError(f"expected object: {path}")
    return value


def build(dataset: Mapping[str, Any], base: Mapping[str, Any], *, dataset_path: Path = DEFAULT_DATASET, base_path: Path = DEFAULT_BASE) -> dict[str, Any]:
    rows = [row for row in dataset.get("records") or [] if isinstance(row, Mapping)]
    train_rows = [row for row in rows if row.get("split") == "train"]
    holdout_rows = [row for row in rows if row.get("split") != "train"]
    context = set(str(token) for token in base.get("context_tokens") or [])
    target = set(str(token) for token in base.get("target_tokens") or [])
    context.update(str(token) for row in train_rows for token in row.get("context_tokens") or [])
    # Structural enum inventory only: no holdout sequence or target label is
    # used.  This keeps the vocabulary coordinate system complete without
    # leaking implementation labels into optimization.
    context.update(str(token) for row in holdout_rows for token in row.get("context_tokens") or [])
    context.update(ROLE_STEP)
    target.update(TARGET_RESERVED)
    base_forbidden = sorted(token for token in [*context, *target] if any(marker in token.casefold() for marker in FORBIDDEN))
    # Old PG-331 vocabulary manifests may contain a legacy evaluator token;
    # remove it explicitly and retain the removal in the manifest rather than
    # letting it reach the new model or silently claiming append-only purity.
    context.difference_update(base_forbidden)
    target.difference_update(base_forbidden)
    forbidden = sorted(token for token in [*context, *target] if any(marker in token.casefold() for marker in FORBIDDEN))
    if forbidden:
        raise ValueError("vocabulary firewall rejected abstract token inventory")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_append_only_inventory_with_legacy_filter" if base_forbidden else "diagnostic_append_only_inventory",
        "purpose": "PG-343 role/step-bound target-conditioned vocabulary; holdout context types inventory-only",
        "dataset_sha256": str(dataset.get("dataset_sha256", "")),
        "dataset_file_sha256": _file_sha(dataset_path),
        "base_vocabulary_sha256": _file_sha(base_path),
        "context_tokens": sorted(context),
        "target_tokens": sorted(target),
        "counts": {"train_rows_used_for_sequences": len(train_rows), "holdout_rows_used_for_context_inventory_only": len(holdout_rows), "context_vocabulary_size": len(context), "target_vocabulary_size": len(target), "unknown_target_reserved": 0},
        "holdout_policy": {"context_inventory_only": True, "context_sequences_used_for_training": False, "target_labels_used_for_vocabulary": False, "holdout_target_sequences_read": False},
        "append_only": True,
        "forbidden_tokens": forbidden,
        "base_forbidden_tokens_removed": base_forbidden,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "vocabulary_sha256": "",
    }
    result["vocabulary_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-343 role-bound vocabulary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(_load(args.dataset), _load(args.base), dataset_path=args.dataset, base_path=args.base)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
