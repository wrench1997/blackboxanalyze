"""Build the PG-331 append-only vocabulary manifest.

The manifest is generated from the versioned ontology plus observed abstract
tokens.  It never drops low-frequency axes to improve a metric.  If the
information audit is blocked, the manifest is still useful for diagnosis but
is explicitly not training-eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg323_decoy_ask_anchor_dataset_v1.json"
ONTOLOGY = RESEARCH / "pg331_web_token_ontology_v1.json"
PAYLOAD_ONTOLOGY = RESEARCH / "pg348_payload_shape_ontology_v1.json"
AUDIT = RESEARCH / "pg331_information_preservation_audit_v1.json"
OUTPUT = RESEARCH / "pg331_web_token_vocabulary_v1.json"
TOKENIZER = ROOT / "app" / "pg331_web_tokenizer.py"
RULES = RESEARCH / "improvement_rules.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path.name)
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build(
    *,
    dataset_path: Path | None = None,
    ontology_path: Path | None = None,
    audit_path: Path | None = None,
    base_vocabulary_path: Path | None = None,
    payload_ontology_path: Path | None = None,
) -> dict[str, Any]:
    selected_dataset = (dataset_path or DATASET).resolve()
    selected_ontology = (ontology_path or ONTOLOGY).resolve()
    selected_audit = (audit_path or AUDIT).resolve()
    selected_base = (base_vocabulary_path or OUTPUT).resolve()
    selected_payload_ontology = (payload_ontology_path or PAYLOAD_ONTOLOGY).resolve()
    dataset = _load(selected_dataset)
    ontology = _load(selected_ontology)
    payload_ontology = _load(selected_payload_ontology) if selected_payload_ontology.exists() else {}
    audit = _load(selected_audit) if selected_audit.exists() else {}
    # The output file is also the default append-only base.  Read it before
    # writing the new manifest so a rebuild cannot silently discard tokens
    # from an earlier ontology/source split.  A first build simply has no
    # existing base and starts from the declared inventory.
    base = _load(selected_base) if selected_base.exists() else {}
    rules = _load(RULES) if RULES.exists() else {}
    # Append-only means a new source-row collection can add observed abstract
    # tokens without erasing tokens learned from earlier implementations.
    context: set[str] = {str(token) for token in base.get("context_tokens") or []}
    target: set[str] = {str(token) for token in base.get("target_tokens") or []}
    for row in dataset.get("records", []):
        context.update(str(token) for token in row.get("context_tokens") or [])
        target.update(str(token) for token in row.get("target_tokens") or [])
    payload_slot = dict(payload_ontology.get("slot") or {})
    payload_values = [str(token) for token in list(payload_slot.get("allowed_tokens") or []) if str(token)]
    target.update(f"payload_shape_ref={token}" for token in payload_values)
    reserved = set(str(token) for token in (ontology.get("reserved_tokens") or {}).get("universal", []))
    reserved.update(str(token) for token in (ontology.get("reserved_tokens") or {}).get("bucket_policy", []))
    observed_context_count = len(context)
    observed_target_count = len(target)
    # Reserve the complete ontology inventory up front.  These are not
    # hand-picked features: every declared field has observed/not_observed
    # slots, so a later collector cannot silently invent a smaller vocabulary.
    ontology_inventory: set[str] = set()
    ontology_inventory.update(
        {
            "chunk_boundary=begin",
            "chunk_boundary=end",
            *{f"chunk_shape={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
            *{f"chunk_index={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
            *{f"chunk_count={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
            *{f"chunk_digest=b{value:02x}" for value in range(256)},
        }
    )
    for axis, spec in dict(ontology.get("axes") or {}).items():
        axis_name = str(axis)
        presence = str(spec.get("presence_token", ""))
        if presence:
            ontology_inventory.update({f"{presence}=observed", f"{presence}=not_observed"})
        ontology_inventory.update({f"axis_begin={axis_name}", f"axis_end={axis_name}"})
        for field in list(spec.get("fields") or []):
            key = f"{axis_name}_field_{str(field)}"
            ontology_inventory.update({f"{key}=observed", f"{key}=absent", f"{key}=not_observed", f"{key}=unknown"})
    # Semantic parameter roles are a versioned ontology contract rather than
    # values discovered opportunistically in one dataset.  Reserve them in
    # the context vocabulary before any live rows arrive, while keeping raw
    # parameter names out of the model stream.
    role_taxonomy = dict((rules.get("pg331_vocabulary_contract_current") or {}).get("parameter_role_taxonomy") or {})
    role_values = [str(role) for role in list(role_taxonomy.get("roles") or [])]
    ontology_inventory.update({f"param_role={role}" for role in role_values if role})
    context.update(ontology_inventory)
    shared = sorted((context | target | reserved) - {"[PAD]", "[UNK]"})
    context_tokens = sorted(context | reserved)
    target_tokens = sorted(target | reserved)
    axes = {
        str(axis): {
            "presence_token": str(spec.get("presence_token", "")),
            "token_prefixes": [str(item) for item in list(spec.get("token_prefixes") or [])],
            "fields": [str(item) for item in list(spec.get("fields") or [])],
        }
        for axis, spec in dict(ontology.get("axes") or {}).items()
    }
    document: dict[str, Any] = {
        "protocol_id": "pg-pk-331-web-token-vocabulary-v1",
        "schema_version": "pg331-web-token-vocabulary-v1",
        "status": "training_eligible" if audit.get("status") == "passed" else "diagnostic_only_audit_blocked",
        "ontology": _display_path(selected_ontology),
        "ontology_sha256": _digest(ontology),
        "dataset": _display_path(selected_dataset),
        "dataset_sha256": _digest(dataset),
        "tokenizer": "app/pg331_web_tokenizer.py",
        "tokenizer_sha256": _file_sha256(TOKENIZER) if TOKENIZER.exists() else "",
        "payload_shape_ontology": _display_path(selected_payload_ontology) if payload_ontology else None,
        "payload_shape_ontology_sha256": _file_sha256(selected_payload_ontology) if payload_ontology else "",
        "audit": _display_path(selected_audit),
        "audit_status": str(audit.get("status", "missing")),
        "axes": axes,
        "vocabulary_policy": {
            "append_only": True,
            "ontology_reserved_tokens_included": True,
            "low_frequency_axis_pruning": False,
            "unknown_and_not_observed_distinct": True,
            "raw_literal_tokens_allowed": False,
            "context_target_vocabularies_separate": True,
            "overflow_requires_loss_report": True,
            "tokenizer_hash_required": True,
            "parameter_role_taxonomy_version": str(role_taxonomy.get("version", "")),
            "parameter_role_values_reserved": role_values,
            "base_vocabulary_append_only": bool(base),
            "base_vocabulary": _display_path(selected_base) if base else None,
            "base_vocabulary_sha256": _file_sha256(selected_base) if base else "",
            "payload_shape_slot_reserved": bool(payload_values),
            "raw_payload_literals_allowed": False,
        },
        "counts": {"context_observed": observed_context_count, "target_observed": observed_target_count, "ontology_inventory": len(ontology_inventory), "reserved": len(reserved), "context_total": len(context_tokens), "target_total": len(target_tokens), "shared_total": len(shared)},
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "shared_tokens": shared,
        "training_eligibility": {"allowed": audit.get("status") == "passed", "reason": "all ontology axes and split/entropy gates must pass before this manifest can feed training"},
    }
    document["vocabulary_sha256"] = ""
    document["vocabulary_sha256"] = _digest(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--ontology", type=Path, default=ONTOLOGY)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--base-vocabulary", type=Path, default=OUTPUT)
    parser.add_argument("--payload-ontology", type=Path, default=PAYLOAD_ONTOLOGY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    document = build(dataset_path=args.dataset, ontology_path=args.ontology, audit_path=args.audit, base_vocabulary_path=args.base_vocabulary, payload_ontology_path=args.payload_ontology)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, ensure_ascii=False, indent=2) if args.json else f"{document['status']}: {document['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
