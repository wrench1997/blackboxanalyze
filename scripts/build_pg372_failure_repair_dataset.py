"""Build a bounded PG-372 failure→repair candidate dataset.

The builder composes only the abstract context/Rule-IR slots already present
in PG-362 and PG-367.  It annotates existing rows with an abstract pairing key
so a decoder can learn a changed action after failure.  It does not synthesize
wire, payloads, URLs or evaluator answers, and it keeps the train-only
vocabulary contract explicit for holdout rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "pg362": ROOT / "research" / "pg362_full_rule_ir_dataset_v1.json",
    "pg367": ROOT / "research" / "pg367_waf_staircase_dataset_v2.json",
}
SLOTS = ("question", "ask_reason", "next_action", "repair_action", "transport_ref", "field_role_ref", "encoding_ref", "syntax_category_ref", "probe_variant_ref", "safe_to_send", "payload_shape_ref", "oracle_ref", "negative_control_presence_ref")
RAW_FRAGMENTS = ("raw_payload=", "payload=", "wire=", "response_body=", "raw_response=", "http://", "https://", "url=")
FAILURE_SIGNAL_KEYS = {
    "failure_signature", "failure_failure_class", "failure_failure_stage",
    "failure_filter_action", "failure_transform_class", "failure_blocked_reason_class",
    "failure_parse_error_class", "failure_encoding_error_class", "failure_redirect_error_class",
    "failure_repair_axis", "failure_repair_outcome",
}
NEUTRAL_FAILURE_VALUES = {"none", "absent", "zero", "empty", "identity", "allow", "not_applicable"}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _values(tokens: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            if key in SLOTS:
                result[key] = value
    return result


def _failure_and_repair(context: list[str], values: Mapping[str, str]) -> tuple[bool, bool]:
    failure = any("=" in token and token.split("=", 1)[0] in FAILURE_SIGNAL_KEYS and token.split("=", 1)[1] not in NEUTRAL_FAILURE_VALUES for token in context) or values.get("question") == "ask_failure" or values.get("next_action") == "repair"
    repair = values.get("repair_action", "none") != "none" or values.get("next_action") == "repair"
    return failure, repair


def _surface_key(source: str, split: str, context: list[str], values: Mapping[str, str]) -> str:
    # Keep only broad page/DOM structure and the two high-information target
    # axes.  Method/encoding/failure tokens stay out of this key, so a pair is
    # formed only as an abstract counterfactual family, never from a literal.
    structural = [token for token in context if token.startswith(("document_", "dom_", "element_", "navigation_", "javascript_", "js_"))]
    return _sha([source, split, structural, values.get("syntax_category_ref"), values.get("payload_shape_ref")])


def build(datasets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    candidates: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    context_train: set[str] = set()
    target_train: set[str] = {"[TARGET_BOS]", "[TARGET_EOS]"}
    for source, dataset in datasets.items():
        for index, raw in enumerate(list(dataset.get("records") or [])):
            if not isinstance(raw, Mapping):
                failures.append(f"{source}:row_{index}:not_mapping")
                continue
            context = [str(token) for token in raw.get("context_tokens") or []] if isinstance(raw.get("context_tokens"), list) else []
            target = [str(token) for token in raw.get("target_tokens") or []] if isinstance(raw.get("target_tokens"), list) else []
            values = _values(target)
            split = str(raw.get("split", ""))
            if not context or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
                failures.append(f"{source}:row_{index}:boundary_or_context")
                continue
            if split not in {"train", "implementation_holdout"} or any(slot not in values for slot in SLOTS):
                failures.append(f"{source}:row_{index}:split_or_slot")
                continue
            if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context, *target]):
                failures.append(f"{source}:row_{index}:raw_token")
                continue
            firewall = raw.get("context_firewall")
            if not isinstance(firewall, Mapping) or firewall.get("sidecars_off_context") is not True:
                failures.append(f"{source}:row_{index}:context_firewall")
                continue
            failure, repair = _failure_and_repair(context, values)
            key = _surface_key(str(source), split, context, values)
            item = {
                "record_id": _sha([source, index, split, context, target]),
                "source": str(source),
                "split": split,
                "context_tokens": context,
                "target_tokens": target,
                "failure_present": failure,
                "repair_present": repair,
                "syntax_category_ref": values["syntax_category_ref"],
                "payload_shape_ref": values["payload_shape_ref"],
                "pair_id": key,
                "pair_role": "failure_repair" if failure and repair else ("failure" if failure else ("repair" if repair else "baseline")),
                "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
                "raw_payload_stored": False,
                "raw_response_body_stored": False,
                "oracle_answer_in_context": False,
                "training_eligible": False,
            }
            item["record_sha256"] = _sha(item)
            row_index = len(candidates)
            candidates.append(item)
            groups[key].append(row_index)
            if split == "train":
                context_train.update(context)
                target_train.update(target)

    # Identical abstract examples can be present in both source implementations.
    # A holdout copy wins over a train copy so no exact sequence leaks into the
    # training side; this is a provenance-only dedupe, not a relabeling.
    deduped: dict[str, dict[str, Any]] = {}
    duplicate_groups = 0
    duplicate_rows_removed = 0
    holdout_precedence_train_rows_removed = 0
    for item in candidates:
        signature = _sha([item["context_tokens"], item["target_tokens"]])
        previous = deduped.get(signature)
        if previous is None:
            deduped[signature] = item
            continue
        duplicate_groups += 1
        if previous["split"] == "train" and item["split"] != "train":
            holdout_precedence_train_rows_removed += 1
            deduped[signature] = item
        elif item["split"] == "train" and previous["split"] != "train":
            holdout_precedence_train_rows_removed += 1
        else:
            # Same-split duplicate: retain first deterministic source row.
            duplicate_rows_removed += 1
    candidates = list(deduped.values())
    # Recompute the vocabulary after dedupe.  A train row removed in favor of
    # its holdout twin must not leave its token in the train vocabulary.
    context_train = {token for row in candidates if row["split"] == "train" for token in row["context_tokens"]}
    target_train = {"[TARGET_BOS]", "[TARGET_EOS]"} | {token for row in candidates if row["split"] == "train" for token in row["target_tokens"]}
    groups = defaultdict(list)
    for row_index, item in enumerate(candidates):
        groups[item["pair_id"]].append(row_index)

    paired = 0
    for key, indexes in groups.items():
        labels = {candidates[index]["pair_role"] for index in indexes}
        has_failure = bool(labels & {"failure", "failure_repair"})
        has_repair = bool(labels & {"repair", "failure_repair"})
        if has_failure and has_repair:
            paired += 1
            for index in indexes:
                candidates[index]["paired_failure_repair"] = True
        else:
            for index in indexes:
                candidates[index]["paired_failure_repair"] = False
    # Pair annotations are part of the abstract record contract; recompute the
    # row digest after annotation so the integrity hash covers the final row.
    for item in candidates:
        item.pop("record_sha256", None)
        item["record_sha256"] = _sha(item)

    holdout = [row for row in candidates if row["split"] != "train"]
    holdout_unknown_context = sorted({token for row in holdout for token in row["context_tokens"]} - context_train)
    holdout_unknown_target = sorted({token for row in holdout for token in row["target_tokens"]} - target_train)
    if holdout_unknown_context or holdout_unknown_target:
        failures.append("holdout_vocabulary_gap")
    if not paired:
        failures.append("failure_repair_pairs_missing")
    declared_context: set[str] = set()
    declared_target: set[str] = set()
    for dataset in datasets.values():
        vocabulary = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
        declared_context.update(str(token) for token in list(vocabulary.get("context_tokens") or []))
        declared_target.update(str(token) for token in list(vocabulary.get("target_tokens") or []))
    return {
        "schema_version": "pg372-failure-repair-dataset-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "records": candidates,
        "vocabulary": {"context_tokens": sorted(context_train), "target_tokens": sorted(target_train), "append_only": True, "built_from_train_only": True},
        "counts": {
            "records": len(candidates),
            "train_rows": sum(row["split"] == "train" for row in candidates),
            "holdout_rows": len(holdout),
            "failure_rows": sum(row["failure_present"] for row in candidates),
            "repair_rows": sum(row["repair_present"] for row in candidates),
            "paired_failure_repair_groups": paired,
            "syntax_category_values": len({row["syntax_category_ref"] for row in candidates}),
            "payload_shape_values": len({row["payload_shape_ref"] for row in candidates}),
            "holdout_unknown_context_tokens": len(holdout_unknown_context),
            "holdout_unknown_target_tokens": len(holdout_unknown_target),
            "training_eligible_rows": 0,
            "duplicate_groups": duplicate_groups,
            "duplicate_rows_removed": duplicate_rows_removed,
            "train_rows_removed_by_holdout_precedence": holdout_precedence_train_rows_removed,
        },
        "declared_ontology_inventory": {"context_token_count": len(declared_context), "target_token_count": len(declared_target), "context_inventory_sha256": _sha(sorted(declared_context)), "target_inventory_sha256": _sha(sorted(declared_target)), "slot_order": list(SLOTS)},
        "train_only_gap": {"context_unknown_count": len(holdout_unknown_context), "target_unknown_count": len(holdout_unknown_target), "unknown_token_hash": _sha([holdout_unknown_context, holdout_unknown_target]), "blocked": bool(holdout_unknown_context or holdout_unknown_target)},
        "holdout_contract": {"train_only_vocabulary": True, "source_split_preserved": True, "holdout_precedence_dedupe": True, "unknown_token_hash": _sha([holdout_unknown_context, holdout_unknown_target])},
        "target_contract": {"failure_repair_pairing": True, "syntax_category_ref": True, "payload_shape_ref": True, "raw_payload_in_context": False, "evaluator_sidecar_read": False},
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "raw_material_available": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-372 abstract failure-repair candidate dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg372_failure_repair_dataset_v1.json")
    args = parser.parse_args()
    datasets = {key: json.loads(path.read_text(encoding="utf-8-sig")) for key, path in SOURCES.items()}
    result = build(datasets)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
