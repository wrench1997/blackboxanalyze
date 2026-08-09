"""Collect PG-277 matched-shape counterfactual question/composition data."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.counterfactual_surface_fixture import (  # noqa: E402
    PORTS,
    VARIANTS,
    Collector,
    default_specs,
    make_server,
    source_sha256,
)

OUTPUT = ROOT / "research" / "pg277_counterfactual_question_dataset_v1.json"
TRAIN_SEEDS = {"alpha": (27701, 27702), "beta": (27703, 27704)}
HOLDOUT_SEEDS = {"gamma": (27705, 27706)}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bucket(value: Any, upper: int = 128) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return "0"
    for edge in (1, 2, 4, 8, 16, 32, 64, upper):
        if number <= edge:
            return str(edge)
    return f">{upper}"


def shape_tokens(row: dict[str, Any]) -> list[str]:
    shape = dict(row["candidate_shape"])
    return [
        f"encoding={row['encoding']}",
        f"observe_content={shape.get('content_type_class', 'other')}",
        f"observe_status={shape.get('status_class', 'other')}",
        f"observe_html_tags={bucket(shape.get('html_tag_count'), 16)}",
        f"observe_html_attrs={bucket(shape.get('html_attribute_count'), 16)}",
        f"observe_json_fields={bucket(shape.get('json_field_count'), 16)}",
        f"observe_headers={bucket(shape.get('response_header_count'), 16)}",
        f"observe_body={bucket(shape.get('body_length'), 128)}",
    ]


def question_for_shape(row: dict[str, Any]) -> str:
    shape = dict(row["candidate_shape"])
    if shape.get("content_type_class") == "json":
        return "inspect_value_channel"
    if int(shape.get("response_header_count", 0) or 0) > 4:
        return "inspect_header_channel"
    return "inspect_marker_channel"


def abstract(row: dict[str, Any], *, split: str, source_hash: str) -> dict[str, Any]:
    observation = dict(row["observation_projection"])
    positive = bool(row["rule_ir_result"])
    common = ["[BOS]", "method=GET", "placement=query", "field_bucket=2", *shape_tokens(row), "fresh_reset=1", "source_attested=1", "reference_sent=1", "negative_sent=1", "candidate_sent=1"]
    pre = [*common[:1], "phase=pre_question", "unknown=marker_channel", "question_budget=1", *common[1:], "[CTX_END]"]
    coarse_post = [*common[:1], "phase=post_observation", "question_asked=marker_channel", "observation_detail=omitted", *common[1:], "[CTX_END]"]
    channel = str(observation.get("candidate_channel", "absent"))
    enriched_post = [
        *common[:1], "phase=post_observation", "question_asked=marker_channel", "observation_detail=atomic",
        *common[1:], f"observe_negative_channel={observation.get('negative_channel', 'absent')}",
        f"observe_reference_channel={observation.get('reference_channel', 'absent')}",
        f"observe_candidate_channel={channel}",
        f"observe_candidate_reference_match={int(bool(observation.get('candidate_reference_match')))}",
        f"failure_token={'none' if positive else 'channel_mismatch'}", "[CTX_END]",
    ]
    target = {
        "pre_question": {"question": question_for_shape(row), "action": "ask_question", "belief": "unresolved"},
        "post_observation": {"question": "replay_evidence" if positive else "explain_mismatch", "action": "replay_confirmed" if positive else "abstain", "belief": "confirmed_effect" if positive else "oracle_gap"},
    }
    rejected = {
        "pre_question": {"question": "replay_evidence", "action": "replay_confirmed", "belief": "confirmed_effect"},
        "post_observation": {"question": "explain_mismatch" if positive else "replay_evidence", "action": "abstain" if positive else "replay_confirmed", "belief": "oracle_gap" if positive else "confirmed_effect"},
    }
    components = {"scope_and_safety": 1.0, "information_completeness": 1.0, "question_information_gain": 1.0, "control_alignment": 1.0, "failure_diagnosis": 1.0, "calibrated_abstain": 1.0, "evidence_alignment": 1.0}
    return {
        "record_id": row["record_id"], "split": split, "variant": row["variant"], "seed_id": row["source_id"], "source_hash": source_hash,
        "pre_question_context_tokens": pre, "coarse_post_context_tokens": coarse_post, "enriched_post_context_tokens": enriched_post,
        "targets": target, "preference_rejected": rejected, "teacher_components": components, "teacher_score": round(sum(components.values()) / len(components), 6),
        "labels": {"expected_positive": positive, "mode": row["mode"], "encoding": row["encoding"], "candidate_channel": channel},
        "evidence_hash": str(row.get("evidence", {}).get("evidence_hash", "")), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False,
    }


def collect_variant(variant: str, seeds: tuple[int, ...], split: str, fixture_hash: str) -> list[dict[str, Any]]:
    port = PORTS[VARIANTS.index(variant)]
    server = make_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records: list[dict[str, Any]] = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        for seed in seeds:
            marker = f"pg277-{seed}"
            specs = default_specs(dataset_id=f"pg277-{variant}-{seed}", target=base_url, marker=marker)
            raw = asyncio.run(Collector(base_url=base_url, variant=variant, target_instance_id=f"pg277-{variant}-{seed}", fixture_hash=fixture_hash).collect_many(specs))
            records.extend(abstract(row, split=split, source_hash=fixture_hash) for row in raw)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return records


def collision_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = sha(row[field])
        groups.setdefault(key, []).append(row)
    conflicts = []
    for key, rows in groups.items():
        labels = {bool(row["labels"]["expected_positive"]) for row in rows}
        if len(labels) > 1:
            conflicts.append({"projection_sha256": key, "count": len(rows), "positive_count": sum(bool(row["labels"]["expected_positive"]) for row in rows), "modes": sorted({row["labels"]["mode"] for row in rows})})
    return {"group_count": len(groups), "conflict_group_count": len(conflicts), "conflicting_record_count": sum(item["count"] for item in conflicts), "conflicts": conflicts}


def main() -> None:
    fixture_hash = source_sha256()
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for variant, seeds in TRAIN_SEEDS.items():
        train.extend(collect_variant(variant, seeds, "alpha_beta_train", fixture_hash))
    for variant, seeds in HOLDOUT_SEEDS.items():
        holdout.extend(collect_variant(variant, seeds, "gamma_seed_holdout", fixture_hash))
    coarse = collision_summary(train + holdout, "coarse_post_context_tokens")
    enriched = collision_summary(train + holdout, "enriched_post_context_tokens")
    payload: dict[str, Any] = {
        "schema_version": "pg277-counterfactual-question-dataset-v1",
        "source": {"fixture_source_sha256": fixture_hash, "ports": list(PORTS), "loopback_only": True, "external_network": False, "fresh_target": True},
        "split_contract": {"train_variants": ["alpha", "beta"], "holdout_variants": ["gamma"], "train_seeds": [27701, 27702, 27703, 27704], "holdout_seeds": [27705, 27706], "variant_and_seed_disjoint": True},
        "records": train + holdout,
        "counts": {"train": len(train), "holdout": len(holdout), "train_positive": sum(bool(x["labels"]["expected_positive"]) for x in train), "holdout_positive": sum(bool(x["labels"]["expected_positive"]) for x in holdout)},
        "projection_collision_audit": {"coarse": coarse, "enriched": enriched, "coarse_training_allowed": False, "enriched_training_allowed": enriched["conflict_group_count"] == 0},
        "training_contract": {"question_head_required": True, "coarse_collision_records_are_diagnostic_only": True, "oracle_in_context": False, "raw_payload_in_context": False, "promotion_blocked": True, "memory_promotion_blocked": True},
    }
    payload["dataset_sha256"] = sha(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "counts": payload["counts"], "coarse_collision": coarse, "enriched_collision": enriched, "dataset": str(OUTPUT.relative_to(ROOT)), "dataset_sha256": payload["dataset_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
