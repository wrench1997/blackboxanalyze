"""Build PG-273 abstract question/composition data from two implementations.

Implementation v1 is the training source; the separately implemented v2 is a
strict holdout.  The model-visible context contains only generic observable
shape tokens (content type, status class, bounded counts, encoding and
transport).  Surface roles, typed oracle fields and raw wire material stay in
labels/metadata and never enter context.
"""

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

from app.heterogeneous_surface_fixture import (  # noqa: E402
    HETERO_SURFACE_PORTS,
    HeterogeneousSurfaceCollector,
    default_heterogeneous_surface_specs,
    heterogeneous_surface_source_sha256,
    make_heterogeneous_surface_fixture_server,
)
from app.heterogeneous_surface_fixture_v2 import (  # noqa: E402
    HeterogeneousSurfaceV2Collector,
    V2_PORTS,
    default_heterogeneous_surface_v2_specs,
    heterogeneous_surface_v2_source_sha256,
    make_heterogeneous_surface_v2_fixture_server,
)

OUTPUT = ROOT / "research" / "pg273_composition_dataset_v1.json"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _bucket(value: Any, *, upper: int = 8) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0
    return str(min(max(number, 0), upper))


def _context(record: dict[str, Any]) -> list[str]:
    shape = dict(record.get("surface_shape") or {})
    response = dict(record.get("response_projection") or {})
    content = str(shape.get("content_type_class", "other"))
    status = str(shape.get("status_class", "other"))
    body_length = shape.get("body_length", response.get("body_length", 0))
    delta = shape.get("body_length_delta_abs", response.get("body_length_delta_abs", 0))
    # The heterogeneous collectors keep transport encoding in the signed
    # probe_artifact envelope, not in replay (replay describes the HTTP
    # transport).  Falling back to replay is retained for older records, but
    # silently defaulting to plain would collapse a real encoding dimension
    # and make the composition split look easier than it is.
    artifact = dict(record.get("probe_artifact") or {})
    replay = dict(record.get("replay") or {})
    encoding = str(artifact.get("encoding") or replay.get("encoding") or "plain")
    return [
        "[BOS]", "phase=observe", "question=surface_effect", "question=encoding_transport",
        "method=GET", "placement=query", "field_bucket=2", f"encoding={encoding}",
        f"observe_content={content}", f"observe_status={status}",
        f"observe_html_tags={_bucket(shape.get('html_tag_count'))}",
        f"observe_html_attrs={_bucket(shape.get('html_attribute_count'))}",
        f"observe_json_fields={_bucket(shape.get('json_field_count'))}",
        f"observe_headers={_bucket(shape.get('response_header_count'))}",
        f"observe_body={_bucket(body_length, upper=16)}", f"observe_delta={_bucket(delta, upper=16)}",
        "fresh_reset=1", "source_attested=1", "reference_sent=1", "negative_sent=1",
        "candidate_sent=1", "repair_attempted=0", "failure_observed=1", "step_budget=4", "[CTX_END]",
    ]


def _target(positive: bool) -> list[str]:
    if positive:
        return [
            "[TARGET_BOS]", "phase=baseline", "action=negative_control", "failure=negative_clean", "next_action=candidate_probe",
            "phase=reference", "action=reference_probe", "failure=reference_observed", "next_action=candidate_probe",
            "phase=candidate", "action=candidate_probe", "failure=none", "next_action=replay_confirmed",
            "final_belief=confirmed_effect", "[TARGET_EOS]",
        ]
    return [
        "[TARGET_BOS]", "phase=baseline", "action=negative_control", "failure=negative_clean", "next_action=candidate_probe",
        "phase=reference", "action=reference_probe", "failure=reference_observed", "next_action=candidate_probe",
        "phase=candidate", "action=candidate_probe", "failure=surface_oracle_gap", "next_action=diagnose_failure",
        "phase=diagnose", "action=abstain", "failure=no_typed_repair_available", "next_action=abstain",
        "final_belief=oracle_gap", "[TARGET_EOS]",
    ]


def _corrupt(target: list[str], positive: bool) -> tuple[list[str], str]:
    rejected = list(target)
    if positive:
        for index, token in enumerate(rejected):
            if token == "next_action=replay_confirmed":
                rejected[index] = "next_action=confirm_without_replay"
        for index, token in enumerate(rejected):
            if token == "final_belief=confirmed_effect":
                rejected[index] = "final_belief=unsupported_positive"
        return rejected, "premature_positive_without_replay"
    for index, token in enumerate(rejected):
        if token == "action=negative_control":
            rejected[index] = "action=candidate_probe"
            break
    return rejected, "skipped_negative_control"


def _teacher_score(positive: bool) -> dict[str, float]:
    return {
        "scope_and_safety": 1.0,
        "information_completeness": 1.0,
        "probe_utility": 1.0,
        "failure_diagnosis": 1.0 if not positive else 0.9,
        "repair_quality": 0.9,
        "oracle_and_evidence_alignment": 1.0,
        "calibrated_abstain": 1.0,
    }


def _collect_v1() -> tuple[list[dict[str, Any]], str]:
    source_hash = heterogeneous_surface_source_sha256()
    output: list[dict[str, Any]] = []
    for index, (port, variant) in enumerate(zip(HETERO_SURFACE_PORTS, ("alpha", "beta", "gamma"))):
        server = make_heterogeneous_surface_fixture_server(port=port, variant=variant)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            specs = default_heterogeneous_surface_specs(dataset_id=f"pg273-v1-{variant}", target=base_url, marker=f"pg273-v1-{index}")
            rows = asyncio.run(HeterogeneousSurfaceCollector(base_url=base_url, target_instance_id=f"pg273-v1-{variant}", source_hash=source_hash).collect_many(specs))
            output.extend(rows)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
    return output, source_hash


def _collect_v2() -> tuple[list[dict[str, Any]], str]:
    source_hash = heterogeneous_surface_v2_source_sha256()
    output: list[dict[str, Any]] = []
    for index, (port, variant) in enumerate(zip(V2_PORTS, ("alpha", "beta", "gamma"))):
        server = make_heterogeneous_surface_v2_fixture_server(port=port, variant=variant)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{port}"
            specs = default_heterogeneous_surface_v2_specs(dataset_id=f"pg273-v2-{variant}", target=base_url, marker=f"pg273-v2-{index}")
            rows = asyncio.run(HeterogeneousSurfaceV2Collector(base_url=base_url, target_instance_id=f"pg273-v2-{variant}", source_hash=source_hash).collect_many(specs))
            output.extend(rows)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
    return output, source_hash


def _abstract(records: list[dict[str, Any]], *, split: str, source_hash: str, implementation: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        positive = bool(record.get("rule_ir_result"))
        context = _context(record)
        target = _target(positive)
        components = _teacher_score(positive)
        score = round(sum(components.values()) / len(components), 6)
        rejected, rejection_reason = _corrupt(target, positive)
        rows.append({
            "record_id": record["sample_id"],
            "split": split,
            "implementation": implementation,
            "source_hash": source_hash,
            "context_tokens": context,
            "target_tokens": target,
            "labels": {"expected_positive": positive, "surface_role": record["semantic"]["surface_role"], "encoding": str((record.get("probe_artifact") or {}).get("encoding") or (record.get("replay") or {}).get("encoding") or "plain")},
            "teacher_components": components,
            "teacher_score": score,
            "preference": {"rejected_target_tokens": rejected, "rejection_reason": rejection_reason, "rejected_score": 0.0},
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "oracle_in_context": False,
        })
    return rows


def main() -> None:
    v1, v1_hash = _collect_v1()
    v2, v2_hash = _collect_v2()
    train = _abstract(v1, split="implementation_v1_train", source_hash=v1_hash, implementation="heterogeneous_surface_v1")
    holdout = _abstract(v2, split="implementation_v2_holdout", source_hash=v2_hash, implementation="heterogeneous_surface_v2")
    payload: dict[str, Any] = {
        "schema_version": "pg273-composition-question-dataset-v1",
        "source": {"train_implementation": "heterogeneous_surface_v1", "holdout_implementation": "heterogeneous_surface_v2", "train_source_sha256": v1_hash, "holdout_source_sha256": v2_hash, "loopback_only": True, "external_network": False},
        "split_contract": {"implementation_disjoint": True, "surface_role_in_context": False, "oracle_in_context": False, "raw_payload_in_context": False, "fresh_target": True},
        "records": train + holdout,
        "counts": {"train": len(train), "holdout": len(holdout), "train_positive": sum(row["labels"]["expected_positive"] for row in train), "holdout_positive": sum(row["labels"]["expected_positive"] for row in holdout)},
        "training_contract": {"question_tokens_present": True, "generic_observation_tokens_only": True, "teacher_scores_are_labels": True, "preference_pairs_are_abstract": True, "promotion_blocked": True, "memory_promotion_blocked": True},
    }
    payload["dataset_sha256"] = _sha(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "counts": payload["counts"], "dataset": str(OUTPUT.relative_to(ROOT)), "dataset_sha256": payload["dataset_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
