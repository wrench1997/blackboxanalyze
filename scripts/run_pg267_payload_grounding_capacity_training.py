# -*- coding: utf-8 -*-
"""PG-267: train the Rule-IR adapter with grounded payload trajectories.

PG-266 is an evidence-producing, local-only replay lane.  This wrapper adds
its *abstract* trajectory records to the audited PG-265 training pool.  The
human catalog keeps the exact wire values for review, but this training lane
receives only bounded Rule-IR tokens and hashes; payload strings, response
bodies, and oracle decisions remain outside the model input.

The underlying PG-265 runner still performs the sequential capacity sweep and
all of its existing split/canary checks.  This file only supplies the fresh
PG-267 records and writes a separate, auditable namespace for the result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG265 = _load(ROOT / "scripts" / "run_pg265_growth_augmented_large_capacity_training.py", "pg267_pg265_base")
PG260 = PG265.PG260
RESEARCH = ROOT / "research"
PG265_BASE_DATASET = RESEARCH / "pg265_growth_augmented_large_capacity_training_dataset_v1.json"
PG267_DATASET = RESEARCH / "pg267_payload_grounding_augmented_dataset_v1.json"
REPORT = RESEARCH / "pg267_payload_grounding_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg267_payload_grounding_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg267_payload_grounding_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg267_payload_grounding_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg267_payload_grounding_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg267-payload-grounding-capacity-v1"
RUN_MARKER = RESEARCH / "pg267_training_running.json"

BASE_LOADER = PG265._load_records
BASE_HOLDOUT = PG265._is_holdout
REPAIR_ACTION_MAP = {
    # PG-267's human catalog uses descriptive actions; the shared legacy
    # adapter has a closed repair vocabulary.  Keep the original action in
    # the dataset, but derive the nearest bounded training target here.
    "replay_confirmed": "retry_candidate",
    "abstain_or_repair": "recheck_oracle",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_records() -> list[dict[str, Any]]:
    """Combine the audited base with the 12 abstract PG-266 records."""
    rows = list(BASE_LOADER())
    payload = _read(PG267_DATASET)
    if str(payload.get("status", "completed_pg267_dataset_build")) not in {
        "completed_pg267_dataset_build",
        "completed",
    }:
        raise RuntimeError("PG-267 abstract dataset is not complete")
    fresh = list(payload.get("records") or [])
    if len(fresh) != 12:
        raise RuntimeError(f"PG-267 expected 12 abstract records, got {len(fresh)}")
    if any(bool(row.get("raw_payload_strings_stored")) or bool(row.get("raw_response_bodies_stored")) for row in fresh if isinstance(row, dict)):
        raise RuntimeError("PG-267 raw payload/response material must stay outside training")
    normalised_fresh: list[dict[str, Any]] = []
    for raw in fresh:
        row = dict(raw)
        action = str(row.get("repair_action", "abstain"))
        row["repair_action_original"] = action
        row["repair_action"] = REPAIR_ACTION_MAP.get(action, action)
        normalised_fresh.append(PG260._normalise(row, "pg267_payload_grounding"))
    rows.extend(normalised_fresh)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("source", "")),
            int(row.get("seed", 0) or 0),
            PG260._route(row),
            str(row.get("route_source_sha256", "")),
            str(row.get("trajectory_hash", row.get("token_hash", ""))),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _is_holdout(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    if source.startswith("pg267_"):
        # Even seeds are never used for fitting.  This leaves six unseen
        # payload-grounding trajectories for the fresh-source gate.
        return int(row.get("seed", 0) or 0) % 2 == 0
    return bool(BASE_HOLDOUT(row))


def _write_marker() -> None:
    RUN_MARKER.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "report": str(REPORT.relative_to(ROOT)),
                "protocol_id": "pg-pk-267-payload-grounding-capacity-v1",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _clear_marker() -> None:
    RUN_MARKER.unlink(missing_ok=True)


def main() -> int:
    _write_marker()
    try:
        # PG-265's main function looks up these names in its own module
        # namespace, so patching the module (not a copied function) keeps the
        # original runner's split and training implementation intact.
        PG265._load_records = _load_records
        PG265._is_holdout = _is_holdout
        PG260.FRESH_SOURCE_PREFIXES = ("pg259_", "pg260_", "pg262_", "pg264_", "pg267_")
        PG260.CAPACITY_VARIANTS = tuple(
            int(item.strip())
            for item in os.environ.get("PG267_CAPACITY_VARIANTS", "8192,12288,16384").split(",")
            if item.strip().isdigit() and int(item.strip()) > 0
        ) or (8192, 12288, 16384)
        PG260.TRAIN_STEPS = max(int(os.environ.get("PG267_TRAIN_STEPS", "170")), 1)
        PG260.MICRO_BATCH_SIZE = max(int(os.environ.get("PG267_MICRO_BATCH_SIZE", "128")), 1)
        # PG-265.main() reads its own environment names and rebinds these
        # values before entering PG-260; mirror the PG-267 knobs explicitly.
        os.environ["PG265_CAPACITY_VARIANTS"] = ",".join(str(x) for x in PG260.CAPACITY_VARIANTS)
        os.environ["PG265_TRAIN_STEPS"] = str(PG260.TRAIN_STEPS)
        os.environ["PG265_MICRO_BATCH_SIZE"] = str(PG260.MICRO_BATCH_SIZE)
        PG265.REPORT = REPORT
        PG265.DATASET = DATASET
        PG265.TRACE = TRACE
        PG265.PROTOCOL = PROTOCOL
        PG265.MARKDOWN = MARKDOWN
        PG265.ARTIFACT_DIR = ARTIFACT_DIR
        code = PG265.main()

        report = _read(REPORT)
        rows = _load_records()
        fresh_rows = [row for row in rows if str(row.get("source", "")).startswith("pg267_")]
        fresh_holdout = [row for row in fresh_rows if _is_holdout(row)]
        report.update(
            {
                "protocol_id": "pg-pk-267-payload-grounding-capacity-v1",
                "schema_version": "pg267-payload-grounding-capacity-training-report-v1",
                "status": "completed_pg267_payload_grounding_capacity_training",
                "capacity_variants": list(PG260.CAPACITY_VARIANTS),
                "architecture_change": {
                    "id": "pg267-payload-grounding-abstract-token-adapter-v1",
                    "base": "pg265-growth-augmented-large-capacity-v1",
                    "fresh_source": "PG-266 local AI/reference/negative payload-grounding replay",
                    "pg266_abstract_records": len(fresh_rows),
                    "payload_strings_off_input": True,
                    "oracle_target_off_input": True,
                    "raw_response_bodies_off_input": True,
                    "legacy_artifacts_unchanged": True,
                },
                "growth_counts": {
                    "combined_records": len(rows),
                    "pg267_records": len(fresh_rows),
                    "pg267_even_seed_holdout": len(fresh_holdout),
                    "pg267_train_records": len(fresh_rows) - len(fresh_holdout),
                },
                "evaluation_audit": {
                    "audit_id": "pg267-payload-grounding-capacity-final-audit-v1",
                    # The catalog is a human-review artifact and is not copied
                    # to the training host.  PG-267 carries its attested hash.
                    "pg266_catalog_sha256": _read(PG267_DATASET).get("source_catalog_sha256", ""),
                    "pg266_dataset_sha256": _read(PG267_DATASET).get("dataset_sha256", ""),
                    "payload_strings_in_model_input": False,
                    "oracle_target_in_model_input": False,
                    "weights_changed": False,
                },
                "promotion": {
                    "training_promotion_allowed": False,
                    "memory_promotion_allowed": False,
                    "payload_catalog_promotion_allowed": False,
                    "vulnerability_claim_allowed": False,
                    "blocked_by": list((report.get("independent_final_judge") or {}).get("reasons") or []),
                },
            }
        )
        report["report_sha256"] = ""
        report["report_sha256"] = _digest(report)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        dataset = _read(DATASET)
        dataset.update(
            {
                "schema_version": "pg267-payload-grounding-capacity-training-dataset-v1",
                "source_datasets": [str(PG265_BASE_DATASET.relative_to(ROOT)), str(PG267_DATASET.relative_to(ROOT))],
                "contract": dict(
                    dataset.get("contract") or {},
                    pg266_payload_grounding_abstract=True,
                    pg266_catalog_raw_material_review_only=True,
                    oracle_target_off_input=True,
                    raw_payload_strings_stored=False,
                    raw_response_bodies_stored=False,
                    training_promotion_allowed=False,
                ),
            }
        )
        dataset["dataset_sha256"] = ""
        dataset["dataset_sha256"] = _digest(dataset)
        DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        protocol = _read(PROTOCOL)
        protocol.update(
            {
                "protocol_id": report["protocol_id"],
                "schema_version": "pg267-payload-grounding-capacity-training-protocol-v1",
                "training_sources": [str(PG265_BASE_DATASET.relative_to(ROOT)), str(PG267_DATASET.relative_to(ROOT))],
                "capacity_variants": list(PG260.CAPACITY_VARIANTS),
                "pg267_even_seed_holdout": True,
                "oracle_target_off_input": True,
                "raw_payload_and_response_excluded": True,
                "promotion_blocked": True,
            }
        )
        protocol["protocol_sha256"] = ""
        protocol["protocol_sha256"] = _digest(protocol)
        PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        MARKDOWN.write_text(
            "\n".join(
                [
                    "# PG-267 payload-grounding capacity training",
                    "",
                    f"records={len(rows)}; pg267={len(fresh_rows)}; fresh_holdout={len(fresh_holdout)}",
                    f"capacities={','.join(str(x) for x in PG260.CAPACITY_VARIANTS)}; selected_hidden={report.get('selected', {}).get('hidden_dim')}",
                    f"judge={(report.get('independent_final_judge') or {}).get('decision')}; promotion_blocked=true",
                    "PG-266 的精确 payload 只保留在人工审核 catalog；模型只接收抽象 Rule-IR token、哈希和受限结果标签。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "combined_records": len(rows),
                    "pg267_records": len(fresh_rows),
                    "fresh_holdout": len(fresh_holdout),
                    "selected": report.get("selected"),
                    "judge": report.get("independent_final_judge"),
                    "report": str(REPORT.relative_to(ROOT)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return code
    finally:
        _clear_marker()


if __name__ == "__main__":
    raise SystemExit(main())
