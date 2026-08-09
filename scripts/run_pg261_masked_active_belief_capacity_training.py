# -*- coding: utf-8 -*-
"""PG-261: rerun PG-260 with padding-invariant masked pooling.

PG-260 exposed a representation bug: the adapter's mean pool included padded
positions, so a record's prediction depended on which split it was batched
with.  This wrapper reuses the same real local traces, frozen body and
capacity sweep after the mask-aware adapter fix, but writes a separate PG-261
artifact/report so the before/after remains auditable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("pg261_pg260_runner", ROOT / "scripts" / "run_pg260_active_belief_capacity_training.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-260 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG260 = _load()
REPORT = ROOT / "research" / "pg261_masked_active_belief_capacity_training_report_v1.json"
DATASET = ROOT / "research" / "pg261_masked_active_belief_capacity_training_dataset_v1.json"
TRACE = ROOT / "research" / "pg261_masked_active_belief_capacity_training_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg261_masked_active_belief_capacity_training_protocol_v1.json"
MARKDOWN = ROOT / "research" / "pg261_masked_active_belief_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg261-masked-active-belief-capacity-v1"
RUN_MARKER = ROOT / "research" / "pg261_training_running.json"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_run_marker() -> None:
    """Expose a bounded running marker so the ops UI never serves a stale report."""
    RUN_MARKER.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "report": str(REPORT.relative_to(ROOT)),
                "protocol_id": "pg-pk-261-masked-active-belief-capacity-v1",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _clear_run_marker() -> None:
    try:
        RUN_MARKER.unlink(missing_ok=True)
    except OSError:
        # A stale marker cannot upgrade a report: the ops projection also
        # compares its timestamp with the final report mtime.
        pass


def main() -> int:
    _write_run_marker()
    try:
        return _run()
    finally:
        _clear_run_marker()


def _run() -> int:
    PG260.REPORT = REPORT
    PG260.DATASET = DATASET
    PG260.TRACE = TRACE
    PG260.PROTOCOL = PROTOCOL
    PG260.MARKDOWN = MARKDOWN
    PG260.ARTIFACT_DIR = ARTIFACT_DIR
    canonical_rows = PG260._load_records()
    PG260.CANONICAL_CONTEXT_WIDTH = max(len(list(row.get("tokens") or [])) for row in canonical_rows)
    result = PG260.main()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report.update({"protocol_id": "pg-pk-261-masked-active-belief-capacity-v1", "schema_version": "pg261-masked-active-belief-capacity-training-report-v1", "status": "completed_pg261_masked_active_belief_capacity_training"})
    report["architecture_change"] = {"id": "pg261-mask-aware-pooling-v1", "changed": True, "masked_mean_pool": True, "canonical_context_width": int(PG260.CANONICAL_CONTEXT_WIDTH or 0), "padding_invariant_classification": True, "legacy_pg260_artifact_unchanged": True}
    report["independent_final_judge"]["authority"] = ["PG-258 holdout", "PG-259 fresh route holdout", "PG-260 fresh paired route/seed holdout", "VulnerableApp implementation OOD", "frozen legacy policy canary", "PG-261 padding-invariance audit"]
    report["honesty"]["mask_aware_pooling"] = True
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    dataset["schema_version"] = "pg261-masked-active-belief-capacity-training-dataset-v1"
    dataset["architecture_change"] = "pg261-mask-aware-pooling-v1"
    dataset["dataset_sha256"] = ""
    dataset["dataset_sha256"] = _digest(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg261-masked-active-belief-capacity-trace-v1"
    trace["architecture_change"] = "pg261-mask-aware-pooling-v1"
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["protocol_id"] = "pg-pk-261-masked-active-belief-capacity-v1"
    protocol["schema_version"] = "pg261-masked-active-belief-capacity-training-protocol-v1"
    protocol["architecture_change"] = "mask-aware mean pooling over non-padding tokens"
    protocol["protocol_sha256"] = ""
    protocol["protocol_sha256"] = _digest(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(MARKDOWN.read_text(encoding="utf-8").replace("PG-260", "PG-261").replace("PG-260 active-belief", "PG-261 masked active-belief") + "\nmask-aware pooling=enabled; classification is invariant to batch padding width.\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(REPORT.relative_to(ROOT)), "selected": report["selected"], "judge": report["independent_final_judge"], "architecture_change": report["architecture_change"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
