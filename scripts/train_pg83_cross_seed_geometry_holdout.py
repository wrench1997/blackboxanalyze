"""PG-83: independent seed holdout on the PG-82 geometry projection.

The model sees two seeds from the two training implementations.  The third
seed is not used for weight updates or vocabulary construction, and the two
PG-36 implementations remain a separate source holdout.  This closes the
cross-seed leakage left open by PG-82 while retaining the same OOD and strict
unknown-family rules.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG82_SCRIPT = ROOT / "scripts" / "train_pg82_effect_geometry_source_holdout.py"
PG82_TRACE = ROOT / "research" / "pg82_canonical_triplet_collector_trace_v1.json"
PG76_TRACE = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg83_cross_seed_geometry_holdout_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg83_cross_seed_geometry_holdout_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg83_cross_seed_geometry_holdout_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg83_cross_seed_geometry_holdout_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg83_cross_seed_geometry_holdout_transformer_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg83-cross-seed-geometry-holdout-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
PROTOCOL_ID = "pg-pk-83-cross-seed-geometry-holdout-v1"
TRAIN_SEEDS = {7901, 7907}
DEV_SEEDS = {7911}
TRAIN_SOURCES = {("pg34", "base"), ("pg35", "alpha")}
HOLDOUT_SOURCES = {("pg36", "north"), ("pg36", "south")}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_rows_hook(original: Any) -> Any:
    def filtered(pg77: Any, trace: dict[str, Any], pg53: Any, *, split: str, allowed_sources: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        if split == "train":
            source_set = TRAIN_SOURCES
            seed_set = TRAIN_SEEDS
        elif split == "dev":
            source_set = TRAIN_SOURCES
            seed_set = DEV_SEEDS
        else:
            source_set = HOLDOUT_SOURCES
            seed_set = None
        rows = original(pg77, trace, pg53, split=split, allowed_sources=source_set)
        if seed_set is not None:
            rows = [row for row in rows if int(row.get("sampling_seed", -1)) in seed_set]
        return rows

    return filtered


def _rewrite(report: dict[str, Any]) -> dict[str, Any]:
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg83-cross-seed-geometry-holdout-transformer-report-v1"
    report["source"]["train_trace"] = str(PG82_TRACE.relative_to(ROOT))
    report["source"]["projection_schema"] = "canonical_effect_projection_v2"
    report["source"]["seed_holdout"] = {"train": sorted(TRAIN_SEEDS), "dev": sorted(DEV_SEEDS), "holdout_sources": sorted("/".join(item) for item in HOLDOUT_SOURCES)}
    report["promotion"]["reason"] = "PG83 cross-seed and source holdout must pass before any promotion"
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["schema_version"] = "pg83-cross-seed-geometry-holdout-trace-dataset-v1"
    dataset["dataset_id"] = "pg83-cross-seed-geometry-holdout-triplets"
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg83-cross-seed-geometry-holdout-transformer-trace-v1"
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["schema_version"] = "pg83-cross-seed-geometry-holdout-transformer-protocol-v1"
    protocol["seed_contract"] = {"train_seeds": sorted(TRAIN_SEEDS), "dev_seeds": sorted(DEV_SEEDS), "holdout_sources": sorted("/".join(item) for item in HOLDOUT_SOURCES), "seed_in_family_source_tokens": False}
    protocol["run_result"] = {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-83 Cross-seed geometry holdout\n\n" + f"train/dev/source-holdout/unknown={report['dataset']['train']}/{report['dataset']['dev']}/{report['dataset']['source_holdout']}/{report['dataset']['unknown_family_holdout']}；train seeds={sorted(TRAIN_SEEDS)}；dev seed={sorted(DEV_SEEDS)}；device={report['source']['device']}。\n\ndev recall={report['metrics']['dev_holdout']['confirm_recall']}；source holdout recall={report['metrics']['source_holdout']['confirm_recall']}；unknown strict abstain={report['metrics']['unknown_family_holdout']['strict_abstain']}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint["schema_version"] = "pg83-cross-seed-geometry-holdout-transformer-checkpoint-v1"
    checkpoint["projection_schema"] = "canonical_effect_projection_v2"
    checkpoint["train_seeds"] = sorted(TRAIN_SEEDS)
    checkpoint["dev_seeds"] = sorted(DEV_SEEDS)
    torch.save(checkpoint, CHECKPOINT_PATH)
    return report


def run() -> dict[str, Any]:
    pg82 = _load(PG82_SCRIPT, "pg83_pg82_runtime")
    pg82.PG82_TRACE = PG82_TRACE
    pg82.PG76_TRACE = PG76_TRACE
    pg82.DATASET_PATH = DATASET_PATH
    pg82.REPORT_PATH = REPORT_PATH
    pg82.PROTOCOL_PATH = PROTOCOL_PATH
    pg82.TRACE_PATH = TRACE_PATH
    pg82.MARKDOWN_PATH = MARKDOWN_PATH
    pg82.OUTPUT_DIR = OUTPUT_DIR
    pg82.CHECKPOINT_PATH = CHECKPOINT_PATH
    pg82.PROTOCOL_ID = PROTOCOL_ID
    pg82.STEP_ROWS_HOOK = _step_rows_hook
    report = pg82.run()
    return _rewrite(report)


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "train_count": result["dataset"]["train"], "dev_confirm_recall": result["metrics"]["dev_holdout"]["confirm_recall"], "source_holdout_confirm_recall": result["metrics"]["source_holdout"]["confirm_recall"], "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"]["strict_abstain"], "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
