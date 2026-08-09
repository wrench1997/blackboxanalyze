"""Run the PG-311 wide symbolic checkpoint through a fresh PG-305 loopback evaluator.

The adapter is reused unchanged; only the frozen model checkpoint and output
artifact names differ.  This keeps the live comparison honest and preserves
PG-305's non-destructive GET/POST, fresh-reset, negative-control and typed
evidence contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    path = ROOT / "scripts" / "run_pg305_live_loopback_evaluator.py"
    spec = importlib.util.spec_from_file_location("pg312_live_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-305 loopback adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCHEMA_VERSION = "pg312-wide-checkpoint-live-loopback-v1"
    module.CHECKPOINT = ROOT / "artifacts" / "pg311-wide-question" / "pg311_wide_question_anchor_moe_local_morning.pt"
    module.REPORT = ROOT / "research" / "pg312_live_wide_checkpoint_replay_report_v1.json"
    module.CATALOG = ROOT / "research" / "pg312_live_wide_checkpoint_human_catalog_v1.json"
    module.DATASET = ROOT / "research" / "pg312_live_wide_checkpoint_training_dataset_v1.json"
    module.TRACE = ROOT / "research" / "pg312_live_wide_checkpoint_trace_v1.json"
    module.PROTOCOL = ROOT / "research" / "pg312_live_wide_checkpoint_protocol_v1.json"
    module.MARKDOWN = ROOT / "research" / "pg312_live_wide_checkpoint_replay_report_v1.md"
    module.SEED = 31201
    module.BASE_PORT = 6145
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
