"""PG-93: replay PG-34 after adding bounded semantic surface signals.

PG-92 intentionally preserved the failure where coarse JSON shape was equal
for control/candidate.  PG-93 consumes the regenerated PG-34 projection that
contains only the label-free ``surface_observation`` and type geometry counts;
it does not alter the frozen checkpoint or use typed oracle labels as input.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG92_SCRIPT = ROOT / "scripts" / "run_pg92_blind_pg34_frozen_replay.py"
PROTOCOL_ID = "pg-pk-93-effect-surface-pg34-replay-v1"
REPORT_PATH = ROOT / "research" / "pg93_effect_surface_pg34_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg93_effect_surface_pg34_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg93_effect_surface_pg34_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg93_effect_surface_pg34_replay_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> dict[str, Any]:
    base = _load(PG92_SCRIPT, "pg93_pg92_runtime")
    base.PROTOCOL_ID = PROTOCOL_ID
    base.REPORT_PATH = REPORT_PATH
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.TRACE_PATH = TRACE_PATH
    base.MARKDOWN_PATH = MARKDOWN_PATH
    report = base.run()
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg93-effect-surface-pg34-replay-report-v1"
    report["source"]["adapter_profile_reused"] = "canonical_effect_projection_v4_pg35_shape_delta_with_pg34_surface_observation"
    report["source"]["semantic_surface_signal"] = "bounded_true_numeric_type_counts_and_key_hash_buckets"
    report["promotion"] = {"training_allowed": False, "memory_promotion_allowed": False, "status": "frozen_effect_surface_replay_only", "reason": "PG93 evaluates a projection repair; it is not a checkpoint or memory promotion authority"}
    report["artifacts"] = {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8")); trace["protocol_id"] = PROTOCOL_ID; trace["schema_version"] = "pg93-effect-surface-pg34-replay-trace-v1"; TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")); protocol["protocol_id"] = PROTOCOL_ID; protocol["schema_version"] = "pg93-effect-surface-pg34-replay-protocol-v1"; protocol["projection_repair"] = "PG34 bounded surface_observation + generic_effect_geometry"; protocol["run_result"] = {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}; PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-93 Effect surface PG-34 replay\n\n" + f"recall={report['metrics']['confirm_recall']}；false_accept={report['metrics']['false_accept_count']}；unknown_tokens={report['metrics']['unknown_token_count']}；projection=`bounded_surface_observation+generic_geometry`。\n\n硬门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "device": result["metrics"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
