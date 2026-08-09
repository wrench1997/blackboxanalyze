"""Train an offline PG-95 candidate with the invariant token funnel.

Only the existing PG-86 train/dev/source-holdout/unknown-family traces feed
this candidate.  PG-94 remains a frozen, unseen evaluation source.  The
checkpoint is written to a new artifact directory and is never promoted to
the active model or long-term memory by this script.
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

PG86_SCRIPT = ROOT / "scripts" / "train_pg86_surface_signal_composite.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg95-invariant-surface-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
DATASET_PATH = ROOT / "research" / "pg95_invariant_surface_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg95_invariant_surface_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg95_invariant_surface_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg95_invariant_surface_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg95_invariant_surface_transformer_report_v1.md"
PROTOCOL_ID = "pg-pk-95-invariant-surface-transformer-v1"

from app.invariant_token_funnel import SCHEMA_VERSION, canonicalize_tokens  # noqa: E402


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> dict[str, Any]:
    pg86 = _load(PG86_SCRIPT, "pg95_pg86_training_runtime")
    original_tokens = pg86._row_tokens

    def invariant_row_tokens(module: Any, step: dict[str, Any], candidate: dict[str, Any], oracle: dict[str, Any]) -> tuple[list[str], int]:
        tokens, oracle_index = original_tokens(module, step, candidate, oracle)
        return canonicalize_tokens(tokens), oracle_index

    pg86.DATASET_PATH = DATASET_PATH
    pg86.REPORT_PATH = REPORT_PATH
    pg86.PROTOCOL_PATH = PROTOCOL_PATH
    pg86.TRACE_PATH = TRACE_PATH
    pg86.MARKDOWN_PATH = MARKDOWN_PATH
    pg86.OUTPUT_DIR = OUTPUT_DIR
    pg86.CHECKPOINT_PATH = CHECKPOINT_PATH
    pg86.PROTOCOL_ID = PROTOCOL_ID
    pg86._row_tokens = invariant_row_tokens
    report = pg86.run()
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg95-invariant-surface-transformer-report-v1"
    report["source"]["token_funnel_schema"] = SCHEMA_VERSION
    report["source"]["token_funnel_profile"] = "absolute_presence_and_generic_effect_delta"
    report["source"]["training_data_excludes_pg94"] = True
    report["promotion"] = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "status": "offline_candidate_only",
        "reason": "PG95 is a representation candidate; PG94 remains frozen and no active checkpoint is replaced",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["schema_version"] = "pg95-invariant-surface-trace-dataset-v1"
    dataset["dataset_id"] = "pg95-invariant-surface"
    dataset["token_funnel_schema"] = SCHEMA_VERSION
    dataset["training_eligible"] = False
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg95-invariant-surface-transformer-trace-v1"
    trace["token_funnel_schema"] = SCHEMA_VERSION
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["schema_version"] = "pg95-invariant-surface-transformer-protocol-v1"
    protocol["token_funnel_schema"] = SCHEMA_VERSION
    protocol["token_funnel_profile"] = "absolute_presence_and_generic_effect_delta"
    protocol["pg94_excluded_from_training"] = True
    protocol["run_result"] = {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-95 invariant surface transformer\n\n" + f"token_funnel=`{SCHEMA_VERSION}`；device={report['source']['device']}；PG94 excluded from training。\n\n能力门：`{report['capability_gate']['status']}`；active/memory promotion=`false`。\n", encoding="utf-8")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint["schema_version"] = "pg95-invariant-surface-transformer-checkpoint-v1"
    checkpoint["token_funnel_schema"] = SCHEMA_VERSION
    checkpoint["token_funnel_profile"] = "absolute_presence_and_generic_effect_delta"
    checkpoint["pg94_excluded_from_training"] = True
    torch.save(checkpoint, CHECKPOINT_PATH)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "dev_confirm_recall": result["metrics"]["dev_holdout"]["confirm_recall"], "cross_dataset_holdout_recall": result["metrics"]["cross_dataset_holdout"]["confirm_recall"], "source_holdout_confirm_recall": result["metrics"]["source_holdout"]["confirm_recall"], "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"]["strict_abstain"], "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
