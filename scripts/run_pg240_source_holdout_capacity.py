"""PG-240b: train a frozen adapter with the source-heldout replay quarantined.

PG-236/237 remain training material; both PG-240 seeds (the upstream source
tree) are never seen by the adapter.  This tests whether the model can carry
the error/repair policy across an application-source change while keeping the
base next-token body frozen.
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


def _load() -> Any:
    path = ROOT / "scripts" / "run_pg237_capacity_training.py"
    spec = importlib.util.spec_from_file_location("pg237_capacity_for_pg240", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-237 capacity trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG = _load()
RESEARCH = ROOT / "research"
PG240_TRACE = RESEARCH / "pg240_pikachu_source_replay_trace_v1.json"
REPORT = RESEARCH / "pg240_source_holdout_capacity_report_v1.json"
DATASET = RESEARCH / "pg240_source_holdout_capacity_dataset_v1.json"
TRACE = RESEARCH / "pg240_source_holdout_capacity_trace_v1.json"
PROTOCOL = RESEARCH / "pg240_source_holdout_capacity_protocol_v1.json"
MARKDOWN = RESEARCH / "pg240_source_holdout_capacity_report_v1.md"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rename(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, str):
        return value.replace("pg237", "pg240_source").replace("PG-237", "PG-240")
    return value


def main() -> int:
    PG.PG237_TRACE = PG240_TRACE
    PG.REPORT = REPORT
    PG.DATASET = DATASET
    PG.TRACE = TRACE
    PG.PROTOCOL = PROTOCOL
    PG.MARKDOWN = MARKDOWN
    PG.FRESH_SOURCE = "pg240_pikachu_source_replay"
    PG.FRESH_HOLDOUT_SEEDS = (24002,)
    PG.EXTRA_HOLDOUT_SOURCE = "pg236_pikachu_fixed_independent"
    PG.EXTRA_HOLDOUT_SEEDS = (23632,)
    PG.ARTIFACT_DIR = ROOT / "artifacts" / "pg240-source-holdout-capacity-v1"
    PG.EXPERIMENT_ID = "pg240-source-holdout"
    PG.main()

    report = _rename(json.loads(REPORT.read_text(encoding="utf-8")))
    dataset = _rename(json.loads(DATASET.read_text(encoding="utf-8")))
    trace = _rename(json.loads(TRACE.read_text(encoding="utf-8")))
    protocol = _rename(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    report.update(
        {
            "schema_version": "pg240-source-holdout-capacity-report-v1",
            "status": "completed_source_heldout_capacity_training",
            "holdout_source": "pg240_pikachu_source_replay",
            "holdout_seeds": [24002, 23632],
            "honesty": {
                **dict(report.get("honesty") or {}),
                "pg240_seed24002_never_in_training": True,
                "pg236_seed23632_never_in_training": True,
                "application_source_holdout_not_full_runtime_independence": True,
                "general_web_capability_not_established": True,
            },
            "promotion": {**dict(report.get("promotion") or {}), "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
        }
    )
    dataset.update(
        {
            "schema_version": "pg240-source-holdout-capacity-dataset-v1",
            "contract": {**dict(dataset.get("contract") or {}), "pg240_source_holdout_seeds_never_in_training": True, "pg240_seed24002_never_in_training": True, "pg236_seed23632_never_in_training": True, "source_holdout_contains_positive_and_abstain": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        }
    )
    protocol.update({"schema_version": "pg240-source-holdout-capacity-protocol-v1", "source_holdout_seeds": [24002, 23632], "source_holdout_required": True, "false_send_is_hard_failure": True, "next_token_loss_not_promotion_gate": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False})
    trace.update({"schema_version": "pg240-source-holdout-capacity-trace-v1", "source_holdout_seeds": [24002, 23632], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    report.pop("report_sha256", None)
    dataset.pop("dataset_sha256", None)
    protocol.pop("protocol_sha256", None)
    report["report_sha256"] = PG.digest(report)
    dataset["dataset_sha256"] = PG.digest(dataset)
    protocol["protocol_sha256"] = PG.digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    metrics = report.get("selected", {}).get("metrics", {}).get("seed_holdout", {})
    MARKDOWN.write_text(
        "\n".join(
            [
                "# PG-240 source-heldout capacity training",
                "",
                f"train={report['counts'].get('train_rows')}; holdout={report['counts'].get('holdout_rows')}; actions={report['counts'].get('holdout_action_counts')}",
                f"hidden={report['selected'].get('hidden_dim')}; next_token={metrics.get('next_token_accuracy')}; positive_send_recall={metrics.get('positive_send_recall')}; abstain_recall={metrics.get('abstain_recall')}; false_send={metrics.get('false_send_count')}; missed_send={metrics.get('missed_send_count')}",
                f"safety_gate={report.get('safety_abstain_gate_pass')}; capability_gate={report.get('capability_gate_pass')}",
                "",
                "PG-240 源码 seed 与 PG-236 seed 均未进入训练；本轮只证明受控本地源代码留出，不等于公网渗透能力。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "counts": report["counts"], "selected": report["selected"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
