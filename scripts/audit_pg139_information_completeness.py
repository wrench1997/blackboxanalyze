"""Audit PG-139 for missing information before any learning promotion.

The script reads fresh bounded traces through the existing local collector,
but writes only field counts, collision counts, and SHA-256 manifests.  Raw
requests, responses, probe strings, and evaluator actions are never emitted.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.information_completeness import finalize_audit, sha256_json


RESEARCH = ROOT / "research"
OUTPUT = RESEARCH / "pg139_information_completeness_audit_v1.json"


def _load_pg139() -> Any:
    path = ROOT / "scripts" / "run_pg139_value_head_loio.py"
    spec = importlib.util.spec_from_file_location("pg139_runner_for_information_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-139 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    runner = _load_pg139()
    targets = asyncio.run(runner._collect())
    folds = runner.PG138._build_folds(targets)
    token_examples: dict[str, list[dict[str, Any]]] = {}
    for fold_name, fold in folds.items():
        train = runner._examples(fold["train"])
        dev = runner._examples(fold["dev"])
        # The audit view joins bounded labels only for collision statistics;
        # it never serialises the rows themselves.
        token_examples[f"{fold_name}:train"] = [
            {"tokens": item["tokens"], "label": row.get("label"), "surface_kind": row.get("surface_kind"), "failure_signature": row.get("failure_signature")}
            for item, row in zip(train, fold["train"])
        ]
        token_examples[f"{fold_name}:dev"] = [
            {"tokens": item["tokens"], "label": row.get("label"), "surface_kind": row.get("surface_kind"), "failure_signature": row.get("failure_signature")}
            for item, row in zip(dev, fold["dev"])
        ]

    dataset = _load_json("research/pg139_value_head_loio_dataset_v1.json")
    visible = _load_json("research/pg139_value_head_loio_visible_dataset_v1.json")
    report = _load_json("research/pg139_value_head_loio_report_v1.json")
    trace = _load_json("research/pg139_value_head_loio_trace_v1.json")
    audit = finalize_audit(
        __import__("app.information_completeness", fromlist=["build_audit"]).build_audit(
            targets=targets,
            dataset=dataset,
            visible=visible,
            report=report,
            trace=trace,
            token_examples=token_examples,
            source_hashes={
                "dataset": _file_hash("research/pg139_value_head_loio_dataset_v1.json"),
                "visible_dataset": _file_hash("research/pg139_value_head_loio_visible_dataset_v1.json"),
                "report": _file_hash("research/pg139_value_head_loio_report_v1.json"),
                "trace": _file_hash("research/pg139_value_head_loio_trace_v1.json"),
                "runner": _file_hash("scripts/run_pg139_value_head_loio.py"),
            },
        )
    )
    _write_json(OUTPUT, audit)
    print(json.dumps({
        "status": audit["status"],
        "hard_gate_passed": audit["hard_gate_passed"],
        "blocking_reasons": audit["blocking_reasons"],
        "step_count": audit["internal_trace"]["step_count"],
        "public_pretrain_rows": audit["public_dataset"]["pretrain_row_count"],
        "public_action_rows": audit["public_dataset"]["action_row_count"],
        "report": str(OUTPUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

