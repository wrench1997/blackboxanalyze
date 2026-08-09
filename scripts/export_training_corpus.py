from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rule_ir import pretty, truthy_result  # noqa: E402
from app.scenarios import SCENARIOS  # noqa: E402
from app.search import enumerate_envelopes  # noqa: E402


SPLITS = {
    "train": ["js_truthy_access", "parity_color", "access_gate", "text_shape"],
    "validation": ["js_boundary_coupon", "sequence_lock"],
    "test": ["js_substring_redirect", "js_sequence_replay"],
}


def trace_rows(scenario: dict[str, Any], max_cases: int) -> list[dict[str, Any]]:
    rows = []
    validation_cases = scenario.get("validation_cases")
    if validation_cases:
        cases = [
            {
                "envelope": {
                    "input": item.get("input", {}),
                    "context": item.get("context", {}),
                    "state": item.get("state", {}),
                },
                "history": item.get("history", []),
            }
            for item in validation_cases[:max_cases]
        ]
    else:
        cases = [
            {"envelope": envelope, "history": []}
            for envelope in enumerate_envelopes(scenario["fields"], max_cases=max_cases)
        ]
    for case in cases:
        rows.append({
            **case,
            "output": truthy_result(scenario["hidden_rule"], case["envelope"], case["history"]),
        })
    return rows


def build_record(scenario: dict[str, Any], split: str, max_cases: int) -> dict[str, Any]:
    traces = trace_rows(scenario, max_cases)
    return {
        "schema_version": "sift-corpus-v1",
        "task_id": scenario["id"],
        "split": split,
        "family": scenario.get("category", scenario["id"]),
        "modalities": {
            "js": scenario.get("js_source"),
            "game": scenario.get("game_rule", scenario.get("description")),
            "trace": traces,
            "rule_ir": scenario["hidden_rule"],
            "evidence": {
                "intended_rule": scenario.get("intended_rule"),
                "cwe": scenario.get("cwe"),
                "severity": scenario.get("severity"),
                "research_question": scenario.get("research_question"),
            },
        },
        "targets": {
            "rule_pretty": pretty(scenario["hidden_rule"]),
            "query_budget": min(18, len(traces)),
            "requires_executable_verification": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export family-separated SIFT training corpus JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sift-corpus"))
    parser.add_argument("--max-cases", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"schema_version": "sift-corpus-v1", "splits": {}}
    for split, scenario_ids in SPLITS.items():
        output = args.output_dir / f"{split}.jsonl"
        records = [build_record(SCENARIOS[scenario_id], split, args.max_cases) for scenario_id in scenario_ids]
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
        manifest["splits"][split] = {"file": output.name, "families": scenario_ids, "records": len(records)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
