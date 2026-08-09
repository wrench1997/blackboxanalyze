#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.closure import analyze_closure  # noqa: E402
from app.search import search_rules  # noqa: E402


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="离线运行 Rule Discovery Closure Analyzer")
    parser.add_argument("--scenario", type=Path, required=True, help="场景 JSON，至少包含 fields")
    parser.add_argument("--observations", type=Path, required=True, help="观测 JSON 数组，或包含 observations 的对象")
    parser.add_argument("--output", type=Path, help="把完整报告写入文件；默认打印到 stdout")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=180)
    parser.add_argument("--history-depth", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=5000)
    parser.add_argument("--coverage-threshold", type=float, default=0.9)
    parser.add_argument(
        "--goal-mode",
        choices=["either", "observation_goal", "output_true"],
        default="either",
    )
    args = parser.parse_args()

    scenario = load_json(args.scenario)
    raw = load_json(args.observations)
    observations = raw.get("observations", []) if isinstance(raw, dict) else raw
    if not isinstance(observations, list):
        raise SystemExit("observations 文件必须是 JSON 数组或 {\"observations\": [...]} 对象")

    candidates = search_rules(
        scenario.get("fields", []),
        observations,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        history_depth=args.history_depth,
    )
    report = analyze_closure(
        scenario=scenario,
        observations=observations,
        raw_candidates=[candidate.to_dict() for candidate in candidates],
        max_cases=args.max_cases,
        history_depth=args.history_depth,
        coverage_threshold=args.coverage_threshold,
        goal_mode=args.goal_mode,
    )

    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"报告已写入：{args.output}")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
