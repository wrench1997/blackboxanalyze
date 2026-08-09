"""Run the frozen PG-318 evaluator with PG-321 role-conditioned checkpoints.

The evaluator owns the local Docker, DOM-oracle, fresh-reset, and evidence
gates.  This wrapper only stages the research checkpoints and gives the run a
new immutable artifact namespace; it never changes the target adapter.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load() -> Any:
    path = ROOT / "scripts" / "run_pg318_family_holdout_replay.py"
    spec = importlib.util.spec_from_file_location("pg318_evaluator_for_pg321", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-318 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _replace(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("PG-318", "PG-321").replace("pg318", "pg321")
    if isinstance(value, list):
        return [_replace(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item) for key, item in value.items()}
    return value


def main() -> int:
    if os.environ.get("PG321_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-321 live replay requires PG321_LOCAL_DOCKER_EVAL=1")
    os.environ["PG318_LOCAL_DOCKER_EVAL"] = "1"
    evaluator = _load()
    source_dir = ROOT / "artifacts" / "pg321-variant-role" / "seeds"
    staging = ROOT / "artifacts" / "pg321-live-checkpoints"
    staging.mkdir(parents=True, exist_ok=True)
    for seed in (31901, 31902, 31903):
        source = source_dir / f"pg321_variant_role_seed_{seed}.pt"
        if not source.exists():
            raise RuntimeError(f"missing PG-321 seed checkpoint: {source}")
        target = staging / f"pg317_question_anchor_moe_seed_{seed}.pt"
        shutil.copy2(source, target)
    evaluator.CHECKPOINT_DIR = staging
    evaluator.SEEDS = (31901, 31902, 31903)
    evaluator.REPORT = ROOT / "research" / "pg321_family_holdout_replay_report_v1.json"
    evaluator.CATALOG = ROOT / "research" / "pg321_family_holdout_human_catalog_v1.json"
    evaluator.TRACE = ROOT / "research" / "pg321_family_holdout_trace_v1.json"
    evaluator.PROTOCOL = ROOT / "research" / "pg321_family_holdout_protocol_v1.json"
    evaluator.main()
    for path, key in ((evaluator.REPORT, "report_sha256"), (evaluator.CATALOG, "catalog_sha256"), (evaluator.TRACE, "trace_sha256"), (evaluator.PROTOCOL, "protocol_sha256")):
        data = _replace(json.loads(path.read_text(encoding="utf-8-sig")))
        data[key] = ""
        data[key] = _digest(data)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = json.loads(evaluator.REPORT.read_text(encoding="utf-8-sig"))
    print(json.dumps({"status": report.get("status"), "counts": report.get("counts"), "worst_seed_metrics": report.get("worst_seed_metrics"), "gate": report.get("hypothesis_gate"), "report": str(evaluator.REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
