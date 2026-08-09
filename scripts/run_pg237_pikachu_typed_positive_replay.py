"""PG-237: fresh Pikachu typed-positive/negative replay for non-trivial holdout.

This runner reuses the audited PG-217 AI-in-the-send-path loop, but uses new
seeds and output artifacts.  It keeps the same bounded local evaluator:
fresh container, database health gate, independent reference, matched
negative, source hash and typed result/effect projection.  Runtime probe
values and response bodies remain memory-only.
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


def _load_pg217() -> Any:
    path = ROOT / "scripts" / "run_pg217_pikachu_typed_sql_oracle.py"
    spec = importlib.util.spec_from_file_location("pg217_for_pg237", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-217 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG217 = _load_pg217()
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg237_pikachu_typed_positive_replay_report_v1.json"
PROTOCOL = RESEARCH / "pg237_pikachu_typed_positive_replay_protocol_v1.json"
TRACE = RESEARCH / "pg237_pikachu_typed_positive_replay_trace_v1.json"
MARKDOWN = RESEARCH / "pg237_pikachu_typed_positive_replay_report_v1.md"


def _rewrite(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for source, target in replacements.items():
            result = result.replace(source, target)
        return result
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    # The imported runner's episode logic is unchanged; only the experiment
    # identity, seeds, ports and artifact paths are changed.
    PG217.SEEDS = (23701, 23702)
    PG217.PG214.BASE_PORT = 8790
    PG217.REPORT_PATH = REPORT
    PG217.PROTOCOL_PATH = PROTOCOL
    PG217.TRACE_PATH = TRACE
    PG217.MARKDOWN_PATH = MARKDOWN
    result = PG217.main()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    replacements = {
        "pg-pk-217-pikachu-typed-sql-oracle-v1": "pg-pk-237-pikachu-typed-positive-replay-v1",
        "pg217-pikachu-typed-sql-oracle": "pg237-pikachu-typed-positive-replay",
    }
    report = _rewrite(report, replacements)
    protocol = _rewrite(protocol, replacements)
    trace = _rewrite(trace, replacements)
    report["schema_version"] = "pg237-pikachu-typed-positive-replay-report-v1"
    report["status"] = "completed_fresh_typed_positive_negative_replay"
    report["seeds"] = [23701, 23702]
    report["training_role"] = "heldout_positive_source_for_pg237_capacity_training"
    report["promotion"] = {
        **dict(report.get("promotion") or {}),
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    protocol["schema_version"] = "pg237-pikachu-typed-positive-replay-protocol-v1"
    protocol["fresh_seed_holdout_source"] = True
    protocol["training_promotion_allowed"] = False
    protocol["memory_promotion_allowed"] = False
    protocol["vulnerability_claim_allowed"] = False
    trace["schema_version"] = "pg237-pikachu-typed-positive-replay-trace-v1"
    trace["training_eligible"] = False
    trace["training_role"] = "heldout_positive_source_for_pg237_capacity_training"
    report.pop("report_sha256", None)
    protocol.pop("protocol_sha256", None)
    import hashlib

    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    report["report_sha256"] = digest(report)
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(PROTOCOL, protocol)
    _write(TRACE, trace)
    print(json.dumps({"status": report["status"], "seeds": report["seeds"], "counts": report["counts"], "report": str(REPORT.relative_to(ROOT)), "trace": str(TRACE.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())

