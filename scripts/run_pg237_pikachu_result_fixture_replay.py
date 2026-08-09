"""PG-237: fresh result-fixture replay with a non-trivial typed positive lane."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg218() -> Any:
    path = ROOT / "scripts" / "run_pg218_pikachu_result_fixture.py"
    spec = importlib.util.spec_from_file_location("pg218_for_pg237", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-218 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG218 = _load_pg218()
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg237_pikachu_result_fixture_replay_report_v1.json"
PROTOCOL = RESEARCH / "pg237_pikachu_result_fixture_replay_protocol_v1.json"
TRACE = RESEARCH / "pg237_pikachu_result_fixture_replay_trace_v1.json"
MARKDOWN = RESEARCH / "pg237_pikachu_result_fixture_replay_report_v1.md"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rename(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, str):
        return value.replace("pg-pk-218-pikachu-result-fixture-v1", "pg-pk-237-pikachu-result-fixture-replay-v1").replace("pg218-pikachu-result-fixture", "pg237-pikachu-result-fixture-replay")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> int:
    PG218.SEEDS = (23701, 23702)
    PG218.PG214.BASE_PORT = 8810
    PG218.REPORT_PATH = REPORT
    PG218.PROTOCOL_PATH = PROTOCOL
    PG218.TRACE_PATH = TRACE
    PG218.MARKDOWN_PATH = MARKDOWN
    result = PG218.main()
    report = _rename(json.loads(REPORT.read_text(encoding="utf-8")))
    protocol = _rename(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    trace = _rename(json.loads(TRACE.read_text(encoding="utf-8")))
    report["schema_version"] = "pg237-pikachu-result-fixture-replay-report-v1"
    report["status"] = "completed_fresh_typed_positive_negative_result_fixture_replay"
    report["seeds"] = [23701, 23702]
    report["training_role"] = "positive_and_negative_source_for_pg237_capacity_training"
    report["promotion"] = {**dict(report.get("promotion") or {}), "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
    protocol["schema_version"] = "pg237-pikachu-result-fixture-replay-protocol-v1"
    protocol["fresh_seed_holdout_source"] = True
    protocol["training_promotion_allowed"] = False
    protocol["memory_promotion_allowed"] = False
    protocol["vulnerability_claim_allowed"] = False
    trace["schema_version"] = "pg237-pikachu-result-fixture-replay-trace-v1"
    trace["training_eligible"] = False
    trace["training_role"] = "positive_and_negative_source_for_pg237_capacity_training"
    report.pop("report_sha256", None)
    protocol.pop("protocol_sha256", None)
    report["report_sha256"] = _digest(report)
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(PROTOCOL, protocol)
    _write(TRACE, trace)
    print(json.dumps({"status": report["status"], "seeds": report["seeds"], "counts": report["counts"], "report": str(REPORT.relative_to(ROOT)), "trace": str(TRACE.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())

