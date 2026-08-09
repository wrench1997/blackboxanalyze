"""PG-238: fresh Pikachu DOM/redirect family replay for OOD abstention."""

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


def _load_pg227() -> Any:
    path = ROOT / "scripts" / "run_pg227_ai_dom_redirect_validation.py"
    spec = importlib.util.spec_from_file_location("pg227_for_pg238", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-227 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG227 = _load_pg227()
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg238_pikachu_surface_replay_report_v1.json"
DATASET = RESEARCH / "pg238_pikachu_surface_replay_dataset_v1.json"
TRACE = RESEARCH / "pg238_pikachu_surface_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg238_pikachu_surface_replay_protocol_v1.json"
MARKDOWN = RESEARCH / "pg238_pikachu_surface_replay_report_v1.md"


def _rename(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, str):
        return value.replace("pg-pk-227-ai-dom-redirect-validation-v1", "pg-pk-238-pikachu-surface-replay-v1").replace("pg227-ai-dom-redirect-validation", "pg238-pikachu-surface-replay")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    PG227.SEEDS = (23801, 23802)
    PG227.PG214.BASE_PORT = 9000
    PG227.REPORT = REPORT
    PG227.DATASET = DATASET
    PG227.TRACE = TRACE
    PG227.PROTOCOL = PROTOCOL
    PG227.MARKDOWN = MARKDOWN
    result = PG227.main()
    report = _rename(json.loads(REPORT.read_text(encoding="utf-8")))
    dataset = _rename(json.loads(DATASET.read_text(encoding="utf-8")))
    trace = _rename(json.loads(TRACE.read_text(encoding="utf-8")))
    protocol = _rename(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    report["schema_version"] = "pg238-pikachu-surface-replay-report-v1"
    report["status"] = "completed_fresh_unseen_surface_replay"
    report["seeds"] = [23801, 23802]
    report["training_role"] = "family_holdout_and_oracle_ablation_source"
    report["promotion"] = {**dict(report.get("promotion") or {}), "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
    dataset["schema_version"] = "pg238-pikachu-surface-replay-dataset-v1"
    dataset["training_contract"] = {**dict(dataset.get("training_contract") or {}), "training_eligible": False, "family_holdout_only": True, "typed_dom_effect_is_not_xss": True, "normal_redirect_is_not_open_redirect": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}
    trace["schema_version"] = "pg238-pikachu-surface-replay-trace-v1"
    trace["training_eligible"] = False
    protocol["schema_version"] = "pg238-pikachu-surface-replay-protocol-v1"
    protocol["family_holdout_only"] = True
    protocol["training_promotion_allowed"] = False
    protocol["memory_promotion_allowed"] = False
    protocol["vulnerability_claim_allowed"] = False
    report.pop("report_sha256", None)
    dataset.pop("dataset_sha256", None)
    protocol.pop("protocol_sha256", None)
    report["report_sha256"] = _digest(report)
    dataset["dataset_sha256"] = _digest(dataset)
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "seeds": report["seeds"], "counts": report["counts"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())

