"""Run a read-only PG-331 loopback adapter smoke and write abstract evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_loopback_adapter import SCHEMA_VERSION, capture_loopback


DEFAULT_ORIGINS = ("http://127.0.0.1:8766/", "http://127.0.0.1:3100/")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _length_bucket(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return "empty" if number <= 0 else "short" if number <= 64 else "medium" if number <= 1024 else "long"


def run(origins: list[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for origin in origins:
        try:
            result = capture_loopback(origin, timeout=8.0)
            observation = result["observation"]
            field_status_counts = {axis: dict(Counter(fields.values())) for axis, fields in dict(result.get("field_capture_manifest") or {}).items()}
            records.append(
                {
                    "origin_digest": result["origin_digest"],
                    "target_contacted": bool(result.get("target_contacted")),
                    "transport": dict(result.get("transport") or {}),
                    "failure_class": str((observation.get("failure_feedback") or {}).get("failure_class", "unknown")),
                    "axis_presence": {"document": "observed", "navigation": "observed", "request": "observed", "response": "observed", "javascript": "observed", "failure": "observed", "belief_replay": "observed"},
                    "field_status_counts": field_status_counts,
                    "shape_counts": {"dom_elements": len((observation.get("document_structure") or {}).get("elements") or []), "links": len((observation.get("navigation") or {}).get("links") or []), "request_parameters": len((observation.get("request_transport") or {}).get("parameters") or []), "scripts": int((observation.get("javascript_surface") or {}).get("script_count", 0) or 0), "event_handlers": int((observation.get("javascript_surface") or {}).get("event_handler_count", 0) or 0)},
                    "response_shape": {"status_class": str((observation.get("response_transport") or {}).get("status_class", "unknown")), "content_type_class": str((observation.get("response_transport") or {}).get("content_type_class", "unknown")), "redirect_hop_count": int((observation.get("response_transport") or {}).get("redirect_hop_count", 0) or 0), "body_length_bucket": _length_bucket((observation.get("response_transport") or {}).get("body_length", "unknown"))},
                    "raw_body_stored": False,
                    "raw_payload_stored": False,
                }
            )
        except Exception as error:  # local target may be stopped; preserve only the error class
            records.append({"origin_digest": _digest({"origin": origin}), "target_contacted": False, "error_class": type(error).__name__, "raw_body_stored": False, "raw_payload_stored": False})
    contacted_count = sum(1 for item in records if item.get("target_contacted") is True)
    report_status = "diagnostic_only" if contacted_count else "target_unavailable"
    interpretation = (
        "真实 loopback 页面结构已采到，但没有 fresh reset/typed evaluator，因此只能作为观察诊断和 ASK 依据，不能进入训练。"
        if contacted_count
        else "本轮没有可用的 loopback 监听目标；仅保留目标不可用的错误类别，不能推断页面结构、漏洞阴性或训练价值。"
    )
    report: dict[str, Any] = {
        "schema_version": "pg331-loopback-adapter-smoke-v1",
        "status": report_status,
        "adapter_schema": SCHEMA_VERSION,
        "adapter_sha256": _sha256_file(ROOT / "app" / "pg331_loopback_adapter.py"),
        "targets": records,
        "target_contacted_count": contacted_count,
        "target_unavailable_count": len(records) - contacted_count,
        "fresh_reset_attested": False,
        "typed_evaluator_available": False,
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": interpretation,
    }
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", action="append", dest="origins")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg331_loopback_adapter_smoke_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(list(args.origins or DEFAULT_ORIGINS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
