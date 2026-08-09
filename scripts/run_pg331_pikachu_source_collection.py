"""PG-331A live source-row collection on a fresh pinned Pikachu target.

This is a narrow collection probe, not a vulnerability scanner.  It sends three
neutral requests (root GET, parameterized GET with empty query fields, and POST
with empty form fields) to separate disposable ``--network none`` containers.  The host-side
relay binds only to loopback and forwards through a PHP process inside the
container.  The adapter keeps structural seven-axis projections in memory;
the output contains no response body or payload string.

The evaluator sidecar is deliberately marked unavailable in this first pass.
Consequently every row is expected to be ``incomplete/ASK`` and promotion is
closed.  A later evaluator-specific run must supply candidate/reference/
negative evidence before any row can be training-eligible.

Run during the local collection window with an explicit operator flag:

    $env:PG331_LOCAL_DOCKER_EVAL='1'
    python scripts/run_pg331_pikachu_source_collection.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_pikachu_docker_relay import DisposablePikachu, IMAGE
from app.pg331_loopback_adapter import capture_loopback
from app.pg331_source_row import collect_pg331_source_row, sha256_json
from app.pg331_trajectory import audit_pg331_trajectory


SCHEMA_VERSION = "pg331-pikachu-source-collection-v1"
REPORT_PATH = ROOT / "research" / "pg331_pikachu_source_collection_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg331_pikachu_source_row_collection_v1.json"
SEED = 33101

ROUTES: tuple[dict[str, Any], ...] = (
    {"id": "root-get", "method": "GET", "path": "/", "fields": []},
    {"id": "sql-search-get", "method": "GET", "path": "/vul/sqli/sqli_search.php", "fields": ["name", "submit"]},
    {"id": "sql-id-post", "method": "POST", "path": "/vul/sqli/sqli_id.php", "fields": ["id", "submit"]},
)


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_window() -> None:
    if os.environ.get("PG331_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-331 live collection requires explicit PG331_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-331 local collection is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")


def _route_url(origin: str, route: Mapping[str, Any]) -> str:
    path = str(route["path"])
    fields = list(route.get("fields") or [])
    if str(route["method"]).upper() == "GET" and fields:
        # Empty values preserve parameter names and placement without sending
        # an attack string or a state-changing value.
        from urllib.parse import urlencode

        query = urlencode({str(field): "" for field in fields})
        return f"{origin}{path}?{query}"
    return f"{origin}{path}"


def _source_meta(route: Mapping[str, Any]) -> dict[str, Any]:
    route_manifest_digest = _sha256({"route_id": route["id"], "method": route["method"], "path_shape": route["path"], "fields": route.get("fields") or []})
    return {
        "source_id": "pg331-pikachu-fixed-neutral",
        "implementation": "pikachu-fixed",
        "collector_id": "pg331-live-source-collector-v1",
        "authorization_id": "operator-authorized-local-docker-loopback",
        "image_digest": IMAGE.split("@sha256:", 1)[1],
        "source_digest": route_manifest_digest,
    }


def _evaluator_sidecar(route: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    # A neutral baseline has no typed candidate/reference/negative oracle.  A
    # deterministic evidence digest documents what was observed without
    # pretending it is a vulnerability verdict.
    return {
        "typed_available": False,
        "negative_control": False,
        "reference_present": False,
        "candidate_present": False,
        "fresh_reset": True,
        "evidence_hash": _sha256({"route_id": route["id"], "observation": observation}),
        "confirmed_positive": False,
        "effect_class": "baseline_structure_only",
        "evaluator_version": "pg331-no-typed-oracle-baseline-v1",
    }


def _target_projection() -> dict[str, Any]:
    return {
        "question": "ask_typed_oracle",
        "next_action": "ask_typed",
        "repair_action": "none",
        "transport_ref": "request_method",
        "field_role_ref": "parameter_role",
        "encoding_ref": "encoding_chain",
        "probe_variant_ref": "none",
        "safe_to_send": False,
    }


def _capture_route(route: Mapping[str, Any], index: int) -> dict[str, Any]:
    name = f"sift-pg331-neutral-{SEED}-{index}"
    target = DisposablePikachu(name, seed=SEED, index=index)
    try:
        reset = target.start()
        form_data = {str(field): "" for field in list(route.get("fields") or [])} if str(route["method"]).upper() == "POST" else None
        capture = capture_loopback(_route_url(target.origin, route), method=str(route["method"]).upper(), form_data=form_data, timeout=15.0)
        observation = dict(capture["observation"])
        evaluator = _evaluator_sidecar(route, observation)
        row = collect_pg331_source_row(
            record_id=f"pg331:{route['id']}:{index}",
            observation=observation,
            source_meta=_source_meta(route),
            reset=reset,
            evaluator=evaluator,
            field_capture_manifest=capture["field_capture_manifest"],
            target_projection=_target_projection(),
            split="unassigned",
            operator_reviewed=False,
            hard_negative=False,
        )
        return {"route_id": str(route["id"]), "method": str(route["method"]).upper(), "target_contacted": bool(capture.get("target_contacted")), "row": row}
    finally:
        target.stop()


def run() -> dict[str, Any]:
    _require_window()
    started = time.monotonic()
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, route in enumerate(ROUTES):
        try:
            episodes.append(_capture_route(route, index))
        except Exception as error:
            errors.append({"route_id": str(route["id"]), "error_class": type(error).__name__})
    rows = [dict(item["row"]) for item in episodes if isinstance(item.get("row"), Mapping)]
    trajectory_steps = [{"step_index": index, "action_role": "baseline_observe", "row": row} for index, row in enumerate(rows)]
    trajectory = audit_pg331_trajectory(trajectory_steps, require_get_post=True) if rows else {"valid": False, "failures": ["no_rows"]}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_diagnostic_only" if rows and not errors else "incomplete",
        "runtime": {"image": IMAGE, "network": "none", "loopback_relay_only": True, "external_network": False, "route_count": len(ROUTES), "elapsed_seconds": round(time.monotonic() - started, 3)},
        "counts": {"route_count": len(ROUTES), "rows": len(rows), "target_contacted": sum(int(bool(item.get("target_contacted"))) for item in episodes), "errors": len(errors), "training_eligible": sum(int(bool(row.get("training_eligible"))) for row in rows), "ask_rows": sum(int(str((row.get("target_projection") or {}).get("next_action")) == "ask_typed") for row in rows)},
        "errors": errors,
        "trajectory_audit": trajectory,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "真实 GET/POST 结构只作为诊断和 ASK 证据；本轮 typed evaluator 缺失，不能训练、不能声明漏洞、不能生成 payload catalog。",
    }
    report["report_sha256"] = sha256_json(report)
    dataset: dict[str, Any] = {
        "schema_version": "pg331-source-row-collection-v1",
        "collector": "scripts/run_pg331_pikachu_source_collection.py",
        "records": rows,
        "counts": {"input": len(ROUTES), "accepted": len(rows), "incomplete": sum(int(not bool(row.get("training_eligible"))) for row in rows), "rejected": len(errors), "training_eligible": 0},
        "source": {"image": IMAGE, "network": "none", "loopback_only": True, "external_network": False},
        "promotion": report["promotion"],
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    _write(REPORT_PATH, report)
    _write(DATASET_PATH, dataset)
    return {"report": report, "dataset": dataset}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    print(json.dumps(result if args.json else {"status": result["report"]["status"], "counts": result["report"]["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
