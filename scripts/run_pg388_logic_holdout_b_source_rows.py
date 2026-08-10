"""Materialize a bounded in-process PG-388 implementation-B holdout.

The runner reuses the audited PG-331 row adapter and Rule-IR collector, but
calls implementation B's WSGI function directly.  It performs no socket,
Docker, GPU, or external-network action.  The output is candidate-only and
must not be treated as live image evidence or training authorization.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

# Keep direct ``python scripts/...`` execution rooted at the repository.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fixtures.pg388.logic_lab_b import application
from scripts.run_pg388_logic_rule_ir_source_rows_live import _digest, run, write_artifacts


DEFAULT_BASE_URL = "http://127.0.0.1:8088"
DEFAULT_REPORT = "research/pg388_logic_holdout_b_source_rows_v1.json"
DEFAULT_ROWS = "research/pg388_logic_holdout_b_source_rows_rows_v1.json"
DEFAULT_SIDECARS = "research/pg388_logic_holdout_b_source_rows_sidecars_v1.json"

# A bounded page shell is used only as an in-memory input to the existing
# whole-page adapter.  It contains no user values, route literals, payloads,
# response bodies, or external resources.
HOLDOUT_PAGE_HTML = "<!doctype html><html lang='en'><head><title>logic holdout</title><script>const stateShape = 'abstract';</script></head><body><main data-lab='logic-holdout'><form method='post' action='/abstract'><input name='case_ref' type='hidden' value='enum'></form></main></body></html>"


def _call_wsgi(_base_url: str, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    body = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")).encode("utf-8") if method == "POST" else b""
    statuses: list[str] = []

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        statuses.append(status)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
        "PG388_NETWORK_MODE": "none",
    }
    document = b"".join(application(environ, start_response))
    if not statuses or int(statuses[0].split()[0]) >= 400:
        raise RuntimeError("logic_holdout_b_wsgi_contract_failure")
    value = json.loads(document.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("logic_holdout_b_non_object_response")
    return value


def _page_request(_base_url: str, _path: str = "/pg388", *, timeout: float = 5.0) -> str:
    return HOLDOUT_PAGE_HTML


def collect(*, authorization_id: str = "pg388-local-logic-holdout-b", timeout: float = 5.0) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    report, rows, sidecars = run(
        DEFAULT_BASE_URL,
        authorization_id=authorization_id,
        timeout=timeout,
        environ={"PG388_LOCAL_EVAL": "1"},
        request=_call_wsgi,
        page_request=_page_request,
    )
    report["execution"] = {
        "local_fixture_in_process": True,
        "local_frontend_contacted": False,
        "target_contacted": False,
        "external_network": False,
        "wire_created": False,
        "docker_started": False,
        "gpu_touched": False,
    }
    report["source_contract"]["image_attested"] = False
    report["source_contract"]["runtime_attested"] = False
    report["training_eligible"] = 0
    report["promotion"] = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    report["report_sha256"] = _digest(report)
    return report, rows, sidecars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-id", default="pg388-local-logic-holdout-b")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--rows-output", default=DEFAULT_ROWS)
    parser.add_argument("--sidecars-output", default=DEFAULT_SIDECARS)
    args = parser.parse_args()
    report, rows, sidecars = collect(authorization_id=args.authorization_id, timeout=args.timeout)
    for value in (args.report_output, args.rows_output, args.sidecars_output):
        Path(value).parent.mkdir(parents=True, exist_ok=True)
    write_artifacts(args.report_output, args.rows_output, args.sidecars_output, report, rows, sidecars)
    print(json.dumps({"status": report["status"], "implementation_id": report.get("implementation_id"), "counts": report["counts"], "execution": report["execution"]}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
