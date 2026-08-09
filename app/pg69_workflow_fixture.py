"""PG-69 independent workflow-invariant fixture.

This is a small, stateless HTTP implementation used only as a family-held-out
evaluation target.  It does not execute markup, SQL, commands or redirects.
The evaluator may inspect the in-memory request values, while the model-facing
projection contains only bounded response shape information.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


PG69_WORKFLOW_SCHEMA = "sift-pg69-workflow-invariant-fixture-v1"
PG69_WORKFLOW_FAMILY = "workflow_invariant"
PG69_WORKFLOW_VARIANTS = ("amber", "violet")
PG69_WORKFLOW_PORTS = (8815, 8816)
PG69_WORKFLOW_ROUTES = frozenset({"/handoff", "/quota", "/heartbeat"})


def source_sha256() -> str:
    from pathlib import Path

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def evaluate_workflow(route: str, values: dict[str, str], variant: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Evaluate a bounded invariant and return response plus evaluator-only oracle."""

    marker = str(values.get("marker", ""))
    if route == "/handoff":
        commit = values.get("verb") == "commit" and values.get("prior") == "verified"
        challenge_matches = values.get("stamp") == values.get("fresh")
        violation = bool(commit and not challenge_matches)
        outcome = "commit" if commit else "deny"
        status = 200 if commit else 403
        if variant == "amber":
            response = {"decision": outcome, "scope": "sealed" if commit else "open", "marker": marker}
        else:
            response = {"accepted": commit, "phase": outcome, "marker": marker}
        oracle = {
            "oracle_id": "pg69-workflow-history-binding-v1",
            "oracle_signal": "history_binding_mismatch",
            "positive": violation,
            "positive_authority": True,
            "typed": {"commit_action": values.get("verb") == "commit", "previous_verified": values.get("prior") == "verified", "challenge_matches": challenge_matches, "unexpected_replay_accept": violation},
        }
        return status, response, oracle
    if route == "/quota":
        member = values.get("member") == "1"
        amount = str(values.get("amount", "0"))
        try:
            numeric = int(amount)
        except (TypeError, ValueError):
            numeric = 0
        issued = bool(member and numeric > 100)
        expected_issued = bool(member and numeric >= 100)
        violation = bool(member and numeric == 100 and not issued)
        if variant == "amber":
            response = {"issued": issued, "amount_class": "high" if numeric > 100 else "boundary", "marker": marker}
        else:
            response = {"grant": "yes" if issued else "no", "threshold": numeric >= 100, "marker": marker}
        oracle = {
            "oracle_id": "pg69-workflow-boundary-v1",
            "oracle_signal": "business_boundary_mismatch",
            "positive": violation,
            "positive_authority": True,
            "typed": {"member": member, "boundary_hit": numeric == 100, "expected_issued": expected_issued, "observed_issued": issued, "invariant_violation": violation},
        }
        return 200, response, oracle
    return 200, {"alive": True, "marker": marker}, {
        "oracle_id": "pg69-workflow-ordinary-v1",
        "oracle_signal": "no_typed_violation",
        "positive": False,
        "positive_authority": True,
        "typed": {"ordinary_response": True, "invariant_violation": False},
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "sift-pg69-workflow/1"

    def _serve(self, method: str, raw_path: str, values: dict[str, str]) -> None:
        route = urlsplit(raw_path).path
        variant = str(getattr(self.server, "fixture_variant", "amber"))
        if route not in PG69_WORKFLOW_ROUTES or variant not in PG69_WORKFLOW_VARIANTS:
            status, response = 404, {"error": "not_found"}
        else:
            status, response, _ = evaluate_workflow(route, values, variant)
        body = json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-Sift-Workflow-Variant", variant)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        values = {str(key): str(items[0]) for key, items in parse_qs(parsed.query, keep_blank_values=True).items() if items}
        self._serve("GET", self.path, values)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or 0)
        encoded = self.rfile.read(min(length, 4096)).decode("utf-8", errors="replace")
        values = {str(key): str(items[0]) for key, items in parse_qs(encoded, keep_blank_values=True).items() if items}
        self._serve("POST", self.path, values)

    def log_message(self, format: str, *args: Any) -> None:
        return


class WorkflowFixtureServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in PG69_WORKFLOW_VARIANTS:
            raise ValueError("unknown PG-69 workflow variant")
        super().__init__(address, _Handler)
        self.fixture_variant = variant


def make_workflow_server(port: int, variant: str) -> WorkflowFixtureServer:
    if int(port) not in PG69_WORKFLOW_PORTS:
        raise ValueError("PG-69 workflow port is not allow-listed")
    return WorkflowFixtureServer(("127.0.0.1", int(port)), variant)


__all__ = [
    "PG69_WORKFLOW_FAMILY",
    "PG69_WORKFLOW_PORTS",
    "PG69_WORKFLOW_ROUTES",
    "PG69_WORKFLOW_SCHEMA",
    "PG69_WORKFLOW_VARIANTS",
    "evaluate_workflow",
    "make_workflow_server",
    "source_sha256",
]
