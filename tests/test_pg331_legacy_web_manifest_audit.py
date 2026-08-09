from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "audit_pg331_legacy_web_manifest.py"
SPEC = importlib.util.spec_from_file_location("pg331_legacy_manifest_audit_test", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_legacy_manifest_is_diagnostic_and_keeps_missing_axes_explicit():
    document = {
        "stats": {"get_query_surface_count": 1, "post_form_surface_count": 1, "incomplete_surface_count": 2},
        "page_summaries": [
            {
                "loaded": True,
                "title": "redacted",
                "link_observation_count": 2,
                "form_count": 1,
                "script_count": 1,
                "response_projection": {"status_chain": [{"status": 200}]},
            }
        ],
        "route_catalog": [{"methods_observed": ["GET", "POST"], "quality_status": "incomplete_missing_request_parameter_context"}],
        "request_response_rows": [{"response_schema": {"parameterized_response_observed": False}}],
        "script_catalog": [{"kind": "external"}],
    }

    report = MODULE.audit_manifest(document, input_path="legacy.json")

    assert report["status"] == "diagnostic_only_blocked"
    assert report["coverage"]["route_count"] == 1
    assert report["axes"]["request_transport"]["status"] == "partial"
    assert report["axes"]["response_transport"]["parameterized_response_observed_count"] == 0
    assert report["axes"]["failure_feedback"]["status"] == "not_observed"
    assert report["axes"]["belief_and_replay"]["status"] == "not_observed"
    assert report["promotion"]["training_allowed"] is False


def test_legacy_manifest_audit_does_not_persist_route_literals():
    document = {"page_summaries": [{"path": "/secret-route", "url": "http://127.0.0.1:1/secret-route", "loaded": True}], "route_catalog": [], "request_response_rows": [], "script_catalog": []}

    report = MODULE.audit_manifest(document, input_path="legacy.json")
    serialized = str(report)

    assert "/secret-route" not in serialized
    assert "http://127.0.0.1:1" not in serialized
