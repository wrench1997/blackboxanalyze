import asyncio

from app.pikachu_active_controller import ACTIVE_CONTROLLER_SCHEMA, PikachuActiveController, _projection_likelihood
from app.pikachu_replay_collector import default_pikachu_paired_specs


class _FakeCollector:
    """Return only the bounded fields consumed by the controller."""

    def __init__(self):
        self.calls = []

    async def collect(self, spec):
        self.calls.append(spec)
        is_candidate = (spec.get("pair") or {}).get("surface_role") == "reflected_get"
        projection = {
            "marker_reflected": is_candidate,
            "marker_in_attribute": is_candidate,
            "marker_in_script_source": False,
            "sql_error_shape": False,
            "body_length_delta_abs": 0,
            "external_redirect": False,
        }
        return {
            "sample_id": f"fake-{len(self.calls)}",
            "pair": spec.get("pair"),
            "oracle_projection": projection,
            "rule_ir_result": is_candidate,
            "evidence": {"evidence_hash": f"{len(self.calls):064x}"},
        }


class _FakeSharedRouter:
    def inspect(self, record):
        return {
            "abstained": False,
            "ood": False,
            "belief_prior": {"xss": 0.70, "injection": 0.20, "url_redirect": 0.10},
            "shared_probabilities": {"control": 0.0, "xss": 0.70, "injection": 0.20, "access_control": 0.05, "logic": 0.05},
            "candidate_family": "xss",
            "candidate_route": "xss",
            "confidence": 0.70,
            "margin": 0.30,
            "route_only": True,
            "positive_authority": False,
        }


def test_active_controller_screens_then_refines_with_budget():
    collector = _FakeCollector()
    result = asyncio.run(
        PikachuActiveController(collector, max_requests=7).run(default_pikachu_paired_specs())
    )

    assert result["schema_version"] == ACTIVE_CONTROLLER_SCHEMA
    assert result["request_count"] == 7
    assert sum(row["stage"] == "screen" for row in result["selection_trace"]) == 6
    assert sum(row["stage"] == "refine" for row in result["selection_trace"]) == 1
    assert result["selection_trace"][-1]["surface_key"].endswith("::reflected_get")
    assert result["selection_trace"][-1]["variant"] != "plain"
    assert all(spec["target"] == "http://127.0.0.1:8766" for spec in collector.calls)
    assert result["safety"] == {
        "loopback_only": True,
        "read_only_get_only": True,
        "fresh_target": False,
        "target_instance_id": "unattested",
        "external_network": False,
        "script_execution": False,
        "database_write": False,
        "raw_body_stored": False,
    }


def test_belief_likelihood_prioritizes_sql_differential_and_typed_sink():
    sql = _projection_likelihood({"oracle_projection": {"controlled_differential": True, "interpreter_boundary": True, "marker_reflected": True}})
    attr = _projection_likelihood({"oracle_projection": {"sink_kind": "html_attribute", "marker_reflected": True}})
    text = _projection_likelihood({"oracle_projection": {"marker_in_html_text": True, "marker_reflected": True}})
    assert sql["injection"] > sql["xss"]
    assert attr["xss"] > attr["injection"]
    assert text["xss"] < attr["xss"]


def test_shared_router_is_only_a_prior_for_active_controller():
    collector = _FakeCollector()
    result = asyncio.run(
        PikachuActiveController(collector, max_requests=7, shared_router=_FakeSharedRouter()).run(default_pikachu_paired_specs())
    )
    assert result["safety"]["shared_router_diagnostic_only"] is True
    assert all("shared_router" in row for row in result["selection_trace"])
