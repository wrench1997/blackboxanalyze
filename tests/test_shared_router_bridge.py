from pathlib import Path

from app.shared_router_bridge import SHARED_ROUTER_BRIDGE_SCHEMA, SharedRouterBridge


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"


def test_shared_router_bridge_is_route_only_and_fail_closed():
    bridge = SharedRouterBridge(CHECKPOINT, strict_ood=True)
    result = bridge.inspect({
        "payload": {"method": "GET", "path": "/health?ok=1", "probe": "opaque", "headers": {"accept": "application/json"}},
        "response_projection": {"status_code": 200, "body_length": 42, "headers": {"content-type": "application/json"}, "json_shape": {"type": "object", "key_count": 2}},
        "evaluator_state_visible": False,
        "safety": {"local_only": True, "read_only": True},
    })
    assert result["schema_version"] == SHARED_ROUTER_BRIDGE_SCHEMA
    assert result["route_only"] is True
    assert result["positive_authority"] is False
    assert set(result["belief_prior"]) >= {"xss", "injection", "access_control"}
    assert set(result.get("shared_probabilities", {})) >= {"xss", "injection", "access_control", "logic"}
