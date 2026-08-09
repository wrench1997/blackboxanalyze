import asyncio

import pytest

from app.juice_shop_shadow_collector import (
    JUICE_SHOP_SHADOW_ORACLE,
    JuiceShopShadowCollector,
    default_juice_shop_shadow_specs,
    validate_juice_shop_shadow_spec,
)


def test_shadow_specs_are_loopback_get_only_and_leave_evaluator_paths_hidden():
    specs = default_juice_shop_shadow_specs()
    assert len(specs) == 5
    normalized = [validate_juice_shop_shadow_spec(spec) for spec in specs]
    assert all(row["target"] == "http://127.0.0.1:3100" for row in normalized)
    assert all(row["method"] == "GET" for row in normalized)
    assert all(row["expected_oracle"] == JUICE_SHOP_SHADOW_ORACLE for row in normalized)
    with pytest.raises(ValueError, match="allow-listed"):
        validate_juice_shop_shadow_spec({**specs[0], "path": "/api/Challenges"})
    with pytest.raises(ValueError, match="read-only GET"):
        validate_juice_shop_shadow_spec({**specs[0], "method": "POST"})


def test_shadow_collector_emits_bounded_unsupported_surface_record():
    record = asyncio.run(JuiceShopShadowCollector(target_instance_id="shadow-test").collect(default_juice_shop_shadow_specs()[2]))
    assert record["candidate_status"] in {"unsupported_surface_abstain", "environment_failure_abstain"}
    if record["candidate_status"] == "environment_failure_abstain":
        assert record["environment_failure"] is True
    assert record["rule_ir_result"] is False
    assert record["safety"]["local_only"] is True
    assert record["safety"]["raw_body_stored"] is False
    assert "body_sha256" in record["response_projection"]
    assert "target_instance_id" in record["evidence"]["reset"]
