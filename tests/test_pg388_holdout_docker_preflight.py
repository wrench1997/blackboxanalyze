from __future__ import annotations

from scripts.preflight_pg388_holdout_docker import preflight


def _env() -> dict[str, str]:
    return {
        "PG388_LOCAL_DOCKER_EVAL": "1",
        "PG388_PYTHON_IMAGE_DIGEST_B": "sha256:" + "b" * 64,
        "PG388_PYTHON_IMAGE_DIGEST": "sha256:" + "a" * 64,
        "PG388_NODE_BASE_IMAGE": "node:20-alpine@sha256:" + "c" * 64,
    }


def test_default_preflight_fails_closed_without_explicit_gate() -> None:
    result = preflight(environ={})
    assert result["status"] == "blocked_holdout_docker_preflight"
    assert "PG388_LOCAL_DOCKER_EVAL=1_required" in result["reasons"]
    assert result["docker_started"] is False
    assert result["image_attested"] is False


def test_reviewed_digest_shape_and_compose_contract_can_parse_read_only() -> None:
    calls: list[tuple[object, ...]] = []

    class Result:
        returncode = 0

    def fake_runner(command, **kwargs):
        calls.append(tuple(command))
        return Result()

    result = preflight(environ=_env(), check_compose=True, runner=fake_runner)
    assert result["status"] == "ready_for_operator_review"
    assert result["checks"]["compose_config"] is True
    assert len(calls) == 1
    assert "up" not in calls[0] and "build" not in calls[0]
    assert result["training_eligible"] == 0
    assert result["promotion"]["vulnerability_claim_allowed"] is False
