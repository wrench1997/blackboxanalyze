import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg220_is_fresh_get_post_shadow_only() -> None:
    report = _load("pg220_live_shadow_replay_report_v1.json")
    counts = report["counts"]
    assert report["status"] == "completed_fresh_local_shadow_replay"
    assert counts["fresh_container_count"] == 4
    assert counts["get_episode_count"] == 2
    assert counts["post_episode_count"] == 2
    assert counts["ai_send_count"] == 4
    assert counts["reference_send_count"] == 4
    assert counts["negative_send_count"] == 4
    assert counts["typed_effect_confirmed_count"] == 4
    assert counts["result_fixture_verified_count"] == 4
    assert counts["shadow_target_action_match_count"] == counts["shadow_row_count"]
    assert counts["shadow_gated_unsafe_allow_count"] == 0
    assert counts["docker_restart_used_count"] == 0
    assert report["safety"]["loopback_only"] is True
    assert report["promotion"]["live_send_takeover_allowed"] is False
    assert all(row["raw_payload_strings_stored"] is False for row in report["episodes"])
    assert all(row["raw_response_bodies_stored"] is False for row in report["episodes"])
