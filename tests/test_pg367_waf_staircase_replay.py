from __future__ import annotations

import json
import re

from scripts.run_pg367_waf_staircase_replay import replay


HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def test_one_policy_replays_real_get_and_post_with_single_axis_repair() -> None:
    report = replay(policy_ids=("delimiter_normalizer",), seed=36711, timeout=2)
    assert report["status"] == "completed_evaluator_only"
    assert report["target_contacted"] is True
    assert report["network_policy"] == "loopback_only"
    assert report["external_network"] is False
    assert report["docker_used"] is False
    assert report["counts"]["episodes"] == 2
    assert report["counts"]["get_episodes"] == 1
    assert report["counts"]["post_episodes"] == 1
    assert report["counts"]["repair_rows"] == 8
    assert report["counts"]["repair_action_changed"] == 8
    assert report["counts"]["negative_violation"] == 0

    methods = {episode["method"] for episode in report["episodes"]}
    assert methods == {"GET", "POST"}
    for episode in report["episodes"]:
        assert episode["checks"]["candidate_typed"] is True
        assert episode["checks"]["reference_typed"] is True
        assert episode["checks"]["negative_clean"] is True
        assert episode["checks"]["replay_consistent"] is True
        assert set(episode["roles"]) == {"candidate", "reference", "negative", "replay"}
        for row in episode["roles"].values():
            assert HASH_RE.fullmatch(row["evidence_sha256"])
            assert row["fresh_reset"]["completed"] is True
            repair = row["repair"]
            assert repair["required"] is True
            assert repair["single_axis_changed"] is True
            assert len(repair["changed_probe_axes"]) == 1
            assert repair["action_changed"] is True
            assert row["negative_control_clean"] is (row["role"] == "negative")
            request = row["baseline"]["request"]
            assert set(request) == {"method", "path_sha256", "body_sha256", "body_length"}
            assert HASH_RE.fullmatch(request["path_sha256"])
            assert HASH_RE.fullmatch(request["body_sha256"])
            if episode["method"] == "GET":
                assert request["body_length"] == 0
            else:
                assert request["body_length"] > 0


def test_replay_report_does_not_persist_canary_url_body_or_raw_response() -> None:
    report = replay(policy_ids=("delimiter_normalizer",), seed=36712, timeout=2)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "pg367-runtime-canary" not in text
    assert "http://127.0.0.1:" not in text
    assert '"url"' not in text
    assert '"body"' not in text
    assert '"response_body"' not in text
    assert report["raw_persistence"] == {
        "request_url_stored": False,
        "request_body_stored": False,
        "response_body_stored": False,
        "runtime_canary_stored": False,
        "model_context_visible": False,
        "training_visible": False,
    }
    assert all(row["raw_request_stored"] is False and row["raw_response_stored"] is False for episode in report["episodes"] for row in episode["roles"].values())
    assert report["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def test_show_wire_is_ephemeral_only(capsys) -> None:
    report = replay(policy_ids=("delimiter_normalizer",), seed=36713, timeout=2, show_wire=True)
    output = capsys.readouterr().out
    assert "[ephemeral wire] GET http://127.0.0.1:" in output
    assert "[ephemeral wire] POST http://127.0.0.1:" in output
    assert "pg367-runtime-canary" in output
    assert "pg367-runtime-canary" not in json.dumps(report, ensure_ascii=False)
