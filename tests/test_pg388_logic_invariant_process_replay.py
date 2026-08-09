from __future__ import annotations

import json
from pathlib import Path


REPORT = Path("research/pg388_logic_invariant_process_replay_v1.json")


def test_process_replay_has_fresh_role_matrix_and_clean_negative_lane() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    counts = report["counts"]
    assert report["status"] == "completed_logic_process_candidate_only"
    assert counts["episodes"] == counts["cases"] * counts["seeds"] * counts["roles"]
    assert counts["typed_effect"] == counts["episodes"] - counts["negative_episodes"]
    assert counts["negative_violation"] == 0
    assert counts["failure_observed"] == counts["action_changed"]
    assert counts["fresh_reset"] == counts["episodes"]
    assert report["safety"] == {
        "in_process_only": True,
        "external_network": False,
        "state_mutated": False,
        "raw_values_stored": False,
        "credentials_accessed": False,
        "wire_created": False,
    }
    assert report["promotion"]["vulnerability_claim_allowed"] is False


def test_process_replay_keeps_evidence_abstract_and_role_bound() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    episodes = report["episodes"]
    assert episodes
    evidence = {(item["case_ref"], item["seed"], item["role"], item["evidence_sha256"]) for item in episodes}
    assert len(evidence) == len(episodes)
    for item in episodes[:64]:
        text = json.dumps({"context": item["context_tokens"], "target": item["target_tokens"]}, ensure_ascii=False).casefold()
        for marker in ("http://", "https://", "payload", "wire", "response_body", "credential", "<script"):
            assert marker not in text
