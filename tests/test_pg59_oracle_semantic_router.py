import json
from pathlib import Path


def _read(name):
    return json.loads(Path("research", name).read_text(encoding="utf-8"))


def test_pg59_uses_post_probe_semantics_without_family_or_raw_inputs():
    protocol = _read("pg59_oracle_semantic_router_protocol_v1.json")
    report = _read("pg59_oracle_semantic_router_report_v1.json")
    contract = report["training_contract"]
    assert contract["oracle_semantics_are_post_probe_evidence"] is True
    assert contract["family_name_in_features"] is False
    assert contract["raw_probe_in_features"] is False
    assert contract["raw_response_body_in_features"] is False
    assert contract["unknown_modality_fail_closed"] is True
    assert protocol["gates"]["post_probe_router_is_not_discovery_claim"] is True


def test_pg59_independent_typed_oracle_routing_is_safe_but_quarantined():
    report = _read("pg59_oracle_semantic_router_report_v1.json")
    holdout = report["metrics"]["holdout_semantic_gate"]
    assert holdout["known_family_recall"] == 1.0
    assert holdout["known_wrong_family_count"] == 0
    assert holdout["unknown_misname_count"] == 0
    assert holdout["negative_false_accept_count"] == 0
    assert holdout["abstain_rate"] == 0.916667
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_capability_claim_allowed"] is False
