import pytest

from app.pikachu_replay_collector import (
    PIKACHU_BASE_URL,
    PIKACHU_FRESH_BASE_URL,
    PikachuReplayCollector,
    SAFE_INVENTORY_PATHS,
    default_pikachu_counterfactual_specs,
    default_pikachu_probe_specs,
    default_pikachu_paired_specs,
    validate_pikachu_spec,
)


def test_default_pikachu_specs_are_loopback_get_canaries():
    specs = default_pikachu_probe_specs()
    assert len(specs) == 7
    normalized = [validate_pikachu_spec(spec) for spec in specs]
    assert all(row["target"] == PIKACHU_BASE_URL for row in normalized)
    assert all(row["method"] == "GET" for row in normalized)
    assert all(row["payload"]["safety"]["does_not_execute"] is True for row in normalized)
    assert {row["family"] for row in normalized} == {"xss", "injection", "url_redirect"}


def test_pikachu_collector_fails_closed_on_scope_mutation_and_unapproved_query():
    spec = default_pikachu_probe_specs()[0]
    with pytest.raises(ValueError, match="127.0.0.1:8766"):
        validate_pikachu_spec({**spec, "target": "https://example.test"})
    with pytest.raises(ValueError, match="only read-only GET"):
        validate_pikachu_spec({**spec, "method": "POST"})
    with pytest.raises(ValueError, match="not allow-listed"):
        validate_pikachu_spec({**spec, "params": {"message": "safe", "callback": "http://127.0.0.1"}})


def test_fresh_loopback_profile_is_explicitly_attested_without_claiming_per_request_reset():
    collector = PikachuReplayCollector(
        base_url=PIKACHU_FRESH_BASE_URL,
        target_instance_id="fresh-test-instance",
        fresh_target=True,
    )
    spec = validate_pikachu_spec(default_pikachu_probe_specs()[0])
    reset = collector._read_only_reset(spec)
    assert collector.base_url == PIKACHU_FRESH_BASE_URL
    assert reset["fresh"] is True
    assert reset["fresh_target"] is True
    assert reset["target_instance_id"] == "fresh-test-instance"
    assert reset["state_change_allowed"] is False


def test_inventory_contains_hazardous_pages_for_explicit_abstention():
    assert "/vul/rce/rce_ping.php" in SAFE_INVENTORY_PATHS
    assert "/vul/ssrf/ssrf_curl.php" in SAFE_INVENTORY_PATHS
    assert "/vul/xxe/xxe_1.php" in SAFE_INVENTORY_PATHS


def test_paired_specs_cover_four_encodings_and_cross_surface_roles():
    specs = default_pikachu_paired_specs()
    assert len(specs) == 24
    assert {spec["pair"]["variant"] for spec in specs} == {
        "plain", "url_percent", "html_entity", "double_html_entity",
    }
    assert {spec["pair"]["surface_role"] for spec in specs} == {
        "reflected_get", "dom_value_source", "sqli_str", "sqli_search",
        "sqli_blind_boolean", "sqli_blind_time",
    }
    normalized = [validate_pikachu_spec(spec) for spec in specs]
    assert all(row["pair"]["pair_id"] for row in normalized)
    with pytest.raises(ValueError, match="variant"):
        validate_pikachu_spec({**specs[0], "pair": {**specs[0]["pair"], "variant": "raw-exploit"}})


def test_counterfactual_specs_use_a_non_matching_inert_marker():
    specs = default_pikachu_counterfactual_specs()
    assert len(specs) == 12
    assert {spec["family"] for spec in specs} == {"xss", "injection"}
    assert {spec["encoding"] for spec in specs} == {
        "counterfactual_marker_substitution_plain",
        "counterfactual_marker_substitution_url_percent",
    }
    assert all(spec["marker"] != next(iter(spec["params"].values())) for spec in specs if spec["family"] == "xss")
    normalized = [validate_pikachu_spec(spec) for spec in specs]
    assert all(row["payload"]["safety"]["does_not_execute"] is True for row in normalized)
