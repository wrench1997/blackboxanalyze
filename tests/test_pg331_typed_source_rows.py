from __future__ import annotations

from scripts.run_pg331_pikachu_typed_source_rows import (
    PROBES,
    ROUTES,
    _indicator,
    _source_meta,
    _target,
    _typed_callback,
)


def test_allowlisted_probe_matrix_is_get_post_and_three_roles() -> None:
    assert {str(route["method"]) for route in ROUTES} == {"GET", "POST"}
    assert {str(route["id"]) for route in ROUTES} == set(PROBES)
    assert all(set(values) == {"candidate", "reference", "negative"} for values in PROBES.values())


def test_evaluator_projection_contains_only_bounded_shapes_and_digest() -> None:
    details: dict[str, object] = {}
    callback = _typed_callback("sql-string-get", "candidate", details)
    projection = callback(b"<html><body>Your UID: 1</body></html>", {"Content-Type": "text/html; charset=utf-8"}, 200)
    assert projection["effect_marker"] == "present"
    assert projection["database_touched"] is False
    assert len(str(details["body_sha256"])) == 64
    assert "body" not in projection
    assert "raw_body" not in projection


def test_indicator_uses_negative_baseline_not_a_raw_body() -> None:
    assert _indicator({"marker_present": False, "body_length": 501}, {"marker_present": False, "body_length": 100}) is True
    assert _indicator({"marker_present": False, "body_length": 101}, {"marker_present": False, "body_length": 100}) is False


def test_target_is_abstract_and_source_meta_digests_only() -> None:
    target = _target("candidate")
    assert target["next_action"] == "send_probe"
    assert target["probe_variant_ref"] == "source_attested_candidate"
    meta = _source_meta(ROUTES[0], {"evidence_sha256": "x"})
    assert len(str(meta["image_digest"])) == 64
    assert len(str(meta["source_digest"])) == 64
    assert "/vul/" not in str(meta)
