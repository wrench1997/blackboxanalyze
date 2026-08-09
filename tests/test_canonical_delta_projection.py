from app.canonical_delta_projection import SCHEMA_VERSION, canonical_delta_tokens


def _projection(*, shape_key_count=1, body_length="256-4095", status="2xx"):
    return {
        "status_class": status,
        "content_type_class": "json",
        "body_length_bucket": body_length,
        "shape": {"array_count": 0, "bool_count": 1, "key_count": shape_key_count, "number_count": 0, "scalar_count": 1, "string_count": 0},
        "location_origin_changed": False,
        "state_changed": False,
        "transport_error": False,
    }


def test_canonical_projection_discards_surface_field_names():
    before = _projection(shape_key_count=1)
    after = _projection(shape_key_count=3)
    assert SCHEMA_VERSION == "canonical-delta-projection-v1"
    assert canonical_delta_tokens(before, after) == ("DELTA_EFFECT_INCREASE",)


def test_canonical_projection_keeps_generic_response_channel_separate():
    before = _projection(status="2xx")
    after = _projection(status="4xx", body_length="4096+")
    assert canonical_delta_tokens(before, after) == ("DELTA_RESPONSE_CHANGE",)


def test_canonical_projection_has_no_change_token_for_matched_negative():
    projection = _projection()
    assert canonical_delta_tokens(projection, projection) == ()
