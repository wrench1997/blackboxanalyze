from app.surface_novelty_discriminator import SurfaceNoveltyDiscriminator, make_surface_observation


def _projection(key_count: int):
    return {
        "status_class": "2xx",
        "content_type_class": "json",
        "body_length_bucket": "256-4095",
        "frame_policy": "unknown",
        "header_names": ["content-type"],
        "location_origin_changed": False,
        "state_changed": False,
        "transport_error": False,
        "shape": {"array_count": 0, "bool_count": 1, "key_count": key_count, "number_count": 0, "scalar_count": 1, "string_count": 0},
    }


def test_surface_novelty_is_bounded_and_fails_closed():
    known = make_surface_observation(_projection(1), _projection(2), method="GET", encoding_class="identity", phase="confirm", safe_probe=True)
    novel = make_surface_observation(_projection(1), _projection(4), method="GET", encoding_class="identity", phase="confirm", safe_probe=True)
    discriminator = SurfaceNoveltyDiscriminator().fit([known])
    assert discriminator.predict(known)["decision"] == "in_domain"
    assert discriminator.predict(novel)["decision"] == "novel_surface"
    marker_free = make_surface_observation({"marker": {}}, {"marker": {}}, method="GET", encoding_class="identity", phase="confirm", safe_probe=True)
    assert "marker" not in marker_free


def test_surface_novelty_rejects_oracle_fields():
    observation = make_surface_observation(_projection(1), _projection(2), method="POST", encoding_class="identity", phase="confirm", safe_probe=True)
    observation["oracle_projection"] = {}
    try:
        SurfaceNoveltyDiscriminator().fit([observation])
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("oracle fields must fail closed")
