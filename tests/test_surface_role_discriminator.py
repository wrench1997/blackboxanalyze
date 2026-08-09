import torch

from app.surface_role_discriminator import (
    SURFACE_ROLE_FEATURE_DIM,
    SURFACE_ROLES,
    SurfaceRoleDiscriminator,
    surface_shape_feature_vector,
)


def test_surface_role_features_use_generic_shape_only():
    shape = {
        "content_type_class": "json",
        "status_class": "2xx",
        "html_tag_count": 0,
        "html_attribute_count": 0,
        "script_count": 0,
        "json_field_count": 2,
        "response_header_count": 4,
        "body_length": 80,
        "body_length_delta_abs": 12,
        "surface_role": "json_echo",
        "marker_in_json_value": True,
    }
    vector = surface_shape_feature_vector(shape)
    assert len(vector) == SURFACE_ROLE_FEATURE_DIM
    assert vector[1] == 1.0
    assert "json_echo" not in str(vector)


def test_surface_role_discriminator_emits_role_or_abstain():
    model = SurfaceRoleDiscriminator()
    output = model.decode(torch.zeros(2, SURFACE_ROLE_FEATURE_DIM), abstain_threshold=0.0)
    assert len(output) == 2
    assert output[0]["role"] in SURFACE_ROLES
    assert output[0]["abstained"] is False
