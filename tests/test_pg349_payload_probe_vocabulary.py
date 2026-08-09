import json
from pathlib import Path


VOCAB = Path(__file__).parents[1] / "research" / "pg349_payload_probe_vocabulary_v1.json"


def _load():
    return json.loads(VOCAB.read_text(encoding="utf-8"))


def test_payload_vocab_is_source_grounded_and_abstract_only():
    data = _load()
    assert data["status"] == "abstract_source_grounded_evaluator_binding_required"
    assert len(data["source_basis"]) >= 4
    policy = data["storage_policy"]
    assert policy["raw_strings_stored"] is False
    assert policy["raw_strings_in_model_context"] is False
    assert policy["raw_examples_allowed_only"] == "ephemeral_evaluator_side_manual_review"
    assert policy["payload_catalog_promotion_allowed"] is False


def test_payload_vocab_has_composable_dimensions_and_oracle_gate():
    data = _load()
    dims = data["token_dimensions"]
    for name in (
        "surface_context",
        "boundary_strategy",
        "syntax_category",
        "encoding_layer",
        "transport_placement",
        "probe_variant",
        "oracle_kind",
        "action_class",
    ):
        assert dims[name]
        assert "unknown" in dims[name] or name == "action_class"
    contract = data["composition_contract"]
    assert set(contract["required_slots"]) >= {
        "surface_context",
        "boundary_strategy",
        "syntax_category",
        "encoding_layer",
        "transport_placement",
        "probe_variant",
        "oracle_kind",
    }
    assert contract["safe_to_send_default"] is False
    assert contract["one_variable_repair"] is True
    assert contract["missing_observation_is_not_negative"] is True
    assert "typed_oracle" in contract["positive_requires"]


def test_payload_vocab_contains_no_literal_payload_or_network_target():
    data = VOCAB.read_text(encoding="utf-8").lower()
    forbidden = ("javascript:", "<script", "alert(", "document.cookie", "http://", "https://example")
    # URLs in source_basis are references, not probe material; inspect the
    # token and family sections independently so citations remain allowed.
    body = json.loads(data)
    token_text = json.dumps(
        {"token_dimensions": body["token_dimensions"], "composition_contract": body["composition_contract"], "family_mapping": body["family_mapping"]},
        ensure_ascii=False,
    ).lower()
    assert all(marker not in token_text for marker in forbidden)

