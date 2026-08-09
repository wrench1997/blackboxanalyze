import copy
import json
from pathlib import Path

import torch

from app.catalog_rule_decoder import (
    CATALOG_DECODER_FAMILIES,
    CatalogRuleIRDecoderV2,
    CatalogRuleIRDecoder,
    abstract_catalog_rule_ir,
    catalog_feature_vector,
    catalog_visible_trace,
)
from app.payload_catalog import flatten_catalog, load_catalog
from app.rule_ir_decoder import FEATURE_DIM, validate_abstract_rule_ir


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_projection_removes_semantic_and_evaluator_labels():
    rows = flatten_catalog(load_catalog(ROOT / "research" / "payload_replay_catalog_v1.json"))
    assert rows
    forbidden = {"family", "source_id", "semantic", "rule_ir", "evaluator", "intended_output", "is_counterexample"}
    for row in rows:
        trace = catalog_visible_trace(row)
        encoded = json.dumps(trace, ensure_ascii=False, sort_keys=True).casefold()
        assert not any(token in encoded for token in forbidden)
        assert trace["output"] is False
        assert trace["input"]["action"]["path"] == "relative_path"
        assert "pair_id" not in encoded


def test_catalog_projection_ignores_recorded_oracle_result():
    rows = flatten_catalog(load_catalog(ROOT / "research" / "payload_replay_catalog_v1.json"))
    original = rows[0]
    changed = copy.deepcopy(original)
    changed["rule_ir_result"] = not bool(original.get("rule_ir_result"))
    changed["semantic"]["family"] = "made_up_label"
    assert catalog_feature_vector(original) == catalog_feature_vector(changed)


def test_catalog_decoder_emits_only_grammar_checked_templates():
    for family in CATALOG_DECODER_FAMILIES:
        validate_abstract_rule_ir(abstract_catalog_rule_ir(family))
    model = CatalogRuleIRDecoder()
    decoded = model.decode(torch.zeros(2, FEATURE_DIM), abstain_threshold=0.0, margin_threshold=0.0)
    assert len(decoded) == 2
    for row in decoded:
        assert row["family"] in CATALOG_DECODER_FAMILIES
        assert row["margin"] >= 0.0
        validate_abstract_rule_ir(row["rule_ir"])


def test_catalog_decoder_can_abstain_on_margin_gate():
    model = CatalogRuleIRDecoder()
    decoded = model.decode(torch.zeros(1, FEATURE_DIM), abstain_threshold=0.0, margin_threshold=1.0)
    assert decoded[0]["family"] is None
    assert decoded[0]["abstained"] is True
    assert decoded[0]["rule_ir"] is None


def test_catalog_decoder_v2_two_view_embedding_and_grammar_output():
    model = CatalogRuleIRDecoderV2()
    features = torch.zeros(3, FEATURE_DIM)
    logits = model(features)
    embeddings = model.encode(features)
    assert logits.shape == (3, len(CATALOG_DECODER_FAMILIES))
    assert embeddings.shape == (3, 96)
    decoded = model.decode(features, abstain_threshold=0.0, margin_threshold=0.0)
    for row in decoded:
        validate_abstract_rule_ir(row["rule_ir"])


def test_catalog_decoder_v2_margin_gate_is_fail_closed():
    model = CatalogRuleIRDecoderV2()
    decoded = model.decode(torch.zeros(1, FEATURE_DIM), abstain_threshold=0.0, margin_threshold=1.0)
    assert decoded[0]["abstained"] is True
    assert decoded[0]["rule_ir"] is None
