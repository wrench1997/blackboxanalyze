import json
import copy
from pathlib import Path

import torch

from app.catalog_rule_decoder import CATALOG_DECODER_FAMILIES
from app.payload_catalog import flatten_catalog, load_catalog
from app.pg23_multitask_decoder import (
    PG23MultiTaskDecoder,
    PG23_SURFACE_ROLES,
    assert_visible_trace_redacted,
    pg23_feature_vector,
    pg23_labels,
    pg23_visible_trace,
)


def _row():
    rows = flatten_catalog(load_catalog(Path("research/pikachu_counterfactual_catalog_v1.json")))
    return rows[0]


def test_pg23_visible_projection_hides_labels_and_raw_marker():
    row = _row()
    visible = pg23_visible_trace(row)
    assert_visible_trace_redacted(row)
    text = json.dumps(visible, ensure_ascii=False).casefold()
    assert "counterfactual" not in text
    assert row["payload"]["probe"] not in text
    assert "xss_reflected_get" not in text
    assert len(pg23_feature_vector(row)) == 256


def test_pg23_labels_and_offline_transport_negative():
    row = _row()
    family, surface, emit = pg23_labels(row)
    assert CATALOG_DECODER_FAMILIES[family] == "xss"
    assert PG23_SURFACE_ROLES[surface] == "xss_reflected_get"
    assert emit == 1.0
    negative = copy.deepcopy(row)
    negative["sample_id"] = f"{row['sample_id']}-transport"
    negative["counterfactual"] = {"kind": "negative_control", "intervention": "transport_failure"}
    negative["rule_ir_result"] = False
    negative["response_projection"] = {"status_code": 0, "body_length": 0, "headers": {}}
    negative["oracle_projection"] = {}
    assert pg23_labels(negative)[2] == 0.0
    assert negative["response_projection"]["status_code"] == 0
    assert_visible_trace_redacted(negative)


def test_pg23_decoder_emits_only_grammar_checked_template():
    model = PG23MultiTaskDecoder(dropout=0.0)
    features = torch.zeros((2, 256), dtype=torch.float32)
    decoded = model.decode(features, family_threshold=1.1, emit_threshold=1.1, margin_threshold=1.1)
    assert len(decoded) == 2
    assert all(item["abstained"] for item in decoded)
    assert all(item["rule_ir"] is None for item in decoded)
