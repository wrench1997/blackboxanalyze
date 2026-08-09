import asyncio
import threading

import torch

from app.logic_access_decoder import LOGIC_ACCESS_FEATURE_DIM, LOGIC_ACCESS_SURFACE_SHORTCUT_INDICES, LogicAccessDecoder, logic_access_feature_vector, logic_access_model_feature_vector
from app.logic_access_fixture import LogicAccessCollector, default_logic_access_fixture_specs, logic_access_fixture_source_sha256, make_logic_access_fixture_server


def _rows():
    server = make_logic_access_fixture_server(port=8795, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return asyncio.run(LogicAccessCollector(target_instance_id="decoder-pg10", source_hash=logic_access_fixture_source_sha256()).collect_many(default_logic_access_fixture_specs(target="http://127.0.0.1:8795")))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_logic_access_features_do_not_include_oracle_values():
    row = _rows()[0]
    row["oracle_projection"] = {"positive": True, "typed": {"secret": "hidden"}}
    vector = logic_access_feature_vector(row)
    assert len(vector) == LOGIC_ACCESS_FEATURE_DIM
    assert "hidden" not in str(vector)


def test_logic_access_decoder_abstain_contract():
    model = LogicAccessDecoder()
    output = model.decode(torch.zeros(2, LOGIC_ACCESS_FEATURE_DIM), abstain_threshold=0.0, margin_threshold=0.0, temperature=1.5)
    assert len(output) == 2
    assert all("rule_ir" in row for row in output)


def test_logic_access_model_view_zeroes_route_vocabulary_shortcuts():
    row = _rows()[0]
    raw = logic_access_feature_vector(row)
    model_view = logic_access_model_feature_vector(row)
    assert len(model_view) == LOGIC_ACCESS_FEATURE_DIM
    assert all(model_view[index] == 0.0 for index in LOGIC_ACCESS_SURFACE_SHORTCUT_INDICES)
    assert any(raw[index] != 0.0 for index in LOGIC_ACCESS_SURFACE_SHORTCUT_INDICES)
