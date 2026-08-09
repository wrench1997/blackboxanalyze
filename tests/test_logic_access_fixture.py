import asyncio
import threading

from app.logic_access_fixture import (
    LogicAccessCollector,
    default_logic_access_fixture_specs,
    logic_access_fixture_source_sha256,
    make_logic_access_fixture_server,
)


def test_logic_access_fixture_emits_typed_positive_and_counterfactual():
    server = make_logic_access_fixture_server(port=8795, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(LogicAccessCollector(target_instance_id="test-pg10", source_hash=logic_access_fixture_source_sha256()).collect_many(default_logic_access_fixture_specs(target="http://127.0.0.1:8795")))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    positives = [row for row in rows if row["rule_ir_result"]]
    controls = [row for row in rows if not row["rule_ir_result"]]
    assert len(rows) == 20
    assert len(positives) == 8
    assert len(controls) == 12
    assert {row["oracle_projection"]["oracle_signal"] for row in positives} == {
        "authorization_boundary_divergence",
        "business_boundary_mismatch",
        "history_binding_mismatch",
    }
    assert all("raw_body" not in row["evidence"] for row in rows)
    assert all(row["evidence"]["reset"]["fresh_target"] for row in rows)


def test_logic_access_encoded_pair_has_same_typed_oracle():
    specs = default_logic_access_fixture_specs(target="http://127.0.0.1:8795")
    pair = [spec for spec in specs if spec["pair"]["pair_id"] == "logic-pg10-gate-truthy"]
    assert pair[0]["path"] != pair[1]["path"]
    assert pair[0]["pair"]["variant"] == "plain"
    assert pair[1]["pair"]["variant"] == "url_percent"
