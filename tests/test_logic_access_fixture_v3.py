import asyncio
import threading

from app.logic_access_fixture_v3 import (
    LogicAccessV3Collector,
    default_logic_access_v3_specs,
    logic_access_v3_source_sha256,
    make_logic_access_v3_fixture_server,
    validate_logic_access_v3_fixture_spec,
)


def test_logic_access_v3_is_a_third_independent_source():
    server = make_logic_access_v3_fixture_server(port=8815, variant="red")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(
            LogicAccessV3Collector(
                base_url="http://127.0.0.1:8815",
                target_instance_id="logic-v3-test",
                source_hash=logic_access_v3_source_sha256(),
            ).collect_many(default_logic_access_v3_specs(target="http://127.0.0.1:8815"))
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(rows) == 20
    assert sum(bool(row["rule_ir_result"]) for row in rows) == 8
    assert all(row["semantic"]["surface"] == "synthetic_logic_access_http_v3" for row in rows)
    assert all(row["safety"]["database_touched"] is False for row in rows)
    assert all(row["safety"]["state_mutated"] is False for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_logic_access_v3_rejects_v2_route_vocabulary():
    try:
        validate_logic_access_v3_fixture_spec({"target": "http://127.0.0.1:8815", "path": "/authorize?subject=member", "source_id": "s", "lab_id": "l"})
    except ValueError:
        pass
    else:
        raise AssertionError("v2 route vocabulary was accepted by the v3 fixture")

