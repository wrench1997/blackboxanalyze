import asyncio
import threading

from app.logic_access_fixture_v2 import (
    LogicAccessV2Collector,
    default_logic_access_v2_specs,
    logic_access_v2_source_sha256,
    make_logic_access_v2_fixture_server,
    validate_logic_access_v2_fixture_spec,
)


def test_logic_access_v2_changes_surface_but_keeps_typed_boundary_contract():
    server = make_logic_access_v2_fixture_server(port=8812, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(
            LogicAccessV2Collector(
                base_url="http://127.0.0.1:8812",
                target_instance_id="logic-v2-test",
                source_hash=logic_access_v2_source_sha256(),
            ).collect_many(default_logic_access_v2_specs(target="http://127.0.0.1:8812"))
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(rows) == 20
    assert sum(bool(row["rule_ir_result"]) for row in rows) == 8
    assert all(row["safety"]["database_touched"] is False for row in rows)
    assert all(row["safety"]["state_mutated"] is False for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)
    assert all(row["semantic"]["surface"] == "synthetic_logic_access_http_v2" for row in rows)


def test_logic_access_v2_pair_and_route_are_not_v1_shortcuts():
    specs = default_logic_access_v2_specs(target="http://127.0.0.1:8812")
    plain = next(spec for spec in specs if spec["pair"]["pair_id"] == "logic-pg16-authorize-boundary" and spec["pair"]["variant"] == "plain")
    encoded = next(spec for spec in specs if spec["pair"]["pair_id"] == "logic-pg16-authorize-boundary" and spec["pair"]["variant"] == "url_percent")
    assert "/authorize" in plain["path"]
    assert "/gate" not in plain["path"]
    assert plain["path"] != encoded["path"]
    try:
        validate_logic_access_v2_fixture_spec({"target": "http://127.0.0.1:8812", "path": "/authorize?role=admin", "source_id": "s", "lab_id": "l"})
    except ValueError:
        pass
    else:
        raise AssertionError("v1 query vocabulary was accepted by the v2 fixture")

