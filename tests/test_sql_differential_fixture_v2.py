import asyncio
import threading

from app.sql_differential_fixture_v2 import (
    SQL_V2_ORACLE,
    SqlV2Collector,
    default_sql_v2_specs,
    make_sql_v2_fixture_server,
    sql_v2_source_sha256,
    validate_sql_v2_spec,
)


def test_sql_v2_uses_renamed_surface_but_same_bounded_oracle():
    server = make_sql_v2_fixture_server(port=8806, variant="beta")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(
            SqlV2Collector(
                base_url="http://127.0.0.1:8806",
                target_instance_id="sql-v2-test",
                source_hash=sql_v2_source_sha256(),
            ).collect_many(default_sql_v2_specs(target="http://127.0.0.1:8806"))
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(rows) == 15
    assert sum(bool(row["rule_ir_result"]) for row in rows) == 12
    assert all(row["semantic"]["expected_oracle"] == SQL_V2_ORACLE for row in rows)
    assert all(row["safety"]["database_touched"] is False for row in rows)
    assert all(row["safety"]["real_sleep_performed"] is False for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_sql_v2_pair_is_transport_distinct_and_rejects_unlisted_channel():
    specs = default_sql_v2_specs(target="http://127.0.0.1:8806")
    plain = next(spec for spec in specs if spec["pair"]["pair_id"] == "sql-pg14-parse_fault" and spec["pair"]["variant"] == "plain")
    encoded = next(spec for spec in specs if spec["pair"]["pair_id"] == "sql-pg14-parse_fault" and spec["pair"]["variant"] == "url_percent")
    assert plain["path"] != encoded["path"]
    try:
        validate_sql_v2_spec({"target": "http://127.0.0.1:8806", "mode": "real_sql", "source_id": "s", "lab_id": "l"})
    except ValueError:
        pass
    else:
        raise AssertionError("unlisted SQL v2 channel was accepted")
