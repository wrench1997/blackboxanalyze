import asyncio
import threading

from app.sql_differential_fixture_v3 import (
    SQL_V3_ORACLE,
    SqlV3Collector,
    default_sql_v3_specs,
    make_sql_v3_fixture_server,
    sql_v3_source_sha256,
    validate_sql_v3_spec,
)


def test_sql_v3_uses_independent_search_surface_and_same_bounded_oracle():
    server = make_sql_v3_fixture_server(port=8809, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(
            SqlV3Collector(
                base_url="http://127.0.0.1:8809",
                target_instance_id="sql-v3-test",
                source_hash=sql_v3_source_sha256(),
            ).collect_many(default_sql_v3_specs(target="http://127.0.0.1:8809"))
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(rows) == 15
    assert sum(bool(row["rule_ir_result"]) for row in rows) == 12
    assert all(row["semantic"]["expected_oracle"] == SQL_V3_ORACLE for row in rows)
    assert all(row["safety"]["database_touched"] is False for row in rows)
    assert all(row["safety"]["real_sleep_performed"] is False for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_sql_v3_pair_changes_path_and_rejects_unlisted_channel():
    specs = default_sql_v3_specs(target="http://127.0.0.1:8809")
    plain = next(spec for spec in specs if spec["pair"]["pair_id"] == "sql-pg15-parse_issue" and spec["pair"]["variant"] == "plain")
    encoded = next(spec for spec in specs if spec["pair"]["pair_id"] == "sql-pg15-parse_issue" and spec["pair"]["variant"] == "url_percent")
    assert plain["path"] != encoded["path"]
    try:
        validate_sql_v3_spec({"target": "http://127.0.0.1:8809", "mode": "real_sql", "source_id": "s", "lab_id": "l"})
    except ValueError:
        pass
    else:
        raise AssertionError("unlisted SQL v3 channel was accepted")

