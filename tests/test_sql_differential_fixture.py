import asyncio
import threading
import time

import httpx
import pytest

from app.sql_differential_fixture import (
    SQL_FIXTURE_BASE_URL,
    SqlDifferentialCollector,
    default_sql_fixture_specs,
    make_sql_fixture_server,
    sql_fixture_source_sha256,
    validate_sql_fixture_spec,
)


def _start_server():
    server = make_sql_fixture_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            if httpx.get(f"{SQL_FIXTURE_BASE_URL}/query?mode=plain", timeout=0.2).status_code == 200:
                return server, thread
        except Exception:
            time.sleep(0.01)
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)
    raise RuntimeError("SQL fixture did not start")


def test_sql_specs_pair_safe_abstract_channels_only():
    specs = default_sql_fixture_specs()
    assert len(specs) == 15
    assert specs[0]["mode"] == "syntax_error"
    assert specs[1]["mode"] != specs[0]["mode"]
    assert "union" not in str(specs).casefold()
    assert validate_sql_fixture_spec(specs[1])["decoded_mode"] == "syntax_error"


def test_sql_collector_emits_ast_differential_without_db_or_sleep():
    server, thread = _start_server()
    try:
        rows = asyncio.run(
            SqlDifferentialCollector(target_instance_id="sql-test", source_hash=sql_fixture_source_sha256()).collect_many(
                default_sql_fixture_specs()
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert len(rows) == 15
    positive = [row for row in rows if row["rule_ir_result"]]
    assert len(positive) == 12
    assert any(row["oracle_projection"]["modality"] == "syntax_error" for row in rows)
    assert any(row["oracle_projection"]["modality"] == "bounded_timing" for row in rows)
    assert all(row["oracle_projection"]["database_touched"] is False for row in rows)
    assert all(row["oracle_projection"]["real_sleep_performed"] is False for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_sql_fixture_rejects_real_target_and_unlisted_mode():
    with pytest.raises(ValueError):
        validate_sql_fixture_spec({"target": "https://example.com", "mode": "plain", "source_id": "s", "lab_id": "l"})
    with pytest.raises(ValueError):
        validate_sql_fixture_spec({"mode": "real_sql", "source_id": "s", "lab_id": "l"})
