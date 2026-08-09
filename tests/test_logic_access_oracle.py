import asyncio
import threading

from app.logic_access_fixture import LogicAccessCollector, default_logic_access_fixture_specs, logic_access_fixture_source_sha256, make_logic_access_fixture_server
from app.logic_access_oracle import revalidate_logic_access_pair


def _rows():
    server = make_logic_access_fixture_server(port=8795, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return asyncio.run(LogicAccessCollector(target_instance_id="oracle-pg10", source_hash=logic_access_fixture_source_sha256()).collect_many(default_logic_access_fixture_specs(target="http://127.0.0.1:8795")))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_typed_oracle_accepts_access_pair_and_rejects_ordinary_200():
    rows = _rows()
    access = [row for row in rows if row["pair"]["pair_id"] == "logic-pg10-gate-truthy"]
    annotated = [dict(row, candidate_family="access_control") for row in access]
    result = revalidate_logic_access_pair(
        annotated,
        authorized_source_hash=logic_access_fixture_source_sha256(),
        expected_family="access_control",
        oracle_name="synthetic_authorization_boundary_v1",
        expected_signal="authorization_boundary_divergence",
    )
    assert result["accepted"] is True
    ordinary = [row for row in rows if row["pair"]["pair_id"] == "logic-pg10-ordinary-200-control"]
    denied = revalidate_logic_access_pair(
        [dict(row, candidate_family="access_control") for row in ordinary],
        authorized_source_hash=logic_access_fixture_source_sha256(),
        expected_family="access_control",
        oracle_name="synthetic_authorization_boundary_v1",
        expected_signal="authorization_boundary_divergence",
    )
    assert denied["accepted"] is False
    assert "oracle_contract_mismatch" in denied["reasons"]


def test_typed_oracle_accepts_business_boundary_pair():
    rows = _rows()
    coupon = [row for row in rows if row["pair"]["pair_id"] == "logic-pg10-coupon-boundary"]
    result = revalidate_logic_access_pair(
        [dict(row, candidate_family="logic") for row in coupon],
        authorized_source_hash=logic_access_fixture_source_sha256(),
        expected_family="logic",
        oracle_name="synthetic_business_invariant_v1",
        expected_signal="business_boundary_mismatch",
    )
    assert result["accepted"] is True
