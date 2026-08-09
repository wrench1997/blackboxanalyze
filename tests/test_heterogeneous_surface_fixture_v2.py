import asyncio
import threading

from app.heterogeneous_surface_fixture_v2 import (
    V2_ORACLE,
    V2_SURFACES,
    HeterogeneousSurfaceV2Collector,
    default_heterogeneous_surface_v2_specs,
    heterogeneous_surface_v2_source_sha256,
    make_heterogeneous_surface_v2_fixture_server,
)


def test_heterogeneous_v2_is_independent_and_keeps_attribute_positive():
    server = make_heterogeneous_surface_v2_fixture_server(port=8803, variant="beta")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(
            HeterogeneousSurfaceV2Collector(
                base_url="http://127.0.0.1:8803",
                target_instance_id="hetero-v2-test",
                source_hash=heterogeneous_surface_v2_source_sha256(),
            ).collect_many(default_heterogeneous_surface_v2_specs(target="http://127.0.0.1:8803"))
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(rows) == len(V2_SURFACES) * 2
    positives = [row for row in rows if row["rule_ir_result"]]
    assert len(positives) == 2
    assert all(row["semantic"]["surface_role"] == "html_attribute" for row in positives)
    assert all(row["semantic"]["expected_oracle"] == V2_ORACLE for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_heterogeneous_v2_encoding_pair_changes_transport():
    specs = default_heterogeneous_surface_v2_specs(target="http://127.0.0.1:8803")
    plain = next(spec for spec in specs if spec["pair"]["pair_id"] == "hetero-pg13-xml_text" and spec["pair"]["variant"] == "plain")
    encoded = next(spec for spec in specs if spec["pair"]["pair_id"] == "hetero-pg13-xml_text" and spec["pair"]["variant"] == "url_percent")
    assert plain["path"] != encoded["path"]
    assert heterogeneous_surface_v2_source_sha256() != ""
