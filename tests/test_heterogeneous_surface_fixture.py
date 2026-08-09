import asyncio
import socket
import threading

from app.heterogeneous_surface_fixture import (
    HETERO_SURFACE_ORACLE,
    HETERO_SURFACE_PORTS,
    HETERO_SURFACES,
    HeterogeneousSurfaceCollector,
    default_heterogeneous_surface_specs,
    heterogeneous_surface_source_sha256,
    make_heterogeneous_surface_fixture_server,
)


def _free_fixture_port() -> int:
    for port in HETERO_SURFACE_PORTS:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            probe.close()
        return port
    raise RuntimeError("no allow-listed heterogeneous fixture port is available")


def test_heterogeneous_fixture_keeps_only_html_attribute_positive():
    port = _free_fixture_port()
    target = f"http://127.0.0.1:{port}"
    server = make_heterogeneous_surface_fixture_server(port=port, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = asyncio.run(HeterogeneousSurfaceCollector(base_url=target, target_instance_id="hetero-test", source_hash=heterogeneous_surface_source_sha256()).collect_many(default_heterogeneous_surface_specs(target=target)))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(rows) == len(HETERO_SURFACES) * 2
    positives = [row for row in rows if row["rule_ir_result"]]
    assert len(positives) == 2
    assert all(row["semantic"]["surface_role"] == "html_attribute" for row in positives)
    assert all(row["semantic"]["expected_oracle"] == HETERO_SURFACE_ORACLE for row in rows)
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_heterogeneous_fixture_has_real_encoding_pairs():
    specs = default_heterogeneous_surface_specs(target="http://127.0.0.1:8800")
    plain = next(spec for spec in specs if spec["pair"]["pair_id"] == "hetero-pg12-xml_text" and spec["pair"]["variant"] == "plain")
    encoded = next(spec for spec in specs if spec["pair"]["pair_id"] == "hetero-pg12-xml_text" and spec["pair"]["variant"] == "url_percent")
    assert plain["path"] != encoded["path"]
