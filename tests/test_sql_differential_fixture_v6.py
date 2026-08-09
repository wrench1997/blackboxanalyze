import threading

from app.sql_differential_fixture_v6 import (
    V6_MODES,
    V6_VARIANTS,
    collect_sql_v6,
    make_sql_v6_fixture_server,
    run_v6_oracle,
    sql_v6_source_sha256,
)


def test_v6_is_independent_and_non_executing() -> None:
    assert sql_v6_source_sha256()
    baseline = run_v6_oracle("baseline", variant="obsidian")
    for mode in V6_MODES - {"baseline"}:
        row = run_v6_oracle(mode, variant="obsidian")
        assert row["oracle"] == "synthetic_sql_shape_differential_v6"
        assert row["candidate_signal"] is True
        assert row["interpreter_boundary"] is True
        assert row["execution"] == "not_run"
        assert row["database_touched"] is False
        assert row["real_sleep_performed"] is False
        assert row["baseline_shape_sha256"] == baseline["baseline_shape_sha256"]


def test_v6_get_post_shape_agreement_for_all_variants() -> None:
    for index, variant in enumerate(sorted(V6_VARIANTS)):
        port = 8840 + index
        server = make_sql_v6_fixture_server(port=port, variant=variant)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for method in ("GET", "POST"):
                for mode in ("branch", "syntax", "row", "timeout_shape", "error_redirect"):
                    row = collect_sql_v6(target=f"http://127.0.0.1:{port}", port=port, variant=variant, method=method, mode=mode, sample_id=f"{variant}-{method.lower()}-{mode}")
                    assert row["oracle_projection"]["interpreter_boundary"] is True
                    assert row["oracle_projection"]["candidate_signal"] is True
                    assert row["fresh_target"] is True
                    assert row["raw_payload_strings_stored"] is False
                    assert row["raw_response_bodies_stored"] is False
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

