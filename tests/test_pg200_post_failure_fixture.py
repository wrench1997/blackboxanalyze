import threading

from app.pg200_post_failure_fixture import (
    PG200_POST_MODES,
    collect_post_failure,
    make_post_failure_server,
)


def test_pg200_unseen_post_failure_shapes_are_bounded() -> None:
    server = make_post_failure_server(port=8850)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for mode in PG200_POST_MODES:
            row = collect_post_failure(target="http://127.0.0.1:8850", port=8850, mode=mode, sample_id=f"post-{mode}")
            assert row["method"] == "POST"
            assert row["failure_signature"]["typed_available"] is False
            assert row["failure_signature"]["positive_authority"] is False
            assert row["fresh_target"] is True
            assert row["raw_payload_strings_stored"] is False
            assert row["raw_response_bodies_stored"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

