from app.safe_marker import fresh_marker, marker_sha256


def test_fresh_markers_are_inert_and_unique():
    first = fresh_marker()
    second = fresh_marker()
    assert first != second
    assert first.startswith("PG25XSS_")
    assert len(marker_sha256(first)) == 64
