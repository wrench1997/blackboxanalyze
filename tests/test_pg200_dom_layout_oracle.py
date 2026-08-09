import json

import pytest

from app.pg200_dom_layout_oracle import PG200_LAYOUTS, run_pg200_dom_layout_oracle


def test_pg200_fourth_layouts_are_typed_and_nonexecuting() -> None:
    for index, layout in enumerate(sorted(PG200_LAYOUTS)):
        marker = f"pg200-layout-{index}"
        result = run_pg200_dom_layout_oracle(
            f'<section><template data-sift-marker="{marker}">{marker}</template></section>',
            marker=marker,
            layout=layout,
        )
        assert result["schema_version"] == "pg200-dom-layout-oracle-v1"
        assert result["dom_change"] is True
        assert result["script_execution"] is False
        assert result["positive"] is False
        assert result["vulnerability_claim_allowed"] is False
        assert result["raw_markup_stored"] is False
        assert "<template" not in json.dumps(result, ensure_ascii=False)


def test_pg200_script_marker_is_not_a_dom_effect() -> None:
    result = run_pg200_dom_layout_oracle("<script>pg200-script-marker</script>", marker="pg200-script-marker", layout="svg_shell")
    assert result["dom_change"] is False
    assert result["script_execution"] is False
    assert result["positive_authority"] is False


def test_pg200_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="layout"):
        run_pg200_dom_layout_oracle("<main>pg200-test</main>", marker="pg200-test", layout="unknown")
