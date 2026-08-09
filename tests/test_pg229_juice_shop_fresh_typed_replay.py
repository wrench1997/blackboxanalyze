import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg229_fresh_seed_reference_and_model_self_error_are_visible() -> None:
    report = json.loads((ROOT / "research" / "pg229_juice_shop_fresh_typed_replay_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg229_juice_shop_fresh_typed_replay_dataset_v1.json").read_text(encoding="utf-8-sig"))
    counts = report["counts"]
    assert report["status"] == "completed_fresh_juice_shop_ai_selected_typed_replay"
    assert counts["fresh_container_count"] == 2
    assert counts["candidate_episode_count"] == 28
    assert counts["reference_pair_count"] == 28
    assert counts["typed_effect_confirmed_count"] == 2
    assert counts["model_self_error_count"] == 2
    assert counts["model_gate_correction_count"] == 2
    assert counts["false_positive_count"] == 0
    assert report["promotion"]["payload_grounded_catalog_promotion_allowed"] is False
    assert dataset["contract"]["evaluator_state_hidden_from_agent"] is True
    assert dataset["contract"]["path_surface_is_not_payload_grounded"] is True
    for row in report["results"]:
        assert row["raw_payload_strings_stored"] is False
        assert row["raw_response_bodies_stored"] is False
        if row["typed_effect_confirmed"]:
            assert row["candidate_reference_agreement"] is True
            assert row["negative_clean"] is True
            assert row["model_self_error_detected"] is True
            assert row["model_gate_corrected_diagnosis"] == "confirmed_local_effect"

