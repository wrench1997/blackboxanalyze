import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg33_formal_candidate_is_reproducible_but_not_promoted():
    report = _load("pg_pk_33_formal_model_candidate_v1.json")
    checkpoint = ROOT / report["training"]["checkpoint"]
    assert report["catalog"]["sample_count"] == 84
    assert report["model"]["visible_projection_labels"] is False
    assert report["model"]["device"] in {"cuda", "cpu"}
    assert report["training"]["train_count"] == 12
    assert report["training"]["train_accuracy"] == 1.0
    assert checkpoint.exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["training"]["checkpoint_sha256"]
    assert len(report["cells"]) == 15
    assert all(cell["metrics_status"] == "completed" for cell in report["cells"])
    assert report["capability_gate"]["status"] == "no_proven_gain"
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["calibration"]["abstain_threshold"] == 1.0
