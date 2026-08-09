"""Train the small logic/access Rule IR head on local abstract fixtures."""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.logic_access_decoder import (  # noqa: E402
    LOGIC_ACCESS_CLASSES,
    LOGIC_ACCESS_FEATURE_DIM,
    LogicAccessDecoder,
    logic_access_feature_vector,
    logic_access_model_feature_vector,
)
from app.confidence_calibration import expected_calibration_error, fit_temperature, temperature_scale  # noqa: E402
from app.rule_ir_decoder import calibrate_abstention_threshold  # noqa: E402
from app.logic_access_fixture import (  # noqa: E402
    LogicAccessCollector,
    default_logic_access_fixture_specs,
    logic_access_fixture_source_sha256,
    make_logic_access_fixture_server,
)


ARTIFACT_DIR = ROOT / "artifacts" / "logic-access-decoder-pg-pk-10"
CHECKPOINT_PATH = ARTIFACT_DIR / "logic_access_decoder.pt"
LEGACY_CHECKPOINT_PATH = ARTIFACT_DIR / "logic_access_decoder_pre_generic_features_v1.pt"
REPORT_PATH = ARTIFACT_DIR / "report.json"
TRAIN_TARGETS = ((8795, "alpha", "logic-access-train-alpha"), (8796, "beta", "logic-access-train-beta"))


def _serve_collect(port: int, variant: str, dataset_id: str) -> list[dict[str, Any]]:
    server = make_logic_access_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg10-train-{port}", daemon=True)
    thread.start()
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                    break
            except OSError:
                threading.Event().wait(0.02)
        # On Windows the first accepted readiness socket can be handed to the
        # request worker just after the listener becomes visible.  Give that
        # worker a bounded hand-off window before opening the async client.
        threading.Event().wait(0.10)
        target = f"http://127.0.0.1:{port}"
        return asyncio.run(
            LogicAccessCollector(base_url=target, target_instance_id=f"train-{variant}-{port}", source_hash=logic_access_fixture_source_sha256()).collect_many(
                default_logic_access_fixture_specs(dataset_id=dataset_id, target=target, marker=f"pg10-{variant}-marker")
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _label(row: dict[str, Any]) -> int:
    projection = row.get("oracle_projection") or {}
    if not bool(projection.get("positive")):
        return LOGIC_ACCESS_CLASSES.index("control")
    family = str(projection.get("oracle_name", ""))
    if family == "synthetic_authorization_boundary_v1":
        return LOGIC_ACCESS_CLASSES.index("access_control")
    if family in {"synthetic_business_invariant_v1", "synthetic_history_binding_v1"}:
        return LOGIC_ACCESS_CLASSES.index("logic")
    return LOGIC_ACCESS_CLASSES.index("control")


def _metrics(model: LogicAccessDecoder, features: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=-1)
        predictions = probabilities.argmax(dim=-1)
        confidence = probabilities.max(dim=-1).values
    accuracy = float((predictions == labels).float().mean()) if len(labels) else 0.0
    controls = labels == LOGIC_ACCESS_CLASSES.index("control")
    non_controls = ~controls
    control_false_accept = float(((predictions != LOGIC_ACCESS_CLASSES.index("control")) & controls).float().mean()) if bool(controls.any()) else 0.0
    positive_recall = float(((predictions == labels) & non_controls).float().sum() / non_controls.float().sum()) if bool(non_controls.any()) else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "control_false_accept_rate": round(control_false_accept, 6),
        "positive_recall": round(positive_recall, 6),
        "mean_confidence": round(float(confidence.mean()), 6) if len(confidence) else 0.0,
    }


def main() -> None:
    random.seed(20260802)
    torch.manual_seed(20260802)
    rows: list[dict[str, Any]] = []
    for port, variant, dataset_id in TRAIN_TARGETS:
        rows.extend(_serve_collect(port, variant, dataset_id))
    features = torch.tensor([logic_access_model_feature_vector(row) for row in rows], dtype=torch.float32)
    labels = torch.tensor([_label(row) for row in rows], dtype=torch.long)
    # Keep one encoded view per pair in validation so the head cannot simply
    # memorise a single transport representation.
    validation_mask = torch.tensor([
        str((row.get("pair") or {}).get("variant", "")) == "url_percent" for row in rows
    ], dtype=torch.bool)
    train_mask = ~validation_mask
    # Fit normalisation on both transport views.  Computing it on plain-only
    # rows would make the URL-percent indicator a zero-variance feature and
    # explode it at inference, which is exactly the encoding shortcut this
    # experiment is meant to remove.
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-4)
    train_features = (features[train_mask] - mean) / std
    validation_features = (features[validation_mask] - mean) / std
    train_labels = labels[train_mask]
    validation_labels = labels[validation_mask]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogicAccessDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.0008)
    # The control class is larger; balanced weights stop the head learning the
    # unsafe shortcut “everything is an ordinary 200 response”.
    counts = torch.bincount(train_labels, minlength=len(LOGIC_ACCESS_CLASSES)).float()
    weights = (counts.sum() / counts.clamp_min(1.0)).to(device)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    best_state: dict[str, Any] | None = None
    best_score = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_features), train_labels)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            val_metrics = _metrics(model, validation_features.to(device), validation_labels.to(device))
            score = float(val_metrics["accuracy"] - 0.5 * val_metrics["control_false_accept_rate"])
            history.append({"epoch": float(epoch), "loss": round(float(loss), 6), "validation_score": round(score, 6)})
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("logic/access training produced no checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    train_metrics = _metrics(model, ((features[train_mask] - mean) / std), labels[train_mask])
    validation_metrics = _metrics(model, ((features[validation_mask] - mean) / std), labels[validation_mask])
    with torch.inference_mode():
        validation_probabilities = torch.softmax(model(((features[validation_mask] - mean) / std)), dim=-1).tolist()
    validation_labels_list = validation_labels.tolist()
    temperature_fit = fit_temperature(validation_probabilities, validation_labels_list)
    scaled_validation_probabilities = temperature_scale(validation_probabilities, temperature_fit["temperature"])
    calibration_predictions = [max(range(len(row)), key=row.__getitem__) for row in scaled_validation_probabilities]
    calibration_correctness = [prediction == label for prediction, label in zip(calibration_predictions, validation_labels_list)]
    abstain_calibration = calibrate_abstention_threshold(scaled_validation_probabilities, validation_labels_list, minimum_precision=0.99)
    confidence_calibration = {
        **temperature_fit,
        "raw_ece": expected_calibration_error([max(row) for row in validation_probabilities], calibration_correctness),
        "scaled_ece": expected_calibration_error([max(row) for row in scaled_validation_probabilities], calibration_correctness),
        "abstain_threshold": float(abstain_calibration["threshold"]),
        "abstain_precision": float(abstain_calibration["precision"]),
        "abstain_coverage": float(abstain_calibration["coverage"]),
        "calibration_sample_count": len(validation_labels_list),
        "calibration_source": "v1_url_percent_encoding_holdout",
        "calibrated_confidence_is_route_only": True,
    }
    calibrated_threshold = float(confidence_calibration["abstain_threshold"])
    raw_outputs = model.decode((features - mean) / std, abstain_threshold=calibrated_threshold, margin_threshold=0.10, temperature=float(temperature_fit["temperature"]))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists() and not LEGACY_CHECKPOINT_PATH.exists():
        shutil.copy2(CHECKPOINT_PATH, LEGACY_CHECKPOINT_PATH)
    checkpoint = {
        "schema_version": "sift-logic-access-decoder-checkpoint-v1",
        "model_state": model.state_dict(),
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "abstain_threshold": calibrated_threshold,
        "margin_threshold": 0.10,
        "temperature": float(temperature_fit["temperature"]),
        "confidence_calibration": confidence_calibration,
        "classes": list(LOGIC_ACCESS_CLASSES),
        "feature_dim": LOGIC_ACCESS_FEATURE_DIM,
        "training_target_ids": [f"train-{variant}-{port}" for port, variant, _ in TRAIN_TARGETS],
        "source_hash": logic_access_fixture_source_sha256(),
        "device": str(device),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    report = {
        "schema_version": "sift-logic-access-decoder-report-v1",
        "protocol_id": "pg-pk-10-logic-access-head-training-v1",
        "status": "accepted_for_oracle_revalidation" if validation_metrics["control_false_accept_rate"] == 0.0 and validation_metrics["positive_recall"] >= 0.90 else "diagnostic_only",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sample_count": len(rows),
        "class_counts": {name: int((labels == index).sum()) for index, name in enumerate(LOGIC_ACCESS_CLASSES)},
        "train_metrics": train_metrics,
        "encoding_holdout_metrics": validation_metrics,
        "confidence_calibration": confidence_calibration,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "source_hash": logic_access_fixture_source_sha256(),
        "history": history,
        "visible_projection_excludes_oracle": True,
        "feature_contract": "v2_generic_value_boundaries_added_for_surface_holdout_and_v1_surface_shortcuts_zeroed",
        "legacy_checkpoint": str(LEGACY_CHECKPOINT_PATH.relative_to(ROOT)) if LEGACY_CHECKPOINT_PATH.exists() else None,
        "sample_predictions": [
            {
                "sample_id": rows[index]["sample_id"],
                "candidate_family": output["candidate_family"],
                "confidence": output["confidence"],
                "abstained": output["abstained"],
            }
            for index, output in enumerate(raw_outputs[:12])
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = {key: report[key] for key in ("protocol_id", "status", "device", "gpu_name", "sample_count", "class_counts", "train_metrics", "encoding_holdout_metrics", "checkpoint")}
    output["report"] = str(REPORT_PATH.relative_to(ROOT))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
