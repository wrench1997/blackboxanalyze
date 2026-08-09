"""Train the small SQL-channel family head on an abstract local fixture."""

from __future__ import annotations

import asyncio
import copy
import json
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rule_ir_decoder import calibrate_abstention_threshold  # noqa: E402
from app.sql_channel_decoder import SQL_CHANNEL_FEATURE_DIM, SqlChannelDecoder, sql_channel_feature_vector  # noqa: E402
from app.sql_differential_fixture import (  # noqa: E402
    SQL_FIXTURE_BASE_URL,
    SqlDifferentialCollector,
    default_sql_fixture_specs,
    make_sql_fixture_server,
    sql_fixture_source_sha256,
)
from app.sql_differential_fixture_v2 import (  # noqa: E402
    SQL_V2_BASE_URL,
    SqlV2Collector,
    default_sql_v2_specs,
    make_sql_v2_fixture_server,
    sql_v2_source_sha256,
)
from app.sql_differential_fixture_v3 import (  # noqa: E402
    SQL_V3_BASE_URL,
    SqlV3Collector,
    default_sql_v3_specs,
    make_sql_v3_fixture_server,
    sql_v3_source_sha256,
)
from app.logic_access_fixture import (  # noqa: E402
    LogicAccessCollector,
    default_logic_access_fixture_specs,
    logic_access_fixture_source_sha256,
    make_logic_access_fixture_server,
)


PROTOCOL_ID = "pg-pk-09-sql-channel-decoder-v1"
SEED = 20260911
OUTPUT_DIR = ROOT / "artifacts" / "sql-channel-decoder-pg-pk-09"
CHECKPOINT = OUTPUT_DIR / "sql_channel_decoder.pt"
REPORT = OUTPUT_DIR / "report.json"
PROTOCOL = ROOT / "research" / "pg_pk_09_sql_differential_protocol_v1.json"


def _collect_target(target_id: str) -> list[dict[str, Any]]:
    server = make_sql_fixture_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{SQL_FIXTURE_BASE_URL}/query?mode=plain", timeout=0.3).status_code == 200:
                    break
            except Exception:
                time.sleep(0.02)
        return asyncio.run(
            SqlDifferentialCollector(target_instance_id=target_id, source_hash=sql_fixture_source_sha256()).collect_many(
                default_sql_fixture_specs()
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([sql_channel_feature_vector(row) for row in rows], dtype=torch.float32)


def _collect_v2_counterfactual_controls() -> list[dict[str, Any]]:
    """Collect only v2 safe controls; v2 positive channels stay held out."""

    server = make_sql_v2_fixture_server(port=8806, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{SQL_V2_BASE_URL}/lookup?channel=baseline", timeout=0.3).status_code == 200:
                    break
            except Exception:
                time.sleep(0.02)
        specs = [
            spec for spec in default_sql_v2_specs(target=SQL_V2_BASE_URL)
            if str(spec.get("mode", "")) in {"value_only", "baseline"}
            or str((spec.get("pair") or {}).get("pair_id", "")) == "sql-pg14-safe"
        ]
        return asyncio.run(
            SqlV2Collector(
                base_url=SQL_V2_BASE_URL,
                target_instance_id="sql-v2-counterfactual-control",
                source_hash=sql_v2_source_sha256(),
            ).collect_many(specs)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _collect_v3_counterfactual_controls() -> list[dict[str, Any]]:
    """Collect only v3 safe controls; all v3 positive channels stay held out."""

    server = make_sql_v3_fixture_server(port=8809, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{SQL_V3_BASE_URL}/search?q=baseline", timeout=0.3).status_code == 200:
                    break
            except Exception:
                time.sleep(0.02)
        specs = [
            spec for spec in default_sql_v3_specs(target=SQL_V3_BASE_URL)
            if str(spec.get("mode", "")) in {"literal_only", "baseline"}
            or str((spec.get("pair") or {}).get("pair_id", "")) == "sql-pg15-safe"
        ]
        return asyncio.run(
            SqlV3Collector(
                base_url=SQL_V3_BASE_URL,
                target_instance_id="sql-v3-counterfactual-control",
                source_hash=sql_v3_source_sha256(),
            ).collect_many(specs)
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _collect_cross_family_negative_controls() -> list[dict[str, Any]]:
    """Use an older logic/access source as SQL-family negatives.

    These rows are not globally clean: some are real logic/access positives.
    They are nevertheless strict negatives for the SQL family.  The v2
    logic/access source remains held out for the cross-family guard.
    """

    server = make_logic_access_fixture_server(port=8795, variant="alpha")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if httpx.get("http://127.0.0.1:8795/health?ok=1", timeout=0.3).status_code == 200:
                    break
            except Exception:
                time.sleep(0.02)
        target = "http://127.0.0.1:8795"
        rows = asyncio.run(
            LogicAccessCollector(
                base_url=target,
                target_instance_id="logic-v1-sql-negative",
                source_hash=logic_access_fixture_source_sha256(),
            ).collect_many(default_logic_access_fixture_specs(dataset_id="logic-v1-sql-negative", target=target, marker="sql-cross-family"))
        )
        return [dict(row, _sql_label=0) for row in rows]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _normalise(train: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (values - mean) / std, mean, std


@torch.inference_mode()
def _evaluate(model: SqlChannelDecoder, features: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], device: torch.device, threshold: float) -> dict[str, Any]:
    probabilities = torch.softmax(model(features.to(device)), dim=-1).cpu()
    predicted = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predicted.eq(labels)
    accepted = confidence >= threshold
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "accepted": 0, "false_accept": 0})
    for row, matched, is_accepted, prediction in zip(rows, correct, accepted, predicted):
        expected = "injection" if bool(row.get("rule_ir_result")) else "control"
        stats = by_class[expected]
        stats["correct"] += int(matched)
        stats["total"] += 1
        stats["accepted"] += int(is_accepted)
        stats["false_accept"] += int(is_accepted and expected == "control" and int(prediction) == 1)
    return {
        "accuracy": round(float(correct.float().mean()), 6),
        "total": len(labels),
        "coverage": round(float(accepted.float().mean()), 6),
        "abstain_rate": round(float((~accepted).float().mean()), 6),
        "accepted_accuracy": round(float(correct[accepted].float().mean()), 6) if bool(accepted.any()) else None,
        "by_class": {key: value for key, value in sorted(by_class.items())},
    }


def main() -> None:
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = _collect_target("sql-train-target")
    v2_counterfactual_rows = _collect_v2_counterfactual_controls()
    v3_counterfactual_rows = _collect_v3_counterfactual_controls()
    cross_family_negative_rows = _collect_cross_family_negative_controls()
    train_rows.extend(v2_counterfactual_rows)
    train_rows.extend(v3_counterfactual_rows)
    train_rows.extend(cross_family_negative_rows)
    calibration_rows = _collect_target("sql-calibration-target")
    fresh_rows = _collect_target("sql-fresh-target")
    train_raw = _features(train_rows)
    calibration_raw = _features(calibration_rows)
    train_features, mean, std = _normalise(train_raw, train_raw)
    calibration_features = (calibration_raw - mean) / std
    labels = torch.tensor([int(row.get("_sql_label", 1 if row.get("rule_ir_result") else 0)) for row in train_rows], dtype=torch.long)
    calibration_labels = torch.tensor([1 if row.get("rule_ir_result") else 0 for row in calibration_rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    model = SqlChannelDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    for _epoch in range(1, 121):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_features.to(device)), labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            accuracy = float(model(calibration_features.to(device)).argmax(dim=-1).eq(calibration_labels.to(device)).float().mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.inference_mode():
        calibration_probabilities = torch.softmax(model(calibration_features.to(device)), dim=-1).cpu().tolist()
    calibration = calibrate_abstention_threshold(calibration_probabilities, calibration_labels.tolist(), minimum_precision=0.99)
    threshold = float(calibration["threshold"])
    fresh_features = (_features(fresh_rows) - mean) / std
    fresh_labels = torch.tensor([1 if row.get("rule_ir_result") else 0 for row in fresh_rows], dtype=torch.long)
    validation = _evaluate(model, calibration_features, calibration_labels, calibration_rows, device, threshold)
    fresh = _evaluate(model, fresh_features, fresh_labels, fresh_rows, device, threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sift-sql-channel-decoder-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": SQL_CHANNEL_FEATURE_DIM,
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "abstain_threshold": threshold,
        "device_at_training": str(device),
        "seed": SEED,
    }, CHECKPOINT)
    acceptance = {
        "fresh_accuracy_min": 0.90,
        "fresh_accuracy": fresh["accuracy"],
        "control_false_accept_rate_max": 0.05,
        "control_false_accept_rate": fresh["by_class"].get("control", {}).get("false_accept", 0) / max(fresh["by_class"].get("control", {}).get("total", 0), 1),
        "passed": bool(
            fresh["accuracy"] >= 0.90
            and fresh["by_class"].get("control", {}).get("false_accept", 0) / max(fresh["by_class"].get("control", {}).get("total", 0), 1) <= 0.05
        ),
        "failure_action": "diagnostic_only_and_abstain",
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-sql-channel-decoder-report-v1",
        "status": "accepted_for_oracle_revalidation" if acceptance["passed"] else "diagnostic_only",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        "target_instances": ["sql-train-target", "sql-calibration-target", "sql-fresh-target"],
        "training": {
            "examples": len(train_rows),
            "oracle_fields_excluded": True,
            "source_hash": sql_fixture_source_sha256(),
            "counterfactual_control_examples": len(v2_counterfactual_rows),
            "counterfactual_control_source_hash": sql_v2_source_sha256(),
            "positive_v2_channels_held_out": True,
            "v3_counterfactual_control_examples": len(v3_counterfactual_rows),
            "v3_counterfactual_control_source_hash": sql_v3_source_sha256(),
            "positive_v3_channels_held_out": True,
            "cross_family_negative_examples": len(cross_family_negative_rows),
            "cross_family_negative_source_hash": logic_access_fixture_source_sha256(),
            "cross_family_positive_logic_rows_are_sql_negative": True,
        },
        "calibration": calibration,
        "validation": validation,
        "fresh": fresh,
        "acceptance": acceptance,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "device": report["device"], "fresh_accuracy": fresh["accuracy"], "control_false_accept_rate": acceptance["control_false_accept_rate"], "checkpoint": report["checkpoint"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
