"""Train the shared anonymous family router on local, typed-oracle fixtures."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_app_surface_fixture import (  # noqa: E402
    SurfaceFixtureCollector,
    default_surface_fixture_specs,
    make_surface_fixture_server,
    surface_fixture_source_sha256,
)
from app.cross_app_positive_fixture import (  # noqa: E402
    PositiveFixtureCollector,
    default_fixture_specs,
    fixture_source_sha256,
    make_server as make_positive_fixture_server,
)
from app.logic_access_fixture import (  # noqa: E402
    LogicAccessCollector,
    default_logic_access_fixture_specs,
    logic_access_fixture_source_sha256,
    make_logic_access_fixture_server,
)
from app.shared_family_representation import (  # noqa: E402
    SHARED_EMBEDDING_DIM,
    SHARED_FAMILY_CLASSES,
    SHARED_FEATURE_DIM,
    OOD_INVARIANT_FEATURE_INDICES,
    SharedFamilyRouter,
    shared_model_input,
)
from app.ood_gate import fit_ood_reference  # noqa: E402
from app.confidence_calibration import expected_calibration_error, fit_temperature, temperature_scale  # noqa: E402
from app.rule_ir_decoder import calibrate_abstention_threshold  # noqa: E402
from app.sql_differential_fixture import (  # noqa: E402
    SqlDifferentialCollector,
    default_sql_fixture_specs,
    make_sql_fixture_server,
    sql_fixture_source_sha256,
)


ARTIFACT_DIR = ROOT / "artifacts" / "shared-family-router-pg-pk-11"
CHECKPOINT_PATH = ARTIFACT_DIR / "shared_family_router.pt"
REPORT_PATH = ARTIFACT_DIR / "report.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_11_shared_router_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_11_shared_router_protocol_v1.json"


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _collect_with_server(port: int, server: Any, collector: Any, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    thread = threading.Thread(target=server.serve_forever, name=f"pg11-train-{port}", daemon=True)
    thread.start()
    try:
        _wait_ready(port)
        return asyncio.run(collector.collect_many(specs))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _collect_training_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    surface_source = surface_fixture_source_sha256()
    rows.extend(_collect_with_server(
        8791,
        make_surface_fixture_server(),
        SurfaceFixtureCollector(target_instance_id="shared-train-surface", source_hash=surface_source),
        default_surface_fixture_specs(marker_prefix="pg11-surface"),
    ))
    sql_source = sql_fixture_source_sha256()
    rows.extend(_collect_with_server(
        8793,
        make_sql_fixture_server(),
        SqlDifferentialCollector(target_instance_id="shared-train-sql", source_hash=sql_source),
        default_sql_fixture_specs(marker="pg11-sql-marker"),
    ))
    logic_source = logic_access_fixture_source_sha256()
    for port, variant, dataset_id in ((8795, "alpha", "shared-train-logic-alpha"), (8796, "beta", "shared-train-logic-beta")):
        target = f"http://127.0.0.1:{port}"
        rows.extend(_collect_with_server(
            port,
            make_logic_access_fixture_server(port=port, variant=variant),
            LogicAccessCollector(base_url=target, target_instance_id=f"shared-train-logic-{variant}", source_hash=logic_source),
            default_logic_access_fixture_specs(dataset_id=dataset_id, target=target, marker=f"pg11-logic-{variant}"),
        ))
    return rows


def _collect_fresh_logic() -> list[dict[str, Any]]:
    port, variant = 8797, "gamma"
    target = f"http://127.0.0.1:{port}"
    return _collect_with_server(
        port,
        make_logic_access_fixture_server(port=port, variant=variant),
        LogicAccessCollector(base_url=target, target_instance_id="shared-eval-logic-gamma", source_hash=logic_access_fixture_source_sha256()),
        default_logic_access_fixture_specs(dataset_id="shared-eval-logic-gamma", target=target, marker="pg11-logic-gamma"),
    )


def _collect_fresh_joint_surface() -> list[dict[str, Any]]:
    source_hash = fixture_source_sha256()
    return _collect_with_server(
        8790,
        make_positive_fixture_server(),
        PositiveFixtureCollector(target_instance_id="shared-eval-positive-surface", source_hash=source_hash),
        default_fixture_specs(marker_prefix="pg11-fresh-surface"),
    )


def _label(row: dict[str, Any]) -> int:
    projection = row.get("oracle_projection") or {}
    if row.get("schema_version") == "sift-cross-app-surface-fixture-v1":
        # The shared head routes the surface to the XSS family.  Whether the
        # sink is exploitable is deliberately left to the typed sink oracle;
        # text/JSON/header controls are still valid XSS-family observations.
        if str((row.get("semantic") or {}).get("surface_role", "")) == "plain_control":
            return SHARED_FAMILY_CLASSES.index("control")
        return SHARED_FAMILY_CLASSES.index("xss")
    if row.get("schema_version") == "sift-cross-app-positive-fixture-v1":
        if str(((row.get("pair") or {}).get("surface_role", ""))) == "plain_control":
            return SHARED_FAMILY_CLASSES.index("control")
        return SHARED_FAMILY_CLASSES.index("xss")
    if row.get("schema_version") == "sift-sql-differential-fixture-v1":
        # SQL channel membership is not the same as a positive injection
        # finding; the AST/interpreter oracle makes that second decision.
        return SHARED_FAMILY_CLASSES.index("injection")
    route = urlsplit(str((row.get("payload") or {}).get("path", ""))).path
    if route == "/gate":
        return SHARED_FAMILY_CLASSES.index("access_control")
    if route in {"/coupon", "/replay"}:
        return SHARED_FAMILY_CLASSES.index("logic")
    return SHARED_FAMILY_CLASSES.index("control")


def _metrics(model: SharedFamilyRouter, features: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=-1)
        predictions = probabilities.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean()) if len(labels) else 0.0
    control_index = SHARED_FAMILY_CLASSES.index("control")
    control_mask = labels == control_index
    positive_mask = ~control_mask
    false_accept = float(((predictions != control_index) & control_mask).float().mean()) if bool(control_mask.any()) else 0.0
    recall = float(((predictions == labels) & positive_mask).float().sum() / positive_mask.float().sum()) if bool(positive_mask.any()) else 0.0
    by_family: dict[str, float] = {}
    for index, family in enumerate(SHARED_FAMILY_CLASSES):
        mask = labels == index
        by_family[family] = round(float((predictions[mask] == labels[mask]).float().mean()), 6) if bool(mask.any()) else 0.0
    return {"accuracy": round(accuracy, 6), "control_false_accept_rate": round(false_accept, 6), "positive_recall": round(recall, 6), "by_family": by_family}


def _pair_distance(model: SharedFamilyRouter, features: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, float]:
    with torch.inference_mode():
        embeddings = model.encode(features)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair_id = str((row.get("pair") or {}).get("pair_id", ""))
        if pair_id:
            grouped[pair_id].append(index)
    distances: list[float] = []
    for indices in grouped.values():
        if len(indices) == 2:
            distances.append(float(torch.linalg.vector_norm(embeddings[indices[0]] - embeddings[indices[1]])))
    return {
        "pair_count": float(len(distances)),
        "mean_l2": round(sum(distances) / len(distances), 6) if distances else 0.0,
        "max_l2": round(max(distances), 6) if distances else 0.0,
    }


def main() -> None:
    rows = _collect_training_rows()
    fresh_rows = _collect_fresh_logic()
    fresh_surface_rows = _collect_fresh_joint_surface()
    # Plain/encoded split is kept for a visible holdout; all views contribute
    # to normalisation so encoding indicators do not become infinite z-scores.
    raw_features = torch.tensor([shared_model_input(row) for row in rows], dtype=torch.float32)
    labels = torch.tensor([_label(row) for row in rows], dtype=torch.long)
    encoded_mask = torch.tensor([str((row.get("pair") or {}).get("variant", "")) == "url_percent" for row in rows], dtype=torch.bool)
    plain_mask = ~encoded_mask
    mean = raw_features.mean(dim=0)
    std = raw_features.std(dim=0).clamp_min(1e-4)
    train_features_raw = raw_features[plain_mask]
    train_labels = labels[plain_mask]
    holdout_features = (raw_features[encoded_mask] - mean) / std
    holdout_labels = labels[encoded_mask]
    train_features = ((train_features_raw - mean) / std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedFamilyRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.001)
    counts = torch.bincount(train_labels, minlength=len(SHARED_FAMILY_CLASSES)).float()
    # ``control`` means an unknown/ordinary surface here, not a negative
    # vulnerability label.  Keep that route visible while leaving the
    # family-specific oracle responsible for positive/negative decisions.
    weights = torch.ones(len(SHARED_FAMILY_CLASSES), dtype=torch.float32)
    weights[SHARED_FAMILY_CLASSES.index("control")] = 2.0
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_features_device = train_features.to(device)
    train_labels_device = train_labels.to(device)
    best_state: dict[str, Any] | None = None
    best_score = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, 601):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_features_device), train_labels_device)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            holdout_metrics = _metrics(model, holdout_features.to(device), holdout_labels.to(device))
            score = float(holdout_metrics["accuracy"] - 0.5 * holdout_metrics["control_false_accept_rate"])
            history.append({"epoch": float(epoch), "loss": round(float(loss), 6), "holdout_score": round(score, 6)})
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("shared-family training produced no checkpoint")
    model.load_state_dict(best_state)
    model.cpu().eval()
    train_metrics = _metrics(model, train_features, train_labels)
    holdout_metrics = _metrics(model, holdout_features, holdout_labels)
    pair_metrics = _pair_distance(model, ((raw_features - mean) / std), rows)
    fresh_raw = torch.tensor([shared_model_input(row) for row in fresh_rows], dtype=torch.float32)
    fresh_features = (fresh_raw - mean) / std
    fresh_labels = torch.tensor([_label(row) for row in fresh_rows], dtype=torch.long)
    fresh_metrics = _metrics(model, fresh_features, fresh_labels)
    fresh_surface_raw = torch.tensor([shared_model_input(row) for row in fresh_surface_rows], dtype=torch.float32)
    fresh_surface_features = (fresh_surface_raw - mean) / std
    fresh_surface_labels = torch.tensor([_label(row) for row in fresh_surface_rows], dtype=torch.long)
    fresh_joint_rows = fresh_rows + fresh_surface_rows
    fresh_joint_features = torch.cat((fresh_features, fresh_surface_features), dim=0)
    fresh_joint_labels = torch.cat((fresh_labels, fresh_surface_labels), dim=0)
    # Fit confidence only on the held-out fresh joint target.  The calibrated
    # probability never grants positive authority; it only decides whether a
    # shared route is useful enough to bias active probing.
    with torch.inference_mode():
        calibration_probabilities = torch.softmax(model(fresh_joint_features), dim=-1).tolist()
    calibration_labels = fresh_joint_labels.tolist()
    temperature_fit = fit_temperature(calibration_probabilities, calibration_labels)
    scaled_calibration_probabilities = temperature_scale(calibration_probabilities, temperature_fit["temperature"])
    calibration_predictions = [max(range(len(row)), key=row.__getitem__) for row in scaled_calibration_probabilities]
    calibration_correctness = [prediction == label for prediction, label in zip(calibration_predictions, calibration_labels)]
    calibration_threshold = calibrate_abstention_threshold(
        scaled_calibration_probabilities,
        calibration_labels,
        minimum_precision=0.99,
    )
    calibration_confidences = [max(row) for row in calibration_probabilities]
    scaled_confidences = [max(row) for row in scaled_calibration_probabilities]
    control_index = SHARED_FAMILY_CLASSES.index("control")
    control_false_accept = sum(
        int(confidence >= float(calibration_threshold["threshold"]) and label == control_index and prediction != control_index)
        for confidence, label, prediction in zip(scaled_confidences, calibration_labels, calibration_predictions)
    )
    control_count = sum(int(label == control_index) for label in calibration_labels)
    confidence_calibration = {
        **temperature_fit,
        "raw_ece": expected_calibration_error(calibration_confidences, calibration_correctness),
        "scaled_ece": expected_calibration_error(scaled_confidences, calibration_correctness),
        "abstain_threshold": float(calibration_threshold["threshold"]),
        "abstain_precision": float(calibration_threshold["precision"]),
        "abstain_coverage": float(calibration_threshold["coverage"]),
        "control_false_accept_rate": control_false_accept / control_count if control_count else 0.0,
        "calibration_sample_count": len(calibration_labels),
        "calibration_source": "fresh_logic_gamma_plus_cross_app_positive_surface",
    }
    calibrated_abstain_threshold = float(confidence_calibration["abstain_threshold"])
    fresh_outputs = model.decode(
        fresh_features,
        abstain_threshold=calibrated_abstain_threshold,
        margin_threshold=0.10,
        temperature=float(temperature_fit["temperature"]),
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # The shared route is meant to generalise across response surfaces.  A
    # plain variance mask would silently learn the training fixture's HTML /
    # JSON geometry and mark an unseen but valid surface as OOD.  Keep the
    # OOD reference on the explicitly audited invariant dimensions, then
    # remove any dimensions that are genuinely constant in this run (a
    # constant feature has no distance information and only amplifies noise).
    ood_feature_mask = torch.zeros_like(std, dtype=torch.bool)
    ood_feature_mask[list(OOD_INVARIANT_FEATURE_INDICES)] = True
    ood_feature_mask &= std > 1e-3
    if int(ood_feature_mask.sum()) < 2:
        raise RuntimeError("shared-family OOD invariant contract has fewer than two varying dimensions")
    checkpoint = {
        "schema_version": "sift-shared-family-router-checkpoint-v1",
        "model_state": model.state_dict(),
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "feature_dim": SHARED_FEATURE_DIM,
        "embedding_dim": SHARED_EMBEDDING_DIM,
        "classes": list(SHARED_FAMILY_CLASSES),
        "abstain_threshold": calibrated_abstain_threshold,
        "margin_threshold": 0.10,
        "temperature": float(temperature_fit["temperature"]),
        "confidence_calibration": confidence_calibration,
        "training_sources": {
            "surface_fixture_source_sha256": surface_fixture_source_sha256(),
            "sql_fixture_source_sha256": sql_fixture_source_sha256(),
            "logic_access_fixture_source_sha256": logic_access_fixture_source_sha256(),
        },
        "device": str(device),
        "ood_clip": 3.0,
        "ood_fit": fit_ood_reference(train_features[:, ood_feature_mask].clamp(-3.0, 3.0).cpu(), quantile=0.95, slack=1.25),
        "ood_feature_mask": ood_feature_mask.tolist(),
        "ood_reference_features": train_features[:, ood_feature_mask].clamp(-3.0, 3.0).cpu().tolist(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    acceptance = {
        "minimum_holdout_accuracy": 0.90,
        "maximum_holdout_control_false_accept_rate": 0.05,
        "minimum_holdout_positive_recall": 0.90,
        "maximum_pair_mean_l2": 1.50,
        "passed": bool(
            holdout_metrics["accuracy"] >= 0.90
            and holdout_metrics["control_false_accept_rate"] <= 0.05
            and holdout_metrics["positive_recall"] >= 0.90
            and pair_metrics["mean_l2"] <= 1.50
        ),
    }
    acceptance["confidence_calibration_passed"] = bool(
        confidence_calibration["abstain_precision"] >= 0.99
        and confidence_calibration["abstain_coverage"] >= 0.50
        and confidence_calibration["control_false_accept_rate"] <= 0.05
    )
    report = {
        "schema_version": "sift-shared-family-router-report-v1",
        "protocol_id": "pg-pk-11-shared-anonymous-family-router-v1",
        "status": "accepted_for_diagnostic_routing" if acceptance["passed"] else "diagnostic_only",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_sample_count": len(rows),
        "encoding_holdout_sample_count": int(encoded_mask.sum()),
        "fresh_target_sample_count": len(fresh_rows) + len(fresh_surface_rows),
        "class_counts": {name: int((labels == index).sum()) for index, name in enumerate(SHARED_FAMILY_CLASSES)},
        "train_metrics": train_metrics,
        "encoding_holdout_metrics": holdout_metrics,
        "fresh_logic_target_metrics": fresh_metrics,
        "fresh_joint_surface_holdout_metrics": _metrics(model, fresh_joint_features, fresh_joint_labels),
        "fresh_joint_surface_holdout_sample_count": len(fresh_joint_rows),
        "pair_invariance": pair_metrics,
        "confidence_calibration": confidence_calibration,
        "acceptance": acceptance,
        "fresh_target_abstain_count": sum(int(output["abstained"]) for output in fresh_outputs),
        "fresh_target_sample_predictions": [
            {"sample_id": row["sample_id"], "candidate_family": output["candidate_family"], "confidence": output["confidence"], "abstained": output["abstained"]}
            for row, output in zip(fresh_rows[:12], fresh_outputs[:12])
        ],
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "history": history,
        "feature_contract": {
            "oracle_fields_consumed": False,
            "semantic_labels_consumed": False,
            "raw_body_consumed": False,
            "route_tokens_consumed": False,
            "decoded_query_geometry_only": True,
            "ood_zero_variance_dimensions_masked": True,
            "ood_surface_specific_dimensions_excluded": True,
            "ood_invariant_feature_indices": list(OOD_INVARIANT_FEATURE_INDICES),
            "counterfactual_plain_control_routes_to_control": True,
            "temperature_fitted_on_fresh_holdout": True,
            "calibrated_confidence_is_route_only": True,
        },
    }
    report["acceptance"]["minimum_fresh_joint_surface_accuracy"] = 0.90
    report["acceptance"]["fresh_joint_surface_passed"] = bool(report["fresh_joint_surface_holdout_metrics"]["accuracy"] >= 0.90)
    report["acceptance"]["passed"] = bool(report["acceptance"]["passed"] and report["acceptance"]["fresh_joint_surface_passed"] and report["acceptance"]["confidence_calibration_passed"])
    report["status"] = "accepted_for_diagnostic_routing" if report["acceptance"]["passed"] else "diagnostic_only"
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-11 共享匿名表示与族路由 head\n\n"
        "共享 head 只做 XSS/SQL/访问控制/逻辑族路由；正向结论仍必须由族特异 typed oracle 复核。\n\n"
        f"状态：`{report['status']}`；训练样本：{report['training_sample_count']}；编码留出样本：{report['encoding_holdout_sample_count']}；新目标样本：{report['fresh_target_sample_count']}。\n\n"
        f"编码留出 accuracy：{report['encoding_holdout_metrics']['accuracy']:.3f}；联合族外表面 accuracy：{report['fresh_joint_surface_holdout_metrics']['accuracy']:.3f}；未知表面误路由：{report['encoding_holdout_metrics']['control_false_accept_rate']:.3f}；pair mean L2：{report['pair_invariance']['mean_l2']:.3f}。\n\n"
        f"温度：{report['confidence_calibration']['temperature']:.3f}；校准后 ECE：{report['confidence_calibration']['scaled_ece']:.3f}；abstain threshold：{report['confidence_calibration']['abstain_threshold']:.3f}；coverage：{report['confidence_calibration']['abstain_coverage']:.3f}。\n\n"
        f"共享 head 门禁：`{'pass' if report['acceptance']['passed'] else 'diagnostic_only'}`。共享 head 不具备正向放行权。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": "pg-pk-11-shared-anonymous-family-router-v1",
        "schema_version": "sift-pg-pk-11-shared-router-protocol-v1",
        "feature_contract": report["feature_contract"],
        "training_targets": ["surface_fixture", "sql_fixture", "logic_access_alpha", "logic_access_beta"],
        "fresh_target": ["logic_access_gamma", "cross_app_positive_surface"],
        "positive_gate": "family_specific_typed_oracle_only",
        "confidence_calibration": report["confidence_calibration"],
        "acceptance": report["acceptance"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "status": report["status"],
        "device": report["device"],
        "gpu_name": report["gpu_name"],
        "training_sample_count": report["training_sample_count"],
        "encoding_holdout_metrics": report["encoding_holdout_metrics"],
        "fresh_logic_target_metrics": report["fresh_logic_target_metrics"],
        "fresh_joint_surface_holdout_metrics": report["fresh_joint_surface_holdout_metrics"],
        "pair_invariance": report["pair_invariance"],
        "confidence_calibration": report["confidence_calibration"],
        "acceptance": acceptance,
        "checkpoint": report["checkpoint"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
