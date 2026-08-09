"""Run PG-PK-06: strict OOD plus explicit positive-oracle revalidation.

The fixture is intentionally not added to the decoder training set.  The run
therefore demonstrates two separate decisions: the global OOD gate abstains,
while a pinned inert oracle can revalidate a pair after the model proposes the
same family for both encodings.  No script is executed and no raw body is
persisted.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.catalog_rule_decoder import CATALOG_DECODER_FAMILIES, catalog_feature_vector, catalog_visible_trace  # noqa: E402
from app.cross_app_positive_fixture import (  # noqa: E402
    FIXTURE_BASE_URL,
    FIXTURE_ORACLE,
    PositiveFixtureCollector,
    default_fixture_specs,
    fixture_source_sha256,
    make_server,
)
from app.ood_gate import fit_ood_reference, nearest_reference_distances, ood_flags  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.oracle_revalidation import revalidate_positive_pair  # noqa: E402
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402
from run_juice_shop_cross_app_shadow import _load_model  # noqa: E402


PROTOCOL_ID = "pg-pk-06-positive-oracle-revalidation-v1"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_06_positive_oracle_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_06_positive_oracle_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_06_positive_oracle_protocol_v1.json"


def _wait_for_fixture() -> None:
    deadline = time.monotonic() + 5.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{FIXTURE_BASE_URL}/plain", timeout=0.5)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - only a startup race
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"positive fixture did not start: {last_error}")


def _training_reference(checkpoint: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    rows = flatten_catalog(load_catalog(ROOT / "research" / "pikachu_paired_catalog_v1.json"))
    rows = [
        row for row in rows
        if row["pair"]["variant"] in {"plain", "url_percent"}
        and row["pair"]["surface_role"] in {"reflected_get", "sqli_str", "sqli_search"}
    ]
    raw = torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    reference = (raw - mean) / std
    return reference, fit_ood_reference(reference)


def _predict(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    ood_fit: dict[str, Any],
    ood_reference: torch.Tensor,
) -> list[dict[str, Any]]:
    raw = torch.tensor([catalog_feature_vector(row) for row in records], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    features = (raw - mean) / std
    distances = nearest_reference_distances(features, ood_reference)
    flags = ood_flags(distances, ood_fit)
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=-1).tolist()
    outputs: list[dict[str, Any]] = []
    for index, (record, values) in enumerate(zip(records, probabilities)):
        order = sorted(range(len(values)), key=lambda item: values[item], reverse=True)
        candidate_index = order[0]
        second_index = order[1]
        family = CATALOG_DECODER_FAMILIES[candidate_index]
        probability = float(values[candidate_index])
        margin = float(values[candidate_index] - values[second_index])
        outputs.append({
            "sample_id": record["sample_id"],
            "pair_id": record.get("pair", {}).get("pair_id"),
            "variant": record.get("pair", {}).get("variant"),
            "surface": record["semantic"]["surface"],
            "candidate_family": family,
            "model_probability": round(probability, 6),
            "margin": round(margin, 6),
            "ood_distance": round(float(distances[index]), 6),
            "ood": bool(flags[index]),
            "model_only_accepted": bool(probability >= 0.45 and margin >= 0.10),
            "strict_ood_action": "abstain" if flags[index] else "continue_to_oracle",
            "oracle_revalidated": False,
        })
    return outputs


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-PK-06 跨应用正向 oracle 复核",
        "",
        "本轮使用仓库内短生命周期 fixture；它只把 canary 放入 HTML 属性并进行 HTML 转义，不执行脚本。fixture 不进入训练集。严格 OOD 层保持 abstain，只有显式绑定的、带源码哈希和证据哈希的正向 oracle 才能复核成对样本。",
        "",
        f"样本：{report['sample_count']}；严格 OOD abstain：{report['strict_ood_abstain_count']}；模型-only 接受：{report['model_only_accepted_count']}；正向 oracle 复核接受样本：{report['oracle_revalidated_sample_count']}（pair：{report['oracle_revalidated_pair_count']}）。",
        f"长期记忆晋升试探：`{report['memory_promotion_probe']['status']}`；原因：{', '.join(report['memory_promotion_probe']['reasons']) or 'none'}。",
        "",
        "| pair | variant | candidate | OOD | model-only | strict action | oracle revalidated |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["predictions"]:
        lines.append(
            f"| `{row['pair_id']}` | `{row['variant']}` | `{row['candidate_family']}` | {'yes' if row['ood'] else 'no'} | "
            f"{'accept' if row['model_only_accepted'] else 'abstain'} | `{row['strict_ood_action']}` | "
            f"{'accept' if row['oracle_revalidated'] else 'abstain'} |"
        )
    lines.extend([
        "",
        "正向复核不是把 OOD 阈值调低：它要求 pair 两个编码、同一 surface、模型族一致、fixture 源码哈希一致、每条证据 SHA-256 有效、属性 oracle 成立且没有脚本信号。plain-control 作为反事实负例必须 abstain。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    server = make_server()
    thread = threading.Thread(target=server.serve_forever, name="sift-pg-pk-06-fixture", daemon=True)
    thread.start()
    try:
        _wait_for_fixture()
        source_hash = fixture_source_sha256()
        records = asyncio.run(
            PositiveFixtureCollector(target_instance_id=f"fixture-pid-{threading.get_ident()}", source_hash=source_hash).collect_many(
                default_fixture_specs()
            )
        )
        # The visible projection is what the decoder receives.  This assertion
        # prevents evaluator labels and the positive oracle from leaking into
        # the model input during the cross-app test.
        visible = json.dumps([catalog_visible_trace(row) for row in records], ensure_ascii=False).casefold()
        forbidden = ("evaluator", "challenge", "family", "source_id", "rule_ir", "marker_in_attribute")
        if any(token in visible for token in forbidden):
            raise RuntimeError("positive fixture visible projection leaked evaluator/oracle labels")

        model, checkpoint = _load_model()
        if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError("decoder checkpoint feature dimension mismatch")
        reference, ood_fit = _training_reference(checkpoint)
        predictions = _predict(model, checkpoint, records, ood_fit=ood_fit, ood_reference=reference)
        by_id = {row["sample_id"]: row for row in predictions}

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record.get("pair", {}).get("pair_id", "")), []).append(record)
        pair_results: list[dict[str, Any]] = []
        for pair_id, pair_records in sorted(grouped.items()):
            candidate_rows = [by_id[record["sample_id"]] for record in pair_records]
            candidate_by_id = {row["sample_id"]: row for row in candidate_rows}
            result = revalidate_positive_pair(
                [
                    dict(record, candidate_family=candidate_by_id[record["sample_id"]]["candidate_family"])
                    for record in pair_records
                ],
                expected_family="xss",
                oracle_name=FIXTURE_ORACLE,
                authorized_source_hash=source_hash,
                required_surface_role="reflected_attribute" if pair_id == "fixture-pair-01" else "reflected_attribute",
                required_sink_kind="html_attribute",
            )
            result["pair_id"] = pair_id
            pair_results.append(result)
            if result["accepted"]:
                for row in candidate_rows:
                    row["oracle_revalidated"] = True

        positive_pair = next((result for result in pair_results if result.get("pair_id") == "fixture-pair-01"), None)
        promotion_ledger = []
        if positive_pair and positive_pair.get("accepted"):
            evidence_hashes = list(positive_pair.get("evidence_hashes") or [])
            for index, seed in enumerate((20260861, 20260867)):
                promotion_ledger.append({
                    "dataset_id": "fixture-pg-pk-06",
                    "sampling_seed": seed,
                    "target_instance_id": f"fixture-pid-{threading.get_ident()}",
                    "rule_key": "xss::reflected_attribute",
                    "accepted": True,
                    "oracle_revalidated": True,
                    "false_positive": False,
                    "evidence_hash": evidence_hashes[index % len(evidence_hashes)] if evidence_hashes else "",
                    "local_only": True,
                })
        promotion_probe = assess_memory_promotion("xss::reflected_attribute", promotion_ledger)

        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": "sift-pg-pk-06-positive-oracle-report-v1",
            "target": {
                "base_url": FIXTURE_BASE_URL,
                "target_instance_id": f"fixture-pid-{threading.get_ident()}",
                "fixture_source_sha256": source_hash,
                "external_network": False,
                "loopback_only": True,
                "fresh_target": True,
            },
            "training_boundary": {
                "training_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
                "training_target": "Pikachu only",
                "fixture_seen_during_training": False,
                "visible_projection_labels_hidden": True,
            },
            "sample_count": len(records),
            "strict_ood_abstain_count": sum(row["ood"] for row in predictions),
            "strict_ood_abstain_rate": sum(row["ood"] for row in predictions) / len(predictions) if predictions else 1.0,
            "model_only_accepted_count": sum(row["model_only_accepted"] for row in predictions),
            "oracle_revalidated_sample_count": sum(row["oracle_revalidated"] for row in predictions),
            "oracle_revalidated_pair_count": sum(result["accepted"] for result in pair_results),
            "negative_control_revalidated_count": sum(
                result["accepted"] for result in pair_results if result.get("pair_id") == "fixture-pair-02"
            ),
            "ood_fit": ood_fit,
            "pair_results": pair_results,
            "memory_promotion_probe": promotion_probe,
            "predictions": predictions,
            "safety": {
                "raw_body_stored": False,
                "evaluator_state_visible": False,
                "external_network": False,
                "script_execution": False,
                "database_write": False,
                "mutations": False,
            },
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "sample_count": report["sample_count"],
            "strict_ood_abstain_count": report["strict_ood_abstain_count"],
            "model_only_accepted_count": report["model_only_accepted_count"],
            "oracle_revalidated_sample_count": report["oracle_revalidated_sample_count"],
            "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"],
            "negative_control_revalidated_count": report["negative_control_revalidated_count"],
            "memory_promotion_status": promotion_probe["status"],
            "memory_promotion_reasons": promotion_probe["reasons"],
            "report": report["report_path"],
            "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
        }, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
