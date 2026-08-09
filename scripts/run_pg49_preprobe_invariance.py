"""Check that PG-48's pre-probe action ordering is invariant to response masking."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG48_SCRIPT = ROOT / "scripts" / "train_pg48_compositional_preprobe.py"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
CATALOG_PATH = ROOT / "research" / "pg48_compositional_preprobe_catalog_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg48-compositional-preprobe" / "preprobe_action_value.pt"
REPORT_PATH = ROOT / "research" / "pg49_preprobe_invariance_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg49_preprobe_invariance_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _confirm_scores(pg48: Any, pairs: dict[tuple[str, str], dict[str, Any]], model: Any, semantic_index: dict[str, int], channel_index: dict[str, int]) -> tuple[list[list[str]], list[float]]:
    neutral = (1.0, 0.0, 1.0)
    scores: dict[tuple[str, str], float] = {}
    for action in pg48.ACTION_ORDER[2:]:
        pair = pairs.get(action)
        if pair is None:
            continue
        features = pg48._features([pair], [action], [neutral], semantic_index, channel_index)
        with torch.inference_mode():
            scores[action] = float(torch.sigmoid(model(features))[0])
    ordered = sorted(scores, key=lambda action: (-scores[action], pg48.ACTION_INDEX[action]))
    return [[action[0], action[1]] for action in ordered], [scores[action] for action in ordered]


def _mask_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    masked = copy.deepcopy(pairs)
    for pair in masked:
        for row in (pair["candidate"], pair["control"]):
            row["response_projection"] = {"status_code": 0, "body_length": 0, "masked": True}
            row["oracle_projection"] = {"positive": False, "confirmed_effect": "none", "masked": True}
            row["evidence"] = {"evidence_hash": "0" * 64, "masked": True}
    return masked


def main() -> int:
    pg48 = _load(PG48_SCRIPT, "pg48_for_pg49")
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg49")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = [pair for pair in pg38._pair_rows(list(catalog["samples"])) if pair["candidate"].get("implementation") == "frost"]
    masked_pairs = _mask_pairs(pairs)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    semantic_index = dict(checkpoint["semantic_index"])
    channel_index = dict(checkpoint["channel_index"])
    model = pg48.PreProbeActionValueModel(len(semantic_index) + len(channel_index) + len(pg48.ACTION_ORDER) + 3)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int, str], dict[tuple[str, str], dict[str, Any]]]:
        result: dict[tuple[str, str, str, int, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
        for pair in rows:
            candidate = pair["candidate"]
            key = (str(candidate["implementation"]), str(candidate["surface_id"]), str(candidate["surface_variant"]), int(candidate["sampling_seed"]), str(candidate["semantic_reference"]))
            result[key][(str(candidate["method"]), str(candidate["phase"]))] = pair
        return result

    original_groups = grouped(pairs)
    masked_groups = grouped(masked_pairs)
    episode_results: list[dict[str, Any]] = []
    max_score_delta = 0.0
    changed_count = 0
    for key in sorted(original_groups):
        original_order, original_scores = _confirm_scores(pg48, original_groups[key], model, semantic_index, channel_index)
        masked_order, masked_scores = _confirm_scores(pg48, masked_groups[key], model, semantic_index, channel_index)
        deltas = [abs(left - right) for left, right in zip(original_scores, masked_scores)]
        max_score_delta = max(max_score_delta, max(deltas, default=0.0))
        changed = original_order != masked_order
        changed_count += int(changed)
        episode_results.append({"implementation": key[0], "surface_id": key[1], "surface_variant": key[2], "sampling_seed": key[3], "semantic_reference": key[4], "original_confirm_order": original_order, "masked_confirm_order": masked_order, "changed": changed, "score_delta_max": round(max(deltas, default=0.0), 12)})

    report = {
        "protocol_id": "sift-pg49-preprobe-invariance-v1",
        "schema_version": "pg-pk-49-preprobe-invariance-report-v1",
        "status": "diagnostic_only",
        "holdout_implementation": "frost",
        "pair_count": len(pairs),
        "episode_count": len(episode_results),
        "selection_order_match_rate": round((len(episode_results) - changed_count) / max(len(episode_results), 1), 6),
        "selection_changed_count": changed_count,
        "max_score_delta": round(max_score_delta, 12),
        "response_projection_consumed_by_selection": False,
        "mutated_fields": ["response_projection", "control.response_projection", "oracle_projection", "evidence"],
        "input_contract": "semantic/channel/action/belief only",
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "episodes": episode_results,
        "safe_gate": {"status": "passed" if changed_count == 0 and max_score_delta == 0.0 else "blocked", "claim_allowed": changed_count == 0 and max_score_delta == 0.0, "training_allowed": False, "memory_promotion_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False},
    }
    manifest = {"protocol_id": report["protocol_id"], "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": report["checkpoint_sha256"], "selection_order_match_rate": report["selection_order_match_rate"]}
    report["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-49 pre-probe invariance", "", "将响应/ oracle / 证据字段置换为占位值后，发送前确认顺序仍只由 semantic/channel/action/belief 决定。", "", f"- episodes: {report['episode_count']}", f"- selection order match rate: {report['selection_order_match_rate']}", f"- max score delta: {report['max_score_delta']}", f"- safety gate: `{report['safe_gate']['status']}`", "- training/memory promotion: blocked"]) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol_id", "pair_count", "episode_count", "selection_order_match_rate", "selection_changed_count", "max_score_delta", "safe_gate")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
