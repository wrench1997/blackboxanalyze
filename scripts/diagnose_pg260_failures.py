# -*- coding: utf-8 -*-
"""Diagnose PG-260 abstract-head failures without storing raw wires."""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG260 = _load("run_pg260_active_belief_capacity_training.py", "pg260_failure_diag")
PG249 = PG260.PG249
PG231 = PG260.PG231
PG237 = PG260.PG237
from app.pg260_active_belief_adapter import (  # noqa: E402
    ABSTAIN_CLASSES,
    BELIEF_CLASSES,
    FAMILY_CLASSES,
    PROBE_CLASSES,
    RULE_IR_CLASSES,
    PG260ActiveBeliefAdapter,
    belief_target,
    evaluate_pg260_adapter,
    family_target,
    probe_target,
    rule_target,
    unknown_abstain_target,
)

REPORT = ROOT / "research" / "pg260_failure_diagnostic_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
LEGACY_ARTIFACT = ROOT / "artifacts" / "pg249-pikachu-route-seed-capacity-v1" / "frozen_xxl_capacity_hidden4096.pt"
ADAPTER_ARTIFACT = ROOT / "artifacts" / "pg260-active-belief-capacity-v1" / "active_belief_hidden4096.pt"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def main() -> int:
    rows = PG260._load_records()
    rows = [row for row in rows if PG260._is_ood(row) or PG260._is_holdout(row)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    if not hasattr(PG249.PG248.PG247, "ORIGINAL_INPUT_TOKEN_ID"):
        PG249.PG248.PG247.ORIGINAL_INPUT_TOKEN_ID = PG231._input_token_id
    PG231._input_token_id = PG249.PG248._patched_input_token_id
    base, _ = PG249.PG248.PG247._load_base(device)
    old_payload = torch.load(LEGACY_ARTIFACT, map_location="cpu", weights_only=False)
    old_vocab = {str(key): int(value) for key, value in old_payload["token_vocabulary"].items()}
    old_policy = PG237.FrozenXXLFailurePolicy(d_model=1024, hidden_dim=int(old_payload["hidden_dim"]), vocab_size=len(old_vocab)).to(device)
    old_policy.load_state_dict(old_payload["state_dict"], strict=True)
    old_policy.eval()
    encoded = PG231._encode(rows, input_vocab, {str(key): int(value) for key, value in torch.load(ADAPTER_ARTIFACT, map_location="cpu", weights_only=False)["token_vocabulary"].items()}, device)
    with torch.no_grad():
        body = base.base.body.encode(encoded[0], encoded[0].ne(0)).detach()
        context = old_policy.context_projection(body).detach()
    payload = torch.load(ADAPTER_ARTIFACT, map_location="cpu", weights_only=False)
    model = PG260ActiveBeliefAdapter(d_model=int(context.shape[-1]), hidden_dim=int(payload["hidden_dim"]), token_vocab_size=len(payload["token_vocabulary"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    positions = PG260._positions(rows, context.shape[1], device)
    with torch.no_grad():
        output = model(context, classification_positions=positions, attention_mask=encoded[0].ne(0))
    rule_ids = output["rule"].argmax(dim=-1).tolist()
    family_ids = output["family"].argmax(dim=-1).tolist()
    belief_ids = output["belief"].argmax(dim=-1).tolist()
    probe_ids = output["probe"].argmax(dim=-1).tolist()
    abstain_ids = output["unknown_abstain"].argmax(dim=-1).tolist()
    records = []
    confusion = collections.Counter()
    family_confusion = collections.Counter()
    for index, row in enumerate(rows):
        target_rule = str(row["rule_ir_class"])
        target_family = str(row["family_class"])
        pred_rule = RULE_IR_CLASSES[rule_ids[index]]
        pred_family = FAMILY_CLASSES[family_ids[index]]
        confusion[(target_rule, pred_rule)] += 1
        family_confusion[(target_family, pred_family)] += 1
        records.append({"source": str(row.get("source", "")), "seed": int(row.get("seed", 0) or 0), "route": PG260._route(row), "lane": str(row.get("lane", "")), "rule_target": target_rule, "rule_pred": pred_rule, "family_target": target_family, "family_pred": pred_family, "family_target_id": family_target(target_family), "family_pred_id": family_ids[index], "belief_target": BELIEF_CLASSES[belief_target(row["belief_class"])], "belief_pred": BELIEF_CLASSES[belief_ids[index]], "probe_target": PROBE_CLASSES[probe_target(row["probe_class"])], "probe_pred": PROBE_CLASSES[probe_ids[index]], "unknown_abstain_target": ABSTAIN_CLASSES[unknown_abstain_target(row)], "unknown_abstain_pred": ABSTAIN_CLASSES[abstain_ids[index]], "rule_correct": target_rule == pred_rule, "family_correct": target_family == pred_family})
    target_vocab = {str(key): int(value) for key, value in payload["token_vocabulary"].items()}
    encoded_targets = PG231._encode(rows, input_vocab, target_vocab, device)
    metrics = evaluate_pg260_adapter(model, context, encoded_targets[1], torch.tensor([rule_target(row["rule_ir_class"]) for row in rows], device=device), torch.tensor([family_target(row["family_class"]) for row in rows], device=device), torch.tensor([belief_target(row["belief_class"]) for row in rows], device=device), torch.tensor([probe_target(row["probe_class"]) for row in rows], device=device), torch.tensor([unknown_abstain_target(row) for row in rows], device=device), positions, attention_mask=encoded_targets[0].ne(0))
    subset_metrics = {}
    context_invariance = {}
    for label, subset in (("ood", [row for row in rows if PG260._is_ood(row)]), ("holdout", [row for row in rows if PG260._is_holdout(row) and not PG260._is_ood(row)])):
        subset_encoded = PG231._encode(subset, input_vocab, target_vocab, device)
        with torch.no_grad():
            subset_body = base.base.body.encode(subset_encoded[0], subset_encoded[0].ne(0)).detach()
            subset_context = old_policy.context_projection(subset_body).detach()
        subset_metrics[label] = PG260._metrics(model, subset_context, subset_encoded, subset, device)
        indices = [index for index, row in enumerate(rows) if (PG260._is_ood(row) if label == "ood" else PG260._is_holdout(row) and not PG260._is_ood(row))]
        # Compare only real-token positions; differing widths expose padding
        # sensitivity in the frozen body before the adapter is even reached.
        max_delta = 0.0
        for local_index, union_index in enumerate(indices):
            width = int(subset_encoded[0][local_index].ne(0).sum().item())
            max_delta = max(max_delta, float((subset_context[local_index, :width] - context[union_index, :width]).abs().max().item()))
        context_invariance[label] = {"max_abs_delta_on_real_tokens": max_delta, "padding_width_union": int(encoded[0].shape[1]), "padding_width_subset": int(subset_encoded[0].shape[1])}
    payload_out = {"schema_version": "pg260-failure-diagnostic-v1", "model_artifact": str(ADAPTER_ARTIFACT.relative_to(ROOT)), "model_artifact_sha256": hashlib.sha256(ADAPTER_ARTIFACT.read_bytes()).hexdigest(), "device": str(device), "record_count": len(records), "ood_count": sum(PG260._is_ood(row) for row in rows), "holdout_count": sum(PG260._is_holdout(row) and not PG260._is_ood(row) for row in rows), "confusion": {f"{left}->{right}": count for (left, right), count in sorted(confusion.items())}, "family_confusion": {f"{left}->{right}": count for (left, right), count in sorted(family_confusion.items())}, "metrics_on_union": metrics, "metrics_by_subset": subset_metrics, "context_invariance": context_invariance, "records": records, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}
    payload_out["report_sha256"] = _digest(payload_out)
    REPORT.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT.relative_to(ROOT)), "record_count": len(records), "ood_count": payload_out["ood_count"], "holdout_count": payload_out["holdout_count"], "family_confusion": payload_out["family_confusion"], "report_sha256": payload_out["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
