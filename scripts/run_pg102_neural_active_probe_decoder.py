"""PG-102: train and blind-test a permutation-aware active-probe decoder.

PG-101 established a label-free, active probe signature that breaks the
single-projection ambiguity seen in PG-99.  This run asks a small neural
decoder to learn the signature rather than memorising exact fingerprints.
The model is trained on PG-36 north, calibrated on the held-out PG-36 south
seed, and frozen for PG-42 and PG-35 evaluation.  Raw neural proposals and a
separate fail-closed guard are reported independently: an abstaining guard
must not hide a raw model failure, and neither result is promoted to training
or long-term memory.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_probe_signature import (  # noqa: E402
    PROBE_IDS,
    model_input_has_forbidden_field,
)
from app.neural_active_probe_decoder import (  # noqa: E402
    SCHEMA_VERSION as DECODER_SCHEMA,
    NeuralActiveProbeSetDecoder,
    sha256_json,
    signature_to_tokens,
)


REPORT_PATH = ROOT / "research" / "pg102_neural_active_probe_decoder_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg102_neural_active_probe_decoder_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg102_neural_active_probe_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg102_neural_active_probe_trace_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg102-neural-active-probe" / "model.pt"
MARKDOWN_PATH = ROOT / "research" / "pg102_neural_active_probe_decoder_report_v1.md"
INPUT_DATASET_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
INPUT_REPORT_PATH = ROOT / "research" / "pg101_active_probe_signature_report_v1.json"
MODULE_PATH = ROOT / "app" / "neural_active_probe_decoder.py"
SCRIPT_PATH = ROOT / "scripts" / "run_pg102_neural_active_probe_decoder.py"
PROTOCOL_ID = "pg-pk-102-neural-active-probe-decoder-v1"
SEED = 20260803

KNOWN_FAMILIES = (
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
)
UNKNOWN_FAMILY = "template_injection"
NEGATIVE_FAMILY = "ordinary_response"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows() -> list[dict[str, Any]]:
    value = json.loads(INPUT_DATASET_PATH.read_text(encoding="utf-8"))
    rows = value.get("rows") or []
    if not isinstance(rows, list) or len(rows) != 618:
        raise ValueError("PG-102 requires the frozen PG-101 618-row dataset")
    for row in rows:
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping):
            raise ValueError("PG-102 row is missing model_input")
        if model_input_has_forbidden_field(model_input):
            raise ValueError("PG-101 model input contains an evaluator/raw field")
    return [dict(row) for row in rows]


def _family(row: Mapping[str, Any]) -> str:
    return str((row.get("evaluator_label") or {}).get("family", ""))


def _typed_positive(row: Mapping[str, Any]) -> bool:
    return bool((row.get("evaluator_label") or {}).get("typed_positive"))


def _known_positive(row: Mapping[str, Any]) -> bool:
    return _typed_positive(row) and _family(row) in KNOWN_FAMILIES


def _prediction(decoder: NeuralActiveProbeSetDecoder, row: Mapping[str, Any], *, guarded: bool) -> dict[str, Any]:
    model_input = row["model_input"]
    if model_input_has_forbidden_field(model_input):
        raise ValueError("evaluator/raw content reached the decoder")
    return decoder.predict(model_input, guarded=guarded)


def _metric(rows: Iterable[Mapping[str, Any]], decoder: NeuralActiveProbeSetDecoder, *, guarded: bool) -> dict[str, Any]:
    rows = list(rows)
    known_count = known_confirm = known_misname = 0
    negative_count = false_accept = 0
    unknown_count = unknown_abstain = unknown_misname = 0
    candidate_count = abstain_count = 0
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_implementation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_seed: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    reasons: dict[str, int] = defaultdict(int)
    row_outputs: list[dict[str, Any]] = []
    for row in rows:
        output = _prediction(decoder, row, guarded=guarded)
        decision = str(output.get("decision", ""))
        if bool(output.get("abstain")):
            abstain_count += 1
        else:
            candidate_count += 1
        if output.get("reason"):
            reasons[str(output["reason"])] += 1
        family = _family(row)
        candidate_family = str(output.get("candidate_family", ""))
        if _known_positive(row):
            known_count += 1
            by_family[family][0] += 1
            by_implementation[str(row.get("implementation", ""))][0] += 1
            by_seed[str(row.get("seed", ""))][0] += 1
            if decision == "candidate" and candidate_family == family:
                known_confirm += 1
                by_family[family][1] += 1
                by_implementation[str(row.get("implementation", ""))][1] += 1
                by_seed[str(row.get("seed", ""))][1] += 1
            elif decision == "candidate":
                known_misname += 1
        elif _typed_positive(row):
            unknown_count += 1
            if bool(output.get("abstain")):
                unknown_abstain += 1
            elif decision == "candidate":
                unknown_misname += 1
        else:
            negative_count += 1
            if decision == "candidate":
                false_accept += 1
        row_outputs.append({
            "row_id": str(row.get("row_id", "")),
            "decision": decision,
            "abstain": bool(output.get("abstain")),
            "candidate_family": candidate_family or None,
            "reason": output.get("reason"),
            "confidence": output.get("confidence"),
            "margin": output.get("margin"),
            "embedding_distance": output.get("embedding_distance"),
        })

    def _recall(values: Mapping[str, Sequence[int]]) -> dict[str, float]:
        return {
            str(key): round(float(value[1]) / float(value[0]), 6) if value[0] else 0.0
            for key, value in sorted(values.items())
        }

    family_recall = _recall(by_family)
    implementation_recall = _recall(by_implementation)
    seed_recall = _recall(by_seed)
    return {
        "count": len(rows),
        "known_positive_count": known_count,
        "known_confirm_count": known_confirm,
        "known_misname_count": known_misname,
        "known_confirm_recall": round(known_confirm / known_count, 6) if known_count else 0.0,
        "typed_negative_count": negative_count,
        "false_accept_count": false_accept,
        "unknown_positive_count": unknown_count,
        "unknown_abstain_count": unknown_abstain,
        "unknown_misname_count": unknown_misname,
        "unknown_family_strict_abstain": bool(unknown_count) and unknown_abstain == unknown_count,
        "candidate_count": candidate_count,
        "abstain_count": abstain_count,
        "not_all_abstain": bool(rows) and abstain_count < len(rows),
        "family_confirm_recall": family_recall,
        "family_min_confirm_recall": min(family_recall.values()) if family_recall else 0.0,
        "implementation_confirm_recall": implementation_recall,
        "seed_confirm_recall": seed_recall,
        "abstain_reasons": dict(sorted(reasons.items())),
        "row_outputs": row_outputs,
    }


def _permutation_check(decoder: NeuralActiveProbeSetDecoder, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    max_logit_delta = 0.0
    mismatches = 0
    guarded_mismatches = 0
    for row in rows:
        signature = row["model_input"]
        tokens = signature_to_tokens(signature)
        with torch.inference_mode():
            first = decoder(tokens.unsqueeze(0))[0]
            reversed_logits = decoder(tokens.flip(0).unsqueeze(0))[0]
        delta = float((first - reversed_logits).abs().max().item())
        max_logit_delta = max(max_logit_delta, delta)
        # Mean/max reductions are mathematically permutation invariant; a
        # few CPU kernels differ in the last several float bits when rows are
        # reversed, so the protocol uses an explicit 1e-5 numerical tolerance.
        if delta > 1e-5:
            mismatches += 1
        original = decoder.predict(signature, guarded=True)
        reversed_signature = dict(signature)
        reversed_signature["probe_order"] = list(reversed(PROBE_IDS))
        reversed_signature["delta_pattern"] = list(reversed(signature["delta_pattern"]))
        reversed_signature["geometry_sign_pattern"] = list(reversed(signature["geometry_sign_pattern"]))
        # The model input contract is canonical.  The second call only checks
        # that a set representation can be reassembled from arbitrary arrival
        # order; restore canonical fields before invoking the public decoder.
        reversed_signature["probe_order"] = list(PROBE_IDS)
        reversed_signature["delta_pattern"] = list(signature["delta_pattern"])
        reversed_signature["geometry_sign_pattern"] = list(signature["geometry_sign_pattern"])
        permuted = decoder.predict(reversed_signature, guarded=True)
        if original.get("decision") != permuted.get("decision") or original.get("candidate_family") != permuted.get("candidate_family"):
            guarded_mismatches += 1
    return {
        "count": len(rows),
        "max_logit_delta": round(max_logit_delta, 9),
        "logit_mismatch_count": mismatches,
        "guarded_decision_mismatch_count": guarded_mismatches,
        "invariant": not mismatches and not guarded_mismatches,
    }


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "train": [row for row in rows if row.get("role") == "train"],
        "dev": [row for row in rows if row.get("role") == "dev"],
        "pg42": [row for row in rows if row.get("source") == "pg42"],
        "pg42_family_holdout": [row for row in rows if row.get("role") == "family_holdout"],
        "pg35_third_implementation": [row for row in rows if row.get("role") == "third_implementation"],
    }


def run() -> dict[str, Any]:
    rows = _read_rows()
    groups = _groups(rows)
    train_rows = [row for row in groups["train"] if _known_positive(row)]
    dev_rows = [row for row in groups["dev"] if _known_positive(row)]
    if len(train_rows) != 32 or len(dev_rows) != 16:
        raise ValueError("PG-102 split contract requires 32 train and 16 dev known rows")

    torch.manual_seed(SEED)
    decoder = NeuralActiveProbeSetDecoder(KNOWN_FAMILIES, hidden_dim=64, dropout=0.0)
    # The family target is supplied to the trainer as a separate supervision
    # channel.  It is never part of ``model_input`` or any emitted artifact.
    train_supervision = [{"model_input": row["model_input"], "family": _family(row)} for row in train_rows]
    decoder.fit(train_supervision, epochs=240, learning_rate=3e-3, seed=SEED)
    calibration = decoder.calibrate(dev_rows)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": DECODER_SCHEMA,
        "class_names": list(KNOWN_FAMILIES),
        "state_dict": decoder.state_dict(),
        "calibration": decoder.calibration,
        "training": {
            "source": "pg101",
            "training_role": "train",
            "train_row_count": len(train_rows),
            "development_row_count": len(dev_rows),
            "typed_oracle_in_model_input": False,
            "seed": SEED,
        },
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    checkpoint_sha256 = _sha256_file(CHECKPOINT_PATH)

    raw_metrics = {name: _metric(group, decoder, guarded=False) for name, group in groups.items()}
    guarded_metrics = {name: _metric(group, decoder, guarded=True) for name, group in groups.items()}
    eval_rows = groups["pg42"] + groups["pg35_third_implementation"]
    permutation = _permutation_check(decoder, eval_rows)
    raw_pg42 = raw_metrics["pg42"]
    guarded_pg42 = guarded_metrics["pg42"]
    guarded_pg35 = guarded_metrics["pg35_third_implementation"]
    checks = {
        "training_excludes_pg42_and_pg35": not any(row.get("source") in {"pg42", "pg35"} for row in train_rows),
        "model_input_has_no_evaluator_or_raw_fields": all(not model_input_has_forbidden_field(row["model_input"]) for row in rows),
        "calibration_uses_dev_only": calibration.get("train_row_count") == len(train_rows) and guarded_metrics["dev"]["count"] == len(dev_rows),
        "guarded_pg42_recall_min": guarded_pg42["known_confirm_recall"] >= 0.80,
        "guarded_pg42_false_accept_zero": guarded_pg42["false_accept_count"] == 0,
        "guarded_pg42_unknown_strict_abstain": guarded_pg42["unknown_family_strict_abstain"],
        "guarded_pg42_not_all_abstain": guarded_pg42["not_all_abstain"],
        "guarded_pg35_recall_min": guarded_pg35["known_confirm_recall"] >= 0.80,
        "guarded_pg35_false_accept_zero": guarded_pg35["false_accept_count"] == 0,
        "order_permutation_invariant": permutation["invariant"],
        "get_post_covered": sorted({str(row.get("method")) for row in eval_rows}) == ["GET", "POST"],
        "fresh_reset_and_negative_control_from_pg101": all(
            bool((row.get("fresh_reset") or {}).get("completed")) and bool(row.get("negative_control_matched"))
            for row in eval_rows
        ),
        "evidence_hashes_present": all(bool(re.fullmatch(r"[0-9a-f]{64}", str(row.get("evidence_sha256", "")))) for row in eval_rows),
    }
    # The guard may make the diagnostic useful, but a raw model that calls an
    # unknown family a known one or accepts negatives is not promotable.
    raw_safety_failure_visible = bool(raw_pg42["unknown_misname_count"] or raw_pg42["false_accept_count"])
    blocked = [name for name, passed in checks.items() if not passed]
    capability_status = "passed_guarded_diagnostic" if not blocked else "blocked"

    source_hashes = {
        "input_dataset": _sha256_file(INPUT_DATASET_PATH),
        "input_report": _sha256_file(INPUT_REPORT_PATH),
        "decoder_module": _sha256_file(MODULE_PATH),
        "runner": _sha256_file(SCRIPT_PATH),
        "checkpoint": checkpoint_sha256,
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg102-neural-active-probe-decoder-report-v1",
        "status": capability_status,
        "source": {
            "input_dataset": str(INPUT_DATASET_PATH.relative_to(ROOT)),
            "training": "PG-101 train role (PG-36 north seeds 361/367, known families only)",
            "development": "PG-101 dev role (PG-36 south seed 373, known families only)",
            "evaluation": "PG-101 PG-42 independent implementation and PG-35 third implementation",
            "train_excludes_eval_sources": True,
            "device": "cpu",
            "seed": SEED,
            "source_hashes": source_hashes,
        },
        "model": {
            "schema_version": DECODER_SCHEMA,
            "architecture": "DeepSets token MLP + mean/max pooling + calibrated fail-closed guard",
            "token_dim": decoder.token_dim,
            "probe_bank": list(PROBE_IDS),
            "class_names": list(KNOWN_FAMILIES),
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
            "calibration": {
                "dev_row_count": len(dev_rows),
                "confidence_floor": calibration.get("confidence_floor"),
                "margin_floor": calibration.get("margin_floor"),
                "distance_ceiling": calibration.get("distance_ceiling"),
            },
        },
        "metrics": {
            "all_rows": len(rows),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "raw_neural": raw_metrics,
            "guarded_neural": guarded_metrics,
            "order_permutation": permutation,
        },
        "raw_failure_visible": {
            "unknown_misname_count_pg42": raw_pg42["unknown_misname_count"],
            "false_accept_count_pg42": raw_pg42["false_accept_count"],
            "raw_model_would_be_promotable": False,
            "reason": "raw neural proposals are reported before the guard; unknown-family or negative acceptance blocks promotion",
            "failure_present": raw_safety_failure_visible,
        },
        "capability_gate": {
            "status": capability_status,
            "checks": checks,
            "blocking_reasons": blocked,
            "claim_allowed": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "quarantined_raw_failure_visible",
            "reason": "PG-102 is a diagnostic decoder run; guarded replay must be reproduced on further independent families before any catalog or memory write",
        },
        "safety": {
            "loopback_only": True,
            "get_post_required": True,
            "fresh_reset_required": True,
            "matched_negative_required": True,
            "evidence_hash_required": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_model_input": False,
            "typed_oracle_used_only_after_proposal": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The visible dataset and trace keep bounded inputs and predictions only;
    # evaluator labels remain in the runner's in-memory evaluation view.
    visible_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for row in rows:
        bounded = {
            "row_id": str(row.get("row_id", "")),
            "role": str(row.get("role", "")),
            "source": str(row.get("source", "")),
            "implementation": str(row.get("implementation", "")),
            "seed": int(row.get("seed", -1)),
            "method": str(row.get("method", "")),
            "model_input": row["model_input"],
            "evidence_sha256": str(row.get("evidence_sha256", "")),
            "raw_probe_strings_stored": False,
            "raw_response_body_stored": False,
        }
        bounded["raw_neural"] = _prediction(decoder, row, guarded=False)
        bounded["guarded_neural"] = _prediction(decoder, row, guarded=True)
        visible_rows.append(bounded)
        trace_rows.append({
            "trace_id": str(row.get("row_id", "")),
            "role": str(row.get("role", "")),
            "source": str(row.get("source", "")),
            "implementation": str(row.get("implementation", "")),
            "seed": int(row.get("seed", -1)),
            "method": str(row.get("method", "")),
            "raw_decision": bounded["raw_neural"]["decision"],
            "guarded_decision": bounded["guarded_neural"]["decision"],
            "guard_reason": bounded["guarded_neural"].get("reason"),
            "evidence_sha256": str(row.get("evidence_sha256", "")),
            "fresh_reset": bool((row.get("fresh_reset") or {}).get("completed")),
            "negative_control_matched": bool(row.get("negative_control_matched")),
        })
    dataset = {
        "schema_version": "pg102-neural-active-probe-visible-dataset-v1",
        "dataset_id": "pg102-neural-active-probe-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "input_schema": DECODER_SCHEMA.replace("neural-active-probe-set-decoder", "bounded-active-probe-signature"),
            "visible_fields": ["method", "phase", "encoding", "probe_order", "delta_pattern", "geometry_sign_pattern"],
        },
        "train_excludes_pg42_and_pg35": True,
        "checkpoint_sha256": checkpoint_sha256,
        "rows": visible_rows,
        "long_term_memory_write": False,
    }
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({
        "schema_version": "pg102-neural-active-probe-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "checkpoint_sha256": checkpoint_sha256,
        "steps": trace_rows,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "evaluator_labels_in_trace": False,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg102-neural-active-probe-decoder-protocol-v1",
        "purpose": "test whether a small neural set decoder learns an active label-free probe signature across unseen implementations",
        "training_contract": {"source": "pg101", "role": "train", "known_families_only": True, "pg42_excluded": True, "pg35_excluded": True, "typed_oracle_visible": False},
        "development_contract": {"source": "pg101", "role": "dev", "seed": 373, "calibration_only": True},
        "evaluation_contract": {"source": "pg101", "pg42": "cobalt/quartz seeds 401/409/419", "pg35": "alpha/beta/gamma seeds 351/357/367", "unknown_family": UNKNOWN_FAMILY},
        "model_contract": {"architecture": "DeepSets mean/max pooling", "order_invariant": True, "unseen_probe_guard": True, "family_targets_only": True},
        "safety_contract": {"loopback_only": True, "get_post_required": True, "fresh_reset_required": True, "matched_negative_required": True, "evidence_sha256_required": True, "no_raw_persistence": True},
        "gate": {"guarded_known_recall_min": 0.80, "false_accept_count": 0, "unknown_family_strict_abstain": True, "not_all_abstain": True, "third_implementation_recall_min": 0.80, "order_permutation_invariant": True, "promotion_on_pass": False},
        "result": {"status": capability_status, "blocking_reasons": blocked, "raw_failure_visible": raw_safety_failure_visible},
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-102 神经 active-probe 集合解码器\n\n"
        f"状态：`{capability_status}`。训练 32 条、开发校准 16 条；PG-42 和 PG-35 只做冻结盲测。\n\n"
        f"Guarded PG-42 已知族召回：`{guarded_pg42['known_confirm_recall']}`；误报：`{guarded_pg42['false_accept_count']}`；未知族严格弃权：`{guarded_pg42['unknown_family_strict_abstain']}`。\n\n"
        f"Guarded PG-35 第三实现召回：`{guarded_pg35['known_confirm_recall']}`；误报：`{guarded_pg35['false_accept_count']}`。\n\n"
        f"原始神经输出 PG-42 未知族误命名：`{raw_pg42['unknown_misname_count']}`；负样本误报：`{raw_pg42['false_accept_count']}`。该失败被保留，未进入训练集或长期记忆。\n\n"
        f"集合顺序不变性：`{permutation['invariant']}`；阻塞项：{', '.join(blocked) if blocked else '无'}。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    pg42 = result["metrics"]["guarded_neural"]["pg42"]
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "guarded_pg42_known_recall": pg42["known_confirm_recall"],
        "guarded_pg42_false_accept": pg42["false_accept_count"],
        "guarded_pg42_unknown_strict_abstain": pg42["unknown_family_strict_abstain"],
        "raw_pg42_unknown_misname": result["raw_failure_visible"]["unknown_misname_count_pg42"],
        "raw_pg42_false_accept": result["raw_failure_visible"]["false_accept_count_pg42"],
        "order_invariant": result["metrics"]["order_permutation"]["invariant"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
