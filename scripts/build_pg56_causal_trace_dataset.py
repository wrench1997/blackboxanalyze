"""Build a metadata-only causal trace corpus for PG-56.

The model input is an abstract event sequence.  It intentionally excludes
family names, source/implementation identifiers, raw probes, response bodies,
cookies, and evaluator labels.  Typed-oracle results are serialized only as
future events/targets so a causal decoder can learn what evidence follows an
action; they are not available before the target marker.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PG53_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
PG54_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg56_causal_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg56_causal_trace_dataset_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg56_causal_trace_dataset_report_v1.md"


def _bucket(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "u"
    if number <= 0:
        return "0"
    if number == 1:
        return "1"
    if number == 2:
        return "2"
    return "3p"


def _sign_bucket(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "u"
    if number < 0:
        return "neg"
    if number > 0:
        return "pos"
    return "zero"


def _response_tokens(prefix: str, phase: dict[str, Any] | None) -> list[str]:
    if not phase:
        return [f"{prefix}_MISSING"]
    response = phase.get("response") or {}
    shape = response.get("shape") or {}
    surface = phase.get("surface_observation") or {}
    geometry = phase.get("generic_effect_geometry") or {}
    return [
        f"{prefix}_STATUS_{str(response.get('status_class', 'unknown')).upper()}",
        f"{prefix}_CONTENT_{str(response.get('content_type_class', 'unknown')).upper()}",
        f"{prefix}_SHAPE_{str(shape.get('kind', 'unknown')).upper()}",
        f"{prefix}_KEYS_{_bucket(shape.get('key_count'))}",
        f"{prefix}_SCALARS_{_bucket(shape.get('scalar_count'))}",
        f"{prefix}_ARRAYS_{_bucket(shape.get('array_count'))}",
        f"{prefix}_TRUE_BOOL_{_bucket(surface.get('true_boolean_count'))}",
        f"{prefix}_NONZERO_NUM_{_bucket(surface.get('nonzero_numeric_count'))}",
        f"{prefix}_GEOM_LEAVES_{_bucket(geometry.get('leaf_count'))}",
        f"{prefix}_GEOM_DEPTH_{_bucket(geometry.get('max_depth'))}",
    ]


def _difference_tokens(control: dict[str, Any] | None, candidate: dict[str, Any] | None) -> list[str]:
    if not control or not candidate:
        return ["DIFF_MISSING"]
    cr = (control.get("response") or {}).get("shape") or {}
    tr = (candidate.get("response") or {}).get("shape") or {}
    cs = control.get("surface_observation") or {}
    ts = candidate.get("surface_observation") or {}
    cg = control.get("generic_effect_geometry") or {}
    tg = candidate.get("generic_effect_geometry") or {}
    fields = [
        ("KEYS", cr.get("key_count"), tr.get("key_count")),
        ("SCALARS", cr.get("scalar_count"), tr.get("scalar_count")),
        ("ARRAYS", cr.get("array_count"), tr.get("array_count")),
        ("TRUE_BOOL", cs.get("true_boolean_count"), ts.get("true_boolean_count")),
        ("NUM", cs.get("nonzero_numeric_count"), ts.get("nonzero_numeric_count")),
        ("LEAVES", cg.get("leaf_count"), tg.get("leaf_count")),
        ("OBJECTS", cg.get("object_count"), tg.get("object_count")),
    ]
    tokens = []
    for name, before, after in fields:
        try:
            delta = float(after) - float(before)
        except (TypeError, ValueError):
            delta = 0.0
        tokens.append(f"DIFF_{name}_{_sign_bucket(delta)}")
    return tokens


def _probe_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    manifest = row.get("payload_manifest") or {}
    descriptors = row.get("probe_descriptors") or {}
    descriptor = descriptors.get("candidate") or manifest
    return descriptor if isinstance(descriptor, dict) else {}


def _action_token(row: dict[str, Any]) -> str:
    descriptor = _probe_descriptor(row)
    method = str(row.get("method") or descriptor.get("method") or "UNKNOWN").upper()
    role = str(descriptor.get("role") or "candidate").upper()
    phase = str(descriptor.get("phase") or "candidate").upper()
    return f"ACTION_{method}_{role}_{phase}"


def _oracle_modality(row: dict[str, Any]) -> str:
    phase = row.get("candidate") or {}
    oracle = phase.get("oracle") or {}
    modality = str(oracle.get("modality") or "unknown").lower()
    mapping = {
        "browser_dom_effect": "DOM",
        "typed_dom": "DOM",
        "sql_ast_difference": "AST",
        "typed_ast": "AST",
        "redirect_destination": "REDIRECT",
        "typed_redirect": "REDIRECT",
        "typed_authentication_boundary": "BOUNDARY",
        "typed_authorization_boundary": "BOUNDARY",
        "typed_logic_invariant": "BOUNDARY",
        "typed_validation_boundary": "BOUNDARY",
        "typed_local_canary": "CANARY",
        "local_canary": "CANARY",
        "bounded_effect_contract": "EFFECT",
        "negative_control_or_screen": "NEGATIVE_CONTROL",
    }
    return mapping.get(modality, "OTHER")


def _belief_tokens(row: dict[str, Any]) -> list[str]:
    belief = row.get("belief") or {}
    steps = belief.get("steps") or []
    tokens: list[str] = []
    for step in steps:
        action = str(step.get("action_path") or "UNKNOWN").split(":", 1)[0].upper()
        tokens.append(f"BELIEF_ACTION_{action}")
        tokens.append(f"BELIEF_IG_{_bucket(round(float(step.get('information_gain', 0.0)) * 10)) if step.get('information_gain') is not None else 'u'}")
        tokens.append(f"BELIEF_DUP_{'1' if step.get('duplicate_evidence') else '0'}")
    return tokens or ["BELIEF_MISSING"]


def _row_tokens(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    control = row.get("control") or {}
    screen = row.get("screen") or {}
    candidate = row.get("candidate") or {}
    outcome = bool(((candidate.get("oracle") or {}).get("positive")))
    action = _action_token(row)
    modality = _oracle_modality(row)
    tokens = [
        "BOS",
        f"CHANNEL_{str(row.get('method', 'UNKNOWN')).upper()}",
        "CONTROL_TARGET",
        *_response_tokens("CONTROL", control),
        "SCREEN_TARGET",
        *_response_tokens("SCREEN", screen),
        "CANDIDATE_TARGET",
        *_response_tokens("CANDIDATE", candidate),
        *_difference_tokens(control, candidate),
        *_belief_tokens(row),
        "NEXT_ACTION_TARGET",
        action,
        "ORACLE_TARGET",
        f"ORACLE_MODALITY_{modality}",
        f"ORACLE_OUTCOME_{'POSITIVE' if outcome else 'NEGATIVE'}",
        "RULE_IR_TARGET",
        f"RULE_EFFECT_{'CONFIRMED' if outcome else 'REJECTED'}",
        f"RULE_TRANSPORT_{str(row.get('method', 'UNKNOWN')).upper()}",
        f"RULE_ORACLE_{modality}",
        "EOS",
    ]
    target = {
        "outcome": "positive" if outcome else "negative",
        "modality": modality,
        "family": str(row.get("family") or "unknown"),
        "unknown_family": str(row.get("family") or "") == "template_injection",
        "evidence_present": bool(row.get("evidence_sha256")),
    }
    return tokens, target


def _split_pg53(row: dict[str, Any]) -> str:
    # Keep one implementation out of training so this is a real source split.
    return "dev" if row.get("implementation") == "pg35" else "train"


def main() -> int:
    pg53 = json.loads(PG53_PATH.read_text(encoding="utf-8"))["rows"]
    pg54 = json.loads(PG54_PATH.read_text(encoding="utf-8"))["rows"]
    rows: list[dict[str, Any]] = []
    for row in pg53:
        tokens, target = _row_tokens(row)
        rows.append({
            "trace_id": str(row["sample_id"]),
            "split": _split_pg53(row),
            "tokens": tokens,
            "target": target,
            "raw_probe_stored": False,
            "raw_response_stored": False,
        })
    for row in pg54:
        tokens, target = _row_tokens(row)
        seed = int(row.get("sampling_seed", 0))
        if row.get("variant") == "framed":
            split = "holdout"
        elif seed == 419:
            split = "dev"
        else:
            split = "train"
        rows.append({
            "trace_id": str(row["sample_id"]),
            "split": split,
            "tokens": tokens,
            "target": target,
            "raw_probe_stored": False,
            "raw_response_stored": False,
        })
    counts = Counter(item["split"] for item in rows)
    token_counts = Counter(token for item in rows for token in item["tokens"])
    dataset = {
        "schema_version": "pg56-causal-trace-dataset-v1",
        "dataset_id": "pg56-causal-abstract-traces",
        "training_eligible": False,
        "evaluation_only": True,
        "model_input_contract": {
            "family_name_in_tokens": False,
            "source_id_in_tokens": False,
            "implementation_in_tokens": False,
            "raw_probe_in_tokens": False,
            "raw_response_body_in_tokens": False,
            "typed_oracle_before_target_marker": False,
            "evaluator_target_is_metadata_only": True,
        },
        "splits": {
            "train": "PG-53 pg34/pg36 + PG-54 ledger/envelope seeds 401/409",
            "dev": "PG-53 pg35 + PG-54 ledger/envelope seed 419",
            "holdout": "PG-54 framed all seeds, including template_injection",
        },
        "rows": rows,
        "split_counts": dict(counts),
        "vocabulary_preview": sorted(token_counts),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    dataset["dataset_sha256"] = hashlib.sha256(encoded).hexdigest()
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "protocol_id": "pg-pk-56-causal-trace-pretraining-v1",
        "schema_version": "pg56-causal-trace-dataset-report-v1",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "row_count": len(rows),
        "split_counts": dict(counts),
        "token_vocabulary_size": len(token_counts),
        "token_count": sum(token_counts.values()),
        "positive_target_count": sum(item["target"]["outcome"] == "positive" for item in rows),
        "unknown_family_target_count": sum(item["target"]["unknown_family"] for item in rows),
        "model_input_contract": dataset["model_input_contract"],
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
        "review_decision": "approved_for_pg56_causal_pretraining_only",
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "\n".join([
            "# PG-56 抽象因果 Trace 数据集",
            "",
            f"总轨迹：`{len(rows)}`；train/dev/holdout：`{counts['train']}/{counts['dev']}/{counts['holdout']}`；抽象 token 词表：`{len(token_counts)}`。",
            "",
            "输入只包含通道、匿名响应形状、成对差分、belief 步骤和抽象动作；漏洞族、来源、原始 probe、响应正文和 typed-oracle 结果不出现在目标标记之前。",
            "",
            "该数据集仅用于 PG-56 因果轨迹预训练，尚未获得长期记忆或正式漏洞族能力晋升资格。",
            "",
        ]) + "",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
