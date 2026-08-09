"""PG-369 static design and input audit for a multi-task Rule-IR MoE.

PG-369 is deliberately a *plan-only* candidate.  It combines the full
abstract Rule-IR rows from PG-362 with the compositional WAF process rows
from PG-367 and describes one decoder-only causal MoE with several auxiliary
losses:

``L = L_next_token + L_slot_query + L_ask + L_repair + L_negative``

The script validates the data contract, computes bounded label/length
statistics, records the expected acceptance gates, and writes a plan.  It
does not import torch, instantiate a model, start a GPU job, send a request,
or create a checkpoint.  Existing training code is checked as source text so
that the future runner can reuse ``app.pg295_causal_moe`` without making this
audit an execution path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pg369-multitask-moe-plan-v1"
SLOTS = (
    "question",
    "ask_reason",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "probe_variant_ref",
    "safe_to_send",
    "payload_shape_ref",
    "oracle_ref",
    "negative_control_presence_ref",
)
TARGET_BOUNDARIES = ("[TARGET_BOS]", "[TARGET_EOS]")
SPLITS = ("train", "implementation_holdout")
RAW_EXACT_PREFIXES = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "response_body_text=",
    "raw_response=",
    "wire=",
    "evaluator=",
    "oracle=",
    "route_literal=",
    "family=",
    "implementation=",
    "image=",
    "source=",
)
RAW_VALUE_MARKERS = (
    "http://",
    "https://",
    "<script",
    "select ",
    "union ",
    "union select",
    "pg367-runtime-canary",
)
PROMOTION_KEYS = (
    "training_allowed",
    "memory_promotion_allowed",
    "payload_catalog_promotion_allowed",
    "vulnerability_claim_allowed",
)
DEFAULT_INPUTS = {
    "pg362": ROOT / "research" / "pg362_full_rule_ir_dataset_v1.json",
    "pg367": ROOT / "research" / "pg367_waf_staircase_dataset_v2.json",
}
DEFAULT_AUDITS = {
    "pg362": ROOT / "research" / "pg362_full_rule_ir_audit_v1.json",
    "pg367": ROOT / "research" / "pg367_waf_staircase_audit_v2.json",
}
DEFAULT_REFERENCE = ROOT / "research" / "pg367_a800_process_candidate_v2.json"
MOE_SOURCE = ROOT / "app" / "pg295_causal_moe.py"
MULTITASK_WEIGHTS = {
    "next_token": 1.0,
    "slot_query": 1.0,
    "ask": 1.5,
    "repair": 1.5,
    "negative": 2.0,
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path, *, max_bytes: int = 256 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"input exceeds bounded audit size: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _slot_values(target_tokens: Sequence[Any]) -> dict[str, str] | None:
    tokens = [str(token) for token in target_tokens]
    if tokens[:1] != [TARGET_BOUNDARIES[0]] or tokens[-1:] != [TARGET_BOUNDARIES[1]]:
        return None
    result: dict[str, str] = {}
    for token in tokens[1:-1]:
        if "=" not in token:
            return None
        key, value = token.split("=", 1)
        if key not in SLOTS or key in result or not value:
            return None
        result[key] = value
    if tuple(result) != SLOTS:
        return None
    return result


def derive_multitask_labels(target_tokens: Sequence[Any]) -> dict[str, Any] | None:
    """Derive all PG-369 supervision lanes from one abstract target.

    The returned object contains only Rule-IR categories and booleans.  It is
    intentionally a pure helper: callers can use it to build a future
    trainer batch, while this plan runner never persists those batches.
    """

    slots = _slot_values(target_tokens)
    if slots is None:
        return None
    question = slots["question"]
    action = slots["next_action"]
    safe = slots["safe_to_send"] == "1"
    return {
        "next_token": {key: slots[key] for key in SLOTS},
        "slot_query": {key: slots[key] for key in SLOTS},
        "ask": {
            "is_ask": question.startswith("ask_"),
            "question": question,
            "ask_reason": slots["ask_reason"],
        },
        "repair": {
            "is_repair": action == "repair",
            "next_action": action,
            "repair_action": slots["repair_action"],
        },
        "negative": {
            "is_negative_or_abstain": not safe,
            "safe_to_send": safe,
            "negative_control_presence_ref": slots["negative_control_presence_ref"],
        },
    }


def _is_raw_token(token: str) -> bool:
    folded = str(token).casefold()
    if any(folded.startswith(prefix) for prefix in RAW_EXACT_PREFIXES):
        return True
    return any(marker in folded for marker in RAW_VALUE_MARKERS)


def _promotion_closed(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    # Dataset rows use ``training_eligible`` while report roots use
    # ``training_allowed``.  Both are fail-closed; accepting either spelling
    # avoids manufacturing a false input failure for an otherwise reviewed
    # row.
    observed = set()
    for key in (*PROMOTION_KEYS, "training_eligible"):
        if key in value:
            observed.add(key)
            if value.get(key) is not False:
                return False
    return bool(observed) and "memory_promotion_allowed" in observed and "payload_catalog_promotion_allowed" in observed and "vulnerability_claim_allowed" in observed


def _audit_dataset(dataset: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    """Audit one abstract dataset without retaining any row material."""

    failures: set[str] = set()
    promotion_failure_count = 0
    records = dataset.get("records")
    if not isinstance(records, list):
        return {
            "source": source_name,
            "status": "blocked_input_audit",
            "counts": {"records": 0, "raw_hits": 0, "missing_vocabulary_tokens": 0},
            "failures": ["records_not_list"],
            "tasks": {},
            "lengths": {"max_context_target": 0, "min_context_target": 0},
        }
    seen_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = {key: Counter() for key in ("question", "next_action", "repair_action", "safe_to_send", "negative_control_presence_ref")}
    lengths: list[int] = []
    raw_hits = 0
    all_tokens: set[str] = set()
    for index, row in enumerate(records):
        prefix = f"row_{index}"
        if not isinstance(row, Mapping):
            failures.add(f"{prefix}:not_mapping")
            continue
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in seen_ids:
            failures.add(f"{prefix}:duplicate_record_id")
        seen_ids.add(record_id)
        split = str(row.get("split", ""))
        split_counts[split] += 1
        if split not in SPLITS:
            failures.add(f"{prefix}:split")
        context = row.get("context_tokens")
        target = row.get("target_tokens")
        if not isinstance(context, list) or not context:
            failures.add(f"{prefix}:context")
            context = []
        if not isinstance(target, list) or _slot_values(target) is None:
            failures.add(f"{prefix}:target_contract")
            target = []
        firewall = row.get("context_firewall")
        if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.add(f"{prefix}:context_firewall")
        for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context"):
            if row.get(flag) is not False:
                failures.add(f"{prefix}:{flag}")
        promotion = row.get("promotion")
        if not _promotion_closed(promotion):
            promotion_failure_count += 1
        values = [str(token) for token in [*context, *target]]
        all_tokens.update(values)
        if any(_is_raw_token(token) for token in values):
            raw_hits += 1
            failures.add(f"{prefix}:raw_token")
        slots = _slot_values(target)
        if slots is not None:
            for key in task_counts:
                task_counts[key][slots[key]] += 1
        lengths.append(len(context) + len(target))
    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    vocabulary = {str(token) for token in [*(vocab.get("context_tokens") or []), *(vocab.get("target_tokens") or [])]}
    missing = sorted(all_tokens - vocabulary)
    if missing:
        failures.add("missing_vocabulary_tokens")
    if not split_counts.get("train") or not split_counts.get("implementation_holdout"):
        failures.add("split_coverage")
    if not records:
        failures.add("empty_dataset")
    if promotion_failure_count:
        failures.add("row_promotion_not_closed")
    status = "passed_candidate_input_audit" if not failures else "blocked_input_audit"
    return {
        "source": source_name,
        "status": status,
        "counts": {
            "records": len(records),
            "train_rows": split_counts.get("train", 0),
            "implementation_holdout_rows": split_counts.get("implementation_holdout", 0),
            "raw_hits": raw_hits,
            "missing_vocabulary_tokens": len(missing),
            "training_eligible_rows": 0,
        },
        "tasks": {key: dict(sorted(values.items())) for key, values in task_counts.items()},
        "lengths": {
            "max_context_target": max(lengths or [0]),
            "min_context_target": min(lengths or [0]),
            "full_context_preserved": True,
            "silent_truncation": False,
        },
        "failures": sorted(failures),
    }


def _audit_report(report: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    failures: list[str] = []
    if not _promotion_closed(report.get("promotion")):
        failures.append("promotion_not_closed")
    if isinstance(report.get("failures"), list) and report.get("failures"):
        failures.append("upstream_failures_present")
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    if int(counts.get("raw_hits", 0) or 0) != 0:
        failures.append("upstream_raw_hits")
    if int(counts.get("missing_vocabulary_tokens", 0) or 0) != 0:
        failures.append("upstream_missing_vocabulary")
    return {
        "source": source_name,
        "status": "passed_candidate_audit" if not failures else "blocked_upstream_audit",
        "failures": sorted(set(failures)),
        "reported_status": str(report.get("status", "missing")),
        "reported_counts": {str(key): value for key, value in counts.items() if isinstance(value, (int, float, str))},
    }


def _model_source_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "blocked_model_source", "failures": ["model_source_missing"]}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return {"status": "blocked_model_source", "failures": ["model_source_unreadable"]}
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    required = {"CausalMoEConfig", "CausalMoELanguageModel", "train_causal_moe"}
    missing = sorted(required - names)
    return {
        "status": "passed_model_source_contract" if not missing else "blocked_model_source",
        "failures": ["missing_reusable_symbols"] if missing else [],
        "required_symbols": sorted(required),
        "reusable_symbols_present": sorted(required - set(missing)),
        "raw_source_loaded": False,
    }


def _combined_task_counts(audits: Sequence[Mapping[str, Any]], split: str) -> dict[str, dict[str, int]]:
    # The row audits retain aggregate task histograms.  Reconstructing a
    # train/holdout view by subtracting is safe only for balanced datasets,
    # so this helper is intentionally not used for exact split counts.  The
    # plan reports source-level histograms and combined row totals instead.
    del split
    result: dict[str, Counter[str]] = {key: Counter() for key in ("question", "next_action", "repair_action", "safe_to_send", "negative_control_presence_ref")}
    for audit in audits:
        tasks = audit.get("tasks") if isinstance(audit.get("tasks"), Mapping) else {}
        for key, values in tasks.items():
            if key not in result or not isinstance(values, Mapping):
                continue
            for value, count in values.items():
                result[key][str(value)] += int(count)
    return {key: dict(sorted(value.items())) for key, value in result.items()}


def _load_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "not_available"}
    try:
        report = _load_json(path, max_bytes=4 * 1024 * 1024)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"status": "unreadable"}
    candidates = report.get("candidates") if isinstance(report.get("candidates"), list) else []
    values = [
        float(item.get("holdout", {}).get("sequence_exact"))
        for item in candidates
        if isinstance(item, Mapping) and isinstance(item.get("holdout"), Mapping) and isinstance(item.get("holdout", {}).get("sequence_exact"), (int, float))
    ]
    return {
        "status": "observed_reference_only",
        "schema_version": report.get("schema_version"),
        "worst_holdout_sequence_exact": min(values) if values else None,
        "report_sha256": _sha_file(path),
        "promotion": {key: False for key in PROMOTION_KEYS},
    }


def build_plan(
    *,
    dataset_paths: Mapping[str, str | Path] = DEFAULT_INPUTS,
    audit_paths: Mapping[str, str | Path] = DEFAULT_AUDITS,
    reference_path: str | Path | None = DEFAULT_REFERENCE,
    model_source_path: str | Path = MOE_SOURCE,
) -> dict[str, Any]:
    """Build a static PG-369 design without invoking a trainer."""

    datasets: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    audit_reports: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    audit_hashes: dict[str, str] = {}
    total_train = 0
    total_holdout = 0
    maximum_length = 0
    for name in ("pg362", "pg367"):
        data_path = Path(dataset_paths[name])
        audit_path = Path(audit_paths[name])
        data = _load_json(data_path)
        upstream_audit = _load_json(audit_path)
        datasets[name] = data
        source_hashes[name] = _sha_file(data_path)
        audit_hashes[name] = _sha_file(audit_path)
        row_audit = _audit_dataset(data, source_name=name)
        audits.append(row_audit)
        audit_reports.append(_audit_report(upstream_audit, source_name=name))
        total_train += int(row_audit["counts"].get("train_rows", 0))
        total_holdout += int(row_audit["counts"].get("implementation_holdout_rows", 0))
        maximum_length = max(maximum_length, int(row_audit["lengths"].get("max_context_target", 0)))
    model_audit = _model_source_audit(Path(model_source_path))
    failures = sorted({failure for item in [*audits, *audit_reports, model_audit] for failure in item.get("failures", [])})
    source_task_counts = _combined_task_counts(audits, "all")
    required_window = max(768, maximum_length)
    reference = _load_reference(Path(reference_path) if reference_path is not None else None)
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_static_candidate_design" if not failures else "blocked_input_audit",
        "execution": {
            "plan_only": True,
            "trainer_invoked": False,
            "checkpoint_written": False,
            "gpu_touched": False,
            "docker_started": False,
            "network_used": False,
            "raw_material_loaded": False,
        },
        "inputs": {
            "sources": {
                "pg362": {"schema": "pg362-full-rule-ir-dataset-v1", "rows": audits[0]["counts"], "file_sha256": source_hashes["pg362"]},
                "pg367": {"schema": "pg367-waf-staircase-dataset-v2", "rows": audits[1]["counts"], "file_sha256": source_hashes["pg367"]},
            },
            "audits": {"pg362": audit_reports[0], "pg367": audit_reports[1]},
            "row_audits": {"pg362": audits[0], "pg367": audits[1]},
            "combined": {
                "train_rows": total_train,
                "implementation_holdout_rows": total_holdout,
                "vocabulary_union_size": len({
                    str(token)
                    for data in datasets.values()
                    for vocab_key in ("context_tokens", "target_tokens")
                    for token in ((data.get("vocabulary") or {}).get(vocab_key) or [])
                }),
                "max_context_target_length": maximum_length,
                "required_context_window": required_window,
                "full_context_preserved": True,
                "silent_truncation": False,
            },
        },
        "model": {
            "family": "decoder_only_causal_transformer_moe",
            "base_module": "app.pg295_causal_moe.py",
            "base_source_sha256": _sha_file(Path(model_source_path)) if Path(model_source_path).is_file() else None,
            "reused_symbols": model_audit.get("reusable_symbols_present", []),
            "config": {
                "d_model": 256,
                "n_heads": 4,
                "n_layers": 4,
                "experts": 4,
                "expert_hidden": 512,
                "max_length": required_window,
            },
            "context_policy": {
                "all_abstract_context_kept": True,
                "target_tokens_as_labels_only": True,
                "raw_payload_or_response_in_context": False,
                "sidecars_off_context": True,
            },
        },
        "multi_task_objective": {
            "formula": "1.0*L_next_token + 1.0*L_slot_query + 1.5*L_ask + 1.5*L_repair + 2.0*L_negative",
            "weights": dict(MULTITASK_WEIGHTS),
            "tasks": [
                {"name": "next_token", "weight": 1.0, "target": "full_ordered_rule_ir", "coverage": "all_13_slots"},
                {"name": "slot_query", "weight": 1.0, "target": "one_slot_value_given_full_context", "coverage": "all_13_slots"},
                {"name": "ask", "weight": 1.5, "target": "question_and_ask_reason", "coverage": "missing_or_failure_rows"},
                {"name": "repair", "weight": 1.5, "target": "next_action=repair_and_repair_action", "coverage": "failure_rows"},
                {"name": "negative", "weight": 2.0, "target": "safe_to_send=0_and_matched_negative_presence", "coverage": "abstain_or_negative_rows"},
            ],
            "label_values_are_abstract_only": True,
            "evaluator_labels_not_used": True,
            "label_builder": "derive_multitask_labels",
        },
        "task_coverage": {
            "combined_all_rows": source_task_counts,
            "slot_count": len(SLOTS),
        },
        "expected_metrics": {
            "status": "not_run_static_plan",
            "reference": reference,
            "acceptance_targets": {
                "holdout_sequence_exact_min": 0.70,
                "slot_query_accuracy_min": 0.90,
                "ask_recall_min": 0.95,
                "repair_recall_min": 0.90,
                "negative_false_allow_max": 0,
                "predictive_entropy_relative_drop_max": 0.25,
            },
            "interpretation": "这些是下一轮候选的验收门，不是已获得的模型成绩。",
        },
        "failures": failures,
        "scientific_gate": {
            "status": "blocked_candidate_only",
            "input_audit_required": True,
            "independent_implementation_required": True,
            "typed_live_replay_required": True,
            "claim_allowed": False,
        },
        "promotion": {key: False for key in PROMOTION_KEYS},
    }
    plan["plan_sha256"] = _sha_json(plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PG-369 plan-only multi-task MoE candidate")
    parser.add_argument("--pg362-dataset", type=Path, default=DEFAULT_INPUTS["pg362"])
    parser.add_argument("--pg367-dataset", type=Path, default=DEFAULT_INPUTS["pg367"])
    parser.add_argument("--pg362-audit", type=Path, default=DEFAULT_AUDITS["pg362"])
    parser.add_argument("--pg367-audit", type=Path, default=DEFAULT_AUDITS["pg367"])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model-source", type=Path, default=MOE_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg369_multitask_moe_plan_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        dataset_paths={"pg362": args.pg362_dataset, "pg367": args.pg367_dataset},
        audit_paths={"pg362": args.pg362_audit, "pg367": args.pg367_audit},
        reference_path=args.reference,
        model_source_path=args.model_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan if args.json else {"status": plan["status"], "failures": plan["failures"], "plan_sha256": plan["plan_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if plan["status"] == "ready_static_candidate_design" else 2


if __name__ == "__main__":
    raise SystemExit(main())
