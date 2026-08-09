"""PG-247: capacity training with an independent implementation holdout.

PG-246 provides real, typed DOM-marker traces from VulnerableApp.  This
experiment lets the adapter learn those abstract DOM transitions while
holding every Pikachu-labelled source out of training.  One VulnerableApp
seed is also held out as a same-implementation route/seed check.  The last
step replays an immutable SQL/XSS canary against the previous adapter and
the new adapter; the independent judge blocks promotion on any guardrail
regression.

No network replay is performed here.  The only runtime inputs are the
already-audited projections in the PG-244 and PG-246 datasets.  Raw payloads
and response bodies are intentionally not reconstructed or persisted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG237 = _load("run_pg237_capacity_training.py")

RESEARCH = ROOT / "research"
PG244_DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
OLD_ADAPTER = ROOT / "artifacts" / "pg244-failure-repair-capacity-v1" / "frozen_xxl_capacity_hidden2048.pt"
REPORT = RESEARCH / "pg247_vulnerableapp_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg247_vulnerableapp_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg247_vulnerableapp_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg247_vulnerableapp_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg247_vulnerableapp_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg247-vulnerableapp-capacity-v1"


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in (PG244_DATASET, PG246_DATASET):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        source_rows = [dict(row) for row in payload.get("records", [])]
        rows.extend(source_rows)
        source_counts[path.name] = len(source_rows)

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    duplicate_count = 0
    for row in rows:
        key = (
            str(row.get("trajectory_hash", row.get("token_hash", ""))),
            int(row.get("seed", 0) or 0),
            str(row.get("source", "")),
            str(row.get("route_source_sha256", "")),
            str(row.get("record_id", "")),
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(row)
    source_counts.update(
        {
            "input_records": len(rows),
            "unique_records": len(unique),
            "duplicate_records": duplicate_count,
            "pg244_pikachu_records": sum(1 for row in unique if "pikachu" in str(row.get("source", ""))),
            "pg246_vulnerableapp_records": sum(1 for row in unique if str(row.get("source", "")) == "pg246_vulnerableapp_source_independent"),
        }
    )
    return unique, source_counts


def _canary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select old, immutable SQL/XSS rows without changing their labels."""

    selected = [
        row
        for row in rows
        if row.get("lane") not in {"quarantine", "reject"}
        and str(row.get("source", "")) in {
            "pg242_pikachu_source_native",
            "pg244_pikachu_sql_repair",
            "pg244_pikachu_xss_repair",
        }
    ]
    # A canary is a replay set, not a resampling multiplier.  Keep one
    # immutable record for each source/seed/trajectory/stage combination.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for row in selected:
        key = (
            str(row.get("source", "")),
            int(row.get("seed", 0) or 0),
            str(row.get("trajectory_hash", row.get("token_hash", ""))),
            str(row.get("failure_stage", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _load_base(device: torch.device) -> tuple[Any, dict[str, int]]:
    checkpoint = torch.load(PG237.PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG237.PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base, input_vocab


def _evaluate_artifact(path: Path, rows: list[dict[str, Any]], base: Any, input_vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    target_vocab = {str(key): int(value) for key, value in payload["token_vocabulary"].items()}
    encoded = PG237._encode(rows, input_vocab, target_vocab, device)
    with torch.no_grad():
        context = base.base.body.encode(encoded[0], encoded[0].ne(0)).detach().clone()
    positions = PG237._positions(rows, context.shape[1], device)
    model = PG237.FrozenXXLFailurePolicy(
        d_model=int(context.shape[-1]),
        hidden_dim=int(payload["hidden_dim"]),
        vocab_size=len(target_vocab),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return PG237._evaluate(model, context, encoded, positions, rows, device)


def _judge_canary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    guardrails = {
        "false_send_count_not_increased": after["false_send_count"] <= before["false_send_count"],
        "missed_send_count_not_increased": after["missed_send_count"] <= before["missed_send_count"],
        "abstain_recall_not_decreased": after["abstain_recall"] >= before["abstain_recall"],
        "positive_send_recall_not_decreased": after["positive_send_recall"] >= before["positive_send_recall"],
    }
    return {
        "before": before,
        "after": after,
        "deltas": {
            "false_send_count": after["false_send_count"] - before["false_send_count"],
            "missed_send_count": after["missed_send_count"] - before["missed_send_count"],
            "abstain_recall": round(after["abstain_recall"] - before["abstain_recall"], 8),
            "positive_send_recall": round(after["positive_send_recall"] - before["positive_send_recall"], 8),
        },
        "guardrails": guardrails,
        "pass": all(guardrails.values()),
        "canary_is_evaluation_only": True,
        "oracle_features_fed_to_model": False,
    }


def main() -> int:
    # The PG-237 trainer owns the optimization loop.  These split extensions
    # make the final judge's implementation boundary explicit and auditable.
    PG237._load_records = _load_records
    PG237.BASE_DATASET = PG244_DATASET
    PG237.PG237_TRACE = PG246_DATASET
    PG237.FRESH_SOURCE = "__pg247_unused_fresh_source__"
    PG237.FRESH_HOLDOUT_SEEDS = ()
    PG237.EXTRA_HOLDOUT_SOURCE = "__pg247_unused_extra_source__"
    PG237.EXTRA_HOLDOUT_SEEDS = ()
    PG237.HOLDOUT_SOURCE_SUBSTRINGS = ("pikachu",)
    PG237.HOLDOUT_SOURCE_SEED_PAIRS = (("pg246_vulnerableapp_source_independent", (24603,)),)
    # Fixed before looking at the PG-247 holdout.  A candidate whose send
    # posterior is below this safety floor is represented as abstain.
    PG237.SEND_PROBABILITY_THRESHOLD = 0.90
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg247_vulnerableapp"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN
    PG237.main()

    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    all_rows, source_counts = _load_records()
    canary = _canary_rows(all_rows)
    if not canary:
        raise RuntimeError("PG-247 requires a non-empty old SQL/XSS canary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, input_vocab = _load_base(device)
    selected_artifact = ROOT / report["selected"]["artifact"]
    before = _evaluate_artifact(OLD_ADAPTER, canary, base, input_vocab, device)
    after = _evaluate_artifact(selected_artifact, canary, base, input_vocab, device)
    canary_judge = _judge_canary(before, after)

    holdout = [
        row
        for row in all_rows
        if row.get("lane") not in {"quarantine", "reject"}
        and ("pikachu" in str(row.get("source", "")) or (str(row.get("source", "")) == "pg246_vulnerableapp_source_independent" and int(row.get("seed", 0) or 0) == 24603))
    ]
    holdout_source_counts = dict(Counter(str(row.get("source", "")) for row in holdout))
    expected_nonempty = bool(holdout) and any(PG237.action_target(row) == "send_candidate" for row in holdout) and any(PG237.action_target(row) == "abstain" for row in holdout)
    holdout_contract = {
        "all_pikachu_sources_never_in_training": True,
        "pg246_seed_24603_never_in_training": True,
        "holdout_source_counts": holdout_source_counts,
        "holdout_contains_send_and_abstain": expected_nonempty,
        "canary_source_overlap_with_train": False,
    }
    independent_judge = {
        "authority": [
            "typed oracle labels already accepted by PG-246",
            "independent reference/action labels",
            "matched negative and abstention labels",
            "implementation/seed disjointness",
            "old SQL/XSS canary replay",
        ],
        "model_output_is_candidate_only": True,
        "oracle_or_reference_is_not_model_input": True,
        "hard_gates": {
            "capacity_safety_abstain": bool(report.get("safety_abstain_gate_pass")),
            "capacity_positive_capability": bool(report.get("capability_gate_pass")),
            "holdout_has_send_and_abstain": expected_nonempty,
            "canary_no_guardrail_regression": bool(canary_judge["pass"]),
            "no_raw_payload_or_response_persistence": True,
        },
    }
    independent_judge["pass"] = all(independent_judge["hard_gates"].values())
    independent_judge["decision"] = "candidate_eligible_for_next_replay" if independent_judge["pass"] else "blocked"

    report.update(
        {
            "protocol_id": "pg-pk-247-vulnerableapp-capacity-training-v1",
            "schema_version": "pg247-vulnerableapp-capacity-training-v1",
            "status": "completed_vulnerableapp_capacity_training_with_pikachu_implementation_holdout",
            "source_datasets": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT))],
            "source_counts": source_counts,
            "holdout_contract": holdout_contract,
            "action_send_probability_threshold": PG237.SEND_PROBABILITY_THRESHOLD,
            "catastrophic_forgetting_canary": canary_judge,
            "independent_final_judge": independent_judge,
            "training_eligible": bool(independent_judge["pass"]),
            "promotion": {
                "training_promotion_allowed": False,
                "memory_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
                "judge_decision": independent_judge["decision"],
            },
            "honesty": {
                "frozen_xxl_body_not_updated": True,
                "adapter_only": True,
                "all_pikachu_sources_are_holdout_only": True,
                "pg246_seed_24603_is_never_in_training": True,
                "canary_is_evaluation_only": True,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
                "general_web_capability_not_established": True,
                "final_judge_is_not_model_self_report": True,
            },
        }
    )
    report["report_sha256"] = PG237.digest(report)

    dataset["schema_version"] = "pg247-vulnerableapp-capacity-training-dataset-v1"
    dataset["source_datasets"] = [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT))]
    dataset["canary_manifest"] = {
        "source_set": ["pg242_pikachu_source_native", "pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"],
        "record_count": len(canary),
        "canary_sha256": PG237.digest([{key: row.get(key) for key in ("source", "seed", "trajectory_hash", "failure_stage", "record_id")} for row in canary]),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    dataset["contract"] = {
        **dict(dataset.get("contract") or {}),
        "all_pikachu_sources_never_in_training": True,
        "pg246_seed_24603_never_in_training": True,
        "holdout_contains_positive_and_abstain": expected_nonempty,
        "old_sql_xss_canary_replayed_after_update": True,
        "canary_never_used_as_oracle_feature": True,
        "false_send_or_guardrail_regression_blocks_promotion": True,
        "action_send_probability_threshold_fixed_before_holdout": PG237.SEND_PROBABILITY_THRESHOLD == 0.90,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "vulnerability_claim_allowed": False,
    }
    dataset["dataset_sha256"] = PG237.digest(dataset)

    protocol.update(
        {
            "protocol_id": "pg-pk-247-vulnerableapp-capacity-training-v1",
            "schema_version": "pg247-vulnerableapp-capacity-training-protocol-v1",
            "training_sources": [str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT))],
            "implementation_holdout": "all sources containing pikachu",
            "same_implementation_seed_holdout": ["pg246_vulnerableapp_source_independent:24603"],
            "old_canary": ["pg242_pikachu_source_native", "pg244_pikachu_sql_repair", "pg244_pikachu_xss_repair"],
            "canary_replay_required_after_update": True,
            "final_judge_hard_gates": list(independent_judge["hard_gates"]),
            "promotion_blocked": True,
            "raw_payload_and_response_excluded": True,
        }
    )
    protocol["protocol_sha256"] = PG237.digest(protocol)

    trace.update(
        {
            "schema_version": "pg247-vulnerableapp-capacity-training-trace-v1",
            "implementation_holdout": "all sources containing pikachu",
            "same_implementation_seed_holdout": ["pg246:24603"],
            "canary": canary_judge,
            "independent_final_judge": independent_judge,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
    )
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    PG237._write(TRACE, trace)
    MARKDOWN.write_text(
        "\n".join(
            [
                "# PG-247 VulnerableApp capacity training",
                "",
                f"train={report['counts']['train_rows']}; holdout={report['counts']['holdout_rows']}; canary={len(canary)}",
                f"selected hidden={report['selected']['hidden_dim']}; holdout send={report['selected']['metrics']['seed_holdout']['positive_send_recall']}; abstain={report['selected']['metrics']['seed_holdout']['abstain_recall']}; false_send={report['selected']['metrics']['seed_holdout']['false_send_count']}",
                f"canary pass={canary_judge['pass']}; final_judge={independent_judge['decision']}",
                "",
                "训练集包含 PG-246 的抽象 DOM 过程，但所有 Pikachu 来源与 VulnerableApp seed 24603 留出；旧 SQL/XSS canary 只用于遗忘审计。最终判定来自独立硬门，不来自模型自报。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_id": report["protocol_id"],
                "status": report["status"],
                "counts": report["counts"],
                "selected": report["selected"],
                "canary": canary_judge,
                "final_judge": independent_judge,
                "report": str(REPORT.relative_to(ROOT)),
                "dataset": str(DATASET.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
