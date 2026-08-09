"""Run PG-02 source-grounding and family-holdout experiments.

The runner consumes only the locally generated, authorization-attested
catalog.  It evaluates two policies:

* ``source_grounded`` learns successful structural features from an authorized
  training split and abstains when a holdout surface has no support;
* ``random_authorized`` samples from the same validated candidate pool.

Family labels and target contracts are evaluator-side metadata.  Policy input
contains a validated payload and opaque provenance attestation only.
"""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dom_oracle import run_dom_oracle
from app.maze_engine import sha256_json
from app.payload_catalog import (
    flatten_catalog,
    load_catalog,
    policy_candidate,
    structural_feature_key,
    validate_policy_candidate,
)
from app.payload_grounding import SourceGroundedMemory


PROTOCOL_ID = "sift-payload-source-holdout-v1"
SEEDS = (20260841, 20260847, 20260853, 20260859, 20260861)
STRATEGIES = ("source_grounded", "random_authorized")


def _sql_oracle(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    # Import lazily so the catalogue module remains independent of the SQL
    # oracle implementation.
    from app.sql_ast_oracle import run_sql_ast_oracle

    evidence = run_sql_ast_oracle(str(payload["probe"])).to_dict()["evidence"]
    return evidence, bool(evidence.get("candidate_signal"))


def _rule_surface_oracle(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    semantic = record["semantic"]
    signal_fields: dict[str, bool] = {}
    for token in str(semantic["expected_signal"]).split("+"):
        signal_fields[token] = True
    evidence = {
        "oracle": "synthetic_rule_surface_v1",
        "surface": semantic["surface"],
        "candidate_signal": True,
        **signal_fields,
        "network_access": False,
        "navigation": False,
        "script_execution": False,
        "database_touched": False,
        "evaluator_state_visible": False,
    }
    evidence["evidence_hash"] = sha256_json(evidence)
    return evidence, True


def run_record_oracle(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    family = str(record["semantic"]["family"])
    payload = record["payload"]
    if family == "xss":
        transforms = ["html_entity_decode", "html_entity_decode"] if payload["probe_kind"] == "encoded_dom_markup" else []
        evidence = run_dom_oracle(
            str(payload["probe"]),
            transforms=transforms,
            marker=str(payload["marker"]),
        ).to_dict()
        return evidence, bool(evidence.get("candidate_signal"))
    if family == "injection":
        return _sql_oracle(payload)
    return _rule_surface_oracle(record)


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        source_id = str(row["provenance"]["source_id"])
        family = str(row["semantic"]["family"])
        result[source_id][family].append(row)
    return result


def _source_suffix(source_id: str) -> str:
    return source_id.rsplit("-", 1)[-1]


def build_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create evaluator-side source/family holdout tasks."""

    by_source_family = _index(rows)
    sources = sorted(by_source_family)
    source_a = [source for source in sources if _source_suffix(source) == "a"]
    source_b = [source for source in sources if _source_suffix(source) == "b"]

    def collect(source_ids: list[str], families: set[str]) -> list[dict[str, Any]]:
        return [
            row
            for source_id in source_ids
            for family in sorted(families)
            for row in by_source_family[source_id].get(family, [])
        ]

    cases: list[dict[str, Any]] = []
    # Source split: same families, unseen source IDs.  XSS/SQL are used here
    # because every abstract feature is distinct, avoiding generic-canary ties.
    train = collect(source_a, {"xss", "injection"})
    test = collect(source_b, {"xss", "injection"})
    generic_distractor = collect(source_b, {"access_control"})[0]
    for target in test:
        cases.append({
            "case_id": f"source_split:{target['sample_id']}",
            "split": "source_split_same_family",
            "mode": "exact_sample",
            "target": target,
            "train": train,
            # The distractor has an unsupported structural feature.  This
            # isolates source transfer from a hidden family label while
            # keeping the random baseline non-trivial.
            "pool": [target, generic_distractor],
            "heldout_family": target["semantic"]["family"],
        })

    # Family-external structural transfer: HTTP canary grammar is observed in
    # access-control training, then applied to unseen logic/redirect surfaces.
    transfer_train = collect(source_a, {"access_control"})
    transfer_distractors = collect(source_b, {"xss", "injection"})
    for family in ("logic", "url_redirect"):
        target_rows = collect(source_b, {family})
        cases.append({
            "case_id": f"family_transfer:{family}",
            "split": "family_holdout_structural_transfer",
            "mode": "exact_sample",
            "target": target_rows[0],
            "train": transfer_train,
            "pool": [target_rows[0], *transfer_distractors],
            "heldout_family": family,
        })

    # Family-external fail-closed check: no HTTP canary appears in training;
    # the grounded policy should abstain instead of hallucinating a payload.
    abstain_train = collect(source_a, {"xss", "injection"})
    abstain_distractor = collect(source_b, {"access_control"})[0]
    for family in ("access_control", "logic"):
        target_rows = collect(source_b, {family})
        cases.append({
            "case_id": f"family_abstain:{family}",
            "split": "family_holdout_unseen_surface",
            "mode": "expected_abstain",
            "target": target_rows[0],
            "train": abstain_train,
            # Both candidates are HTTP canaries, and neither is supported by
            # the training split.  A grounded policy must abstain.
            "pool": [target_rows[0], abstain_distractor],
            "heldout_family": family,
        })
    return cases


def _train_memory(records: list[dict[str, Any]], *, seed: int) -> SourceGroundedMemory:
    memory = SourceGroundedMemory(seed=seed)
    for record in records:
        candidate = policy_candidate(record)
        evidence, signal = run_record_oracle(record)
        memory.observe(
            candidate,
            status="observable_success" if signal else "dead_end",
            evidence=evidence,
            evaluator_confirmed=False,
        )
    return memory


def _random_choice(pool: list[dict[str, Any]], *, rng: random.Random) -> dict[str, Any]:
    candidate = copy.deepcopy(rng.choice(pool))
    # Validate the authorization marker before any policy can use it.
    candidate = validate_policy_candidate(candidate)
    candidate["selection_mode"] = "random_authorized"
    candidate["selection_score"] = None
    return candidate


def run_case(case: dict[str, Any], *, seed: int, strategy: str) -> dict[str, Any]:
    memory = _train_memory(case["train"], seed=seed)
    pool = [policy_candidate(row) for row in case["pool"]]
    random.Random(seed ^ 0xBEEF ^ len(case["case_id"])).shuffle(pool)
    supported_before = set(memory.supported_features())
    rng = random.Random(seed ^ 0xCAFE)
    if strategy == "source_grounded":
        selected = memory.select(pool, require_supported=True)
    elif strategy == "random_authorized":
        selected = _random_choice(pool, rng=rng)
    else:
        raise ValueError(f"unknown PG-02 strategy: {strategy}")

    target = case["target"]
    target_feature = str(target["structural_feature"])
    selected_feature = None
    evidence = None
    oracle_signal = False
    authorization_valid = True
    status = "abstain"
    target_match = False
    structural_match = False
    unsupported_feature_selected = False
    if selected is not None:
        try:
            selected = validate_policy_candidate(selected)
            selected_feature = structural_feature_key(selected["payload"])
            unsupported_feature_selected = selected_feature not in supported_before
            selected_record = next(
                row for row in case["pool"] if row["sample_id"] == selected["candidate_id"]
            )
            evidence, oracle_signal = run_record_oracle(selected_record)
            target_match = selected["candidate_id"] == target["sample_id"]
            structural_match = selected_feature == target_feature
            if target_match and oracle_signal:
                status = "observable_success"
            else:
                status = "dead_end"
        except (KeyError, StopIteration, ValueError):
            authorization_valid = False
            status = "rejected"
    else:
        # A grounded abstention is expected when the holdout feature is absent
        # from the authorized training evidence.
        status = "abstain"

    if strategy == "source_grounded" and selected is not None:
        memory.observe(
            selected,
            status="observable_success" if status == "observable_success" else "dead_end",
            evidence=evidence,
            evaluator_confirmed=False,
        )
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "mode": case["mode"],
        "heldout_family": case["heldout_family"],
        "seed": seed,
        "strategy": strategy,
        "train_sample_count": len(case["train"]),
        "pool_count": len(pool),
        "target_sample_id": target["sample_id"],
        "target_feature": target_feature,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_feature": selected_feature,
        "status": status,
        "target_match": target_match,
        "structural_match": structural_match,
        "oracle_signal": oracle_signal,
        "authorization_valid": authorization_valid,
        "unsupported_feature_selected": unsupported_feature_selected,
        "abstained": selected is None,
        "evidence_hash": evidence.get("evidence_hash") if evidence else None,
        "evaluator_confirmed": False,
        "policy_uses_evaluator": False,
        "supported_features_before": sorted(supported_before),
        "memory_summary": memory.summary(),
    }


def authorization_guard_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    memory = SourceGroundedMemory(seed=20260802)
    tampered = policy_candidate(rows[0])
    tampered["source_attestation"]["source_sha256"] = "0" * 64
    try:
        memory.select([tampered], require_supported=False)
    except ValueError as exc:
        return {
            "rejected": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {"rejected": False, "error_type": None, "error": None}


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["strategy"])].append(row)
    result: list[dict[str, Any]] = []
    for (split, strategy), group in sorted(grouped.items()):
        result.append({
            "split": split,
            "strategy": strategy,
            "case_count": len(group),
            "exact_success_rate": round(sum(row["status"] == "observable_success" for row in group) / len(group), 4),
            "structural_transfer_rate": round(sum(row["structural_match"] for row in group) / len(group), 4),
            "abstention_rate": round(sum(row["abstained"] for row in group) / len(group), 4),
            "unsupported_selection_rate": round(sum(row["unsupported_feature_selected"] for row in group) / len(group), 4),
            "authorization_valid_rate": round(sum(row["authorization_valid"] for row in group) / len(group), 4),
            "evaluator_confirmation_rate": 0.0,
        })
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-02 Source-Grounded Payload Holdout",
        "",
        "本实验只读取带授权证明的本地 safe detection manifest。source split 检查换来源迁移，family holdout 检查族外结构迁移与无证据时的 fail-closed abstention。",
        "",
        "| split | 策略 | exact success | structural transfer | abstention | unsupported selection | authorization valid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"| {row['split']} | {row['strategy']} | {row['exact_success_rate']:.2f} | "
            f"{row['structural_transfer_rate']:.2f} | {row['abstention_rate']:.2f} | "
            f"{row['unsupported_selection_rate']:.2f} | {row['authorization_valid_rate']:.2f} |"
        )
    lines.extend([
        "",
        "边界：source_grounded 是结构记忆控制器，不是神经模型；族外 transfer 的 exact success 仍来自本地合成 oracle，不代表真实站点漏洞确认。公网语料和 evaluator 状态均未接入。",
        "",
        f"原始 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    catalog_path = ROOT / "research" / "payload_source_catalog_v1.json"
    if not catalog_path.exists():
        from build_payload_source_catalog import build_catalog
        from app.payload_catalog import write_catalog
        write_catalog(catalog_path, build_catalog())
    catalog = load_catalog(catalog_path)
    records = flatten_catalog(catalog)
    cases = build_cases(records)
    guard = authorization_guard_check(records)
    if not guard["rejected"]:
        raise RuntimeError("authorization guard failed to reject tampered provenance")

    rows = [
        run_case(case, seed=seed, strategy=strategy)
        for case in cases
        for seed in SEEDS
        for strategy in STRATEGIES
    ]

    checkpoint_root = ROOT / "artifacts" / "payload-grounding" / PROTOCOL_ID
    # Keep one auditable training checkpoint per split/strategy/seed; individual
    # case rows retain the same pre-evaluation memory summary.
    checkpoint_paths: list[str] = []
    for split in sorted({case["split"] for case in cases}):
        case = next(case for case in cases if case["split"] == split)
        for seed in SEEDS:
            for strategy in ("source_grounded",):
                memory = _train_memory(case["train"], seed=seed)
                checkpoint = checkpoint_root / f"{split}-{strategy}-seed{seed}.json"
                memory.save(checkpoint)
                checkpoint_paths.append(str(checkpoint.relative_to(ROOT)))

    report_path = ROOT / "research" / "payload_source_holdout_v1.json"
    markdown_path = ROOT / "research" / "payload_source_holdout_v1.md"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-payload-source-holdout-report-v1",
        "catalog": {
            "path": str(catalog_path.relative_to(ROOT)),
            "catalog_sha256": catalog["catalog_sha256"],
            "source_count": len(catalog["sources"]),
            "sample_count": len(records),
        },
        "controls": {
            "policy_labels_visible": False,
            "provenance_visible": True,
            "source_ids_opaque": True,
            "probe_markers_opaque": True,
            "evaluator_state_visible": False,
            "public_payload_corpus_ingested": False,
            "network_execution": False,
            "script_execution": False,
            "database_execution": False,
            "real_sleep_performed": False,
            "seeds": list(SEEDS),
            "strategies": list(STRATEGIES),
        },
        "authorization_guard": guard,
        "case_count": len(cases),
        "aggregate": _aggregate(rows),
        "runs": rows,
        "checkpoints": checkpoint_paths,
        "interpretation": [
            "source_split 测量同一抽象 grammar 换来源后的迁移，而不是随机切分泄漏。",
            "family_holdout_structural_transfer 允许从已授权的 generic HTTP grammar 迁移到未见族，但不把它标记为 evaluator_confirmed。",
            "family_holdout_unseen_surface 检查训练证据不足时是否 abstain，防止 AI 猜测无来源 payload。",
            "授权证明、原始 probe、编码、预期 oracle 与 SHA-256 均在 catalog 中绑定。",
        ],
        "report_path": str(report_path.relative_to(ROOT)),
        "protocol_path": "research/payload_source_holdout_protocol_v1.json",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "catalog": report["catalog"],
        "authorization_guard": guard,
        "aggregate": report["aggregate"],
        "report": str(report_path.relative_to(ROOT)),
        "markdown": str(markdown_path.relative_to(ROOT)),
        "run_count": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
