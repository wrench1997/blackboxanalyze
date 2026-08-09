"""PG-109: cross-seed and cross-implementation Rule IR fragment composition."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg107_multistep_generic_belief as pg107  # noqa: E402
from app.active_goal_label_inducer import ActiveGoalLabelInducer  # noqa: E402
from app.active_probe_signature import model_input_has_forbidden_field  # noqa: E402
from app.probe_binding_attestation import CANONICAL_BINDING_SHA256, add_binding_attestation, binding_attestation_valid  # noqa: E402
from app.rule_fragment_assembler import assemble_rule_fragments, fragment_from_row  # noqa: E402


PROTOCOL_ID = "pg-pk-109-fragment-composition-v1"
TRAIN_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
PG105_DATASET_PATH = ROOT / "research" / "pg105_observable_projection_visible_dataset_v1.json"
PG105_TRACE_PATH = ROOT / "research" / "pg105_observable_projection_trace_v1.json"
PG106_DATASET_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_visible_dataset_v1.json"
PG106_TRACE_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_trace_v1.json"
ASSEMBLER_PATH = ROOT / "app" / "rule_fragment_assembler.py"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg109_fragment_composition.py"
REPORT_PATH = ROOT / "research" / "pg109_fragment_composition_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg109_fragment_composition_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg109_fragment_composition_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg109_fragment_composition_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg109_fragment_composition_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg109_fragment_composition_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training() -> list[dict[str, Any]]:
    dataset = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for original in dataset.get("rows", []):
        if original.get("role") != "train":
            continue
        row = dict(original)
        row["model_input"] = add_binding_attestation(row["model_input"])
        rows.append(row)
    if len(rows) != 32:
        raise ValueError("PG-109 requires the frozen 32-row PG-101 training role")
    return rows


def _load_evaluation() -> list[dict[str, Any]]:
    rows = pg107._load_rows(PG105_DATASET_PATH, PG105_TRACE_PATH, sources={"pg42", "pg35", "pg76", "pg69"})
    rows.extend(pg107._load_rows(PG106_DATASET_PATH, PG106_TRACE_PATH, sources={"pg106"}))
    return rows


def _raw_effect_slots(row: dict[str, Any]) -> tuple[str, ...]:
    pattern = list(row.get("model_input", {}).get("delta_pattern") or [])
    return tuple(f"p{index}" for index, changed in enumerate(pattern) if bool(changed))


def _is_effect_pair(rows: list[dict[str, Any]], supported_slots: set[str]) -> bool:
    if len(rows) != 2 or {str(row.get("method")) for row in rows} != {"GET", "POST"}:
        return False
    return (
        all(len(_raw_effect_slots(row)) == 1 for row in rows)
        and len({_raw_effect_slots(row)[0] for row in rows}) == 1
        and _raw_effect_slots(rows[0])[0] in supported_slots
    )


def _make_episode(rows: list[dict[str, Any]], supported_slots: set[str]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (0 if str(row.get("method")) == "GET" else 1, str(row.get("row_id", ""))))
    fragments = [fragment_from_row(row) for row in ordered]
    assembly = assemble_rule_fragments(fragments, supported_slots=sorted(supported_slots))
    reversed_assembly = assemble_rule_fragments(list(reversed(fragments)), supported_slots=sorted(supported_slots))
    return {
        "episode_group": str(ordered[0].get("episode_group", "")) if ordered else "",
        "source": str(ordered[0].get("source", "")) if ordered else "",
        "implementations": sorted({str(row.get("implementation", "")) for row in ordered}),
        "methods": sorted({str(row.get("method", "")) for row in ordered}),
        "fragments": fragments,
        "assembly": assembly,
        "reverse_assembly": reversed_assembly,
        "order_invariant": assembly.get("canonical_sha256") == reversed_assembly.get("canonical_sha256"),
        "fresh_reset_per_fragment": all(bool(row.get("fresh_reset", {}).get("fresh_target")) for row in ordered),
        "negative_control_per_fragment": all(bool(row.get("negative_control_matched")) for row in ordered),
        "typed_oracle_called": False,
        "confirmed_positive": False,
        "training_eligible": False,
        "long_term_memory_write": False,
    }


def _find_effect_row(rows: list[dict[str, Any]], *, source: str, method: str, slot: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("source")) != source or str(row.get("method")) != method:
            continue
        fragment = fragment_from_row(row)
        if fragment["relation_atom"] == "effect_present" and str(fragment.get("slot")) == slot:
            return row
    return None


def run() -> dict[str, Any]:
    train_rows = _load_training()
    evaluation_rows = _load_evaluation()
    inducer = ActiveGoalLabelInducer(
        minimum_support=2,
        require_get_post=True,
        require_binding_attestation=True,
        expected_binding_sha256=CANONICAL_BINDING_SHA256,
    ).fit([{"model_input": row["model_input"]} for row in train_rows])
    proposal = inducer.proposal()
    supported_slots = {str(slot) for slot in proposal.get("supported_slots", [])}
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        groups[str(row["episode_group"])].append(row)
    grouped_rows = list(groups.values())
    episodes = [_make_episode(group_rows, supported_slots) for group_rows in grouped_rows]
    # The denominator is defined from the raw bounded response-delta pattern,
    # before the assembler is called.  This prevents the reported recall from
    # becoming a tautology based on the assembler's own proposal.
    known_effect_expected = [
        episode for group_rows, episode in zip(grouped_rows, episodes)
        if episode["source"] in {"pg42", "pg35"} and _is_effect_pair(group_rows, supported_slots)
    ]
    known_effect_assembled = [
        episode for episode in known_effect_expected
        if episode["assembly"].get("decision") == "await_typed_oracle"
    ]
    unknown_episodes = [episode for episode in episodes if episode["source"] in {"pg69", "pg106"}]

    # Cross-sample copy/paste: pair a GET fragment from one implementation
    # with a POST fragment from another implementation, slot by slot.
    cross_sample: list[dict[str, Any]] = []
    for slot in sorted(supported_slots):
        get_row = _find_effect_row(evaluation_rows, source="pg42", method="GET", slot=slot)
        post_row = _find_effect_row(evaluation_rows, source="pg35", method="POST", slot=slot)
        if get_row is None or post_row is None:
            continue
        fragments = [fragment_from_row(get_row), fragment_from_row(post_row)]
        forward = assemble_rule_fragments(fragments, supported_slots=sorted(supported_slots))
        reverse = assemble_rule_fragments(list(reversed(fragments)), supported_slots=sorted(supported_slots))
        cross_sample.append({
            "slot": slot,
            "source_pair": ["pg42", "pg35"],
            "methods": ["GET", "POST"],
            "assembly": forward,
            "reverse_assembly": reverse,
            "order_invariant": forward.get("canonical_sha256") == reverse.get("canonical_sha256"),
        })

    effect_rows = [row for row in evaluation_rows if fragment_from_row(row)["relation_atom"] == "effect_present"]
    p0_get = _find_effect_row(evaluation_rows, source="pg42", method="GET", slot="p0")
    p1_post = _find_effect_row(evaluation_rows, source="pg42", method="POST", slot="p1")
    decoy_post = next((row for row in evaluation_rows if row.get("source") == "pg106" and row.get("method") == "POST" and fragment_from_row(row)["relation_atom"] == "input_only_anomaly"), None)
    negative_cases: dict[str, dict[str, Any]] = {}
    if p0_get is not None and p1_post is not None:
        negative_cases["slot_conflict"] = assemble_rule_fragments([fragment_from_row(p0_get), fragment_from_row(p1_post)], supported_slots=sorted(supported_slots))
    if p0_get is not None:
        same = fragment_from_row(p0_get)
        negative_cases["duplicate_evidence"] = assemble_rule_fragments([same, same], supported_slots=sorted(supported_slots))
        bad_binding = dict(same)
        bad_binding["binding_sha256"] = "0" * 64
        negative_cases["invalid_binding"] = assemble_rule_fragments([same, bad_binding], supported_slots=sorted(supported_slots))
    if p0_get is not None and decoy_post is not None:
        negative_cases["input_only_decoy"] = assemble_rule_fragments([fragment_from_row(p0_get), fragment_from_row(decoy_post)], supported_slots=sorted(supported_slots))

    checks = {
        "training_row_count": len(train_rows) == 32,
        "evaluation_row_count": len(evaluation_rows) == 578,
        "episode_count": len(episodes) == 289,
        "supported_slots_are_generic": supported_slots == {f"p{index}" for index in range(8)},
        "binding_attestation_valid": all(binding_attestation_valid(row["model_input"], expected_sha256=CANONICAL_BINDING_SHA256) for row in train_rows + evaluation_rows),
        "model_input_oracle_blind": all(not model_input_has_forbidden_field(row["model_input"]) for row in train_rows + evaluation_rows),
        "fragments_have_bounded_context": all(fragment_from_row(row)["negative_control_clear"] and fragment_from_row(row)["fresh_target"] for row in train_rows + evaluation_rows),
        "known_effect_pair_count_positive": len(known_effect_expected) >= 200,
        "known_effect_recall": bool(known_effect_expected) and (len(known_effect_assembled) / len(known_effect_expected)) >= 0.80,
        "unknown_and_decoy_abstain": all(episode["assembly"].get("decision") != "await_typed_oracle" for episode in unknown_episodes),
        "cross_sample_recombination_count": len(cross_sample) == len(supported_slots),
        "cross_sample_recombination_valid": all(item["assembly"].get("decision") == "await_typed_oracle" for item in cross_sample),
        "cross_sample_order_invariant": all(item["order_invariant"] for item in cross_sample),
        "episode_order_invariant": all(bool(episode["order_invariant"]) for episode in episodes if episode["assembly"].get("decision") == "await_typed_oracle"),
        "slot_conflict_abstain": negative_cases.get("slot_conflict", {}).get("decision") == "abstain" and negative_cases.get("slot_conflict", {}).get("reason") == "slot_conflict",
        "duplicate_evidence_abstain": negative_cases.get("duplicate_evidence", {}).get("reason") == "duplicate_evidence",
        "invalid_binding_abstain": negative_cases.get("invalid_binding", {}).get("reason") == "invalid_fragment_binding",
        "input_only_decoy_abstain": negative_cases.get("input_only_decoy", {}).get("reason") == "input_only_fragment_cannot_supply_effect",
        "typed_oracle_not_called": all(not episode["typed_oracle_called"] for episode in episodes) and all(not episode["confirmed_positive"] for episode in episodes),
        "assembly_non_executable": all(episode["assembly"].get("executable") is False for episode in episodes),
        "promotion_disabled": all(not episode["training_eligible"] and not episode["long_term_memory_write"] for episode in episodes),
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_fragment_composition_diagnostic" if not blocked else "blocked"
    metrics = {
        "training_row_count": len(train_rows),
        "evaluation_row_count": len(evaluation_rows),
        "episode_count": len(episodes),
        "supported_slot_count": len(supported_slots),
        "known_effect_expected_pair_count": len(known_effect_expected),
        "known_effect_assembled_pair_count": len(known_effect_assembled),
        "known_effect_pair_count": len(known_effect_expected),
        "known_effect_assembly_recall": round(len(known_effect_assembled) / len(known_effect_expected), 6) if known_effect_expected else 0.0,
        "unknown_or_decoy_episode_count": len(unknown_episodes),
        "unknown_or_decoy_abstain_rate": round(sum(int(item["assembly"].get("decision") != "await_typed_oracle") for item in unknown_episodes) / len(unknown_episodes), 6) if unknown_episodes else 0.0,
        "cross_sample_recombination_count": len(cross_sample),
        "cross_sample_valid_rate": round(sum(int(item["assembly"].get("decision") == "await_typed_oracle") for item in cross_sample) / len(cross_sample), 6) if cross_sample else 0.0,
        "negative_case_count": len(negative_cases),
        "negative_case_abstain_rate": round(sum(int(item.get("decision") == "abstain") for item in negative_cases.values()) / len(negative_cases), 6) if negative_cases else 0.0,
        "typed_oracle_called_count": 0,
        "confirmed_positive_count": 0,
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg109-fragment-composition-report-v1",
        "status": status,
        "source": {
            "training_source": "PG101 train role PG36 north seeds 361/367",
            "evaluation_source": "PG105 PG42/PG35/PG76/PG69 plus PG106 independent decoy",
            "training_row_count": len(train_rows),
            "evaluation_row_count": len(evaluation_rows),
            "episode_count": len(episodes),
            "source_hashes": {
                "train_dataset": _sha256_file(TRAIN_PATH),
                "pg105_dataset": _sha256_file(PG105_DATASET_PATH),
                "pg105_trace": _sha256_file(PG105_TRACE_PATH),
                "pg106_dataset": _sha256_file(PG106_DATASET_PATH),
                "pg106_trace": _sha256_file(PG106_TRACE_PATH),
                "assembler": _sha256_file(ASSEMBLER_PATH),
                "inducer": _sha256_file(INDUCER_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "controller": {
            "architecture": "family-free atomic Rule IR fragment assembler",
            "atoms": ["effect_present", "probe_binding_valid", "get_post_repeat", "negative_control_clear"],
            "copy_paste_order_invariant": True,
            "cross_sample_recombination": True,
            "family_labels_in_fragments": False,
            "oracle_labels_in_fragments": False,
            "typed_oracle_called": False,
            "promotion_allowed": False,
        },
        "metrics": metrics,
        "checks": checks,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "negative_cases": negative_cases,
        "cross_sample": cross_sample,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "fragment_composition_evaluation_only", "reason": "composition proposals still require independent typed replay, OOD validation and human/Codex review"},
        "safety": {"loopback_only": True, "external_network": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "request_field_names_stored": False, "request_values_stored": False, "evaluator_labels_in_model_input": False, "family_labels_in_fragments": False, "fresh_reset_per_fragment": True, "matched_negative_controls": True, "evidence_hashes_verified": True, "long_term_memory_write": False},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible_episodes = []
    trace_steps = []
    for episode in episodes:
        visible = {
            "episode_group": episode["episode_group"],
            "source": episode["source"],
            "implementations": episode["implementations"],
            "methods": episode["methods"],
            "fragments": episode["fragments"],
            "assembly": episode["assembly"],
            "reverse_assembly": episode["reverse_assembly"],
            "order_invariant": episode["order_invariant"],
            "fresh_reset_per_fragment": episode["fresh_reset_per_fragment"],
            "negative_control_per_fragment": episode["negative_control_per_fragment"],
            "typed_oracle_called": False,
            "confirmed_positive": False,
        }
        visible_episodes.append(visible)
        for fragment in episode["fragments"]:
            trace_steps.append({
                "episode_group": episode["episode_group"],
                "source": episode["source"],
                "method": fragment["method"],
                "fragment": fragment,
                "assembly_decision": episode["assembly"].get("decision"),
                "assembly_reason": episode["assembly"].get("reason"),
                "evidence_sha256": fragment["evidence_sha256"],
                "fresh_reset": fragment["fresh_target"],
                "negative_control_matched": fragment["negative_control_clear"],
                "typed_oracle_called": False,
                "confirmed_positive": False,
            })
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg109-fragment-composition-visible-dataset-v1", "dataset_id": "pg109-fragment-composition-visible", "evaluation_only": True, "training_eligible": False, "proposal_sha256": proposal["proposal_sha256"], "episodes": visible_episodes, "cross_sample": cross_sample, "negative_cases": negative_cases, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg109-fragment-composition-trace-v1", "evaluation_only": True, "training_eligible": False, "proposal_sha256": proposal["proposal_sha256"], "steps": trace_steps, "evaluator_labels_in_trace": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg109-fragment-composition-protocol-v1", "purpose": "test family-free copy/paste assembly across seeds, order and independent implementations", "training_contract": {"row_count": len(train_rows), "family_visible": False, "oracle_visible": False}, "required_atoms": ["effect_present", "probe_binding_valid", "get_post_repeat", "negative_control_clear"], "negative_contract": ["slot_conflict_abstain", "duplicate_evidence_abstain", "invalid_binding_abstain", "input_only_decoy_abstain"], "gate": {"known_effect_assembly_recall_min": 0.80, "unknown_or_decoy_abstain_rate": 1.0, "cross_sample_valid_rate": 1.0, "promotion_on_pass": False}, "result": {"status": status, "blocking_reasons": blocked}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(f"# PG-109 Rule Fragment Composition / 原子片段组装\n\n状态：`{status}`；episode：`{metrics['episode_count']}`；已知效果组装召回：`{metrics['known_effect_assembly_recall']}`；族外/decoy 弃权率：`{metrics['unknown_or_decoy_abstain_rate']}`。\n\n跨样本重组：`{metrics['cross_sample_recombination_count']}`；重组有效率：`{metrics['cross_sample_valid_rate']}`；负例硬门弃权率：`{metrics['negative_case_abstain_rate']}`。\n\n该 Rule IR 只由可复用原子组成、不可执行、必须等待 typed oracle；训练集和长期记忆关闭。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["status"], "episode_count": result["metrics"]["episode_count"], "known_effect_assembly_recall": result["metrics"]["known_effect_assembly_recall"], "unknown_or_decoy_abstain_rate": result["metrics"]["unknown_or_decoy_abstain_rate"], "cross_sample_valid_rate": result["metrics"]["cross_sample_valid_rate"], "negative_case_abstain_rate": result["metrics"]["negative_case_abstain_rate"], "training_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2))
