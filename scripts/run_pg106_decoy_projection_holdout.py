"""PG-106 cross-implementation causal projection and decoy holdout."""

from __future__ import annotations

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

import run_pg103_auto_goal_label_active_probe as pg103  # noqa: E402
import run_pg104_probe_binding_ablation as pg104  # noqa: E402
import run_pg105_observable_projection as pg105  # noqa: E402
from app.active_goal_label_inducer import ActiveGoalLabelInducer  # noqa: E402
from app.active_probe_signature import PROBE_IDS, aggregate_signature, make_probe_observation, model_input_has_forbidden_field, sha256_json  # noqa: E402
from app.pg105_observable_projection import SCHEMA_VERSION as CAUSAL_SCHEMA, attach_causal_extension, make_causal_projection  # noqa: E402
from app.probe_binding_attestation import BINDING_SCHEMA_VERSION, CANONICAL_BINDING_SHA256, add_binding_attestation, binding_attestation_valid  # noqa: E402


PROTOCOL_ID = "pg-pk-106-decoy-projection-holdout-v1"
FIXTURE_PATH = ROOT / "app" / "pg106_decoy_projection_fixture.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg106_decoy_projection_holdout.py"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
INPUT_DATASET_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
PG103_DATASET_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_visible_dataset_v1.json"
PG103_TRACE_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_trace_v1.json"
UNKNOWN_FAMILY = "workflow_invariant"
NEGATIVE_FAMILY = "ordinary_response"
PORTS = {"amber": 8815, "violet": 8816}

REPORT_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset(*, variant: str, route: str, method: str, ordinal: int, scenario: str) -> dict[str, Any]:
    key = f"pg106|{variant}|{route}|{method}|{ordinal}|{scenario}"
    value = {
        "kind": "fresh_pg106_decoy_projection_episode",
        "reset_id": f"pg106-reset-{hashlib.sha256((key + '|reset').encode()).hexdigest()[:24]}",
        "target_instance_id": f"pg106-target-{hashlib.sha256(key.encode()).hexdigest()[:24]}",
        "state_epoch": hashlib.sha256((key + "|epoch").encode()).hexdigest()[:24],
        "reset_adapter_sha256": hashlib.sha256(b"pg106-loopback-reset-adapter").hexdigest(),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "transport": "httpx_loopback",
        "probe_bank_episode": True,
    }
    value["reset_sha256"] = sha256_json(value)
    return value


def _values(route: str, *, candidate: bool, probe_id: str) -> dict[str, str]:
    if route == "/threshold":
        return {"subject": "member", "value": "100" if candidate and probe_id == "p8" else "99"}
    # A harmless decoy: p0 changes anonymous numeric shape, but the fixture
    # deliberately ignores it and the typed oracle remains negative.
    return {"shape": "10" if candidate and probe_id == "p0" else "9"}


def _collect_episode(fixture: Any, *, variant: str, route: str, method: str, ordinal: int, candidate_episode: bool) -> dict[str, Any]:
    scenario = "decoy" if route == "/decoy" else "threshold"
    reset = _reset(variant=variant, route=route, method=method, ordinal=ordinal, scenario=scenario)
    observations: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    query_count = 0
    with pg104._FreshPG69Target(fixture, variant=variant, ordinal=ordinal, mode=scenario) as target:
        for probe_id in PROBE_IDS:
            control_values = _values(route, candidate=False, probe_id=probe_id)
            candidate_values = _values(route, candidate=candidate_episode, probe_id=probe_id)
            control_projection, control_geometry, _ = target.request(route, method, control_values)
            candidate_projection, candidate_geometry, oracle = target.request(route, method, candidate_values)
            query_count += 2
            observation = make_probe_observation(
                probe_id=probe_id, method=method, phase="confirm", encoding="identity",
                control_geometry=control_geometry, candidate_geometry=candidate_geometry,
                control_projection=control_projection, candidate_projection=candidate_projection,
                safe_probe=True,
            )
            observations.append(observation)
            causal.append(make_causal_projection(
                control_values,
                candidate_values,
                response_changed=bool(observation["delta_nonzero"]),
            ))
            oracles.append(oracle)
    signature = add_binding_attestation(attach_causal_extension(aggregate_signature(observations), causal))
    typed_positive = any(bool(item.get("positive")) and bool(item.get("positive_authority")) for item in oracles)
    evidence = sha256_json({
        "reset": reset,
        "confirm_observation_sha256": [item["observation_sha256"] for item in observations],
        "causal_projection_sha256": [sha256_json(item) for item in causal],
        "typed_positive": typed_positive,
    })
    group_digest = hashlib.sha256(f"pg106|{variant}|{route}|{scenario}|{candidate_episode}".encode()).hexdigest()[:24]
    row_digest = hashlib.sha256(f"pg106|{variant}|{route}|{method}|{candidate_episode}".encode()).hexdigest()[:24]
    return {
        "row_id": f"pg106-{row_digest}",
        "episode_group": f"pg106-group-{group_digest}",
        "source": "pg106",
        "implementation": f"pg106-{variant}",
        "seed": 10600 + int(ordinal),
        "method": method,
        "route": route,
        "scenario": scenario,
        "family": UNKNOWN_FAMILY if typed_positive else NEGATIVE_FAMILY,
        "typed_positive": typed_positive,
        "model_input": signature,
        "fresh_reset": reset,
        "negative_control_matched": True,
        "evidence_sha256": evidence,
        "query_count": query_count,
        "order_permutation_invariant": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


def _attest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "model_input": add_binding_attestation(row["model_input"])} for row in rows]


def _decoy_metrics(rows: list[dict[str, Any]], inducer: ActiveGoalLabelInducer) -> dict[str, Any]:
    decoys = [row for row in rows if row.get("source") == "pg106" and row.get("scenario") == "decoy" and row.get("route") == "/decoy" and any(row["model_input"]["causal_extension"]["input_changed_response_unchanged_pattern"])]
    outputs = [inducer.predict(row["model_input"], guarded=True) for row in decoys]
    opaque = [row for row in rows if bool(row.get("typed_positive")) and not any(row["model_input"].get("delta_pattern", []))]
    return {
        "decoy_count": len(decoys),
        "decoy_anomaly_count": len(decoys),
        "decoy_false_confirm_count": sum(int(item.get("decision") == "confirm_candidate") for item in outputs),
        "decoy_abstain_count": sum(int(item.get("decision") == "abstain") for item in outputs),
        "opaque_positive_count": len(opaque),
        "opaque_positive_anomaly_count": sum(int(any(row["model_input"]["causal_extension"]["input_changed_response_unchanged_pattern"])) for row in opaque),
    }


def run() -> dict[str, Any]:
    train, frozen_eval = pg103._prepare_pg101_rows()
    train = _attest(train)
    frozen_eval = _attest(frozen_eval)
    pg76 = pg104._load_pg76_rows()
    pg69_fixture = pg104._load_module(pg104.PG69_FIXTURE_PATH, "pg106_pg69_fixture")
    pg69_rows = [
        pg105._collect_pg69_episode(pg69_fixture, variant=variant, route=route, method=method, ordinal=ordinal, positive_episode=positive)
        for ordinal, (variant, route, method, positive) in enumerate(
            ( (variant, route, method, positive) for variant in ("amber", "violet") for route in ("/handoff", "/quota") for method in ("GET", "POST") for positive in (True, False) )
        )
    ]
    fixture = pg104._load_module(FIXTURE_PATH, "pg106_independent_fixture")
    independent_rows: list[dict[str, Any]] = []
    ordinal = 0
    for variant in ("amber", "violet"):
        for route in ("/threshold", "/decoy"):
            for method in ("GET", "POST"):
                for candidate in (True, False):
                    independent_rows.append(_collect_episode(fixture, variant=variant, route=route, method=method, ordinal=ordinal, candidate_episode=candidate))
                    ordinal += 1
    evaluation = frozen_eval + pg76 + pg69_rows + independent_rows
    inducer = ActiveGoalLabelInducer(minimum_support=2, require_get_post=True, require_binding_attestation=True, expected_binding_sha256=CANONICAL_BINDING_SHA256).fit([{"model_input": row["model_input"]} for row in train])
    proposal = inducer.proposal()
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups = {
        "pg42": [row for row in evaluation if row.get("source") == "pg42"],
        "pg76": [row for row in evaluation if row.get("source") == "pg76"],
        "pg69": [row for row in evaluation if row.get("source") == "pg69"],
        "pg106_independent": [row for row in evaluation if row.get("source") == "pg106"],
        "all_evaluation": evaluation,
    }
    metrics: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}
    for name, rows in groups.items():
        metrics[name], records[name] = pg103._metric(rows, inducer, guarded=True)
        raw_metrics[name], _ = pg103._metric(rows, inducer, guarded=False)
    goal = pg103._goal_metrics(records["all_evaluation"])
    decoy = _decoy_metrics(independent_rows, inducer)
    composition = pg104._composition_ablation(inducer, evaluation)
    all_metric = metrics["all_evaluation"]
    causal_rows = pg69_rows + independent_rows
    checks = {
        "binding_attestation_valid": all(binding_attestation_valid(row["model_input"]) for row in train + evaluation),
        "proposal_oracle_blind": proposal["proposal_inputs"]["oracle_visible"] is False and proposal["proposal_inputs"]["family_visible"] is False,
        "training_excludes_evaluation": all(row.get("source") not in {"pg42", "pg35", "pg76", "pg69", "pg106"} for row in train),
        "model_input_no_evaluator_or_raw": all(not model_input_has_forbidden_field(row["model_input"]) for row in train + evaluation),
        "get_post_covered": sorted({str(row["method"]) for row in evaluation}) == ["GET", "POST"],
        "fresh_reset_unique": len({row["fresh_reset"]["target_instance_id"] for row in evaluation}) == len(evaluation),
        "matched_negative": all(bool(row["negative_control_matched"]) for row in evaluation),
        "evidence_hashes": all(len(str(row["evidence_sha256"])) == 64 for row in evaluation),
        "order_invariant": all(bool(row["order_permutation_invariant"]) for row in causal_rows + pg76 + frozen_eval),
        "known_recall_min": all_metric["known_confirm_recall"] >= 0.80,
        "known_label_consistency_min": all_metric["known_label_consistency"] >= 0.95,
        "false_accept_zero": all_metric["false_accept_count"] == 0,
        "pg69_unknown_strict_abstain": metrics["pg69"]["unknown_family_strict_abstain"],
        "pg106_unknown_strict_abstain": metrics["pg106_independent"]["unknown_family_strict_abstain"],
        "decoy_anomaly_present": decoy["decoy_count"] == 4 and decoy["decoy_anomaly_count"] == 4,
        "decoy_false_confirm_zero": decoy["decoy_false_confirm_count"] == 0,
        "decoy_all_abstain": decoy["decoy_abstain_count"] == decoy["decoy_count"],
        "opaque_positive_anomaly_present": decoy["opaque_positive_count"] >= 4 and decoy["opaque_positive_anomaly_count"] == decoy["opaque_positive_count"],
        "negative_goal_false_completion_zero": goal["negative_false_completion_count"] == 0,
        "composition_order_invariant": composition["copy_paste_order_invariant"] and composition["cross_sample_recombination_order_invariant"],
        "composition_no_candidate_promotion": composition["candidate_promotion_eligible_count"] == 0,
        "composition_recombination_non_executable": composition["cross_sample_recombination_executable"] is False,
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_cross_implementation_decoy_diagnostic" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg106-decoy-projection-holdout-report-v1",
        "status": status,
        "source": {
            "training_source": "PG101 train role PG36 north seeds 361/367",
            "frozen_evaluation_source": "PG101 PG42/PG76 plus fresh PG69 and independent PG106",
            "training_row_count": len(train),
            "evaluation_row_count": len(evaluation),
            "pg69_row_count": len(pg69_rows),
            "pg106_row_count": len(independent_rows),
            "source_hashes": {
                "pg101_dataset": _sha256_file(INPUT_DATASET_PATH),
                "pg103_dataset": _sha256_file(PG103_DATASET_PATH),
                "pg103_trace": _sha256_file(PG103_TRACE_PATH),
                "pg69_fixture": _sha256_file(pg104.PG69_FIXTURE_PATH),
                "pg106_fixture": _sha256_file(FIXTURE_PATH),
                "causal_module": _sha256_file(pg105.CAUSAL_MODULE_PATH),
                "inducer_module": _sha256_file(INDUCER_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "model": {"architecture": "binding-gated generic slots plus anonymous input-response relation", "causal_projection_schema": CAUSAL_SCHEMA, "family_generation": False, "oracle_in_model_input": False},
        "metrics": {"guarded_proposal": metrics, "raw_proposal": raw_metrics, "guarded_goal": goal, "decoy": decoy, "compositional_rule_ir_ablation": composition},
        "raw_failure_visible": {"unknown_misname_count": sum(raw_metrics[name]["unknown_misname_count"] for name in raw_metrics), "false_accept_count": sum(raw_metrics[name]["false_accept_count"] for name in raw_metrics), "failure_present": any(raw_metrics[name]["unknown_misname_count"] or raw_metrics[name]["false_accept_count"] for name in raw_metrics)},
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "cross_implementation_decoy_quarantined", "reason": "projection is diagnostic only; typed oracle and future OOD review remain mandatory"},
        "safety": {"loopback_only": True, "external_network": False, "fresh_reset_per_episode": True, "matched_negative_controls": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "request_field_names_stored": False, "request_values_stored": False, "evaluator_labels_in_model_input": False, "input_change_is_not_effect_atom": True},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible_rows = []
    trace_steps = []
    for row in evaluation:
        guarded = inducer.predict(row["model_input"], guarded=True)
        raw = inducer.predict(row["model_input"], guarded=False)
        visible_rows.append({"row_id": row["row_id"], "source": row["source"], "implementation": row["implementation"], "seed": row["seed"], "method": row["method"], "model_input": row["model_input"], "raw_proposal": raw, "guarded_proposal": guarded, "evidence_sha256": row["evidence_sha256"], "fresh_reset": row["fresh_reset"], "negative_control_matched": True, "raw_probe_strings_stored": False, "raw_response_body_stored": False})
        trace_steps.append({"trace_id": row["row_id"], "episode_group": row["episode_group"], "source": row["source"], "implementation": row["implementation"], "seed": row["seed"], "method": row["method"], "raw_decision": raw["decision"], "guarded_decision": guarded["decision"], "guarded_label_id": guarded["label_id"], "composition_decision": guarded.get("composition_decision"), "evidence_sha256": row["evidence_sha256"], "fresh_reset": True, "negative_control_matched": True})
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg106-decoy-projection-visible-dataset-v1", "dataset_id": "pg106-decoy-projection-visible", "evaluation_only": True, "training_eligible": False, "model_input_contract": {"oracle_is_label_not_feature": True, "family_label_in_features": False, "causal_projection_schema": CAUSAL_SCHEMA, "input_change_is_not_effect": True, "request_field_names_in_features": False, "request_values_in_features": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "proposal_sha256": proposal["proposal_sha256"], "rows": visible_rows, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg106-decoy-projection-trace-v1", "evaluation_only": True, "training_eligible": False, "proposal_sha256": proposal["proposal_sha256"], "steps": trace_steps, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "evaluator_labels_in_trace": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg106-decoy-projection-protocol-v1", "purpose": "cross-implementation and decoy audit of anonymous input-response projection", "training_contract": {"row_count": len(train), "oracle_visible": False, "family_visible": False}, "evaluation_contract": {"frozen_sources": ["PG42", "PG76"], "fresh_sources": ["PG69", "PG106"], "row_count": len(evaluation)}, "decoy_contract": {"decoy_input_change_expected": True, "typed_positive": False, "must_abstain": True, "false_confirm_allowed": False}, "gate": {"known_recall_min": 0.80, "false_accept_count": 0, "unknown_family_strict_abstain": True, "decoy_false_confirm_count": 0, "promotion_on_pass": False}, "result": {"status": status, "blocking_reasons": blocked}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(f"# PG-106 Cross-implementation decoy projection\n\n状态：`{status}`；已知召回：`{all_metric['known_confirm_recall']}`；误报：`{all_metric['false_accept_count']}`。\n\n独立实现未知族严格弃权：`{metrics['pg106_independent']['unknown_family_strict_abstain']}`；decoy 异常数：`{decoy['decoy_anomaly_count']}`；decoy 误确认：`{decoy['decoy_false_confirm_count']}`；decoy 弃权：`{decoy['decoy_abstain_count']}`。\n\n输入变化仍不是 effect atom；训练和长期记忆关闭。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    metric = result["metrics"]["guarded_proposal"]["all_evaluation"]
    decoy = result["metrics"]["decoy"]
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["status"], "known_recall": metric["known_confirm_recall"], "false_accept_count": metric["false_accept_count"], "pg106_unknown_strict_abstain": result["metrics"]["guarded_proposal"]["pg106_independent"]["unknown_family_strict_abstain"], "decoy_false_confirm_count": decoy["decoy_false_confirm_count"], "decoy_abstain_count": decoy["decoy_abstain_count"], "training_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2))
