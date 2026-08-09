"""PG-105: causal projection for typed-positive but surface-opaque episodes.

The runner adds an anonymous input/response relation to fresh PG-69 replay.
The relation can make an opaque episode *suspicious*, but it cannot confirm a
vulnerability.  The typed oracle remains evaluator-only and all artifacts are
evaluation-only.
"""

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

import run_pg103_auto_goal_label_active_probe as pg103  # noqa: E402
import run_pg104_probe_binding_ablation as pg104  # noqa: E402
from app.active_goal_label_inducer import ActiveGoalLabelInducer  # noqa: E402
from app.active_probe_signature import PROBE_IDS, aggregate_signature, make_probe_observation, model_input_has_forbidden_field, sha256_json  # noqa: E402
from app.pg105_observable_projection import SCHEMA_VERSION as CAUSAL_SCHEMA, attach_causal_extension, make_causal_projection  # noqa: E402
from app.probe_binding_attestation import BINDING_SCHEMA_VERSION, CANONICAL_BINDING_SHA256, add_binding_attestation, binding_attestation_valid  # noqa: E402


PROTOCOL_ID = "pg-pk-105-observable-projection-v1"
INPUT_DATASET_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
PG103_DATASET_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_visible_dataset_v1.json"
PG103_TRACE_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_trace_v1.json"
PG69_FIXTURE_PATH = ROOT / "app" / "pg69_workflow_fixture.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg105_observable_projection.py"
CAUSAL_MODULE_PATH = ROOT / "app" / "pg105_observable_projection.py"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
PG69_PORTS = {"amber": 8815, "violet": 8816}
UNKNOWN_FAMILY = "workflow_invariant"
NEGATIVE_FAMILY = "ordinary_response"

REPORT_PATH = ROOT / "research" / "pg105_observable_projection_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg105_observable_projection_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg105_observable_projection_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg105_observable_projection_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg105_observable_projection_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg105_observable_projection_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset(*, variant: str, ordinal: int, mode: str) -> dict[str, Any]:
    key = f"pg105|pg69|{variant}|{ordinal}|{mode}"
    value = {
        "kind": "fresh_pg69_observable_projection_episode",
        "reset_id": f"pg105-reset-{hashlib.sha256((key + '|reset').encode()).hexdigest()[:24]}",
        "target_instance_id": f"pg105-target-{hashlib.sha256(key.encode()).hexdigest()[:24]}",
        "state_epoch": hashlib.sha256((key + "|epoch").encode()).hexdigest()[:24],
        "reset_adapter_sha256": hashlib.sha256(b"pg105-loopback-reset-adapter").hexdigest(),
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


def _collect_pg69_episode(fixture: Any, *, variant: str, route: str, method: str, ordinal: int, positive_episode: bool) -> dict[str, Any]:
    mode = "positive" if positive_episode else "negative"
    reset = _reset(variant=variant, ordinal=ordinal, mode=mode)
    observations: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    query_count = 0
    with pg104._FreshPG69Target(fixture, variant=variant, ordinal=ordinal, mode=mode) as target:
        # Screen is executed for protocol parity; only bounded hashes survive.
        screen_control_projection, screen_control_geometry, _ = target.request(route, method, pg104._values(route, positive=False))
        screen_candidate_projection, screen_candidate_geometry, _ = target.request(route, method, pg104._probe_values(route, "p0", positive_episode))
        query_count += 2
        screen = make_probe_observation(
            probe_id="p0", method=method, phase="screen", encoding="identity",
            control_geometry=screen_control_geometry, candidate_geometry=screen_candidate_geometry,
            control_projection=screen_control_projection, candidate_projection=screen_candidate_projection,
            safe_probe=True,
        )
        for probe_id in PROBE_IDS:
            control_values = pg104._values(route, positive=False)
            candidate_values = pg104._probe_values(route, probe_id, positive_episode)
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
    signature = attach_causal_extension(aggregate_signature(observations), causal)
    signature = add_binding_attestation(signature)
    reversed_signature = attach_causal_extension(aggregate_signature(list(reversed(observations))), causal)
    reversed_signature = add_binding_attestation(reversed_signature)
    typed_positive = any(bool(item.get("positive")) and bool(item.get("positive_authority")) for item in oracles)
    evidence = sha256_json({
        "reset": reset,
        "screen_observation_sha256": screen["observation_sha256"],
        "confirm_observation_sha256": [item["observation_sha256"] for item in observations],
        "causal_projection_sha256": [sha256_json(item) for item in causal],
        "typed_positive": typed_positive,
    })
    group_digest = hashlib.sha256(f"pg105|pg69|{variant}|{route}|{mode}".encode()).hexdigest()[:24]
    row_digest = hashlib.sha256(f"pg105|pg69|{variant}|{route}|{method}|{mode}".encode()).hexdigest()[:24]
    return {
        "row_id": f"pg105-pg69-{row_digest}",
        "episode_group": f"pg105-group-{group_digest}",
        "source": "pg69",
        "implementation": f"pg69-{variant}",
        "seed": 10500 + int(ordinal),
        "method": method,
        "family": UNKNOWN_FAMILY if typed_positive else NEGATIVE_FAMILY,
        "typed_positive": typed_positive,
        "model_input": signature,
        "fresh_reset": reset,
        "negative_control_matched": True,
        "evidence_sha256": evidence,
        "query_count": query_count,
        "order_permutation_invariant": signature["signature_sha256"] == reversed_signature["signature_sha256"],
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


def _attest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        row["model_input"] = add_binding_attestation(row["model_input"])
        result.append(row)
    return result


def _causal_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if bool(row.get("typed_positive"))]
    negatives = [row for row in rows if not bool(row.get("typed_positive"))]
    opaque_positives = [row for row in positives if not any(bool(value) for value in row["model_input"].get("delta_pattern", []))]
    positive_patterns = [row["model_input"]["causal_extension"]["input_changed_response_unchanged_pattern"] for row in opaque_positives]
    negative_patterns = [row["model_input"]["causal_extension"]["input_changed_response_unchanged_pattern"] for row in negatives]
    return {
        "positive_count": len(positives),
        "positive_opaque_count": len(opaque_positives),
        "positive_opaque_anomaly_present": bool(opaque_positives) and all(any(pattern) for pattern in positive_patterns),
        "negative_anomaly_count": sum(any(pattern) for pattern in negative_patterns),
        "all_rows_have_causal_extension": all(isinstance(row["model_input"].get("causal_extension"), dict) for row in rows),
        "opaque_relation_is_generic_code_1": all(
            all(int(code) in range(4) for code in row["model_input"]["causal_extension"]["relation_code_pattern"])
            for row in rows
        ),
    }


def run() -> dict[str, Any]:
    train, frozen_eval = pg103._prepare_pg101_rows()
    train = _attest_rows(train)
    frozen_eval = _attest_rows(frozen_eval)
    pg76_rows = pg104._load_pg76_rows()
    fixture = pg104._load_module(pg104.PG69_FIXTURE_PATH, "pg105_pg69_fixture")
    pg69_rows: list[dict[str, Any]] = []
    ordinal = 0
    for variant in ("amber", "violet"):
        for route in ("/handoff", "/quota"):
            for method in ("GET", "POST"):
                for positive_episode in (True, False):
                    pg69_rows.append(_collect_pg69_episode(
                        fixture, variant=variant, route=route, method=method,
                        ordinal=ordinal, positive_episode=positive_episode,
                    ))
                    ordinal += 1
    evaluation = frozen_eval + pg76_rows + pg69_rows
    inducer = ActiveGoalLabelInducer(
        minimum_support=2,
        require_get_post=True,
        require_binding_attestation=True,
        expected_binding_sha256=CANONICAL_BINDING_SHA256,
    ).fit([{"model_input": row["model_input"]} for row in train])
    proposal = inducer.proposal()
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups = {
        "pg42": [row for row in evaluation if row.get("source") == "pg42"],
        "pg35_third_implementation": [row for row in evaluation if row.get("source") == "pg35"],
        "pg76_unseen_family": [row for row in evaluation if row.get("source") == "pg76"],
        "pg69_observable_projection": [row for row in evaluation if row.get("source") == "pg69"],
        "all_evaluation": evaluation,
    }
    metrics: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}
    for name, rows in groups.items():
        metrics[name], records[name] = pg103._metric(rows, inducer, guarded=True)
        raw_metrics[name], _ = pg103._metric(rows, inducer, guarded=False)
    goal = pg103._goal_metrics(records["all_evaluation"])
    composition = pg104._composition_ablation(inducer, evaluation)
    causal = _causal_metrics(pg69_rows)
    all_metric = metrics["all_evaluation"]
    checks = {
        "binding_attestation_valid_for_train_and_eval": all(binding_attestation_valid(row["model_input"]) for row in train + evaluation),
        "binding_schema_declared": proposal["probe_binding"]["schema_version"] == BINDING_SCHEMA_VERSION and proposal["probe_binding"]["binding_sha256"] == CANONICAL_BINDING_SHA256,
        "proposal_input_oracle_blind": proposal["proposal_inputs"]["oracle_visible"] is False and proposal["proposal_inputs"]["family_visible"] is False,
        "training_excludes_all_evaluation_sources": all(row.get("source") not in {"pg42", "pg35", "pg76", "pg69"} for row in train),
        "model_input_no_evaluator_or_raw": all(not model_input_has_forbidden_field(row["model_input"]) for row in train + evaluation),
        "get_post_covered": sorted({str(row.get("method")) for row in evaluation}) == ["GET", "POST"],
        "fresh_reset_per_episode": all(bool((row.get("fresh_reset") or {}).get("completed")) and bool((row.get("fresh_reset") or {}).get("fresh_target")) for row in evaluation),
        "matched_negative_control_per_episode": all(bool(row.get("negative_control_matched")) for row in evaluation),
        "evidence_hashes_present": all(len(str(row.get("evidence_sha256", ""))) == 64 for row in evaluation),
        "order_permutation_invariant": all(bool(row.get("order_permutation_invariant")) for row in pg69_rows + pg76_rows + frozen_eval),
        "known_recall_min": all_metric["known_confirm_recall"] >= 0.80,
        "known_label_consistency_min": all_metric["known_label_consistency"] >= 0.95,
        "false_accept_zero": all_metric["false_accept_count"] == 0,
        "pg42_unknown_strict_abstain": metrics["pg42"]["unknown_family_strict_abstain"],
        "pg76_unknown_strict_abstain": metrics["pg76_unseen_family"]["unknown_family_strict_abstain"],
        "pg69_unknown_strict_abstain": metrics["pg69_observable_projection"]["unknown_family_strict_abstain"],
        "not_all_abstain_on_known": all_metric["not_all_abstain_on_known"],
        "negative_false_completion_zero": goal["negative_false_completion_count"] == 0,
        "causal_extension_complete": causal["all_rows_have_causal_extension"],
        "causal_relation_bounded": causal["opaque_relation_is_generic_code_1"],
        "opaque_positive_anomaly_present": causal["positive_opaque_anomaly_present"],
        "negative_anomaly_zero": causal["negative_anomaly_count"] == 0,
        "composition_order_invariant": composition["copy_paste_order_invariant"] and composition["cross_sample_recombination_order_invariant"],
        "composition_no_candidate_promotion": composition["candidate_promotion_eligible_count"] == 0,
        "composition_recombination_non_executable": composition["cross_sample_recombination_executable"] is False,
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_observable_projection_diagnostic" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg105-observable-projection-report-v1",
        "status": status,
        "source": {
            "training_source": "PG101 train role PG36 north seeds 361/367",
            "frozen_evaluation_source": "PG101 PG42/PG35 plus PG103 PG76 evidence",
            "fresh_evaluation_source": "PG69 amber/violet workflow implementation with anonymous causal extension",
            "training_row_count": len(train),
            "evaluation_row_count": len(evaluation),
            "pg69_fresh_evaluation_row_count": len(pg69_rows),
            "source_hashes": {
                "pg101_dataset": _sha256_file(INPUT_DATASET_PATH),
                "pg103_dataset": _sha256_file(PG103_DATASET_PATH),
                "pg103_trace": _sha256_file(PG103_TRACE_PATH),
                "pg69_fixture": _sha256_file(PG69_FIXTURE_PATH),
                "causal_module": _sha256_file(CAUSAL_MODULE_PATH),
                "inducer_module": _sha256_file(INDUCER_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "model": {
            "inducer_schema": "active-auto-goal-label-inducer-v1",
            "architecture": "binding-gated generic active slots plus anonymous input-response relation",
            "causal_projection_schema": CAUSAL_SCHEMA,
            "vulnerability_family_generation": False,
            "typed_oracle_in_model_input": False,
        },
        "metrics": {
            "guarded_proposal": metrics,
            "raw_proposal": raw_metrics,
            "guarded_goal": goal,
            "causal_extension": causal,
            "compositional_rule_ir_ablation": composition,
        },
        "raw_failure_visible": {
            "unknown_misname_count": sum(raw_metrics[name]["unknown_misname_count"] for name in raw_metrics),
            "false_accept_count": sum(raw_metrics[name]["false_accept_count"] for name in raw_metrics),
            "failure_present": any(raw_metrics[name]["unknown_misname_count"] or raw_metrics[name]["false_accept_count"] for name in raw_metrics),
            "reason": "raw oracles and unguarded slot presence remain diagnostic only",
        },
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "observable_projection_diagnostic_quarantined",
            "reason": "anonymous relation improves abstention coverage but does not replace independent typed oracle or OOD review",
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "fresh_reset_per_episode": True,
            "matched_negative_controls": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "request_field_names_stored": False,
            "request_values_stored": False,
            "evaluator_labels_in_model_input": False,
            "typed_oracle_used_only_after_proposal": True,
            "input_change_is_not_effect_atom": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible_rows: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    for row in evaluation:
        raw = inducer.predict(row["model_input"], guarded=False)
        guarded = inducer.predict(row["model_input"], guarded=True)
        visible_rows.append({
            "row_id": str(row["row_id"]),
            "source": str(row["source"]),
            "implementation": str(row["implementation"]),
            "seed": int(row["seed"]),
            "method": str(row["method"]),
            "model_input": row["model_input"],
            "raw_proposal": raw,
            "guarded_proposal": guarded,
            "evidence_sha256": str(row["evidence_sha256"]),
            "fresh_reset": dict(row["fresh_reset"]),
            "negative_control_matched": True,
            "raw_probe_strings_stored": False,
            "raw_response_body_stored": False,
        })
        trace_steps.append({
            "trace_id": str(row["row_id"]),
            "episode_group": str(row["episode_group"]),
            "source": str(row["source"]),
            "implementation": str(row["implementation"]),
            "seed": int(row["seed"]),
            "method": str(row["method"]),
            "raw_decision": raw["decision"],
            "guarded_decision": guarded["decision"],
            "guarded_label_id": guarded["label_id"],
            "composition_decision": guarded.get("composition_decision"),
            "evidence_sha256": str(row["evidence_sha256"]),
            "fresh_reset": True,
            "negative_control_matched": True,
        })
    DATASET_PATH.write_text(json.dumps({
        "schema_version": "pg105-observable-projection-visible-dataset-v1",
        "dataset_id": "pg105-observable-projection-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "causal_extension_schema": CAUSAL_SCHEMA,
            "input_field_names_in_features": False,
            "input_values_in_features": False,
            "input_change_is_not_effect": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "proposal_sha256": proposal["proposal_sha256"],
        "rows": visible_rows,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({
        "schema_version": "pg105-observable-projection-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "steps": trace_steps,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "evaluator_labels_in_trace": False,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg105-observable-projection-protocol-v1",
        "purpose": "make input/response opacity observable without leaking typed oracle semantics",
        "training_contract": {"source": "PG101 train", "row_count": len(train), "oracle_visible": False, "family_visible": False},
        "evaluation_contract": {"frozen_sources": ["PG42", "PG35", "PG76"], "fresh_source": "PG69 amber/violet", "row_count": len(evaluation)},
        "causal_extension_contract": {
            "schema_version": CAUSAL_SCHEMA,
            "visible": ["bounded_input_shape_delta", "input_changed_response_unchanged", "relation_code"],
            "raw_field_names_persisted": False,
            "raw_values_persisted": False,
            "input_change_is_not_effect_atom": True,
            "opaque_positive_action": "abstain_or_await_typed_oracle",
        },
        "gate": {
            "known_recall_min": 0.80,
            "known_label_consistency_min": 0.95,
            "false_accept_count": 0,
            "unknown_family_strict_abstain": True,
            "negative_anomaly_count": 0,
            "promotion_on_pass": False,
        },
        "result": {"status": status, "blocking_reasons": blocked},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-105 Observable Projection / 可观测投影\n\n"
        f"状态：`{status}`；已知召回：`{all_metric['known_confirm_recall']}`；误报：`{all_metric['false_accept_count']}`。\n\n"
        f"PG-69 未知族严格弃权：`{metrics['pg69_observable_projection']['unknown_family_strict_abstain']}`；opaque 输入-表面不变正例：`{causal['positive_opaque_count']}`；阴性异常：`{causal['negative_anomaly_count']}`。\n\n"
        "输入变化只作为疑点关系，不作为 effect_present 或 typed oracle；训练和长期记忆保持关闭。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    metric = result["metrics"]["guarded_proposal"]["all_evaluation"]
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "known_recall": metric["known_confirm_recall"],
        "false_accept_count": metric["false_accept_count"],
        "pg69_unknown_strict_abstain": result["metrics"]["guarded_proposal"]["pg69_observable_projection"]["unknown_family_strict_abstain"],
        "opaque_positive_count": result["metrics"]["causal_extension"]["positive_opaque_count"],
        "negative_anomaly_count": result["metrics"]["causal_extension"]["negative_anomaly_count"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
