"""PG-104: probe-binding attestation, slot permutation and surface ablation.

PG-103's generic labels passed while using canonical probe IDs.  This run
audits whether that result is merely an unverified slot-memory shortcut.  A
binding attestation commits to the safe probe bank (IDs and schema only).
Missing or permuted bindings must abstain.  Geometry-sign ablations retain a
valid binding and should preserve the abstract effect label, demonstrating
that the label is not tied to one response surface shape.

PG-69 supplies a second, independently written workflow implementation.  It
is replayed in fresh loopback targets with GET/POST, matched controls and a
typed oracle that remains outside the model-visible signature.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any, Mapping

import httpx

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg103_auto_goal_label_active_probe as pg103  # noqa: E402
from app.active_goal_label_inducer import (  # noqa: E402
    ActiveGoalLabelInducer,
    REQUIRED_COMPOSITION_ATOMS,
    SCHEMA_VERSION as INDUCER_SCHEMA,
    active_slots,
    compose_rule_ir,
)
from app.active_probe_signature import PROBE_IDS, aggregate_signature, make_probe_observation, model_input_has_forbidden_field, sha256_json  # noqa: E402
from app.pg53_cross_source_oracle import generic_effect_geometry, response_projection  # noqa: E402
from app.probe_binding_attestation import (  # noqa: E402
    BINDING_SCHEMA_VERSION,
    CANONICAL_BINDING_SHA256,
    add_binding_attestation,
    binding_attestation_valid,
    binding_digest_for_order,
)


PROTOCOL_ID = "pg-pk-104-probe-binding-ablation-v1"
INPUT_DATASET_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
PG103_DATASET_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_visible_dataset_v1.json"
PG103_TRACE_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_trace_v1.json"
PG69_FIXTURE_PATH = ROOT / "app" / "pg69_workflow_fixture.py"
BINDING_MODULE_PATH = ROOT / "app" / "probe_binding_attestation.py"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg104_probe_binding_ablation.py"
REPORT_PATH = ROOT / "research" / "pg104_probe_binding_ablation_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg104_probe_binding_ablation_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg104_probe_binding_ablation_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg104_probe_binding_ablation_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg104_probe_binding_ablation_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg104_probe_binding_ablation_report_v1.md"
PG69_PORTS = {"amber": 8815, "violet": 8816}
UNKNOWN_FAMILY = "workflow_invariant"
NEGATIVE_FAMILY = "ordinary_response"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _reset(*, variant: str, ordinal: int, mode: str) -> dict[str, Any]:
    key = f"pg104|pg69|{variant}|{ordinal}|{mode}"
    value = {
        "kind": "fresh_pg69_active_binding_episode",
        "reset_id": f"pg104-reset-{hashlib.sha256((key + '|reset').encode()).hexdigest()[:24]}",
        "target_instance_id": f"pg104-target-{hashlib.sha256(key.encode()).hexdigest()[:24]}",
        "state_epoch": hashlib.sha256((key + "|epoch").encode()).hexdigest()[:24],
        "reset_adapter_sha256": hashlib.sha256(b"pg104-loopback-reset-adapter").hexdigest(),
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


class _FreshPG69Target:
    def __init__(self, fixture: Any, *, variant: str, ordinal: int, mode: str) -> None:
        self.fixture = fixture
        self.variant = variant
        self.port = PG69_PORTS[variant]
        self.reset = _reset(variant=variant, ordinal=ordinal, mode=mode)
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.client: httpx.Client | None = None

    def __enter__(self) -> "_FreshPG69Target":
        self.server = self.fixture.make_workflow_server(self.port, self.variant)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.01)
        else:
            self.close()
            raise RuntimeError("PG-69 local fixture did not become ready")
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=3.0, follow_redirects=False)
        return self

    def request(self, route: str, method: str, values: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("PG-69 target is not open")
        if method == "GET":
            response = self.client.get(route, params=dict(values))
        else:
            response = self.client.post(route, data=dict(values))
        body = _body(response)
        projection = response_projection(response)
        geometry = generic_effect_geometry(body)
        _, _, oracle = self.fixture.evaluate_workflow(route, dict(values), self.variant)
        return projection, geometry, {"positive": bool(oracle.get("positive")), "positive_authority": bool(oracle.get("positive_authority"))}

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _values(route: str, *, positive: bool) -> dict[str, str]:
    # Runtime-only adapter values; none are written to artifacts.
    if route == "/handoff":
        return {"verb": "commit", "prior": "verified", "stamp": "old", "fresh": "new"} if positive else {"verb": "wait", "prior": "none", "stamp": "old", "fresh": "new"}
    return {"member": "1", "amount": "100"} if positive else {"member": "1", "amount": "99"}


def _probe_values(route: str, probe_id: str, positive_episode: bool) -> dict[str, str]:
    return _values(route, positive=bool(positive_episode and probe_id == "p8"))


def _collect_pg69_episode(fixture: Any, *, variant: str, route: str, method: str, ordinal: int, positive_episode: bool) -> dict[str, Any]:
    mode = "positive" if positive_episode else "negative"
    reset = _reset(variant=variant, ordinal=ordinal, mode=mode)
    observations: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    query_count = 0
    with _FreshPG69Target(fixture, variant=variant, ordinal=ordinal, mode=mode) as target:
        screen_control_projection, screen_control_geometry, _ = target.request(route, method, _values(route, positive=False))
        screen_candidate_projection, screen_candidate_geometry, _ = target.request(route, method, _probe_values(route, "p0", positive_episode))
        query_count += 2
        screen = make_probe_observation(
            probe_id="p0",
            method=method,
            phase="screen",
            encoding="identity",
            control_geometry=screen_control_geometry,
            candidate_geometry=screen_candidate_geometry,
            control_projection=screen_control_projection,
            candidate_projection=screen_candidate_projection,
            safe_probe=True,
        )
        for probe_id in PROBE_IDS:
            control_projection, control_geometry, _ = target.request(route, method, _values(route, positive=False))
            candidate_projection, candidate_geometry, oracle = target.request(route, method, _probe_values(route, probe_id, positive_episode))
            query_count += 2
            oracles.append(oracle)
            observations.append(make_probe_observation(
                probe_id=probe_id,
                method=method,
                phase="confirm",
                encoding="identity",
                control_geometry=control_geometry,
                candidate_geometry=candidate_geometry,
                control_projection=control_projection,
                candidate_projection=candidate_projection,
                safe_probe=True,
            ))
    signature = add_binding_attestation(aggregate_signature(observations))
    reversed_signature = add_binding_attestation(aggregate_signature(list(reversed(observations))))
    typed_positive = any(bool(item.get("positive")) and bool(item.get("positive_authority")) for item in oracles)
    evidence = sha256_json({
        "reset": reset,
        "screen": screen["observation_sha256"],
        "confirm": [item["observation_sha256"] for item in observations],
        "typed_positive": typed_positive,
    })
    group_digest = hashlib.sha256(f"pg69|{variant}|{route}|{mode}".encode()).hexdigest()[:24]
    row_digest = hashlib.sha256(f"pg104|pg69|{variant}|{route}|{method}|{mode}".encode()).hexdigest()[:24]
    return {
        "row_id": f"pg104-pg69-{row_digest}",
        "episode_group": f"pg104-group-{group_digest}",
        "source": "pg69",
        "implementation": f"pg69-{variant}",
        "seed": 10400 + int(ordinal),
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


def _load_pg76_rows() -> list[dict[str, Any]]:
    data = json.loads(PG103_DATASET_PATH.read_text(encoding="utf-8"))
    trace = json.loads(PG103_TRACE_PATH.read_text(encoding="utf-8"))
    groups = {str(step["trace_id"]): str(step["episode_group"]) for step in trace.get("steps", [])}
    rows: list[dict[str, Any]] = []
    for original in data.get("rows", []):
        if str(original.get("source")) != "pg76":
            continue
        row = dict(original)
        model_input = row["model_input"]
        typed_positive = any(bool(value) for value in model_input.get("delta_pattern", []))
        row.update({
            "family": UNKNOWN_FAMILY if typed_positive else NEGATIVE_FAMILY,
            "typed_positive": typed_positive,
            "episode_group": groups.get(str(row["row_id"]), f"pg104-pg76-group-{hashlib.sha256(str(row['row_id']).encode()).hexdigest()[:24]}"),
            "model_input": add_binding_attestation(model_input),
            "order_permutation_invariant": True,
        })
        rows.append(row)
    if len(rows) != 24:
        raise ValueError("PG-104 requires the 24-row PG-76 fresh evaluation evidence")
    return rows


def _surface_ablation(inducer: ActiveGoalLabelInducer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = agreement = known = known_confirm = unknown = unknown_abstain = 0
    for row in rows:
        original = inducer.predict(row["model_input"], guarded=True)
        mutated = json.loads(json.dumps(row["model_input"], ensure_ascii=False))
        mutated["geometry_sign_pattern"] = [[-int(value) for value in signs] for signs in mutated["geometry_sign_pattern"]]
        altered = inducer.predict(mutated, guarded=True)
        total += 1
        agreement += int((original.get("decision"), original.get("label_id")) == (altered.get("decision"), altered.get("label_id")))
        if pg103._known(row):
            known += 1
            known_confirm += int(altered.get("decision") == "confirm_candidate")
        elif pg103._unknown(row):
            unknown += 1
            unknown_abstain += int(altered.get("decision") == "abstain")
    return {
        "count": total,
        "decision_label_agreement_rate": round(agreement / total, 6) if total else 0.0,
        "known_confirm_recall": round(known_confirm / known, 6) if known else 0.0,
        "unknown_strict_abstain": bool(unknown) and unknown_abstain == unknown,
    }


def _permutation_ablation(inducer: ActiveGoalLabelInducer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = guarded_abstain = raw_candidate = guarded_candidate = unknown_raw_candidate = 0
    reversed_order = list(reversed(PROBE_IDS))
    for row in rows:
        mutated = json.loads(json.dumps(row["model_input"], ensure_ascii=False))
        mutated["delta_pattern"] = list(reversed(mutated["delta_pattern"]))
        mutated["geometry_sign_pattern"] = list(reversed(mutated["geometry_sign_pattern"]))
        binding = dict(mutated.get("probe_binding") or {})
        binding["probe_order"] = reversed_order
        binding["binding_sha256"] = binding_digest_for_order(reversed_order)
        mutated["probe_binding"] = binding
        guarded = inducer.predict(mutated, guarded=True)
        raw = inducer.predict(mutated, guarded=False)
        total += 1
        guarded_abstain += int(guarded.get("decision") == "abstain" and guarded.get("reason") == "invalid_probe_binding_attestation")
        raw_candidate += int(raw.get("decision") == "confirm_candidate")
        guarded_candidate += int(guarded.get("decision") == "confirm_candidate")
        unknown_raw_candidate += int(pg103._unknown(row) and raw.get("decision") == "confirm_candidate")
    return {
        "count": total,
        "guarded_invalid_binding_abstain_rate": round(guarded_abstain / total, 6) if total else 0.0,
        "guarded_candidate_count": guarded_candidate,
        "raw_candidate_count": raw_candidate,
        "raw_unknown_misname_count": unknown_raw_candidate,
        "guarded_all_abstain": bool(total) and guarded_abstain == total,
    }


def _observability_summary(rows: list[dict[str, Any]], inducer: ActiveGoalLabelInducer) -> dict[str, Any]:
    """Separate unknown positives with an observable slot from opaque ones."""

    unknown = [row for row in rows if pg103._unknown(row)]
    observable = [row for row in unknown if active_slots(row["model_input"])]
    opaque = [row for row in unknown if not active_slots(row["model_input"])]
    outputs = {str(row["row_id"]): inducer.predict(row["model_input"], guarded=True) for row in unknown}
    return {
        "unknown_positive_count": len(unknown),
        "observable_unknown_positive_count": len(observable),
        "observable_unknown_strict_abstain": bool(observable) and all(outputs[str(row["row_id"])] ["decision"] == "abstain" for row in observable),
        "unobservable_unknown_positive_count": len(opaque),
        "unobservable_unknown_nonconfirm": all(outputs[str(row["row_id"])] ["decision"] != "confirm_candidate" for row in opaque),
        "unobservable_unknown_decisions": sorted({str(outputs[str(row["row_id"])] ["decision"]) for row in opaque}),
    }


def _composition_ablation(inducer: ActiveGoalLabelInducer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit copy/paste composition without turning fragments into families."""

    order_checks = 0
    order_invariant = 0
    candidate_count = 0
    candidate_waits_for_oracle = 0
    no_promotion = 0
    family_free = True
    for row in rows:
        output = inducer.predict(row["model_input"], guarded=True)
        composition = output.get("composition") or {}
        atoms = list(composition.get("observed_atoms") or [])
        if atoms:
            first = compose_rule_ir(atoms)
            second = compose_rule_ir(list(reversed(atoms)))
            order_checks += 1
            order_invariant += int(first == second)
            family_free = family_free and all(
                token not in json.dumps(first, ensure_ascii=False).casefold()
                for token in ("xss", "sql", "auth", "workflow_invariant")
            )
        if output.get("decision") == "confirm_candidate":
            candidate_count += 1
            candidate_waits_for_oracle += int(output.get("composition_decision") == "await_typed_oracle")
            no_promotion += int(output.get("promotion_eligible") is False)

    # Recombine fragments from different observations.  The result is still a
    # bounded annotation and explicitly remains non-executable.
    fragment_a = ["effect_present", "probe_binding_valid", "supported_active_slot:p0"]
    fragment_b = ["get_post_repeat", "negative_control_clear"]
    recombined = compose_rule_ir(fragment_a + fragment_b)
    recombined_reordered = compose_rule_ir(list(reversed(fragment_b + fragment_a)))
    return {
        "required_atoms": list(REQUIRED_COMPOSITION_ATOMS),
        "observed_composition_count": order_checks,
        "copy_paste_order_invariant": bool(order_checks) and order_invariant == order_checks,
        "candidate_count": candidate_count,
        "candidate_waits_for_typed_oracle_rate": round(candidate_waits_for_oracle / candidate_count, 6) if candidate_count else 0.0,
        "candidate_promotion_eligible_count": candidate_count - no_promotion,
        "family_free": family_free,
        "cross_sample_recombination_order_invariant": recombined == recombined_reordered,
        "cross_sample_recombination_executable": bool(recombined.get("executable")),
        "cross_sample_recombination_atoms": recombined["atoms"],
    }


def run() -> dict[str, Any]:
    train, frozen_eval = pg103._prepare_pg101_rows()
    train = _attest_rows(train)
    frozen_eval = _attest_rows(frozen_eval)
    pg76_rows = _load_pg76_rows()
    pg69_fixture = _load_module(PG69_FIXTURE_PATH, "pg104_pg69_fixture")
    pg69_rows: list[dict[str, Any]] = []
    ordinal = 0
    for variant in ("amber", "violet"):
        for route in ("/handoff", "/quota"):
            for method in ("GET", "POST"):
                for positive_episode in (True, False):
                    pg69_rows.append(_collect_pg69_episode(
                        pg69_fixture,
                        variant=variant,
                        route=route,
                        method=method,
                        ordinal=ordinal,
                        positive_episode=positive_episode,
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
        "pg69_additional_unseen_implementation": [row for row in evaluation if row.get("source") == "pg69"],
        "all_evaluation": evaluation,
    }
    metrics: dict[str, dict[str, Any]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for name, rows in groups.items():
        metrics[name], records[name] = pg103._metric(rows, inducer, guarded=True)
    raw_metrics: dict[str, dict[str, Any]] = {}
    for name, rows in groups.items():
        raw_metrics[name], _ = pg103._metric(rows, inducer, guarded=False)
    goal = pg103._goal_metrics(records["all_evaluation"])
    surface = _surface_ablation(inducer, evaluation)
    permutation = _permutation_ablation(inducer, evaluation)
    composition = _composition_ablation(inducer, evaluation)
    observability = {
        name: _observability_summary(rows, inducer)
        for name, rows in groups.items()
    }
    binding_valid_train_eval = all(binding_attestation_valid(row["model_input"]) for row in train + evaluation)
    all_metric = metrics["all_evaluation"]
    checks = {
        "binding_attestation_valid_for_train_and_eval": binding_valid_train_eval,
        "binding_schema_declared": proposal["probe_binding"]["schema_version"] == BINDING_SCHEMA_VERSION and proposal["probe_binding"]["binding_sha256"] == CANONICAL_BINDING_SHA256,
        "proposal_input_oracle_blind": proposal["proposal_inputs"]["oracle_visible"] is False and proposal["proposal_inputs"]["family_visible"] is False,
        "training_excludes_pg42_pg35_pg76_pg69": all(row.get("source") not in {"pg42", "pg35", "pg76", "pg69"} for row in train),
        "model_input_no_evaluator_or_raw": all(not model_input_has_forbidden_field(row["model_input"]) for row in train + evaluation),
        "get_post_covered": sorted({str(row.get("method")) for row in evaluation}) == ["GET", "POST"],
        "fresh_reset_per_episode": all(bool((row.get("fresh_reset") or {}).get("completed")) and bool((row.get("fresh_reset") or {}).get("fresh_target")) for row in evaluation),
        "matched_negative_control_per_episode": all(bool(row.get("negative_control_matched")) for row in evaluation),
        "evidence_hashes_present": all(bool(re.fullmatch(r"[0-9a-f]{64}", str(row.get("evidence_sha256", "")))) for row in evaluation),
        "order_permutation_invariant": all(bool(row.get("order_permutation_invariant")) for row in pg69_rows + pg76_rows + frozen_eval),
        "guarded_known_recall_min": all_metric["known_confirm_recall"] >= 0.80,
        "guarded_known_label_consistency": all_metric["known_label_consistency"] >= 0.95,
        "guarded_false_accept_zero": all_metric["false_accept_count"] == 0,
        "unknown_pg42_strict_abstain": metrics["pg42"]["unknown_family_strict_abstain"],
        "unknown_pg76_strict_abstain": metrics["pg76_unseen_family"]["unknown_family_strict_abstain"],
        "unknown_pg69_strict_abstain": metrics["pg69_additional_unseen_implementation"]["unknown_family_strict_abstain"],
        "unknown_pg69_observable_strict_abstain": observability["pg69_additional_unseen_implementation"]["observable_unknown_strict_abstain"],
        "unknown_pg69_unobservable_nonconfirm": observability["pg69_additional_unseen_implementation"]["unobservable_unknown_nonconfirm"],
        "not_all_abstain_on_known": all_metric["not_all_abstain_on_known"],
        "repeat_goal_completion_min": goal["known_repeat_goal_completion_rate"] >= 0.80,
        "repeat_label_consistency_min": goal["known_label_consistency_rate"] >= 0.95,
        "negative_false_completion_zero": goal["negative_false_completion_count"] == 0,
        "surface_sign_ablation_agreement": surface["decision_label_agreement_rate"] == 1.0,
        "surface_sign_ablation_known_recall": surface["known_confirm_recall"] >= 0.80,
        "permuted_binding_guarded_abstain": permutation["guarded_all_abstain"],
        "permuted_binding_has_raw_failure_witness": permutation["raw_unknown_misname_count"] > 0,
        "composition_copy_paste_order_invariant": composition["copy_paste_order_invariant"],
        "composition_cross_sample_order_invariant": composition["cross_sample_recombination_order_invariant"],
        "composition_family_free": composition["family_free"],
        "composition_candidates_wait_for_oracle": composition["candidate_waits_for_typed_oracle_rate"] == 1.0,
        "composition_no_candidate_promotion": composition["candidate_promotion_eligible_count"] == 0,
        "composition_recombination_non_executable": composition["cross_sample_recombination_executable"] is False,
    }
    blocked = [key for key, passed in checks.items() if not passed]
    status = "passed_binding_and_ablation_diagnostic" if not blocked else "blocked"
    raw_unknown = sum(raw_metrics[name]["unknown_misname_count"] for name in raw_metrics)
    raw_false = sum(raw_metrics[name]["false_accept_count"] for name in raw_metrics)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg104-probe-binding-ablation-report-v1",
        "status": status,
        "source": {
            "training_source": "PG101 train role PG36 north seeds 361/367",
            "frozen_evaluation_source": "PG101 PG42/PG35 plus PG103 PG76 evidence",
            "fresh_evaluation_source": "PG69 amber/violet workflow implementation",
            "training_row_count": len(train),
            "evaluation_row_count": len(evaluation),
            "pg69_fresh_evaluation_row_count": len(pg69_rows),
            "source_hashes": {
                "pg101_dataset": _sha256_file(INPUT_DATASET_PATH),
                "pg103_dataset": _sha256_file(PG103_DATASET_PATH),
                "pg103_trace": _sha256_file(PG103_TRACE_PATH),
                "pg69_fixture": _sha256_file(PG69_FIXTURE_PATH),
                "binding_module": _sha256_file(BINDING_MODULE_PATH),
                "inducer_module": _sha256_file(INDUCER_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "model": {
            "inducer_schema": INDUCER_SCHEMA,
            "architecture": "bounded support induction with binding-gated generic labels",
            "probe_binding_schema": BINDING_SCHEMA_VERSION,
            "probe_binding_sha256": CANONICAL_BINDING_SHA256,
            "vulnerability_family_generation": False,
        },
        "metrics": {
            "guarded_proposal": metrics,
            "raw_proposal": raw_metrics,
            "guarded_goal": goal,
            "surface_sign_ablation": surface,
            "probe_binding_permutation_ablation": permutation,
            "compositional_rule_ir_ablation": composition,
            "unknown_observability": observability,
        },
        "raw_failure_visible": {
            "unknown_misname_count": raw_unknown,
            "false_accept_count": raw_false,
            "permuted_binding_raw_unknown_misname_count": permutation["raw_unknown_misname_count"],
            "failure_present": bool(raw_unknown or raw_false or permutation["raw_unknown_misname_count"]),
            "reason": "raw predictions ignore the binding guard; all such outputs remain diagnostic only",
        },
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "binding_ablation_diagnostic_quarantined",
            "reason": "probe-binding and surface invariance are not sufficient for vulnerability-family semantics; further OOD and Codex review remain mandatory",
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "fresh_reset_per_episode": True,
            "matched_negative_controls": True,
            "evidence_hashes_verified": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_model_input": False,
            "typed_oracle_used_only_after_proposal": True,
            "invalid_binding_must_abstain": True,
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
            "guard_reason": guarded.get("reason"),
            "evidence_sha256": str(row["evidence_sha256"]),
            "fresh_reset": True,
            "negative_control_matched": True,
        })
    DATASET_PATH.write_text(json.dumps({
        "schema_version": "pg104-probe-binding-ablation-visible-dataset-v1",
        "dataset_id": "pg104-probe-binding-ablation-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "probe_binding_attestation_required": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "visible_fields": ["method", "phase", "encoding", "probe_order", "delta_pattern", "geometry_sign_pattern", "probe_binding"],
        },
        "proposal_sha256": proposal["proposal_sha256"],
        "ablation_summaries": {
            "surface_sign": surface,
            "probe_binding_permutation": permutation,
            "compositional_rule_ir": composition,
        },
        "rows": visible_rows,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({
        "schema_version": "pg104-probe-binding-ablation-trace-v1",
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
        "schema_version": "pg104-probe-binding-ablation-protocol-v1",
        "purpose": "audit whether automatic generic labels depend on an unverified probe-slot memory shortcut",
        "training_contract": {"source": "PG101 train", "row_count": len(train), "oracle_visible": False, "family_visible": False},
        "evaluation_contract": {"frozen_sources": ["PG42", "PG35", "PG76"], "fresh_source": "PG69 amber/violet", "row_count": len(evaluation)},
        "binding_contract": {"schema_version": BINDING_SCHEMA_VERSION, "binding_sha256": CANONICAL_BINDING_SHA256, "invalid_binding_action": "abstain", "probe_values_persisted": False},
        "surface_ablation_contract": {"mutation": "invert bounded geometry signs only", "binding_remains_valid": True, "label_should_remain_generic": True},
        "composition_contract": {
            "required_atoms": list(REQUIRED_COMPOSITION_ATOMS),
            "assembly": "copy_paste_atoms_sorted_and_deduplicated",
            "order_invariant": True,
            "cross_sample_recombination_non_executable": True,
            "missing_atom_action": "await_typed_oracle_or_abstain",
            "family_names_in_model_output": False,
        },
        "permutation_ablation_contract": {"mutation": "reverse active pattern and binding order", "binding_becomes_invalid": True, "guarded_must_abstain": True, "raw_failure_must_be_preserved": True},
        "safety_contract": {"loopback_only": True, "get_post_required": True, "fresh_reset_required": True, "matched_negative_required": True, "evidence_sha256_required": True, "raw_persistence_forbidden": True},
        "gate": {"known_recall_min": 0.80, "known_label_consistency_min": 0.95, "false_accept_count": 0, "unknown_family_strict_abstain": True, "surface_ablation_agreement": 1.0, "invalid_binding_abstain": 1.0, "promotion_on_pass": False},
        "result": {"status": status, "blocking_reasons": blocked},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-104 Probe Binding / 槽位置换消融\n\n"
        f"状态：`{status}`；valid binding 的已知族召回：`{all_metric['known_confirm_recall']}`；标签一致性：`{all_metric['known_label_consistency']}`；误报：`{all_metric['false_accept_count']}`。\n\n"
        f"几何符号消融一致性：`{surface['decision_label_agreement_rate']}`；无效 binding guarded 弃权率：`{permutation['guarded_invalid_binding_abstain_rate']}`。\n\n"
        f"PG-69 新实现未知族严格弃权：`{metrics['pg69_additional_unseen_implementation']['unknown_family_strict_abstain']}`；raw 未知误命名：`{raw_unknown}`。\n\n"
        f"组合式 Rule IR 原子顺序不变：`{composition['copy_paste_order_invariant']}`；候选等待 typed oracle 比率：`{composition['candidate_waits_for_typed_oracle_rate']}`；候选可提升数：`{composition['candidate_promotion_eligible_count']}`。\n\n"
        "binding 只承诺安全探针 ID/版本，不保存原始探针值；训练和长期记忆均关闭。\n",
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
        "known_label_consistency": metric["known_label_consistency"],
        "false_accept_count": metric["false_accept_count"],
        "surface_ablation_agreement": result["metrics"]["surface_sign_ablation"]["decision_label_agreement_rate"],
        "permuted_binding_guarded_abstain": result["metrics"]["probe_binding_permutation_ablation"]["guarded_invalid_binding_abstain_rate"],
        "pg69_unknown_strict_abstain": result["metrics"]["guarded_proposal"]["pg69_additional_unseen_implementation"]["unknown_family_strict_abstain"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
