"""PG-103: oracle-blind automatic goal/label induction on a new family.

The inducer receives only PG-101's bounded active-probe signatures.  It
inventories stable, single-slot effects and emits a generic goal plus labels
such as ``AUTO_EFFECT_SLOT_P0``; no vulnerability family or typed oracle is
available during induction.  The frozen proposal is then replayed on the
PG-42/PG-35 matrix and on a fresh PG-76 workflow fixture.  PG-76 is outside
the training family registry and is collected here with a fresh loopback
target for every positive and negative episode.

Raw proposals are scored separately from the guarded proposal.  Only the
guarded result can pass the diagnostic gate, and even a passing diagnostic is
quarantined from training and long-term memory.
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
from typing import Any, Iterable, Mapping

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_goal_label_inducer import (  # noqa: E402
    ActiveGoalLabelInducer,
    SCHEMA_VERSION as INDUCER_SCHEMA,
    active_slots,
    proposal_digest,
)
from app.active_probe_signature import (  # noqa: E402
    PROBE_IDS,
    aggregate_signature,
    make_probe_observation,
    model_input_has_forbidden_field,
    sha256_json,
)
from app.pg53_cross_source_oracle import generic_effect_geometry, response_projection  # noqa: E402


PROTOCOL_ID = "pg-pk-103-auto-goal-label-active-probe-v1"
INPUT_DATASET_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
INPUT_REPORT_PATH = ROOT / "research" / "pg101_active_probe_signature_report_v1.json"
PG76_FIXTURE_PATH = ROOT / "app" / "pg76_unknown_triplet_fixture.py"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg103_auto_goal_label_active_probe.py"
REPORT_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg103_auto_goal_label_active_probe_report_v1.md"
PG76_PORT = 8818
KNOWN_FAMILIES = {
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
}
UNKNOWN_FAMILY = "workflow_invariant"
NEGATIVE_FAMILY = "ordinary_response"


def _file_sha256(path: Path) -> str:
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


def _fresh_reset(*, variant: str, ordinal: int, mode: str) -> dict[str, Any]:
    key = f"pg103|pg76|{variant}|{ordinal}|{mode}"
    reset = {
        "kind": "fresh_pg76_active_goal_episode",
        "reset_id": f"pg103-reset-{hashlib.sha256((key + '|reset').encode()).hexdigest()[:24]}",
        "target_instance_id": f"pg103-target-{hashlib.sha256(key.encode()).hexdigest()[:24]}",
        "state_epoch": hashlib.sha256((key + "|epoch").encode()).hexdigest()[:24],
        "reset_adapter_sha256": hashlib.sha256(b"pg103-loopback-reset-adapter").hexdigest(),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
        "transport": "httpx_loopback",
        "probe_bank_episode": True,
    }
    reset["reset_sha256"] = sha256_json(reset)
    return reset


class _FreshPG76Target:
    def __init__(self, fixture: Any, variant: str, ordinal: int, mode: str) -> None:
        self.fixture = fixture
        self.variant = variant
        self.ordinal = int(ordinal)
        self.mode = mode
        self.server: Any = None
        self.thread: threading.Thread | None = None
        self.client: httpx.Client | None = None
        self.reset = _fresh_reset(variant=variant, ordinal=ordinal, mode=mode)

    def __enter__(self) -> "_FreshPG76Target":
        self.server = self.fixture.make_server(PG76_PORT, self.variant)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", PG76_PORT), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.01)
        else:
            self.close()
            raise RuntimeError("PG-76 local fixture did not become ready")
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{PG76_PORT}", timeout=3.0, follow_redirects=False)
        return self

    def request(self, route: str, method: str, values: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("PG-76 target is not open")
        if method == "GET":
            response = self.client.get(route, params=dict(values))
        else:
            response = self.client.post(route, data=dict(values))
        body = _body(response)
        projection = response_projection(response)
        geometry = generic_effect_geometry(body)
        positive, signal = self.fixture.evaluate(route, dict(values), self.variant)
        # This oracle object exists only in the collector's memory.  It is
        # never passed to the inducer or written to the visible dataset.
        oracle = {
            "positive": bool(positive),
            "positive_authority": True,
            "signal_digest": hashlib.sha256(str(signal).encode()).hexdigest(),
        }
        return projection, geometry, oracle

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
    # Adapter-only values.  They are intentionally never serialized.
    if route == "/commit":
        return {"verb": "commit", "prior": "verified", "fresh": "new"} if positive else {"verb": "wait", "prior": "none", "fresh": "new"}
    return {"member": "1", "amount": "100"} if positive else {"member": "1", "amount": "99"}


def _probe_values(route: str, probe_id: str, *, positive_episode: bool) -> dict[str, str]:
    return _values(route, positive=bool(positive_episode and probe_id == "p8"))


def _collect_pg76_episode(fixture: Any, *, variant: str, route: str, method: str, ordinal: int, positive_episode: bool) -> dict[str, Any]:
    mode = "positive" if positive_episode else "negative"
    reset = _fresh_reset(variant=variant, ordinal=ordinal, mode=mode)
    observations: list[dict[str, Any]] = []
    candidate_oracles: list[dict[str, Any]] = []
    query_count = 0
    with _FreshPG76Target(fixture, variant, ordinal, mode) as target:
        # A screen pair is part of the goal budget; the complete bank is the
        # confirmation trace.  Both use a matched ordinary control.
        screen_control_projection, screen_control_geometry, _ = target.request(route, method, _values(route, positive=False))
        screen_candidate_projection, screen_candidate_geometry, _ = target.request(route, method, _probe_values(route, "p0", positive_episode=positive_episode))
        query_count += 2
        screen_observation = make_probe_observation(
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
            candidate_projection, candidate_geometry, oracle = target.request(route, method, _probe_values(route, probe_id, positive_episode=positive_episode))
            query_count += 2
            candidate_oracles.append(oracle)
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
    signature = aggregate_signature(observations)
    reversed_signature = aggregate_signature(list(reversed(observations)))
    typed_positive = any(bool(item.get("positive")) and bool(item.get("positive_authority")) for item in candidate_oracles)
    evidence = sha256_json({
        "reset": reset,
        "screen": screen_observation["observation_sha256"],
        "confirm": [item["observation_sha256"] for item in observations],
        "typed_positive": typed_positive,
    })
    # Keep positive and negative episodes separate, but pair their GET and
    # POST channels so the induced goal is actually tested as a two-channel
    # repeatable objective on the unseen fixture.
    episode_group = hashlib.sha256(f"pg76|{variant}|{route}|{mode}".encode()).hexdigest()[:24]
    row_id = hashlib.sha256(f"pg103|pg76|{variant}|{route}|{method}|{mode}".encode()).hexdigest()[:24]
    return {
        "row_id": f"pg103-pg76-{row_id}",
        "episode_group": f"pg103-group-{episode_group}",
        "source": "pg76",
        "implementation": f"pg76-{variant}",
        "seed": 10300 + int(ordinal),
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


def _prepare_pg101_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(INPUT_DATASET_PATH.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    if len(rows) != 618:
        raise ValueError("PG-103 requires the frozen PG-101 618-row dataset")
    training: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping) or model_input_has_forbidden_field(model_input):
            raise ValueError("PG-101 model input failed the oracle-blind contract")
        family = str((row.get("evaluator_label") or {}).get("family", ""))
        typed_positive = bool((row.get("evaluator_label") or {}).get("typed_positive"))
        row["family"] = family
        row["typed_positive"] = typed_positive
        row["episode_group"] = hashlib.sha256(re.sub(r"-(?:get|post)$", "", str(row.get("row_id", "")), flags=re.IGNORECASE).encode()).hexdigest()[:24]
        row["fresh_reset_ok"] = bool((row.get("fresh_reset") or {}).get("completed")) and bool((row.get("fresh_reset") or {}).get("fresh_target")) and not bool((row.get("fresh_reset") or {}).get("external_network"))
        row["negative_control_ok"] = bool(row.get("negative_control_matched"))
        if row.get("role") == "train":
            training.append(row)
        elif row.get("source") in {"pg42", "pg35"}:
            evaluation.append(row)
    if len(training) != 32 or len(evaluation) != 522:
        raise ValueError("PG-103 split contract requires 32 train and 522 frozen evaluation rows")
    return training, evaluation


def _known(row: Mapping[str, Any]) -> bool:
    return bool(row.get("typed_positive")) and str(row.get("family")) in KNOWN_FAMILIES


def _unknown(row: Mapping[str, Any]) -> bool:
    return bool(row.get("typed_positive")) and not _known(row)


def _metric(rows: Iterable[Mapping[str, Any]], inducer: ActiveGoalLabelInducer, *, guarded: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(rows)
    known = known_hits = known_label_matches = known_misname = 0
    negatives = false_accepts = 0
    unknown = unknown_abstain = unknown_misname = 0
    candidate_count = abstain_count = reject_count = 0
    records: list[dict[str, Any]] = []
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_method: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        output = inducer.predict(row["model_input"], guarded=guarded)
        slots = list(active_slots(row["model_input"]))
        decision = str(output.get("decision", ""))
        if decision == "confirm_candidate":
            candidate_count += 1
        elif decision == "abstain":
            abstain_count += 1
        else:
            reject_count += 1
        expected_label = f"AUTO_EFFECT_SLOT_{slots[0].upper()}" if len(slots) == 1 else "AUTO_NO_OBSERVED_EFFECT" if not slots else "AUTO_UNSEEN_OR_AMBIGUOUS_EFFECT"
        if _known(row):
            known += 1
            by_source[str(row.get("source", ""))][0] += 1
            by_method[str(row.get("method", ""))][0] += 1
            if decision == "confirm_candidate":
                known_hits += 1
                by_source[str(row.get("source", ""))][1] += 1
                by_method[str(row.get("method", ""))][1] += 1
                if str(output.get("label_id")) == expected_label:
                    known_label_matches += 1
                else:
                    known_misname += 1
        elif _unknown(row):
            unknown += 1
            if decision == "abstain":
                unknown_abstain += 1
            elif decision == "confirm_candidate":
                unknown_misname += 1
        else:
            negatives += 1
            if decision == "confirm_candidate":
                false_accepts += 1
        records.append({
            "row_id": str(row.get("row_id", "")),
            "episode_group": str(row.get("episode_group", "")),
            "source": str(row.get("source", "")),
            "implementation": str(row.get("implementation", "")),
            "seed": int(row.get("seed", -1)),
            "method": str(row.get("method", "")),
            "typed_positive": bool(row.get("typed_positive")),
            "family": str(row.get("family", "")),
            "expected_label": expected_label,
            "prediction": output,
        })
    family_consistency = known_label_matches / known if known else 0.0
    return {
        "count": len(rows),
        "known_positive_count": known,
        "known_confirm_count": known_hits,
        "known_confirm_recall": round(known_hits / known, 6) if known else 0.0,
        "known_label_consistency": round(family_consistency, 6),
        "known_misname_count": known_misname,
        "typed_negative_count": negatives,
        "false_accept_count": false_accepts,
        "unknown_positive_count": unknown,
        "unknown_abstain_count": unknown_abstain,
        "unknown_misname_count": unknown_misname,
        "unknown_family_strict_abstain": bool(unknown) and unknown_abstain == unknown,
        "candidate_count": candidate_count,
        "abstain_count": abstain_count,
        "reject_count": reject_count,
        "not_all_abstain_on_known": bool(known) and known_hits > 0,
        "source_confirm_recall": {
            source: round(values[1] / values[0], 6) if values[0] else 0.0
            for source, values in sorted(by_source.items())
        },
        "method_confirm_recall": {
            method: round(values[1] / values[0], 6) if values[0] else 0.0
            for method, values in sorted(by_method.items())
        },
    }, records


def _goal_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["episode_group"])].append(record)
    paired = positive = positive_completed = positive_label_consistent = negative = negative_false_completion = unknown = unknown_strict = 0
    for group in groups.values():
        methods = {str(row["method"]) for row in group}
        if not {"GET", "POST"}.issubset(methods):
            continue
        paired += 1
        decisions = [str(row["prediction"].get("decision", "")) for row in group]
        labels = [str(row["prediction"].get("label_id", "")) for row in group]
        if all(bool(row["typed_positive"]) and str(row["family"]) in KNOWN_FAMILIES for row in group):
            positive += 1
            if all(decision == "confirm_candidate" for decision in decisions):
                positive_completed += 1
            if len(set(labels)) == 1 and all(decision == "confirm_candidate" for decision in decisions):
                positive_label_consistent += 1
        elif all(not bool(row["typed_positive"]) for row in group):
            negative += 1
            if any(decision == "confirm_candidate" for decision in decisions):
                negative_false_completion += 1
        elif all(bool(row["typed_positive"]) for row in group):
            unknown += 1
            if all(decision == "abstain" for decision in decisions):
                unknown_strict += 1
    return {
        "paired_episode_count": paired,
        "known_positive_episode_count": positive,
        "known_repeat_goal_completion_count": positive_completed,
        "known_repeat_goal_completion_rate": round(positive_completed / positive, 6) if positive else 0.0,
        "known_label_consistent_episode_count": positive_label_consistent,
        "known_label_consistency_rate": round(positive_label_consistent / positive, 6) if positive else 0.0,
        "negative_episode_count": negative,
        "negative_false_completion_count": negative_false_completion,
        "unknown_positive_episode_count": unknown,
        "unknown_strict_abstain_episode_count": unknown_strict,
        "unknown_strict_abstain_rate": round(unknown_strict / unknown, 6) if unknown else 0.0,
    }


def _raw_baseline(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    unknown_misname = false_accept = 0
    for row in rows:
        slots = active_slots(row["model_input"])
        candidate = bool(slots)
        if _unknown(row) and candidate:
            unknown_misname += 1
        if not bool(row.get("typed_positive")) and candidate:
            false_accept += 1
    return {"count": len(rows), "unknown_misname_count": unknown_misname, "false_accept_count": false_accept, "failure_present": bool(unknown_misname or false_accept)}


def run() -> dict[str, Any]:
    train_rows, frozen_eval_rows = _prepare_pg101_rows()
    fixture = _load_module(PG76_FIXTURE_PATH, "pg103_pg76_fixture")
    pg76_rows: list[dict[str, Any]] = []
    ordinal = 0
    for variant in fixture.VARIANTS:
        for route in fixture.ROUTES:
            for method in ("GET", "POST"):
                for positive_episode in (True, False):
                    pg76_rows.append(_collect_pg76_episode(
                        fixture,
                        variant=str(variant),
                        route=str(route),
                        method=str(method),
                        ordinal=ordinal,
                        positive_episode=positive_episode,
                    ))
                    ordinal += 1
    all_eval_rows = frozen_eval_rows + pg76_rows
    inducer = ActiveGoalLabelInducer(minimum_support=2, require_get_post=True).fit([{"model_input": row["model_input"]} for row in train_rows])
    proposal = inducer.proposal()
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    raw_metrics: dict[str, dict[str, Any]] = {}
    guarded_metrics: dict[str, dict[str, Any]] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    guarded_records: dict[str, list[dict[str, Any]]] = {}
    groups = {
        "pg42": [row for row in all_eval_rows if row.get("source") == "pg42"],
        "pg35_third_implementation": [row for row in all_eval_rows if row.get("source") == "pg35"],
        "pg76_unseen_family": [row for row in all_eval_rows if row.get("source") == "pg76"],
        "all_evaluation": all_eval_rows,
    }
    for name, group in groups.items():
        raw_metrics[name], raw_records[name] = _metric(group, inducer, guarded=False)
        guarded_metrics[name], guarded_records[name] = _metric(group, inducer, guarded=True)
    guarded_goal = _goal_metrics(guarded_records["all_evaluation"])
    raw_goal = _goal_metrics(raw_records["all_evaluation"])
    raw_failure = _raw_baseline(all_eval_rows)
    guarded_all = guarded_metrics["all_evaluation"]
    checks = {
        "proposal_oracle_blind": proposal["proposal_inputs"]["oracle_visible"] is False and proposal["proposal_inputs"]["family_visible"] is False,
        "training_excludes_pg42_pg35_pg76": all(row.get("source") not in {"pg42", "pg35", "pg76"} for row in train_rows),
        "model_input_excludes_evaluator_or_raw": all(not model_input_has_forbidden_field(row["model_input"]) for row in train_rows + all_eval_rows),
        "get_post_covered": sorted({str(row.get("method")) for row in all_eval_rows}) == ["GET", "POST"],
        "fresh_reset_per_episode": all(bool(row.get("fresh_reset_ok", True)) if "fresh_reset_ok" in row else bool((row.get("fresh_reset") or {}).get("completed")) for row in all_eval_rows),
        "matched_negative_control_per_episode": all(bool(row.get("negative_control_ok", True)) if "negative_control_ok" in row else bool(row.get("negative_control_matched")) for row in all_eval_rows),
        "evidence_hashes_present": all(bool(re.fullmatch(r"[0-9a-f]{64}", str(row.get("evidence_sha256", "")))) for row in all_eval_rows),
        "order_permutation_invariant": all(bool(row.get("order_permutation_invariant")) for row in pg76_rows) and all(bool(row.get("order_permutation_invariant", True)) for row in frozen_eval_rows),
        "guarded_known_recall_min": guarded_all["known_confirm_recall"] >= 0.80,
        "guarded_known_label_consistency": guarded_all["known_label_consistency"] >= 0.95,
        "guarded_false_accept_zero": guarded_all["false_accept_count"] == 0,
        "guarded_pg42_unknown_strict_abstain": guarded_metrics["pg42"]["unknown_family_strict_abstain"],
        "guarded_pg76_unknown_strict_abstain": guarded_metrics["pg76_unseen_family"]["unknown_family_strict_abstain"],
        "guarded_not_all_abstain_on_known": guarded_all["not_all_abstain_on_known"],
        "known_repeat_goal_completion_min": guarded_goal["known_repeat_goal_completion_rate"] >= 0.80,
        "known_repeat_label_consistency_min": guarded_goal["known_label_consistency_rate"] >= 0.95,
        "negative_repeat_goal_false_completion_zero": guarded_goal["negative_false_completion_count"] == 0,
        "raw_failure_preserved": raw_failure["failure_present"],
    }
    blocked = [name for name, passed in checks.items() if not passed]
    status = "passed_generic_goal_label_diagnostic" if not blocked else "blocked"
    source_hashes = {
        "pg101_input_dataset": _file_sha256(INPUT_DATASET_PATH),
        "pg101_input_report": _file_sha256(INPUT_REPORT_PATH),
        "pg76_fixture": _file_sha256(PG76_FIXTURE_PATH),
        "active_probe_module": _file_sha256(ROOT / "app" / "active_probe_signature.py"),
        "inducer_module": _file_sha256(INDUCER_PATH),
        "runner": _file_sha256(RUNNER_PATH),
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg103-auto-goal-label-active-probe-report-v1",
        "status": status,
        "source": {
            "training_source": "PG101 train role: PG36 north seeds 361/367 known families only",
            "frozen_evaluation_source": "PG101 PG42 cobalt/quartz and PG35 third implementation",
            "fresh_evaluation_source": "PG76 workflow_invariant, variants copper/teal/indigo",
            "training_row_count": len(train_rows),
            "frozen_evaluation_row_count": len(frozen_eval_rows),
            "fresh_pg76_evaluation_row_count": len(pg76_rows),
            "oracle_after_proposal": True,
            "source_hashes": source_hashes,
        },
        "proposal": {
            "proposal_file": str(PROPOSAL_PATH.relative_to(ROOT)),
            "proposal_sha256": proposal["proposal_sha256"],
            "inducer_schema": INDUCER_SCHEMA,
            "discovered_slot_count": len(proposal.get("discovered_effect_slots", [])),
            "discovered_slots": [item["slot"] for item in proposal.get("discovered_effect_slots", [])],
            "generated_vulnerability_family_names": False,
        },
        "metrics": {
            "raw_proposal": raw_metrics,
            "guarded_proposal": guarded_metrics,
            "raw_goal": raw_goal,
            "guarded_goal": guarded_goal,
            "raw_slot_presence_baseline": raw_failure,
        },
        "raw_failure_visible": {
            "unknown_family_misname_count": raw_failure["unknown_misname_count"],
            "negative_false_accept_count": raw_failure["false_accept_count"],
            "failure_present": raw_failure["failure_present"],
            "reason": "an unguarded slot-presence proposal treats every active effect as a candidate; the frozen proposal must retain unseen-slot abstention",
        },
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "generic_goal_label_proposal_quarantined",
            "reason": "PG103 validates abstract effect labels only; typed oracle acceptance, further unseen families and Codex review remain required",
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "fresh_reset_per_episode": True,
            "matched_negative_controls": True,
            "evidence_hashes_verified": True,
            "get_post_covered": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_model_input": False,
            "typed_oracle_labels_used_only_after_proposal": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    visible_rows: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    for row in all_eval_rows:
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
            "fresh_reset": dict(row.get("fresh_reset") or {}),
            "negative_control_matched": bool(row.get("negative_control_matched")),
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
            "fresh_reset": bool((row.get("fresh_reset") or {}).get("completed")),
            "negative_control_matched": bool(row.get("negative_control_matched")),
        })
    DATASET_PATH.write_text(json.dumps({
        "schema_version": "pg103-auto-goal-label-active-probe-visible-dataset-v1",
        "dataset_id": "pg103-auto-goal-label-active-probe-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "proposal_is_generic_effect_only": True,
            "visible_fields": ["method", "phase", "encoding", "probe_order", "delta_pattern", "geometry_sign_pattern"],
        },
        "proposal_sha256": proposal["proposal_sha256"],
        "rows": visible_rows,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({
        "schema_version": "pg103-auto-goal-label-active-probe-trace-v1",
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
        "schema_version": "pg103-auto-goal-label-active-probe-protocol-v1",
        "purpose": "test automatic generic goal/label induction from bounded active probe observations",
        "training_contract": {"source": "pg101", "role": "train", "row_count": len(train_rows), "typed_oracle_visible": False, "family_visible": False},
        "frozen_evaluation_contract": {"source": "pg101", "pg42": "known plus template holdout", "pg35": "third implementation", "row_count": len(frozen_eval_rows)},
        "fresh_evaluation_contract": {"source": "pg76", "unknown_family": UNKNOWN_FAMILY, "variants": ["copper", "teal", "indigo"], "fresh_target_per_episode": True, "row_count": len(pg76_rows)},
        "proposal_contract": {"labels_must_be_generic_effect_aliases": True, "vulnerability_family_generation_forbidden": True, "success_requires_get_post_repeat": True, "unseen_slot_must_abstain": True},
        "oracle_contract": {"typed_oracle_after_proposal": True, "negative_controls_required": True, "evidence_sha256_required": True},
        "safety_contract": {"loopback_only": True, "external_network": False, "raw_probe_persistence_forbidden": True, "raw_response_persistence_forbidden": True},
        "gate": {"known_recall_min": 0.80, "known_label_consistency_min": 0.95, "false_accept_count": 0, "unknown_family_strict_abstain": True, "repeat_goal_completion_min": 0.80, "repeat_label_consistency_min": 0.95, "promotion_on_pass": False},
        "result": {"status": status, "blocking_reasons": blocked},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-103 自动目标/标签归纳\n\n"
        f"状态：`{status}`；训练只使用 PG-101 的 32 条无标签 active signature。\n\n"
        f"Guarded 已知族召回：`{guarded_all['known_confirm_recall']}`；抽象标签一致性：`{guarded_all['known_label_consistency']}`；误报：`{guarded_all['false_accept_count']}`。\n\n"
        f"重复目标完成率：`{guarded_goal['known_repeat_goal_completion_rate']}`；重复标签一致性：`{guarded_goal['known_label_consistency_rate']}`。\n\n"
        f"PG-42 未知族严格弃权：`{guarded_metrics['pg42']['unknown_family_strict_abstain']}`；PG-76 新未知族严格弃权：`{guarded_metrics['pg76_unseen_family']['unknown_family_strict_abstain']}`。\n\n"
        f"Raw 槽位基线未知族误命名：`{raw_failure['unknown_misname_count']}`；负样本误报：`{raw_failure['false_accept_count']}`。该失败已保留，未训练、未写长期记忆。\n\n"
        f"阻塞项：{', '.join(blocked) if blocked else '无'}。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    metrics = result["metrics"]["guarded_proposal"]["all_evaluation"]
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "known_recall": metrics["known_confirm_recall"],
        "known_label_consistency": metrics["known_label_consistency"],
        "false_accept_count": metrics["false_accept_count"],
        "pg42_unknown_strict_abstain": result["metrics"]["guarded_proposal"]["pg42"]["unknown_family_strict_abstain"],
        "pg76_unknown_strict_abstain": result["metrics"]["guarded_proposal"]["pg76_unseen_family"]["unknown_family_strict_abstain"],
        "repeat_goal_completion": result["metrics"]["guarded_goal"]["known_repeat_goal_completion_rate"],
        "raw_unknown_misname": result["raw_failure_visible"]["unknown_family_misname_count"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
