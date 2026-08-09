"""Build PG-278: multi-family, counterfactual, failure-to-repair task data.

This is intentionally a bounded *controlled* study.  It uses four independent
loopback fixtures (DOM surface, SQL-shaped response, redirect contract and
logic/access state) and preserves only abstract request/response projections.
The raw probe, response body and evaluator result never enter model context.

The study's narrow claim is slot binding under an implementation holdout:
given an explicitly represented missing observation, can a policy request the
right observation class, then update from the returned evidence?  It is not a
claim of real-world vulnerability discovery or memory promotion.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heterogeneous_surface_fixture_v3 import (  # noqa: E402
    HeterogeneousSurfaceV3Collector,
    V3_PORTS,
    default_heterogeneous_surface_v3_specs,
    heterogeneous_surface_v3_source_sha256,
    make_heterogeneous_surface_v3_fixture_server,
)
from app.logic_access_fixture_v3 import (  # noqa: E402
    LOGIC_ACCESS_V3_PORTS,
    LogicAccessV3Collector,
    _evaluate as logic_evaluate,
    default_logic_access_v3_specs,
    logic_access_v3_source_sha256,
    make_logic_access_v3_fixture_server,
)
from app.pg278_redirect_fixture import (  # noqa: E402
    PORTS as REDIRECT_PORTS,
    VARIANTS as REDIRECT_VARIANTS,
    collect as collect_redirect,
    make_server as make_redirect_server,
    source_sha256 as redirect_source_sha256,
)
from app.sql_differential_fixture_v6 import (  # noqa: E402
    V6_PORTS,
    V6_VARIANTS,
    collect_sql_v6,
    make_sql_v6_fixture_server,
    sql_v6_source_sha256,
)


OUTPUT = ROOT / "research" / "pg278_multifamily_question_dataset_v1.json"
SEEDS = (27801, 27802, 27803)
SPLIT_BY_VARIANT = {"train_a": "implementation_train", "train_b": "implementation_train", "holdout": "implementation_holdout"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bucket(value: Any, *, edges: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return "0"
    for edge in edges:
        if number <= edge:
            return str(edge)
    return f">{edges[-1]}"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    variants: tuple[str, str, str]
    ports: tuple[int, int, int]
    slots: tuple[tuple[str, str], tuple[str, str]]
    source_hash: str


FAMILIES = {
    "dom_effect": FamilySpec(
        "dom_effect",
        ("alpha", "beta", "gamma"),
        tuple(int(port) for port in V3_PORTS),
        (("dom_render_channel", "inspect_effect_channel"), ("dom_control_alignment", "inspect_control_comparison")),
        heterogeneous_surface_v3_source_sha256(),
    ),
    "sql_differential": FamilySpec(
        "sql_differential",
        ("obsidian", "pearl", "saffron"),
        tuple(int(port) for port in V6_PORTS),
        (("sql_response_shape", "inspect_effect_channel"), ("sql_baseline_delta", "inspect_control_comparison")),
        sql_v6_source_sha256(),
    ),
    "redirect_contract": FamilySpec(
        "redirect_contract",
        tuple(str(value) for value in REDIRECT_VARIANTS),
        tuple(int(port) for port in REDIRECT_PORTS),
        (("redirect_status_hop", "inspect_effect_channel"), ("redirect_location_scope", "inspect_control_comparison")),
        redirect_source_sha256(),
    ),
    "logic_access": FamilySpec(
        "logic_access",
        ("red", "blue", "green"),
        tuple(int(port) for port in LOGIC_ACCESS_V3_PORTS),
        (("logic_outcome_transition", "inspect_effect_channel"), ("logic_invariant_control", "inspect_control_comparison")),
        logic_access_v3_source_sha256(),
    ),
}


def serve(server: Any) -> tuple[Any, threading.Thread]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=3.0)


def _encoding_variant(seed: int) -> tuple[str, str]:
    # Both entries are observable request encodings.  The second also exercises
    # a POST/form channel in SQL/redirect fixtures.
    return ("plain", "url_percent") if seed % 2 else ("plain", "url_percent")


def _family_pre_tokens(*, family: str, slot: str, question: str, method: str, placement: str, encoding: str) -> list[str]:
    role = "effect" if question == "inspect_effect_channel" else "control_comparison"
    return [
        "[BOS]",
        "phase=pre_question",
        f"family={family}",
        f"method={method}",
        f"placement={placement}",
        f"encoding={encoding}",
        "scope=loopback_fixture",
        "fresh_reset=1",
        "reference_sent=1",
        "negative_sent=1",
        "candidate_sent=1",
        "question_budget=1",
        "unknown_slot_count=1",
        f"unknown_role={role}",
        f"unknown_slot={slot}",
        "[CTX_END]",
    ]


def _coarse_pre_tokens(*, family: str, method: str, placement: str, encoding: str) -> list[str]:
    return [
        "[BOS]",
        "phase=pre_question",
        f"family={family}",
        f"method={method}",
        f"placement={placement}",
        f"encoding={encoding}",
        "scope=loopback_fixture",
        "fresh_reset=1",
        "reference_sent=1",
        "negative_sent=1",
        "candidate_sent=1",
        "question_budget=1",
        "unknown_slot_count=1",
        "unknown_slot=omitted",
        "[CTX_END]",
    ]


def _post_tokens(*, family: str, slot: str, question: str, observable: list[str]) -> list[str]:
    return [
        "[BOS]",
        "phase=post_observation",
        f"family={family}",
        f"question_asked={question}",
        f"bound_slot={slot}",
        "fresh_reset=1",
        "reference_sent=1",
        "negative_sent=1",
        "candidate_sent=1",
        *observable,
        "[CTX_END]",
    ]


def _observable_dom(row: dict[str, Any]) -> list[str]:
    shape = dict(row.get("surface_shape") or {})
    projection = dict(row.get("oracle_projection") or {})
    channel = "attribute" if projection.get("marker_in_attribute") else "html_text" if projection.get("marker_in_html_text") else "header" if projection.get("marker_in_header") else "json" if projection.get("marker_in_json_value") else "absent"
    return [
        f"observed_status={dict(row.get('response_projection') or {}).get('status_code', 0) // 100}xx",
        f"observed_content={shape.get('content_type_class', 'other')}",
        f"observed_tags={bucket(shape.get('html_tag_count'), edges=(1, 2, 4, 8, 16))}",
        f"observed_attrs={bucket(shape.get('html_attribute_count'), edges=(1, 2, 4, 8, 16))}",
        f"observed_marker_channel={channel}",
    ]


def _observable_sql(row: dict[str, Any]) -> list[str]:
    response = dict(row.get("response_projection") or {})
    shape = dict(response.get("json_shape") or {})
    return [
        f"observed_status={int(response.get('status_code', 0) or 0) // 100}xx",
        f"observed_json_type={shape.get('type', 'other')}",
        f"observed_json_keys={bucket(shape.get('key_count'), edges=(1, 2, 3, 4, 5, 8))}",
        f"observed_location={int(bool(response.get('location_present')))}",
        f"observed_body_bucket={bucket(response.get('body_length'), edges=(32, 64, 96, 128, 192, 256))}",
    ]


def _observable_redirect(row: dict[str, Any]) -> list[str]:
    response = dict(row.get("response_projection") or {})
    return [
        f"observed_status={response.get('status_class', 'other')}",
        f"observed_content={response.get('content_type_class', 'other')}",
        f"observed_location={response.get('location_class', 'absent')}",
        f"observed_location_present={int(bool(response.get('location_present')))}",
        f"observed_body_bucket={bucket(response.get('body_length'), edges=(32, 64, 96, 128, 192, 256))}",
    ]


def _logic_visible_projection(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    decision = str(value.get("decision", value.get("phase", value.get("result", "absent"))))
    scope = str(value.get("zone", value.get("scope", value.get("area", "absent"))))
    issued = value.get("issued", "absent")
    if isinstance(issued, bool):
        issued_class = "bool_true" if issued else "bool_false"
    elif isinstance(issued, int):
        issued_class = "int_nonzero" if issued else "int_zero"
    elif isinstance(issued, str):
        issued_class = "text_yes" if issued.casefold() == "yes" else "text_other"
    else:
        issued_class = "absent"
    return {
        "status_class": f"{response.status_code // 100}xx",
        "json_key_count": len(value),
        "decision": decision,
        "scope": scope,
        "issued_class": issued_class,
        "body_length": len(response.content),
        "body_sha256": hashlib.sha256(response.content).hexdigest(),
    }


def _observable_logic(row: dict[str, Any]) -> list[str]:
    response = dict(row.get("visible_response_projection") or {})
    return [
        f"observed_status={response.get('status_class', 'other')}",
        f"observed_json_keys={bucket(response.get('json_key_count'), edges=(1, 2, 3, 4, 5, 8))}",
        f"observed_decision={response.get('decision', 'absent')}",
        f"observed_scope={response.get('scope', 'absent')}",
        f"observed_issued={response.get('issued_class', 'absent')}",
    ]


def _collect_twice(collect_one: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    first, second = collect_one(), collect_one()
    first_hash = str(first.get("evidence_hash") or dict(first.get("evidence") or {}).get("evidence_hash") or "")
    second_hash = str(second.get("evidence_hash") or dict(second.get("evidence") or {}).get("evidence_hash") or "")
    if not first_hash or first_hash != second_hash:
        raise RuntimeError("fresh replay evidence was not stable")
    returned = dict(first)
    returned["fresh_replay_evidence_hashes"] = [first_hash, second_hash]
    returned["fresh_replay_count"] = 2
    return returned


def _collect_dom(*, variant: str, port: int, seed: int, encoding: str, case: str, source_hash: str) -> dict[str, Any]:
    marker = f"pg278-dom-{seed}"
    base_url = f"http://127.0.0.1:{port}"
    desired_lab = f"{'attr' if case == 'positive' else 'text'}-{'plain' if encoding == 'plain' else 'url-percent'}"
    specs = default_heterogeneous_surface_v3_specs(dataset_id=f"pg278-dom-{variant}-{seed}", target=base_url, marker=marker)
    spec = next(item for item in specs if str(item.get("lab_id")) == desired_lab)
    def collect_one() -> dict[str, Any]:
        return asyncio.run(HeterogeneousSurfaceV3Collector(base_url=base_url, target_instance_id=f"pg278-dom-{variant}-{seed}", source_hash=source_hash).collect(spec))
    raw = _collect_twice(collect_one)
    return {"method": "GET", "placement": "query", "encoding": encoding, "expected_positive": bool(raw.get("rule_ir_result")), "source_evidence_hash": str(raw.get("evidence", {}).get("evidence_hash", "")), "observable": _observable_dom(raw), "raw": raw}


def _collect_sql(*, variant: str, port: int, seed: int, encoding: str, case: str) -> dict[str, Any]:
    method = "GET" if encoding == "plain" else "POST"
    mode = "branch" if case == "positive" else "baseline"
    target = f"http://127.0.0.1:{port}"
    def collect_one() -> dict[str, Any]:
        return collect_sql_v6(target=target, port=port, variant=variant, method=method, mode=mode, sample_id=f"pg278-sql-{seed}-{encoding}-{case}")
    raw = _collect_twice(collect_one)
    oracle = dict(raw.get("oracle_projection") or {})
    return {"method": method, "placement": "query" if method == "GET" else "form", "encoding": "query_plain" if method == "GET" else "form_urlencoded", "expected_positive": bool(oracle.get("candidate_signal")), "source_evidence_hash": str(raw.get("evidence_hash", "")), "observable": _observable_sql(raw), "raw": raw}


def _collect_redirect(*, variant: str, port: int, seed: int, encoding: str, case: str) -> dict[str, Any]:
    method = "GET" if encoding == "plain" else "POST"
    mode = "hop" if case == "positive" else "baseline"
    target = f"http://127.0.0.1:{port}"
    def collect_one() -> dict[str, Any]:
        return collect_redirect(target=target, port=port, variant=variant, method=method, mode=mode, sample_id=f"pg278-redirect-{seed}-{encoding}-{case}")
    raw = _collect_twice(collect_one)
    oracle = dict(raw.get("oracle_projection") or {})
    return {"method": method, "placement": "query" if method == "GET" else "form", "encoding": "query_plain" if method == "GET" else "form_urlencoded", "expected_positive": bool(oracle.get("typed_positive")), "source_evidence_hash": str(raw.get("evidence_hash", "")), "observable": _observable_redirect(raw), "raw": raw}


def _logic_spec(*, target: str, seed: int, encoding: str, case: str) -> dict[str, Any]:
    marker = f"pg278-logic-{seed}"
    suffix = "plain" if encoding == "plain" else "url-percent"
    prefix = "logic-pg18-permit-boundary" if case == "positive" else "logic-pg18-permit-owner-control"
    specs = default_logic_access_v3_specs(dataset_id=f"pg278-logic-{seed}", target=target, marker=marker)
    return next(item for item in specs if str(item.get("lab_id")) == f"{prefix}-{suffix}")


def _collect_logic(*, variant: str, port: int, seed: int, encoding: str, case: str, source_hash: str) -> dict[str, Any]:
    target = f"http://127.0.0.1:{port}"
    spec = _logic_spec(target=target, seed=seed, encoding=encoding, case=case)
    parsed = urlsplit(str(spec["path"]))
    values = {str(key): str(items[0]) for key, items in parse_qs(parsed.query, keep_blank_values=True).items() if items}
    def collect_one() -> dict[str, Any]:
        collected = asyncio.run(LogicAccessV3Collector(base_url=target, target_instance_id=f"pg278-logic-{variant}-{seed}", source_hash=source_hash).collect(spec))
        with httpx.Client(base_url=target, timeout=5.0, follow_redirects=False, cookies={}) as client:
            visible = client.get(str(spec["path"]))
        collected["visible_response_projection"] = _logic_visible_projection(visible)
        return collected
    raw = _collect_twice(collect_one)
    _, _, oracle = logic_evaluate(parsed.path, values, variant)
    # This is request semantics, not the evaluator's answer.  In the prior
    # draft, an owner control and an unexpected member grant collapsed to the
    # same public response projection (200 / grant / private).  A policy that
    # cannot see the abstract request condition cannot learn whether that grant
    # is expected.  Preserve the minimum generalizable condition while still
    # excluding raw query text, the full URL, and evaluator output.
    request_semantics = {
        "subject_role": "owner" if values.get("actor") == "owner" else "non_owner",
        "credit_state": "nonzero" if values.get("credit", "0") not in {"", "0"} else "zero",
        "route_role": "protected_access",
    }
    observable = _observable_logic(raw) + [f"request_subject_role={request_semantics['subject_role']}", f"request_credit_state={request_semantics['credit_state']}", f"request_route_role={request_semantics['route_role']}"]
    return {"method": "GET", "placement": "query", "encoding": encoding, "expected_positive": bool(oracle.get("positive")), "source_evidence_hash": sha({"collector": str(raw.get("evidence", {}).get("evidence_hash", "")), "visible": raw.get("visible_response_projection"), "request_semantics": request_semantics}), "observable": observable, "request_semantics": request_semantics, "raw": raw}


def _family_collectors(family: str, *, variant: str, port: int, seed: int, encoding: str, case: str, source_hash: str) -> dict[str, Any]:
    if family == "dom_effect":
        return _collect_dom(variant=variant, port=port, seed=seed, encoding=encoding, case=case, source_hash=source_hash)
    if family == "sql_differential":
        return _collect_sql(variant=variant, port=port, seed=seed, encoding=encoding, case=case)
    if family == "redirect_contract":
        return _collect_redirect(variant=variant, port=port, seed=seed, encoding=encoding, case=case)
    if family == "logic_access":
        return _collect_logic(variant=variant, port=port, seed=seed, encoding=encoding, case=case, source_hash=source_hash)
    raise ValueError(f"unknown PG-278 family: {family}")


def _server_for(family: str, *, variant: str, port: int) -> Any:
    if family == "dom_effect":
        return make_heterogeneous_surface_v3_fixture_server(port=port, variant=variant)
    if family == "sql_differential":
        return make_sql_v6_fixture_server(port=port, variant=variant)
    if family == "redirect_contract":
        return make_redirect_server(port=port, variant=variant)
    if family == "logic_access":
        return make_logic_access_v3_fixture_server(port=port, variant=variant)
    raise ValueError(f"unknown PG-278 family: {family}")


def _rejected(slot: str, question: str, all_slots: tuple[tuple[str, str], tuple[str, str]]) -> list[dict[str, str]]:
    other_slot, other_question = next(item for item in all_slots if item[0] != slot)
    other_question = other_question if other_question != question else "inspect_control_comparison"
    return [
        {"question": other_question, "action": "ask_observation", "belief": "unresolved", "slot": other_slot},
        {"question": "replay_evidence", "action": "review_evidence", "belief": "supported", "slot": slot},
    ]


def _task_rows(*, family_spec: FamilySpec, variant: str, seed: int, split: str, encoding: str, case: str, row: dict[str, Any], pair_ids: dict[str, str]) -> list[dict[str, Any]]:
    method, placement = str(row["method"]), str(row["placement"])
    expected_positive = bool(row["expected_positive"])
    outcome = {"question": "replay_evidence", "action": "review_evidence", "belief": "supported"} if expected_positive else {"question": "explain_failure", "action": "abstain", "belief": "rejected"}
    result: list[dict[str, Any]] = []
    for slot, question in family_spec.slots:
        pair_id = f"pg278:{family_spec.name}:{variant}:{seed}:{encoding}:{slot}"
        record_id = f"{pair_id}:{case}"
        pre = _family_pre_tokens(family=family_spec.name, slot=slot, question=question, method=method, placement=placement, encoding=str(row["encoding"]))
        coarse = _coarse_pre_tokens(family=family_spec.name, method=method, placement=placement, encoding=str(row["encoding"]))
        post = _post_tokens(family=family_spec.name, slot=slot, question=question, observable=list(row["observable"]))
        evidence_hash = sha({"source": row["source_evidence_hash"], "family": family_spec.name, "variant": variant, "seed": seed, "encoding": row["encoding"], "case": case, "slot": slot, "observable": row["observable"]})
        result.append({
            "schema_version": "pg278-question-slot-record-v1",
            "record_id": record_id,
            "pair_id": pair_id,
            "paired_opposite_record_id": pair_ids.get(slot, ""),
            "split": split,
            "family": family_spec.name,
            "implementation": variant,
            "collection_seed": seed,
            "encoding": str(row["encoding"]),
            "request_projection": {"method": method, "placement": placement, "parameter_count_bucket": "2", "abstract_condition": dict(row.get("request_semantics") or {}), "fresh_reset": True, "replay_count": int(row["raw"].get("fresh_replay_count", 0))},
            "missing_observation_slot": slot,
            "question_class": question,
            "coarse_pre_question_context_tokens": coarse,
            "pre_question_context_tokens": pre,
            "post_observation_context_tokens": post,
            "targets": {"pre_question": {"question": question, "action": "ask_observation", "belief": "unresolved", "slot": slot}, "post_observation": {**outcome, "slot": slot}},
            "preference_rejected": {"pre_question": _rejected(slot, question, family_spec.slots), "post_observation": [{"question": "explain_failure" if expected_positive else "replay_evidence", "action": "abstain" if expected_positive else "review_evidence", "belief": "rejected" if expected_positive else "supported", "slot": slot}]},
            "failure_repair": {"failure_before_question": "missing_required_observation", "failure_after_observation": "effect_not_supported" if not expected_positive else "none", "repair_action": "abstain_with_evidence_link" if not expected_positive else "review_confirmed_evidence", "repair_succeeded": True},
            "labels": {"expected_positive": expected_positive, "case": case, "post_observation_outcome": "supported" if expected_positive else "rejected"},
            "source": {"kind": "controlled_loopback_fixture", "fixture_source_sha256": family_spec.source_hash, "source_evidence_hash": row["source_evidence_hash"], "fresh_replay_evidence_hashes": list(row["raw"].get("fresh_replay_evidence_hashes") or [])},
            "evidence_hash": evidence_hash,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "oracle_in_context": False,
            "training_lane": "controlled_research_only",
            "memory_promotion_allowed": False,
        })
    return result


def _collision_summary(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(sha(row[field]), []).append(row)
    conflicts: list[dict[str, Any]] = []
    for key, items in groups.items():
        targets = {sha(item["targets"]["pre_question"]) for item in items}
        if len(targets) > 1:
            conflicts.append({"context_sha256": key, "count": len(items), "target_count": len(targets), "families": sorted({str(item["family"]) for item in items})})
    return {"group_count": len(groups), "conflict_group_count": len(conflicts), "conflicting_record_count": sum(int(item["count"]) for item in conflicts), "conflicts": conflicts}


def _pair_rows(*, family_spec: FamilySpec, variant: str, seed: int, split: str, encoding: str, positive: dict[str, Any], negative: dict[str, Any]) -> list[dict[str, Any]]:
    placeholders = {slot: f"pg278:{family_spec.name}:{variant}:{seed}:{encoding}:{slot}:negative" for slot, _ in family_spec.slots}
    negatives = _task_rows(family_spec=family_spec, variant=variant, seed=seed, split=split, encoding=encoding, case="negative", row=negative, pair_ids={slot: f"pg278:{family_spec.name}:{variant}:{seed}:{encoding}:{slot}:positive" for slot, _ in family_spec.slots})
    positives = _task_rows(family_spec=family_spec, variant=variant, seed=seed, split=split, encoding=encoding, case="positive", row=positive, pair_ids=placeholders)
    return positives + negatives


def main() -> None:
    records: list[dict[str, Any]] = []
    implementation_map: dict[str, Any] = {}
    for family, spec in FAMILIES.items():
        implementation_map[family] = {"train": list(spec.variants[:2]), "holdout": spec.variants[2], "ports": list(spec.ports), "fixture_source_sha256": spec.source_hash}
        for index, (variant, port) in enumerate(zip(spec.variants, spec.ports)):
            split = "implementation_train" if index < 2 else "implementation_holdout"
            server, thread = serve(_server_for(family, variant=variant, port=port))
            try:
                for seed in SEEDS:
                    for encoding in _encoding_variant(seed):
                        positive = _family_collectors(family, variant=variant, port=port, seed=seed, encoding=encoding, case="positive", source_hash=spec.source_hash)
                        negative = _family_collectors(family, variant=variant, port=port, seed=seed, encoding=encoding, case="negative", source_hash=spec.source_hash)
                        records.extend(_pair_rows(family_spec=spec, variant=variant, seed=seed, split=split, encoding=encoding, positive=positive, negative=negative))
            finally:
                stop(server, thread)
    train = [row for row in records if row["split"] == "implementation_train"]
    holdout = [row for row in records if row["split"] == "implementation_holdout"]
    slot_counts = {slot: sum(row["missing_observation_slot"] == slot for row in records) for spec in FAMILIES.values() for slot, _ in spec.slots}
    family_counts = {family: {"total": sum(row["family"] == family for row in records), "positive": sum(row["family"] == family and bool(row["labels"]["expected_positive"]) for row in records), "negative": sum(row["family"] == family and not bool(row["labels"]["expected_positive"]) for row in records)} for family in FAMILIES}
    coarse = _collision_summary(records, "coarse_pre_question_context_tokens")
    enriched = _collision_summary(records, "pre_question_context_tokens")
    post = _collision_summary(records, "post_observation_context_tokens")
    payload: dict[str, Any] = {
        "schema_version": "pg278-multifamily-question-dataset-v1",
        "purpose": "Controlled multi-family slot-binding and failure-to-repair study; not real-target capability data.",
        "source": {"loopback_only": True, "external_network": False, "fixtures": implementation_map, "real_multifamily_gold_rows": 0},
        "split_contract": {"train_implementations": {family: list(spec.variants[:2]) for family, spec in FAMILIES.items()}, "holdout_implementations": {family: spec.variants[2] for family, spec in FAMILIES.items()}, "collection_seeds": list(SEEDS), "encodings_per_family": 2, "implementation_disjoint": True},
        "records": records,
        "counts": {"total": len(records), "train": len(train), "holdout": len(holdout), "families": family_counts, "slots": slot_counts},
        "projection_collision_audit": {"coarse": coarse, "enriched": enriched, "post": post, "coarse_training_allowed": False, "enriched_training_allowed": enriched["conflict_group_count"] == 0 and post["conflict_group_count"] == 0},
        "data_contract": {"required_families": 4, "required_implementations_per_family": 3, "required_seeds_per_implementation": 3, "required_encodings_per_seed": 2, "required_missing_slots": 8, "required_fresh_replays": 2, "required_gold_per_family": 24, "required_hard_negative_per_family": 12, "raw_payload_in_context": False, "raw_response_body_in_context": False, "oracle_in_context": False, "promotion_blocked": True},
        "training_contract": {"use": "controlled_research_only", "model_heads": ["question", "action", "belief", "slot"], "train_on": ["pre_question_context_tokens", "post_observation_context_tokens"], "never_train_on": ["coarse_pre_question_context_tokens", "source.oracle", "raw payload", "raw response body"], "memory_promotion_blocked": True},
    }
    payload["dataset_sha256"] = sha(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "dataset": str(OUTPUT.relative_to(ROOT)), "dataset_sha256": payload["dataset_sha256"], "counts": payload["counts"], "collision": payload["projection_collision_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
