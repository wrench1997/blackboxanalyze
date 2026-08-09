"""PG-68: adapt the real local/Docker typed-oracle run into a quarantined trace.

PG-52 is the first lane in this repository whose effect labels come from a
real disposable Pikachu image rather than an in-process fixture.  PG-68 does
not treat that evidence as training data.  It audits the boundary between a
real evaluator result and a learnable trace, then writes an evaluation-only
catalog and a rejected trace manifest.

Two checks are deliberately strict:

* a fresh reset must be a distinct target instance for every action; reusing
  one GET container for several routes is not per-action freshness;
* family-held-out evidence must contain a family absent from the complete
  training registry, not merely a route that was not used in this run.

The adapter keeps only safe abstract probe classes, bounded projections and
hashes.  It never writes the runtime canary, response body or credentials.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.detection_payload import build_detection_payload  # noqa: E402
from app.payload_catalog import write_catalog  # noqa: E402
from app.trace_aligned_dataset import sha256_json, validate_trace_step  # noqa: E402


PROTOCOL_ID = "pg-pk-68-real-local-docker-typed-oracle-adapter-v1"
SCHEMA_VERSION = "sift-pg68-real-local-docker-typed-oracle-adapter-v1"
IMAGE = "tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
PG52_REPORT_PATH = ROOT / "research" / "pg52_authoritative_local_oracle_report_v1.json"
REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
REPORT_PATH = ROOT / "research" / "pg68_real_local_docker_typed_oracle_adapter_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg68_real_local_docker_typed_oracle_adapter_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg68_real_local_docker_typed_oracle_adapter_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg68_real_local_docker_typed_oracle_adapter_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg68_real_local_docker_typed_oracle_adapter_report_v1.md"

REQUIRED_CONTRACT = (
    "browser_execution",
    "browser_offline_response_renderer",
    "static_resource_tags_stripped",
    "controlled_event_dispatch_is_explicit",
    "sql_ast_differential",
    "controlled_redirect",
    "positive_requires_negative_control",
    "positive_requires_fresh_reset",
    "positive_requires_evidence_hash",
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _safe_instance(value: Any) -> str:
    return str(value)[:24]


def _bounded_projection(value: Any) -> dict[str, Any]:
    """Keep a non-empty model projection when a browser oracle owns the response.

    PG-52's browser path intentionally returns no HTTP body projection because
    the offline page is consumed by Playwright.  The trace contract still
    requires a bounded projection, so use a typed absence marker rather than
    smuggling evaluator output into the model channel.
    """

    if isinstance(value, dict) and value:
        return dict(value)
    projection = {
        "status_class": "unknown",
        "content_type": "unknown",
        "body_length_bucket": "unknown",
        "response_projection_available": False,
    }
    projection["projection_sha256"] = sha256_json(projection)
    return projection


def _audited_reset(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize the source reset claim to the adapter's stricter result."""

    reset = dict(row.get("fresh_reset") or {})
    instance = _safe_instance(reset.get("target_instance_id", ""))
    reset["reset_id"] = f"pg68-reset-round-{instance}"
    reset.update({"fresh_target": False, "reset_scope": "transport_container_round", "per_action_required": True, "reuse_detected": True})
    return reset


def _adapter_evidence_hash(row: dict[str, Any], index: int) -> str:
    """Bind the adapter projections and its reset audit without raw data."""

    return sha256_json({
        "opaque_step": f"pg68-step-{int(index):02d}",
        "candidate_response": _bounded_projection(row.get("candidate_response")),
        "control_response": _bounded_projection(row.get("control_response")),
        "oracle": dict(row.get("oracle") or {}),
        "control_oracle": dict(row.get("control_oracle") or {}),
        "reset": _audited_reset(row),
    })


def _probe_spec(row: dict[str, Any]) -> tuple[str, str, str, str]:
    family = str(row["family"])
    mode = str(row.get("oracle", {}).get("oracle_id", ""))
    if family == "xss":
        return "inert_dom_markup", "dom_event_class", "controlled_detached_dom_v1", "typed_dom_execution"
    if family == "injection":
        surface = str(row.get("surface", ""))
        if "search" in surface:
            fragment = "operator_like"
        elif "blind" in surface:
            fragment = "blind_boolean"
        else:
            fragment = "quoted_value"
        return "sql_channel_class", fragment, "synthetic_sql_ast_differential_v1", "typed_ast_differential"
    if family == "url_redirect":
        return "http_canary", "loopback_destination_class", "pikachu_bounded_http_projection_v1", "typed_redirect_destination"
    raise ValueError(f"unsupported PG-68 family: {family}")


def _source_scope(method: str) -> str:
    return "http://127.0.0.1:8768" if method == "POST" else "http://127.0.0.1:8767"


def _route_id(index: int) -> str:
    return f"pg68-route-{int(index):02d}"


def _marker(case_id: str) -> str:
    return "pg68-" + re.sub(r"[^A-Za-z0-9.-]", "-", case_id) + "-probe"


def _catalog_sample(row: dict[str, Any], source_scope: str, index: int) -> dict[str, Any]:
    probe_kind, probe, expected_oracle, expected_signal = _probe_spec(row)
    marker = _marker(str(row["case_id"]))
    method = str(row["method"]).upper()
    field = str(row["path"]).split("/")[-1].replace(".php", "")
    # The field name is retained only as a bounded form/query slot; no runtime
    # canary value is persisted.  The replay marker is an inert abstract tag.
    field = {"xss_reflected_get": "message", "xss_dom_source": "text", "xss_reflected_post": "message", "sqli_str": "name", "sqli_search": "name", "sqli_blind_b": "name", "url_redirect": "url"}.get(str(row["surface"]), "value")
    payload = build_detection_payload(
        target=source_scope,
        method=method,
        path=str(row["path"]),
        headers={"accept": "text/html", "x-sift-probe": marker},
        marker=marker,
        probe=probe,
        probe_kind=probe_kind,
        form={field: marker} if method == "POST" else {},
        expected={"signal": expected_signal, "negative_control": "matched_pair", "typed_oracle": True},
    )
    reset = _audited_reset(row)
    replay: dict[str, Any] = {
        "target": source_scope,
        "method": method,
        "path": str(row["path"]),
        "params": {field: marker} if method == "GET" else {},
        "fresh_reset": reset,
        "transport": "docker_loopback",
    }
    if method == "POST":
        replay["form"] = {field: marker}
    rule_ir = {
        "op": "and",
        "args": [
            {"op": "eq", "left": {"op": "field", "path": "oracle.positive"}, "right": {"op": "const", "value": True}},
            {"op": "eq", "left": {"op": "field", "path": "oracle.positive_authority"}, "right": {"op": "const", "value": True}},
        ],
    }
    return {
        "sample_id": "pg68-" + str(row["case_id"]),
        "payload": payload,
        "probe_artifact": {"original": probe, "encoding": "abstract_class", "probe_sha256": _sha256_text(probe)},
        "semantic": {"family": str(row["family"]), "surface": str(row["surface"]), "expected_oracle": expected_oracle, "expected_signal": expected_signal},
        "pair": {"pair_id": "pg68-pair-" + str(row["case_id"]), "variant": "abstract_class", "surface_role": str(row["surface"]), "encoding_depth": 0},
        "counterfactual": {"kind": "negative_control", "intervention": "matched_control", "source_sample_id": "pg68-" + str(row["case_id"])},
        "replay": replay,
        "response_projection": _bounded_projection(row.get("candidate_response")),
        "oracle_projection": dict(row.get("oracle") or {}),
        "evidence": {"source_candidate_evidence_sha256": str(row["evidence_sha256"]), "source_control_evidence_sha256": str(row["negative_control"]["control_evidence_sha256"]), "adapter_evidence_sha256": _adapter_evidence_hash(row, index)},
        "rule_ir": rule_ir,
        "rule_ir_result": bool((row.get("oracle") or {}).get("positive")),
        "evaluator_state_visible": False,
    }


def _trace_step(row: dict[str, Any], index: int) -> dict[str, Any]:
    method = str(row["method"]).upper()
    family = str(row["family"])
    surface = str(row["surface"])
    field = {"xss_reflected_get": "message", "xss_dom_source": "text", "xss_reflected_post": "message", "sqli_str": "name", "sqli_search": "name", "sqli_blind_b": "name", "url_redirect": "url"}.get(surface, "value")
    probe_kind, probe, _oracle_name, _signal = _probe_spec(row)
    step_id = f"pg68-step-{index:02d}"
    action: dict[str, Any] = {
        "method": method,
        "route_template_id": _route_id(index),
        "placement": "form" if method == "POST" else "query",
        "encoding_chain": ["identity"],
        "probe_ref": f"pg68-probe-{index:02d}",
        "probe_sha256": _sha256_text(probe),
        "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True},
    }
    if method == "POST":
        action["form_field_names"] = [field]
    # The adapter converts the claimed PG-52 reset into an explicit audit
    # projection.  Reused containers are not fresh per action.
    reset = _audited_reset(row)
    oracle = dict(row.get("oracle") or {})
    oracle["negative_control_pair_id"] = f"pg68-control-{index:02d}"
    step: dict[str, Any] = {
        "episode_id": "pg68-real-local-replay",
        "step_id": step_id,
        "parent_step_id": None if index == 0 else f"pg68-step-{index - 1:02d}",
        "sampling_seed": 6800 + index,
        "hypothesis": "surface_hypothesis",
        "belief_before": {"unknown_surface": 1.0},
        "action_manifest": action,
        "baseline_projection": _bounded_projection(row.get("control_response")),
        "response_projection": _bounded_projection(row.get("candidate_response")),
        "oracle_projection": oracle,
        "belief_after": {"candidate_surface": 1.0},
        "decision": "confirmed_positive" if bool(oracle.get("positive")) else "confirmed_negative",
        "next_action": "stop_confirmed" if bool(oracle.get("positive")) else "abstain",
        "fresh_reset": reset,
        "evidence_sha256": _adapter_evidence_hash(row, index),
        "dataset_stage": "evaluation_only_rejected",
        "echo": {},
    }
    echo_body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action")}
    step["echo"] = {"sha256": sha256_json(echo_body)}
    return step


def _family_set_from_registry() -> set[str]:
    registry = _load(REGISTRY_PATH)
    families: set[str] = set()
    for target in registry.get("targets", []):
        if bool(target.get("training_eligible")):
            families.update(str(item) for item in target.get("family_set", []))
    return families


def main() -> int:
    source = _load(PG52_REPORT_PATH)
    rows = list(source.get("detection_results") or [])
    if not rows:
        raise RuntimeError("PG-68 requires the completed PG-52 real-local report")
    contract = dict(source.get("oracle_contract") or {})
    contract_ok = all(bool(contract.get(key)) for key in REQUIRED_CONTRACT)
    target = dict(source.get("target") or {})
    target_ok = target.get("image") == IMAGE and bool(target.get("loopback_only")) and not bool(target.get("external_network"))
    families = sorted({str(row["family"]) for row in rows})
    methods = {str(row["method"]).upper() for row in rows}
    unique_instances = sorted({_safe_instance(row.get("fresh_reset", {}).get("target_instance_id", "")) for row in rows})
    valid_hashes = sum(bool(HASH_RE.fullmatch(_adapter_evidence_hash(row, index).casefold())) for index, row in enumerate(rows))
    negative_matches = sum(bool(row.get("negative_control", {}).get("matched")) for row in rows)
    negative_oracles = sum(not bool(row.get("control_oracle", {}).get("positive")) for row in rows)
    modalities = sorted({str(row.get("oracle", {}).get("modality", "")) for row in rows})
    global_training_families = sorted(_family_set_from_registry())
    unseen_families = sorted(set(families) - set(global_training_families))
    reset_per_action = len(unique_instances) == len(rows) and len(rows) > 0
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Build an evaluation-only catalog.  Its presence does not authorize a
    # training write; the report records the hard gate below.
    source_groups: dict[str, list[dict[str, Any]]] = {"GET": [], "POST": []}
    for index, row in enumerate(rows):
        source_groups[str(row["method"]).upper()].append(_catalog_sample(row, _source_scope(str(row["method"]).upper()), index))
    catalog_sources = []
    for method, samples in source_groups.items():
        if not samples:
            continue
        scope = _source_scope(method)
        catalog_sources.append({
            "provenance": {
                "source_id": f"pg68-pikachu-{method.casefold()}-typed",
                "source_type": "authorized_local_container",
                "origin": "research/pg52_authoritative_local_oracle_report_v1.json",
                "license": "local_container",
                "authorization": "workspace_local_only",
                "scope": [scope],
                "captured_at": captured_at,
                "authorized_for": ["training", "local_replay", "holdout_evaluation"],
                "external_network": False,
                "evaluator_state_visible": False,
                "container_image_digest": IMAGE.split("@", 1)[1],
            },
            "samples": samples,
        })
    catalog = write_catalog(CATALOG_PATH, {"schema_version": "sift-authorized-payload-catalog-v1", "catalog_id": "pg68-real-local-docker-evaluation-only", "sources": catalog_sources})

    rejected_steps: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}
    for index, row in enumerate(rows):
        step = _trace_step(row, index)
        try:
            validate_trace_step(step)
            error = "unexpectedly_accepted"
        except ValueError as exc:
            error = str(exc)
        rejection_reasons[error] = rejection_reasons.get(error, 0) + 1
        rejected_steps.append({
            "step_id": step["step_id"],
            "method": step["action_manifest"]["method"],
            "route_template_id": step["action_manifest"]["route_template_id"],
            "probe_ref": step["action_manifest"]["probe_ref"],
            "probe_sha256": step["action_manifest"]["probe_sha256"],
            "target_instance_id": _safe_instance(row["fresh_reset"].get("target_instance_id", "")),
            "oracle_after_action": step["oracle_projection"],
            "decision": step["decision"],
            "fresh_reset_audit": step["fresh_reset"],
            "evidence_sha256": step["evidence_sha256"],
            "validation_error": error,
            "raw_probe_stored": False,
            "raw_response_stored": False,
        })

    hard_checks = {
        "real_image_pinned": target_ok,
        "typed_oracle_contract_complete": contract_ok,
        "get_post_both_covered": {"GET", "POST"}.issubset(methods),
        "matched_negative_controls": negative_matches == len(rows) and negative_oracles == len(rows),
        "evidence_hash_per_action": valid_hashes == len(rows),
        "fresh_reset_per_action": reset_per_action,
        "family_heldout_replay": bool(unseen_families),
        "independent_source_count": False,
    }
    hard_gate_passed = all(hard_checks.values())
    blockers = [name for name, passed in hard_checks.items() if not passed]
    trace = {
        "schema_version": "sift-pg68-evaluation-replay-trace-v1",
        "protocol_id": PROTOCOL_ID,
        "evaluation_only": True,
        "training_eligible": False,
        "model_retrained_on_pg68": False,
        "model_input_family_leakage": False,
        "episode_count": 0,
        "candidate_step_count": len(rejected_steps),
        "methods": sorted(methods),
        "replay_status": "rejected_before_training",
        "rejection_reasons": rejection_reasons,
        "steps": rejected_steps,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "real_local_adapter_audit_completed",
        "source": {
            "report": str(PG52_REPORT_PATH.relative_to(ROOT)),
            "source_status": source.get("status"),
            "image": target.get("image"),
            "source_instance_count": len(unique_instances),
            "independent_source_count": 1,
            "independent_seed_count": 1,
            "independent_implementation_count": 1,
        },
        "scope": {"case_count": len(rows), "families": families, "methods": sorted(methods), "modalities": modalities, "loopback_only": target_ok, "raw_payloads_stored": False, "raw_response_bodies_stored": False},
        "metrics": {
            "typed_positive_count": sum(bool(row.get("oracle", {}).get("positive")) for row in rows),
            "matched_negative_control_count": negative_matches,
            "negative_control_oracle_false_count": negative_oracles,
            "evidence_hash_valid_count": valid_hashes,
            "get_post_covered": {"GET": sum(str(row["method"]).upper() == "GET" for row in rows), "POST": sum(str(row["method"]).upper() == "POST" for row in rows)},
            "unique_target_instance_count": len(unique_instances),
            "fresh_reset_per_action": reset_per_action,
            "training_registry_family_set": global_training_families,
            "family_holdout_candidate_count": len(unseen_families),
            "family_holdout_candidates": unseen_families,
            "oracle_modalities": modalities,
            "catalog_sample_count": sum(len(source_item["samples"]) for source_item in catalog["sources"]),
        },
        "oracle_contract": contract,
        "hard_gate": {"status": "passed" if hard_gate_passed else "blocked", "checks": hard_checks, "blocking_reasons": blockers, "claim_allowed": hard_gate_passed},
        "promotion": {
            "status": "eligible_for_training" if hard_gate_passed else "quarantined_evaluation_only",
            "evaluation_catalog_generated": True,
            "training_catalog_generated": False,
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "reason": ";".join(blockers) if blockers else "all preregistered gates passed",
        },
        "artifacts": {"catalog": str(CATALOG_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT))},
        "formal_claim": {"allowed": False, "reason": "one pinned image and no per-action fresh reset or family-heldout evidence"},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg68-real-local-docker-typed-oracle-adapter-protocol-v1",
        "authorized_scope": {"target_host": "127.0.0.1", "pinned_image": IMAGE, "external_network": False, "state_change_allowed": False, "raw_persistence": False},
        "input_contract": {"source_report": "PG-52 real local typed oracle", "family_before_action_forbidden": True, "oracle_after_action_only": True, "catalog_role": "evaluation_only_quarantined", "trace_role": "rejected_until_hard_gate"},
        "required_gates": {"get_post_both": True, "matched_negative_control": True, "fresh_reset_per_action": True, "evidence_hash_per_action": True, "family_heldout": True, "independent_source": True, "raw_probe_and_response_persistence_forbidden": True},
        "run_result": {"status": "passed" if hard_gate_passed else "blocked", "hard_gate_checks": hard_checks, "training_allowed": False, "memory_promotion_allowed": False},
        "next_experiment": "PG69 per-action disposable Docker reset plus a genuinely unseen family/implementation before any training write",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PG-68 真实本地/Docker typed-oracle adapter",
        "",
        f"真实 PG-52 cases: {len(rows)}；typed positive: {report['metrics']['typed_positive_count']}；匹配阴性对照: {negative_matches}。",
        f"容器实例: {len(unique_instances)}；动作数: {len(rows)}；fresh reset/动作: `{reset_per_action}`；族外候选: {len(unseen_families)}。",
        "",
        "硬门: " + ("通过" if hard_gate_passed else "阻塞（不进入训练）"),
        "",
        "阻塞项: " + (", ".join(blockers) if blockers else "无"),
        "",
        "该 Catalog 仅用于评估审计；training_catalog_generated=false，长期记忆写入=false。",
        "",
        f"JSON: `{REPORT_PATH.relative_to(ROOT)}`",
        f"协议: `{PROTOCOL_PATH.relative_to(ROOT)}`",
        f"评估 Catalog: `{CATALOG_PATH.relative_to(ROOT)}`",
        f"拒绝 Trace: `{TRACE_PATH.relative_to(ROOT)}`",
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["hard_gate"]["status"], "case_count": len(rows), "fresh_reset_per_action": reset_per_action, "family_holdout_candidate_count": len(unseen_families), "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
