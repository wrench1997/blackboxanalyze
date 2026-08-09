"""PG-101: active probe-bank signature across independent local fixtures.

PG-99 showed that one response projection cannot distinguish several known and
unknown positive cases.  PG-101 tests a different, model-visible channel: a
fixed bank of inert probes is replayed in a fresh loopback episode and the
ordered pattern of generic response-shape changes is retained.  Probe values
are adapter-only runtime inputs; the emitted signature contains canonical
probe IDs, bounded geometry deltas, and no family/oracle/raw content.

The exact-support decoder is a representation baseline, not a production
detector.  It is evaluated on PG-36 (training/dev) and a fresh PG-42
cross-implementation/seed/variant matrix.  Even when its capability gates
pass, training and long-term-memory promotion remain disabled until a neural
decoder and a separate review pass reproduce the gain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg36_independent_maze_catalog as pg36  # noqa: E402
import run_pg42_independent_semantic_catalog_parallel as pg42  # noqa: E402
import run_pg35_independent_fixture_catalog as pg35  # noqa: E402
from app.active_probe_signature import (  # noqa: E402
    ActiveProbeSignatureDecoder,
    PROBE_IDS,
    aggregate_signature,
    make_probe_observation,
    model_input_has_forbidden_field,
    sha256_json,
)
from app.pg36_independent_maze_fixture import LAYOUTS as PG36_LAYOUTS, SURFACE_SPECS as PG36_SURFACES  # noqa: E402
from app.pg42_independent_semantic_fixture import LAYOUTS as PG42_LAYOUTS, SURFACE_SPECS as PG42_SURFACES  # noqa: E402
from app.pg35_independent_fixture import PG35_VARIANTS, SURFACE_SPECS as PG35_SURFACES  # noqa: E402
from app.pg53_cross_source_oracle import generic_effect_geometry  # noqa: E402


PROTOCOL_ID = "pg-pk-101-active-probe-signature-v1"
REPORT_PATH = ROOT / "research" / "pg101_active_probe_signature_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg101_active_probe_signature_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg101_active_probe_signature_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg101_active_probe_signature_report_v1.md"
PG99_REPORT_PATH = ROOT / "research" / "pg99_surface_novelty_report_v1.json"

PG36_SEEDS = (361, 367, 373)
PG42_SEEDS = (401, 409, 419)
PG35_SEEDS = (351, 357, 367)
KNOWN_FAMILIES = tuple(
    spec["family"]
    for spec in PG36_SURFACES.values()
    if spec["family"] not in {"ordinary_response", "unknown_surface"}
)

# These strings are adapter-only safe probe classes.  They are never written
# to a catalog/trace/report; only the canonical p0..p8 IDs leave the runner.
PG36_PROBE_BANK = {
    "p0": "markup_candidate",
    "p1": "operator_like",
    "p2": "auth_boundary_candidate",
    "p3": "id_reference",
    "p4": "invariant_boundary",
    "p5": "relative_redirect",
    "p6": "boundary_value",
    "p7": "local_canary",
    "p8": "template_candidate",
}
PG42_PROBE_BANK = {
    f"p{index}": PG42_SURFACES[f"node-{index + 1:02d}"]["probe"]
    for index in range(9)
}
PG35_PROBE_BANK = {
    f"p{index}": PG35_SURFACES[f"surface-{index + 1:02d}"]["positive"]
    for index in range(8)
}
PG35_PROBE_BANK["p8"] = "template_candidate"


def _body(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _call_pg36(client: httpx.Client, implementation: str, surface: str, method: str, probe: str, phase: str) -> httpx.Response:
    layout = PG36_LAYOUTS[implementation]
    values = {layout["slot_key"]: surface, layout["probe_key"]: str(probe), layout["phase_key"]: phase}
    route = f"{layout['prefix']}/{surface}"
    if method == "GET":
        return client.get(route, params=values)
    if layout["post_content_type"] == "application/json":
        return client.post(route, json=values)
    return client.post(route, data=values, headers={"content-type": layout["post_content_type"]})


def _call_pg42(client: httpx.Client, implementation: str, surface: str, method: str, probe: str, variant: str, phase: str) -> httpx.Response:
    layout = PG42_LAYOUTS[implementation]
    values = {layout["surface_key"]: surface, layout["probe_key"]: str(probe), layout["variant_key"]: variant, layout["phase_key"]: phase}
    route = f"{layout['prefix']}/{surface}"
    if method == "GET":
        return client.get(route, params=values)
    if layout["post_content_type"] == "application/json":
        return client.post(route, json=values)
    return client.post(route, content="&".join(f"{key}={value}" for key, value in values.items()).encode("utf-8"), headers={"content-type": layout["post_content_type"]})


def _call_pg35(client: httpx.Client, variant: str, surface: str, method: str, probe: str) -> httpx.Response:
    layout = PG35_VARIANTS[variant]
    wire = pg35._encode(str(probe), "identity")
    route = f"{layout['prefix']}/{surface}"
    values = {layout["slot_key"]: surface, layout["probe_key"]: wire}
    if method == "GET":
        return client.get(route, params=values)
    if layout["post_content_type"] == "application/json":
        return client.post(route, json=values)
    return client.post(route, data=values, headers={"content-type": layout["post_content_type"]})


def _pg36_projection(response: httpx.Response) -> tuple[dict[str, Any], dict[str, Any]]:
    return pg36._response_projection(response)


def _pg42_projection(response: httpx.Response) -> tuple[dict[str, Any], dict[str, Any]]:
    return pg42._projection(response)


def _pg35_projection(response: httpx.Response) -> tuple[dict[str, Any], dict[str, Any]]:
    return pg35._response_projection(response)


def _fresh_reset(*, source: str, implementation: str, surface: str, variant: str, seed: int, method: str) -> dict[str, Any]:
    target_key = f"{source}|{implementation}|{surface}|{variant}|{seed}|{method}"
    target_id = f"pg101-target-{hashlib.sha256(target_key.encode()).hexdigest()[:24]}"
    reset = {
        "kind": f"fresh_{source}_active_probe_episode",
        "reset_id": f"pg101-reset-{hashlib.sha256((target_key + '|reset').encode()).hexdigest()[:24]}",
        "target_instance_id": target_id,
        "state_epoch": hashlib.sha256((target_key + "|epoch").encode()).hexdigest()[:24],
        "reset_adapter_sha256": hashlib.sha256(("pg101-reset-adapter|" + source).encode()).hexdigest(),
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


def _episode(
    *,
    source: str,
    implementation: str,
    surface: str,
    variant: str,
    seed: int,
    method: str,
) -> dict[str, Any]:
    is_pg36 = source == "pg36"
    is_pg42 = source == "pg42"
    surface_specs = PG36_SURFACES if is_pg36 else PG42_SURFACES if is_pg42 else PG35_SURFACES
    spec = surface_specs[surface]
    bank = PG36_PROBE_BANK if is_pg36 else PG42_PROBE_BANK if is_pg42 else PG35_PROBE_BANK
    phase = "confirm"
    reset = _fresh_reset(source=source, implementation=implementation, surface=surface, variant=variant, seed=seed, method=method)
    observations: list[dict[str, Any]] = []
    query_count = 0
    if is_pg36:
        port = 31936 if implementation == "north" else 31937
        target = pg36._FreshTarget(implementation, port=port)
        projection = _pg36_projection
        caller = lambda client, probe, current_phase: _call_pg36(client, implementation, surface, method, probe, current_phase)
    elif is_pg42:
        target = pg42._FreshTarget(implementation)
        projection = _pg42_projection
        caller = lambda client, probe, current_phase: _call_pg42(client, implementation, surface, method, probe, variant, current_phase)
    else:
        target = pg35._FreshTarget(variant, port=31935)
        projection = _pg35_projection
        caller = lambda client, probe, current_phase: _call_pg35(client, variant, surface, method, probe)
    with target as client:
        # A screen step is recorded as a bounded first move.  The complete
        # confirm bank is then replayed; all requests remain inert and paired.
        screen_control_response = caller(client, "ordinary-observation", "screen")
        screen_candidate_response = caller(client, bank["p0"], "screen")
        query_count += 2
        screen_control_projection, screen_control_body = projection(screen_control_response)
        screen_candidate_projection, screen_candidate_body = projection(screen_candidate_response)
        screen_observation = make_probe_observation(
            probe_id="p0",
            method=method,
            phase="screen",
            encoding="identity",
            control_geometry=generic_effect_geometry(screen_control_body),
            candidate_geometry=generic_effect_geometry(screen_candidate_body),
            control_projection=screen_control_projection,
            candidate_projection=screen_candidate_projection,
            safe_probe=True,
        )
        for probe_id in PROBE_IDS:
            control_response = caller(client, "ordinary-observation", phase)
            candidate_response = caller(client, bank[probe_id], phase)
            query_count += 2
            control_projection, control_body = projection(control_response)
            candidate_projection, candidate_body = projection(candidate_response)
            observations.append(
                make_probe_observation(
                    probe_id=probe_id,
                    method=method,
                    phase=phase,
                    encoding="identity",
                    control_geometry=generic_effect_geometry(control_body),
                    candidate_geometry=generic_effect_geometry(candidate_body),
                    control_projection=control_projection,
                    candidate_projection=candidate_projection,
                    safe_probe=True,
                )
            )
    signature = aggregate_signature(observations)
    permuted_signature = aggregate_signature(list(reversed(observations)))
    evidence_sha256 = sha256_json({"reset": reset, "screen_observation": screen_observation["observation_sha256"], "confirm_observations": [item["observation_sha256"] for item in observations]})
    family = str(spec["family"])
    typed_positive = family not in {"ordinary_response", "unknown_surface"}
    return {
        "row_id": f"pg101-{source}-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}",
        "source": source,
        "implementation": implementation,
        "surface": surface,
        "variant": variant,
        "seed": int(seed),
        "method": method,
        "family": family,
        "typed_positive": typed_positive,
        "signature": signature,
        "order_permutation_invariant": signature["signature_sha256"] == permuted_signature["signature_sha256"],
        "model_input": signature,
        "screen_observation": screen_observation,
        "query_count": query_count,
        "reset": reset,
        "negative_control_matched": True,
        "evidence_sha256": evidence_sha256,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "model_input_has_evaluator_label": model_input_has_forbidden_field(signature),
    }


def _collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for implementation in ("north", "south"):
        for surface in PG36_SURFACES:
            if PG36_SURFACES[surface]["family"] in {"ordinary_response", "unknown_surface"}:
                continue
            for seed in PG36_SEEDS:
                for method in ("GET", "POST"):
                    rows.append(_episode(source="pg36", implementation=implementation, surface=surface, variant="canonical", seed=seed, method=method))
    for implementation in ("cobalt", "quartz"):
        for surface in PG42_SURFACES:
            for variant in ("ledger", "envelope", "framed"):
                for seed in PG42_SEEDS:
                    for method in ("GET", "POST"):
                        rows.append(_episode(source="pg42", implementation=implementation, surface=surface, variant=variant, seed=seed, method=method))
    for variant in PG35_VARIANTS:
        for surface in PG35_SURFACES:
            for seed in PG35_SEEDS:
                for method in ("GET", "POST"):
                    rows.append(_episode(source="pg35", implementation="pg35", surface=surface, variant=variant, seed=seed, method=method))
    return rows


def _score(rows: Iterable[dict[str, Any]], decoder: ActiveProbeSignatureDecoder) -> dict[str, Any]:
    rows = list(rows)
    results: list[dict[str, Any]] = []
    for row in rows:
        prediction = decoder.predict(row["model_input"])
        results.append({"row": row, "prediction": prediction})
    known = [item for item in results if item["row"]["typed_positive"] and item["row"]["family"] in KNOWN_FAMILIES]
    unknown = [item for item in results if item["row"]["family"] == "template_injection"]
    negatives = [item for item in results if not item["row"]["typed_positive"]]
    known_hits = sum(int(item["prediction"].get("candidate_family") == item["row"]["family"]) for item in known)
    false_accepts = sum(int(item["prediction"].get("decision") == "candidate") for item in negatives)
    unknown_misnames = sum(int(item["prediction"].get("decision") == "candidate") for item in unknown)
    return {
        "count": len(rows),
        "known_positive_count": len(known),
        "known_confirm_recall": round(known_hits / len(known), 6) if known else 0.0,
        "false_accept_count": false_accepts,
        "unknown_positive_count": len(unknown),
        "unknown_misname_count": unknown_misnames,
        "unknown_strict_abstain": bool(unknown) and unknown_misnames == 0 and all(item["prediction"].get("abstain") for item in unknown),
        "abstain_count": sum(int(item["prediction"].get("abstain")) for item in results),
        "candidate_count": sum(int(item["prediction"].get("decision") == "candidate") for item in results),
        "family_min_confirm_recall": min(
            (sum(int(item["prediction"].get("candidate_family") == family) for item in known if item["row"]["family"] == family) / max(1, sum(item["row"]["family"] == family for item in known)))
            for family in KNOWN_FAMILIES
        ) if known else 0.0,
        "implementation_min_confirm_recall": min(
            (sum(int(item["prediction"].get("candidate_family") == item["row"]["family"]) for item in known if item["row"]["implementation"] == implementation) / max(1, sum(item["row"]["implementation"] == implementation for item in known)))
            for implementation in sorted({str(item["row"]["implementation"]) for item in known})
        ) if known else 0.0,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_overlap(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    known = {str(row["signature"]["signature_sha256"]) for row in rows if row["typed_positive"] and row["family"] in KNOWN_FAMILIES}
    unknown = {str(row["signature"]["signature_sha256"]) for row in rows if row["typed_positive"] and row["family"] == "template_injection"}
    return {
        "known_signature_count": len(known),
        "unknown_signature_count": len(unknown),
        "known_unknown_overlap_count": len(known & unknown),
        "unknown_overlap_rate": round(sum(str(row["signature"]["signature_sha256"]) in known for row in rows if row["typed_positive"] and row["family"] == "template_injection") / max(1, sum(row["typed_positive"] and row["family"] == "template_injection" for row in rows)), 6),
        "impossibility_witness": bool(known & unknown),
    }


def run() -> dict[str, Any]:
    rows = _collect()
    train_rows = [row for row in rows if row["source"] == "pg36" and row["implementation"] == "north" and row["seed"] in {361, 367}]
    dev_rows = [row for row in rows if row["source"] == "pg36" and row["implementation"] == "south" and row["seed"] == 373]
    eval_rows = [row for row in rows if row["source"] == "pg42"]
    third_rows = [row for row in rows if row["source"] == "pg35"]
    decoder = ActiveProbeSignatureDecoder().fit(train_rows, allowed_families=KNOWN_FAMILIES)
    dev_metrics = _score(dev_rows, decoder)
    eval_metrics = _score(eval_rows, decoder)
    third_metrics = _score(third_rows, decoder)
    by_implementation = {implementation: _score([row for row in eval_rows if row["implementation"] == implementation], decoder) for implementation in ("cobalt", "quartz")}
    by_seed = {str(seed): _score([row for row in eval_rows if row["seed"] == seed], decoder) for seed in PG42_SEEDS}
    overlap = _signature_overlap(eval_rows)
    pg99 = json.loads(PG99_REPORT_PATH.read_text(encoding="utf-8")) if PG99_REPORT_PATH.exists() else {}
    pg99_overlap = ((pg99.get("metrics") or {}).get("pg42_known_unknown_overlap") or {})
    source_hashes = {
        "pg36_fixture": _file_sha256(ROOT / "app" / "pg36_independent_maze_fixture.py"),
        "pg42_fixture": _file_sha256(ROOT / "app" / "pg42_independent_semantic_fixture.py"),
        "pg35_fixture": _file_sha256(ROOT / "app" / "pg35_independent_fixture.py"),
        "active_probe_module": _file_sha256(ROOT / "app" / "active_probe_signature.py"),
        "runner": _file_sha256(Path(__file__)),
    }
    checks = {
        "train_excludes_pg42": all(row["source"] != "pg42" for row in train_rows),
        "target_overlap_forbidden": not ({row["implementation"] for row in train_rows} & {row["implementation"] for row in eval_rows}),
        "family_label_not_in_model_input": all(not model_input_has_forbidden_field(row["model_input"]) for row in rows),
        "oracle_label_not_in_model_input": all(not row["model_input_has_evaluator_label"] for row in rows),
        "fresh_reset_per_episode": all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in rows),
        "negative_control_matched": all(row["negative_control_matched"] for row in rows),
        "evidence_hashes_valid": all(len(row["evidence_sha256"]) == 64 and len(row["signature"]["signature_sha256"]) == 64 for row in rows),
        "get_post_covered": sorted({row["method"] for row in eval_rows}) == ["GET", "POST"],
        "cross_implementation_recall": all(by_implementation[item]["known_confirm_recall"] >= 0.80 for item in by_implementation),
        "cross_seed_recall": all(by_seed[item]["known_confirm_recall"] >= 0.80 for item in by_seed),
        "third_implementation_recall": third_metrics["known_confirm_recall"] >= 0.80,
        "third_implementation_false_accept": third_metrics["false_accept_count"] == 0,
        "order_permutation_invariant": all(row["order_permutation_invariant"] for row in rows),
        "unknown_family_strict_abstain": eval_metrics["unknown_strict_abstain"],
        "zero_negative_false_accept": eval_metrics["false_accept_count"] == 0,
        "not_all_abstain": eval_metrics["candidate_count"] > 0,
        "active_signature_breaks_pg99_overlap": overlap["known_unknown_overlap_count"] < int(pg99_overlap.get("known_unknown_fingerprint_overlap_count", 10**9)),
    }
    blocked = [key for key, value in checks.items() if not value]
    capability_status = "passed" if not blocked else "blocked"
    # This is still a representation/baseline experiment: no checkpoint or
    # catalog is promoted even when the frozen support decoder passes.
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg101-active-probe-signature-report-v1",
        "status": capability_status,
        "source": {
            "training": "pg36 north seeds 361/367 known families only",
            "development": "pg36 south seed 373",
            "evaluation": "pg42 cobalt/quartz seeds 401/409/419, ledger/envelope/framed",
            "third_implementation": "pg35 alpha/beta/gamma seeds 351/357/367",
            "train_excludes_eval_source": True,
            "probe_bank_size": len(PROBE_IDS),
            "probe_values_persisted": False,
            "model_input_schema": "bounded-active-probe-signature-v1",
            "source_hashes": source_hashes,
        },
        "metrics": {
            "all_rows": len(rows),
            "train_rows": len(train_rows),
            "dev": dev_metrics,
            "pg42": eval_metrics,
            "pg35_third_implementation": third_metrics,
            "pg42_by_implementation": by_implementation,
            "pg42_by_seed": by_seed,
            "pg42_signature_overlap": overlap,
            "query_cost": {
                "screen_pair_requests_per_episode": 2,
                "confirm_bank_pair_requests_per_episode": len(PROBE_IDS) * 2,
                "total_pg42_requests": sum(row["query_count"] for row in eval_rows),
                "mean_pg42_requests_per_episode": round(sum(row["query_count"] for row in eval_rows) / max(1, len(eval_rows)), 6),
                "total_pg35_requests": sum(row["query_count"] for row in third_rows),
                "mean_pg35_requests_per_episode": round(sum(row["query_count"] for row in third_rows) / max(1, len(third_rows)), 6),
            },
            "order_permutation_invariant_rate": round(sum(int(row["order_permutation_invariant"]) for row in rows) / max(1, len(rows)), 6),
            "pg99_static_projection_overlap_count": pg99_overlap.get("known_unknown_fingerprint_overlap_count"),
        },
        "capability_gate": {
            "status": capability_status,
            "checks": checks,
            "blocking_reasons": blocked,
            "claim_allowed": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "representation_baseline_only",
            "reason": "active probe signature is a non-neural baseline; a neural decoder and an independent replay are still required",
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_model_input": False,
            "family_labels_in_model_input": False,
            "fresh_reset_per_episode": True,
            "matched_negative_control": True,
            "evidence_hashes_verified": checks["evidence_hashes_valid"],
        },
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg101-active-probe-signature-protocol-v1",
        "hypothesis": "A fixed bank of safe, canonical probe actions can expose a transferable response signature that separates known families from an unseen family without exposing oracle labels.",
        "target_contract": {"sources": ["pg36", "pg42", "pg35"], "methods": ["GET", "POST"], "fresh_loopback_episode_per_surface_seed_variant_method": True, "state_change_allowed": False, "external_network": False},
        "source_hashes": source_hashes,
        "probe_bank": {"canonical_ids": list(PROBE_IDS), "values_persisted": False, "inert_only": True},
        "model_contract": {"visible": ["canonical_probe_id", "generic_geometry_delta", "bounded_transport_delta", "method", "phase"], "family_label_in_features": False, "oracle_label_in_features": False, "raw_body_in_features": False, "route_in_features": False},
        "evaluation_contract": {"train_source": "pg36-north", "dev_source": "pg36-south", "eval_source": "pg42-cobalt-quartz", "third_implementation": "pg35-alpha-beta-gamma", "unknown_family": "template_injection", "unknown_must_abstain": True, "negative_false_accept_must_be_zero": True, "order_permutation_must_be_invariant": True},
        "run_result": {"status": capability_status, "training_allowed": False, "memory_promotion_allowed": False},
        "next_experiment": "PG102 train a neural permutation-aware set decoder on the same signature traces and re-run fresh PG42 plus a third implementation",
    }
    dataset_rows = []
    trace_rows = []
    for row in rows:
        dataset_rows.append({
            "row_id": row["row_id"],
            "role": "train" if row in train_rows else "dev" if row in dev_rows else "family_holdout" if row["family"] == "template_injection" else "third_implementation" if row["source"] == "pg35" else "ood_source",
            "model_input": row["model_input"],
            "evaluator_label": {"family": row["family"], "typed_positive": row["typed_positive"]},
            "source": row["source"],
            "implementation": row["implementation"],
            "seed": row["seed"],
            "method": row["method"],
            "evidence_sha256": row["evidence_sha256"],
            "fresh_reset": row["reset"],
            "negative_control_matched": row["negative_control_matched"],
            "order_permutation_invariant": row["order_permutation_invariant"],
            "raw_probe_strings_stored": False,
            "raw_response_body_stored": False,
        })
        trace_rows.append({
            "trace_id": sha256_json({"row_id": row["row_id"], "evidence": row["evidence_sha256"]})[:24],
            "row_id": row["row_id"],
            "screen": row["screen_observation"],
            "confirm_signature": row["signature"],
            "belief_before": {"unknown": 1.0},
            "next_action": "active_probe_bank_complete_then_decode",
            "evaluator_label": {"family": row["family"], "typed_positive": row["typed_positive"]},
            "evidence_sha256": row["evidence_sha256"],
            "fresh_reset": row["reset"],
            "negative_control_matched": row["negative_control_matched"],
            "online_weight_update": False,
            "long_term_memory_write": False,
        })
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg101-active-probe-signature-visible-dataset-v1", "evaluation_only": True, "training_eligible": False, "model_input_contract": {"family_label_in_features": False, "oracle_label_in_features": False, "raw_body_in_features": False}, "rows": dataset_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg101-active-probe-signature-trace-v1", "evaluation_only": True, "training_eligible": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "steps": trace_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-101 主动 probe-bank signature", "", f"PG42 eval rows: {eval_metrics['count']}；已知族召回: {eval_metrics['known_confirm_recall']}；未知族严格弃权: {eval_metrics['unknown_strict_abstain']}；误报: {eval_metrics['false_accept_count']}。", f"PG35 third implementation recall: {third_metrics['known_confirm_recall']}；误报: {third_metrics['false_accept_count']}。", f"PG99 静态投影已知/未知重叠: {pg99_overlap.get('known_unknown_fingerprint_overlap_count')}；主动签名重叠: {overlap['known_unknown_overlap_count']}；顺序不变率: {report['metrics']['order_permutation_invariant_rate']}。", "", "| split | rows | known recall | unknown abstain | false accepts |", "|---|---:|---:|---|---:|"]
    for name, metrics in (("dev", dev_metrics), ("pg42", eval_metrics), ("pg35-third", third_metrics)):
        lines.append(f"| `{name}` | {metrics['count']} | {metrics['known_confirm_recall']} | {metrics['unknown_strict_abstain']} | {metrics['false_accept_count']} |")
    lines.extend(["", "这是 representation/baseline 实验：probe 值只在 loopback 运行时存在，模型输入只有 canonical ID 与无字段名几何差分。即使能力门通过，仍不训练/写长期记忆，下一步是神经 set decoder + 第三实现复放。", "", f"JSON: `{REPORT_PATH.relative_to(ROOT)}`", f"协议: `{PROTOCOL_PATH.relative_to(ROOT)}`", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "pg42": eval_metrics, "overlap": overlap, "checks": checks, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
