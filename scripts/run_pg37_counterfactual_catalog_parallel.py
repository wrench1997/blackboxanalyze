"""Bounded-parallel PG-37 collector.

Parallelism changes throughput only: each individual control and candidate
still receives a newly created loopback server and a fresh reset proof.  A
fixed worker/port budget prevents target reuse or unbounded fan-out.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, build_catalog, sha256_json  # noqa: E402
from app.pg37_counterfactual_fixture import LAYOUTS, PHASES, SURFACE_SPECS, VARIANTS  # noqa: E402
from app.trace_aligned_dataset import evaluate_episode, sha256_json as trace_sha256_json  # noqa: E402
try:  # Running as ``python scripts/foo.py`` puts scripts/ on sys.path.
    from scripts.run_pg37_counterfactual_catalog import (  # type: ignore[no-redef]  # noqa: E402
        ORACLE_CONTRACT_SHA256,
        ROLE_BY_SURFACE,
        ROLE_FAMILIES,
        SEEDS,
        TARGET_PORT,
        _FreshTarget,
        _call,
        _manifest,
        _oracle,
        _reset,
        _response_projection,
        _rule_ir,
        _source,
        _trace_step,
    )
except ModuleNotFoundError:
    from run_pg37_counterfactual_catalog import (  # type: ignore[no-redef]  # noqa: E402
        ORACLE_CONTRACT_SHA256,
        ROLE_BY_SURFACE,
        ROLE_FAMILIES,
        SEEDS,
        TARGET_PORT,
        _FreshTarget,
        _call,
        _manifest,
        _oracle,
        _reset,
        _response_projection,
        _rule_ir,
        _source,
        _trace_step,
    )


CATALOG_OUTPUT = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
TRACE_OUTPUT = ROOT / "research" / "pg37_counterfactual_trace_v1.json"
REGISTRY_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
MAX_WORKERS = 8
_WORKER_COUNTER = 0
_WORKER_COUNTER_LOCK = threading.Lock()
_WORKER_LOCAL = threading.local()


def _worker_port() -> int:
    """Assign exactly one deterministic loopback port to each executor worker."""
    global _WORKER_COUNTER
    port = getattr(_WORKER_LOCAL, "port", None)
    if port is None:
        with _WORKER_COUNTER_LOCK:
            port = TARGET_PORT + _WORKER_COUNTER
            _WORKER_COUNTER += 1
        _WORKER_LOCAL.port = port
    return int(port)


def _episode(task: tuple[str, str, str, int, int, dict[str, Any], str, str]) -> dict[str, Any]:
    implementation, surface, variant, seed, _port_hint, source, collector_hash, fixture_hash = task
    port = _worker_port()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector = ReadOnlySafeCatalogCollector(source, registry=registry)
    spec = SURFACE_SPECS[surface]
    role = ROLE_BY_SURFACE[surface]
    episode_id = f"pg37-episode-{implementation}-{surface}-{variant}-s{seed}"
    records: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    previous: str | None = None
    belief = {"unknown": 1.0}
    for method in ("GET", "POST"):
        for phase in PHASES:
            with _FreshTarget(implementation, port=port) as client:
                control_response = _call(client, implementation, surface, variant, method, phase, positive=False)
            with _FreshTarget(implementation, port=port) as client:
                candidate_response = _call(client, implementation, surface, variant, method, phase, positive=role != "negative_control")
            baseline, _ = _response_projection(control_response)
            control_projection, control_parsed = _response_projection(control_response)
            candidate_projection, candidate_parsed = _response_projection(candidate_response)
            control_oracle = _oracle(implementation, surface, variant, method, phase, control_parsed)
            candidate_oracle = _oracle(implementation, surface, variant, method, phase, candidate_parsed)
            candidate_positive = bool(candidate_oracle["positive"])
            control_id = f"pg37-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-control"
            candidate_id = f"pg37-{implementation}-{surface}-{variant}-s{seed}-{method.casefold()}-{phase}-candidate"
            control = collector.collect(sample_id=control_id, sample_role="negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, variant, seed, method, phase, role, "control", baseline), payload_manifest=_manifest(implementation, surface, variant, seed, method, phase, role, "control"), response_projection=control_projection, oracle_projection=control_oracle, rule_ir=_rule_ir(surface, variant, method, phase))
            candidate = collector.collect(sample_id=candidate_id, sample_role="candidate" if candidate_positive else "negative_control", sampling_seed=seed, reset=_reset(collector.source, implementation, surface, variant, seed, method, phase, role, "candidate", baseline), payload_manifest=_manifest(implementation, surface, variant, seed, method, phase, role, "candidate"), response_projection=candidate_projection, oracle_projection=candidate_oracle, rule_ir=_rule_ir(surface, variant, method, phase), negative_control=({"control_sample_id": control["sample_id"], "control_evidence_hash": control["evidence"]["evidence_hash"], "intervention": "typed-class-vs-normal-control", "verdict": "confirmed_negative", "same_source": True, "same_surface": True, "same_variant": True} if candidate_positive else None))
            for row, sample_role, pair_role in ((control, "negative_control", "control"), (candidate, "candidate" if candidate_positive else "negative_control", "candidate")):
                row.update({"dataset_role": role, "implementation": implementation, "surface_id": surface, "surface_variant": variant, "family": spec["family"], "method": method, "phase": phase, "sample_role": sample_role, "pair_role": pair_role})
                records.append(row)
            for row, pair_id, action in ((control, None, "replay_candidate"), (candidate, control["sample_id"] if candidate_positive else None, "confirm_same_surface" if phase == "screen" else "stop_episode" if candidate_positive else "next_probe")):
                step = _trace_step(row, episode_id=episode_id, step_id=f"{episode_id}-{method.casefold()}-{phase}-{row['pair_role']}", parent_step_id=previous, next_action=action, negative_control_pair_id=pair_id, belief_before=belief)
                previous = step["step_id"]
                belief = step["belief_after"]
                steps.append(step)
    return {"implementation": implementation, "records": records, "steps": steps, "episode": evaluate_episode(steps)}


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256((ROOT / "app" / "pg37_counterfactual_fixture.py").read_bytes()).hexdigest()
    raw_sources = {implementation: _source(implementation, collector_hash, fixture_hash) for implementation in LAYOUTS}
    tasks: list[tuple[str, str, str, int, int, dict[str, Any], str, str]] = []
    index = 0
    for implementation in LAYOUTS:
        for surface in SURFACE_SPECS:
            for variant in VARIANTS:
                for seed in SEEDS:
                    tasks.append((implementation, surface, variant, seed, 0, raw_sources[implementation], collector_hash, fixture_hash))
                    index += 1
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="pg37") as executor:
        futures = [executor.submit(_episode, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["implementation"], item["records"][0]["sample_id"]))
    all_records = [row for result in results for row in result["records"]]
    trace_steps = [step for result in results for step in result["steps"]]
    episode_reports = [result["episode"] for result in results]
    sources: list[dict[str, Any]] = []
    source_catalogs: list[dict[str, Any]] = []
    for implementation in LAYOUTS:
        source = ReadOnlySafeCatalogCollector(raw_sources[implementation], registry=registry).source
        sources.append(source)
        implementation_records = [row for row in all_records if row["implementation"] == implementation]
        variant_catalog = build_catalog(f"pg37-counterfactual-{implementation}-catalog", source, implementation_records)
        source_catalogs.append({"source_id": source["source_id"], "source_sha256": source["source_sha256"], "catalog_sha256": variant_catalog["catalog_sha256"], "sample_count": len(implementation_records), "training_eligible": variant_catalog["training_eligible"]})
    dataset_tests: list[dict[str, Any]] = []
    for role, families in ROLE_FAMILIES.items():
        for seed in SEEDS:
            rows = [row for row in all_records if row["dataset_role"] == role and int(row["sampling_seed"]) == seed]
            sample_ids = sorted(row["sample_id"] for row in rows)
            targets = sorted({row["target_instance_id"] for row in rows})
            source_hashes = sorted({row["source_sha256"] for row in rows})
            summary = {"sample_id": f"pg37-test-{role}-s{seed}", "dataset_id": f"pg37-{role}-s{seed}-v1", "source_id": f"pg37-role-source-{role}-s{seed}", "source_hash": sha256_json(source_hashes), "target_instance_ids": targets, "family_set": families, "sampling_seed": seed, "role": role, "sample_count": len(rows), "unique_sample_count": len(sample_ids), "denominator": len(rows), "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in rows), "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in rows), "abstain_count": 0, "method_set": ["GET", "POST"], "phase_set": list(PHASES), "surface_variant_set": list(VARIANTS), "source_count": len(source_hashes), "dataset_manifest_sha256": sha256_json({"role": role, "seed": seed, "samples": sample_ids}), "split_manifest_sha256": sha256_json({"role": role, "seed": seed, "targets": targets, "families": families}), "probe_sha256": sha256_json([row["payload_manifest"]["payload_sha256"] for row in rows]), "oracle_contract_sha256": ORACLE_CONTRACT_SHA256, "checkpoint_sha256": sha256_json("pending-model-run"), "metrics_status": "pending_model_run", "metrics": {"typed_recall": 0.0, "precision": 0.0, "false_positive_rate": 0.0, "abstain_precision": 0.0, "ece": 0.0, "median_queries": 0.0}}
            summary["evidence_hash"] = sha256_json(summary)
            dataset_tests.append(summary)
    episode_reports.sort(key=lambda item: item["episode_id"])
    trace_steps.sort(key=lambda item: (item["episode_id"], item["step_id"]))
    catalog = {"schema_version": "pg-pk-37-counterfactual-catalog-v1", "catalog_id": "pg37-counterfactual-v1", "purpose": "same-family multi-surface counterfactual GET/POST typed replay", "runtime_replay": True, "independent_target_implementation": True, "evaluation_only": True, "training_eligible": True, "training_artifact_generated": False, "model_evaluation_completed": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": list(VARIANTS), "seeds": list(SEEDS), "implementations": list(LAYOUTS), "sources": sources, "source_catalogs": source_catalogs, "samples": all_records, "dataset_tests": dataset_tests, "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "trace_episode_count": len(episode_reports), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "typed_positive_count": sum(int(row["oracle_projection"]["positive"]) for row in all_records), "negative_control_count": sum(int(not row["oracle_projection"]["positive"]) for row in all_records), "counterfactual_pair_count": len({(row["implementation"], row["surface_id"], row["sampling_seed"], row["method"], row["phase"]) for row in all_records}), "fresh_reset_count": len(all_records), "source_count": len({row["source_sha256"] for row in all_records}), "target_instance_count": len({row["target_instance_id"] for row in all_records}), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "external_network": False, "authorization": "workspace_local_only", "family_policy": ROLE_FAMILIES, "manifest_sha256": sha256_json({"samples": [row["evidence"]["evidence_hash"] for row in all_records], "dataset_tests": dataset_tests})}
    trace_dataset = {"schema_version": "pg-pk-37-counterfactual-trace-v1", "purpose": "counterfactual surface trace with typed oracle after probe", "evaluation_only": True, "training_eligible": False, "independent_target_implementation": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "methods": ["GET", "POST"], "phases": list(PHASES), "surface_variants": list(VARIANTS), "episode_count": len(episode_reports), "accepted_evaluation_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "episodes": episode_reports, "steps": trace_steps, "catalog_manifest_sha256": catalog["manifest_sha256"], "trace_manifest_sha256": trace_sha256_json([step["trace_sha256"] for step in trace_steps])}
    CATALOG_OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_OUTPUT.write_text(json.dumps(trace_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"catalog": str(CATALOG_OUTPUT.relative_to(ROOT)), "trace_dataset": str(TRACE_OUTPUT.relative_to(ROOT)), "sample_count": len(all_records), "typed_positive_count": catalog["typed_positive_count"], "negative_control_count": catalog["negative_control_count"], "counterfactual_pair_count": catalog["counterfactual_pair_count"], "source_count": catalog["source_count"], "episode_count": len(episode_reports), "accepted_evaluation_episodes": catalog["accepted_evaluation_episode_count"], "training_eligible": catalog["training_eligible"], "max_workers": MAX_WORKERS}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
