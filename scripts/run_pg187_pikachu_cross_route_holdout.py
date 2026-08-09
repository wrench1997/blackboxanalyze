"""PG-187: double holdout on unseen Pikachu GET routes.

PG-186's frozen model/encoding sweep is reused, but the routes are changed to
two observed-by-browser, unseen-by-PG-185 surfaces.  The run remains local,
read-only, inert-DOM-only, and evaluation-only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg186() -> Any:
    path = ROOT / "scripts" / "run_pg186_pikachu_dom_capacity_encoding.py"
    spec = importlib.util.spec_from_file_location("pg186_runner_for_pg187", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-186 runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG186 = _load_pg186()
PG185 = PG186.PG185
RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg187_pikachu_cross_route_holdout_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg187_pikachu_cross_route_holdout_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg187_pikachu_cross_route_holdout_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg187_pikachu_cross_route_holdout_report_v1.md"
ROUTE_SPECS = (
    ("/vul/xss/xss_01.php", "xss_01_unseen_get", ("message", "submit")),
    ("/vul/xss/xss_04.php", "xss_04_unseen_get", ("message", "submit")),
)
ENCODINGS = (PG186.ENCODING_PLANS[0], PG186.ENCODING_PLANS[2])


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_unseen_routes() -> list[dict[str, Any]]:
    crawl = json.loads((RESEARCH / "pg179_pikachu_browser_crawl_manifest_v1.json").read_text(encoding="utf-8"))
    rows = crawl.get("request_response_rows", [])
    routes: list[dict[str, Any]] = []
    for path, surface, fields in ROUTE_SPECS:
        matches = [row for row in rows if row.get("method") == "GET" and row.get("route_path") == path and sorted(row.get("request_schema", {}).get("query_params", [])) == sorted(fields)]
        if len(matches) != 1:
            raise ValueError(f"PG-187 expected one parameterized GET observation for {surface}, got {len(matches)}")
        routes.append({"path": path, "surface": surface, "field_names": list(fields), "crawl_row_sha256": _sha256_json(matches[0])})
    return routes


def main() -> int:
    routes = _load_unseen_routes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    container_id = PG185._start_container()
    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        episode_index = 0
        for model_name, checkpoint_path in PG186.CHECKPOINTS:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
            variant = str(checkpoint["variant"])
            model = PG186.build_model(len(vocabulary), variant).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            model_runs: list[dict[str, Any]] = []
            for route in routes:
                for encoding_name, encoding_chain in ENCODINGS:
                    if episode_index:
                        PG186._restart_and_wait()
                    target_hash = hashlib.sha256(f"{container_id}:{episode_index}:{model_name}:{route['surface']}:{encoding_name}".encode("utf-8")).hexdigest()
                    run = PG186._replay_episode(model, vocabulary, route, device, encoding_name=encoding_name, encoding_chain=encoding_chain, target_hash=target_hash)
                    run["model"] = model_name
                    run["checkpoint_sha256"] = checkpoint_hash
                    model_runs.append(run)
                    runs.append(run)
                    episode_index += 1
            summaries.append({"model": model_name, "variant": variant, "parameter_count": int(sum(p.numel() for p in model.parameters())), "checkpoint_sha256": checkpoint_hash, "episode_count": len(model_runs), "sent_count": sum(r["sent_count"] for r in model_runs), "candidate_sent_count": sum(r["candidate_sent_count"] for r in model_runs), "typed_surface_effect_count": sum(r["typed_surface_effect_count"] for r in model_runs), "controller_abstain_count": sum(r["controller_abstain_count"] for r in model_runs), "typed_positive_count": 0})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        PG185._stop_container()

    report = {
        "protocol_id": "pg-pk-187-pikachu-cross-route-holdout-v1",
        "schema_version": "pg187-pikachu-cross-route-holdout-report-v1",
        "status": "completed_unseen_route_double_holdout",
        "source": {"crawl_manifest": "research/pg179_pikachu_browser_crawl_manifest_v1.json", "routes_unseen_in_pg185": [surface for _, surface, _ in ROUTE_SPECS], "encoding_holdout": [name for name, _ in ENCODINGS], "image": PG185.IMAGE, "loopback_port": PG185.PORT, "fresh_restart_per_episode": True},
        "device": str(device),
        "model_summaries": summaries,
        "counts": {"episode_count": len(runs), "sent_count": sum(r["sent_count"] for r in runs), "candidate_sent_count": sum(r["candidate_sent_count"] for r in runs), "typed_surface_effect_count": sum(r["typed_surface_effect_count"] for r in runs), "typed_positive_count": 0, "controller_abstain_count": sum(r["controller_abstain_count"] for r in runs)},
        "runs": runs,
        "holdout": {"route_holdout": True, "encoding_holdout": True, "model_input_route_present": False, "model_input_family_present": False, "false_vulnerability_positive_count": 0, "unknown_oracle_abstain_required": True},
        "selection": {"selected_variant": None, "cross_route_capability_claim_allowed": False, "training_eligible": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "unseen route/encoding evaluation only; detached DOM effect is not script execution"},
        "safety": {"loopback_only": True, "external_network": False, "fresh_container": True, "inert_dom_markup_only": True, "script_execution": False, "database_write": False, "credentials": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg187-pikachu-cross-route-holdout-trace-v1", "evaluation_only": True, "training_eligible": False, "runs": runs, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False})
    protocol = {"protocol_id": "pg-pk-187-pikachu-cross-route-holdout-v1", "schema_version": "pg187-pikachu-cross-route-holdout-protocol-v1", "route_holdout": [surface for _, surface, _ in ROUTE_SPECS], "encoding_holdout": [name for name, _ in ENCODINGS], "frozen_checkpoints": [name for name, _ in PG186.CHECKPOINTS], "fresh_restart_per_episode": True, "model_input_excludes_route": True, "model_input_excludes_family": True, "manifest_validator_before_send": True, "typed_dom_effect_not_vulnerability": True, "gates": {"loopback_only": True, "inert_dom_markup_only": True, "training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-187 Pikachu unseen-route double holdout", "", f"models={len(summaries)}; episodes={len(runs)}; sent={report['counts']['sent_count']}; candidates={report['counts']['candidate_sent_count']}; typed_surface_effects={report['counts']['typed_surface_effect_count']}", "", "xss_01/xss_04 是浏览器清单中的真实 GET 参数表面，但未用于 PG-185 回放；同时留出多层编码，检查路由泛化和错误阳性。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "models": len(summaries), "episodes": len(runs), "sent_count": report["counts"]["sent_count"], "candidate_sent_count": report["counts"]["candidate_sent_count"], "typed_surface_effect_count": report["counts"]["typed_surface_effect_count"], "typed_positive_count": 0, "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
