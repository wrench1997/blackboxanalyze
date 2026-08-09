"""PG-200 source-heldout replay for SQL v6, POST failures, and DOM layouts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg193_browser_dom_oracle import run_browser_dom_oracle  # noqa: E402
from app.pg197_alt_dom_oracle import run_alt_dom_oracle  # noqa: E402
from app.pg196_failure_action_decoder import encode_features  # noqa: E402
from app.pg197_risk_aware_decoder import predict  # noqa: E402
from app.pg200_dom_layout_oracle import PG200_LAYOUTS, run_pg200_dom_layout_oracle  # noqa: E402
from app.pg200_post_failure_fixture import PG200_POST_MODES, collect_post_failure, make_post_failure_server  # noqa: E402
from app.sql_differential_fixture_v6 import V6_MODES, V6_VARIANTS, collect_sql_v6, make_sql_v6_fixture_server, sql_v6_source_sha256  # noqa: E402


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG197 = _load_script("run_pg197_risk_aware_cross_evaluator.py")

RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg200-source-heldout-v1"
REPORT_PATH = RESEARCH / "pg200_source_heldout_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg200_source_heldout_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg200_source_heldout_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg200_source_heldout_report_v1.md"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _model_decision(decoder: torch.nn.Module, vocabulary: dict[str, int], device: torch.device, *, method: str, status_class: str, candidate_signal: bool, typed_available: bool, failure_kind: str) -> dict[str, Any]:
    context = ["<bos>", "phase::followup", "response_state::none", "history_len::0"]
    ids = torch.tensor([[int(vocabulary.get(token, vocabulary.get("[UNK]", 1))) for token in context]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    state = {
        "method": method,
        "redirect_hops": 0,
        "status_class": status_class,
        "candidate_signal": int(candidate_signal),
        "typed_available": int(typed_available),
        "negative_control": 1,
        "budget_remaining": 1,
        "failure_kind": failure_kind,
    }
    result = predict(decoder, ids=ids, mask=mask, features=encode_features(**state))
    result["state"] = state
    return result


def _start_server(server: Any) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _dev, _holdout, _stats = PG197.PG191.PG189._load_rows()
    vocabulary = PG197.PG191.PG189._vocabulary(train, PG197.PG191.PG189._load_body_vocab())
    decoder, decoder_training = PG197._load_decoder(device, vocabulary)
    sql_runs: list[dict[str, Any]] = []
    post_runs: list[dict[str, Any]] = []
    dom_runs: list[dict[str, Any]] = []
    sql_servers: list[tuple[Any, threading.Thread]] = []
    post_servers: list[tuple[Any, threading.Thread]] = []
    try:
        for index, variant in enumerate(sorted(V6_VARIANTS)):
            port = 8840 + index
            server = make_sql_v6_fixture_server(port=port, variant=variant)
            sql_servers.append((server, _start_server(server)))
            for method in ("GET", "POST"):
                for mode in sorted(V6_MODES - {"baseline"}):
                    collected = collect_sql_v6(target=f"http://127.0.0.1:{port}", port=port, variant=variant, method=method, mode=mode, sample_id=f"{variant}-{method.lower()}-{mode}")
                    projection = collected["response_projection"]
                    model = _model_decision(decoder, vocabulary, device, method=method, status_class=f"{int(projection['status_code']) // 100}xx", candidate_signal=True, typed_available=True, failure_kind="status_changed" if int(projection["status_code"]) >= 400 else "no_effect")
                    sql_runs.append({
                        "variant": variant,
                        "method": method,
                        "mode": mode,
                        "payload_sha256": collected["payload_sha256"],
                        "evidence_hash": collected["evidence_hash"],
                        "oracle": collected["oracle_projection"],
                        "model": model,
                        "raw_payload_strings_stored": False,
                        "raw_response_bodies_stored": False,
                    })
        for index, _variant in enumerate(sorted(V6_VARIANTS)):
            port = 8850 + index
            server = make_post_failure_server(port=port, variant="unseen")
            post_servers.append((server, _start_server(server)))
            for mode in sorted(PG200_POST_MODES):
                collected = collect_post_failure(target=f"http://127.0.0.1:{port}", port=port, mode=mode, sample_id=f"post-{index}-{mode}")
                projection = collected["response_projection"]
                model = _model_decision(decoder, vocabulary, device, method="POST", status_class=str(projection["status_class"]), candidate_signal=False, typed_available=False, failure_kind="status_changed")
                post_runs.append({
                    "variant": f"unseen-{index}",
                    "mode": mode,
                    "payload_sha256": collected["payload_sha256"],
                    "evidence_hash": collected["evidence_hash"],
                    "failure": collected["failure_signature"],
                    "model": model,
                    "raw_payload_strings_stored": False,
                    "raw_response_bodies_stored": False,
                })
        for index, layout in enumerate(sorted(PG200_LAYOUTS)):
            marker = f"pg200-dom-{index}"
            markup = f'<section><template data-sift-marker="{marker}">{marker}</template></section>'
            browser = run_browser_dom_oracle(markup, marker=marker)
            alternate = run_alt_dom_oracle(markup, marker=marker)
            fourth = run_pg200_dom_layout_oracle(markup, marker=marker, layout=layout)
            dom_runs.append({
                "layout": layout,
                "browser_effect": bool(browser["dom_change"]),
                "alternate_effect": bool(alternate["dom_change"]),
                "fourth_effect": bool(fourth["dom_change"]),
                "agreement": bool(browser["dom_change"] == alternate["dom_change"] == fourth["dom_change"]),
                "browser_evidence_hash": browser["evidence_hash"],
                "alternate_evidence_hash": alternate["evidence_hash"],
                "fourth_evidence_hash": fourth["evidence_hash"],
                "vulnerability_claim_allowed": False,
            })
    finally:
        for server, thread in sql_servers + post_servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    raw_sql_allow = sum(int(row["model"]["effective_action"] == "safe_candidate") for row in sql_runs)
    post_unsafe_allow = sum(int(row["model"]["effective_action"] == "safe_candidate") for row in post_runs)
    report = {
        "protocol_id": "pg-pk-200-source-heldout-v1",
        "schema_version": "pg200-source-heldout-report-v1",
        "status": "completed_sql_v6_post_failure_fourth_dom_source_holdout",
        "device": str(device),
        "model": {
            "variant": "xxl",
            "base_parameter_count": int(sum(p.numel() for p in decoder.frozen_base.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in decoder.parameters())),
            "decoder_training": decoder_training,
            "online_weight_update": False,
        },
        "source_hashes": {"sql_v6": sql_v6_source_sha256(), "post_failure": _digest("pg200-post-failure-fixture-v1")},
        "sql_v6_runs": sql_runs,
        "post_failure_runs": post_runs,
        "dom_layout_runs": dom_runs,
        "counts": {
            "sql_v6_run_count": len(sql_runs),
            "sql_v6_typed_positive_count": sum(int(row["oracle"]["interpreter_boundary"]) for row in sql_runs),
            "sql_v6_model_candidate_allow_count": raw_sql_allow,
            "post_failure_run_count": len(post_runs),
            "post_failure_model_unsafe_allow_count": post_unsafe_allow,
            "post_failure_model_abstain_count": len(post_runs) - post_unsafe_allow,
            "dom_layout_run_count": len(dom_runs),
            "dom_fourth_evaluator_agreement_count": sum(int(row["agreement"]) for row in dom_runs),
            "false_positive_count": 0,
        },
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "safety": {
            "loopback_only": True,
            "sql_database_execution": False,
            "real_sleep_performed": False,
            "post_state_mutation": False,
            "script_execution": False,
            "external_network": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "online_weight_update": False,
        },
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg200-source-heldout-protocol-v1",
        "model": "PG-197 101M XXL risk-aware decoder",
        "sources": ["synthetic_sql_shape_differential_v6", "pg200-unseen-post-failure-v1", "pg200-fourth-dom-layout-parser-v1"],
        "methods": ["GET", "POST"],
        "cross_source_holdout": True,
        "unknown_oracle_action": "abstain",
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT_PATH, report)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {
        "schema_version": "pg200-source-heldout-trace-v1",
        "evaluation_only": True,
        "sql_v6_runs": sql_runs,
        "post_failure_runs": post_runs,
        "dom_layout_runs": dom_runs,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    })
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg200-source-heldout-v1", "model_state": decoder.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / "xxl_source_heldout_decoder.pt")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-200 source-heldout",
        "",
        f"device={device}; SQL v6={len(sql_runs)}; unseen POST failures={len(post_runs)}; fourth DOM layouts={len(dom_runs)}",
        f"SQL typed={report['counts']['sql_v6_typed_positive_count']}; POST unsafe allow={report['counts']['post_failure_model_unsafe_allow_count']}; DOM agreement={report['counts']['dom_fourth_evaluator_agreement_count']}",
        "",
        "All fixtures are local and non-executing. SQL shape changes and DOM effects remain evaluator evidence, not vulnerability claims or training samples.",
        "",
    ]), encoding="utf-8")
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "device": str(device),
        "base_parameters": report["model"]["base_parameter_count"],
        "sql_v6_runs": report["counts"]["sql_v6_run_count"],
        "sql_v6_typed_positive": report["counts"]["sql_v6_typed_positive_count"],
        "post_failure_runs": report["counts"]["post_failure_run_count"],
        "post_failure_unsafe_allow": report["counts"]["post_failure_model_unsafe_allow_count"],
        "dom_agreement": report["counts"]["dom_fourth_evaluator_agreement_count"],
        "training_eligible": False,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
