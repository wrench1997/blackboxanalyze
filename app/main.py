from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .authorized_target_session import analyze_authorized_target, iter_authorized_origins
from .authorized_remote_docker import (
    RemoteDockerConfig,
    analyze_authorized_remote_target,
    iter_remote_scope,
    probe_authorized_remote_docker,
)
from .closure import analyze_closure
from .detection_payload import validate_detection_payload
from .dom_oracle import run_dom_oracle
from .maze_engine import DEFAULT_ARTIFACT_ROOT, latest_manifest, load_manifest, verify_ledger
from .maturity import evaluate_research_maturity, triage_scale_failure
from .model_capability_gate import evaluate_model_capability
from .pg282_evaluator_binding import bind_abstract_plan
from .pg284_evaluator_contract import evaluate_typed_replay
from .pg286_live_collection import collect_pg286_live_record
from .pg287_live_collection import collect_pg287_live_record
from .pg292_live import evaluate_pg292_live
from .logic_replay_oracle import PROBE_CLASSES, SURFACES, run_logic_replay_oracle
from .maze_labs import get_maze_lab, public_maze_labs
from .rule_ir import pretty, truthy_result
from .research_events import emit_event, list_events
from .research_ops import build_payload_review, build_research_ops_snapshot
from .scenarios import get_scenario, public_scenarios
from .sql_ast_oracle import FRAGMENT_CLASSES, run_sql_ast_oracle
from .search import build_histories_before, enumerate_envelopes, search_rules, suggest_query
from .store import SessionStore
from .trace_aligned_dataset import evaluate_episode, validate_trace_step

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="SIFT AI Research Lab", version="0.3.0")
store = SessionStore()


class CustomScenario(BaseModel):
    name: str = "自定义黑盒"
    description: str = "用户定义的 Rule IR 黑盒"
    fields: list[dict[str, Any]]
    hidden_rule: Optional[dict[str, Any]] = None
    mode: Literal["oracle", "manual"] = "oracle"
    stateful: bool = False


class CreateSessionRequest(BaseModel):
    scenario_id: Optional[str] = None
    custom: Optional[CustomScenario] = None


class ProbeRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    state_after: Optional[dict[str, Any]] = None
    episode_id: str = "default"
    step: Optional[int] = None
    goal: Optional[bool] = None
    terminal: Optional[bool] = None
    available_actions: Optional[list[dict[str, Any]]] = None
    history: Optional[list[dict[str, Any]]] = None


class ObserveRequest(ProbeRequest):
    output: bool
    source: str = "manual"


class ImportObservationsRequest(BaseModel):
    observations: list[ObserveRequest]


class SearchRequest(BaseModel):
    max_depth: int = Field(default=3, ge=1, le=5)
    beam_width: int = Field(default=120, ge=10, le=500)
    history_depth: int = Field(default=1, ge=0, le=3)


class ValidateRequest(BaseModel):
    rule: dict[str, Any]
    max_cases: int = Field(default=5000, ge=1, le=20000)


class ClosureRequest(BaseModel):
    max_cases: int = Field(default=5000, ge=10, le=50000)
    top_candidates: int = Field(default=48, ge=2, le=200)
    accuracy_tolerance: float = Field(default=0.001, ge=0.0, le=0.2)
    history_depth: int = Field(default=1, ge=0, le=5)
    coverage_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    goal_mode: Literal["observation_goal", "output_true", "either"] = "either"
    auto_search: bool = True
    max_depth: int = Field(default=3, ge=1, le=5)
    beam_width: int = Field(default=180, ge=10, le=800)


class ResearchEventRequest(BaseModel):
    actor: str = "frontend"
    tool: str
    phase: str
    status: str = "info"
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact: Optional[str] = None


class ResearchMaturityRequest(BaseModel):
    reproducible: bool = False
    independent_seeds: int = Field(default=0, ge=0)
    family_holdout_runs: int = Field(default=0, ge=0)
    ablation_supports_mechanism: bool = False
    preregistered_target_met: bool = False
    guardrails_passed: bool = False
    data_lineage_complete: bool = False


class ModelCapabilityRequest(BaseModel):
    evidence: dict[str, Any] = Field(default_factory=dict)
    policy: Optional[dict[str, Any]] = None


class TraceStepRequest(BaseModel):
    step: dict[str, Any] = Field(default_factory=dict)


class TraceEpisodeRequest(BaseModel):
    steps: list[dict[str, Any]] = Field(default_factory=list)


class ScaleFailureTriageRequest(BaseModel):
    failure_reproduces_at_small_scale: bool = False
    seed_or_split_sensitive: bool = False
    family_holdout_regression: bool = False
    metric_or_oracle_definition_changed: bool = False
    single_node_passes_but_distributed_fails: bool = False
    data_hash_or_lineage_mismatch: bool = False
    oom_timeout_or_checkpoint_failure: bool = False
    throughput_or_io_regression: bool = False
    nondeterministic_pipeline: bool = False


class DomOracleRequest(BaseModel):
    value: str = Field(default="", max_length=65536)
    sink: Literal["innerHTML", "template.innerHTML"] = "innerHTML"
    transforms: list[str] = Field(default_factory=list, max_length=4)
    marker: str = Field(default="sift-marker", min_length=1, max_length=64)


class SqlOracleRequest(BaseModel):
    fragment_class: str


class LogicReplayRequest(BaseModel):
    probe_class: str = "normal"
    surface: str = "authorization_boundary"


class DetectionPayloadRequest(BaseModel):
    target: str = "http://127.0.0.1:3100"
    method: Literal["GET", "HEAD", "OPTIONS", "POST"] = "GET"
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    form: dict[str, str] = Field(default_factory=dict)
    marker: str = "sift-probe"
    probe_kind: str = "http_canary"
    probe: Optional[str] = None
    expected: dict[str, Any] = Field(default_factory=dict)


class AuthorizedTargetAnalyzeRequest(BaseModel):
    """Operator-controlled Docker-local observation request.

    The explicit confirmation is intentionally separate from the URL.  It
    prevents a pasted URL from silently starting network traffic, while the
    server-side origin allowlist remains the actual scope boundary.
    """

    target_url: str = Field(min_length=1, max_length=2048)
    authorization: Literal["docker_local"] = "docker_local"
    operator_confirmed: bool = False
    allow_safe_post: bool = False


class RemoteDockerProbeRequest(BaseModel):
    """Explicit operator confirmation for a read-only remote Docker probe."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False


class RemoteDockerAnalyzeRequest(BaseModel):
    """Bind an inert observer to one allowlisted private container port."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False
    container_name: str = Field(min_length=1, max_length=63)
    container_port: int = Field(ge=1, le=65535)
    path: str = Field(default="/", min_length=1, max_length=2048)
    allow_safe_post: bool = False


class RemoteDockerBindRequest(BaseModel):
    """Bind a PG-281 abstract plan to an observed authorized surface."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False
    plan: dict[str, Any] = Field(default_factory=dict)
    surface: dict[str, Any] = Field(default_factory=dict)
    evaluator_evidence: dict[str, Any] = Field(default_factory=dict)
    hard_negative: bool = False


class RemoteDockerEvaluateRequest(BaseModel):
    """Evaluator-only GET/POST replay evidence; no raw wire fields accepted."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False
    surface: dict[str, Any] = Field(default_factory=dict)
    reset: dict[str, Any] = Field(default_factory=dict)
    reference: dict[str, Any] = Field(default_factory=dict)
    negative: dict[str, Any] = Field(default_factory=dict)
    candidate: dict[str, Any] = Field(default_factory=dict)
    replay: dict[str, Any] = Field(default_factory=dict)
    typed_evidence: dict[str, Any] = Field(default_factory=dict)
    hard_negative: bool = False


class RemoteDockerPg292LiveRequest(BaseModel):
    """Join a PG-292 gate, abstract Rule-IR plan and typed replay projection."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False
    context_tokens: list[str] = Field(min_length=1, max_length=192)
    plan_tokens: list[str] = Field(min_length=3, max_length=64)
    gate_probability: float = Field(ge=0.0, le=1.0)
    gate_threshold: float = Field(ge=0.0, le=1.0)
    surface: dict[str, Any] = Field(default_factory=dict)
    reset: dict[str, Any] = Field(default_factory=dict)
    reference: dict[str, Any] = Field(default_factory=dict)
    negative: dict[str, Any] = Field(default_factory=dict)
    candidate: dict[str, Any] = Field(default_factory=dict)
    replay: dict[str, Any] = Field(default_factory=dict)
    typed_evidence: dict[str, Any] = Field(default_factory=dict)
    hard_negative: bool = False
    operator_reviewed: bool = False
    independent_audit_pass: bool = False
    cross_seed_reviewed: bool = False


class RemoteDockerObservationRequest(BaseModel):
    """Turn target-side bounded evaluator output into one PG-286 token record."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False
    record_id: str = Field(min_length=1, max_length=160)
    surface: dict[str, Any] = Field(default_factory=dict)
    reset: dict[str, Any] = Field(default_factory=dict)
    baseline: dict[str, Any] = Field(default_factory=dict)
    reference: dict[str, Any] = Field(default_factory=dict)
    negative: dict[str, Any] = Field(default_factory=dict)
    candidate: dict[str, Any] = Field(default_factory=dict)
    replay: dict[str, Any] = Field(default_factory=dict)
    typed_evidence: dict[str, Any] = Field(default_factory=dict)
    fields: list[str] = Field(default_factory=list, max_length=32)
    modality_projection: Optional[dict[str, Any]] = None
    hard_negative: bool = False
    operator_reviewed: bool = False


class RemoteDockerIdentifiabilityRequest(BaseModel):
    """Bind a complete PG-286 observation to a PG-287 live target plan."""

    authorization: Literal["remote_docker"] = "remote_docker"
    operator_confirmed: bool = False
    observation_record: dict[str, Any] = Field(default_factory=dict)
    observed_binding: dict[str, Any] = Field(default_factory=dict)
    reference_plan: dict[str, Any] = Field(default_factory=dict)
    source_attestation: dict[str, Any] = Field(default_factory=dict)
    split: Literal["train", "route_dev", "family_holdout", "unassigned"] = "unassigned"
    hard_negative: bool = False
    operator_reviewed: bool = False


def _public_session(session: dict[str, Any]) -> dict[str, Any]:
    result = dict(session)
    scenario = dict(result["scenario"])
    scenario.pop("hidden_rule", None)
    scenario.pop("validation_cases", None)
    result["scenario"] = scenario
    return result


def _append_observation(session: dict[str, Any], payload: dict[str, Any], output: bool, source: str) -> dict[str, Any]:
    observation = {
        "input": payload.get("input", {}),
        "context": payload.get("context", {}),
        "state": payload.get("state", {}),
        "output": bool(output),
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "episode_id": payload.get("episode_id", "default"),
    }
    for key in ("state_after", "step", "goal", "terminal", "available_actions", "history"):
        if payload.get(key) is not None:
            observation[key] = payload[key]
    session["observations"].append(observation)
    session["query_count"] += 1
    session["candidates"] = []
    session["last_closure_report"] = None
    return observation


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": app.version}


@app.get("/api/research/events")
def research_events(limit: int = 100) -> dict[str, Any]:
    events = list_events(limit=max(1, min(limit, 500)))
    return {"events": events, "count": len(events)}


@app.get("/api/research/frontend-loop")
def frontend_loop_summary() -> dict[str, Any]:
    path = BASE_DIR / "research" / "frontend_loop_08_summary.json"
    if not path.exists():
        raise HTTPException(404, "frontend research loop summary not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/research/ops")
def research_ops_snapshot() -> dict[str, Any]:
    """Return bounded collector/reviewer/trainer workbench data."""

    return build_research_ops_snapshot()


@app.get("/api/research/ops/payloads")
def research_ops_payloads() -> dict[str, Any]:
    """Return the bounded, human-readable local payload review projection."""

    return build_payload_review()


@app.post("/api/research/events")
def create_research_event(request: ResearchEventRequest) -> dict[str, Any]:
    return emit_event(**request.model_dump())


@app.post("/api/research/maturity/evaluate")
def research_maturity(request: ResearchMaturityRequest) -> dict[str, Any]:
    result = evaluate_research_maturity(request.model_dump())
    emit_event(
        actor="maturity-gate",
        tool="research.maturity.evaluate",
        phase="scale_readiness",
        status="complete" if result["ready"] else "blocked",
        message=f"工程化扩展门禁：{result['state']}",
        payload=result,
        artifact="research/improvement_rules.json",
    )
    return result


@app.post("/api/research/model-capability/evaluate")
def model_capability(request: ModelCapabilityRequest) -> dict[str, Any]:
    try:
        result = evaluate_model_capability(request.evidence, policy=request.policy)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    emit_event(
        actor="model-capability-gate",
        tool="model.capability.evaluate",
        phase="dataset-holdout",
        status="complete" if result["status"] == "pass" else "blocked",
        message=f"模型能力数据集门禁：{result['status']}",
        payload={"status": result["status"], "reasons": result["reasons"], "summary": result["summary"]},
        artifact="research/pg_pk_30_model_capability_dataset_gate_v1.json",
    )
    return result


@app.post("/api/research/trace/step")
def trace_step(request: TraceStepRequest) -> dict[str, Any]:
    try:
        result = validate_trace_step(request.step)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    emit_event(
        actor="trace-aligned-model",
        tool="research.trace.step",
        phase="local-replay",
        status="complete",
        message="已回显安全探测步骤并生成影子数据行",
        payload={"episode_id": result["episode_id"], "step_id": result["step_id"], "trace_sha256": result["trace_sha256"]},
        artifact="research/trace_aligned_replay_learning_policy_v1.json",
    )
    return result


@app.post("/api/research/trace/episode")
def trace_episode(request: TraceEpisodeRequest) -> dict[str, Any]:
    try:
        result = evaluate_episode(request.steps)
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    emit_event(
        actor="trace-aligned-model",
        tool="research.trace.episode",
        phase="local-replay",
        status="complete" if result["status"] == "accepted_evaluation" else "warning",
        message=f"逐步回放数据集状态：{result['status']}",
        payload={"episode_id": result["episode_id"], "step_count": result["step_count"], "reasons": result["reasons"]},
        artifact="research/trace_aligned_replay_learning_policy_v1.json",
    )
    return result


@app.post("/api/research/scale-failure/triage")
def scale_failure_triage(request: ScaleFailureTriageRequest) -> dict[str, Any]:
    result = triage_scale_failure(request.model_dump())
    emit_event(
        actor="root-cause-router",
        tool="scale_failure.triage",
        phase="root_cause",
        status="warning" if result["classification"] in {"mixed", "inconclusive"} else "complete",
        message=f"扩展故障归因：{result['classification']}",
        payload=result,
        artifact="research/RESEARCH_RULES.md",
    )
    return result


@app.get("/api/scenarios")
def scenarios() -> list[dict[str, Any]]:
    return public_scenarios()


@app.get("/api/maze/labs")
def maze_labs() -> list[dict[str, Any]]:
    """List safe local maze laboratories and their semantic exit oracles."""
    return public_maze_labs()


@app.get("/api/maze/labs/{lab_id}")
def maze_lab(lab_id: str) -> dict[str, Any]:
    lab = get_maze_lab(lab_id)
    if lab is None:
        raise HTTPException(404, "maze lab not found")
    return lab


@app.post("/api/maze/oracle/dom")
def dom_oracle(request: DomOracleRequest) -> dict[str, Any]:
    try:
        return run_dom_oracle(
            request.value,
            sink=request.sink,
            transforms=request.transforms,
            marker=request.marker,
        ).to_dict()
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/maze/oracle/sql")
def sql_oracle(request: SqlOracleRequest) -> dict[str, Any]:
    if request.fragment_class not in FRAGMENT_CLASSES:
        raise HTTPException(400, "unknown synthetic SQL fragment class")
    return run_sql_ast_oracle(request.fragment_class).to_dict()


@app.get("/api/maze/replay/dom")
def replay_dom(
    value: str = "",
    sink: Literal["innerHTML", "template.innerHTML"] = "innerHTML",
    transforms: str = "",
    marker: str = "sift-marker",
) -> dict[str, Any]:
    """Read-only GET adapter used by the localhost replay collector."""

    if len(value) > 2048 or len(marker) > 64 or len(transforms) > 128:
        raise HTTPException(400, "replay probe exceeds the local bound")
    transform_list = [item for item in (part.strip() for part in transforms.split(",")) if item]
    try:
        return run_dom_oracle(value, sink=sink, transforms=transform_list, marker=marker).to_dict()
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/maze/replay/dom")
def replay_dom_post(request: DomOracleRequest) -> dict[str, Any]:
    """Read-only POST adapter for the same detached DOM oracle.

    This is intentionally paired with the GET route so the local collector can
    test transport changes without introducing a state mutation or executable
    browser content.
    """

    try:
        return run_dom_oracle(
            request.value,
            sink=request.sink,
            transforms=request.transforms,
            marker=request.marker,
        ).to_dict()
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/maze/replay/sql")
def replay_sql(fragment_class: str) -> dict[str, Any]:
    """Read-only GET adapter for the abstract SQL AST oracle."""

    if fragment_class not in FRAGMENT_CLASSES:
        raise HTTPException(400, "unknown synthetic SQL fragment class")
    return run_sql_ast_oracle(fragment_class).to_dict()


@app.post("/api/maze/replay/sql")
def replay_sql_post(request: SqlOracleRequest) -> dict[str, Any]:
    """Read-only POST adapter for the abstract SQL AST oracle."""

    if request.fragment_class not in FRAGMENT_CLASSES:
        raise HTTPException(400, "unknown synthetic SQL fragment class")
    return run_sql_ast_oracle(request.fragment_class).to_dict()


@app.get("/api/maze/replay/logic")
def replay_logic(probe_class: str = "normal", surface: str = "authorization_boundary") -> dict[str, Any]:
    if probe_class not in PROBE_CLASSES or surface not in SURFACES:
        raise HTTPException(400, "unknown abstract logic replay class")
    return run_logic_replay_oracle(probe_class, surface=surface)


@app.post("/api/maze/replay/logic")
def replay_logic_post(request: LogicReplayRequest) -> dict[str, Any]:
    if request.probe_class not in PROBE_CLASSES or request.surface not in SURFACES:
        raise HTTPException(400, "unknown abstract logic replay class")
    return run_logic_replay_oracle(request.probe_class, surface=request.surface)


@app.post("/api/maze/detection-payload")
def detection_payload(request: DetectionPayloadRequest) -> dict[str, Any]:
    try:
        return validate_detection_payload(request.model_dump())
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/maze/target/scope")
def authorized_target_scope() -> dict[str, Any]:
    """Expose the non-secret Docker origin allowlist to the operator UI."""

    return {
        "authorized_origins": list(iter_authorized_origins()),
        "loopback_any_port": True,
        "private_docker_networks": "explicit SIFT_AUTHORIZED_DOCKER_TARGETS only",
        "external_network": False,
        "raw_body_stored": False,
        "typed_oracle_required_for_confirmation": True,
    }


@app.post("/api/maze/target/analyze")
async def authorized_target_analyze(request: AuthorizedTargetAnalyzeRequest) -> dict[str, Any]:
    if request.authorization != "docker_local":
        raise HTTPException(400, "only docker_local authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before network observation")
    try:
        result = await analyze_authorized_target(
            target_url=request.target_url,
            allow_safe_post=request.allow_safe_post,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    emit_event(
        actor="operator-target-runner",
        tool="maze.target.analyze",
        phase="authorized_docker_observation",
        status="warning" if result["candidate_status"] == "candidate" else "complete",
        message=f"Docker-local target observation: {result['candidate_status']}",
        payload={
            "target": result["target"],
            "request_count": result["request_count"],
            "candidate_status": result["candidate_status"],
            "evidence_sha256": result["evidence_sha256"],
            "promotion": result["promotion"],
        },
        artifact="app/authorized_target_session.py",
    )
    return result


@app.get("/api/maze/remote-docker/scope")
def authorized_remote_docker_scope() -> dict[str, Any]:
    """Expose the fixed remote host and its non-mutating command contract."""

    config = RemoteDockerConfig.from_environment()
    return {
        "schema_version": "pg280-authorized-remote-docker-adapter-v1",
        "scope": config.scope(),
        "operator_scope_lines": list(iter_remote_scope()),
        "target_container_allowlist_configured": bool(os.environ.get("SIFT_REMOTE_DOCKER_TARGET_CONTAINERS", "").strip()),
        "typed_oracle_required": True,
        "fresh_reset_required": True,
        "training_or_replay_started": False,
    }


@app.post("/api/maze/remote-docker/probe")
def authorized_remote_docker_probe(request: RemoteDockerProbeRequest) -> dict[str, Any]:
    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before remote probing")
    result = probe_authorized_remote_docker()
    emit_event(
        actor="operator-remote-docker-probe",
        tool="maze.remote_docker.probe",
        phase="authorized_remote_docker_scope",
        status="complete" if result["status"] == "available" else "blocked",
        message=f"PG-280 remote Docker probe: {result['status']}",
        payload={
            "status": result["status"],
            "docker_binary": result["docker_binary"],
            "docker_server": result["docker_server"],
            "running_containers": result["running_containers"],
            "evidence_sha256": result["evidence_sha256"],
        },
        artifact="research/pg280_remote_docker_probe_v2.json",
    )
    return result


@app.post("/api/maze/remote-docker/analyze")
async def authorized_remote_docker_analyze(request: RemoteDockerAnalyzeRequest) -> dict[str, Any]:
    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before remote observation")
    try:
        result = await analyze_authorized_remote_target(
            container_name=request.container_name,
            container_port=request.container_port,
            path=request.path,
            allow_safe_post=request.allow_safe_post,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from error
    emit_event(
        actor="operator-remote-docker-runner",
        tool="maze.remote_docker.analyze",
        phase="authorized_remote_docker_observation",
        status="complete" if result["status"] == "observed" else "blocked",
        message=f"PG-280 remote target observation: {result['status']}",
        payload={
            "status": result["status"],
            "remote": result["remote"],
            "promotion": result["promotion"],
            "evidence_sha256": result["evidence_sha256"],
        },
        artifact="app/authorized_remote_docker.py",
    )
    return result


@app.post("/api/maze/remote-docker/bind")
def authorized_remote_docker_bind(request: RemoteDockerBindRequest) -> dict[str, Any]:
    """Apply the PG-282 evaluator-only binding gate after a remote probe.

    The endpoint returns an abstract wire shape and evidence projection only;
    it never sends the plan or constructs a literal payload.  A live positive
    therefore requires the separate target-side evaluator contract.
    """

    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before remote binding")
    remote_probe = probe_authorized_remote_docker()
    try:
        result = bind_abstract_plan(
            request.plan,
            request.surface,
            remote_probe=remote_probe,
            evaluator_evidence=request.evaluator_evidence,
            hard_negative=request.hard_negative,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    emit_event(
        actor="operator-remote-docker-binding",
        tool="maze.remote_docker.bind",
        phase="authorized_remote_docker_evaluator",
        status="complete" if result["status"] == "confirmed_positive" else "blocked",
        message=f"PG-282 evaluator binding: {result['status']}",
        payload={
            "status": result["status"],
            "decision": result["decision"],
            "checks": result["checks"],
            "binding_evidence_sha256": result["binding_evidence_sha256"],
        },
        artifact="app/pg282_evaluator_binding.py",
    )
    return result


@app.post("/api/maze/remote-docker/evaluate")
def authorized_remote_docker_evaluate(request: RemoteDockerEvaluateRequest) -> dict[str, Any]:
    """Validate target-side typed GET/POST replay evidence.

    This endpoint deliberately does not send a request.  The remote target
    runner supplies bounded projections and a fresh-reset attestation; this
    API only applies the PG-284 acceptance gate and returns hashes.
    """

    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before evaluator validation")
    remote_probe = probe_authorized_remote_docker()
    try:
        result = evaluate_typed_replay(
            surface=request.surface,
            reset=request.reset,
            reference=request.reference,
            negative=request.negative,
            candidate=request.candidate,
            replay=request.replay,
            typed_evidence=request.typed_evidence,
            remote_probe=remote_probe,
            hard_negative=request.hard_negative,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    emit_event(
        actor="operator-remote-docker-evaluator",
        tool="maze.remote_docker.evaluate",
        phase="authorized_remote_docker_evaluator",
        status="complete" if result["status"] == "confirmed_effect" else "blocked",
        message=f"PG-284 typed evaluator: {result['status']}",
        payload={
            "status": result["status"],
            "decision": result["decision"],
            "checks": result["checks"],
            "evidence_projection_sha256": result["evidence_projection_sha256"],
        },
        artifact="app/pg284_evaluator_contract.py",
    )
    return result


@app.post("/api/maze/remote-docker/pg292-live")
def authorized_remote_docker_pg292_live(request: RemoteDockerPg292LiveRequest) -> dict[str, Any]:
    """Join PG-292, PG-288 and PG-284 without emitting a wire request.

    The caller supplies only abstract context/Rule-IR tokens and bounded
    target-side projections.  The server performs a fresh read-only remote
    Docker probe, then returns a fail-closed replay decision.  Even a complete
    typed replay remains a candidate for explicit training review; this route
    never creates a literal payload or authorizes network emission.
    """

    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before PG-292-live evaluation")
    remote_probe = probe_authorized_remote_docker()
    try:
        result = evaluate_pg292_live(
            context_tokens=request.context_tokens,
            plan_tokens=request.plan_tokens,
            gate_probability=request.gate_probability,
            gate_threshold=request.gate_threshold,
            surface=request.surface,
            reset=request.reset,
            reference=request.reference,
            negative=request.negative,
            candidate=request.candidate,
            replay=request.replay,
            typed_evidence=request.typed_evidence,
            remote_probe=remote_probe,
            hard_negative=request.hard_negative,
            operator_reviewed=request.operator_reviewed,
            independent_audit_pass=request.independent_audit_pass,
            cross_seed_reviewed=request.cross_seed_reviewed,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    result["remote_probe"] = {
        "status": remote_probe.get("status", "unknown"),
        "evidence_sha256": remote_probe.get("evidence_sha256", ""),
    }
    emit_event(
        actor="operator-remote-docker-pg292-live",
        tool="maze.remote_docker.pg292_live",
        phase="pg292_live_typed_evaluator",
        status="complete" if result["status"] == "typed_replay_candidate_for_training_review" else "blocked",
        message=f"PG-292-live: {result['status']}",
        payload={
            "status": result["status"],
            "decision": result["decision"],
            "checks": result["checks"],
            "reasons": result["reasons"],
            "evidence_sha256": result["evidence_sha256"],
        },
        artifact="app/pg292_live.py",
    )
    return result


@app.post("/api/maze/remote-docker/observation")
def authorized_remote_docker_observation(request: RemoteDockerObservationRequest) -> dict[str, Any]:
    """Ingest one authorized evaluator projection into the PG-286 token lane.

    The endpoint is intentionally an ingest boundary rather than a scanner:
    the target-side runner performs the authorized GET/POST/fresh-reset work
    and submits only bounded projections.  This API applies the remote probe,
    typed replay and shared-token gates; it never sends a request itself.
    """

    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before observation ingestion")
    remote_probe = probe_authorized_remote_docker()
    try:
        result = collect_pg286_live_record(
            record_id=request.record_id,
            surface=request.surface,
            reset=request.reset,
            baseline=request.baseline,
            reference=request.reference,
            negative=request.negative,
            candidate=request.candidate,
            replay=request.replay,
            typed_evidence=request.typed_evidence,
            remote_probe=remote_probe,
            fields=request.fields,
            modality_projection=request.modality_projection,
            hard_negative=request.hard_negative,
            operator_reviewed=request.operator_reviewed,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    result["remote_probe"] = {
        "status": remote_probe.get("status", "unknown"),
        "evidence_sha256": remote_probe.get("evidence_sha256", ""),
    }
    emit_event(
        actor="operator-remote-docker-observation",
        tool="maze.remote_docker.observation",
        phase="pg286_observation_token_collection",
        status="complete" if result["decision"] == "eligible_for_cross_seed_review" else "blocked",
        message=f"PG-286 observation collection: {result['decision']}",
        payload={
            "record_id": result["record_id"],
            "decision": result["decision"],
            "token_evidence_status": result["token_evidence_status"],
            "evaluator_status": result["evaluator_status"],
            "training_eligible": result["training_eligible"],
            "evidence_hash": result["evidence_hash"],
            "record_sha256": result["record_sha256"],
        },
        artifact="app/pg286_live_collection.py",
    )
    return result


@app.post("/api/maze/remote-docker/identifiability")
def authorized_remote_docker_identifiability(request: RemoteDockerIdentifiabilityRequest) -> dict[str, Any]:
    """Ingest one complete evaluator observation for PG-287-live.

    The target-side runner must already have performed the authorised
    GET/POST/fresh-reset replay.  This endpoint only validates bounded
    projections and binds the observed encoding to an abstract reference
    plan; it never creates a request or promotes a training row.
    """

    if request.authorization != "remote_docker":
        raise HTTPException(400, "only remote_docker authorization is supported")
    if not request.operator_confirmed:
        raise HTTPException(400, "operator confirmation is required before identifiability ingestion")
    remote_probe = probe_authorized_remote_docker()
    if remote_probe.get("status") != "available":
        raise HTTPException(409, "authorized remote Docker is unavailable; PG-287-live remains blocked")
    try:
        result = collect_pg287_live_record(
            observation_record=request.observation_record,
            observed_binding=request.observed_binding,
            reference_plan=request.reference_plan,
            source_attestation=request.source_attestation,
            remote_probe=remote_probe,
            split=request.split,
            hard_negative=request.hard_negative,
            operator_reviewed=request.operator_reviewed,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    result["remote_probe"] = {"status": remote_probe.get("status", "unknown"), "evidence_sha256": remote_probe.get("evidence_sha256", "")}
    emit_event(
        actor="operator-remote-docker-identifiability",
        tool="maze.remote_docker.identifiability",
        phase="pg287_live_identifiability_collection",
        status="blocked" if not result.get("training_eligible") else "complete",
        message=f"PG-287-live identifiability collection: {result.get('variant', 'unknown')} / training_eligible={bool(result.get('training_eligible'))}",
        payload={
            "record_id": result.get("record_id"),
            "variant": result.get("variant"),
            "training_eligible": result.get("training_eligible"),
            "source_evidence_hash": result.get("source_evidence_hash"),
            "record_sha256": result.get("record_sha256"),
        },
        artifact="app/pg287_live_collection.py",
    )
    return result


@app.get("/api/maze/runs/latest")
def latest_maze_run() -> dict[str, Any]:
    path = latest_manifest(DEFAULT_ARTIFACT_ROOT)
    if path is None:
        raise HTTPException(404, "no maze run manifest found")
    return load_manifest(path)


@app.get("/api/maze/runs/{run_id}")
def maze_run(run_id: str) -> dict[str, Any]:
    path = DEFAULT_ARTIFACT_ROOT / run_id / "manifest.json"
    if not path.is_file() or not path.resolve().is_relative_to(DEFAULT_ARTIFACT_ROOT.resolve()):
        raise HTTPException(404, "maze run not found")
    manifest = load_manifest(path)
    ledger_path = DEFAULT_ARTIFACT_ROOT / run_id / "evidence.jsonl"
    manifest["ledger_verification"] = verify_ledger(ledger_path)
    return manifest


@app.post("/api/sessions")
def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    if request.custom is not None:
        scenario = request.custom.model_dump()
        scenario["id"] = "custom"
        if scenario["mode"] == "oracle" and scenario.get("hidden_rule") is None:
            raise HTTPException(400, "oracle 模式必须提供 hidden_rule")
    elif request.scenario_id:
        scenario = get_scenario(request.scenario_id)
        if scenario is None:
            raise HTTPException(404, "scenario not found")
    else:
        raise HTTPException(400, "scenario_id 或 custom 至少提供一个")
    result = _public_session(store.create(scenario))
    emit_event(
        actor="research-engine",
        tool="session.create",
        phase="experiment",
        status="complete",
        message=f"建立独立实验会话：{scenario.get('name', scenario.get('id'))}",
        payload={"session_id": result["id"], "scenario_id": scenario.get("id"), "field_count": len(scenario.get("fields", []))},
    )
    return result


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return _public_session(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    if not store.delete(session_id):
        raise HTTPException(404, "session not found")
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/probe")
def probe(session_id: str, request: ProbeRequest) -> dict[str, Any]:
    def action(session: dict[str, Any]):
        scenario = session["scenario"]
        hidden_rule = scenario.get("hidden_rule")
        if scenario.get("mode") == "manual" or hidden_rule is None:
            raise HTTPException(400, "该会话是手工标注模式，请使用 /observe")
        envelope = request.model_dump()
        if envelope.get("history") is not None:
            history = envelope["history"]
        else:
            episode = str(envelope.get("episode_id", "default"))
            history = [obs for obs in session["observations"] if str(obs.get("episode_id", "default")) == episode]
        output = truthy_result(hidden_rule, envelope, history)
        observation = _append_observation(session, envelope, output, "oracle")
        return {"output": output, "observation": observation, "query_count": session["query_count"]}

    result = store.mutate(session_id, action)
    if result is None:
        raise HTTPException(404, "session not found")
    emit_event(
        actor="oracle",
        tool="blackbox.probe",
        phase="observation",
        status="complete",
        message=f"Oracle 返回 {'TRUE' if result['output'] else 'FALSE'}",
        payload={"session_id": session_id, "input": request.input, "context": request.context, "state": request.state, "output": result["output"]},
    )
    return result


@app.post("/api/sessions/{session_id}/observe")
def observe(session_id: str, request: ObserveRequest) -> dict[str, Any]:
    def action(session: dict[str, Any]):
        payload = request.model_dump()
        output = payload.pop("output")
        source = payload.pop("source")
        observation = _append_observation(session, payload, output, source)
        return {"observation": observation, "query_count": session["query_count"]}

    result = store.mutate(session_id, action)
    if result is None:
        raise HTTPException(404, "session not found")
    return result


@app.post("/api/sessions/{session_id}/observations/import")
def import_observations(session_id: str, request: ImportObservationsRequest) -> dict[str, Any]:
    def action(session: dict[str, Any]):
        for item in request.observations:
            payload = item.model_dump()
            output = payload.pop("output")
            source = payload.pop("source")
            _append_observation(session, payload, output, source)
        return {"imported": len(request.observations), "query_count": session["query_count"]}

    result = store.mutate(session_id, action)
    if result is None:
        raise HTTPException(404, "session not found")
    return result


@app.post("/api/sessions/{session_id}/search")
def run_search(session_id: str, request: SearchRequest) -> dict[str, Any]:
    def action(session: dict[str, Any]):
        scenario = session["scenario"]
        candidates = search_rules(
            scenario["fields"],
            session["observations"],
            max_depth=request.max_depth,
            beam_width=request.beam_width,
            history_depth=request.history_depth,
        )
        session["candidates"] = [candidate.to_dict() for candidate in candidates]
        session["last_closure_report"] = None
        return {"candidates": session["candidates"], "count": len(session["candidates"])}

    result = store.mutate(session_id, action)
    if result is None:
        raise HTTPException(404, "session not found")
    best = result["candidates"][0] if result.get("candidates") else None
    emit_event(
        actor="rule-inducer",
        tool="rule.search",
        phase="induction",
        status="complete",
        message=f"生成 {result['count']} 个候选规则",
        payload={"session_id": session_id, "candidate_count": result["count"], "best_accuracy": best.get("accuracy") if best else None, "best_rule": best.get("pretty") if best else None},
    )
    return result


@app.get("/api/sessions/{session_id}/suggest")
def suggest(session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    raw = session.get("candidates", [])
    if len(raw) < 2:
        raise HTTPException(400, "请先搜索并保留至少两个候选规则")

    # Re-score candidate expressions to recover the internal Candidate representation.
    from .search import score_expr

    candidates = [score_expr(item["expr"], session["observations"]) for item in raw]
    suggestion = suggest_query(session["scenario"]["fields"], candidates, session["observations"])
    if suggestion is None:
        return {"suggestion": None, "reason": "候选规则在当前输入域内没有可用分歧"}
    emit_event(
        actor="active-policy",
        tool="query.suggest",
        phase="counterexample-search",
        status="complete",
        message="选择最大候选分歧探针",
        payload={"session_id": session_id, "suggestion": suggestion},
    )
    return {"suggestion": suggestion}


@app.post("/api/sessions/{session_id}/closure/analyze")
def closure_analyze(session_id: str, request: ClosureRequest) -> dict[str, Any]:
    def action(session: dict[str, Any]):
        if request.auto_search and (not session.get("candidates")) and session.get("observations"):
            candidates = search_rules(
                session["scenario"]["fields"],
                session["observations"],
                max_depth=request.max_depth,
                beam_width=request.beam_width,
                history_depth=request.history_depth,
            )
            session["candidates"] = [candidate.to_dict() for candidate in candidates]

        report = analyze_closure(
            scenario=session["scenario"],
            observations=session["observations"],
            raw_candidates=session.get("candidates", []),
            previous_reports=session.get("closure_history", []),
            max_cases=request.max_cases,
            top_candidates=request.top_candidates,
            accuracy_tolerance=request.accuracy_tolerance,
            history_depth=request.history_depth,
            coverage_threshold=request.coverage_threshold,
            goal_mode=request.goal_mode,
        )
        session.setdefault("closure_history", []).append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": report["closure_status"],
            "score": report["closure_score"],
            "fingerprint": report["fingerprint"],
        })
        session["closure_history"] = session["closure_history"][-20:]
        session["last_closure_report"] = report
        return report

    result = store.mutate(session_id, action)
    if result is None:
        raise HTTPException(404, "session not found")
    emit_event(
        actor="evidence-engine",
        tool="closure.analyze",
        phase="verification",
        status="complete",
        message=f"闭环状态：{result['closure_status']}",
        payload={"session_id": session_id, "closure_status": result["closure_status"], "closure_score": result["closure_score"], "confidence": result["confidence"], "reasons": result.get("reasons", [])},
    )
    return result


@app.post("/api/sessions/{session_id}/validate")
def validate(session_id: str, request: ValidateRequest) -> dict[str, Any]:
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    scenario = session["scenario"]
    hidden_rule = scenario.get("hidden_rule")

    if hidden_rule is None:
        correct = 0
        histories = build_histories_before(session["observations"])
        details = []
        for obs, history in zip(session["observations"], histories):
            envelope = {"input": obs.get("input", {}), "context": obs.get("context", {}), "state": obs.get("state", {})}
            prediction = truthy_result(request.rule, envelope, history)
            correct += prediction == bool(obs["output"])
            details.append({"envelope": envelope, "expected": bool(obs["output"]), "predicted": prediction})
        total = len(session["observations"])
        return {
            "mode": "observed_only",
            "rule_pretty": pretty(request.rule),
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "note": "手工模式没有隐藏测试 oracle，结果只覆盖已录入观测。",
            "failures": [item for item in details if item["expected"] != item["predicted"]][:30],
        }

    cases = scenario.get("validation_cases")
    if cases:
        prepared = []
        for case in cases:
            prepared.append({
                "envelope": {"input": case.get("input", {}), "context": case.get("context", {}), "state": case.get("state", {})},
                "history": case.get("history", []),
            })
    else:
        prepared = [{"envelope": envelope, "history": []} for envelope in enumerate_envelopes(scenario["fields"], request.max_cases)]

    correct = 0
    failures = []
    for case in prepared:
        expected = truthy_result(hidden_rule, case["envelope"], case["history"])
        predicted = truthy_result(request.rule, case["envelope"], case["history"])
        if expected == predicted:
            correct += 1
        elif len(failures) < 30:
            failures.append({**case, "expected": expected, "predicted": predicted})
    total = len(prepared)
    return {
        "mode": "hidden_validation",
        "rule_pretty": pretty(request.rule),
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "failures": failures,
    }


@app.post("/api/expr/evaluate")
def evaluate_expression(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = truthy_result(payload["rule"], payload.get("envelope", {}), payload.get("history", []))
        return {"result": result, "pretty": pretty(payload["rule"])}
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
