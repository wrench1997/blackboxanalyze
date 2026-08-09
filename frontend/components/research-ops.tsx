"use client";

import { useEffect, useMemo, useState } from "react";

type View = "tasks" | "architecture" | "capability";
type Task = {
  id: string;
  role: "collector" | "reviewer" | "trainer";
  owner: string;
  human_required: boolean;
  status: string;
  label: string;
  route: string;
  seed: number;
  method: string;
  typed_effect: boolean;
  confirmed_positive: boolean;
  reasons: string[];
  evidence_hash: string;
  instruction: string;
  raw_material_available: boolean;
};

type RequestShape = {
  logical_values: Record<string, string>;
  encoded_values: Record<string, string>;
  wire: string;
};

type PayloadChannel = {
  status: string;
  request?: RequestShape;
  true?: RequestShape;
  false?: RequestShape;
};

type PayloadEntry = {
  id: string;
  family: string;
  route: string;
  method: string;
  fields: string[];
  validation_status: string;
  expected_oracle: string;
  effect_claim: string;
  source: string;
  ai: PayloadChannel;
  reference: PayloadChannel;
  negative: PayloadChannel;
  oracle_evidence: {
    available: boolean;
    status: string;
    source_report: string;
    seed?: number;
    pattern_id: string;
    matched: boolean;
    match_count?: number | string | null;
    span_buckets: string[];
    reasons: string[];
    candidate_reference_agreement?: boolean | null;
    negative_clean?: boolean | null;
    evidence_sha256: string;
    oracle_facts: Record<string, unknown>;
    candidate_projection: Record<string, unknown>;
    candidate_true_projection: Record<string, unknown>;
    candidate_false_projection: Record<string, unknown>;
    reference_projection: Record<string, unknown>;
    negative_projection: Record<string, unknown>;
    raw_response_body_stored: boolean;
    raw_payload_stored: boolean;
  };
  notes: string[];
  review_only: boolean;
  persisted: boolean;
  training_eligible: boolean;
};

type PayloadReview = {
  schema_version: string;
  generated_at: string;
  review_only: boolean;
  persisted: boolean;
  training_eligible: boolean;
  target_scope: { kind: string; allowed_origins: string[]; arbitrary_target_input: boolean };
  safety: Record<string, boolean>;
  disclaimer: string;
  entries: PayloadEntry[];
};

type ProcessStage = { id: string; label: string; state: string; detail: string };
type ProcessTrace = {
  id: string;
  family: string;
  route: string;
  method: string;
  seed: number;
  fresh_reset: boolean;
  database_health: string;
  target_hash: string;
  evidence_hash: string;
  ai_sent: boolean;
  reference_sent: boolean;
  negative_sent: boolean;
  confirmed_positive: boolean;
  oracle_available: boolean;
  abstract_probe: string;
  stages: ProcessStage[];
};

type LearningFinding = {
  id: string;
  severity: string;
  title: string;
  evidence: string;
  action: string;
};

type LearningQueue = {
  id: string;
  priority: string;
  owner: string;
  title: string;
  why: string;
  status: string;
  current: string;
  minimum_quota: string[];
  collect: string[];
  acceptance: string[];
  output_lane: string;
  prevents: string[];
};

type LearningRequirements = {
  schema_version: string;
  title: string;
  principle: string;
  evidence: {
    source_report: string;
    source_audit: string;
    report_ready: boolean;
    audit_pass: boolean;
    controlled_rows: number;
    real_multifamily_gold_rows: number;
    coarse_conflict_groups: number;
    coarse_conflicting_rows: number;
    process_question_recovery_worst_seed: number;
    conservative_question_recovery_worst_seed: number;
    dpo_question_recovery_worst_seed: number;
    pg278_enriched_conflict_groups?: number;
    pg278_post_conflict_groups?: number;
    pg278_pre_transition_worst_seed?: number;
    pg278_post_transition_worst_seed?: number;
    pg278_slot_binding_worst_seed?: number;
    pg278_pair_flip_worst_seed?: number;
    pg278_missing_safe_worst_seed?: number;
    pg278_family_question_worst_seed?: number;
    pg278_gate_status?: string;
    pg279_enriched_conflict_groups?: number;
    pg279_post_conflict_groups?: number;
    pg279_get_rows?: number;
    pg279_post_rows?: number;
    pg279_failure_repair_rows?: number;
    pg279_typed_effect_rows?: number;
    pg279_abstain_rows?: number;
    pg279_family_question_worst_seed?: number;
    pg279_gate_status?: string;
    pg279_retention_pre_min?: number;
    pg279_retention_post_min?: number;
    pg279_retention_missing_safe_min?: number;
    pg279_training_mix_sha256?: string;
    pg280_conditional_entropy_bits?: number;
    pg280_bayes_error_lower_bound?: number;
    pg280_final_only_pre_supervision_rows?: number;
    pg280_final_only_post_accuracy?: number;
    pg280_final_only_ask_rate?: number;
    pg280_process_pre_supervision_rows?: number;
    pg280_process_post_accuracy?: number;
    pg280_process_ask_rate?: number;
    pg280_process_safe_rate?: number;
    pg280_hard_negative_rows?: number;
    pg280_docker_status?: string;
    pg280_gate_status?: string;
    pg281_train_rows?: number;
    pg281_route_dev_rows?: number;
    pg281_family_holdout_rows?: number;
    pg281_hard_negative_rows?: number;
    pg281_route_positive_recall_min?: number;
    pg281_family_positive_recall_min?: number;
    pg281_hard_negative_reject_min?: number;
    pg281_hard_negative_false_allow_max?: number;
    pg281_gate_status?: string;
    pg281_docker_status?: string;
    claim: string;
  };
  findings: LearningFinding[];
  queues: LearningQueue[];
  record_contract: Array<{ group: string; fields: string[]; rule: string }>;
  resources: Array<{ category: string; required: boolean; items: string[]; why: string }>;
  forbidden: Array<{ id: string; title: string; reason: string; lane: string }>;
  promotion_gate: { current_status: string; conditions: string[]; next_experiment: string };
  latest_experiment?: { id: string; status: string; report: string; audit: string; dataset_audit?: string; independent_audit_pass?: boolean; controlled_rows: number; families: number; implementations_per_family: number; seeds_per_implementation: number; encodings_per_seed: number; pre_transition_worst_seed: number; post_transition_worst_seed: number; pair_flip_worst_seed: number; real_multifamily_gold_rows: number; promotion_blocked: boolean };
};

type Snapshot = {
  schema_version: string;
  generated_at: string;
  execution_location_policy?: { rule?: string; remote_executor?: { address?: string; gpu?: string; cuda_visible_devices?: string; other_gpus?: string }; local_runtime?: { services_allowed?: boolean; training_allowed?: boolean; docker_allowed?: boolean; browser_replay_allowed?: boolean } };
  research_goal: { title: string; objective: string; priority_order: string[]; training_stack: Array<{ stage: string; purpose: string; gate: string; data?: string; preferred_trajectory?: string[]; rejected_trajectory?: string[]; reward_components?: string[] }>; mentor_judge_loop: { role?: string; teacher_actions?: string[]; reference_answer_levels?: string[]; episode_score?: string; graduation_gate?: string }; next_experiment: string; non_goal: string[] };
  source_reports: Array<{ name: string; updated_at: string | null; sha256: string }>;
  judge: { name: string; scope: string; training_promotion_allowed: boolean; memory_promotion_allowed: boolean; vulnerability_claim_allowed: boolean };
  tasks: { all: Task[]; collector: Task[]; reviewer: Task[]; trainer: Task[] };
  process_traces: ProcessTrace[];
  surface_catalog: { manifest_id: string; generated_at: string | null; counts: { routes: number; with_parameter_context: number; parameterized_response_observed: number; training_eligible: number; missing_parameter_context: number }; routes: Array<{ path: string; methods: string[]; query_params: string[]; form_params: string[]; post_form_params: string[]; has_parameter_context: boolean; parameterized_response_observed: boolean; status: string; training_eligible: boolean; evidence_sha256: string }> };
  learning_requirements: LearningRequirements;
  architecture: Array<{ id: string; title: string; subtitle: string; detail: string; owner: string }>;
  capability: {
    metrics: Array<{ id: string; label: string; value: string; status: string; note: string }>;
    model: { selected_hidden_dim: number; adapter_parameter_count: number; route_holdout_pass: boolean; model_input_uses_oracle: boolean; pg257?: { status: string; selected_hidden_dim: number; seed_holdout_rule_accuracy: number; seed_holdout_widebyte_recall: number; seed_holdout_next_token_accuracy: number; record_count: number; promotion_blocked: boolean }; pg258?: { status: string; selected_hidden_dim: number; holdout_rule_accuracy: number; holdout_family_accuracy: number; implementation_ood_family_accuracy: number; record_count: number; canary_pass: boolean; promotion_blocked: boolean }; pg259?: { status: string; selected_hidden_dim: number; fresh_route_rule_accuracy: number; fresh_route_family_accuracy: number; fresh_route_belief_accuracy: number; fresh_route_probe_accuracy: number; implementation_ood_family_accuracy: number; record_count: number; canary_pass: boolean; promotion_blocked: boolean }; pg260?: { status: string; selected_hidden_dim: number; adapter_parameter_count: number; fresh_route_rule_accuracy: number; fresh_route_family_accuracy: number; fresh_route_unknown_abstain_accuracy: number; implementation_ood_family_accuracy: number; record_count: number; canary_pass: boolean; judge_pass: boolean; promotion_blocked: boolean }; pg261?: { status: string; selected_hidden_dim: number; adapter_parameter_count: number; fresh_route_rule_accuracy: number; fresh_route_family_accuracy: number; fresh_route_unknown_abstain_accuracy: number; implementation_ood_family_accuracy: number; record_count: number; canary_pass: boolean; judge_pass: boolean; promotion_blocked: boolean }; pg262?: { status: string; record_count: number; sql_count: number; xss_count: number; audit_complete: boolean; training_eligible: boolean; evidence_hash: string }; pg263?: { status: string; selected_hidden_dim: number; adapter_parameter_count: number; record_count: number; fresh_route_rule_accuracy: number; fresh_route_family_accuracy: number; implementation_ood_family_accuracy: number; canary_pass: boolean; judge_pass: boolean; promotion_blocked: boolean; evidence_hash: string; resource_profile?: Record<string, unknown> }; pg264?: { status: string; record_count: number; audit_record_count: number; sql_count: number; xss_count: number; boolean_count: number; widebyte_count: number; audit_complete: boolean; training_eligible: boolean; evidence_hash: string }; pg265?: { status: string; selected_hidden_dim: number; adapter_parameter_count: number; record_count: number; fresh_route_rule_accuracy: number; fresh_route_family_accuracy: number; judge_pass: boolean; promotion_blocked: boolean; audit_pass: boolean; evidence_hash: string }; pg267?: { status: string; selected_hidden_dim: number; adapter_parameter_count: number; capacity_variants: number[]; record_count: number; pg267_record_count: number; fresh_holdout_rule_accuracy: number; fresh_holdout_family_accuracy: number; route_seed_rule_accuracy: number; implementation_ood_family_accuracy: number; judge_pass: boolean; structural_audit_pass: boolean; canary_pass: boolean; promotion_blocked: boolean; evidence_hash: string }; pg268?: { status: string; surface_count: number; get_count: number; post_count: number; complete_replayed_surface_count: number; unsupported_multipart_count: number; ai_send_count: number; reference_send_count: number; negative_send_count: number; confirmed_local_effect_count: number; abstain_count: number; false_positive_count: number; fresh_reset_count: number; source_attested_count: number; audit_pass: boolean; training_promotion_blocked: boolean; memory_promotion_blocked: boolean; raw_payloads_human_review_only: boolean; oracle_target_in_model_input: boolean; manifest_parameterized_surface_count: number; evidence_hash: string }; pg279?: { status: string; controlled_row_count: number; family_count: number; get_rows: number; post_rows: number; failure_repair_rows: number; typed_effect_rows: number; abstain_rows: number; coarse_conflict_groups: number; enriched_conflict_groups: number; post_conflict_groups: number; family_question_worst_seed: number; retention_status: string; retention_pre_min: number; retention_post_min: number; retention_missing_safe_min: number; operational_audit_pass: boolean; scientific_gate_status: string; real_application_gold_rows: number; promotion_blocked: boolean; evidence_hash: string }; pg280?: { status: string; controlled_row_count: number; family_count: number; shared_slot_token_count: number; conditional_entropy_bits: number; bayes_error_lower_bound: number; final_only_pre_supervision_rows: number; final_only_post_accuracy: number; final_only_ask_rate: number; process_pre_supervision_rows: number; process_post_accuracy: number; process_ask_rate: number; process_safe_rate: number; hard_negative_rows: number; hard_negative_training_eligible: boolean; docker_status: string; scientific_gate_status: string; operational_audit_pass: boolean; promotion_blocked: boolean; real_application_gold_rows: number; evidence_hash: string }; pg281?: { status: string; record_count: number; train_count: number; route_dev_count: number; family_holdout_count: number; hard_negative_count: number; route_positive_recall: number; family_positive_recall: number; route_plan_exact_accuracy: number; family_plan_exact_accuracy: number; hard_negative_safe_reject: number; hard_negative_false_allow: number; literal_payload_generation: boolean; live_send: boolean; docker_status: string; scientific_gate_status: string; operational_audit_pass: boolean; promotion_blocked: boolean; real_application_gold_rows: number; evidence_hash: string } };
    limits: string[];
    next: string;
  };
  instructions: { collector: string; reviewer: string; trainer: string };
};

type Pg282Capability = {
  status: string;
  record_count: number;
  await_evaluator_count: number;
  abstain_count: number;
  confirmed_positive_count: number;
  hard_negative_abstain: boolean;
  remote_docker_status: string;
  operational_audit_pass: boolean;
  literal_payload_generation: boolean;
  live_replay: boolean;
  real_application_gold_rows: number;
  promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg283Capability = {
  status: string;
  train_count: number;
  route_dev_count: number;
  family_holdout_count: number;
  hard_negative_count: number;
  selected_variant: string;
  route_action_safe_exact: number;
  family_action_safe_exact: number;
  hard_negative_safe_reject: number;
  hard_negative_false_allow: number;
  engineering_gate_status: string;
  scientific_gate_status: string;
  remote_docker_status: string;
  live_send: boolean;
  literal_payload_generation: boolean;
  operational_audit_pass: boolean;
  real_application_gold_rows: number;
  promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg284Capability = {
  status: string;
  contract_rows: number;
  blocked_rows: number;
  confirmed_effect_rows: number;
  hard_negative_blocked: boolean;
  engineering_gate_status: string;
  scientific_gate_status: string;
  remote_docker_status: string;
  live_replay: boolean;
  operational_audit_pass: boolean;
  real_application_gold_rows: number;
  promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg285Capability = {
  status: string;
  train_count: number;
  route_dev_count: number;
  family_holdout_count: number;
  hard_negative_count: number;
  selected_variant: string;
  route_sequence_exact: number;
  family_sequence_exact: number;
  route_action_accuracy: number;
  family_action_accuracy: number;
  hard_negative_safe_reject: number;
  hard_negative_false_allow: number;
  engineering_gate_status: string;
  scientific_gate_status: string;
  remote_docker_status: string;
  literal_payload_generation: boolean;
  runtime_canary_placeholder: boolean;
  live_send: boolean;
  operational_audit_pass: boolean;
  real_application_gold_rows: number;
  promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg286Capability = {
  status: string;
  total_rows: number;
  complete_rows: number;
  incomplete_rows: number;
  sql_rows: number;
  xss_rows: number;
  redirect_rows: number;
  hard_negative_rows: number;
  sql_ast_available_rows: number;
  training_eligible_rows: number;
  memory_promotion_allowed_rows: number;
  shared_slot_count: number;
  family_hidden_in_context: boolean;
  oracle_label_in_context: boolean;
  remote_docker_status: string;
  operational_audit_pass: boolean;
  scientific_gate_status: string;
  real_application_gold_rows: number;
  batch_status: string;
  batch_record_count: number;
  batch_training_eligible_rows: number;
  batch_audit_sha256: string;
  promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg287Capability = {
  status: string;
  train_count: number;
  route_dev_count: number;
  family_holdout_count: number;
  hard_negative_count: number;
  ambiguous_count: number;
  resolved_count: number;
  selected_variant: string;
  route_ambiguous_ask_recall: number;
  route_resolved_encoding_accuracy: number;
  family_ambiguous_ask_recall: number;
  family_resolved_encoding_accuracy: number | null;
  family_resolved_encoding_count: number;
  hard_negative_ask_recall: number;
  hard_negative_false_allow: number;
  route_sequence_exact: number;
  family_sequence_exact: number;
  engineering_gate_status: string;
  scientific_gate_status: string;
  remote_docker_status: string;
  live_send: boolean;
  literal_payload_generation: boolean;
  operational_audit_pass: boolean;
  real_application_gold_rows: number;
  promotion_blocked: boolean;
  training_eligible_rows: number;
  live_protocol_status: string;
  live_protocol_sha256: string;
  live_batch_status: string;
  live_batch_record_count: number;
  live_batch_family_resolved_count: number;
  live_batch_blocking_reasons: string[];
  live_batch_audit_sha256: string;
  checkpoint_sha256: string;
  evidence_hash: string;
  promotion_reason: string;
};

type Pg292Capability = {
  status: string;
  mixed_train_count: number;
  counterfactual_train_count: number;
  route_holdout_count: number;
  family_holdout_count: number;
  hard_negative_count: number;
  feature_count: number;
  selected_variant: string;
  selected_threshold: number;
  route_positive_recall: number;
  family_positive_recall: number;
  hard_negative_false_allow: number;
  hard_negative_safe_reject: number;
  engineering_gate_status: string;
  scientific_gate_status: string;
  remote_docker_status: string;
  literal_payload_generation: boolean;
  live_send: boolean;
  operational_audit_pass: boolean;
  real_application_gold_rows: number;
  promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg269Guided = {
  status: string;
  surface_count: number;
  get_count: number;
  post_count: number;
  complete_replayed_surface_count: number;
  initial_confirmed_count: number;
  final_confirmed_count: number;
  repair_attempt_count: number;
  repair_success_count: number;
  abstain_count: number;
  false_positive_count: number;
  fresh_reset_count: number;
  source_attested_count: number;
  audit_pass: boolean;
  context_target_split_pass: boolean;
  training_promotion_blocked: boolean;
  memory_promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg270Ablation = {
  status: string;
  train_count: number;
  route_dev_count: number;
  family_holdout_count: number;
  preference_pair_count: number;
  process_reward_count: number;
  guided_token_accuracy: number;
  guided_next_action_accuracy: number;
  guided_preference_win_rate: number;
  audit_pass: boolean;
  cuda_assignment?: { device_name?: string; cuda_visible_devices?: string; visible_device_count?: number; current_device?: number | null };
  training_promotion_blocked: boolean;
  memory_promotion_blocked: boolean;
  evidence_hash: string;
};

type Pg271Replay = {
  status: string;
  fresh_seed: number;
  surface_count: number;
  next_action_accuracy: number;
  final_belief_accuracy: number;
  abstain_calibration_accuracy: number;
  family_holdout_count: number;
  family_next_action_accuracy: number;
  family_final_belief_accuracy: number;
  family_abstain_calibration_accuracy: number;
  unsupported_positive_count: number;
  audit_pass: boolean;
  training_promotion_blocked: boolean;
  memory_promotion_blocked: boolean;
  vulnerability_claim_blocked: boolean;
  evidence_hash: string;
};

const fallback: Snapshot = {
  schema_version: "sift-research-ops-snapshot-v1",
  generated_at: "",
  execution_location_policy: { rule: "本地不运行实验；等待远程执行策略。", remote_executor: {}, local_runtime: { services_allowed: false, training_allowed: false, docker_allowed: false, browser_replay_allowed: false } },
  research_goal: { title: "", objective: "", priority_order: [], training_stack: [], mentor_judge_loop: {}, next_experiment: "", non_goal: [] },
  source_reports: [],
  judge: { name: "SIFT final judge", scope: "loopback-only", training_promotion_allowed: false, memory_promotion_allowed: false, vulnerability_claim_allowed: false },
  tasks: { all: [], collector: [], reviewer: [], trainer: [] },
  process_traces: [],
  surface_catalog: { manifest_id: "", generated_at: null, counts: { routes: 0, with_parameter_context: 0, parameterized_response_observed: 0, training_eligible: 0, missing_parameter_context: 0 }, routes: [] },
  learning_requirements: {
    schema_version: "sift-learning-requirements-v1",
    title: "等待数据任务书",
    principle: "API 就绪后显示采集与验收要求。",
    evidence: { source_report: "", source_audit: "", report_ready: false, audit_pass: false, controlled_rows: 0, real_multifamily_gold_rows: 0, coarse_conflict_groups: 0, coarse_conflicting_rows: 0, process_question_recovery_worst_seed: 0, conservative_question_recovery_worst_seed: 0, dpo_question_recovery_worst_seed: 0, claim: "" },
    findings: [], queues: [], record_contract: [], resources: [], forbidden: [],
    promotion_gate: { current_status: "BLOCKED", conditions: [], next_experiment: "等待 API" },
  },
  architecture: [],
  capability: { metrics: [], model: { selected_hidden_dim: 0, adapter_parameter_count: 0, route_holdout_pass: false, model_input_uses_oracle: false, pg257: undefined, pg258: undefined, pg259: undefined, pg260: undefined, pg261: undefined, pg262: undefined, pg263: undefined, pg264: undefined, pg265: undefined, pg267: undefined, pg268: undefined }, limits: [], next: "等待 API" },
  instructions: { collector: "", reviewer: "", trainer: "" },
};

const payloadFallback: PayloadReview = {
  schema_version: "sift-review-payloads-v1",
  generated_at: "",
  review_only: true,
  persisted: false,
  training_eligible: false,
  target_scope: { kind: "loopback_only", allowed_origins: ["127.0.0.1", "localhost"], arbitrary_target_input: false },
  safety: {},
  disclaimer: "等待本地 payload 审查接口。",
  entries: [],
};

async function loadSnapshot(): Promise<Snapshot> {
  const response = await fetch("/api/research/ops", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<Snapshot>;
}

async function loadPayloadReview(): Promise<PayloadReview> {
  const response = await fetch("/api/research/ops/payloads", { cache: "no-store" });
  if (!response.ok) throw new Error(`payload review HTTP ${response.status}`);
  return response.json() as Promise<PayloadReview>;
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    needs_oracle: "待补 oracle",
    needs_review: "待复核",
    ready_for_review: "可复核",
    promotion_blocked: "晋级冻结",
  };
  return labels[status] || status;
}

function roleLabel(role: string) {
  return role === "collector" ? "采集员" : role === "reviewer" ? "复核员" : "训练员";
}

function OpsHeader({ view, online, updated, onRefresh }: { view: View; online: boolean; updated: string; onRefresh: () => void }) {
  const links: Array<[View, string, string]> = [["tasks", "任务台", "/ops"], ["architecture", "架构理念", "/ops/architecture"], ["capability", "能力证据", "/ops/capability"]];
  return (
    <header className="ops-header">
      <a className="ops-brand" href="/ops"><span>Σ</span><div><b>SIFT / OPS</b><small>RESEARCH CONTROL PLANE</small></div></a>
      <nav className="ops-nav">{links.map(([id, label, href]) => <a className={view === id ? "active" : ""} href={href} key={id}>{label}</a>)}</nav>
      <div className="ops-header-actions"><span className={`ops-live ${online ? "online" : ""}`}><i />{online ? "LIVE" : "OFFLINE"}</span><button className="ops-refresh" onClick={onRefresh}>刷新 {updated ? new Date(updated).toLocaleTimeString("zh-CN", { hour12: false }) : ""}</button></div>
    </header>
  );
}

function MetricStrip({ snapshot }: { snapshot: Snapshot }) {
  const metrics = snapshot.capability.metrics;
  return <div className="ops-metric-strip">{metrics.map((metric) => <article key={metric.id} className={`ops-metric ${metric.status}`}><span>{metric.label}</span><strong>{metric.value}</strong><small>{metric.note}</small></article>)}</div>;
}

function TaskCard({ task, selected, onClick }: { task: Task; selected: boolean; onClick: () => void }) {
  return <button className={`ops-task-card ${selected ? "selected" : ""} ${task.human_required ? "escalated" : ""}`} onClick={onClick}>
    <div className="ops-task-top"><span className={`ops-role role-${task.role}`}>{roleLabel(task.role)}</span><em>{statusLabel(task.status)}</em></div>
    <b>{task.route}</b>
    <p>{task.instruction}</p>
    <div className="ops-task-foot"><span>seed {task.seed}</span><span>{task.owner}</span><span>{task.evidence_hash || "no hash"}</span></div>
  </button>;
}

function TaskInspector({ task }: { task?: Task }) {
  if (!task) return <div className="ops-empty"><span>INSPECTOR</span><h3>选择一条任务</h3><p>AI 默认执行本地采集、复核和训练准备；只有 scope、oracle 或证据缺失时才升级人工。</p></div>;
  return <div className="ops-inspector-content">
    <div className="ops-inspector-head"><div><span>{roleLabel(task.role)} / {task.id}</span><h2>{task.label}</h2></div><strong className={task.confirmed_positive ? "pass" : task.human_required ? "warn" : "pending"}>{task.confirmed_positive ? "CONFIRMED LOCAL" : task.human_required ? "ESCALATE" : "IN PROGRESS"}</strong></div>
    <dl className="ops-facts"><div><dt>Route</dt><dd>{task.route}</dd></div><div><dt>Owner</dt><dd>{task.owner}</dd></div><div><dt>AI typed effect</dt><dd>{task.typed_effect ? "yes" : "no"}</dd></div><div><dt>Human required</dt><dd>{task.human_required ? "yes" : "no"}</dd></div><div><dt>Evidence hash</dt><dd>{task.evidence_hash || "missing"}</dd></div><div><dt>Raw material</dt><dd>{task.raw_material_available ? "restricted" : "ephemeral only"}</dd></div></dl>
    <div className="ops-instruction"><span>NEXT ACTION</span><p>{task.instruction}</p></div>
    <div className="ops-reasons"><span>判官原因</span>{task.reasons.length ? task.reasons.map((reason) => <code key={reason}>{reason}</code>) : <code>all registered gates passed</code>}</div>
  </div>;
}

function PayloadRequest({ label, request }: { label: string; request?: RequestShape }) {
  if (!request) return <div className="ops-wire-card missing"><div className="ops-wire-label"><span>{label}</span><em>未生成</em></div><p>AI 当前没有声称生成这条请求。</p></div>;
  return <div className="ops-wire-card"><div className="ops-wire-label"><span>{label}</span><em>可读 wire</em></div><pre>{request.wire}</pre><details><summary>查看编码前后</summary><div className="ops-wire-values"><div><small>逻辑值</small><code>{JSON.stringify(request.logical_values, null, 2)}</code></div><div><small>线上编码</small><code>{JSON.stringify(request.encoded_values, null, 2)}</code></div></div></details></div>;
}

function PayloadChannelView({ title, channel }: { title: string; channel: PayloadChannel }) {
  if (channel.true || channel.false) return <div className="ops-channel"><div className="ops-channel-head"><b>{title}</b><span>{channel.status}</span></div><PayloadRequest label="TRUE branch" request={channel.true} /><PayloadRequest label="FALSE branch" request={channel.false} /></div>;
  return <div className="ops-channel"><div className="ops-channel-head"><b>{title}</b><span>{channel.status}</span></div><PayloadRequest label="request" request={channel.request} /></div>;
}

function OracleEvidenceView({ evidence }: { evidence: PayloadEntry["oracle_evidence"] }) {
  const projection = (label: string, value?: Record<string, unknown> | null) => {
    const bounded = value && typeof value === "object" ? value : {};
    return <div className="ops-evidence-projection"><small>{label}</small><code>{Object.keys(bounded).length ? JSON.stringify(bounded, null, 2) : "projection unavailable"}</code></div>;
  };
  return <div className={`ops-oracle-evidence ${evidence.matched ? "confirmed" : "gap"}`}><div className="ops-evidence-head"><span>RUNTIME ORACLE / BOUNDED EVIDENCE</span><b>{evidence.matched ? "MATCHED" : "NOT CONFIRMED"}</b></div><div className="ops-evidence-grid"><span>source <b>{evidence.source_report || "not run"}</b></span><span>seed <b>{evidence.seed || "—"}</b></span><span>pattern <b>{evidence.pattern_id}</b></span><span>match count <b>{evidence.match_count ?? "—"}</b></span><span>candidate↔reference <b>{evidence.candidate_reference_agreement == null ? "—" : String(evidence.candidate_reference_agreement)}</b></span><span>negative clean <b>{evidence.negative_clean == null ? "—" : String(evidence.negative_clean)}</b></span><span>evidence SHA-256 <b>{evidence.evidence_sha256 || "missing"}</b></span></div><div className="ops-evidence-projections">{projection("oracle facts", evidence.oracle_facts)}{projection("candidate response projection", evidence.candidate_projection)}{projection("reference response projection", evidence.reference_projection)}{projection("negative response projection", evidence.negative_projection)}{projection("candidate TRUE branch", evidence.candidate_true_projection)}{projection("candidate FALSE branch", evidence.candidate_false_projection)}</div>{evidence.reasons.length > 0 && <div className="ops-evidence-reasons">{evidence.reasons.map((reason) => <code key={reason}>{reason}</code>)}</div>}<p>这是可复核的状态/形状/匹配投影，不是响应正文；raw payload/response body 不会写入训练集或长期记忆。</p></div>;
}

function PayloadReviewPanel({ payloads }: { payloads: PayloadReview }) {
  const [selectedId, setSelectedId] = useState<string>();
  const [family, setFamily] = useState("all");
  const families = useMemo(() => [...new Set(payloads.entries.map((entry) => entry.family))], [payloads.entries]);
  const filtered = family === "all" ? payloads.entries : payloads.entries.filter((entry) => entry.family === family);
  const selected = filtered.find((entry) => entry.id === selectedId) || filtered[0];
  useEffect(() => { if (selected && !filtered.some((entry) => entry.id === selectedId)) setSelectedId(selected.id); }, [selected, selectedId, filtered]);
  return <section id="payload-review" className="ops-wrap ops-payload-review"><div className="ops-section-head"><div><span>PAYLOAD REVIEW / HUMAN VISIBLE</span><h2>AI 到底发了什么</h2></div><div className="ops-review-badges"><b>REVIEW ONLY</b><b>LOOPBACK</b><b>NOT TRAINING</b><b>ORACLE EVIDENCE</b></div></div><p className="ops-review-note">{payloads.disclaimer} 这里展示 AI candidate、reference、negative 三组完整 GET/POST wire 与编码，下面再展开真实回放的响应投影、匹配模式、计数和证据哈希；没有 oracle 就明确显示未确认。</p><div className="ops-payload-layout"><div className="ops-payload-list"><div className="ops-filter ops-family-filter"><button className={family === "all" ? "active" : ""} onClick={() => setFamily("all")}>全部</button>{families.map((item) => <button className={family === item ? "active" : ""} onClick={() => setFamily(item)} key={item}>{item}</button>)}</div>{filtered.map((entry) => <button className={`ops-payload-row ${selected?.id === entry.id ? "selected" : ""}`} key={entry.id} onClick={() => setSelectedId(entry.id)}><span>{entry.family}</span><b>{entry.method} {entry.route}</b><em>{entry.validation_status}</em></button>)}</div>{selected ? <div className="ops-payload-inspector"><div className="ops-payload-title"><div><span>{selected.family.toUpperCase()} / {selected.id}</span><h3>{selected.method} {selected.route}</h3></div><strong className={selected.validation_status === "validated_local_effect" ? "pass" : "warn"}>{selected.validation_status}</strong></div><div className="ops-payload-facts"><span>fields <b>{selected.fields.join(" · ")}</b></span><span>oracle <b>{selected.expected_oracle}</b></span><span>source <b>{selected.source}</b></span></div><div className="ops-payload-channels"><PayloadChannelView title="AI candidate" channel={selected.ai} /><PayloadChannelView title="Reference" channel={selected.reference} /><PayloadChannelView title="Negative control" channel={selected.negative} /></div><OracleEvidenceView evidence={selected.oracle_evidence} /><div className="ops-effect-card"><span>判定说明</span><p>{selected.effect_claim}</p>{selected.notes.map((note) => <code key={note}>{note}</code>)}</div></div> : <div className="ops-empty"><h3>没有 payload</h3><p>接口返回空集时不伪造能力。</p></div>}</div></section>;
}

function SurfaceCoveragePanel({ catalog }: { catalog: Snapshot["surface_catalog"] }) {
  const incomplete = catalog.routes.filter((route) => !route.parameterized_response_observed).slice(0, 18);
  const observed = catalog.counts.parameterized_response_observed;
  const contextual = catalog.counts.with_parameter_context;
  return <section className="ops-wrap ops-surface-catalog"><div className="ops-section-head"><div><span>SURFACE CATALOG / PARAMETER GROUNDING</span><h2>先补齐 GET / POST，再谈训练</h2></div><span className="ops-chip">{catalog.counts.routes} routes</span></div><p className="ops-review-note">浏览器爬取只提供基线时，不能当作训练样本；这里把缺参数上下文和未完成参数化回放的真实清单摊开。</p><div className="ops-surface-stats"><b>{contextual}<small>有参数上下文</small></b><b>{observed}<small>已参数化回放</small></b><b>{catalog.counts.missing_parameter_context}<small>缺参数上下文</small></b><b>{catalog.counts.training_eligible}<small>可训练</small></b></div><div className="ops-surface-list">{incomplete.length ? incomplete.map((route) => <div className="ops-surface-row" key={`${route.path}-${route.methods.join("-")}-${route.query_params.join("-")}-${route.form_params.join("-")}-${route.post_form_params.join("-")}`}><b>{route.path}</b><span>{route.methods.join(" / ") || "GET"} · {route.query_params.concat(route.form_params, route.post_form_params).join(", ") || "参数未观测"}</span><em>{route.status}</em></div>) : <div className="ops-empty"><h3>参数化回放已覆盖</h3><p>当前 manifest 没有待补齐的路由。</p></div>}</div></section>;
}

function DataMissionPanel({ brief }: { brief: LearningRequirements }) {
  const [selectedId, setSelectedId] = useState<string>();
  const selected = brief.queues.find((queue) => queue.id === selectedId) || brief.queues[0];
  const percent = (value: number) => `${(value * 100).toFixed(0)}%`;
  return <section id="data-mission" className="ops-wrap ops-data-mission">
    <div className="ops-section-head"><div><span>DATA MISSION / PG-280 EVIDENCE BACKED</span><h2>{brief.title}</h2></div><div className="ops-review-badges"><b>{brief.evidence.audit_pass ? "AUDIT PASS" : "AUDIT WAITING"}</b><b>REAL GOLD {brief.evidence.real_multifamily_gold_rows}</b><b className="blocked">PROMOTION {brief.promotion_gate.current_status}</b></div></div>
    <p className="ops-mission-principle">{brief.principle}</p>
    {brief.latest_experiment && <div className="ops-latest-experiment"><div><span>LATEST FILLED DATA / {brief.latest_experiment.id}</span><strong>{brief.latest_experiment.status.toUpperCase()}</strong><p>{brief.latest_experiment.controlled_rows} 条受控记录 · {brief.latest_experiment.families} 族 · 每族 {brief.latest_experiment.implementations_per_family} 实现 · {brief.latest_experiment.seeds_per_implementation} seed · {brief.latest_experiment.encodings_per_seed} 编码</p></div><div><b>REAL GOLD {brief.latest_experiment.real_multifamily_gold_rows}</b><small>{brief.latest_experiment.independent_audit_pass ? "independent audit pass · promotion blocked" : "independent audit waiting"}</small></div></div>}
    <div className="ops-mission-evidence">
      <article><span>CONTROLLED ROWS</span><strong>{brief.evidence.controlled_rows}</strong><p>只作受控假设证据</p></article>
      <article className="warn"><span>INPUT COLLISIONS</span><strong>{brief.evidence.coarse_conflicting_rows}</strong><p>{brief.evidence.coarse_conflict_groups} 个冲突组</p></article>
      <article className="warn"><span>PROCESS QUESTION / WORST</span><strong>{percent(brief.evidence.process_question_recovery_worst_seed)}</strong><p>会 ASK，但未必问对</p></article>
      <article className="pass"><span>CONSERVATIVE / WORST</span><strong>{percent(brief.evidence.conservative_question_recovery_worst_seed)}</strong><p>本受控族三 seed</p></article>
      <article className="warn"><span>DPO QUESTION / WORST</span><strong>{percent(brief.evidence.dpo_question_recovery_worst_seed)}</strong><p>平均分会掩盖退化</p></article>
      {typeof brief.evidence.pg278_post_transition_worst_seed === "number" && <article className="pass"><span>PG-278 POST / WORST</span><strong>{percent(brief.evidence.pg278_post_transition_worst_seed)}</strong><p>收到观测后的修复转移</p></article>}
      {typeof brief.evidence.pg278_pair_flip_worst_seed === "number" && <article className="pass"><span>PG-278 PAIR FLIP</span><strong>{percent(brief.evidence.pg278_pair_flip_worst_seed)}</strong><p>正/负反事实同时复核</p></article>}
      {typeof brief.evidence.pg278_post_conflict_groups === "number" && <article className={brief.evidence.pg278_post_conflict_groups === 0 ? "pass" : "warn"}><span>POST COLLISIONS</span><strong>{brief.evidence.pg278_post_conflict_groups}</strong><p>必须为 0 才能训练</p></article>}
      {typeof brief.evidence.pg279_get_rows === "number" && <article className="pass"><span>PG-279 GET / POST</span><strong>{brief.evidence.pg279_get_rows} / {brief.evidence.pg279_post_rows}</strong><p>真实远程 loopback 传输投影</p></article>}
      {typeof brief.evidence.pg279_failure_repair_rows === "number" && <article className="pass"><span>FAILURE → REPAIR</span><strong>{brief.evidence.pg279_failure_repair_rows}</strong><p>typed {brief.evidence.pg279_typed_effect_rows} · abstain {brief.evidence.pg279_abstain_rows}</p></article>}
      {typeof brief.evidence.pg279_family_question_worst_seed === "number" && <article className="warn"><span>FAMILY HOLDOUT / WORST</span><strong>{percent(brief.evidence.pg279_family_question_worst_seed)}</strong><p>科学 gate {brief.evidence.pg279_gate_status}</p></article>}
      {typeof brief.evidence.pg279_retention_post_min === "number" && <article className="pass"><span>RETENTION CANARY</span><strong>{percent(brief.evidence.pg279_retention_post_min)}</strong><p>pre {percent(brief.evidence.pg279_retention_pre_min ?? 0)} · missing-safe {percent(brief.evidence.pg279_retention_missing_safe_min ?? 0)}</p></article>}
      {typeof brief.evidence.pg280_conditional_entropy_bits === "number" && <article className="warn"><span>PG-280 IDENTIFIABILITY</span><strong>{brief.evidence.pg280_conditional_entropy_bits.toFixed(1)} bit</strong><p>Bayes error 下界 {percent(brief.evidence.pg280_bayes_error_lower_bound ?? 0)}；缺观测不可硬猜</p></article>}
      {typeof brief.evidence.pg280_final_only_post_accuracy === "number" && <article className="warn"><span>FINAL-ONLY / PROCESS</span><strong>{percent(brief.evidence.pg280_final_only_post_accuracy)} / {percent(brief.evidence.pg280_process_post_accuracy ?? 0)}</strong><p>final-only pre/ASK {brief.evidence.pg280_final_only_pre_supervision_rows ?? 0} / {percent(brief.evidence.pg280_final_only_ask_rate ?? 0)} · process ASK {percent(brief.evidence.pg280_process_ask_rate ?? 0)}</p></article>}
      {typeof brief.evidence.pg280_hard_negative_rows === "number" && <article className="warn"><span>OOD HARD NEGATIVE</span><strong>{brief.evidence.pg280_hard_negative_rows}</strong><p>evaluation-only · Docker {brief.evidence.pg280_docker_status}</p></article>}
      {typeof brief.evidence.pg281_route_positive_recall_min === "number" && <article className="warn"><span>PG-281 ABSTRACT PLAN</span><strong>{percent(brief.evidence.pg281_route_positive_recall_min)} / {percent(brief.evidence.pg281_family_positive_recall_min ?? 0)}</strong><p>route / family 正例 replay recall；literal payload 不进模型</p></article>}
      {typeof brief.evidence.pg281_hard_negative_reject_min === "number" && <article className="pass"><span>SAFE SEND GATE</span><strong>{percent(brief.evidence.pg281_hard_negative_reject_min)}</strong><p>hard-negative reject · false-allow {brief.evidence.pg281_hard_negative_false_allow_max ?? 0} · Docker {brief.evidence.pg281_docker_status}</p></article>}
    </div>
    <div className="ops-finding-grid">{brief.findings.map((finding) => <article key={finding.id}><div><b>{finding.severity}</b><span>{finding.id}</span></div><h3>{finding.title}</h3><p>{finding.evidence}</p><code>NEXT → {finding.action}</code></article>)}</div>
    <div className="ops-mission-layout">
      <div className="ops-mission-list"><div className="ops-mission-list-head"><span>COLLECTION QUEUE</span><b>{brief.queues.length} 个数据包</b></div>{brief.queues.map((queue, index) => <button key={queue.id} className={selected?.id === queue.id ? "selected" : ""} onClick={() => setSelectedId(queue.id)}><i>{String(index + 1).padStart(2, "0")}</i><div><span>{queue.priority} · {queue.owner}</span><b>{queue.title}</b><small>{queue.current}</small></div><em>{queue.status}</em></button>)}</div>
      {selected ? <div className="ops-mission-inspector"><div className="ops-mission-title"><div><span>{selected.id} / {selected.priority}</span><h3>{selected.title}</h3></div><strong>{selected.owner}</strong></div><div className="ops-mission-why"><span>为什么采</span><p>{selected.why}</p></div><div className="ops-mission-columns"><article><span>最低配额</span>{selected.minimum_quota.map((item) => <p key={item}><i>Q</i>{item}</p>)}</article><article><span>必须采集</span>{selected.collect.map((item) => <p key={item}><i>+</i>{item}</p>)}</article><article><span>验收硬门</span>{selected.acceptance.map((item) => <p key={item}><i>✓</i>{item}</p>)}</article></div><div className="ops-mission-output"><span>OUTPUT LANE</span><b>{selected.output_lane}</b><small>防止：{selected.prevents.join(" · ")}</small></div></div> : <div className="ops-empty"><h3>等待任务</h3><p>API 没有返回采集队列。</p></div>}
    </div>
    <div className="ops-contract-grid">
      <article><div className="ops-contract-title"><span>RECORD CONTRACT</span><h3>每条记录必须长什么样</h3></div>{brief.record_contract.map((group) => <details key={group.group}><summary>{group.group}<b>{group.fields.length} fields</b></summary><p>{group.fields.join(" · ")}</p><code>{group.rule}</code></details>)}</article>
      <article><div className="ops-contract-title"><span>RESOURCE MANIFEST</span><h3>要准备哪些资源</h3></div>{brief.resources.map((resource) => <details key={resource.category}><summary>{resource.category}<b>{resource.required ? "REQUIRED" : "OPTIONAL"}</b></summary><p>{resource.items.join(" · ")}</p><code>{resource.why}</code></details>)}</article>
    </div>
    <div className="ops-dirty-data"><div className="ops-contract-title"><span>DIRTY DATA / NEVER TRAIN</span><h3>这些东西不许混进训练</h3></div><div>{brief.forbidden.map((item) => <article key={item.id}><span>{item.lane}</span><b>{item.title}</b><p>{item.reason}</p></article>)}</div></div>
    <div className="ops-promotion-gate"><div><span>PROMOTION GATE</span><strong>{brief.promotion_gate.current_status}</strong><p>{brief.evidence.claim}</p></div><div>{brief.promotion_gate.conditions.map((condition) => <p key={condition}><i>○</i>{condition}</p>)}</div><code>NEXT → {brief.promotion_gate.next_experiment}</code></div>
  </section>;
}

function TasksView({ snapshot, payloads }: { snapshot: Snapshot; payloads: PayloadReview }) {
  const [filter, setFilter] = useState<"all" | "collector" | "reviewer" | "trainer">("all");
  const [selectedId, setSelectedId] = useState<string>();
  const tasks = snapshot.tasks[filter] || [];
  const selected = tasks.find((task) => task.id === selectedId) || tasks[0];
  const humanCount = snapshot.tasks.all.filter((task) => task.human_required).length;
  return <>
    <section className="ops-hero"><div><span className="ops-overline">SIFT / HUMAN-AI HANDOFF</span><h1>把研究流程<br /><em>交给任务。</em></h1><p>AI 先负责采集、复核和训练准备；页面只把真正缺 oracle、缺 scope 或缺复现证据的任务升级给人。</p></div><div className="ops-judge-card"><span>FINAL JUDGE</span><strong>LOCAL ONLY</strong><p>{snapshot.judge.scope}</p><div><b>{snapshot.tasks.all.length}</b><small>live tasks</small><b>{humanCount}</b><small>human escalations</small></div></div></section>
    {snapshot.execution_location_policy && <section className="ops-execution-banner"><div><span>EXECUTION LOCATION / HARD RULE</span><strong>REMOTE A800 ONLY</strong><p>{snapshot.execution_location_policy.rule}</p></div><div><b>{snapshot.execution_location_policy.remote_executor?.address || "remote executor"}</b><small>{snapshot.execution_location_policy.remote_executor?.gpu || "A800 GPU0"} · CUDA_VISIBLE_DEVICES={snapshot.execution_location_policy.remote_executor?.cuda_visible_devices || "0"}</small><small>{snapshot.execution_location_policy.remote_executor?.other_gpus || "other GPUs untouched"}</small></div><div><b>LOCAL SERVICES OFF</b><small>训练 / Docker / 浏览器回放均禁用</small></div></section>}
    <DataMissionPanel brief={snapshot.learning_requirements} />
    <div className="ops-wrap"><MetricStrip snapshot={snapshot} /></div>
    <section className="ops-wrap ops-task-layout"><div className="ops-task-board"><div className="ops-section-head"><div><span>WORK QUEUE / AI FIRST</span><h2>今日任务</h2></div><div className="ops-filter">{(["all", "collector", "reviewer", "trainer"] as const).map((id) => <button className={filter === id ? "active" : ""} key={id} onClick={() => setFilter(id)}>{id === "all" ? "全部" : roleLabel(id)}</button>)}</div></div><div className="ops-task-list">{tasks.map((task) => <TaskCard key={task.id} task={task} selected={task.id === selected?.id} onClick={() => setSelectedId(task.id)} />)}{!tasks.length && <div className="ops-empty"><h3>暂无任务</h3><p>新的实验报告写入后，刷新页面即可生成任务。</p></div>}</div></div><aside className="ops-inspector"><TaskInspector task={selected} /></aside></section>
     <section className="ops-wrap ops-instruction-grid"><article><span>COLLECTOR</span><h3>采集员</h3><p>{snapshot.instructions.collector}</p><b>AI 默认执行 · 缺字段才升级</b></article><article><span>REVIEWER</span><h3>复核员</h3><p>{snapshot.instructions.reviewer}</p><b>最终判定永远独立于模型</b></article><article><span>TRAINER</span><h3>训练员</h3><p>{snapshot.instructions.trainer}</p><b>没有 gold 就不晋级</b></article></section>
     <SurfaceCoveragePanel catalog={snapshot.surface_catalog} />
     <PayloadReviewPanel payloads={payloads} />
  </>;
}

function ArchitectureView({ snapshot }: { snapshot: Snapshot }) {
  const [selected, setSelected] = useState(snapshot.architecture[0]?.id || "surface");
  const active = snapshot.architecture.find((layer) => layer.id === selected) || snapshot.architecture[0];
  const composition = (snapshot.research_goal as Snapshot["research_goal"] & { question_composition_loop?: { principle?: string; loop?: string[]; success_definition?: string } }).question_composition_loop;
  return <>
    <section className="ops-subhero"><span className="ops-overline">MODEL DESIGN / RULE-MAZE</span><h1>不是一条黑盒链，<br /><em>是一条可回放的证据链。</em></h1><p>每一层都只接收它应该接收的 token；模型可以提出下一步，但不能替代独立判官。</p></section>
    <section className="ops-wrap ops-goal-card"><div className="ops-section-head"><div><span>REDEFINED RESEARCH GOAL / V2</span><h2>{snapshot.research_goal.title || "可泛化验证智能"}</h2></div><span className="ops-chip">导师 → SFT → PREFERENCE → RL</span></div><p>{snapshot.research_goal.objective}</p><div className="ops-goal-priorities">{snapshot.research_goal.priority_order.map((item, index) => <span key={item}><b>{String(index + 1).padStart(2, "0")}</b>{item}</span>)}</div><div className="ops-goal-mentor"><span>MENTOR / JUDGE LOOP</span><p>{snapshot.research_goal.mentor_judge_loop.role || "导师提供参考策略，独立 oracle 负责复放判定。"}</p><code>{snapshot.research_goal.mentor_judge_loop.episode_score || "步骤分 + fresh replay + 误报惩罚"}</code></div>{composition?.principle && <div className="ops-goal-mentor"><span>QUESTION → COMPOSE → VERIFY</span><p>{composition.principle}</p><code>{composition.success_definition || "疑问、失败、修复和 verdict 成对保存"}</code></div>}<code>下一实验：{snapshot.research_goal.next_experiment}</code></section>
    <section className="ops-wrap ops-architecture"><div className="ops-section-head"><div><span>PIPELINE / CLICK A LAYER</span><h2>从页面到记忆</h2></div><span className="ops-chip">oracle-out-of-input</span></div><div className="ops-flow">{snapshot.architecture.map((layer, index) => <button className={`ops-flow-node ${layer.id === selected ? "active" : ""}`} key={layer.id} onClick={() => setSelected(layer.id)}><i>{String(index + 1).padStart(2, "0")}</i><b>{layer.title}</b><small>{layer.subtitle}</small>{index < snapshot.architecture.length - 1 && <span className="ops-arrow">→</span>}</button>)}</div>{active && <div className="ops-layer-detail"><div><span>SELECTED LAYER / {active.owner.toUpperCase()}</span><h2>{active.title}</h2><p>{active.detail}</p></div><pre>{JSON.stringify({ layer: active.id, owner: active.owner, input: active.id === "oracle" ? "bounded evidence projection" : "abstract tokens", output: active.id === "memory" ? "versioned dataset lane" : "next transition" }, null, 2)}</pre></div>}</section>
    <section className="ops-wrap ops-principles"><article><span>01 / CAUSALITY</span><h3>失败信息先进入 belief</h3><p>模型不是靠一次猜测闭环，而是用失败阶段、环境归因和最小修复改变下一步 token 权重。</p></article><article><span>02 / GENERALIZATION</span><h3>Route 只是外壳</h3><p>训练和留出按 seed、route、family、implementation 分隔；抽象 Rule IR 才是可迁移的核心。</p></article><article><span>03 / HONESTY</span><h3>oracle 不进模型输入</h3><p>response/AST/DOM evaluator 只在发送后负责判定，避免标签泄漏和自我安慰。</p></article></section>
  </>;
}

function ProcessReplayPanel({ traces }: { traces: ProcessTrace[] }) {
  const [selectedId, setSelectedId] = useState<string>();
  const [family, setFamily] = useState("all");
  const families = useMemo(() => [...new Set(traces.map((trace) => trace.family))], [traces]);
  const filtered = family === "all" ? traces : traces.filter((trace) => trace.family === family);
  const selected = filtered.find((trace) => trace.id === selectedId) || filtered[0];
  useEffect(() => { if (selected && !filtered.some((trace) => trace.id === selectedId)) setSelectedId(selected.id); }, [filtered, selected, selectedId]);
  return <section id="process-replay" className="ops-wrap ops-process-replay"><div className="ops-section-head"><div><span>AI PROCESS REPLAY / LOCAL LAB ONLY</span><h2>看见 AI 怎么走迷宫</h2></div><div className="ops-review-badges"><b>FRESH RESET</b><b>GET / POST</b><b>ORACLE AFTER SEND</b></div></div><p className="ops-review-note">这里不是把结果分数包装成“攻击成功”，而是逐步展示真实本地回放：AI 先观察并选抽象探针，随后由独立 reference、negative 和 typed oracle 决定下一步。</p><div className="ops-process-layout"><div className="ops-process-list"><div className="ops-filter ops-family-filter"><button className={family === "all" ? "active" : ""} onClick={() => setFamily("all")}>全部</button>{families.map((item) => <button className={family === item ? "active" : ""} onClick={() => setFamily(item)} key={item}>{item}</button>)}</div>{filtered.map((trace) => <button className={`ops-process-row ${selected?.id === trace.id ? "selected" : ""}`} key={trace.id} onClick={() => setSelectedId(trace.id)}><span>{trace.family} · seed {trace.seed}</span><b>{trace.route}</b><em className={trace.confirmed_positive ? "pass" : "warn"}>{trace.confirmed_positive ? "LOCAL CONFIRMED" : trace.oracle_available ? "ABSTAIN / REVIEW" : "ORACLE GAP"}</em></button>)}</div>{selected ? <div className="ops-process-inspector"><div className="ops-process-meta"><div><span>{selected.family.toUpperCase()} / seed {selected.seed}</span><h3>{selected.route}</h3></div><strong className={selected.confirmed_positive ? "pass" : "warn"}>{selected.confirmed_positive ? "CONFIRMED LOCAL" : "NOT CONFIRMED"}</strong></div><div className="ops-process-facts"><span>AI <b>{selected.ai_sent ? "sent" : "abstain"}</b></span><span>reference <b>{selected.reference_sent ? "sent" : "missing"}</b></span><span>negative <b>{selected.negative_sent ? "sent" : "missing"}</b></span><span>fresh <b>{selected.fresh_reset ? "yes" : "no"}</b></span><span>probe <b>{selected.abstract_probe}</b></span><span>evidence <b>{selected.evidence_hash || "missing"}</b></span></div><div className="ops-process-timeline">{selected.stages.map((stage, index) => <div className={`ops-process-step ${stage.state}`} key={stage.id}><i>{String(index + 1).padStart(2, "0")}</i><div><b>{stage.label}</b><p>{stage.detail}</p></div>{index < selected.stages.length - 1 && <span className="ops-process-arrow">→</span>}</div>)}</div><div className="ops-effect-card"><span>人工复核动作</span><p>先在 <a href="/ops#payload-review">Payload Review</a> 查看三组 wire，再按 evidence hash 回到对应报告；没有 typed oracle 就保持 abstain。</p></div></div> : <div className="ops-empty"><h3>暂无回放</h3><p>等待本地报告。</p></div>}</div></section>;
}

function LatestExperimentPanel({ snapshot, pg263Ready }: { snapshot: Snapshot; pg263Ready: boolean }) {
  const traces = snapshot.capability.model.pg262;
  const model = snapshot.capability.model.pg263;
  const growth = snapshot.capability.model.pg264;
  const large = snapshot.capability.model.pg265;
  const grounded = snapshot.capability.model.pg267;
  const replay = snapshot.capability.model.pg268;
  const guided = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg269?: Pg269Guided }).pg269;
  const teacher = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg270?: Pg270Ablation }).pg270;
  const fresh = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg271?: Pg271Replay }).pg271;
  const hypothesis = snapshot.capability.model as Snapshot["capability"]["model"] & { pg272?: any; pg274_score_rl?: any; pg275_hypothesis_ablation?: any; pg276_third_implementation?: any; pg277_question_composition?: any };
  const remote = snapshot.capability.model.pg279;
  type RemoteOntology = NonNullable<Snapshot["capability"]["model"]["pg280"]> & { remote_adapter_status?: string; remote_adapter_audit_pass?: boolean };
  const ontology = snapshot.capability.model.pg280 as RemoteOntology | undefined;
  type PayloadPolicy = NonNullable<Snapshot["capability"]["model"]["pg281"]> & { selected_variant?: string; risk4_false_allow?: number };
  const payloadPolicy = snapshot.capability.model.pg281 as PayloadPolicy | undefined;
  const evaluatorBinding = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg282?: Pg282Capability }).pg282;
  const feedbackPolicy = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg283?: Pg283Capability }).pg283;
  const evaluatorContract = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg284?: Pg284Capability }).pg284;
  const payloadGrounding = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg285?: Pg285Capability }).pg285;
  const observationTokens = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg286?: Pg286Capability }).pg286;
  const identifiability = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg287?: Pg287Capability }).pg287;
  const featureGate = (snapshot.capability.model as Snapshot["capability"]["model"] & { pg292?: Pg292Capability }).pg292;
  return <section className="ops-wrap ops-capability">
    <div className="ops-section-head"><div><span>LATEST EXPERIMENT / PG-287 IDENTIFIABILITY GATE</span><h2>当前训练链路</h2></div><span className="ops-chip">LIVE REPORT-BACKED</span></div>
    <div className="ops-cap-grid">
      <article className="ops-cap-card"><span>PG-262 DATA</span><strong className={traces?.audit_complete ? "pass" : "warn"}>{traces?.status === "collecting" ? "RUNNING" : traces?.record_count ? `${traces.record_count}` : "PENDING"}</strong><p>{traces ? `fresh paired · SQL ${traces.sql_count} · XSS ${traces.xss_count}` : "等待采集报告"}</p><code>{traces?.audit_complete ? "audit complete · training locked" : "完整性审计未完成"}</code></article>
      <article className="ops-cap-card"><span>PG-263 AUGMENTED</span><strong className={model?.judge_pass ? "pass" : "warn"}>{pg263Ready ? `${((model?.fresh_route_rule_accuracy ?? 0) * 100).toFixed(0)}%` : "RUNNING"}</strong><p>{pg263Ready ? `hidden ${model?.selected_hidden_dim.toLocaleString()} · ${model?.record_count} 条` : "PG-262 fresh holdout / GPU"}</p><code>{pg263Ready ? `family ${((model?.fresh_route_family_accuracy ?? 0) * 100).toFixed(0)}% · OOD ${((model?.implementation_ood_family_accuracy ?? 0) * 100).toFixed(0)}% · canary ${model?.canary_pass ? "pass" : "fail"}` : "capacity sweep in progress"}</code></article>
      <article className="ops-cap-card"><span>PG-264 GROWTH</span><strong className={growth?.audit_complete ? "pass" : "warn"}>{growth?.status === "collecting" ? "RUNNING" : growth?.record_count ? `${growth.record_count}` : "PENDING"}</strong><p>{growth ? `SQL ${growth.sql_count} · XSS ${growth.xss_count} · boolean ${growth.boolean_count} · widebyte ${growth.widebyte_count}` : "等待 32 个 fresh seed"}</p><code>{growth?.audit_complete ? "audit complete · PG-265 ready" : "audit required before training"}</code></article>
      <article className="ops-cap-card"><span>PG-265 LARGE</span><strong className={large?.audit_pass ? "pass" : "warn"}>{large?.status === "training_running" ? "RUNNING" : large?.status === "stopped_waiting_external_device" ? "STOPPED" : large?.selected_hidden_dim ? `${large.selected_hidden_dim}` : "WAITING"}</strong><p>{large?.record_count ? `${large.record_count} 条 · ${large.adapter_parameter_count.toLocaleString()} params` : large?.status === "stopped_waiting_external_device" ? "等待其他授权设备" : "等待 PG-264 审计"}</p><code>{large?.audit_pass ? `fresh ${((large.fresh_route_rule_accuracy ?? 0) * 100).toFixed(0)}% · judge ${large.judge_pass ? "pass" : "blocked"}` : "4096 / 8192 / 12288"}</code></article>
      <article className="ops-cap-card"><span>PG-267 GROUNDED</span><strong className={grounded?.judge_pass ? "pass" : "warn"}>{grounded?.status ? `${((grounded.fresh_holdout_rule_accuracy ?? 0) * 100).toFixed(0)}%` : "WAITING"}</strong><p>{grounded?.record_count ? `${grounded.record_count} 条 · hidden ${grounded.selected_hidden_dim.toLocaleString()}` : "等待 PG-266 抽象 token"}</p><code>{grounded?.status ? `fresh family ${((grounded.fresh_holdout_family_accuracy ?? 0) * 100).toFixed(0)}% · route-seed ${((grounded.route_seed_rule_accuracy ?? 0) * 100).toFixed(0)}% · ${grounded.judge_pass ? "candidate" : "blocked"}` : "8192 / 12288 / 16384"}</code></article>
      <article className="ops-cap-card"><span>PG-268B REPLAY</span><strong className={replay?.audit_pass ? "pass" : "warn"}>{replay?.status ? `${replay.confirmed_local_effect_count}/${replay.surface_count}` : "WAITING"}</strong><p>{replay?.status ? `GET ${replay.get_count} · POST ${replay.post_count} · complete ${replay.complete_replayed_surface_count}` : "等待参数化 GET/POST 回放"}</p><code>{replay?.status ? `AI/ref/neg ${replay.ai_send_count}/${replay.reference_send_count}/${replay.negative_send_count} · abstain ${replay.abstain_count} · audit ${replay.audit_pass ? "pass" : "blocked"}` : "fresh reset + typed oracle"}</code></article>
      <article className="ops-cap-card"><span>PG-269 GUIDED LOOP</span><strong className={guided?.audit_pass ? "pass" : "warn"}>{guided?.status ? `${guided.final_confirmed_count}/${guided.surface_count}` : "WAITING"}</strong><p>{guided?.status ? `initial ${guided.initial_confirmed_count} · repair ${guided.repair_success_count}/${guided.repair_attempt_count} · abstain ${guided.abstain_count}` : "等待失败引导回放"}</p><code>{guided?.status ? `GET ${guided.get_count} · POST ${guided.post_count} · context/target ${guided.context_target_split_pass ? "split" : "blocked"} · audit ${guided.audit_pass ? "pass" : "blocked"}` : "teacher → diagnose → repair → replay"}</code></article>
      <article className="ops-cap-card"><span>PG-270 TEACHER SFT</span><strong className={teacher?.audit_pass ? "pass" : "warn"}>{teacher?.status ? `${(teacher.guided_token_accuracy * 100).toFixed(1)}%` : "WAITING"}</strong><p>{teacher?.status ? `family holdout ${teacher.family_holdout_count} · preference ${teacher.guided_preference_win_rate.toFixed(2)}` : "等待 A800 教师消融"}</p><code>{teacher?.status ? `next-action ${(teacher.guided_next_action_accuracy * 100).toFixed(0)}% · process ${teacher.process_reward_count} · ${teacher.cuda_assignment?.device_name || "GPU"} · promotion blocked` : "疑问 → 组装 → 失败更新"}</code></article>
      <article className="ops-cap-card"><span>PG-271 FRESH SEED</span><strong className={fresh?.audit_pass ? "pass" : "warn"}>{fresh?.status ? `${(fresh.family_next_action_accuracy * 100).toFixed(0)}%` : "WAITING"}</strong><p>{fresh?.status ? `seed ${fresh.fresh_seed} · family holdout ${fresh.family_holdout_count} · belief ${(fresh.family_final_belief_accuracy * 100).toFixed(0)}%` : "等待独立 fresh 回放"}</p><code>{fresh?.status ? `abstain ${(fresh.family_abstain_calibration_accuracy * 100).toFixed(0)}% · unsupported positive ${fresh.unsupported_positive_count} · oracle claim blocked` : "new seed candidate replay"}</code></article>
      <article className="ops-cap-card"><span>PG-272 DIAGNOSIS</span><strong className="warn">{hypothesis.pg272?.status ? `${((hypothesis.pg272.positive_recall ?? 0) * 100).toFixed(0)}%` : "WAITING"}</strong><p>{hypothesis.pg272?.status ? `negative reject ${((hypothesis.pg272.negative_reject ?? 0) * 100).toFixed(0)}% · false negatives ${hypothesis.pg272.false_negative_count ?? 0}` : "等待独立实现诊断"}</p><code>全 abstain 不是成功 · 表示瓶颈</code></article>
      <article className="ops-cap-card"><span>PG-274 RL REGRESSION</span><strong className="warn">{hypothesis.pg274_score_rl?.status ? "BLOCKED" : "WAITING"}</strong><p>weighted SFT recall 100% · REINFORCE 33%</p><code>奖励错配；KL/行为策略约束必需</code></article>
      <article className="ops-cap-card"><span>PG-275 ATOMIC / DPO</span><strong className="pass">{hypothesis.pg275_hypothesis_ablation?.status ? "100%" : "WAITING"}</strong><p>atomic recall · conservative/DPO 保持</p><code>minimal/collapsed recall 0% · promotion blocked</code></article>
      <article className="ops-cap-card"><span>PG-276 V3 + CANARY</span><strong className="pass">{hypothesis.pg276_third_implementation?.status ? "PASS" : "WAITING"}</strong><p>第三实现 100% · v2 canary 保持</p><code>仍为单族小样本，不是漏洞能力声明</code></article>
      <article className="ops-cap-card"><span>PG-277 QUESTION RECOVERY</span><strong className="warn">{hypothesis.pg277_question_composition?.status ? `${((hypothesis.pg277_question_composition.process_question_recovery_min ?? 0) * 100).toFixed(0)}%` : "WAITING"}</strong><p>{hypothesis.pg277_question_composition?.status ? `coarse conflicts ${hypothesis.pg277_question_composition.coarse_conflicting_record_count} · process recall ${((hypothesis.pg277_question_composition.process_positive_recall ?? 0) * 100).toFixed(0)}%` : "等待信息碰撞/过程监督消融"}</p><code>{hypothesis.pg277_question_composition?.status ? `conservative worst ${((hypothesis.pg277_question_composition.conservative_question_recovery_min ?? 0) * 100).toFixed(0)}% · DPO worst ${((hypothesis.pg277_question_composition.dpo_question_recovery_min ?? 0) * 100).toFixed(0)}% · promotion blocked` : "exact question recovery across seeds"}</code></article>
      <article className="ops-cap-card"><span>PG-279 REMOTE REPLAY</span><strong className={remote?.operational_audit_pass ? "pass" : "warn"}>{remote?.status ? `${remote.get_rows}/${remote.post_rows}` : "WAITING"}</strong><p>{remote?.status ? `GET/POST · failure→repair ${remote.failure_repair_rows} · typed/abstain ${remote.typed_effect_rows}/${remote.abstain_rows}` : "等待远程 A800 回放报告"}</p><code>{remote?.status ? `retention ${remote.retention_status} · family gate ${remote.scientific_gate_status} · gold ${remote.real_application_gold_rows} · promotion blocked` : "GPU0 only · fresh replay ×2"}</code></article>
      <article className="ops-cap-card"><span>PG-280 SHARED ONTOLOGY</span><strong className="warn">{ontology?.status ? `${(ontology.process_ask_rate * 100).toFixed(0)}% ASK` : "WAITING"}</strong><p>{ontology?.status ? `H=${ontology.conditional_entropy_bits.toFixed(1)} bit · Bayes ≥ ${(ontology.bayes_error_lower_bound * 100).toFixed(0)}% · ${ontology.controlled_row_count} rows` : "等待 ontology policy 报告"}</p><code>{ontology?.status ? `final-only post ${(ontology.final_only_post_accuracy * 100).toFixed(0)}% / pre ${ontology.final_only_pre_supervision_rows} · process safe ${(ontology.process_safe_rate * 100).toFixed(0)}% · OOD ${ontology.hard_negative_rows} eval-only · Docker ${ontology.docker_status} · remote adapter ${ontology.remote_adapter_status ?? "not_run"} / audit ${ontology.remote_adapter_audit_pass ? "pass" : "blocked"}` : "shared slot · family-OOD · promotion blocked"}</code></article>
      <article className="ops-cap-card"><span>PG-281 ABSTRACT PLAN</span><strong className="warn">{payloadPolicy?.status ? `${(payloadPolicy.route_positive_recall * 100).toFixed(0)}%` : "WAITING"}</strong><p>{payloadPolicy?.status ? `family ${(payloadPolicy.family_positive_recall * 100).toFixed(0)}% · plan exact ${(payloadPolicy.route_plan_exact_accuracy * 100).toFixed(0)}%/${(payloadPolicy.family_plan_exact_accuracy * 100).toFixed(0)}%` : "等待 abstract probe-policy 报告"}</p><code>{payloadPolicy?.status ? `selected ${payloadPolicy.selected_variant ?? "plain_sft"} · hard reject ${(payloadPolicy.hard_negative_safe_reject * 100).toFixed(0)}% · false-allow ${payloadPolicy.hard_negative_false_allow} · literal payload ${payloadPolicy.literal_payload_generation ? "on" : "off"} · live send ${payloadPolicy.live_send ? "on" : "off"}` : "probe class · channel · encoding · safe gate"}</code></article>
      <article className="ops-cap-card"><span>PG-282 EVALUATOR BINDING</span><strong className="warn">{evaluatorBinding?.status ? `${evaluatorBinding.await_evaluator_count} PENDING` : "WAITING"}</strong><p>{evaluatorBinding?.status ? `rows ${evaluatorBinding.record_count} · abstain ${evaluatorBinding.abstain_count} · confirmed ${evaluatorBinding.confirmed_positive_count}` : "等待 evaluator binding 报告"}</p><code>{evaluatorBinding?.status ? `hard-negative ${evaluatorBinding.hard_negative_abstain ? "abstain pass" : "blocked"} · Docker ${evaluatorBinding.remote_docker_status} · live replay ${evaluatorBinding.live_replay ? "on" : "off"}` : "abstract plan → authorized surface → typed evidence"}</code></article>
      <article className="ops-cap-card"><span>PG-283 FEEDBACK POLICY</span><strong className="warn">{feedbackPolicy?.status ? `${(feedbackPolicy.route_action_safe_exact * 100).toFixed(0)}%` : "WAITING"}</strong><p>{feedbackPolicy?.status ? `route/family ${((feedbackPolicy.route_action_safe_exact ?? 0) * 100).toFixed(0)}% / ${((feedbackPolicy.family_action_safe_exact ?? 0) * 100).toFixed(0)}% · hard ${feedbackPolicy.hard_negative_count}` : "等待多步 feedback policy 报告"}</p><code>{feedbackPolicy?.status ? `engineering ${feedbackPolicy.engineering_gate_status} · scientific ${feedbackPolicy.scientific_gate_status} · selected ${feedbackPolicy.selected_variant} · live send ${feedbackPolicy.live_send ? "on" : "off"}` : "failure → diagnose → repair → replay"}</code></article>
      <article className="ops-cap-card"><span>PG-284 TYPED EVALUATOR</span><strong className="warn">{evaluatorContract?.status ? `${evaluatorContract.blocked_rows} BLOCKED` : "WAITING"}</strong><p>{evaluatorContract?.status ? `contract ${evaluatorContract.contract_rows} · effect ${evaluatorContract.confirmed_effect_rows} · hard-negative ${evaluatorContract.hard_negative_blocked ? "blocked" : "pending"}` : "等待 evaluator contract 报告"}</p><code>{evaluatorContract?.status ? `engineering ${evaluatorContract.engineering_gate_status} · scientific ${evaluatorContract.scientific_gate_status} · Docker ${evaluatorContract.remote_docker_status} · live replay ${evaluatorContract.live_replay ? "on" : "off"}` : "fresh reset + typed effect + replay hash"}</code></article>
      <article className="ops-cap-card"><span>PG-285 PAYLOAD GROUNDING</span><strong className="warn">{payloadGrounding?.status ? `${(payloadGrounding.route_sequence_exact * 100).toFixed(0)}%` : "WAITING"}</strong><p>{payloadGrounding?.status ? `route/family sequence ${((payloadGrounding.route_sequence_exact ?? 0) * 100).toFixed(0)}% / ${((payloadGrounding.family_sequence_exact ?? 0) * 100).toFixed(0)}% · ${payloadGrounding.train_count} train` : "等待结构化 payload grounding 报告"}</p><code>{payloadGrounding?.status ? `selected ${payloadGrounding.selected_variant} · hard reject ${(payloadGrounding.hard_negative_safe_reject * 100).toFixed(0)}% · false-allow ${payloadGrounding.hard_negative_false_allow} · placeholder ${payloadGrounding.runtime_canary_placeholder ? "on" : "off"} · Docker ${payloadGrounding.remote_docker_status}` : "failure → repair → wire-plan next token"}</code></article>
      <article className="ops-cap-card"><span>PG-286 OBSERVATION TOKENS</span><strong className="warn">{observationTokens?.status ? `${observationTokens.complete_rows}/${observationTokens.total_rows}` : "WAITING"}</strong><p>{observationTokens?.status ? `complete / catalog · SQL AST ${observationTokens.sql_ast_available_rows} · hard-negative ${observationTokens.hard_negative_rows}` : "等待 bounded evidence catalog"}</p><code>{observationTokens?.status ? `incomplete ${observationTokens.incomplete_rows} · slots ${observationTokens.shared_slot_count} · batch ${observationTokens.batch_status} (${observationTokens.batch_record_count}) · gold ${observationTokens.batch_training_eligible_rows} · Docker ${observationTokens.remote_docker_status}` : "shared slots · family hidden · promotion blocked"}</code></article>
      <article className="ops-cap-card"><span>PG-287 IDENTIFIABILITY</span><strong className="warn">{identifiability?.status ? (identifiability.family_resolved_encoding_accuracy === null ? "N/A" : `${(identifiability.family_resolved_encoding_accuracy * 100).toFixed(0)}%`) : "WAITING"}</strong><p>{identifiability?.status ? `family resolved encoding · coverage ${identifiability.family_resolved_encoding_count} · ASK ${(identifiability.family_ambiguous_ask_recall * 100).toFixed(0)}%` : "等待族外识别性训练报告"}</p><code>{identifiability?.status ? `route resolved ${(identifiability.route_resolved_encoding_accuracy * 100).toFixed(0)}% · hard ASK ${(identifiability.hard_negative_ask_recall * 100).toFixed(0)}% · false-allow ${identifiability.hard_negative_false_allow} · live batch ${identifiability.live_batch_status} (${identifiability.live_batch_record_count}) · Docker ${identifiability.remote_docker_status}` : "ambiguous → ask_typed · resolved → bounded plan"}</code></article>
      <article className="ops-cap-card"><span>PG-292 FEATURE GATE</span><strong className="warn">{featureGate?.status ? `${(featureGate.route_positive_recall * 100).toFixed(0)}% / ${featureGate.hard_negative_false_allow}` : "WAITING"}</strong><p>{featureGate?.status ? `route/family recall ${(featureGate.route_positive_recall * 100).toFixed(0)}% / ${(featureGate.family_positive_recall * 100).toFixed(0)}% · feature ${featureGate.feature_count}` : "等待 key/value OOD gate 报告"}</p><code>{featureGate?.status ? `threshold ${featureGate.selected_threshold} · hard reject ${(featureGate.hard_negative_safe_reject * 100).toFixed(0)}% · Docker ${featureGate.remote_docker_status} · real gold ${featureGate.real_application_gold_rows} · promotion blocked` : "unknown evaluator → learned abstain boundary"}</code></article>
      <article className="ops-cap-card dark"><span>DECISION</span><strong>{model?.judge_pass ? "REVIEW" : "BLOCKED"}</strong><p>独立审计后再谈晋级</p><code>{model?.evidence_hash || "evidence pending"}</code></article>
    </div>
    <p className="ops-review-note">资源策略：容量分支顺序运行；后续复跑默认使用 micro-batch 16 + gradient accumulation，具体值以报告中的 resource profile 为准。</p>
  </section>;
}

function CapabilityView({ snapshot, payloads }: { snapshot: Snapshot; payloads: PayloadReview }) {
  const grouped = useMemo(() => {
    const groups = new Map<string, { route: string; confirmed: number; total: number; reasons: string[] }>();
    snapshot.tasks.all.forEach((task) => { const current = groups.get(task.route) || { route: task.route, confirmed: 0, total: 0, reasons: [] }; current.total += 1; current.confirmed += Number(task.confirmed_positive); current.reasons.push(...task.reasons); groups.set(task.route, current); });
    return [...groups.values()];
  }, [snapshot.tasks.all]);
  const pg261Ready = Boolean(snapshot.capability.model.pg261 && snapshot.capability.model.pg261.status !== "running" && snapshot.capability.model.pg261.status !== "training_running");
  const pg263Ready = Boolean(snapshot.capability.model.pg263 && snapshot.capability.model.pg263.status !== "not_run" && snapshot.capability.model.pg263.status !== "training_running");
  return <>
    <LatestExperimentPanel snapshot={snapshot} pg263Ready={pg263Ready} />
    <section className="ops-subhero capability"><span className="ops-overline">CAPABILITY / EVIDENCE NOT PROMISE</span><h1>模型现在能做什么，<br /><em>证据说了算。</em></h1><p>能力页将“会发探针”“观察到效果”“允许晋级”拆开显示，避免把一个漂亮分数误读成渗透能力。<br /><a href="/ops#payload-review">打开人工 Payload 审核台，查看 AI/reference/negative 三组 wire ↗</a></p></section>
    <section className="ops-wrap ops-capability"><MetricStrip snapshot={snapshot} /><div className="ops-cap-grid"><article className="ops-cap-card"><span>ADAPTER</span><strong>{snapshot.capability.model.selected_hidden_dim.toLocaleString()}</strong><p>selected hidden dimension</p><code>{snapshot.capability.model.adapter_parameter_count.toLocaleString()} trainable parameters</code></article><article className="ops-cap-card"><span>ROUTE HOLDOUT</span><strong className={snapshot.capability.model.route_holdout_pass ? "pass" : "warn"}>{snapshot.capability.model.route_holdout_pass ? "PASS" : "PENDING"}</strong><p>PG-254 route-family holdout</p><code>oracle_is_model_input = {String(snapshot.capability.model.model_input_uses_oracle)}</code></article><article className="ops-cap-card"><span>PG-257 RULE IR</span><strong className={snapshot.capability.model.pg257?.status === "completed_rule_ir_class_capacity_training" ? "pass" : "warn"}>{snapshot.capability.model.pg257 ? `${(snapshot.capability.model.pg257.seed_holdout_rule_accuracy * 100).toFixed(0)}%` : "PENDING"}</strong><p>{snapshot.capability.model.pg257 ? `偶数 seed 留出 · ${snapshot.capability.model.pg257.record_count} 条` : "等待训练报告"}</p><code>{snapshot.capability.model.pg257 ? `widebyte recall ${(snapshot.capability.model.pg257.seed_holdout_widebyte_recall * 100).toFixed(0)}% · next-token ${(snapshot.capability.model.pg257.seed_holdout_next_token_accuracy * 100).toFixed(2)}%` : "promotion = blocked"}</code></article><article className="ops-cap-card"><span>PG-258 UNIFIED IR</span><strong className={snapshot.capability.model.pg258?.status === "candidate_eligible_for_next_replay" ? "pass" : "warn"}>{snapshot.capability.model.pg258 ? `${(snapshot.capability.model.pg258.holdout_rule_accuracy * 100).toFixed(0)}%` : "PENDING"}</strong><p>{snapshot.capability.model.pg258 ? `seed/route 留出 · ${snapshot.capability.model.pg258.record_count} 条` : "等待训练报告"}</p><code>{snapshot.capability.model.pg258 ? `family ${(snapshot.capability.model.pg258.holdout_family_accuracy * 100).toFixed(0)}% · OOD ${(snapshot.capability.model.pg258.implementation_ood_family_accuracy * 100).toFixed(0)}% · canary ${snapshot.capability.model.pg258.canary_pass ? "pass" : "fail"}` : "promotion = blocked"}</code></article><article className="ops-cap-card"><span>PG-259 ACTIVE BELIEF</span><strong className={snapshot.capability.model.pg259?.status === "candidate_eligible_for_next_replay" ? "pass" : "warn"}>{snapshot.capability.model.pg259 ? `${(snapshot.capability.model.pg259.fresh_route_rule_accuracy * 100).toFixed(0)}%` : "PENDING"}</strong><p>{snapshot.capability.model.pg259 ? `fresh route 留出 · ${snapshot.capability.model.pg259.record_count} 条` : "等待训练报告"}</p><code>{snapshot.capability.model.pg259 ? `family ${(snapshot.capability.model.pg259.fresh_route_family_accuracy * 100).toFixed(0)}% · belief ${(snapshot.capability.model.pg259.fresh_route_belief_accuracy * 100).toFixed(0)}% · OOD ${(snapshot.capability.model.pg259.implementation_ood_family_accuracy * 100).toFixed(0)}%` : "promotion = blocked"}</code></article><article className="ops-cap-card"><span>PG-260 CAPACITY</span><strong className={snapshot.capability.model.pg260?.judge_pass ? "pass" : "warn"}>{snapshot.capability.model.pg260?.status && snapshot.capability.model.pg260.status !== "not_run" ? `${(snapshot.capability.model.pg260.fresh_route_rule_accuracy * 100).toFixed(0)}%` : "PENDING"}</strong><p>{snapshot.capability.model.pg260?.status && snapshot.capability.model.pg260.status !== "not_run" ? `hidden ${snapshot.capability.model.pg260.selected_hidden_dim.toLocaleString()} · ${snapshot.capability.model.pg260.record_count} 条` : "等待 GPU 报告"}</p><code>{snapshot.capability.model.pg260?.status && snapshot.capability.model.pg260.status !== "not_run" ? `family ${(snapshot.capability.model.pg260.fresh_route_family_accuracy * 100).toFixed(0)}% · abstain ${(snapshot.capability.model.pg260.fresh_route_unknown_abstain_accuracy * 100).toFixed(0)}% · OOD ${(snapshot.capability.model.pg260.implementation_ood_family_accuracy * 100).toFixed(0)}%` : "promotion = blocked"}</code></article><article className="ops-cap-card"><span>PG-261 MASKED</span><strong className={snapshot.capability.model.pg261?.judge_pass ? "pass" : "warn"}>{pg261Ready ? `${((snapshot.capability.model.pg261?.fresh_route_rule_accuracy ?? 0) * 100).toFixed(0)}%` : "RUNNING"}</strong><p>{pg261Ready ? `hidden ${snapshot.capability.model.pg261?.selected_hidden_dim.toLocaleString()} · ${snapshot.capability.model.pg261?.record_count} 条` : "mask-aware pooling / GPU"}</p><code>{pg261Ready ? `family ${((snapshot.capability.model.pg261?.fresh_route_family_accuracy ?? 0) * 100).toFixed(0)}% · OOD ${((snapshot.capability.model.pg261?.implementation_ood_family_accuracy ?? 0) * 100).toFixed(0)}% · canary ${snapshot.capability.model.pg261?.canary_pass ? "pass" : "fail"}` : "padding invariance under evaluation"}</code></article><article className="ops-cap-card dark"><span>JUDGE SCOPE</span><strong>LOOPBACK</strong><p>local typed effect only</p><code>promotion = blocked</code></article></div></section>
    <section className="ops-wrap ops-route-table"><div className="ops-section-head"><div><span>ROUTE / EFFECT SEPARATION</span><h2>逐条能力矩阵</h2></div><span className="ops-chip">{grouped.length} route families</span></div><div className="ops-table"><div className="ops-table-head"><span>ROUTE</span><span>RUNS</span><span>CONFIRMED</span><span>DECISION</span></div>{grouped.map((row) => <div className="ops-table-row" key={row.route}><b>{row.route}</b><span>{row.total}</span><span className={row.confirmed ? "pass" : "warn"}>{row.confirmed}/{row.total}</span><span>{row.confirmed ? "本地可复核" : row.reasons[0] || "待 oracle"}</span></div>)}</div></section>
    <ProcessReplayPanel traces={snapshot.process_traces} />
    <PayloadReviewPanel payloads={payloads} />
    <section className="ops-wrap ops-limits"><div><span>KNOWN LIMITS</span><h2>能力边界</h2></div><div>{snapshot.capability.limits.map((limit) => <p key={limit}><i>!</i>{limit}</p>)}</div><strong>NEXT → {snapshot.capability.next}</strong></section>
  </>;
}

export default function ResearchOps({ view }: { view: View }) {
  const [snapshot, setSnapshot] = useState<Snapshot>(fallback);
  const [payloads, setPayloads] = useState<PayloadReview>(payloadFallback);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState("");
  const refresh = () => Promise.all([loadSnapshot(), loadPayloadReview()]).then(([data, review]) => { setSnapshot(data); setPayloads(review); setOnline(true); setError(""); }).catch((reason: Error) => { setOnline(false); setError(reason.message); });
  useEffect(() => { refresh(); const timer = window.setInterval(refresh, 15000); return () => window.clearInterval(timer); }, []);
  return <main className="ops-page"><OpsHeader view={view} online={online} updated={snapshot.generated_at} onRefresh={refresh} />{error && <div className="ops-error">数据快照暂不可用：{error}。页面保留空状态，不伪造指标。</div>}{view === "tasks" && <TasksView snapshot={snapshot} payloads={payloads} />}{view === "architecture" && <ArchitectureView snapshot={snapshot} />}{view === "capability" && <CapabilityView snapshot={snapshot} payloads={payloads} />}<footer className="ops-footer"><span>SIFT / RESEARCH OPS / {snapshot.schema_version}</span><a href="/">返回主实验控制台 ↗</a><span>AI FIRST · HUMAN ESCALATION · FINAL JUDGE</span></footer></main>;
}
