"use client";

import { useEffect, useMemo, useState } from "react";

type ScenarioKey = "reachable" | "opaque";
type NodeKind = "start" | "probe" | "junction" | "dead" | "candidate" | "control" | "oracle" | "exit";
type StepStatus = "ready" | "probe" | "candidate" | "deadend" | "backtrack" | "negative" | "oracle" | "exit" | "abstain";

type Belief = {
  effect: number;
  inputOnly: number;
  none: number;
  unknown: number;
};

type MazeEvent = {
  node: string;
  phase: string;
  status: StepStatus;
  action: string;
  observation: string;
  detail: string;
  belief: Belief;
  rule: string;
  evidence: string;
};

type DistanceVector = {
  remaining: number;
  gate: number;
  uncertainty: number;
  risk: number;
  cost: number;
  display: number;
};

type MazeNode = {
  id: string;
  label: string;
  caption: string;
  col: number;
  row: number;
  kind: NodeKind;
};

type TargetAnalysis = {
  target: { origin: string; path: string; authorized_scope: string };
  request_count: number;
  maze_events: MazeEvent[];
  candidate_status: "candidate" | "abstain";
  exit_unlocked: boolean;
  typed_oracle: { status: string; confirmed_positive: boolean };
  safe_probe_manifests: Array<{ method: string; path: string; payload_sha256: string }>;
  evidence_sha256: string;
  promotion: { training_sample: boolean; long_term_memory: boolean; reason: string };
};

const nodes: MazeNode[] = [
  { id: "start", label: "起点", caption: "fresh reset", col: 1, row: 3, kind: "start" },
  { id: "scan", label: "观察", caption: "surface", col: 2, row: 3, kind: "probe" },
  { id: "junction", label: "分叉", caption: "belief", col: 3, row: 3, kind: "junction" },
  { id: "dead", label: "死路", caption: "decoy", col: 4, row: 2, kind: "dead" },
  { id: "candidate", label: "候选", caption: "GET / POST", col: 4, row: 3, kind: "candidate" },
  { id: "false-exit", label: "假出口", caption: "shape only", col: 5, row: 2, kind: "dead" },
  { id: "replay", label: "复放", caption: "negative", col: 5, row: 3, kind: "control" },
  { id: "oracle", label: "验收", caption: "typed oracle", col: 6, row: 3, kind: "oracle" },
  { id: "exit", label: "出口", caption: "confirmed", col: 7, row: 3, kind: "exit" },
];

const edges = [
  ["start", "scan"],
  ["scan", "junction"],
  ["junction", "dead"],
  ["dead", "false-exit"],
  ["junction", "candidate"],
  ["candidate", "replay"],
  ["replay", "oracle"],
  ["oracle", "exit"],
] as const;

const coordinates: Record<string, [number, number]> = {
  start: [50, 270],
  scan: [150, 270],
  junction: [250, 270],
  dead: [350, 180],
  candidate: [350, 270],
  "false-exit": [450, 180],
  replay: [450, 270],
  oracle: [550, 270],
  exit: [650, 270],
};

const reachableEvents: MazeEvent[] = [
  {
    node: "start",
    phase: "RESET",
    status: "ready",
    action: "RESET · fresh target",
    observation: "episode fresh / state cleared",
    detail: "每一轮从新的本地实例开始，避免把上一次状态当成证据。",
    belief: { effect: 0, inputOnly: 0, none: 0, unknown: 1 },
    rule: "episode := fresh_reset()",
    evidence: "sha256: 7e31…c0a2",
  },
  {
    node: "scan",
    phase: "PROBE",
    status: "probe",
    action: "GET · safe_probe.control",
    observation: "baseline shape / no effect",
    detail: "先拿到无效对照，记录响应形状；此时没有漏洞结论。",
    belief: { effect: 0, inputOnly: 0.08, none: 0.52, unknown: 0.4 },
    rule: "surface_delta := compare(control, baseline)",
    evidence: "sha256: b118…4fd9",
  },
  {
    node: "junction",
    phase: "BELIEF",
    status: "candidate",
    action: "GET · safe_probe.candidate",
    observation: "response shape changed",
    detail: "形状变化只是候选信号，不能直接当作出口；模型把概率推向 candidate。",
    belief: { effect: 0.12, inputOnly: 0.56, none: 0.14, unknown: 0.18 },
    rule: "candidate := shape_changed AND typed_effect_unknown",
    evidence: "sha256: 2a67…9140",
  },
  {
    node: "dead",
    phase: "PROBE",
    status: "deadend",
    action: "GET · surface.decoy",
    observation: "redirect-like shape / no typed effect",
    detail: "这是一条故意设置的假路：表面像变化，但 oracle 没有确认预期效果。",
    belief: { effect: 0.04, inputOnly: 0.61, none: 0.2, unknown: 0.15 },
    rule: "shape_delta ≠ confirmed_effect",
    evidence: "sha256: 91af…17e8",
  },
  {
    node: "false-exit",
    phase: "ORACLE",
    status: "abstain",
    action: "typed oracle · reject false exit",
    observation: "expected effect absent / matched negative",
    detail: "验证门未通过，模型必须停在 abstain，而不是为了‘过关’强行报漏洞。",
    belief: { effect: 0.02, inputOnly: 0.66, none: 0.25, unknown: 0.07 },
    rule: "confirmed_positive := effect AND negative_control_passed",
    evidence: "sha256: 4c2a…dbe7",
  },
  {
    node: "dead",
    phase: "BACKTRACK",
    status: "backtrack",
    action: "BACKTRACK · prune branch",
    observation: "dead end marked / return to junction",
    detail: "把死路写进轨迹，回退到分叉点；这一步才是迷宫控制层的核心。",
    belief: { effect: 0.05, inputOnly: 0.34, none: 0.46, unknown: 0.15 },
    rule: "if oracle_failed: mark_dead_end(); backtrack()",
    evidence: "sha256: 6bd0…0f32",
  },
  {
    node: "candidate",
    phase: "PROBE",
    status: "candidate",
    action: "POST · safe_probe.candidate",
    observation: "same abstract signal on POST",
    detail: "换通道复放，检查模型是否只记住 GET 表面，而不是理解抽象规则。",
    belief: { effect: 0.24, inputOnly: 0.53, none: 0.08, unknown: 0.15 },
    rule: "candidate := invariant_signal(GET, POST)",
    evidence: "sha256: e8c1…62a4",
  },
  {
    node: "replay",
    phase: "CONTROL",
    status: "negative",
    action: "POST · matched_negative_control",
    observation: "negative control stays inert",
    detail: "阴性对照没有出现同样效果，候选信号因此获得更高可信度。",
    belief: { effect: 0.48, inputOnly: 0.37, none: 0.07, unknown: 0.08 },
    rule: "specificity := candidate_effect - control_effect",
    evidence: "sha256: 1f2e…c6d0",
  },
  {
    node: "oracle",
    phase: "ORACLE",
    status: "oracle",
    action: "typed oracle · replay + evidence",
    observation: "expected effect confirmed / hash frozen",
    detail: "只有预期效果、阴性对照、fresh reset、证据哈希同时满足，才允许绑定 Rule IR。",
    belief: { effect: 0.87, inputOnly: 0.08, none: 0.02, unknown: 0.03 },
    rule: "slot.surface_transition := {method, delta, oracle, evidence_hash}",
    evidence: "sha256: 9a1d…55be",
  },
  {
    node: "exit",
    phase: "CONFIRM",
    status: "exit",
    action: "EXIT · confirmed_positive",
    observation: "maze exit unlocked",
    detail: "出口不是‘看起来像’，而是通过 typed oracle 的可复放证据后解锁。",
    belief: { effect: 0.96, inputOnly: 0.02, none: 0.01, unknown: 0.01 },
    rule: "confirmed_positive → bind Rule IR → replay on fresh target",
    evidence: "sha256: 0d4b…a1f6",
  },
];

const opaqueEvents: MazeEvent[] = [
  ...reachableEvents.slice(0, 9),
  {
    node: "oracle",
    phase: "ORACLE",
    status: "abstain",
    action: "typed oracle · insufficient evidence",
    observation: "effect not observable on this target",
    detail: "族外目标没有给出可验证的预期效果，正确结果是 abstain，而不是把表面变化当漏洞。",
    belief: { effect: 0.21, inputOnly: 0.2, none: 0.09, unknown: 0.5 },
    rule: "if typed_effect == unknown: abstain()",
    evidence: "sha256: 4f91…cc13",
  },
];

const scenarios: Record<ScenarioKey, { title: string; note: string; events: MazeEvent[] }> = {
  reachable: {
    title: "可达出口 · 回退后确认",
    note: "先走假路，再通过 GET/POST、阴性对照和 typed oracle 解锁出口。",
    events: reachableEvents,
  },
  opaque: {
    title: "不可判定 · 正确弃权",
    note: "表面有信号但 oracle 不可见，模型应该停在 abstain。",
    events: opaqueEvents,
  },
};

const statusLabels: Record<StepStatus, string> = {
  ready: "准备",
  probe: "探测",
  candidate: "候选",
  deadend: "死路",
  backtrack: "回退",
  negative: "阴性对照",
  oracle: "oracle",
  exit: "出口确认",
  abstain: "弃权",
};

const statusClass = (status: StepStatus) => `maze-status-${status}`;

function clamp(value: number) {
  return Math.min(1, Math.max(0, value));
}

function deriveDisplayDistance(event: MazeEvent, index: number, total: number): DistanceVector {
  // The demo has no target evaluator for the static maze, so these are
  // explicitly display-only approximations.  The backend research contract
  // uses app/maze_distance.py with evaluator-side gate facts instead.
  const gates = [
    index >= 0, // authorized scope
    index >= 0, // fresh reset
    index >= 1, // safe probe
    event.phase === "CONTROL" || event.status === "oracle" || event.status === "exit", // matched negative
    event.action.includes("POST") || event.phase === "REPLAY" || event.status === "exit", // cross-channel replay
    event.status === "oracle" || event.status === "exit", // typed effect gate
    index >= 1, // evidence is present in the trace
    event.status === "exit", // Rule IR binding
  ];
  const remaining = gates.filter((value) => !value).length;
  const gate = remaining / gates.length;
  const uncertainty = clamp(event.belief.unknown * 1.25);
  const risk = event.status === "candidate" ? 0.65 : event.status === "deadend" ? 0.55 : event.status === "abstain" ? 0.15 : event.status === "exit" ? 0.02 : 0.35;
  const cost = clamp((index + 1) / Math.max(total, 1));
  return { remaining, gate, uncertainty, risk, cost, display: 0.45 * gate + 0.25 * uncertainty + 0.2 * risk + 0.1 * cost };
}

function DistanceView({ distance }: { distance: DistanceVector }) {
  const rows = [
    ["验收门剩余", distance.gate, "#c8f25a"],
    ["belief 未知", distance.uncertainty, "#ff7143"],
    ["误报风险", distance.risk, "#d5c35b"],
    ["动作成本", distance.cost, "#6c84ff"],
  ] as const;
  return (
    <div className="maze-distance-view">
      <div className="maze-distance-total"><strong>{distance.remaining}</strong><span> gates remain</span><b>{Math.round(distance.display * 100)}%</b></div>
      {rows.map(([label, value, color]) => (
        <div className="maze-distance-row" key={label}><div><span>{label}</span><b>{Math.round(value * 100)}%</b></div><div className="maze-belief-track"><i style={{ width: `${Math.max(3, value * 100)}%`, background: color }} /></div></div>
      ))}
      <small className="maze-distance-note">多轴展示距离；不是漏洞置信度，也不能单独解锁出口。</small>
    </div>
  );
}

function BeliefBars({ belief }: { belief: Belief }) {
  const rows = [
    ["effect", "预期效果", belief.effect, "#c8f25a"],
    ["inputOnly", "仅输入回显", belief.inputOnly, "#6c84ff"],
    ["none", "无变化", belief.none, "#8d958b"],
    ["unknown", "未知", belief.unknown, "#ff7143"],
  ] as const;

  return (
    <div className="maze-belief-bars">
      {rows.map(([key, label, value, color]) => (
        <div className="maze-belief-row" key={key}>
          <div><span>{label}</span><b>{Math.round(value * 100)}%</b></div>
          <div className="maze-belief-track"><i style={{ width: `${Math.max(3, value * 100)}%`, background: color }} /></div>
        </div>
      ))}
    </div>
  );
}

export default function MazeLab() {
  const [scenario, setScenario] = useState<ScenarioKey>("reachable");
  const [stepIndex, setStepIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const [targetUrl, setTargetUrl] = useState("http://127.0.0.1:3100/");
  const [operatorConfirmed, setOperatorConfirmed] = useState(false);
  const [allowSafePost, setAllowSafePost] = useState(false);
  const [targetBusy, setTargetBusy] = useState(false);
  const [targetError, setTargetError] = useState<string | null>(null);
  const [targetResult, setTargetResult] = useState<TargetAnalysis | null>(null);
  const [targetEvents, setTargetEvents] = useState<MazeEvent[] | null>(null);
  const [authorizedOrigins, setAuthorizedOrigins] = useState<string[]>([]);
  const activeScenario = scenarios[scenario];
  const events = targetEvents ?? activeScenario.events;
  const current = events[Math.min(stepIndex, events.length - 1)];
  const currentDistance = deriveDisplayDistance(current, stepIndex, events.length);

  useEffect(() => {
    fetch("/api/maze/target/scope")
      .then((response) => response.ok ? response.json() as Promise<{ authorized_origins?: string[] }> : Promise.reject(new Error("scope unavailable")))
      .then((scope) => setAuthorizedOrigins(scope.authorized_origins ?? []))
      .catch(() => setAuthorizedOrigins([]));
  }, []);

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => {
      setStepIndex((index) => {
        if (index >= events.length - 1) {
          setRunning(false);
          return index;
        }
        return index + 1;
      });
    }, 850);
    return () => window.clearInterval(timer);
  }, [events.length, running]);

  const visited = useMemo(() => new Set(events.slice(0, stepIndex + 1).map((event) => event.node)), [events, stepIndex]);
  const exploredCount = visited.size;
  const exitFound = current.status === "exit";
  const isFinished = stepIndex >= events.length - 1;

  const reset = () => {
    setRunning(false);
    setStepIndex(0);
  };

  const changeScenario = (next: ScenarioKey) => {
    setRunning(false);
    setScenario(next);
    setStepIndex(0);
    setTargetResult(null);
    setTargetEvents(null);
    setTargetError(null);
  };

  const analyzeTarget = async () => {
    if (targetBusy) return;
    setTargetBusy(true);
    setTargetError(null);
    setRunning(false);
    try {
      const response = await fetch("/api/maze/target/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_url: targetUrl.trim(),
          authorization: "docker_local",
          operator_confirmed: operatorConfirmed,
          allow_safe_post: allowSafePost,
        }),
      });
      const body = await response.json() as TargetAnalysis & { detail?: string };
      if (!response.ok) throw new Error(body.detail ?? "目标观察失败");
      setTargetResult(body);
      setTargetEvents(body.maze_events);
      setStepIndex(0);
    } catch (error) {
      setTargetError(error instanceof Error ? error.message : "目标观察失败");
    } finally {
      setTargetBusy(false);
    }
  };

  const start = () => {
    if (running) {
      setRunning(false);
      return;
    }
    if (isFinished) setStepIndex(0);
    setRunning(true);
  };

  const step = () => {
    setRunning(false);
    setStepIndex((index) => Math.min(index + 1, events.length - 1));
  };

  return (
    <main className="maze-page">
      <header className="maze-header">
        <a className="maze-brand" href="/ops/capability"><span>↗</span><b>SIFT / MAZE LAB</b></a>
        <div className="maze-header-state"><i /> LOCAL TARGET · INERT SIMULATION</div>
      </header>

      <section className="maze-hero">
        <div>
          <p className="maze-overline">TRACE CONTROL GAME · PG-MAZE-01</p>
          <h1>把黑盒问题，<em>画成一座迷宫。</em></h1>
          <p>AI 不靠猜“这是不是漏洞”，而是在节点之间走：探测、更新 belief、撞上死路、回退，再用 typed oracle 验收出口。</p>
        </div>
        <div className="maze-hero-ticket">
          <span>CONTROL LOOP</span>
          <strong>PROBE → BELIEF → ORACLE</strong>
          <small>每一步都留下可复放证据</small>
        </div>
      </section>

      <section className="maze-target-console">
        <div className="maze-target-copy"><span>05 / OPERATOR TARGET</span><h2>输入获授权的 Docker URL</h2><p>后台会做有限的 GET 基线、带 canary 的 GET、可选安全 POST 和一次复放，再把响应投影成迷宫轨迹。正文不保存，typed oracle 不可见时只会 abstain。</p></div>
        <div className="maze-target-form">
          <label htmlFor="maze-target-url">TARGET URL<input id="maze-target-url" value={targetUrl} onChange={(event) => setTargetUrl(event.target.value)} placeholder="http://pikachu:80/vulnerabilities/..." spellCheck={false} /></label>
          <div className="maze-target-checks">
            <label><input type="checkbox" checked={operatorConfirmed} onChange={(event) => setOperatorConfirmed(event.target.checked)} />我确认这是我控制的 Docker 靶场</label>
            <label><input type="checkbox" checked={allowSafePost} onChange={(event) => setAllowSafePost(event.target.checked)} />允许发送 sift_probe 安全 POST</label>
          </div>
          <button type="button" className="maze-button maze-button-primary maze-target-run" onClick={analyzeTarget} disabled={targetBusy || !operatorConfirmed}>{targetBusy ? "抓取并抽象中…" : "开始抓取 → 建立迷宫"}</button>
          {targetError ? <p className="maze-target-error">{targetError}</p> : null}
          {targetResult ? <p className="maze-target-result"><b>{targetResult.candidate_status === "candidate" ? "发现候选表面差分" : "没有稳定差分"}</b> · {targetResult.request_count} requests · {targetResult.evidence_sha256}</p> : null}
          {targetResult ? <details className="maze-probe-manifests" open><summary>生成的可复现安全 probe（非攻击 payload）</summary><pre>{JSON.stringify(targetResult.safe_probe_manifests, null, 2)}</pre></details> : null}
        </div>
        <div className="maze-authorized-scope"><span>AUTHORIZED ORIGINS</span><code>{authorizedOrigins.length ? authorizedOrigins.join("  ·  ") : "由后端 SIFT_AUTHORIZED_DOCKER_TARGETS 配置"}</code></div>
      </section>

      <section className="maze-toolbar" aria-label="迷宫控制">
        <div className="maze-scenario-picker">
          <label htmlFor="maze-scenario">SCENARIO</label>
          <select id="maze-scenario" value={scenario} onChange={(event) => changeScenario(event.target.value as ScenarioKey)}>
            <option value="reachable">可达出口 · 回退后确认</option>
            <option value="opaque">不可判定 · 正确弃权</option>
          </select>
        </div>
        <p>{activeScenario.note}</p>
        <div className="maze-controls">
          <button type="button" className="maze-button maze-button-primary" onClick={start}>{running ? "暂停" : isFinished ? "再跑一遍" : "开始寻路"}</button>
          <button type="button" className="maze-button" onClick={step} disabled={isFinished}>单步</button>
          <button type="button" className="maze-button maze-button-quiet" onClick={reset}>重置</button>
        </div>
      </section>

      <section className="maze-stats" aria-label="当前统计">
        <div><span>STEP</span><strong>{String(stepIndex + 1).padStart(2, "0")} / {String(events.length).padStart(2, "0")}</strong></div>
        <div><span>EXPLORED NODES</span><strong>{exploredCount} / {nodes.length}</strong></div>
        <div><span>CURRENT STATUS</span><strong className={statusClass(current.status)}>{statusLabels[current.status]}</strong></div>
        <div><span>EXIT</span><strong className={exitFound ? "maze-status-exit" : "maze-status-abstain"}>{exitFound ? "UNLOCKED" : "LOCKED"}</strong></div>
      </section>

      <section className="maze-main-grid">
        <article className="maze-card maze-map-card">
          <div className="maze-card-head"><div><span>01 / SEARCH SPACE</span><h2>{targetResult ? "实时目标迷宫" : "迷宫状态"}</h2></div><b>{targetResult ? "TARGET TRACE" : running ? "RUNNING" : isFinished ? "PAUSED AT END" : "READY"}</b></div>
          <div className="maze-map-wrap">
            <svg className="maze-edges" viewBox="0 0 700 360" aria-hidden="true">
              {edges.map(([from, to]) => {
                const [x1, y1] = coordinates[from];
                const [x2, y2] = coordinates[to];
                const active = visited.has(from) && visited.has(to);
                return <line key={`${from}-${to}`} x1={x1} y1={y1} x2={x2} y2={y2} className={active ? "active" : ""} />;
              })}
            </svg>
            <div className="maze-grid">
              {Array.from({ length: 28 }, (_, index) => {
                const col = (index % 7) + 1;
                const row = Math.floor(index / 7) + 1;
                const node = nodes.find((item) => item.col === col && item.row === row);
                if (!node) return <div className="maze-cell maze-wall" key={`${col}-${row}`} />;
                const isCurrent = current.node === node.id;
                const isVisited = visited.has(node.id);
                return (
                  <div className={`maze-cell maze-node maze-node-${node.kind} ${isVisited ? "visited" : ""} ${isCurrent ? "current" : ""}`} key={node.id}>
                    <span className="maze-node-dot" />
                    <strong>{node.label}</strong>
                    <small>{node.caption}</small>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="maze-legend"><span><i className="legend-active" />已探索</span><span><i className="legend-dead" />死路 / 假出口</span><span><i className="legend-exit" />出口</span><span><i className="legend-wall" />不可通行</span></div>
        </article>

        <aside className="maze-card maze-trace-card">
          <div className="maze-card-head"><div><span>02 / TRACE STREAM</span><h2>AI 行走记录</h2></div><b>{events.length} EVENTS</b></div>
          <div className="maze-trace-list" aria-live="polite">
            {events.map((event, index) => (
              <button type="button" className={`maze-trace-event ${index === stepIndex ? "selected" : ""} ${index <= stepIndex ? "seen" : ""}`} key={`${event.node}-${index}`} onClick={() => { setRunning(false); setStepIndex(index); }}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <i className={statusClass(event.status)} />
                <div><b>{event.action}</b><small>{event.phase} · {event.observation}</small></div>
                <em>{statusLabels[event.status]}</em>
              </button>
            ))}
          </div>
        </aside>
      </section>

      <section className="maze-detail-grid">
        <article className="maze-card maze-detail-card">
          <div className="maze-card-head"><div><span>03 / CURRENT EVENT</span><h2>{current.action}</h2></div><b className={statusClass(current.status)}>{statusLabels[current.status]}</b></div>
          <p className="maze-detail-copy">{current.detail}</p>
          <div className="maze-detail-columns">
            <div><span>OBSERVATION DIFF</span><pre>{current.observation}</pre></div>
            <div><span>BELIEF POSTERIOR</span><BeliefBars belief={current.belief} /></div>
            <div><span>TRACE DISTANCE · DISPLAY ONLY</span><DistanceView distance={currentDistance} /></div>
          </div>
        </article>

        <article className="maze-card maze-rule-card">
          <div className="maze-card-head"><div><span>04 / RULE IR BINDING</span><h2>中间层记忆</h2></div><b>{current.status === "exit" ? "BOUND" : "PROVISIONAL"}</b></div>
          <pre className="maze-rule-code">{current.rule}</pre>
          <div className="maze-evidence"><span>FROZEN EVIDENCE</span><code>{current.evidence}</code><small>{current.status === "exit" ? "可在 fresh target 上复放" : "尚未满足长期记忆门"}</small></div>
        </article>
      </section>

      <section className="maze-principle">
        <span>WHY THIS MATTERS</span>
        <div><h2>“找到出口”不是命中一个字符串，而是完成一条可解释、可回退、可复放的轨迹。</h2><p>这个页面展示的是模型的控制层能力：它如何处理死路和不确定性。它不是公网扫描器，也不会把表面变化自动升级成漏洞结论；只有本地授权 oracle 给出预期效果，出口才会亮起。</p></div>
      </section>

      <footer className="maze-footer"><a href="/ops/capability">← 返回能力与回放台</a><span>SAFE DEMO · NO EXTERNAL TARGETS · NO REAL PAYLOADS</span></footer>
    </main>
  );
}
