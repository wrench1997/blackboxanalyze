"use client";

import { useEffect, useMemo, useState } from "react";

type Json = Record<string, unknown>;
type FieldSpec = { path: string; label?: string; type: string; domain?: unknown[] };
type Scenario = {
  id: string;
  name: string;
  description?: string;
  category?: string;
  severity?: string;
  cwe?: string;
  research_question?: string;
  hypothesis?: string;
  game_rule?: string;
  intended_rule?: string;
  js_source?: string;
  tags?: string[];
  stateful?: boolean;
  fields: FieldSpec[];
};
type Candidate = { pretty: string; accuracy: number; complexity: number; score: number; expr: Json };
type ClosureReport = {
  closure_status: string;
  closure_score: number;
  confidence: string;
  reasons?: string[];
  recommendations?: string[];
  coverage?: { envelope_coverage?: number };
  hypotheses?: { max_disagreement?: number; behavior_class_count?: number };
};
type ActivityEvent = {
  id: string;
  timestamp: string;
  actor: string;
  tool: string;
  phase: string;
  status: string;
  message: string;
  payload: Json;
  artifact?: string;
};
type NeuralUrlSummary = {
  status: string;
  scope: string;
  model: { parameters: number; learned_set_head_parameters: number; target_training_examples: number };
  final_confirmation: {
    fresh_seeds: number[];
    candidate_neural_mean: number;
    candidate_neural_min: number;
    frozen_neural_mean: number;
    minimum_gain_over_frozen: number;
    minimum_same_checkpoint_set_head_gain: number;
    worst_old_family_regression: number;
    counterexample_top1_mean: number;
    random_top1_mean: number;
    all_preregistered_checks_passed: boolean;
  };
  per_seed: Array<{
    seed: number;
    dataset_sha256: string;
    candidate_neural: number;
    same_checkpoint_without_set_head: number;
    frozen_neural: number;
    c5_rule_system: number;
    counterexample_top1: number;
    worst_old_family_regression: number;
  }>;
  failed_branches: Array<{ name: string; result: string; reason: string }>;
  separation_of_claims: { neural: string; c5: string; engineering: string };
};

const navItems = [
  ["lab", "控制台"],
  ["activity", "活动流"],
  ["results", "实验结果"],
  ["model", "模型"],
  ["memory", "记忆"],
  ["rules", "改进规则"],
];

const closureNames: Record<string, string> = {
  insufficient_data: "数据不足",
  open: "假设仍开放",
  identified: "规则已识别",
  observationally_closed: "行为闭环",
  observationally_closed_low_coverage: "低覆盖闭环",
  context_incomplete_or_nondeterministic: "隐藏上下文",
  dsl_or_search_insufficient: "搜索空间不足",
  budget_or_domain_limited: "预算受限",
};

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    play: <><path d="m9 7 8 5-8 5V7Z" /></>,
    arrow: <><path d="M5 12h14m-5-5 5 5-5 5" /></>,
    spark: <><path d="m12 3 1.4 4.6L18 9l-4.6 1.4L12 15l-1.4-4.6L6 9l4.6-1.4L12 3Zm6 12 .7 2.3L21 18l-2.3.7L18 21l-.7-2.3L15 18l2.3-.7L18 15Z" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
    memory: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m14 0h3M2 15h3m14 0h3M9 10h6v4H9z" /></>,
    code: <><path d="m8 8-4 4 4 4m8-8 4 4-4 4m-3-10-2 12" /></>,
    flask: <><path d="M9 3h6m-5 0v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3M8 15h8" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body as T;
}

function setPath(root: Record<string, unknown>, path: string, value: unknown) {
  const parts = path.split(".");
  let cursor = root;
  parts.slice(0, -1).forEach((part) => {
    const next = (cursor[part] as Record<string, unknown>) || {};
    cursor[part] = next;
    cursor = next;
  });
  cursor[parts.at(-1)!] = value;
}

function cartesianCases(fields: FieldSpec[], limit = 8) {
  let rows: Json[] = [{ input: {}, context: {}, state: {} }];
  for (const field of fields) {
    const next: Json[] = [];
    for (const row of rows) {
      for (const value of field.domain?.length ? field.domain : [""]) {
        const copy = JSON.parse(JSON.stringify(row)) as Json;
        setPath(copy, field.path, value);
        next.push(copy);
      }
    }
    rows = next;
  }
  if (rows.length <= limit) return rows;
  const indices = new Set([0, rows.length - 1]);
  for (let i = 1; indices.size < limit; i += 1) indices.add(Math.round(i * (rows.length - 1) / (limit - 1)));
  return [...indices].sort((a, b) => a - b).slice(0, limit).map((index) => rows[index]);
}

export default function ResearchDashboard() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("js_truthy_access");
  const [apiOnline, setApiOnline] = useState(false);
  const [runState, setRunState] = useState<"idle" | "running" | "complete" | "error">("idle");
  const [runStep, setRunStep] = useState(0);
  const [runMessage, setRunMessage] = useState("选择一个缺陷语料，运行可复现黑盒实验。");
  const [sessionId, setSessionId] = useState<string>();
  const [observationCount, setObservationCount] = useState(0);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [closure, setClosure] = useState<ClosureReport>();
  const [probeBudget, setProbeBudget] = useState(8);
  const [beamWidth, setBeamWidth] = useState(180);
  const [maxDepth, setMaxDepth] = useState(3);
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<string>();
  const [eventViewStart, setEventViewStart] = useState(0);
  const [neuralUrlSummary, setNeuralUrlSummary] = useState<NeuralUrlSummary>();

  useEffect(() => {
    Promise.all([api<{ status: string }>("/api/health"), api<Scenario[]>("/api/scenarios"), api<NeuralUrlSummary>("/research/neural_url_loop_11_summary.json")])
      .then(([, loaded, neuralSummary]) => {
        setApiOnline(true);
        setScenarios(loaded);
        setNeuralUrlSummary(neuralSummary);
        if (!loaded.some((item) => item.id === scenarioId) && loaded[0]) setScenarioId(loaded[0].id);
      })
      .catch((error: Error) => setRunMessage(`研究引擎未连接：${error.message}`));
  }, []);

  useEffect(() => {
    let mounted = true;
    const refresh = () => api<{ events: ActivityEvent[] }>("/api/research/events?limit=120")
      .then((result) => {
        if (!mounted) return;
        setEvents(result.events.slice().reverse());
        if (!selectedEventId && result.events.length) setSelectedEventId(result.events.at(-1)?.id);
      })
      .catch(() => undefined);
    refresh();
    const timer = window.setInterval(refresh, 900);
    return () => { mounted = false; window.clearInterval(timer); };
  }, [selectedEventId]);

  const scenario = useMemo(() => scenarios.find((item) => item.id === scenarioId) || scenarios[0], [scenarios, scenarioId]);
  const bestCandidate = candidates[0];
  const visibleEvents = events.filter((event) => new Date(event.timestamp).getTime() >= eventViewStart);
  const selectedEvent = visibleEvents.find((event) => event.id === selectedEventId) || visibleEvents[0];

  async function runProtocol() {
    if (!scenario || runState === "running") return;
    setRunState("running");
    setRunStep(1);
    setCandidates([]);
    setClosure(undefined);
    setObservationCount(0);
    setRunMessage("正在生成边界与对抗探针…");
    try {
      const session = await api<{ id: string }>("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenario.id }),
      });
      setSessionId(session.id);
      const probe = (payload: Json) => api<Json>(`/api/sessions/${session.id}/probe`, { method: "POST", body: JSON.stringify(payload) });
      let probes = 0;
      if (scenario.stateful && scenario.fields[0]?.path === "input.action") {
        const sequences = [["verify", "commit"], ["wait", "commit"], ["verify", "cancel"], ["commit", "commit"]];
        for (let episode = 0; episode < sequences.length; episode += 1) {
          for (let step = 0; step < sequences[episode].length; step += 1) {
            await probe({ input: { action: sequences[episode][step] }, context: {}, state: {}, episode_id: `seed-${episode + 1}`, step });
            probes += 1;
            setObservationCount(probes);
          }
        }
      } else {
        for (const [index, testCase] of cartesianCases(scenario.fields, probeBudget).entries()) {
          await probe({ ...testCase, episode_id: `seed-${index + 1}`, step: 0 });
          probes += 1;
          setObservationCount(probes);
        }
      }

      setRunStep(2);
      setRunMessage("正在把行为压缩为可执行 Rule IR…");
      let search = await api<{ candidates: Candidate[] }>(`/api/sessions/${session.id}/search`, {
        method: "POST",
        body: JSON.stringify({ max_depth: maxDepth, beam_width: beamWidth, history_depth: scenario.stateful ? 1 : 0 }),
      });
      setCandidates(search.candidates);

      setRunStep(3);
      setRunMessage("正在选择候选分歧最大的后续实验…");
      for (let round = 0; round < 2 && search.candidates.length >= 2; round += 1) {
        const suggested = await api<{ suggestion?: Json }>(`/api/sessions/${session.id}/suggest`);
        if (!suggested.suggestion) break;
        await probe({ ...suggested.suggestion, episode_id: `active-${round + 1}`, step: 0 });
        probes += 1;
        setObservationCount(probes);
        search = await api<{ candidates: Candidate[] }>(`/api/sessions/${session.id}/search`, {
          method: "POST",
          body: JSON.stringify({ max_depth: maxDepth, beam_width: beamWidth, history_depth: scenario.stateful ? 1 : 0 }),
        });
        setCandidates(search.candidates);
      }

      setRunStep(4);
      setRunMessage("正在验证覆盖、行为等价类与结论边界…");
      const report = await api<ClosureReport>(`/api/sessions/${session.id}/closure/analyze`, {
        method: "POST",
        body: JSON.stringify({ max_cases: 5000, coverage_threshold: 0.9, history_depth: scenario.stateful ? 1 : 0, auto_search: true }),
      });
      setClosure(report);
      setRunState("complete");
      setRunMessage("实验完成。规则、反例与闭环限制均可重放验证。");
    } catch (error) {
      setRunState("error");
      setRunMessage(error instanceof Error ? error.message : "实验失败");
    }
  }

  function jumpTo(id: string) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function exportEvents() {
    const blob = new Blob([JSON.stringify(visibleEvents, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `sift-activity-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#lab" aria-label="SIFT 首页"><span>S</span><b>SIFT</b><small>RESEARCH SYSTEM</small></a>
        <nav>{navItems.map(([id, label]) => <button key={id} onClick={() => jumpTo(id)}>{label}</button>)}</nav>
        <div className={`node-state ${apiOnline ? "online" : ""}`}><i />{apiOnline ? "ENGINE ONLINE" : "CONNECTING"}</div>
      </header>

      <section className="operator-shell" id="lab">
        <div className="operator-top wrap">
          <div><p className="overline">SIFT / OPERATOR WORKBENCH</p><h1>实验控制台</h1><p>控制黑盒探针、规则归纳、训练与证据闭环；所有操作写入 MCP 风格活动信封。</p></div>
          <div className="operator-actions"><a className="button quiet" href="/pg385">过滤反馈实验 ↗</a><a className="button quiet" href="/ops">研究运营台 ↗</a><a className="button quiet" href="/maze">迷宫回放台 ↗</a><button className="button quiet" onClick={exportEvents}>导出活动 JSON</button><button className="button primary" onClick={runProtocol} disabled={!apiOnline || runState === "running"}><Icon name="play" />{runState === "running" ? "运行中" : "启动实验"}</button></div>
        </div>
        <div className="console-grid wrap">
          <section className="command-panel console-panel">
            <div className="console-title"><span>RUN CONFIGURATION</span><b>{runState.toUpperCase()}</b></div>
            <label>缺陷语料<select value={scenarioId} onChange={(event) => setScenarioId(event.target.value)} disabled={runState === "running"}>{scenarios.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            <div className="parameter-grid">
              <label>初始探针<input type="number" min="4" max="18" value={probeBudget} onChange={(event) => setProbeBudget(Number(event.target.value))} /></label>
              <label>Beam<input type="number" min="20" max="500" value={beamWidth} onChange={(event) => setBeamWidth(Number(event.target.value))} /></label>
              <label>Rule depth<input type="number" min="1" max="5" value={maxDepth} onChange={(event) => setMaxDepth(Number(event.target.value))} /></label>
            </div>
            <div className="command-rule"><span>RESEARCH QUESTION</span><p>{scenario?.research_question || scenario?.description || "等待语料"}</p></div>
            <button className="console-run" onClick={runProtocol} disabled={!apiOnline || runState === "running"}><Icon name="play" />运行：探针 → 归纳 → 反例 → 闭环</button>
          </section>

          <section className="activity-console console-panel" id="activity">
            <div className="console-title"><span>LIVE ACTIVITY / MCP ENVELOPES</span><div><i className={apiOnline ? "online" : ""} /><b>{visibleEvents.length} EVENTS</b><button onClick={() => setEventViewStart(Date.now())}>清空视图</button></div></div>
            <div className="event-stream">
              {visibleEvents.length ? visibleEvents.map((event) => <button key={event.id} className={selectedEvent?.id === event.id ? "selected" : ""} onClick={() => setSelectedEventId(event.id)}>
                <time>{new Date(event.timestamp).toLocaleTimeString("zh-CN", { hour12: false })}</time><span className={`event-status ${event.status}`} />
                <div><b>{event.tool}</b><p>{event.message}</p></div><code>{event.actor}</code>
              </button>) : <div className="event-empty"><Icon name="spark" /><p>等待研究事件。启动实验后，Oracle、搜索器、训练器和证据引擎会在这里实时汇报。</p></div>}
            </div>
          </section>

          <section className="event-inspector console-panel">
            <div className="console-title"><span>EVENT INSPECTOR</span><b>{selectedEvent?.phase || "NO SELECTION"}</b></div>
            {selectedEvent ? <>
              <div className="inspector-meta"><span><b>ACTOR</b>{selectedEvent.actor}</span><span><b>TOOL</b>{selectedEvent.tool}</span><span><b>STATUS</b>{selectedEvent.status}</span></div>
              <pre>{JSON.stringify(selectedEvent.payload, null, 2)}</pre>
              {selectedEvent.artifact && <div className="artifact-link"><span>ARTIFACT</span><code>{selectedEvent.artifact}</code></div>}
            </> : <div className="event-empty"><p>点击活动流中的任意事件查看完整 payload。</p></div>}
          </section>

          <section className="run-monitor console-panel">
            <div className="console-title"><span>RUN MONITOR</span><b>{sessionId?.slice(0, 8).toUpperCase() || "NO RUN"}</b></div>
            <div className="monitor-metrics"><div><strong>{observationCount}</strong><span>OBS</span></div><div><strong>{candidates.length}</strong><span>RULES</span></div><div><strong>{bestCandidate ? `${(bestCandidate.accuracy * 100).toFixed(0)}%` : "—"}</strong><span>FIT</span></div><div><strong>{closure ? `${(closure.closure_score * 100).toFixed(0)}%` : "—"}</strong><span>EVIDENCE</span></div></div>
            <div className="monitor-progress"><span style={{ width: `${runStep * 25}%` }} /></div>
            <p>{runMessage}</p>
            {bestCandidate && <code className="leading-rule">{bestCandidate.pretty}</code>}
          </section>
        </div>
      </section>

      <section className="loop-results" id="results">
        <div className="wrap">
          <div className="loop-result-head">
            <div><p className="overline">FRESH COMPLETE-FAMILY HOLDOUT / LOOP 11</p><h2>URL 泛化从负迁移，走到可消融的神经学习。</h2></div>
            <div className={`research-verdict ${neuralUrlSummary?.final_confirmation.all_preregistered_checks_passed ? "passed" : "pending"}`}><span>PREREGISTERED STATUS</span><b>{neuralUrlSummary?.final_confirmation.all_preregistered_checks_passed ? "CONFIRMED" : "LOADING"}</b><small>科研试验通过 · 尚非生产安全声明</small></div>
          </div>
          <div className="headline-metrics">
            <article><span>NEURAL HOLDOUT</span><strong>{neuralUrlSummary ? `${(neuralUrlSummary.final_confirmation.candidate_neural_mean * 100).toFixed(2)}%` : "—"}</strong><p>目标家族训练样本为 0</p></article>
            <article><span>FROZEN BASELINE</span><strong>{neuralUrlSummary ? `${(neuralUrlSummary.final_confirmation.frozen_neural_mean * 100).toFixed(2)}%` : "—"}</strong><p>旧冻结神经路径</p></article>
            <article className="effect"><span>MIN NEURAL GAIN</span><strong>{neuralUrlSummary ? `+${(neuralUrlSummary.final_confirmation.minimum_gain_over_frozen * 100).toFixed(2)}` : "—"}</strong><p>percentage points · 两种子最小值</p></article>
            <article><span>COUNTEREXAMPLE TOP-1</span><strong>{neuralUrlSummary ? `${(neuralUrlSummary.final_confirmation.counterexample_top1_mean * 100).toFixed(0)}%` : "—"}</strong><p>随机基线 37.50%</p></article>
          </div>
          <div className="experiment-matrix panel">
            <div className="matrix-head"><div><span>SEED / DATA FINGERPRINT</span></div><span>NEURAL / FROZEN</span><span>HEAD ON / OFF</span><span>TOP-1 / RANDOM</span><span>C5 RULE PATH</span></div>
            {neuralUrlSummary?.per_seed.map((row) => <div className="matrix-row" key={row.seed}>
              <div><b>{row.seed}</b><code>{row.dataset_sha256.slice(0, 14)}</code></div>
              <span><b>{(row.candidate_neural * 100).toFixed(2)}%</b><small>/ {(row.frozen_neural * 100).toFixed(2)}%</small></span>
              <span><b>{(row.candidate_neural * 100).toFixed(2)}%</b><small>/ {(row.same_checkpoint_without_set_head * 100).toFixed(2)}%</small></span>
              <span><b>{(row.counterexample_top1 * 100).toFixed(0)}%</b><small>/ 37.50%</small></span>
              <span><b>{(row.c5_rule_system * 100).toFixed(0)}%</b><small>非神经结果</small></span>
            </div>)}
          </div>
          <div className="causal-grid">
            <article><span>ROOT CAUSE</span><h3>episode 标签绑定不足</h3><p>旧模型认识 URL 原语，却依赖字段名与家族捷径，不能稳定把正负轨迹绑定到新查询。</p></article>
            <article><span>ARCHITECTURE / +128</span><h3>可学习集合比较头</h3><p>规范 URL 槽位配合查询—正负样例比较；同 checkpoint 关闭头部，至少损失 28.83pp。</p></article>
            <article><span>FAILED BRANCHES</span><h3>三次失败都保留</h3><p>表示单改无效；全局元标签伤及 truthiness；v1 又伤及 substring。阈值未放宽，失败种子未复用。</p></article>
            <article className="engineering-card"><span>CLAIM BOUNDARY</span><h3>神经、规则、工程分开算</h3><p>神经 88.42%；C5 规则路径 100% 但不计学习。下一阶段才验证跨语言、真实解析器、噪声与吞吐。</p><b>{neuralUrlSummary?.final_confirmation.all_preregistered_checks_passed ? "4/4 CHECKS PASS" : "CHECKING"}</b></article>
          </div>
          <div className="metric-warning"><b>SCIENTIFIC BOUNDARY</b><p>这是合成完整家族留出上的因果证据，不是生产漏洞检出率。旧家族最差回归为 {neuralUrlSummary ? `${(neuralUrlSummary.final_confirmation.worst_old_family_regression * 100).toFixed(2)}pp` : "—"}，仍在预注册 −2pp 限制内。</p></div>
        </div>
      </section>

      <section className="hero wrap" id="about">
        <div className="hero-copy">
          <p className="overline">MEMORY-FIRST · ROOT-CAUSE DRIVEN · EXECUTABLE EVIDENCE</p>
          <h1>让模型从失败里，<br /><em>长出新的结构。</em></h1>
          <p className="hero-lead">不是把参数做大，而是让一个小模型学会观察黑盒、记住关键经验、找到失败根因，再决定该改数据、记忆、奖励还是架构。</p>
          <div className="hero-actions">
            <button className="button primary" onClick={runProtocol} disabled={!apiOnline || runState === "running"}><Icon name="play" />{runState === "running" ? "实验运行中" : "运行黑盒实验"}</button>
            <button className="button quiet" onClick={() => jumpTo("memory")}>查看记忆消融 <Icon name="arrow" /></button>
          </div>
        </div>
        <aside className="model-ticket">
          <div className="ticket-top"><span>CURRENT RESEARCH PILOT</span><b>03 / SEEDS</b></div>
          <h2>SIFT–0.9M</h2>
          <p>GPT-2 style decoder<br />+ persistent rule memory</p>
          <div className="ticket-grid"><div><b>908K</b><span>PARAMS</span></div><div><b>640</b><span>CONTEXT</span></div><div><b>8</b><span>MEMORY ITEMS</span></div><div><b>4</b><span>LAYERS</span></div></div>
          <div className="ticket-foot"><span>ARCHITECTURE IS MUTABLE</span><i>↗</i></div>
        </aside>
      </section>

      <section className="ticker"><div className="wrap ticker-inner"><span>1,800 PROGRAMS / SEED</span><i /> <span>21,148+ VERIFIED TRACES</span><i /> <span>908,546 PARAMETERS</span><i /> <span>100% DOUBLE HOLDOUT</span><i /> <span>+39.52 PP DOM EFFECT</span></div></section>

      <section className="discovery-modes wrap" aria-label="规则发现模式">
        <article><span>01 · WHITE BOX</span><h3>明文抽取</h3><p>源码、AST、字节码直接转成语言无关候选规则。</p><b>成熟解析器优先</b></article>
        <article><span>02 · GRAY BOX</span><h3>静态 × 运行时</h3><p>用真实轨迹验证候选，定位动态配置与静态语义冲突。</p><b>冲突保留双版本</b></article>
        <article className="black-box-mode"><span>03 · BLACK BOX</span><h3>主动探测补足</h3><p>只凭输入输出筛选假设、打边界、探索状态并重放反例。</p><b>黑盒是一等公民</b></article>
      </section>

      <section className="experiment-section wrap">
        <div className="section-heading"><div><p className="overline">01 · LIVE LAB</p><h2>一条能重放的研究运行</h2></div><p>输入与输出来自真实 Oracle；自然语言解释不能替代执行证据。</p></div>
        <div className="experiment-grid">
          <article className="scenario-panel panel">
            <div className="panel-label"><span>DEFECT CORPUS</span><b>{scenario?.cwe || "LOADING"}</b></div>
            <select value={scenarioId} onChange={(event) => setScenarioId(event.target.value)} disabled={runState === "running"} aria-label="缺陷语料">
              {scenarios.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
            <span className={`severity ${scenario?.severity?.toLowerCase()}`}>{scenario?.severity || "—"}</span>
            <h3>{scenario?.name || "载入语料…"}</h3>
            <p>{scenario?.research_question || scenario?.description}</p>
            <div className="rule-quote"><span>GAME ABSTRACTION</span><p>{scenario?.game_rule || "等待规则抽象"}</p></div>
          </article>

          <article className="run-panel panel">
            <div className="panel-label"><span>EXPERIMENT RUN</span><b className={runState}>{runState.toUpperCase()}</b></div>
            <div className="run-steps">
              {["边界探针", "规则归纳", "主动反例", "闭环验证"].map((label, index) => <div key={label} className={runStep > index ? "done" : runStep === index + 1 && runState === "running" ? "active" : ""}><i>{index + 1}</i><span>{label}</span><b>{runStep > index ? "✓" : ""}</b></div>)}
            </div>
            <div className="run-message">{runMessage}</div>
            <div className="run-id"><span>RUN</span><code>{sessionId?.slice(0, 18).toUpperCase() || "NOT STARTED"}</code></div>
          </article>

          <article className="result-panel panel">
            <div className="panel-label"><span>MEASUREMENTS</span><b>LIVE</b></div>
            <div className="result-numbers"><div><strong>{observationCount}</strong><span>OBSERVATIONS</span></div><div><strong>{candidates.length}</strong><span>HYPOTHESES</span></div><div><strong>{bestCandidate ? `${(bestCandidate.accuracy * 100).toFixed(0)}%` : "—"}</strong><span>BEST FIT</span></div><div><strong>{closure ? `${(closure.closure_score * 100).toFixed(0)}%` : "—"}</strong><span>EVIDENCE</span></div></div>
            <div className="result-verdict"><span>{closure ? closureNames[closure.closure_status] || closure.closure_status : "等待实验"}</span><p>{closure?.reasons?.[0] || "完成运行后，这里会显示有限域内的证据强度和结论边界。"}</p></div>
          </article>
        </div>
      </section>

      <section className="model-section" id="model">
        <div className="wrap">
          <div className="section-heading inverse"><div><p className="overline">02 · MODEL</p><h2>0.9M 先找机制，不给规模幻觉买单</h2></div><p>只有小模型在规则族外饱和，且消融证明容量是瓶颈，才升级 Dense 或 MoE。</p></div>
          <div className="semantic-layer">
            <div className="language-adapters">
              {['JavaScript','Python','Rust','Java / JVM'].map((language) => <span key={language}>{language}<i>ADAPTER</i></span>)}
            </div>
            <div className="semantic-core">
              <div><span>COMMON SEMANTIC RULE</span><b>语言无关规则层</b></div>
              <code>coercion · boundary · state · invariant · effect · evidence</code>
              <p>通用规则骨架用于跨语言学习；溢出、异常、求值顺序和 nullish 行为作为显式语言参数保留。</p>
            </div>
            <div className="semantic-outputs">
              <span><b>HUMAN</b>可读游戏规则</span><span><b>MACHINE</b>可执行 Rule IR</span><span><b>MEMORY</b>可检索语义特征</span>
            </div>
          </div>
          <div className="architecture-line">
            <div><span>01</span><Icon name="code" size={22} /><b>多模态序列</b><small>JS · GAME · TRACE</small></div><i>→</i>
            <div><span>02</span><b>4 层 Decoder</b><small>d=128 · 4 heads</small></div><i>→</i>
            <div className="memory-node"><span>03</span><Icon name="memory" size={22} /><b>长期记忆</b><small>Episode + Rule IR</small></div><i>→</i>
            <div><span>04</span><b>四任务头</b><small>Rule · Probe · Evidence</small></div>
          </div>
          <div className="curriculum-grid">
            {[['P0','15–30M','合成规则预训练','5,000 个自动变异程序；执行器验真。'],['P1','60K','多任务监督','抽象、预测、归纳、查询与拒答。'],['P2','20K','主动实验 RL','奖励信息增益与最小反例。'],['P3','10K','记忆策略 RL','训练写入、检索、合并和遗忘。']].map(([stage, scale, title, text]) => <article key={stage}><div><span>{stage}</span><b>{scale}</b></div><h3>{title}</h3><p>{text}</p></article>)}
          </div>
        </div>
      </section>

      <section className="memory-section wrap" id="memory">
        <div className="section-heading"><div><p className="overline">03 · MEMORY ABLATION</p><h2>长上下文 ≠ 长期记忆</h2></div><p>32 个 Episode，关键事件延迟 6 步；结果来自本地可复现实验。</p></div>
        <div className="neural-pilot panel">
          <div className="pilot-intro"><span>ACTUAL GPU TRAINING · RTX 3060 · 3 SEEDS</span><h3>Frontend Rule Memory 族外实验</h3><p>输入只包含黑盒 TRACE、QUERY 和确定性输入投影；源码、家族、Rule IR、CWE 与目标答案不可见。</p></div>
          <div className="pilot-metric"><span>WITH EXECUTABLE MEMORY</span><strong>100%</strong><small>two unseen frontend families</small></div>
          <div className="pilot-metric muted-metric"><span>WITHOUT MEMORY</span><strong>45.57%</strong><small>three-seed mean</small></div>
          <div className="pilot-metric gain-metric"><span>CAUSAL MEMORY GAIN</span><strong>+54.43</strong><small>percentage points</small></div>
          <div className="pilot-finding"><b>DEEPEST ROOT CAUSE</b><p>正确记忆没有稳定进入 logits，且 &lt; 运算符与提示协议冲突。零参数可执行门 + 语言中立 Rule IR 修复了两层故障，无需扩模型。</p></div>
        </div>
        <div className="memory-layout">
          <div className="memory-chart panel">
            <div className="chart-head"><span>RULE RECOVERY ACCURACY</span><b>HIGHER IS BETTER</b></div>
            <div className="bar-row"><label><b>无 Rule Memory</b><small>structured query only</small></label><div><i style={{ width: "45.57%" }} /></div><strong>45.57%</strong></div>
            <div className="bar-row"><label><b>原始文本 + 记忆</b><small>DOM family holdout</small></label><div><i style={{ width: "60.48%" }} /></div><strong>60.48%</strong></div>
            <div className="bar-row winner"><label><b>结构化 + 可执行记忆</b><small>two-family holdout</small></label><div><i style={{ width: "100%" }} /></div><strong>100%</strong></div>
            <div className="chart-note"><Icon name="spark" /><p>结构化 DOM 语义带来 <b>+39.52pp</b>；Episode Rule Memory 则负责跨轨迹绑定和稳定执行。</p></div>
          </div>
          <div className="memory-stack">
            <article><span>WORKING</span><b>4K 当前上下文</b><p>保留本次推理所需的源码、最近轨迹和候选。</p></article>
            <article><span>EPISODIC</span><b>跨会话经验</b><p>按程序指纹、状态签名和漏洞族检索历史反例。</p></article>
            <article><span>SEMANTIC</span><b>版本化 Rule IR</b><p>沉淀规则、置信度、适用边界和证据指针。</p></article>
          </div>
        </div>
      </section>

      <section className="rules-section" id="rules">
        <div className="wrap">
          <div className="section-heading"><div><p className="overline">04 · IMPROVEMENT CONSTITUTION</p><h2>架构可以改，证据标准不能改</h2></div><p>实验成熟才进入工程扩展；扩展失败必须区分科学问题与工程能力问题。</p></div>
          <div className="root-loop">
            {[['01','复现','保存模型、数据、种子、记忆与最小失败样本。'],['02','根因分层','区分数据、表示、记忆、架构、目标、搜索和评估。'],['03','反事实消融','注入正确记忆、改变数据或关闭组件，定位因果层。'],['04','最小改动','先修数据与 reward，再考虑上下文、容量或 MoE。'],['05','族外验证','用完整未见漏洞族检查效果、误报、ECE 与成本。'],['06','保留 / 回滚','达成预注册指标才进入主线，否则保留证据并回滚。'],['07','成熟度门禁','至少 3 个种子、族外验证、机制消融和完整数据血缘通过后，才能扩大数据与算力。'],['08','双路径诊断','扩展失败先判断实验问题、工程能力问题、混合问题或证据不足，再分别修复。']].map(([number, title, text]) => <article key={number}><span>{number}</span><div><h3>{title}</h3><p>{text}</p></div></article>)}
          </div>
          <div className="constitution"><Icon name="flask" size={24} /><p><b>Failure → Reproduce → Root cause → Ablation → Holdout → Maturity gate → Experiment / Engineering triage → Keep / Roll back</b><span>不以堆基础设施掩盖失败实验，也不以修改科学假设掩盖工程故障；两条路径独立验收。</span></p></div>
        </div>
      </section>

      <footer><div className="wrap"><div className="wordmark footer-mark"><span>S</span><b>SIFT</b></div><p>Small model. Long memory. Executable evidence.</p><a href="#lab">回到顶部 ↑</a></div></footer>
    </main>
  );
}
